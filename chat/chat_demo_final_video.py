"""
Video-LLaVA Chatbot (multi-turn)
- Load Video-LLaVA model (HF Hub or local)
- Upload a video + hints
- DV/OV classifier + Fault-BERT(.pth) 사용
- (옵션) Fault-BERT encoder의 잠재벡터로 soft token을 만들어 Video-LLaVA 텍스트 임베딩 시퀀스에 삽입 (inputs_embeds)
- (옵션) 해석가능성 모드: 비디오만으로 초안 생성 → 소프트 토큰으로 짧게 수정(보정)
"""

import os, glob, shutil 
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import cv2
import json

from torchvision.models.video import r3d_18, R3D_18_Weights
from transformers import (
    AutoModel, AutoTokenizer,
    TimesformerModel, VideoMAEModel,
    VideoLlavaProcessor, VideoLlavaForConditionalGeneration
)

import re

# ▼ 파일 상단 util 근처에 추가
ADMIN_PREFIX_RE = re.compile(r"^\s*(revised\s*draft|draft|hint|assistant|user|description|evidence)\s*:?", re.I)
TAG_BLOCK_RE    = re.compile(r"\s*<hint>.*?</hint>\s*", re.I | re.S)
PROMPT_TAIL_RE  = re.compile(
    r"Describe who enters first, each vehicle’s turn or straight path, where their paths converge, and the immediate visible outcome in 4–6 sentences\.?",
    re.I
)

import random
import hashlib

def _pick_by_seed(candidates, seed_key: str | None):
    if not candidates:
        return ""
    if seed_key:
        h = int(hashlib.md5(seed_key.encode("utf-8")).hexdigest(), 16)
        idx = h % len(candidates)
        return candidates[idx]
    return random.choice(candidates)

def _assistant_text_bubble(text: str):
    # Chatbot(type="messages") 안전한 텍스트 버블
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}

WHO_EV_VICTIM = [
    "From the visible maneuvers, the ego vehicle seems to be the victim, with the other vehicle primarily at fault.",
    "Based on the scene, the ego vehicle appears to be the impacted party, while the other vehicle bears the main responsibility.",
    "The evidence suggests the ego vehicle is the victim and the other vehicle holds the greater fault.",
    "It looks like the ego vehicle suffered the harm, and the other vehicle is chiefly at fault.",
    "Observations indicate the ego vehicle is the victim; primary fault lies with the other vehicle.",
]

WHO_OV_VICTIM = [
    "From the visible maneuvers, the other vehicle seems to be the victim, with the ego vehicle primarily at fault.",
    "Based on the scene, the other vehicle appears to be the impacted party, while the ego vehicle bears the main responsibility.",
    "The evidence suggests the other vehicle is the victim and the ego vehicle holds the greater fault.",
    "It looks like the other vehicle suffered the harm, and the ego vehicle is chiefly at fault.",
    "Observations indicate the other vehicle is the victim; primary fault lies with the ego vehicle.",
]

WHO_UNCERTAIN = [
    "Given the visible evidence, identifying the victim and the primarily at-fault party is inconclusive.",
    "With the current view, it’s difficult to determine clear victim and fault.",
    "Evidence is insufficient to confidently decide who is the victim and who is primarily at fault.",
    "The scene doesn’t clearly reveal a single primarily at-fault party.",
    "It remains ambiguous which party is the victim or mainly responsible.",
]



def pick_who_line(ev: float, ov: float, seed_key: str | None = None) -> str:
    # seed_key를 넘기면 "같은 상황 = 같은 문장" 보장, None이면 매 클릭마다 랜덤
    if ev < ov:
        return _pick_by_seed(WHO_EV_VICTIM, seed_key)
    elif ev > ov:
        return _pick_by_seed(WHO_OV_VICTIM, seed_key)
    else:
        return _pick_by_seed(WHO_UNCERTAIN, seed_key)

def _render_who_line(ev: float, ov: float, style: str = "short", seed_key: str = "") -> str:
    """
    ev < ov  -> EV 피해자 (OV 과실↑)
    ev > ov  -> OV 피해자 (EV 과실↑)
    ev == ov -> 판단 곤란
    style: "short" | "brief" | "detailed"
    """
    tie_lines = [
        "From visible evidence alone, it’s difficult to identify a clear victim or at-fault party.",
        "Based on what is visible, liability is ambiguous and neither side is clearly the victim.",
        "The frames do not provide enough clarity to determine who is victim or primarily at fault.",
        "With only the visible cues, it’s not possible to decisively determine victim and fault.",
    ]

    ev_victim_lines = [
        "Based on the visible maneuvers, the ego vehicle is the victim, and the other vehicle is primarily at fault.",
        "Given the observed entry and path interaction, the ego vehicle is the victim; the other vehicle bears the primary fault.",
        "From the visible trajectories, the ego vehicle is the victim and the other vehicle holds the greater share of fault.",
        "Considering the visible approach and conflict point, the ego vehicle is the victim while the other vehicle is mainly at fault.",
        "By the visible evidence, the ego vehicle is the victim; the other vehicle is chiefly responsible.",
    ]

    ov_victim_lines = [
        "Based on the visible maneuvers, the other vehicle is the victim, and the ego vehicle is primarily at fault.",
        "Given the observed entry and path interaction, the other vehicle is the victim; the ego vehicle bears the primary fault.",
        "From the visible trajectories, the other vehicle is the victim and the ego vehicle holds the greater share of fault.",
        "Considering the visible approach and conflict point, the other vehicle is the victim while the ego vehicle is mainly at fault.",
        "By the visible evidence, the other vehicle is the victim; the ego vehicle is chiefly responsible.",
    ]

    if abs(ev - ov) < 1e-6:
        base = _pick_by_seed(tie_lines, seed_key)
    elif ev < ov:
        base = _pick_by_seed(ev_victim_lines, seed_key)   # EV 피해자
    else:
        base = _pick_by_seed(ov_victim_lines, seed_key)   # OV 피해자

    if style == "detailed":
        # 아주 짧은 근거 한 토막 덧붙이기(문장 길이 과도하게 늘리지 않게 중립 문구 사용)
        tail = " This judgment is drawn strictly from visible motion, entry order, and where their paths converge."
        return base if base.endswith(".") else (base + ".") + tail
    return base

def _is_model_loaded_text(msg: str) -> bool:
    # ui_load_model의 반환 문자열이 "Loaded model: ..." 형태면 True
    m = (msg or "").strip().lower()
    return m.startswith("loaded model:") or ("loaded" in m and "model" in m)

def gate_controls(video_name_val, model_ok: bool):
    """
    - 영상 선택됨 + 모델 로드됨 = 챗 버튼 활성화
    - 그 외에는 비활성화
    """
    video_ok = bool((video_name_val or "").strip())
    ok = bool(video_ok and model_ok)
    return (
        gr.update(interactive=ok),  # btn_ratio
        gr.update(interactive=ok),  # btn_who
    )

def _strip_admin_and_hint(text: str) -> str:
    """Revised DRAFT:, HINT:, <hint>...</hint>, 프롬프트 꼬리문 제거 + 라인 정리"""
    if not text:
        return ""
    # 1) <hint> ... </hint> 블록 통째로 제거
    text = TAG_BLOCK_RE.sub(" ", text)

    # 2) 'HINT:' 이후는 싹 절단
    text = re.split(r"(?i)\bHINT\s*:", text)[0]

    # # 3) 라벨/머리말 제거 (Revised DRAFT:, DRAFT:, Description:, Evidence:, etc.)
    # lines = []
    # for ln in text.splitlines():
    #     ln = ln.strip()
    #     if not ln:
    #         continue
    #     if ADMIN_PREFIX_RE.match(ln):
    #         # 'Revised DRAFT: 내용'처럼 콜론 뒤 본문만 살릴 수 있으면 살림
    #         parts = re.split(r":\s*", ln, maxsplit=1)
    #         if len(parts) == 2 and parts[1].strip():
    #             lines.append(parts[1].strip())
    #         continue
    #     lines.append(ln)
    # text = " ".join(lines).strip()

    # # 4) 프롬프트 꼬리문(“Describe who enters first…”) 제거
    # text = PROMPT_TAIL_RE.sub("", text).strip()

    # 5) 선행 마침표/중복 공백 정리
    text = re.sub(r"^\s*[\.·•]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 모호/저정보 표현 패턴
GENERIC_PHRASES = re.compile(
    r"\b("
    r"busy intersection|in the middle(?: of the intersection)?|near the center|"
    r"on the side of the road|the vehicles are parked|parked(?: along the street| on the side)?"
    r")\b",
    re.I,
)

def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))

def _enforce_min_words(text: str, min_words: int = 80) -> str:
    """짧으면 자연스럽게 분량을 늘리도록 꼬리문을 추가(한 번만)."""
    if _word_count(text) >= int(min_words):
        return text
    tail = (
        " Describe who enters first, each vehicle’s turn or straight path, where their paths converge, "
        "and the immediate visible outcome in 4–6 sentences."
    )
    text = (text or "").strip()
    if text and not text.endswith((".", "!", "?")):
        text += "."
    return (text + " " + tail).strip()

def _is_low_info(text: str) -> bool:
    """1) 너무 짧거나 2) 모호한 상투구 포함 시 저정보로 간주."""
    if _word_count(text) < 50:
        return True
    if GENERIC_PHRASES.search(text or ""):
        return True
    return False

def _degenericize(text: str, dv_hint: str = "", ov_hint: str = "") -> str:
    """
    'parked', 'busy intersection' 같은 모호한 표현을
    DV/OV 힌트를 활용한 중립적·구체적 표현으로 교체.
    """
    replacements = [
        (r"\bthe vehicles are parked\b", "both vehicles are moving and converging based on their visible paths"),
        (r"\bparked(?: along the street| on the side)?\b", "moving along their lanes"),
        (r"\bbusy intersection\b", "an intersection with multiple approach lanes"),
        (r"\bin the middle(?: of the intersection)?\b", "near the conflict point of their trajectories"),
        (r"\bnear the center\b", "near the conflict point"),
        (r"\bon the side of the road\b", "along the approach lane"),
    ]
    out = text or ""
    for pat, rep in replacements:
        out = re.sub(pat, rep, out, flags=re.I)

    # 힌트로 약간 구체화(너무 길어지지 않게 짧게만)
    add_bits = []
    if dv_hint:
        add_bits.append(f"Ego vehicle appears {dv_hint.lower()}.")
    if ov_hint:
        add_bits.append(f"Other vehicle appears {ov_hint.lower()}.")
    if add_bits and _word_count(out) < 90:
        if not out.endswith((".", "!", "?")):
            out += "."
        out += " " + " ".join(add_bits)
    return out.strip()

def _regen_if_low_info(
    frames,
    text: str,
    soft_tokens,
    temperature: float,
    sampling: bool,
    engine,                      # VideoLLaVAChatEngine 인스턴스(여기서는 ENGINE)
    processor_prompt_boost: str = "",
    max_new_tokens: int = 220,
):
    """저정보면 한 번 더 재생성해서 구체화."""
    if not _is_low_info(text):
        return text
    boost = (
        "Write 4–6 concrete sentences (90–140 words). "
        "Name relative positions (left/right/front/back), entry order, and turning/straight trajectories. "
        "Avoid generic phrases like 'busy intersection', 'in the middle', or 'parked'."
    )
    if processor_prompt_boost:
        boost += " " + processor_prompt_boost

    new_text = engine.generate_with_soft_tokens_from_fault(
        frames=frames,
        chat_prompt=boost,
        soft_tokens=soft_tokens,
        insert_pos=2,
        temperature=max(0.3, float(temperature)),
        do_sample=bool(sampling),
        max_new_tokens=int(max_new_tokens),
    )
    return new_text or text

def _summary_only(text: str) -> str:
    """HINT/비율/템플릿 라인 제거하고, 요약 문장만 1~2문장 남김."""
    if not text:
        return ""
    # HINT/비율/템플릿/숫자라인 제거
    drop = re.compile(r"^\s*(HINT:|Fault\b|Predicted fault|Template:|Ego=|Other=)", re.I)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not drop.match(ln)]

    # 'Description:'가 있으면 그 부분만 사용
    for ln in lines:
        if ln.lower().startswith("description:"):
            ln = re.sub(r"(?i)^description:\s*", "", ln).strip()
            sents = re.split(r"(?<=[.!?])\s+", ln)
            out = " ".join(sents[:2]).strip()
            return out if out else ln

    # 없으면 남은 텍스트에서 1~2문장
    sents = re.split(r"(?<=[.!?])\s+", " ".join(lines))
    return " ".join(sents[:2]).strip()

# --- 기존 import/전역 하단 어딘가에 유틸 추가 ---
def _default_session():
    return {
        "video_path": None,
        "video_name": None,
        "preview_sent": False,
        "dv_hint": None,
        "ov_hint": None,
        "topk_text": None,
        "fault_hint": None,
        "dc_f": None,
        "ov_f": None,
        "victim": None,
        "sentence": None,
        "soft_tokens": None,
        "model_id": None,
    }

def _assistant_video_preview_bubble(video_path: str, caption: str = "I’ll analyze this video."):
    # ✅ Gradio Chatbot(type="messages") 최신 스키마에 맞춤
    # - 파일/비디오 메시지는 단일 dict로 보내고
    # - 경로 키는 "path"를 사용
    # - 캡션은 별도의 text 아이템이 아니라 alt_text로 전달
    return {
        "role": "assistant",
        "content": [
            {"type": "video", "video": video_path, "alt_text": caption}
        ]
    }

def _intent_from_message(msg: str):
    m = (msg or "").strip().lower()

    # summary
    if any(k in m for k in [
        "summary", "summarize", "describe", "explain", "overview", "what happened", "accident summary"
    ]):
        return "summary"

    # victim / at-fault party
    if any(k in m for k in [
        "who is the victim", "victim", "at fault", "who is at fault", "liable", "responsible party"
    ]):
        return "victim"

    # ratio / liability percentage
    if any(k in m for k in [
        "fault ratio", "liability ratio", "negligence", "percentage", "share of fault", "tell me the ratio"
    ]):
        return "ratio"

    # default
    return "summary"

# -------------------------
# 검색 경로 / 확장자
# -------------------------
SEARCH_ROOTS = [
    "/mnt/data/videos",
    "/app/data/raw/videos/training_reencoded",
    "/app/data/raw/videos/validation_reencoded",
    "/app/data",
    "/app/chat/data/samples",
]

ALLOWED_PATHS = [
    "/mnt/data/videos",
    "/app/data/raw/videos/training_reencoded",
    "/app/data/raw/videos/validation_reencoded",
    "/app/data",
    "/app/chat/data/samples",
]
VIDEO_EXTS = [".mp4", ".mkv", ".mov", ".avi", ".webm"]

SAMPLE_VIDEOS = [
    "bb_1_170415_vehicle_193_011.mp4",
    # "bb_1_130527_vehicle_119_219.mp4"
    "bb_1_170630_vehicle_192_035.mp4",
    "bb_1_220613_vehicle_199_116.mp4"
    
]
def find_video_path_by_name(name: str) -> Optional[str]:
    name = name.strip().strip('"').strip("'")
    if not name:
        return None
    has_ext = os.path.splitext(name)[1].lower() in VIDEO_EXTS

    for root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        if any(ch in name for ch in "*?[]"):
            patterns = [os.path.join(root, name)]
        else:
            patterns = [os.path.join(root, name)] if has_ext else [os.path.join(root, name + ext) for ext in VIDEO_EXTS]
        for pat in patterns:
            hits = sorted(glob.glob(pat))
            if hits:
                hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return hits[0]
    return None

def _resolve_sample_path(p: str) -> str:
    if p and os.path.isabs(p) and os.path.exists(p):
        return p
    q = find_video_path_by_name(p)  # SEARCH_ROOTS에서 탐색
    return q if q and os.path.exists(q) else p

def _build_sample_list() -> list[tuple[str, str]]:
    """UI 그릴 때 호출: SAMPLE_VIDEOS(문자열) → [(resolved_path, label)]"""
    out = []
    for p in SAMPLE_VIDEOS:
        path = _resolve_sample_path(p)
        # 라벨은 파일명(확장자 제외)에서 자동 생성
        label = os.path.splitext(os.path.basename(path or p))[0]
        # 파일이 없으면 스킵(원하면 경고 라벨로 넣어도 됨)
        if path and os.path.exists(path):
            out.append((path, label))
    return out

import re

def _ensure_analysis(session, device, video_path, clf_ckpt, clf_backbone, clf_pretrained,
                     clf_topk, fault_ckpt, fault_model, fault_basis, num_frames, frame_size):
    """세션 캐시에 DV/OV, Fault, 템플릿, soft-token을 준비. 이미 있으면 재계산 안 함."""
    # (1) DV/OV 분류기 캐시
    if session["dv_hint"] is None or session["ov_hint"] is None or session["topk_text"] is None:
        try:
            clf = _ensure_classifier(clf_ckpt, clf_backbone, clf_pretrained, device)
            dv_hint, ov_hint, topk_text = predict_classifier(
                video_path, clf, device=device, topk=int(clf_topk)
            )
        except Exception as e:
            dv_hint, ov_hint, topk_text = "UNKNOWN_DV", "UNKNOWN_OV", f"[classifier error: {e}]"
        session["dv_hint"], session["ov_hint"], session["topk_text"] = dv_hint, ov_hint, topk_text
        session["sentence"] = render_template_from_labels(dv_hint, ov_hint)

    # (2) Fault-BERT 수치/힌트 캐시
    if session["fault_hint"] is None or session["dc_f"] is None or session["ov_f"] is None or session["victim"] is None:
        try:
            fr_m, fr_tok = _ensure_fault(fault_ckpt, fault_model, device)
            y = predict_fault_ratio(fr_m, fr_tok, session["sentence"], device=device)
            dc_f, ov_f = float(y[0]), float(y[1])
            victim = "Undetermined" if abs(dc_f - ov_f) < 1e-6 else ("Ego Vehicle" if dc_f < ov_f else "Other Vehicle")
            session["dc_f"], session["ov_f"], session["victim"] = dc_f, ov_f, victim
            session["fault_hint"] = (
                f"Predicted fault (basis {int(fault_basis)}): Ego={dc_f:.1f}, Other={ov_f:.1f}; "
                f"Likely victim: {victim}; Template: {session['sentence']}"
            )
        except Exception as e:
            session["dc_f"], session["ov_f"], session["victim"] = 5.0, 5.0, "Undetermined"
            session["fault_hint"] = f"[fault error: {e}]; Template: {session['sentence']}"

    # (3) Soft token 캐시
    if session["soft_tokens"] is None:
        try:
            fr_m, fr_tok = _ensure_fault(fault_ckpt, fault_model, device)
            bert_hint_text = f"Ego={session['dv_hint']}; Other={session['ov_hint']}; {session['fault_hint']}"
            session["soft_tokens"] = ENGINE.make_soft_tokens_from_trained_fault(
                fault_m=fr_m, fault_tok=fr_tok, text=bert_hint_text, take="cls", max_tokens=4
            )
        except Exception:
            session["soft_tokens"] = None

# def build_ratio_draft_prompt():
#     return (
#         "You are a traffic-accident legal analyst.\n"
#         "Using ONLY what is VISIBLE in the frames, write 1–2 sentences that justify which side bears more fault.\n"
#         "Base the reasoning strictly on: (a) entry order into the conflict zone, (b) turning vs straight trajectories, "
#         "and (c) where/why their paths converge (cut-in, encroachment, failure to yield). "
#         "Do NOT mention numbers, percentages, timestamps, signals/lights, lane counts, speed, or unseen context.\n"
#         "Focus on WHY one vehicle’s visible maneuver created the conflict point."
#     )
def build_ratio_draft_prompt():
    return (
        "You are a traffic-accident legal analyst.\n"
        "Using ONLY what is VISIBLE in the frames, write 1–2 sentences that justify which side bears more fault.\n"
        "Refer to vehicles ONLY as **Ego Vehicle (EV)** and **Other Vehicle (OV)**.\n"
        "Do NOT mention colors, makes, models, or generic labels (e.g., 'the car', 'the truck').\n"
        "Base the reasoning strictly on: (a) entry order, (b) turning vs straight trajectories, (c) conflict point.\n"
        "Do NOT mention numbers, percentages, timestamps, signals/lights, lane counts, speed, or unseen context.\n"
        "Focus on WHY one vehicle’s visible maneuver created the conflict point."
    )

def _normalize_why(text: str) -> str:
    """이유 설명만 1–2문장으로 정리 + 금지어 제거"""
    if not text: 
        return ""
    text = _strip_admin_and_hint(text)
    # 숫자·신호·시간 등 금지 단어 방어적 제거
    text = re.sub(r"\b(\d+(\.\d+)?%|\d+\s*(km/h|mph|m|s|sec|seconds|minutes))\b", "", text, flags=re.I)
    text = re.sub(r"\b(traffic\s*light|signal(?:ized)?|timestamp|AM|PM|lane\s*count|speed)\b", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    sents = sents[:2] if sents else []
    out = " ".join(sents).strip()
    if out and out[-1] not in ".!?":
        out += "."
    return out

def _answer_for_intent(intent, frames, history, style, temperature, sampling,
                       use_soft, soft_gain, fault_basis, session):
    """
    의도별로 '필요한 내용만' 리턴.
    - summary: 요약만
    - victim: 피해자/가해자 판단만 (한 줄)
    - ratio  : "DC x.x / OV x.x (basis N)" 한 줄 (+선택적 한 줄 설명)
    """
    # 공통 프롬프트(영어 지시)는 모델용으로만 내부에서 사용
    if intent == "summary":
        draft_prompt = build_video_only_prompt()
        draft = ENGINE.generate_video_only(
            frames=frames, chat_prompt=draft_prompt,
            temperature=float(temperature), do_sample=bool(sampling), max_new_tokens=150
        )
        revise_prompt = build_revise_prompt(draft, "", session["fault_hint"])
        text = ENGINE.revise_with_soft_tokens(
            frames=frames,
            chat_prompt=revise_prompt,
            soft_tokens=(session["soft_tokens"] if use_soft else None),
            gain=float(soft_gain),
            temperature=float(temperature),
            do_sample=bool(sampling),
            max_new_tokens=120,
        )
        # ✂️ 요약만 남기기
        text = _clean_generated(text)
        text = _enforce_min_words(text, min_words=120)   # ← 최소 80단어 보장(필요시 아래 유틸 참조)
        text = _degenericize(text, session.get("dv_hint",""), session.get("ov_hint",""))

        text = _strip_admin_and_hint(text)
        return (text or "No summary generated.").strip()

    if intent == "victim":
        ev = session.get("dc_f")
        ov = session.get("ov_f")
        basis = session.get("basis", int(fault_basis))
        # 누가 더 과실이 큰지
        if ev is None or ov is None:
            return "Unable to determine who is at fault from the current frames."

        # if ev < ov:
        #     # OV 과실이 더 큼 → EV가 피해자
        #     who_line = "Based on the visible maneuvers, the ego vehicle appears to be the victim, and the other vehicle appears primarily at fault."
        # elif ev > ov:
        #     # EV 과실이 더 큼 → OV가 피해자
        #     who_line = "Based on the visible maneuvers, the other vehicle appears to be the victim, and the ego vehicle appears primarily at fault."
        # else:
        #     who_line = "From the visible evidence, it is difficult to determine the victim and the at-fault party."

        seed_key = f"{session.get('video_name','')}-{ev:.2f}-{ov:.2f}"
        who_line = pick_who_line(ev, ov, seed_key=None) 

        # (선택) 한 문장 더: 근거를 1문장으로 정리
        why_draft = ENGINE.generate_video_only(
            frames=frames,
            chat_prompt=build_ratio_draft_prompt(),
            temperature=max(0.2, float(temperature)),
            do_sample=bool(sampling),
            max_new_tokens=70,
        )
        why_line = _normalize_why(why_draft)  # 1–2문장으로 압축, 숫자/신호 단어 제거

        # 스타일에 따라 한 문장만 / 한두 문장 반환
        if style == "short":
            return who_line
        else:
            # 필요시 비어있으면 한 문장만
            return who_line if not why_line else f"{who_line} {why_line}"

    if intent == "ratio":
        # 1) 숫자 한 줄 (Fault-BERT 결과; basis 반영되어 있어야 함)
        ratio_line = f"Split-Liability Ratio: EV {session['dc_f']:.1f} / OV {session['ov_f']:.1f}"

        # 2) 비디오만으로 '이유 초안' 생성 (환각 방지 bad_words는 엔진 내부에서 이미 사용)
        draft = ENGINE.generate_video_only(
            frames=frames,
            chat_prompt=build_ratio_draft_prompt(),
            temperature=max(0.2, float(temperature)),
            do_sample=bool(sampling),
            max_new_tokens=90,
        )

        # 3) 소프트 토큰으로 '이유'만 짧게 보정 (힌트 충돌 시 비디오 우선)
        revise_prompt = (
            "Revise the DRAFT into 1–2 sentences that justify the allocation using ONLY visible entry order, "
            "turning/straight trajectories, and the conflict point. "
            "Do NOT output any headings or numbers. "
            "If the hint conflicts with the frames, prefer the video evidence.\n"
            f"DRAFT:\n{draft}\n"
            f"<hint>\n{session.get('fault_hint','')}\n</hint>\n"
        )
        revised = ENGINE.revise_with_soft_tokens(
            frames=frames,
            chat_prompt=revise_prompt,
            soft_tokens=(session["soft_tokens"] if use_soft else None),
            gain=float(soft_gain),
            temperature=max(0.2, float(temperature)),
            do_sample=bool(sampling),
            max_new_tokens=80,
        )

        # 4) 후처리: 라벨/힌트/꼬리문 제거 + 1–2문장 정리 + 금지어 제거
        explain = _normalize_why(revised)

        return ratio_line if not explain else (ratio_line + "\n" + explain)
    # fallback: 안전하게 간결 요약
    draft_prompt = build_video_only_prompt()
    draft = ENGINE.generate_video_only(
        frames=frames, chat_prompt=draft_prompt,
        temperature=float(temperature), do_sample=bool(sampling), max_new_tokens=180
    )
    return draft.strip() or "No description generated."

def _clean_generated(text: str) -> str:
    if not text:
        return ""
    # 과도한 프롬프트 제거 로직 축소
    text = re.sub(r"(?is)\b(USER|ASSISTANT):.*", "", text).strip()
    # 문장 2~3개 유지
    sents = re.split(r"(?<=[.!?])\s+", text)
    text = " ".join(sents[:3]).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text

def _enforce_two_line_revise(text: str) -> str:
    # Keep only the first “Description:” and “Evidence:” lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    desc = next((l for l in lines if l.lower().startswith("description:")), "")
    evid = next((l for l in lines if l.lower().startswith("evidence:")), "")
    # Fallbacks if model didn’t prefix correctly
    if not desc:
        desc = "Description: A collision scenario is visible based on motion and entry order."
    if not evid:
        evid = "Evidence: Paths cross near the center of the intersection."
    # Ensure no truncation mid-word
    if not re.search(r"[.!?]$", desc):
        desc += "."
    if not re.search(r"[.!?]$", evid):
        evid += "."
    return desc + "\n" + evid

def _reset_session_analysis(session: dict):
    """DV/OV 분류·Fault-BERT·소프트토큰 등 '분석 결과' 캐시만 초기화."""
    if session is None:
        return
    for k in [
        "dv_hint", "ov_hint", "topk_text",   # 분류기 힌트
        "fault_hint", "dc_f", "ov_f", "victim",  # Fault-BERT 결과
        "sentence", "soft_tokens"            # 템플릿 문장/소프트토큰
    ]:
        session[k] = None
# =========================================================
# Fault-BERT (.pth) - 네가 학습한 모델로부터 임베딩을 뽑는다
# =========================================================
class TextToFaultRatio(nn.Module):
    def __init__(self, model_name="bert-base-uncased", hidden_dim=768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.regressor(cls)

@torch.no_grad()
def load_fault_model(path: str, model_name="bert-base-uncased", device="cuda"):
    """네가 학습한 .pth를 로드하여 encoder까지 가중치가 반영된 Fault-BERT를 리턴"""
    obj = torch.load(path, map_location="cpu")
    model = TextToFaultRatio(model_name=model_name)
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
        # 필요시 prefix 정리
        new_sd = { (k.replace("module.", "")): v for k, v in sd.items() }
        model.load_state_dict(new_sd, strict=False)
    elif isinstance(obj, dict):
        new_sd = { (k.replace("module.", "")): v for k, v in obj.items() }
        model.load_state_dict(new_sd, strict=False)
    elif hasattr(obj, "state_dict"):
        model = obj
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

@torch.no_grad()
def predict_fault_ratio(model, tokenizer, text: str, device="cuda", max_length=256, total=10.0, temperature: float = 1.0):
    inputs = tokenizer(text, return_tensors="pt",
                       padding="max_length", truncation=True, max_length=max_length).to(device)
    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])  # [1,2]
    probs = F.softmax(logits / temperature, dim=-1).squeeze(0).float().cpu().numpy()        # [2], sum=1
    return probs * float(total)  # sum=total

def project_pair_to_basis(v, total=10.0):
    v = np.maximum(np.asarray(v, dtype=float), 0.0)
    s = float(v.sum())
    if s <= 0:
        return np.array([total/2.0, total/2.0], dtype=float)
    return v * (total / s)


# -------------------------
# Label 텍스트
# -------------------------
real_categories_ids_2nd = {
  1 : "Changing Lanes Within Intersection",
  2 : "Lane Change Path Change",
  3 : "Lanes Wide Enough For Two Vehicles Side By Side",
  4 : "Main Road And Side Road",
  5 : "Other Vehicle Enters From The Side",
  6 : "Roads Of Equal Width",
  7 : "Start After Stopping Accident",
  8 : "Turning Angle Less Than 90 Degrees",
  9 : "Two Vehicles Turning Right Simultaneously"
}
dashcam_vehicle_info = {
    1 : "Facing Each Other Going Straight",
    2 : "Following Vehicle Going Straight After Leaving Safety Zone",
    3 : "Following Vehicle Going Straight Right Side Of Lane",
    4 : "Going Straight From Left Road",
    5 : "Going Straight From Main Road",
    6 : "Going Straight From Main Road Entered Earlier",
    7 : "Going Straight From Main Road Entered Later",
    8 : "Going Straight From Right Road",
    9 : "Going Straight From Side Road Left",
    10 : "Going Straight From Side Road Right",
    11 : "Going Straight From The Right",
    12 : "Going Straight From The Right Entered Earlier",
    13 : "Going Straight From The Right Entered Later",
    14 : "Going Straight Lane Change Inside Intersection",
    15 : "Green Light Going Straight",
    16 : "Green Light Going Straight Collided With Red Light Vehicle",
    17 : "Left Turn From Right Road",
    18 : "Left Turn From Side Road",
    19 : "Right Turn",
    20 : "Right Turn Entered Earlier",
    21 : "Right Turn Entered Later",
    22 : "Right Turn From Main Road",
    23 : "Right Turn From Main Road Entered Later",
    24 : "Right Turn From Side Road",
    25 : "Right Turn From Side Road Entered Earlier",
    26 : "Right Turn From Side Road Entered Later",
    27 : "Right Turn Right Lane",
    28 : "Simultaneous Lane Change",
    29 : "Yellow Light Going Straight",
    30 : "Yellow Light Left Turn Collided With Red Light Vehicle"
}
other_vehicle_info = {
  1 : "Departing After Stop",
  2 : "Facing Each Other Going Straight",
  3 : "Following Vehicle Going Straight",
  4 : "Following Vehicle Going Straight Left Side Of Lane",
  5 : "Following Vehicle Going Straight Right Side Of Lane",
  6 : "Going Straight From Left Road",
  7 : "Going Straight From Main Road",
  8 : "Going Straight From Main Road Entered Earlier",
  9 : "Going Straight From Main Road Entered Later",
  10 : "Going Straight From Right Road",
  11 : "Going Straight From Side Road Left",
  12 : "Going Straight From Side Road Right",
  13 : "Going Straight From The Right",
  14 : "Going Straight From The Right Entered Earlier",
  15 : "Going Straight From The Right Entered Later",
  16 : "Green Left Turn Signal Left Turn Collided With Red Light Vehicle",
  17 : "Green Light Going Straight",
  18 : "Left Turn From Right Road",
  19 : "Left Turn From Side Road",
  20 : "Right Turn",
  21 : "Right Turn Entered Earlier",
  22 : "Right Turn Entered Later",
  23 : "Right Turn From Main Road",
  24 : "Right Turn From Main Road Entered Earlier",
  25 : "Right Turn From Main Road Entered Later",
  26 : "Right Turn From Side Road",
  27 : "Right Turn From Side Road Entered Later",
  28 : "Right Turn Right Lane"
}
def _dict_to_list_by_id(d: dict): return [d[k] for k in sorted(d.keys())]
LABELS = {
    "dv": _dict_to_list_by_id(dashcam_vehicle_info),
    "pl": _dict_to_list_by_id(real_categories_ids_2nd),
    "ov": _dict_to_list_by_id(other_vehicle_info),
}
def render_template_from_labels(dv_text: str, ov_text: str) -> str:
    return (f"At an unsignalized intersection, the Ego Vehicle was {dv_text}, "
            f"while the Other Vehicle was {ov_text}.")

# =========================================================
# 비디오 분류기 (DV/OV 라벨 예측)
# =========================================================
# def build_video_only_prompt():
#     return (
#         "You are a traffic-accident legal analyst.\n"
#         "Describe ONLY what is VISIBLE in the frames.\n"
#         "No traffic lights, numbers, timestamps, speeds, lane counts, or unseen context.\n"
#         "Write 3–5 short factual sentences (70–120 words total) covering:\n"
#         "1) initial positions and approach directions of both vehicles,\n"
#         "2) entry order into the intersection or merge zone,\n"
#         "3) turning/straight trajectories and where the paths converge,\n"
#         "4) immediate visible outcome (e.g., evasive action, contact, stop).\n"
#         "Avoid vague phrases like 'parked' unless the vehicles are clearly stationary for the entire clip."
#     )
def build_video_only_prompt():
    return (
        "You are a traffic-accident legal analyst.\n"
        "Describe ONLY what is VISIBLE in the frames.\n"
        "Use only the terms **Ego Vehicle (EV)** and **Other Vehicle (OV)** to refer to vehicles.\n"
        "Never mention colors, makes, models, or generic labels like 'the car' or 'the truck'.\n"
        "No traffic lights, numbers, timestamps, speeds, lane counts, or unseen context.\n"
        "Write 3–5 short factual sentences (70–120 words total) covering:\n"
        "1) initial positions and approach directions of EV and OV,\n"
        "2) entry order into the intersection or merge zone,\n"
        "3) turning/straight trajectories and where the paths converge,\n"
        "4) immediate visible outcome.\n"
        "Avoid vague phrases like 'parked' unless vehicles are clearly stationary for the entire clip."
    )

def build_prompt(
    user_message: str,
    dashcam_info: Optional[str],
    other_info: Optional[str],
    classifier_topk: Optional[str],
    style: str = "brief",
    history: Optional[list] = None,
    history_max_turns: int = 6,
    fault_hint: Optional[str] = None,
) -> str:
    system_header = (
        "You are a **traffic-accident legal analyst**. "
        "Always prioritize **what is visible in the video**. "
        "If any hint (classification/fault estimate) conflicts with the frames, briefly note the conflict and follow the visual evidence. "
        "Do NOT invent traffic lights, numbers, timestamps, lane counts, speeds, parking or any unseen objects.\n"
    )

    if style == "short":
        length_rule = "Write exactly one concise sentence (~30 words)."
    elif style == "detailed":
        length_rule = "Write 4–6 factual sentences (~80–120 words)."
    else:
        style = "brief"
        length_rule = "Write 2–3 factual sentences (under ~90 words)."

    hint_lines = []
    if dashcam_info:
        hint_lines.append(f"- Ego Vehicle: {dashcam_info}")
    if other_info:
        hint_lines.append(f"- Other Vehicle: {other_info}")
    if classifier_topk:
        hint_lines.append(f"- Classifier outputs (top-k): {classifier_topk}")
    if fault_hint:
        hint_lines.append(f"- Fault analysis [Ego:Other]: {fault_hint}")
    hint_block = ("Include the following hints **exactly once** if they agree with visible evidence:\n" + "\n".join(hint_lines) + "\n") if hint_lines else ""

    # Include recent chat context (if any)
    hist_txt = ""
    if history:
        turns = history[-history_max_turns:]
        for m in turns:
            role = (m.get("role") or "").upper()
            msg  = m.get("content") or ""
            if msg:
                hist_txt += f"{role}:\n{msg}\n\n"

    user_header = (
        f"{hist_txt}"
        "USER:\n"
        "Describe the accident focusing on **motion, entry order, and relative positions** visible in the frames.\n"
        f"{length_rule}\n"
        f"{hint_block}"
        "Output requirements:\n"
        "1) **Description**: 2–3 factual sentences based ONLY on visible evidence.\n"
        "2) **Fault (basis N)**: predicted fault ratio (basis N, one decimal), and identify **Victim** (lower share) and **At-fault** vehicle.\n"
        "3) **Cause**: one-sentence cause grounded in visible evidence (and the hints if they agree).\n"
        "Avoid: traffic lights/signals, arbitrary numbers (except the fault ratio), dates/timestamps, unseen objects/lanes/speeds. "
        "If a hint conflicts with the frames, state the conflict briefly and follow the video evidence.\n"
    )
    if user_message and user_message.strip():
        user_header += f"\nAdditional instruction: {user_message.strip()}\n"

    placeholder = "<video>"
    return f"{placeholder}\n" + system_header + user_header + "ASSISTANT:"

# def build_revise_prompt(draft_text: str, user_message: str, fault_hint_line: str) -> str:
#     return (
#         "You are a traffic-accident legal analyst. Revise the DRAFT based ONLY on visible evidence in the frames. "
#         "If the hint agrees, reflect it ONCE; if it conflicts, briefly note the conflict and keep the visual evidence.\n"
#         "Return EXACTLY TWO LINES:\n"
#         "Description: <one or two concise factual sentences>\n"
#         "Evidence: <the single most decisive visible cue>\n"
#         f"DRAFT:\n{draft_text}\n"
#         f"HINT:\n{fault_hint_line}\n"
#         f"{('Additional instruction: ' + user_message.strip()) if user_message else ''}"
#     )
def build_revise_prompt(draft_text: str, user_message: str, fault_hint_line: str, session=None) -> str:
    return (
        "You are a traffic-accident legal analyst. Revise the DRAFT based ONLY on visible evidence in the frames. "
        "If the hint agrees, reflect it ONCE; if it conflicts, briefly note the conflict and keep the visual evidence.\n"
        "Description: <one or two concise factual sentences>\n"
        "Do NOT copy tag names literally (do not output 'HINT', 'hint', or XML tags).\n"
        f"DRAFT:\n{draft_text}\n"
        f"<hint>\n{fault_hint_line}\n</hint>\n"
        f"{('Additional instruction: ' + user_message.strip()) if user_message else ''}"
    )

def _ensure_video_token(processor, text: str) -> str:
    # 모델/프로세서에 따라 토큰 이름이 다를 수 있어 안전하게 처리
    video_token = getattr(processor, "video_token", None)
    if not video_token:
        video_token = getattr(processor, "image_token", None) or "<video>"
    if video_token not in text:
        return f"{video_token}\n{text}"
    return text

# =========================================================
# 분류기
# =========================================================
class TripleHeadVideoClassifier(nn.Module):
    def __init__(self, n_dv, n_ov, backbone="r3d18", pretrained=False):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "r3d18":
            self.backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1 if pretrained else None)
            feat = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "timesformer":
            self.backbone = TimesformerModel.from_pretrained(
                "facebook/timesformer-base-finetuned-k400", use_safetensors=True
            )
            feat = self.backbone.config.hidden_size
        elif backbone == "videomae":
            self.backbone = VideoMAEModel.from_pretrained(
                "MCG-NJU/videomae-base-finetuned-kinetics", use_safetensors=True
            )
            feat = self.backbone.config.hidden_size
        else:
            raise ValueError(backbone)

        self.dv = nn.Linear(feat, n_dv)
        self.ov = nn.Linear(feat, n_ov)

    def forward(self, x):
        if self.backbone_name == "r3d18":
            z = self.backbone(x)              # (B, feat)
        else:
            x = x.permute(0,2,1,3,4)          # (B,T,C,H,W)
            out = self.backbone(x)
            z = out.last_hidden_state.mean(1) # (B, feat)
        return self.dv(z), self.ov(z)

@torch.no_grad()
def load_video_tensor_for_clf(path, num_frames=16, size=224, device="cuda"):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"no frames in {path}")
    idxs = np.linspace(0, total-1, num_frames).astype(int)

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok: continue
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        fr = cv2.resize(fr, (size, size))
        fr = torch.from_numpy(fr).permute(2,0,1).float()/255.0  # (C,H,W)
        frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError(f"no readable frames in {path}")

    vid = torch.stack(frames, 0)                        # (T,C,H,W)
    vid = vid.permute(1,0,2,3).unsqueeze(0).to(device)  # (1,C,T,H,W)

    # 디바이스/정규화 안정화
    # mean = torch.tensor([0.485, 0.456, 0.406], device=vid.device).view(1,3,1,1)
    # std  = torch.tensor([0.229, 0.224, 0.225], device=vid.device).view(1,3,1,1)
    # vid = (vid - mean) / std
    mean = torch.tensor([0.485, 0.456, 0.406], device=vid.device).view(1, 3, 1, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=vid.device).view(1, 3, 1, 1, 1)
    vid = (vid - mean) / std
    return vid

def load_classifier(ckpt_path: str, backbone="r3d18", device="cuda", pretrained=False):
    model = TripleHeadVideoClassifier(
        n_dv=len(LABELS["dv"]), n_ov=len(LABELS["ov"]),
        backbone=backbone, pretrained=pretrained
    )
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    new_sd = { (k[len("module."):] if k.startswith("module.") else k): v for k,v in sd.items() }
    model.load_state_dict(new_sd, strict=False)
    model.to(device).eval()
    return model

@torch.no_grad()
def predict_classifier(video_path, clf_model, device="cuda", topk=3):
    x = load_video_tensor_for_clf(video_path, num_frames=16, size=224, device=device)
    ld, lv = clf_model(x)   # dv, ov

    def _topk_text(logits, label_list, k):
        probs = logits.softmax(dim=-1)[0]
        val, idx = probs.topk(k=k, dim=-1)
        names = [label_list[i] for i in idx.tolist()]
        top1 = names[0]
        return top1, ", ".join(names)

    dv1, dv_topk = _topk_text(ld, LABELS["dv"], topk)
    ov1, ov_topk = _topk_text(lv, LABELS["ov"], topk)

    dashcam_info   = dv1
    other_info     = ov1
    classifier_top = f"DV[{dv_topk}] | OV[{ov_topk}]"
    return dashcam_info, other_info, classifier_top

# =========================================================
# 파일/비디오 유틸
# =========================================================
def list_candidates(limit=50):
    items = []
    for root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        for f in os.listdir(root):
            fname = f.lower()
            for ext in VIDEO_EXTS:
                if fname.endswith(ext):
                    items.append(os.path.join(root, f))
    items = [p for p in items if os.path.isfile(p)]
    items.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return items[:limit]

def find_video_path_by_name(name: str) -> Optional[str]:
    name = name.strip().strip('"').strip("'")
    if not name:
        return None
    has_ext = os.path.splitext(name)[1].lower() in VIDEO_EXTS

    for root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        if any(ch in name for ch in "*?[]"):
            patterns = [os.path.join(root, name)]
        else:
            patterns = [os.path.join(root, name)] if has_ext else [os.path.join(root, name + ext) for ext in VIDEO_EXTS]
        for pat in patterns:
            hits = sorted(glob.glob(pat))
            if hits:
                hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return hits[0]
    return None

def _ensure_upload_dir():
    upload_dir = "/mnt/data/videos"
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir

def on_video_changed(video):
    """
    gr.Video에 사용자가 파일을 넣으면:
    - /mnt/data/videos/ 로 복사
    - video_name Textbox 에는 베이스파일명만 세팅
    - 좌측 미리보기는 그대로 유지(재생 가능)
    """
    try:
        upload_dir = _ensure_upload_dir()
        # 그라도 입력은 dict 또는 str일 수 있음
        src = None
        if isinstance(video, str) and os.path.exists(video):
            src = video
        elif isinstance(video, dict):
            # gradio가 넘기는 dict 스키마 호환
            cand = video.get("path") or video.get("name") or video.get("video")
            if cand and os.path.exists(cand):
                src = cand

        if not src:
            # 업로드가 제거된 경우 등
            return gr.update(), "⚠️ No video selected."

        base = os.path.basename(src)
        dst = os.path.join(upload_dir, base)
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)

        # Textbox(video_name) 업데이트, 상태 메시지(여기선 load_status 재활용)
        return base, f"📹 Loaded: {base} (saved to /mnt/data/videos)"
    except Exception as e:
        return gr.update(), f"❌ Video load error: {e}"

def _resolve_video_path(video, video_name: Optional[str]) -> Optional[str]:
    # gr.Video는 dict가 들어오기도 함
    path = None
    if isinstance(video, str) and os.path.exists(video):
        path = video
    elif isinstance(video, dict):
        cand = video.get("name") or video.get("video") or video.get("path")
        if cand and os.path.exists(cand):
            path = cand
    if not path and video_name:
        cand = find_video_path_by_name(video_name)
        if cand and os.path.exists(cand):
            path = cand
    if not path and video_name and os.path.exists(video_name):
        path = video_name
    return path

def pick_video_and_preview(selected):
    """
    드롭다운에서 선택 → video_name에 파일명만, video에는 재생가능 경로 세팅
    """
    if not selected:
        return gr.update(), gr.update()
    # selected 는 전체경로일 가능성이 높음. 그래도 안전하게 확인.
    path = selected if os.path.exists(selected) else find_video_path_by_name(selected)
    if not path or not os.path.exists(path):
        return gr.update(), gr.update()

    base = os.path.basename(path)
    # gr.Video에 경로를 value로 주면 곧바로 재생 가능
    return base, gr.update(value=path)

# ---------------- Frame Sampler ----------------
def sample_frames(video_path: str, num_frames: int = 8, size: int = 224) -> List[Image.Image]:
    frames = []
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        if total <= 0:
            raise RuntimeError("Empty video (decord).")
        idxs = (np.linspace(0, total - 1, num_frames)).astype(int).tolist()
        for i in idxs:
            arr = vr[i].asnumpy()  # Decord는 RGB 반환 (BGR 변환 금지)
            frames.append(Image.fromarray(arr).resize((size, size)))
    except Exception:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise RuntimeError("Empty/unreadable video (OpenCV).")
        idxs = (np.linspace(0, total - 1, num_frames)).astype(int).tolist()
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if not ok:
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8)))
                continue
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(fr).resize((size, size)))
        cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames[:num_frames]

# ---------------- Prompt Builders ----------------
def _ensure_pad_token(tok, model):
    eos_id = tok.eos_token_id or getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if tok.pad_token_id is None:
        if eos_id is not None:
            tok.pad_token_id = eos_id
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tok))
    return tok.pad_token_id, (tok.eos_token_id or eos_id)

def _apply_chat_template(processor, user_text: str) -> str:
    """
    VideoLLaVA Processor/Tokenizer에 chat template가 없으면 조용히 폴백한다.
    """
    # 1) 메시지 포맷
    messages = [{
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text": user_text},
        ],
    }]

    # 2) 템플릿 함수 탐색
    apply_fn = getattr(processor, "apply_chat_template", None)
    if apply_fn is None:
        apply_fn = getattr(getattr(processor, "tokenizer", None), "apply_chat_template", None)

    # 3) 없으면 폴백: <video> 토큰만 보장
    if apply_fn is None:
        return _ensure_video_token(processor, user_text)

    # 4) 있으면 사용 (여기서도 예외 터지면 폴백)
    try:
        return apply_fn(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return _ensure_video_token(processor, user_text)

def _find_video_token_insert_pos(processor, input_ids: torch.Tensor) -> int:
    """비디오 토큰이 있으면 그 바로 뒤에 soft token 삽입"""
    vid_token = getattr(processor, "video_token", None) or getattr(processor, "image_token", None)
    if not vid_token:
        return 2
    tok = processor.tokenizer
    vid_ids = tok(vid_token, add_special_tokens=False).input_ids
    ids = input_ids[0].tolist()
    # 간단 탐색(단일 토큰 가정)
    if len(vid_ids) == 1:
        tid = vid_ids[0]
        for i, t in enumerate(ids):
            if t == tid:
                return i + 1
    # 못 찾으면 기본값
    return 2

# =========================================================
# Video-LLaVA 엔진 + soft token 삽입 경로
# =========================================================
class VideoLLaVAChatEngine:
    def __init__(self):
        self.model_id = None
        self.processor = None
        self.model = None
        self.device = self._pick_device()

        # BERT-soft 토큰 투영기 (Fault-BERT hidden -> LLaVA hidden)
        self._hint_proj: Optional[nn.Module] = None
        self._hint_proj_dim: Optional[int] = None

    def _pick_device(self):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self, model_id: str):
        if self.model_id == model_id and self.model is not None:
            return f"Model already loaded: {model_id}"
        self.processor = VideoLlavaProcessor.from_pretrained(model_id)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = VideoLlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=None,
        ).to(self.device)
        self.model.config.use_cache = False
        self.model_id = model_id
        return f"Loaded model: {model_id}"

    def _ensure_hint_proj(self, llm_hidden_size: int, bert_hidden_size: int):
        if (self._hint_proj is None) or (self._hint_proj_dim != llm_hidden_size):
            self._hint_proj = nn.Sequential(
                nn.Linear(bert_hidden_size, llm_hidden_size),
                nn.Tanh(),
                nn.LayerNorm(llm_hidden_size)
            ).to(self.device).eval()
            self._hint_proj_dim = llm_hidden_size
        return self._hint_proj

    @torch.no_grad()
    def make_soft_tokens_from_trained_fault(
        self,
        fault_m: TextToFaultRatio,
        fault_tok: AutoTokenizer,
        text: str,
        take: str = "cls",
        max_tokens: int = 4
    ) -> torch.Tensor:
        """Fault-BERT encoder 히든을 LLaVA 차원으로 투영 → soft tokens [1,M,H]"""
        tok_inputs = fault_tok(text, return_tensors="pt", truncation=True, max_length=256).to(self.device)
        hs = fault_m.encoder(**tok_inputs).last_hidden_state         # [1, Lb, Hb]
        Hb = hs.size(-1)
        H  = self.model.get_input_embeddings().embedding_dim
        proj = self._ensure_hint_proj(H, Hb)

        hs = hs.to(next(proj.parameters()).dtype)

        if take == "cls":
            z = hs[:, 0, :]
            s = proj(z).unsqueeze(1)                                 # [1,1,H]
        elif take == "mean":
            z = hs.mean(dim=1)
            s = proj(z).unsqueeze(1)
        else:
            K = min(hs.size(1), max_tokens)
            z = hs[:, :K, :]
            s = proj(z)

        max_norm = 2.0
        s_norm = s.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        s = s * (max_norm / s_norm).clamp_max(1.0)

        s = s.to(self.model.dtype)
        return s

    @torch.no_grad()
    def generate_video_only(self, frames: List[Image.Image], chat_prompt: str,
                            temperature: float = 0.2, do_sample: bool = False,
                            max_new_tokens: int = 256) -> str:
        model = self.model
        processor = self.processor
        device = self.device
        tok = processor.tokenizer

        # 비디오 토큰 보장
        safe_prompt = _apply_chat_template(processor, chat_prompt)

        # tokenizer pad/eos 안전장치
        pad_id, eos_id = _ensure_pad_token(tok, model)

        # 템플릿 없이 평문 + 비디오만
        proc = processor(videos=[frames], text=[safe_prompt],         # ✅ 변경
                            max_length=2048, padding="longest", truncation=False, return_tensors="pt")

        if "pixel_values_videos" in proc:
            vision = {"pixel_values_videos": proc["pixel_values_videos"].to(device, dtype=model.dtype)}
        else:
            vision = {"pixel_values_videos": proc["pixel_values"].to(device, dtype=model.dtype)}

        input_ids = proc["input_ids"].to(device)
        attn      = proc["attention_mask"].to(device)

        bad_words = ["traffic light","signalized","timestamp","AM","PM"]
        bad_ids = []
        for w in bad_words:
            ids = tok(w, add_special_tokens=False).input_ids
            if ids:  # 빈 리스트 방지
                bad_ids.append(ids)

        gen_ids = model.generate(
            **vision,
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=1,
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
            bad_words_ids=bad_ids if bad_ids else None,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
        gen_only = gen_ids[0, input_ids.shape[1]:]
        text_out = tok.decode(gen_only, skip_special_tokens=True).strip()

        # >>> NEW: never fall back to full decode (that re-inserts the prompt)
        text_out = _clean_generated(text_out)
        if not text_out:
            # one retry with mild sampling to avoid empty output
            gen_ids = model.generate(
                **vision,
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                temperature=max(0.3, temperature),
                do_sample=True,
                num_beams=1,
                no_repeat_ngram_size=3,
                repetition_penalty=1.05,
                bad_words_ids=bad_ids if bad_ids else None,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )
            gen_only = gen_ids[0, input_ids.shape[1]:]
            text_out = _clean_generated(tok.decode(gen_only, skip_special_tokens=True).strip())
        return text_out or "The vehicles converge in the intersection based on their visible paths."

    @torch.no_grad()
    def generate_with_soft_tokens_from_fault(
        self,
        frames: List[Image.Image],
        chat_prompt: str,
        soft_tokens: Optional[torch.Tensor],   # Optional
        insert_pos: int = 2,
        temperature: float = 0.2,
        do_sample: bool = False,
        max_new_tokens: int = 256,
    ) -> str:
        model = self.model
        processor = self.processor
        device = self.device
        tok = processor.tokenizer

        # 안전한 프롬프트(비디오 토큰 보장)
        safe_prompt = _ensure_video_token(processor, chat_prompt)

        pad_id, eos_id = _ensure_pad_token(tok, model)

        proc = processor(videos=[frames], text=[safe_prompt],       # ✅ 변경
                        max_length=2048, padding="longest", truncation=False, return_tensors="pt")

        if "pixel_values_videos" in proc:
            vision = {"pixel_values_videos": proc["pixel_values_videos"].to(device, dtype=model.dtype)}
        else:
            vision = {"pixel_values_videos": proc["pixel_values"].to(device, dtype=model.dtype)}

        input_ids = proc["input_ids"].to(device)
        attn      = proc["attention_mask"].to(device)

        bad_words = ["traffic light","signalized","timestamp","AM","PM"]
        bad_ids = [tok(w, add_special_tokens=False).input_ids
                   for w in bad_words if tok(w, add_special_tokens=False).input_ids]

        # (A) 소프트 토큰 미사용
        if soft_tokens is None:
            gen_ids = model.generate(
                **vision,
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                num_beams=1,
                no_repeat_ngram_size=3,
                repetition_penalty=1.05,
                bad_words_ids=bad_ids if bad_ids else None,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )
            gen_only = gen_ids[0, input_ids.shape[1]:]               # ✨
            text_out = tok.decode(gen_only, skip_special_tokens=True).strip()
            if not text_out:
                text_out = tok.decode(gen_ids[0], skip_special_tokens=True).strip()
            return text_out

        # (B) 소프트 토큰 사용
        text_embeds = model.get_input_embeddings()(input_ids).to(device, dtype=model.dtype)
        S = soft_tokens.to(device, dtype=model.dtype)

        # 비디오 토큰 바로 뒤에 삽입 시도
        pos = _find_video_token_insert_pos(processor, input_ids)
        L = text_embeds.size(1)
        pos = max(1, min(pos, L - 1))

        inputs_embeds = torch.cat([text_embeds[:, :pos, :], S, text_embeds[:, pos:, :]], dim=1)
        new_mask = torch.cat([
            attn[:, :pos],
            torch.ones(attn.size(0), S.size(1), device=device, dtype=attn.dtype),
            attn[:, pos:]
        ], dim=1)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            min_new_tokens=80,          # ✨ 최소 생성 길이
            early_stopping=False,
            num_beams=1,
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
        if do_sample:
            gen_kwargs.update(dict(do_sample=True, temperature=temperature))

        # ❗ inputs_embeds 사용 시에는 보통 출력이 "생성 토큰만" 반환됩니다.
        gen_ids = model.generate(**vision, inputs_embeds=inputs_embeds, attention_mask=new_mask, **gen_kwargs)

        # ❗❗ 슬라이스 하지 말고 바로 디코딩
        text_out = tok.decode(gen_ids[0], skip_special_tokens=True).strip()
        if not text_out:
            text_out = tok.decode(gen_ids[0], skip_special_tokens=False).strip()
        return text_out

    @torch.no_grad()
    def revise_with_soft_tokens(self, frames: List[Image.Image], chat_prompt: str,
                                soft_tokens: Optional[torch.Tensor], gain: float = 0.5,
                                temperature: float = 0.2, do_sample: bool = False,
                                max_new_tokens: int = 180) -> str:
        # gain으로 세기 조절
        if soft_tokens is not None:
            soft_tokens = soft_tokens * float(gain)
        return self.generate_with_soft_tokens_from_fault(
            frames=frames,
            chat_prompt=chat_prompt,
            soft_tokens=soft_tokens,
            insert_pos=2,
            temperature=temperature,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
        )

ENGINE = VideoLLaVAChatEngine()

# =========================================================
# Gradio UI
# =========================================================
def ui_load_model(model_id):
    try:
        msg = ENGINE.load_model(model_id)
    except Exception as e:
        msg = f"Load error: {e}"
    return msg

_CLF_CACHE = {}
_FAULT_CACHE = {}

def _ensure_classifier(ckpt_path, backbone, pretrained, device):
    key = (ckpt_path, backbone, bool(pretrained), device)
    if key not in _CLF_CACHE:
        mdl = load_classifier(ckpt_path, backbone=backbone, device=device, pretrained=pretrained)
        _CLF_CACHE[key] = mdl
    return _CLF_CACHE[key]

def _ensure_fault(ckpt_path, model_name, device):
    key = (ckpt_path, model_name, device)
    if key not in _FAULT_CACHE:
        mdl, tok = load_fault_model(ckpt_path, model_name=model_name, device=device)
        _FAULT_CACHE[key] = (mdl, tok)
    return _FAULT_CACHE[key]

def _append_assistant_once(history, text: str):
    """마지막 assistant 버블과 동일하면 추가하지 않음(디듀프)."""
    if history and isinstance(history[-1], dict):
        if history[-1].get("role") == "assistant" and history[-1].get("content", "") == (text or ""):
            return history
    return history + [{"role": "assistant", "content": (text or "")}]

def ui_turn(
        history, session, user_msg, video, video_name, model_id,
        clf_ckpt, clf_backbone, clf_pretrained, clf_topk,
        fault_ckpt, fault_model, fault_basis,
        style, num_frames, frame_size, temperature, sampling,
        use_soft, interp_mode, soft_gain,
        skip_user_echo=False   # ✅ 버튼 체인에서는 True
    ):
    print("[ui_turn] fired")
    history = history or []
    session = session or _default_session()

    # ---- 실행 중복 방지 락 ----
    if session.get("busy"):
        # 이미 실행 중이면 아무 것도 추가하지 않고 그대로 반환
        return history, session, history
    session["busy"] = True

    try:
        # 0) 모델 로드(한 번만)
        try:
            msg = ENGINE.load_model(model_id)
            session["model_id"] = model_id
        except Exception as e:
            history = _append_assistant_once(history, f"❌ Model load error: {e}")
            return history, session, history

        # 1) 영상 경로 확정
        if session["video_path"] is None:
            vpath = _resolve_video_path(video, video_name)
            if not vpath or (not os.path.exists(vpath)) or os.path.getsize(vpath) == 0:
                history = _append_assistant_once(
                    history,
                    f"❌ Video not found or unreadable. name='{video_name}'\n검색 루트: {SEARCH_ROOTS}"
                )
                return history, session, history
            session["video_path"] = vpath
            session["video_name"] = os.path.basename(vpath)

        # 2) 분석 캐시 준비
        device = ENGINE.device
        _ensure_analysis(session, device, session["video_path"], clf_ckpt, clf_backbone, clf_pretrained,
                         clf_topk, fault_ckpt, fault_model, fault_basis, num_frames, frame_size)

        # 3) 프레임 샘플
        frames = sample_frames(session["video_path"], num_frames=int(num_frames), size=int(frame_size))
        print(f"[frames] sampled={len(frames)} size={frame_size}", flush=True)

        # 4) 의도 라우팅 + 답변 생성
        intent = _intent_from_message(user_msg or "")
        text = _answer_for_intent(
            intent=intent,
            frames=frames,
            history=history,          # 참고용
            style=style,
            temperature=temperature,
            sampling=sampling,
            use_soft=use_soft,
            soft_gain=soft_gain,
            fault_basis=fault_basis,
            session=session
        )

        # 5) 히스토리 업데이트
        # 버튼 체인에서는 이미 사용자 버블 출력했으므로 중복 방지
        if user_msg and not skip_user_echo:
            history = history + [{"role": "user", "content": user_msg}]
        history = _append_assistant_once(history, text)

        return history, session, history
    finally:
        session["busy"] = False


def ui_generate(
        history, user_msg, video, video_name, model_id,
        clf_ckpt, clf_backbone, clf_pretrained, clf_topk,
        fault_ckpt, fault_model, fault_basis,
        style, num_frames, frame_size, temperature, sampling,
        use_soft, interp_mode, soft_gain
    ):
    history = history or []

    # 0) 비디오 찾기
    video_path = _resolve_video_path(video, video_name)
    print(f"[Chat] video_path={video_path}, video_name={video_name}")
    if not video_path:
        history = history + [{"role":"assistant","content": f"❌ Video not found. name='{video_name}'\n검색 루트: {SEARCH_ROOTS}"}]
        return history
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        history = history + [{"role":"assistant","content": f"❌ Video not readable: {video_path}"}]
        return history

    device = ENGINE.device

    # 1) DV/OV 분류 → top-k 텍스트
    try:
        clf = _ensure_classifier(clf_ckpt, clf_backbone, clf_pretrained, device)
        dv_hint, ov_hint, topk_text = predict_classifier(
            video_path, clf, device=device, topk=int(clf_topk)
        )
    except Exception as e:
        dv_hint, ov_hint = "UNKNOWN_EV", "UNKNOWN_OV"
        topk_text = f"[classifier error: {e}]"
    print(f"[Hint] DV={dv_hint} | OV={ov_hint} | TOPK={topk_text}")

    # 2) 템플릿 문장
    sentence = render_template_from_labels(dv_hint, ov_hint)

    # 3) Fault-BERT 추론
    try:
        fr_m, fr_tok = _ensure_fault(fault_ckpt, fault_model, device)
        y = predict_fault_ratio(fr_m, fr_tok, sentence, device=device)  # [2]
        dc_f, ov_f = float(y[0]), float(y[1])
        if abs(dc_f - ov_f) < 1e-6:
            victim = "Undetermined"
        else:
            victim = "Ego Vehicle" if dc_f < ov_f else "Other Vehicle"
        fault_hint = f"Predicted fault (basis {int(fault_basis)}): Ego={dc_f:.1f}, Other={ov_f:.1f}; Likely victim: {victim}; Template: {sentence}"
    except Exception as e:
        fault_hint = f"[fault error: {e}]; Template: {sentence}"
        dc_f, ov_f, victim = 5.0, 5.0, "Undetermined"


    # 4) Video-LLaVA 로드
    load_msg = ENGINE.load_model(model_id)
    print(load_msg)

    # 5) 프롬프트 + 프레임 준비
    frames = sample_frames(video_path, num_frames=int(num_frames), size=int(frame_size))
    prompt_text = build_prompt(
        user_message=user_msg or "",
        dashcam_info=dv_hint,
        other_info=ov_hint,
        classifier_topk=(None if interp_mode else topk_text),
        style=style,
        history=history,
        fault_hint=fault_hint
    )
    print("\n========== INITIAL ANALYSIS (ONCE) ==========", flush=True)
    print(f"[SUMMARY]\n{dv_hint}\n{ov_hint}\n", flush=True)
    print(f"[RATIO]\n{fault_hint}", flush=True)

    # 6) Soft-token 준비(옵션)
    bert_hint_text = f"Ego={dv_hint}; Other={ov_hint}; {fault_hint}"
    soft = None
    if use_soft:
        try:
            fr_m, fr_tok = _ensure_fault(fault_ckpt, fault_model, device)
            soft = ENGINE.make_soft_tokens_from_trained_fault(
                fault_m=fr_m, fault_tok=fr_tok, text=bert_hint_text, take="cls", max_tokens=4
            )
        except Exception as e:
            soft = None
            print(f"[SoftToken] error: {e}")

    # 7) 생성 경로: (A) 해석가능성 2패스 or (B) 단일패스
    try:
        if interp_mode:
            draft_prompt = build_video_only_prompt()
            draft = ENGINE.generate_video_only(
                frames=frames, chat_prompt=draft_prompt,
                temperature=float(temperature), do_sample=bool(sampling), max_new_tokens=150
            )
            revise_prompt = build_revise_prompt(draft, user_msg or "", fault_hint)
            text = ENGINE.revise_with_soft_tokens(
                frames=frames,
                chat_prompt=revise_prompt,
                soft_tokens=soft,
                gain=float(soft_gain),
                temperature=float(temperature),
                do_sample=bool(sampling),
                max_new_tokens=120,
            )
            aux_header = f"📝 DRAFT (video-only): {draft}\n✏️ REVISED: {text}\n⚙️ soft_gain={float(soft_gain):.2f}, use_soft={bool(use_soft)}"
        else:
            text = ENGINE.generate_with_soft_tokens_from_fault(
                frames=frames,
                chat_prompt=prompt_text,
                soft_tokens=soft,
                insert_pos=2,
                temperature=float(temperature),
                do_sample=bool(sampling),
                max_new_tokens=180,
            )
            aux_header = f"📝 {sentence}"
    except Exception as e:
        text = f"Generation error: {e}"
        aux_header = ""

    # 8) 채팅창 출력 (send_btn 경로: 여기서 유저 버블도 추가)
    aux_tail = f"\n⚖️ Fault(basis {int(fault_basis)}): DC {dc_f:.1f} / OV {ov_f:.1f} | Victim: {victim}\n"
    if user_msg:
        history = history + [{"role": "user", "content": user_msg}]
    main = (text or "").strip()
    if not main:
        main = "⚠️ No description generated."
    meta = (aux_tail + aux_header).strip()
    bubble = main if not meta else f"{main}\n\n{meta}"
    history = history + [{"role": "assistant", "content": bubble}]
    return history


# ✅ 사용자 버블을 먼저 찍는 헬퍼 (버튼 클릭 시 사용)
def ui_push_user(history, session, intent_text):
    """사용자 버블을 먼저 찍고, 동일 intent_text를 다음 체인에 넘긴다.
       state와 chatbot을 동시에 갱신하기 위해 history를 두 번 반환한다."""
    history = history or []
    session = session or _default_session()
    new_history = history + [{"role": "user", "content": intent_text}]
    # 반환 순서: state, session, chatbot, user_msg
    return new_history, session, new_history, intent_text


# =========================================================
# Gradio App
# =========================================================
def _fill_dropdown():
    items = list_candidates()
    return gr.Dropdown.update(choices=items, value=(items[0] if items else None))

# 선택한 파일 경로를 미리보기 + 파일명만 세팅
def pick_video_and_preview(picked_path: str):
    if not picked_path:
        return gr.update(value=""), gr.update()  # no-op
    fname = os.path.basename(picked_path)
    return fname, gr.Video.update(value=picked_path, autoplay=True)

# 업로드/드래그된 비디오를 /mnt/data/videos에 복사하고 파일명만 세팅
def on_video_changed(video_event):
    # gr.Video change 이벤트는 dict/str 등으로 들어올 수 있음
    src = None
    if isinstance(video_event, dict):
        # gradio가 넘겨주는 필드 중 사용 가능한 것 우선
        src = video_event.get("name") or video_event.get("video") or video_event.get("path")
    elif isinstance(video_event, str):
        src = video_event
    if not src or not os.path.exists(src):
        return gr.update(value=""), "❌ Uploaded video not readable."

    os.makedirs("/mnt/data/videos", exist_ok=True)
    dst = os.path.join("/mnt/data/videos", os.path.basename(src))
    try:
        if os.path.abspath(src) != os.path.abspath(dst):
            import shutil
            shutil.copy2(src, dst)
        msg = f"✅ Uploaded & registered: {dst}"
    except Exception as e:
        msg = f"⚠️ Copy failed (using source directly): {e}"
        dst = src

    return os.path.basename(dst), msg

# 샘플 선택: 서버에는 파일명만 넘기고, 채팅창에 미리보기 추가
def _use_sample(session, chatbot, sample_path):
    session = session or _default_session()
    session["video_path"] = None
    session["video_name"] = os.path.basename(sample_path)
    _reset_session_analysis(session)
    # chat = (chatbot or []) + [_assistant_video_preview_bubble(sample_path, f"Selected {session['video_name']}")]
    note = f"Selected **{session['video_name']}**. Use **🔍 Enlarge** to preview."
    chat = (chatbot or []) + [_assistant_text_bubble(note)]

    return session, chat, session["video_name"]

FIGMA_URL = "https://www.figma.com/proto/84JeG8TuIXS1nbqvp0kw4U/%EC%A1%B8%EC%A0%84%ED%8C%90%EB%84%AC?page-id=245%3A2&node-id=245-119&viewport=116%2C366%2C0.11&t=b76hxXeN7oTO1v2r-1&scaling=contain&content-scaling=fixed&starting-point-node-id=245%3A119"

with gr.Blocks(title="Video-LLaVA Chatbot") as demo:
    # with gr.Row():
    #     ppt_btn = gr.Button("📑 PPT로 이동 (Figma)", variant="primary", elem_id="ppt-open-btn")
    #     ppt_btn.click(
    #         fn=None,                   # ← 파이썬 함수 안 씀
    #         inputs=[],
    #         outputs=[],
    #         # ← JS는 반드시 화살표 함수 형태여야 함
    #         js=f"() => {{ window.open({json.dumps(FIGMA_URL)}, '_blank'); }}",
    #         queue=False,
    #         show_progress="hidden",
    #     )
    # gr.Markdown("## 🎥 Video-LLaVA Chatbot — Multi-turn video-first accident analysis")
    title_md = gr.Markdown(
        "## 🎥 Video-LLaVA Chatbot — Multi-turn video-first accident analysis",
        elem_id="app-title"
    )

    gr.HTML("""
    <style>
    /* 샘플 리스트 스크롤 */
    #sample-list {
        max-height: 250px;   /* 필요 시 높이 조절 */
        overflow-y: auto;
        padding-right: 6px;
    }
    /* 스크롤바 살짝 보이게(선택) */
    #sample-list::-webkit-scrollbar { width: 8px; }
    #sample-list::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.25);
        border-radius: 6px;
    }
    /* 각 아이템 간격(선택) */
    .sample-item { margin-bottom: 8px; }
            
    /* 제목 아래 여백 압축 */
    #app-title { 
        margin: 0 !important; 
        padding: 0 !important; 
    }
    #app-title h2 {
        margin: 0 !important;          /* 기본 h2 margin 제거 */
        line-height: 1.2 !important;   /* 높이도 살짝 압축 */
    }

    /* 제목 바로 다음 섹션과의 간격도 줄이고 싶다면: */
    #app-title + .gradio-container, 
    #app-title + div {
        margin-top: 6px !important;
    }
            
    #howto-body {
        background: linear-gradient(180deg, rgba(33, 150, 243, 0.06) 0%, rgba(33, 150, 243, 0.03) 100%);
        border: 1px solid rgba(33, 150, 243, 0.18);
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: 8px;
    }
    #howto-body .gr-box,
    #howto-body .gr-panel,
    #howto-body .gr-group,
    #howto-body .gr-block,
    #howto-body .gr-markdown,
    #howto-body * {
        background: transparent !important;
        box-shadow: none !important;
    }

    /* ✅ 핵심: 코드/프리 배경, 테두리, 폰트 제거 */
    #howto-body pre,
    #howto-body code,
    #howto-body .gr-markdown pre,
    #howto-body .gr-markdown code,
    #howto-body .prose pre,
    #howto-body .prose code {
        background: transparent !important;
    }
    /* 풀스크린 모달 오버레이 */
    #video-modal {
        position: fixed; inset: 0; 
        background: rgba(0,0,0,0.65);
        align-items: center; justify-content: center;
        z-index: 9999; padding: 24px;
    }
    #video-modal .modal-card {
        position: relative; width: min(92vw, 1080px);
        background: rgba(20,20,20,0.85);
        border-radius: 14px; padding: 12px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    }
    #video-modal .modal-close {
        position: absolute;
        top: 8px;
        right: 10px;
        z-index: 3;                         /* 비디오/카드보다 위 */
        display: inline-block;
        padding: 8px 12px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.7);
        background: rgba(0,0,0,0.7);        /* 어두운 배경 → 대비 ↑ */
        color: #fff !important;             /* 테마 색상 무시하고 흰색 고정 */
        font-size: 16px;
        font-weight: 700;
        line-height: 1;
        text-shadow: 0 1px 2px rgba(0,0,0,0.6);
        cursor: pointer;
        pointer-events: auto;               /* 비디오가 포인터 잡아먹는 경우 방지 */
    }
    #video-modal .modal-close:hover {
        background: rgba(0,0,0,0.9);
        border-color: #fff;
    }
    </style>
    """)

    def open_modal(path):
        # 모달 비디오에 경로 세팅 + 표시
        return gr.update(visible=True), gr.update(value=path)

    def close_modal():
        # 모달 숨기기
        return gr.update(visible=False), gr.update(value=None)

    # 먼저 상태/채팅 컴포넌트를 만들어두고(이후 왼쪽 패널 핸들러에서 참조)
    # ===== 모달 구성 (맨 바닥에 하나만) =====
    with gr.Group(visible=False, elem_id="video-modal") as modal_group:
        with gr.Column(elem_classes=["modal-card"]):
            modal_close = gr.Button(value="✖ Close", variant="secondary", elem_classes=["modal-close"])
            modal_video = gr.Video(label=None, autoplay=True, height=520, interactive=False)

    # ===== 상단 2열 레이아웃 (같은 Row) =====
    with gr.Row():
        model_ready = gr.State(False) 
        # ---- 왼쪽: 채팅영역 ----
        with gr.Column(scale=5):
            chatbot = gr.Chatbot(height=560, show_copy_button=True, type="messages")
            state   = gr.State([])                 # 대화 히스토리
            session = gr.State(_default_session()) # 세션 캐시

            with gr.Row():
                btn_ratio = gr.Button("Give split-liability ratio", interactive=False)
                btn_who   = gr.Button("Who is at fault?",           interactive=False)
        # ---- 오른쪽: 동영상/모델/옵션 ----
        with gr.Column(scale=5):
            # 👉 먼저 선언 (아래 버튼 outputs에서 참조하므로)

            video_name = gr.Textbox(label="Video Name", placeholder="예) crash_000123.mp4 또는 2024-09-*.mp4", visible=False)

            with gr.Accordion("✨ 데모 실행방법"):
                gr.Markdown(
                            """
                    **1) 샘플 영상 선택**
                    - `🎥 Use`: 해당 영상을 분석 대상으로 등록합니다.
                    - `🔍 Enlarge`: 선택한 영상을 큰 화면으로 미리보기 합니다.
                    - 다른 영상으로 챗봇을 이용하려면 하단의 페이지 새로고침 버튼을 눌러주세요.

                    **2) 챗봇 실행**
                    - `EV`: Ego-Vehicle | `OV`: Other-Vehicle
                    - `Give split-liability ratio`: 차량의 과실비율을 산출하고, 사고 상황을 서술합니다.
                    - `Who is at fault?`: 피해자와 가해자를 판단합니다.
                            """
                        )

                    # 🔄 Reset: 현재 페이지 전체 새로고침 (세션/캐시 초기화 효과)
                reset_btn = gr.Button("🔄 Reset (페이지 새로고침)", variant="secondary")

                # 👉 Reset 버튼 동작: JS로 브라우저 새로고침
                reset_btn.click(
                    fn=lambda: None,
                    inputs=[],
                    outputs=[],
                    js="() => { window.location.reload(); }",  # ← 함수 리터럴 형태로!
                    queue=False,
                    show_progress="hidden",
                )

            with gr.Accordion("📦 샘플 영상 (preview ▶, then Use / Zoom)", open=True):
                samples = _build_sample_list()
                if not samples:
                    gr.Markdown("> No samples found. Put files under a search root (e.g., /mnt/data/videos).")
                else:
                    with gr.Column(elem_id="sample-list"):
                        for path, label in samples:
                            with gr.Column():
                                thumb = gr.Video(label=label, value=path, interactive=False, height=140)
                                with gr.Row(scale=1):
                                    gr.Button(f"🎥 Use").click(
                                        _use_sample,
                                        inputs=[session, chatbot, gr.State(_resolve_sample_path(path))],
                                        outputs=[session, chatbot, video_name],
                                        queue=False, show_progress="hidden"
                                    ).then(
                                        fn=gate_controls,
                                        inputs=[video_name, model_ready],
                                        outputs=[btn_ratio, btn_who],
                                        queue=False
                                    )

                                    gr.Button("🔍 Enlarge").click(
                                        lambda p=_resolve_sample_path(path): (gr.update(visible=True), gr.update(value=p)),
                                        inputs=None,
                                        outputs=[modal_group, modal_video]
                                    )

            model_id = gr.Textbox(label="HF Model ID / local path", value="LanguageBind/Video-LLaVA-7B-hf", visible=False)

            with gr.Column():
                # load_btn = gr.Button("Load / Reload Model")
                load_status = gr.Textbox(label="Load status", value="", interactive=False, visible=False)

            demo.load(
                fn=ui_load_model,
                inputs=[model_id],
                outputs=[load_status],
                queue=False
            ).then(
                fn=lambda s: _is_model_loaded_text(s),
                inputs=[load_status],
                outputs=[model_ready],
                queue=False
            ).then(
                fn=gate_controls,
                inputs=[video_name, model_ready],
                outputs=[btn_ratio, btn_who],
                queue=False
            )

            # 옵션들
            with gr.Accordion("⚙️ Options", open=False):
                with gr.Row():
                    use_soft    = gr.Checkbox(value=True,  label="Use Fault-BERT Soft Token")
                    interp_mode = gr.Checkbox(value=True,  label="Two-pass interpretability")
                    soft_gain   = gr.Slider(0.0, 1.5, value=0.5, step=0.05, label="Soft token gain")
                with gr.Row():
                    style      = gr.Dropdown(label="Output style", choices=["short", "brief", "detailed"], value="brief")
                    num_frames = gr.Slider(4, 16, value=8, step=1, label="Frames")
                with gr.Row():
                    frame_size   = gr.Slider(160, 336, value=224, step=16, label="Frame size")
                    temperature  = gr.Slider(0.0, 1.0, value=0.35, step=0.05, label="Temperature")
                    sampling     = gr.Checkbox(value=True, label="Enable sampling")
                with gr.Row():
                    clf_ckpt     = gr.Textbox(label="Classifier CKPT", value="/app/checkpoints/best_exact_ep13_r3d18.pth")
                    clf_backbone = gr.Dropdown(label="Backbone", choices=["r3d18","timesformer","videomae"], value="r3d18")
                with gr.Row():
                    clf_pretrained = gr.Checkbox(value=False, label="Use pretrained backbone")
                    clf_topk       = gr.Slider(1,5,value=3,step=1,label="Classifier Top-K")
                with gr.Row():
                    fault_ckpt  = gr.Textbox(label="Fault-BERT CKPT", value="/app/text-train/fault_ratio_bert_modify_softmax.pt")
                    fault_model = gr.Textbox(label="Fault-BERT model", value="bert-base-uncased")
                    fault_basis = gr.Slider(2, 20, value=10, step=1, label="Fault ratio basis")

            # video      = gr.Video(label="Preview / Upload", interactive=True, autoplay=True, height=220)

            # with gr.Column():
            #     scan_btn    = gr.Button("Scan Videos")
            #     # video_picker = gr.Dropdown(label="Pick found", choices=[], value=None)
            #     # 현재 프리뷰 확대 버튼 (업로드/드롭다운 선택 비디오 확대)
            #     enlarge_current = gr.Button("🔍 Enlarge preview")

    # ===== 버튼/이벤트 바인딩 (기존 로직 유지) =====
    # load_btn.click(
    #     fn=ui_load_model,
    #     inputs=[model_id],
    #     outputs=[load_status],
    #     queue=False,
    #     show_progress="minimal",
    # ).then(
    #     fn=lambda s: _is_model_loaded_text(s),
    #     inputs=[load_status],
    #     outputs=[model_ready],
    #     queue=False
    # ).then(
    #     fn=gate_controls,
    #     inputs=[video_name, model_ready],
    #     outputs=[btn_ratio, btn_who],
    #     queue=False
    # )
    # model_id.submit(
    #     ui_load_model, inputs=[model_id], outputs=[load_status],
    #     queue=False, show_progress="minimal",
    # )

    # 스캔 → 드롭다운 채우기
    # scan_btn.click(_fill_dropdown, None, [video_picker], queue=False, show_progress="minimal")
    # scan_btn.click(_fill_dropdown, None, queue=False, show_progress="minimal")

    # 드롭다운 선택 → 파일명/미리보기 갱신
    # video_picker.change(pick_video_and_preview, [video_picker], [video_name, video],
    #                     queue=False, show_progress="minimal")
    # 업로드/드래그 → 복사 + 파일명만 세팅
    # video.change(on_video_changed, [video], [video_name, load_status],
    #              queue=False, show_progress="hidden")

    # 현재 프리뷰 확대
    # enlarge_current.click(
    #     lambda v: (True, v if isinstance(v, str) else (v.get("name") or v.get("path") or v.get("video"))),
    #     inputs=[video],
    #     outputs=[modal_group, modal_video],
    # )

    # 모달 닫기
    modal_close.click(
        close_modal,
        inputs=None,
        outputs=[modal_group, modal_video],  # ← 모달 그룹 + 모달 비디오 둘 다 업데이트
    )

    # 채팅 동작(기존 체인 유지)
    INTENT_RATIO = "Watch the video and give a split-liability ratio."
    INTENT_WHO   = "Watch the video and give a who is at fault."

    # btn_ratio.click(
    #     ui_push_user, inputs=[state, session, gr.State(INTENT_RATIO)],
    #     outputs=[state, session, chatbot, user_msg],
    # ).then(
    #     ui_turn,
    #     inputs=[
    #         state, session, user_msg, video, video_name, model_id,
    #         clf_ckpt, clf_backbone, clf_pretrained, clf_topk,
    #         fault_ckpt, fault_model, fault_basis,
    #         style, num_frames, frame_size, temperature, sampling,
    #         use_soft, interp_mode, soft_gain,
    #         gr.State(True)
    #     ],
    #     outputs=[chatbot, session, state],
    # ).then(lambda: gr.update(value=""), None, [user_msg])
    btn_ratio.click(
    ui_push_user,
    inputs=[state, session, gr.State(INTENT_RATIO)],
    outputs=[state, session, chatbot, gr.State()]   # 마지막 값은 이어서 넘길 user_msg 자리 확보용 (dummy ok)
    ).then(
        ui_turn,
        inputs=[
            state, session,
            gr.State(INTENT_RATIO),     # user_msg
            gr.State(None),             # video (우리는 세션/파일명으로만 찾음)
            video_name, model_id,
            clf_ckpt, clf_backbone, clf_pretrained, clf_topk,
            fault_ckpt, fault_model, fault_basis,
            style, num_frames, frame_size, temperature, sampling,
            use_soft, interp_mode, soft_gain,
            gr.State(True)              # skip_user_echo
        ],
        outputs=[chatbot, session, state]
    )

    # Who is at fault
    btn_who.click(
        ui_push_user,
        inputs=[state, session, gr.State(INTENT_WHO)],
        outputs=[state, session, chatbot, gr.State()]
    ).then(
        ui_turn,
        inputs=[
            state, session,
            gr.State(INTENT_WHO),       # user_msg
            gr.State(None),             # video
            video_name, model_id,
            clf_ckpt, clf_backbone, clf_pretrained, clf_topk,
            fault_ckpt, fault_model, fault_basis,
            style, num_frames, frame_size, temperature, sampling,
            use_soft, interp_mode, soft_gain,
            gr.State(True)
        ],
        outputs=[chatbot, session, state]
    )

    # btn_who.click(
    #     ui_push_user, inputs=[state, session, gr.State(INTENT_WHO)],
    #     outputs=[state, session, chatbot, user_msg],
    # ).then(
    #     ui_turn,
    #     inputs=[
    #         state, session, user_msg, video, video_name, model_id,
    #         clf_ckpt, clf_backbone, clf_pretrained, clf_topk,
    #         fault_ckpt, fault_model, fault_basis,
    #         style, num_frames, frame_size, temperature, sampling,
    #         use_soft, interp_mode, soft_gain,
    #         gr.State(True)
    #     ],
    #     outputs=[chatbot, session, state],
    # ).then(lambda: gr.update(value=""), None, [user_msg])


    # send_btn.click(
    #     ui_turn,
    #     inputs=[
    #         state, session, user_msg, video, video_name, model_id,
    #         clf_ckpt, clf_backbone, clf_pretrained, clf_topk,
    #         fault_ckpt, fault_model, fault_basis,
    #         style, num_frames, frame_size, temperature, sampling,
    #         use_soft, interp_mode, soft_gain,
    #         gr.State(False)
    #     ],
    #     outputs=[chatbot, session, state],
    # ).then(lambda: gr.Textbox.update(value=""), None, [user_msg])
    # send_btn.click(
    #     ui_turn,
    #     inputs=[
    #         state, session, video, video_name, model_id,
    #         clf_ckpt, clf_backbone, clf_pretrained, clf_topk,
    #         fault_ckpt, fault_model, fault_basis,
    #         style, num_frames, frame_size, temperature, sampling,
    #         use_soft, interp_mode, soft_gain,
    #         gr.State(False)
    #     ],
    #     outputs=[chatbot, session, state],
    # ).then(lambda: gr.Textbox.update(value=""), None)

if __name__ == "__main__":
    try:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", 7861)))
    share = os.environ.get("GRADIO_SHARE", "1") == "1"

    demo.queue()
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_error=True,
        inbrowser=False,
        prevent_thread_lock=False,
        allowed_paths=ALLOWED_PATHS,
    )
    print(f"\n[Gradio] listening on http://{server_name}:{server_port}  (share={share})")
