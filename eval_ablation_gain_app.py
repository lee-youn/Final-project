# -*- coding: utf-8 -*-
"""
Ablation (use_soft off/on) + gain sensitivity
- Interpretability two-pass (DRAFT -> REVISED)
- 자동 스캔 평가 (최대 MAX_VIDEOS)
- 시드별 텍스트(.txt) 저장 제거
- 추가 메트릭: ILC / BD@N / ODTR
"""

import os, re, json, random, csv
from itertools import combinations
from statistics import mean, pstdev
from typing import Optional, List, Tuple, Dict, Set
import numpy as np
import torch
import torch.nn.functional as F

import prompt_video_llava_chat_with_rate_bert as app

# -----------------------
# 실험 설정
# -----------------------
SEEDS = [42, 1337, 2027]
GAINS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.2]
MAX_VIDEOS = 30

MODEL_ID   = "LanguageBind/Video-LLaVA-7B-hf"
CLF_CKPT   = "/app/checkpoints/best_exact_ep13_r3d18.pth"
FAULT_CKPT = "/app/text-train/fault_ratio_bert_modify_softmax.pt"
FAULT_MODEL= "bert-base-uncased"

NUM_FRAMES = 8
FRAME_SIZE = 224
TEMPERATURE = 0.6
SAMPLING = True
STYLE = "brief"
FAULT_BASIS = 10

VIDEO_EXTS = [".mp4", ".mkv", ".mov", ".avi", ".webm"]

FORBIDDEN = [
    r"\btraffic light(s)?\b", r"\bsignal(ized|s)?\b", r"\btimestamp(s)?\b",
    r"\bAM\b", r"\bPM\b", r"\bspeed(s)?\b", r"\blane(s)?\b", r"\b\d{1,3}\b"
]
HEDGING = [r"\blikely\b", r"\bprobably\b", r"\bappears?\s+to\b", r"\bmay\b", r"\bseems?\b"]

# ===== 행동 어휘 사전 =====
BEHAVIOR_LEXICON: List[str] = [
    # ===== Direction / Turning =====
    "left turn", "right turn", "going straight",
    "slight left", "slight right", "sharp left", "sharp right",
    "wide turn", "tight turn", "u-turn",
    "late left turn", "late right turn", "early apex turn",
    "swinging wide on turn", "cutting the corner", "overshooting the turn",
    "counter-steer correction", "understeer correction", "oversteer correction",
    "premature turn", "delayed turn", "commit then abandon turn",

    # ===== Lane / Merge / Positioning =====
    "lane change", "lane change left", "lane change right",
    "double lane change", "simultaneous lane change",
    "merge left", "merge right", "zipper merge", "merge into traffic",
    "hesitant merge", "assertive merge", "aborted merge",
    "drift left", "drift right", "weaving", "lane wandering",
    "straddling lanes", "partial lane encroachment", "lane departure",
    "cut-in", "hard cut-in", "soft cut-in", "late cut-in",
    "keep to right", "keep to left", "centered in lane",
    "overlap in adjacent lane", "squeeze into gap", "threading through gap",

    # ===== Entry / Priority / Right-of-way (no signals) =====
    "enter from side road", "enter from main road",
    "enter from left road", "enter from right road",
    "joined from curb", "pull out from driveway", "pull out from parking",
    "entered earlier", "entered later", "claiming entry",
    "yield to oncoming", "yield to crossing traffic", "give way",
    "fail to yield", "force entry", "encroach into path", "block the box",
    "creep into mainline", "probe then enter", "rolling creep then go",

    # ===== Interaction / Crossing / Simultaneity =====
    "crossing path", "head-on approach", "counterflow approach",
    "turning simultaneously", "two vehicles turning right simultaneously",
    "opposed turn conflict", "turn across path", "across-the-bow turn",
    "staggered approach", "side-by-side approach",
    "follow closely", "tailgating", "closing gap", "short gap acceptance",
    "reject gap then go", "misjudged gap", "gap forced open",

    # ===== Stopping / Accel–Decel (no speeds) =====
    "stop after", "rolling stop", "stop-and-go", "hesitant stop",
    "hard brake", "sudden stop", "progressive braking", "late braking",
    "brake then release", "pulse braking",
    "coast then stop", "coast then go", "hesitant start", "hesitant go",
    "accelerate out", "accelerate to clear", "surge then settle",

    # ===== Parking / Reversing / Pull-out =====
    "reverse", "backing up", "back into lane",
    "three-point maneuver", "multi-point maneuver",
    "parallel park exit", "angled park exit",
    "door opening conflict", "parked vehicle pull-out",
    "backing from driveway", "backing from shoulder",

    # ===== Intersection / Roundabout (no signals) =====
    "enter intersection", "commit to intersection", "clear intersection",
    "block intersection", "queue spillback into intersection",
    "change lanes inside intersection", "turn across path inside intersection",
    "stop line overrun", "creep past stop line",
    "enter roundabout", "circulate in roundabout", "exit roundabout",
    "missed exit then cut across", "late exit selection",

    # ===== Road Position / Edges / Medians =====
    "hug right edge", "hug left edge", "hug center line",
    "cross center line", "stray into opposing lane",
    "cross median", "median encroachment",
    "mount curb", "mount curb and return", "brush curb",
    "shoulder ride", "use shoulder to pass", "squeeze through narrow gap",

    # ===== Conflict / Contact Style =====
    "rear approach close", "rear closing rapidly",
    "rear approach with brake flash", "rear approach then lane change",
    "side-swipe tendency", "side brush trajectory",
    "t-bone approach", "oblique impact trajectory",
    "pinch against curb", "squeeze against barrier", "box-in maneuver",

    # ===== Maneuver Quality / Intent Cues =====
    "indecisive lane change", "decisive lane change",
    "feint then change", "abort then reattempt",
    "rolling creep", "staged creep then go", "nudge forward then yield",
    "over-correction", "under-correction", "oscillating corrections",
    "lane keeping degraded", "path commitment delayed", "late path choice",

    # ===== Obstruction / Visibility / Surface (neutral) =====
    "sightline blocked by large vehicle", "occluded approach from near side",
    "occluded approach from far side", "hidden entry from driveway",
    "surface edge drop-off encounter", "puddle avoidance swerve",
    "debris avoidance swerve", "work zone channelization follow",
    "temporary cone line follow", "misread temporary channelization",

    # ===== Non-motor Interaction (neutral) =====
    "yield to pedestrian crossing", "fail to yield to pedestrian",
    "yield to cyclist", "close pass to cyclist",
    "pedestrian encroaching into lane", "cyclist encroaching into lane",

    # ===== Label-mirroring neutral bigrams (no numbers) =====
    "changing lanes within intersection", "lanes wide enough for two side by side",
    "reduced turning angle maneuver", "main road and side road pattern",
    "other vehicle enters from the side", "simultaneous lane change",
    "turn across opposing path",

    # ===== Additional fine-grained cues (behavioral atoms) =====
    "late merge decision", "early merge decision",
    "aiming for gap ahead", "aiming for gap behind",
    "shadowing adjacent vehicle", "door zone pass",
    "nose-out probe", "bumper-to-bumper follow",
    "slotting into platoon", "overtake on the left", "overtake on the right",
    "abort overtake", "return to lane after overtake",
    "prepare to turn then continue straight", "fake turn then straighten",
    "lane change under load", "lane change while braking",
    "lane change while accelerating", "lane change while coasting",
    "merge while braking", "merge while accelerating",
    "yield wave ignored", "hand signal yield accepted",
    "rolling hand-off between lanes", "drift toward exit", "late exit drift",
    "hover at conflict point", "commit through conflict point",
    "linger in conflict zone", "clear conflict zone decisively",
    "filter into mainline", "filter out of mainline",
    "tuck in behind lead", "slot ahead of lead",
    "squeeze between turning streams", "thread between opposing flows",
    "angled approach to lane", "shallow angle approach to lane",
    "stall then proceed", "hesitate then retreat",
    "short advance then stop", "micro-movements at line",
    "nose blocks crossing path", "tail blocks crossing path",
    "edge into cross traffic", "edge across center line",
    "late gap selection", "misread opposing intent",
    "assertive claim of right-of-way", "passive yield of right-of-way",
    "path deviation to avoid obstacle", "path deviation to claim gap",
    "overtake set-up", "overtake commit", "overtake abort",
    "shadow pass attempt", "side-by-side dwell", "side-by-side clearance seek",
    "lane position adjustment near curb", "lane position adjustment near center",
    "micro-corrections around parked vehicles", "micro-corrections around cones",
    "entry angle mismatch", "exit angle mismatch",
    "approach overlap", "turn-in overlap", "exit overlap",
    "staggered merge entry", "zipper merge coordination",
    "incomplete lane change", "late lane centering after change",
    "hover next to blind spot", "escape from blind spot",
    "probe for courtesy gap", "accept courtesy gap", "decline courtesy gap",
    "queue jump via shoulder", "queue reentry from shoulder",
    "nose-to-tail accordion", "rolling platoon join", "rolling platoon leave",
    "path cut across by other", "path cut across by ego",
    "misaligned stop position", "encroachment on stop line",

    # ===== Canonicalized phrases (signal-free) =====
    "opposed straight approach",
    "going straight from main road", "going straight from side road",
    "going straight from left road", "going straight from right road",
    "right turn from main road", "right turn from side road",
    "right turn from right lane", "left turn from side road", "left turn from right road",
    "entered earlier into mainline", "entered later into mainline",
    "following straight in lane", "following straight on right side of lane",
    "lane change inside intersection",
    "other vehicle enters from the side",
    "two vehicles turning right simultaneously",
    "enter from right then left turn",
    "side road entry conflict",
    "main road vs side road merge pattern",
]

# ===== ODTR 계산을 위한 보조 사전 =====
STOPWORDS: Set[str] = {
    # 간단 영어 불용어 (필요시 확장)
    "a","an","the","and","or","but","if","then","so","because","as","of","in","on","at","to","from",
    "for","with","without","by","is","are","was","were","be","been","being","it","this","that","these","those",
    "there","here","over","under","into","out","up","down","while","when","where","who","whom","which","what",
    "do","does","did","doing","done","can","could","may","might","should","would","will","shall","than","too",
    "very","more","most","such","also","just","not","no","nor","own","same","both","all","any","each","few","many",
    "some","much","other","another","between","within","across","about","around","per"
}

# 도메인 중립/핵심 명사(ODTR에서 제외): off-domain이 아님
DOMAIN_NEUTRAL: Set[str] = {
    "vehicle","car","van","truck","bus","motorcycle","bike","cyclist","pedestrian","driver","ego","other",
    "road","street","lane","shoulder","median","curb","intersection","roundabout","traffic","flow","path",
    "gap","merge","turn","left","right","straight","entry","exit","approach","conflict","side","main","signal",
    "cross","overtake","stop","brake","accelerate","yield","drift","change","line","center","edge","queue"
}

# ===== 임베더 (behavior alignment) =====
_EMBEDDER = None
_LEX_EMB = None
_EMB_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(model_name, device=_EMB_DEVICE)
    except Exception:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        mdl = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to(_EMB_DEVICE).eval()
        class HFEmbedder:
            def __init__(self, tok, mdl): self.tok, self.mdl = tok, mdl
            @torch.no_grad()
            def encode(self, texts: List[str], convert_to_tensor=True, normalize_embeddings=True):
                if isinstance(texts, str): texts = [texts]
                outs = []
                for i in range(0, len(texts), 16):
                    batch = texts[i:i+16]
                    enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(_EMB_DEVICE)
                    hs = mdl(**enc).last_hidden_state
                    emb = hs.mean(1)
                    outs.append(emb)
                Z = torch.cat(outs, 0)
                if normalize_embeddings: Z = torch.nn.functional.normalize(Z, dim=-1)
                return Z if convert_to_tensor else Z.cpu().numpy()
        _EMBEDDER = HFEmbedder(tok, mdl)
    return _EMBEDDER

def _ensure_lexicon_embeddings():
    global _LEX_EMB
    if _LEX_EMB is not None:
        return _LEX_EMB
    emb = get_embedder()
    with torch.no_grad():
        _LEX_EMB = emb.encode(BEHAVIOR_LEXICON, convert_to_tensor=True, normalize_embeddings=True)
    return _LEX_EMB

def behavior_aligned_score(text: str) -> float:
    t = (text or "").strip()
    if not t: return 0.0
    emb = get_embedder()
    lex = _ensure_lexicon_embeddings()
    with torch.no_grad():
        z = emb.encode([t], convert_to_tensor=True, normalize_embeddings=True)
        sims = torch.matmul(z, lex.T).squeeze(0)
        return float(sims.max().item())

# -----------------------
# 토크나이즈 & 행동 어휘 매칭 (ILC/BD/ODTR)
# -----------------------
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

def tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text or "")]

def compile_phrase_regex(phrase: str) -> re.Pattern:
    # 공백은 \s+로, 단어경계 보장
    p = r"\b" + re.sub(r"\s+", r"\\s+", re.escape(phrase.strip())) + r"\b"
    return re.compile(p, flags=re.IGNORECASE)

# 사전 정렬: 긴 구문 우선(겹침 최소화)
SORTED_PHRASES = sorted(BEHAVIOR_LEXICON, key=lambda s: len(s.split()), reverse=True)
PHRASE_REGEX = [compile_phrase_regex(p) for p in SORTED_PHRASES]

def compute_ilc_bd_odtr(text: str) -> Tuple[float, float, float]:
    """
    ILC: 행동 어휘로 커버된 토큰 / 총 토큰
    BD@N: 고유 행동 구문 매칭 수
    ODTR: (행동 어휘로 커버되지 않고, stopword 아니고, DOMAIN_NEUTRAL에도 없는 토큰) / 총 토큰
    """
    text = text or ""
    tokens = tokenize_with_spans(text)
    n_tokens = max(1, len(tokens))

    covered_token_idx: Set[int] = set()
    matched_phrases: Set[str] = set()

    # 문서 내 행동 어휘 매칭 → 토큰 커버 집합 생성
    for phrase, rgx in zip(SORTED_PHRASES, PHRASE_REGEX):
        found = False
        for m in rgx.finditer(text):
            span_start, span_end = m.start(), m.end()
            # 매칭된 char-span에 걸치는 토큰 index 표시
            for i, (_, s, e) in enumerate(tokens):
                if s >= span_start and e <= span_end:
                    covered_token_idx.add(i)
            found = True
        if found:
            matched_phrases.add(phrase)

    # ILC
    ilc = len(covered_token_idx) / n_tokens

    # BD@N (고유 행동 프레이즈 수)
    bd = float(len(matched_phrases))

    # ODTR
    off_domain_cnt = 0
    for i, (w, _, _) in enumerate(tokens):
        wl = w.lower()
        if i in covered_token_idx:        # 행동 어휘에 포함 -> off-domain 아님
            continue
        if wl in STOPWORDS:               # 불용어 제외
            continue
        if wl in DOMAIN_NEUTRAL:          # 도메인 중립/핵심 단어는 off-domain으로 보지 않음
            continue
        off_domain_cnt += 1
    odtr = off_domain_cnt / n_tokens

    return ilc, bd, odtr

# -----------------------
# 유틸
# -----------------------
def _is_video_file(path: str) -> bool:
    return os.path.isfile(path) and os.path.splitext(path.lower())[1] in VIDEO_EXTS

def _first_video_under(dir_path: str) -> Optional[str]:
    for root, _, files in os.walk(dir_path):
        for f in sorted(files):
            if os.path.splitext(f.lower())[1] in VIDEO_EXTS:
                p = os.path.join(root, f)
                if os.path.getsize(p) > 0:
                    return p
    return None

def find_videos(max_videos=20) -> List[str]:
    try:
        cands = app.list_candidates(limit=max_videos*3)
    except Exception:
        cands = []
    vids = []
    for p in cands:
        if os.path.isdir(p):
            first = _first_video_under(p)
            if first and _is_video_file(first): vids.append(first)
        elif _is_video_file(p):
            vids.append(p)
        if len(vids) >= max_videos:
            break
    uniq, seen = [], set()
    for v in vids:
        if v not in seen and _is_video_file(v):
            uniq.append(v); seen.add(v)
    return uniq[:max_videos]

def paired_bootstrap(a_list, b_list, iters=10000, seed=123):
    if len(a_list) < 2:
        return 0.0, 0.0, 0.0, None
    if np.std(np.array(a_list) - np.array(b_list)) == 0:
        return float(np.mean(np.array(b_list)-np.array(a_list))), 0.0, 0.0, None
    rng = np.random.default_rng(seed)
    diffs = []
    a_arr, b_arr = np.array(a_list), np.array(b_list)
    n = len(a_arr)
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        diffs.append(b_arr[idx].mean() - a_arr[idx].mean())
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs >= 0).mean(), (diffs <= 0).mean())
    return float(diffs.mean()), float(lo), float(hi), float(p)

def jaccard(a: str, b: str):
    A = set(re.findall(r"[a-zA-Z]+", a.lower()))
    B = set(re.findall(r"[a-zA-Z]+", b.lower()))
    if not A and not B: return 1.0
    return len(A & B) / max(1, len(A | B))

def extract_meta(full_text: str):
    m_desc = re.search(r"(?im)^\s*Description:\s*(.+)$", full_text)
    m_evid = re.search(r"(?im)^\s*Evidence:\s*(.+)$", full_text)
    desc = (m_desc.group(1).strip() if m_desc else "")
    evid = (m_evid.group(1).strip() if m_evid else "")
    m_fault = re.search(r"Fault\(basis\s+(\d+)\):\s*DC\s*([0-9.]+)\s*/\s*OV\s*([0-9.]+)\s*\|\s*Victim:\s*([A-Za-z ]+)", full_text)
    basis = dc = ov = victim = None
    if m_fault:
        basis = int(m_fault.group(1)); dc = float(m_fault.group(2)); ov = float(m_fault.group(3))
        victim = m_fault.group(4).strip()
    m_dv = re.search(r"-\s*Dashcam Vehicle:\s*([^\n]+)", full_text)
    m_ov = re.search(r"-\s*Other Vehicle:\s*([^\n]+)", full_text)
    dv_hint = m_dv.group(1).strip() if m_dv else ""
    ov_hint = m_ov.group(1).strip() if m_ov else ""
    return desc, evid, basis, dc, ov, victim, dv_hint, ov_hint

def metric_forbidden_rate(s: str):
    return 1.0 if any(re.search(pat, s, flags=re.IGNORECASE) for pat in FORBIDDEN) else 0.0

def metric_two_line_format(desc: str, evid: str):
    return 1.0 if (desc and evid and desc[-1:] in ".!?" and evid[-1:] in ".!?") else 0.0

def metric_hedging_rate(s: str):
    words = re.findall(r"\w+", s); n = max(1, len(words))
    cnt = sum(len(re.findall(pat, s, flags=re.IGNORECASE)) for pat in HEDGING)
    return 100.0 * cnt / n

def metric_length(s: str):
    return len(re.findall(r"\w+", s))

def metric_reproducibility(texts):
    if len(texts) < 2: return 1.0
    sims = [jaccard(a,b) for a,b in combinations(texts, 2)]
    return float(mean(sims))

def infer_motion_tokens(lbl: str):
    s = (lbl or "").lower()
    if "unknown" in s or not s.strip():
        return {"left","right","straight"}
    tokens = set()
    if "left" in s: tokens.add("left")
    if "right" in s: tokens.add("right")
    if "straight" in s or "going straight" in s: tokens.add("straight")
    if "lane change" in s: tokens.update({"left","right"})
    return tokens or {"straight"}

def metric_fault_alignment(dc, ov, victim):
    if dc is None or ov is None or not victim: return 0.0
    implied = "Ego Vehicle" if dc < ov else ("Other Vehicle" if ov < dc else "Undetermined")
    return 1.0 if victim.strip().lower() == implied.strip().lower() else 0.0

def metric_frame_text_conflict(desc: str, dv_hint: str, ov_hint: str):
    s = (desc or "").lower()
    toks = set()
    if "left" in s: toks.add("left")
    if "right" in s: toks.add("right")
    if "straight" in s: toks.add("straight")
    allow = infer_motion_tokens(dv_hint) | infer_motion_tokens(ov_hint)
    if not toks: return 0.0
    return 1.0 if toks.isdisjoint(allow) else 0.0

# -----------------------
# 실행기
# -----------------------
def run_once(video_path, seed, use_soft, gain):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    app.ENGINE.load_model(MODEL_ID)

    history = []
    user_msg = ""
    video = {"name": video_path}
    video_name = os.path.basename(video_path)

    out_hist = app.ui_generate(
        history, user_msg, video, video_name, MODEL_ID,
        CLF_CKPT, "r3d18", False, 3,
        FAULT_CKPT, FAULT_MODEL, FAULT_BASIS,
        STYLE, NUM_FRAMES, FRAME_SIZE, TEMPERATURE, SAMPLING,
        use_soft, True, gain
    )
    return out_hist[-1]["content"]

def evaluate_condition(video_path, use_soft, gain):
    texts = [run_once(video_path, s, use_soft, gain) for s in SEEDS]

    descs, evids = [], []
    bases, dcs, ovs, victims, dvhints, ovhints = [], [], [], [], [], []
    m_forb, m_format, m_hedge, m_len, m_conf, m_align = [], [], [], [], [], []
    m_beh_desc = []
    m_ilc, m_bd, m_odtr = [], [], []

    for t in texts:
        desc, evid, basis, dc, ov, victim, dvh, ovh = extract_meta(t)
        descs.append(desc); evids.append(evid)
        bases.append(basis); dcs.append(dc); ovs.append(ov); victims.append(victim)
        dvhints.append(dvh); ovhints.append(ovh)

        joined = ((desc or "") + " " + (evid or "")).strip()

        m_forb.append(metric_forbidden_rate(joined))
        m_format.append(metric_two_line_format(desc, evid))
        m_hedge.append(metric_hedging_rate(joined))
        m_len.append(metric_length(desc))
        m_conf.append(metric_frame_text_conflict(desc, dvh, ovh))
        m_align.append(metric_fault_alignment(dc, ov, victim))
        m_beh_desc.append(behavior_aligned_score(joined))

        ilc, bd, odtr = compute_ilc_bd_odtr(joined)
        m_ilc.append(ilc); m_bd.append(bd); m_odtr.append(odtr)

    results = {
        "texts_n": len(texts),
        "desc_mean_len": float(mean(m_len)),
        "forbidden_rate": float(mean(m_forb)),
        "two_line_format_rate": float(mean(m_format)),
        "hedging_per100w": float(mean(m_hedge)),
        "frame_text_conflict_rate": float(mean(m_conf)),
        "fault_alignment_rate": float(mean(m_align)),
        "reproducibility": float(metric_reproducibility(descs)),
        "length_variance": float(pstdev(m_len) ** 2),
        "behavior_align_score_desc": float(mean(m_beh_desc)),
        # 신규 지표
        "ilc_coverage": float(mean(m_ilc)),
        "bd_unique": float(mean(m_bd)),
        "odtr_offdomain": float(mean(m_odtr)),
    }
    return results

def main():
    os.makedirs("results", exist_ok=True)

    VIDEO_LIST = find_videos(MAX_VIDEOS)
    if not VIDEO_LIST:
        print("[WARN] No videos found."); return
    print(f"[INFO] Evaluating {len(VIDEO_LIST)} videos")

    all_rows = []
    summary = {"A0_vs_A1_g=0.5": {}}

    # --- A0 ---
    per_video_A0 = {}
    for v in VIDEO_LIST:
        try:
            r = evaluate_condition(v, use_soft=False, gain=0.0)
            per_video_A0[v] = r
            all_rows.append({"video": v, "cond": "A0", **r})
        except Exception as e:
            print(f"[SKIP A0] {v}: {e}")

    # --- A1 (gains) ---
    per_video_A1 = {g: {} for g in GAINS}
    for g in GAINS:
        for v in VIDEO_LIST:
            try:
                r = evaluate_condition(v, use_soft=True, gain=g)
                per_video_A1[g][v] = r
                all_rows.append({"video": v, "cond": f"A1_g{g}", **r})
            except Exception as e:
                print(f"[SKIP A1 g={g}] {v}: {e}")

    # CSV 저장
    csv_path = "results/ablation_gain_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = None
        for row in all_rows:
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)
    print(f"[OK] saved: {csv_path}")

    # 통계 비교(A0 vs A1 g=0.5)
    def collect(metric):
        vids = [v for v in VIDEO_LIST if v in per_video_A0 and v in per_video_A1.get(0.5, {})]
        a0 = [per_video_A0[v][metric] for v in vids]
        a1 = [per_video_A1[0.5][v][metric] for v in vids]
        return vids, a0, a1

    key_metrics = [
        "forbidden_rate",
        "two_line_format_rate",
        "frame_text_conflict_rate",
        "hedging_per100w",
        "reproducibility",
        "length_variance",
        "fault_alignment_rate",
        "behavior_align_score_desc",
        # 신규 지표 요약 포함
        "ilc_coverage",
        "bd_unique",
        "odtr_offdomain",
    ]
    for m in key_metrics:
        vids, a0, a1 = collect(m)
        diff, lo, hi, p = paired_bootstrap(a0, a1)
        summary["A0_vs_A1_g=0.5"][m] = {
            "N_videos": len(vids),
            "A0_mean": float(np.mean(a0) if a0 else float("nan")),
            "A1_mean": float(np.mean(a1) if a1 else float("nan")),
            "diff_mean": diff, "95CI": [lo, hi],
            "p_value": (None if p is None else float(p))
        }

    # Gain sensitivity
    sens = {}
    for m in key_metrics:
        sens[m] = []
        vidsA0 = [v for v in VIDEO_LIST if v in per_video_A0]
        for g in GAINS:
            vidsBoth = [v for v in vidsA0 if v in per_video_A1.get(g, {})]
            a0v = [per_video_A0[v][m] for v in vidsBoth]
            a1v = [per_video_A1[g][v][m] for v in vidsBoth]
            diff, lo, hi, p = paired_bootstrap(a0v, a1v)
            sens[m].append({
                "gain": g,
                "N_videos": len(vidsBoth),
                "A0_mean": float(np.mean(a0v) if a0v else float("nan")),
                "A1_mean": float(np.mean(a1v) if a1v else float("nan")),
                "diff_mean": diff, "95CI": [lo, hi],
                "p_value": (None if p is None else float(p))
            })
    summary["gain_sensitivity"] = sens

    jpath = "results/summary.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved: {jpath}")

    # 콘솔 요약
    print("\n=== A0 vs A1(g=0.5) summary ===")
    for m in key_metrics:
        s = summary["A0_vs_A1_g=0.5"][m]
        ptxt = "N/A" if s["p_value"] is None else f"{s['p_value']:.3f}"
        print(f"{m:>24}: N={s['N_videos']} | A0={s['A0_mean']:.3f} | A1={s['A1_mean']:.3f} | "
              f"Δ={s['diff_mean']:.3f} [{s['95CI'][0]:.3f},{s['95CI'][1]:.3f}] p={ptxt}")

    print("\n=== Gain sensitivity (Δ vs A0) ===")
    for m in key_metrics:
        print(f"\n[{m}]")
        for item in summary["gain_sensitivity"][m]:
            ptxt = "N/A" if item["p_value"] is None else f"{item['p_value']:.3f}"
            print(f"  g={item['gain']}: N={item['N_videos']} | Δ={item['diff_mean']:.3f} "
                  f"[{item['95CI'][0]:.3f},{item['95CI'][1]:.3f}] p={ptxt}")

if __name__ == "__main__":
    main()
