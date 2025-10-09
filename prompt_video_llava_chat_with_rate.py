# -*- coding: utf-8 -*-
"""
Video-LLaVA Chatbot (multi-turn)
- Load Video-LLaVA model (HF Hub or local)
- Upload a video + hints
- Chat window: user sends prompts repeatedly, model replies with explanations
- Hints must be reflected once in every answer
"""

import os, glob
from typing import List, Optional
import numpy as np
from PIL import Image
import torch
import gradio as gr
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, TimesformerModel, VideoMAEModel
from torchvision.models.video import r3d_18, R3D_18_Weights

from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
import cv2

SEARCH_ROOTS = [
    "/mnt/data/videos",
    "/app/data/raw/videos/training_reencoded",
    "/app/data/raw/videos/validation_reencoded",
    "/data"
]
# 허용할 확장자
VIDEO_EXTS = [".mp4", ".mkv", ".mov", ".avi", ".webm"]

# ===== Fault ratio (BERT) =====
from transformers import AutoModel, AutoTokenizer

class TextToFaultRatio(nn.Module):
    def __init__(self, model_name="bert-base-uncased", hidden_dim=768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:,0]
        return self.regressor(cls)

@torch.no_grad()
def load_fault_model(path: str, model_name="bert-base-uncased", device="cuda"):
    obj = torch.load(path, map_location="cpu")
    model = TextToFaultRatio(model_name=model_name)
    if isinstance(obj, dict) and "state_dict" in obj:
        model.load_state_dict(obj["state_dict"], strict=True)
    elif isinstance(obj, dict):
        model.load_state_dict(obj, strict=False)
    elif hasattr(obj, "state_dict"):
        model = obj
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

@torch.no_grad()
def predict_fault_ratio(model, tokenizer, text: str, device="cuda", max_length=256):
    inputs = tokenizer(text, return_tensors="pt",
                       padding="max_length", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pred = model(input_ids=input_ids, attention_mask=attention_mask).squeeze(0).float().cpu().numpy()  # [2]
    return pred  # (dashcam, other) 임의 스케일

def project_pair_to_basis(v, total=10.0):
    v = np.maximum(np.asarray(v, dtype=float), 0.0)
    s = float(v.sum())
    if s <= 0:
        return np.array([total/2.0, total/2.0], dtype=float)
    return v * (total / s)

def render_template_from_labels(dv_text: str, ov_text: str) -> str:
    return (f"At an unsignalized intersection, the Dashcam Vehicle was {dv_text}, "
            f"while the Other Vehicle was {ov_text}.")


# =========================
# (0) 라벨 정의 (네가 쓰던 dict)
# =========================
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

# =========================
# (3) 분류 모델 (3-헤드)
# =========================
class TripleHeadVideoClassifier(nn.Module):
    def __init__(self, n_dv, n_ov, backbone="r3d18", pretrained=False):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "r3d18":
            if pretrained:
                self.backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            else:
                self.backbone = r3d_18(weights=None)
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
    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    vid = (vid - mean) / std
    vid = vid.permute(1,0,2,3).unsqueeze(0).to(device)  # (1,C,T,H,W)
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
def _load_video_tensor_for_classifier(path, num_frames=16, size=224, device="cuda"):
    import cv2
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total-1, num_frames).astype(int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok: continue
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        fr = cv2.resize(fr, (size, size))
        ten = torch.from_numpy(fr).permute(2,0,1).float()/255.0
        frames.append(ten)
    cap.release()
    vid = torch.stack(frames, 0)                     # (T,C,H,W)
    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    vid = (vid-mean)/std
    vid = vid.permute(1,0,2,3).unsqueeze(0).to(device)  # (1,C,T,H,W)
    return vid
@torch.no_grad()
def predict_classifier(video_path, clf_model, device="cuda", topk=3):
    x = load_video_tensor_for_clf(video_path, num_frames=16, size=224, device=device)
    ld, lv = clf_model(x)   # dv, ov

    def _topk_text(logits, label_list, k):
        probs = logits.softmax(dim=-1)[0]
        val, idx = probs.topk(k=k, dim=-1)
        pairs = [f"{label_list[i]}:{val[j].item():.2f}" for j,i in enumerate(idx.tolist())]
        return label_list[idx[0].item()], ", ".join(pairs)

    dv1, dv_topk = _topk_text(ld, LABELS["dv"], topk)
    ov1, ov_topk = _topk_text(lv, LABELS["ov"], topk)

    dashcam_info   = dv1
    other_info     = ov1
    classifier_top = f"DV[{dv_topk}] | OV[{ov_topk}]"
    return dashcam_info, other_info, classifier_top

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

def find_video_path_by_name(name: str) -> str | None:
    """
    name: 'abcd1234' 또는 'abcd1234.mp4' 처럼 입력.
    - 확장자가 없으면 VIDEO_EXTS를 돌면서 후보를 찾습니다.
    - 와일드카드(*, ?)도 허용합니다. (예: 2024-09-*.mp4)
    """
    name = name.strip().strip('"').strip("'")
    if not name:
        return None

    has_ext = os.path.splitext(name)[1].lower() in VIDEO_EXTS

    # 1) 루트별 탐색
    for root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue

        # 와일드카드가 포함되면 glob로 그대로 탐색
        if any(ch in name for ch in "*?[]"):
            patterns = [os.path.join(root, name)]
        else:
            # 확장자 없으면 모든 확장자 후보를 만든다
            if has_ext:
                patterns = [os.path.join(root, name)]
            else:
                patterns = [os.path.join(root, name + ext) for ext in VIDEO_EXTS]

        for pat in patterns:
            hits = sorted(glob.glob(pat))
            if hits:
                # 가장 최근 수정된 파일을 선택 (동일 이름 다수일 때 안정적)
                hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return hits[0]

        # 하위 디렉토리까지 재귀적으로 찾고 싶으면 아래 블록 주석 해제
        # for dirpath, _, files in os.walk(root):
        #     if has_ext:
        #         cands = [os.path.join(dirpath, name)]
        #     else:
        #         cands = [os.path.join(dirpath, name + ext) for ext in VIDEO_EXTS]
        #     for p in cands:
        #         if os.path.exists(p):
        #             return p

    return None

def _resolve_video_path(video, video_name: str | None) -> str | None:
    # gr.Video 값은 str 또는 dict일 수 있음
    path = None
    if isinstance(video, str) and os.path.exists(video):
        path = video
    elif isinstance(video, dict) and "name" in video and os.path.exists(video["name"]):
        path = video["name"]

    # 업로드 파일이 없으면 video_name으로 탐색
    if not path and video_name:
        cand = find_video_path_by_name(video_name)
        if cand and os.path.exists(cand):
            path = cand

    # 절대경로/상대경로를 직접 입력한 경우도 허용
    if not path and video_name and os.path.exists(video_name):
        path = video_name

    return path
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
            arr = vr[i].asnumpy()[:, :, ::-1]  # BGR->RGB
            frames.append(Image.fromarray(arr).resize((size, size)))
    except Exception:
        import cv2
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

# ---------------- Prompt Builder ----------------
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
        "You are an expert at analyzing dashcam accident videos. "
        "Base your description strictly on what is visible. "
        "Do not invent traffic lights, lane markings, numbers, timestamps, or unseen objects.\n"
    )
    if style == "short":
        length_rule = "Write exactly one concise sentence (under ~30 words)."
    elif style == "detailed":
        length_rule = "Write 4–6 factual sentences (~80–120 words)."
    else:
        style = "brief"
        length_rule = "Write 2–3 factual sentences (under ~90 words)."

    # Hints
    hint_lines = []
    if dashcam_info:    
        hint_lines.append(f"- Dashcam Vehicle: {dashcam_info}")
        print(f"[Hint] Dashcam Vehicle: {dashcam_info}")
    if other_info:      
        hint_lines.append(f"- Other Vehicle: {other_info}")
        print(f"[Hint] Other Vehicle: {other_info}")
    if classifier_topk: 
        hint_lines.append(f"- Classifier outputs (top-k): {classifier_topk}")
        print(f"[Hint] Classifier top-k: {classifier_topk}")
    if fault_hint:      
        hint_lines.append(f"- Fault analysis[Dashcam Vehicle:Other Vehicle]: {fault_hint}")
        print(f"[Hint] Fault analysis: {fault_hint}")
    hint_block = ("Always include the following hints exactly once in natural English:\n" + "\n".join(hint_lines) + "\n") if hint_lines else ""

    # History
    hist_txt = ""
    if history:
        turns = history[-history_max_turns:]
        for role, msg in turns:
            if not msg: continue
            hist_txt += f"{role.upper()}:\n{msg}\n\n"

    user_header = (
        f"{hist_txt}"
        "USER:\n"
        "Describe the accident scene focusing on visible motion, entry order, and relative positions.\n"
        f"{length_rule}\n"
        f"{hint_block}"
        "You MUST apply the provided hints in your answer (reflect them at least once).\n"
        "Then add: the predicted fault ratio (basis 10), identify the likely victim (lower fault share), "
        "and one-sentence cause reasoning grounded in visible evidence and the classification hints.\n"
        "Avoid: the words 'traffic light', numbers unrelated to fault ratio, dates, or invented objects.\n"
        "Use 'Dashcam Vehicle' and 'Other Vehicle' exactly once each.\n"
    )
    if user_message and user_message.strip():
        user_header += f"\nAdditional instruction from user: {user_message.strip()}\n"

    return system_header + user_header + "ASSISTANT:"

# ---------------- Engine ----------------
class VideoLLaVAChatEngine:
    def __init__(self):
        self.model_id = None
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_id: str):
        if self.model_id == model_id and self.model is not None:
            return f"Model already loaded: {model_id}"
        self.processor = VideoLlavaProcessor.from_pretrained(model_id)
        self.model = VideoLlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            low_cpu_mem_usage=True,
            device_map=None,
        ).to(self.device)
        self.model.config.use_cache = False
        self.model_id = model_id
        return f"Loaded model: {model_id}"

    @torch.no_grad()
    def generate(
        self,
        video_path: str,
        user_message: str,
        dashcam_info: str,
        other_info: str,
        classifier_topk: str,
        style: str = "brief",
        num_frames: int = 8,
        frame_size: int = 224,
        temperature: float = 0.2,
        do_sample: bool = False,
        max_new_tokens: int = 180,
        history: Optional[list] = None,
        fault_hint: Optional[str] = None
    ) -> str:
        if not self.model or not self.processor:
            raise RuntimeError("Model is not loaded yet.")

        frames = sample_frames(video_path, num_frames=num_frames, size=frame_size)
        tok = self.processor.tokenizer
        placeholder = "<video>"
        if tok.convert_tokens_to_ids("<video>") in (None, tok.unk_token_id):
            placeholder = "<image>"

        prompt_text = build_prompt(
            user_message, dashcam_info, other_info, classifier_topk,
            style=style, history=history, fault_hint=fault_hint   # ★
        )
        chat_prompt = f"{placeholder}\n{prompt_text}"

        proc = self.processor(
            videos=[frames],
            text=[chat_prompt],
            padding="longest",
            truncation=False,
            return_tensors="pt",
        )
        if "pixel_values_videos" in proc:
            vision = {"pixel_values_videos": proc["pixel_values_videos"].to(self.device, dtype=self.model.dtype)}
        elif "pixel_values" in proc:
            vision = {"pixel_values_videos": proc["pixel_values"].to(self.device, dtype=self.model.dtype)}
        else:
            raise KeyError(f"Processor outputs missing vision tensors. Keys: {list(proc.keys())}")

        input_ids = proc["input_ids"].to(self.device)
        attention_mask = proc["attention_mask"].to(self.device)

        bad_words = ["traffic light","lights","signal","signals","signalized","timestamp","AM","PM"] + [str(d) for d in range(10)]
        bad_ids = []
        for w in bad_words:
            ids = tok(w, add_special_tokens=False).input_ids
            if ids: bad_ids.append(ids)

        gen_ids = self.model.generate(
            **vision,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=1,
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
            bad_words_ids=bad_ids,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        gen_only = gen_ids[:, input_ids.size(1):]
        text = tok.decode(gen_only[0], skip_special_tokens=True).strip()
        if style == "short":
            text = text.split("\n")[0].strip()
            if not text.endswith("."): text += "."
        return text

ENGINE = VideoLLaVAChatEngine()

# ---------------- Gradio UI ----------------
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

def ui_generate(
        history, user_msg, video, video_name, model_id,
        clf_ckpt, clf_backbone, clf_pretrained, clf_topk,
        fault_ckpt, fault_model, fault_basis,
        style, num_frames, frame_size, temperature, sampling
    ):
    history = history or []

    # 0) 비디오 경로 해석
    video_path = _resolve_video_path(video, video_name)
    print(f"[Chat] video_path={video_path}, video_name={video_name}")
    if not video_path:
        history = history + [{"role":"assistant","content": f"❌ Video not found. name='{video_name}'\n검색 루트: {SEARCH_ROOTS}"}]
        return history
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        history = history + [{"role":"assistant","content": f"❌ Video not readable: {video_path}"}]
        return history

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) DV/OV 분류 → top-k 텍스트
    try:
        clf = _ensure_classifier(clf_ckpt, clf_backbone, clf_pretrained, device)
        dv_hint, ov_hint, topk_text = predict_classifier(
            video_path, clf, device=device, topk=int(clf_topk)
        )
    except Exception as e:
        dv_hint, ov_hint = "UNKNOWN_DV", "UNKNOWN_OV"
        topk_text = f"[classifier error: {e}]"
    print(f"[Hint] DV={dv_hint} | OV={ov_hint} | TOPK={topk_text}")

    # 2) 템플릿 문장 생성
    sentence = render_template_from_labels(dv_hint, ov_hint)

    # 3) BERT 과실비율 추론
    try:
        fr_m, fr_tok = _ensure_fault(fault_ckpt, fault_model, device)
        y = predict_fault_ratio(fr_m, fr_tok, sentence, device=device)  # [2]
        pred = project_pair_to_basis(y, total=float(fault_basis))       # (dashcam, other) 합 = basis
        dc_f, ov_f = float(pred[0]), float(pred[1])
        if abs(dc_f - ov_f) < 1e-6:
            victim = "Undetermined"
        else:
            victim = "Dashcam Vehicle" if dc_f < ov_f else "Other Vehicle"
        fault_hint = f"Predicted fault (basis {int(fault_basis)}): Dashcam={dc_f:.1f}, Other={ov_f:.1f}; Likely victim: {victim}; Template: {sentence}"
    except Exception as e:
        fault_hint = f"[fault error: {e}]; Template: {sentence}"

    # 4) LLaVA 호출
    ENGINE.load_model(model_id)
    try:
        text = ENGINE.generate(
            video_path=video_path,
            user_message=user_msg or "",
            dashcam_info=dv_hint,
            other_info=ov_hint,
            classifier_topk=topk_text,
            style=style,
            num_frames=int(num_frames),
            frame_size=int(frame_size),
            temperature=float(temperature),
            do_sample=bool(sampling),
            history=history,
            fault_hint=fault_hint,   # ★ 추가 전달
        )
    except Exception as e:
        text = f"Generation error: {e}"

    # 5) 채팅창 출력(추가로 구조화 보조정보도 한 줄 더)
    aux = f"⚖️ Fault(basis {int(fault_basis)}): DC {dc_f:.1f} / OV {ov_f:.1f} | Victim: {victim}\n📝 {sentence}"
    if user_msg:
        history = history + [{"role": "user", "content": user_msg}]
    history = history + [{"role": "assistant", "content": text + "\n\n" + aux}]
    return history

with gr.Blocks(title="Video-LLaVA Chatbot") as demo:
    gr.Markdown("## 🎥 Video-LLaVA Chatbot — Multi-turn video-first accident analysis")
    with gr.Row():
        with gr.Column(scale=1):
            model_id = gr.Textbox(label="HF Model ID / local path", value="LanguageBind/Video-LLaVA-7B-hf")
            load_btn = gr.Button("Load / Reload Model")
            load_status = gr.Textbox(label="Load status", value="", interactive=False)
            video_name = gr.Textbox(label="Video Name (search in folders)",
                                    placeholder="예) crash_000123.mp4 또는 2024-09-*.mp4")
            scan_btn = gr.Button("Scan Videos")
            video_picker = gr.Dropdown(label="Pick found video (fills Video Name)", choices=[], value=None)

            video = gr.Video(label="Video (mp4/mov/webm)", interactive=True)
            # dashcam_info = gr.Textbox(label="Dashcam Vehicle hint (optional)")
            # other_info = gr.Textbox(label="Other Vehicle hint (optional)")
            # classifier_topk = gr.Textbox(label="Classifier top-k (optional)")
            clf_ckpt     = gr.Textbox(label="Classifier CKPT path", value="/app/checkpoints/best_exact_ep13_r3d18.pth")
            clf_backbone = gr.Dropdown(label="Classifier backbone", choices=["r3d18","timesformer","videomae"], value="r3d18")
            clf_pretrained = gr.Checkbox(label="Use pretrained backbone", value=False)
            clf_topk     = gr.Slider(1,5,value=3,step=1,label="Classifier Top-K")

            fault_ckpt   = gr.Textbox(label="Fault-BERT CKPT path", value="/app/text-train/fault_ratio_bert.pt")
            fault_model  = gr.Textbox(label="Fault-BERT model_name", value="bert-base-uncased")
            fault_basis  = gr.Slider(2, 20, value=10, step=1, label="Fault ratio basis")

            style = gr.Dropdown(label="Output style", choices=["short", "brief", "detailed"], value="brief")
            num_frames = gr.Slider(4, 16, value=8, step=1, label="Frames")
            frame_size = gr.Slider(160, 336, value=224, step=16, label="Frame size")
            temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")
            sampling = gr.Checkbox(value=False, label="Enable sampling")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=520, show_copy_button=True, type="messages")
            state = gr.State([])  # 대화 히스토리
            user_msg = gr.Textbox(label="Your message", placeholder="추가 지시 입력")
            send_btn = gr.Button("Send ▶️")

    load_btn.click(ui_load_model, inputs=[model_id], outputs=[load_status])
    scan_btn.click(
        lambda: list_candidates(),
        inputs=[],
        outputs=[video_picker]
    )
    video_picker.change(lambda p: os.path.basename(p), inputs=[video_picker], outputs=[video_name])
    send_btn.click(
        ui_generate,
        inputs=[
            state, user_msg, video, video_name, model_id,
            clf_ckpt, clf_backbone, clf_pretrained, clf_topk,
            fault_ckpt, fault_model, fault_basis,           # ★ 추가
            style, num_frames, frame_size, temperature, sampling
        ],
        outputs=[chatbot],
    ).then(lambda h: h, inputs=[chatbot], outputs=[state])

if __name__ == "__main__":
    try:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    # ✅ 포트/호스트/공유 URL 안전 설정
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", 7860)))
    share = os.environ.get("GRADIO_SHARE", "1") == "1"  # 원격/도커면 True 유지 추천

    demo.queue()  # 대기열 활성화(응답 안뜨는 이슈 예방)
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_error=True,
        inbrowser=False,  # 서버에서 자동 브라우저 오픈 방지
        prevent_thread_lock=False,  # 일부 환경에서 블랭크 방지
    )
    print(f"\n[Gradio] listening on http://{server_name}:{server_port}  (share={share})")
