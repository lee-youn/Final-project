# eval_cls2ratio.py
# pip install torch torchvision transformers opencv-python numpy pandas scikit-learn matplotlib wandb
import os, re, json, glob, argparse, collections
from typing import List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import wandb
from tqdm.auto import tqdm
import cv2

from transformers import AutoModel, AutoTokenizer, TimesformerModel, VideoMAEModel
from torchvision.models.video import r3d_18, R3D_18_Weights

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
def _compute_motion_roi_boxes(frames_rgb, thr=25, min_area=0.005, expand=0.15):
    import cv2, numpy as np
    boxes=[]; H,W=frames_rgb[0].shape[:2]; min_px=int(min_area*H*W)
    prev=cv2.cvtColor(frames_rgb[0], cv2.COLOR_RGB2GRAY)
    for t in range(1,len(frames_rgb)):
        g=cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2GRAY)
        diff=cv2.absdiff(g,prev); _,m=cv2.threshold(diff,thr,255,cv2.THRESH_BINARY)
        m=cv2.medianBlur(m,5); cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c=max(cnts,key=cv2.contourArea)
            if cv2.contourArea(c)>=min_px:
                x,y,w,h=cv2.boundingRect(c); cx,cy=x+w/2,y+h/2
                w=int(w*(1+expand)); h=int(h*(1+expand))
                x1=max(0,int(cx-w/2)); y1=max(0,int(cy-h/2))
                x2=min(W-1,x1+w); y2=min(H-1,y1+h)
                boxes.append((x1,y1,x2,y2)); prev=g; continue
        boxes.append(None); prev=g
    boxes=[boxes[0] if boxes and boxes[0] is not None else None]+boxes
    return boxes

def _crop_roi_video(frames_rgb, boxes, out_size=224):
    import cv2, torch, numpy as np
    import torchvision.transforms.functional as VF
    H,W=frames_rgb[0].shape[:2]
    def center_box():
        s=min(H,W); x1=(W-s)//2; y1=(H-s)//2; return (x1,y1,x1+s,y1+s)
    outs=[]
    for i,fr in enumerate(frames_rgb):
        x1,y1,x2,y2=(boxes[i] or center_box())
        crop=cv2.resize(fr[y1:y2, x1:x2], (out_size,out_size))
        outs.append(torch.from_numpy(crop).permute(2,0,1).float()/255.0)
    vid=torch.stack(outs,0)
    vid=VF.normalize(vid, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    return vid.permute(1,0,2,3)  # (C,T,H,W)

@torch.no_grad()
def build_classifier_input(vpath, model, num_frames=16, size=224):
    import cv2, numpy as np, torch
    # 1) 원본 프레임 추출
    cap = cv2.VideoCapture(vpath)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(0, total-1), num_frames).astype(int)
    frames_rgb = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if ok:
            frames_rgb.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames_rgb:
        raise RuntimeError(f"no decoded frames: {vpath}")

    # 2) Global 텐서
    import torch
    g = [cv2.resize(fr, (size, size)) for fr in frames_rgb]
    g = torch.from_numpy(np.stack(g)).permute(0,3,1,2).float()/255.0  # (T,C,H,W)
    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    g = (g - mean)/std
    xg = g.permute(1,0,2,3).unsqueeze(0)  # (1,C,T,H,W)

    # 3) Two-Branch면 ROI도 생성
    if getattr(model, "is_two_branch", False):
        boxes = _compute_motion_roi_boxes(frames_rgb, thr=25, min_area=0.005, expand=0.15)
        xr = _crop_roi_video(frames_rgb, boxes, out_size=size).unsqueeze(0)  # (1,C,T,H,W)
        return (xg, xr)
    else:
        return xg



def _dict_to_list_by_id(d: dict): return [d[k] for k in sorted(d.keys())]
LABELS = {
    "dv": _dict_to_list_by_id(dashcam_vehicle_info),
    "pl": _dict_to_list_by_id(real_categories_ids_2nd),
    "ov": _dict_to_list_by_id(other_vehicle_info),
}

# ================
# (1) 유틸들
# ================

def snap_pair_to_integer_basis(v, total=10):
    """
    v: array-like [a, b] (합이 total 근처라고 가정)
    반환: 합=total, 각 성분이 정수인 쌍 (대시캠 a를 반올림, 나머지는 보전)
    """
    v = np.asarray(v, dtype=float)
    v = np.maximum(v, 0.0)
    s = float(v.sum())
    if s <= 0:
        a = int(total // 2)
        return np.array([float(a), float(total - a)], dtype=float)
    v = v * (total / s)  # 합 total로 재정규화
    a_int = int(np.floor(v[0] + 0.5))
    a_int = max(0, min(total, a_int))
    b_int = total - a_int
    return np.array([float(a_int), float(b_int)], dtype=float)

def normalize_pair_100(p):
    a, b = float(p[0]), float(p[1])
    if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
        a *= 100.0; b *= 100.0
    a = max(a, 0.0); b = max(b, 0.0)
    s = a + b
    if s == 0.0:
        return [50.0, 50.0]
    scale = 100.0 / s
    return [a*scale, b*scale]

class TwoBranchVideoClassifier(nn.Module):
    def __init__(self, n_dv, n_ov):
        super().__init__()
        self.is_two_branch = True
        self.g = r3d_18(weights=R3D_18_Weights.KINETICS400_V1); dim_g = self.g.fc.in_features; self.g.fc = nn.Identity()
        self.r = r3d_18(weights=R3D_18_Weights.KINETICS400_V1); dim_r = self.r.fc.in_features; self.r.fc = nn.Identity()
        self.proj_g = nn.Sequential(nn.LayerNorm(dim_g), nn.Linear(dim_g, dim_g), nn.ReLU(), nn.Dropout(0.3))
        self.proj_r = nn.Sequential(nn.LayerNorm(dim_r), nn.Linear(dim_r, dim_r), nn.ReLU(), nn.Dropout(0.3))
        self.dv = nn.Linear(dim_g, n_dv)
        self.fuse = nn.Sequential(nn.LayerNorm(dim_g+dim_r), nn.Linear(dim_g+dim_r, dim_g), nn.ReLU(), nn.Dropout(0.3))
        self.ov = nn.Linear(dim_g, n_ov)
        self.is_two_branch = True

    def forward(self, x_pair):
        xg, xr = x_pair
        zg = self.g(xg); zr = self.r(xr)
        hg = self.proj_g(zg); hr = self.proj_r(zr)
        ldv = self.dv(hg)
        hf  = self.fuse(torch.cat([hg, hr], dim=-1))
        lov = self.ov(hf)
        return ldv, lov
    
def to_basis(pair100, target_basis=10.0):
    factor = 100.0 / target_basis
    return [p / factor for p in pair100]

def project_pair_to_basis(v, total=10.0):
    v = np.maximum(np.asarray(v, dtype=float), 0.0)
    s = float(v.sum())
    if s <= 0:
        return np.array([total/2.0, total/2.0], dtype=float)
    return v * (total / s)

def ratio_from_pairs(y):
    s = np.clip(y.sum(axis=1, keepdims=True), 1e-6, None)
    return (y[:, [0]] / s).ravel()

def pairs_from_ratio(p, total=10.0):
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    a = total * p
    b = total * (1.0 - p)
    return np.stack([a, b], axis=1)

def find_video_file(video_root: str, name_or_rel: str) -> Optional[str]:
    cand = os.path.join(video_root, name_or_rel)
    if os.path.exists(cand):
        return cand
    stem = os.path.splitext(name_or_rel)[0]
    for ext in (".mp4",".avi",".mov",".mkv"):
        p = os.path.join(video_root, stem + ext)
        if os.path.exists(p):
            return p
    g = glob.glob(os.path.join(video_root, stem + "*"))
    return g[0] if g else None

def compute_metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y, yhat)
    try:
        rmse = mean_squared_error(y, yhat, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y, yhat))
    r2 = r2_score(y, yhat)
    mae_dc = mean_absolute_error(y[:, 0], yhat[:, 0])
    mae_ov = mean_absolute_error(y[:, 1], yhat[:, 1])
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2),
            "MAE_dashcam": float(mae_dc), "MAE_other": float(mae_ov), "count": int(len(y))}

def save_plots(y: np.ndarray, yhat: np.ndarray, target_basis: float, out_prefix: str) -> dict:
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    paths = {}
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    for i,title in enumerate(["Dashcam","Other Vehicle"]):
        ax[i].scatter(y[:,i], yhat[:,i], alpha=0.5)
        ax[i].plot([0,target_basis],[0,target_basis],"--",color="gray")
        ax[i].set_title(f"{title} Fault (basis={target_basis})")
        ax[i].set_xlabel("True"); ax[i].set_ylabel("Pred")
        ax[i].set_xlim(0,target_basis); ax[i].set_ylim(0,target_basis)
    plt.tight_layout()
    p_sc = f"{out_prefix}_scatter.png"
    plt.savefig(p_sc); plt.close(fig)
    paths["scatter"] = p_sc

    err = np.abs(yhat - y)
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.hist(err[:,0], bins=20, alpha=0.6, label="Dashcam")
    ax2.hist(err[:,1], bins=20, alpha=0.6, label="Other")
    ax2.set_title(f"Absolute Error Histogram (basis={target_basis})")
    ax2.set_xlabel("Abs Error"); ax2.set_ylabel("Count"); ax2.legend()
    plt.tight_layout()
    p_hist = f"{out_prefix}_err_hist.png"
    plt.savefig(p_hist); plt.close(fig2)
    paths["err_hist"] = p_hist
    return paths

# =========================
# (2) 비디오 로딩 (분류기 입력)
# =========================
@torch.no_grad()
def build_classifier_input(vpath, model, num_frames=16, size=224):
    # 공통: 원본 프레임도 확보
    cap = cv2.VideoCapture(vpath)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(0, total-1), num_frames).astype(int)
    frames_rgb = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok: continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_rgb.append(frame)
    cap.release()
    if not frames_rgb:
        raise RuntimeError(f"no decoded frames: {vpath}")

    # Global
    g = [cv2.resize(fr, (size,size)) for fr in frames_rgb]
    g = torch.from_numpy(np.stack(g)).permute(0,3,1,2).float()/255.0  # (T,C,H,W)
    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    g = (g - mean)/std
    xg = g.permute(1,0,2,3).unsqueeze(0)  # (1,C,T,H,W)

    if getattr(model, "is_two_branch", False):
        boxes = _compute_motion_roi_boxes(frames_rgb, thr=25, min_area=0.005, expand=0.15)
        xr = _crop_roi_video(frames_rgb, boxes, out_size=size).unsqueeze(0)  # (1,C,T,H,W)
        return (xg, xr)
    else:
        return xg
    
@torch.no_grad()
def load_video_tensor(path, num_frames=16, size=224):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"no frames: {path}")
    idxs = np.linspace(0, total-1, num_frames).astype(int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok: continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size,size))
        ten = torch.from_numpy(frame).permute(2,0,1).float()/255.0
        frames.append(ten)
    cap.release()
    if not frames:
        raise RuntimeError(f"no decoded frames: {path}")
    vid = torch.stack(frames, 0)  # (T,C,H,W)
    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    vid = (vid - mean)/std
    return vid.permute(1,0,2,3)  # (C,T,H,W)

# =========================
# (3) 분류 모델 (3-헤드)
# =========================
class TripleHeadVideoClassifier(nn.Module):
    def __init__(self, n_dv, n_pl, n_ov, backbone="r3d18", pretrained=False):
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
        self.pl = nn.Linear(feat, n_pl)
        self.ov = nn.Linear(feat, n_ov)

    def forward(self, x):
        if self.backbone_name == "r3d18":
            z = self.backbone(x)              # (B, feat)
        else:
            x = x.permute(0,2,1,3,4)          # (B,T,C,H,W)
            out = self.backbone(x)
            z = out.last_hidden_state.mean(1) # (B, feat)
        return self.dv(z), self.pl(z), self.ov(z)

def load_classifier(ckpt_path: str, backbone="r3d18", device="cuda",
                    pretrained=False, force_two_branch: bool=False):
    sd_raw = torch.load(ckpt_path, map_location="cpu")
    sd = sd_raw.get("model", sd_raw) if isinstance(sd_raw, dict) else sd_raw

    # DataParallel prefix 제거
    new_sd = {}
    for k, v in sd.items():
        new_sd[k[7:]] = v if k.startswith("module.") else v

    # 🔴 무조건 Two-Branch로 강제
    print("[load_classifier] ✅ force Two-Branch (ignore auto-detect)")
    model = TwoBranchVideoClassifier(n_dv=len(LABELS['dv']), n_ov=len(LABELS['ov']))
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"[load_classifier] missing={len(missing)} unexpected={len(unexpected)}")

    model.to(device).eval()
    return model
# =========================
# (4) Text→Fault 회귀
# =========================
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
    tok = AutoTokenizer.from_pretrained(model_name)
    return model, tok

@torch.no_grad()
def predict_fault_ratio(model, tokenizer, text: str, device="cuda", max_length=256):
    inputs = tokenizer(text, return_tensors="pt",
                       padding="max_length", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pred = model(input_ids=input_ids, attention_mask=attention_mask)  # [1,2]
    return pred.squeeze(0).float().cpu().numpy()

# =========================
# (5) 분류→문장 템플릿
# =========================
def render_template_from_labels(dv_text: str, ov_text: str, place_text: Optional[str]=None) -> str:
    # 최소 요구형 (요청 템플릿)
    # Example:
    # "At an unsignalized intersection, the Dashcam Vehicle was <dashcam_info>, while the Other Vehicle was <other_info>."
    if place_text:
        # 원하면 place도 앞에 녹일 수 있음 (논문 버전용)
        # e.g., "At an unsignalized intersection (Roads Of Equal Width), ..."
        return (f"At an unsignalized intersection ({place_text}), the Dashcam Vehicle was {dv_text}, "
                f"while the Other Vehicle was {ov_text}.")
    else:
        return (f"At an unsignalized intersection, the Dashcam Vehicle was {dv_text}, "
                f"while the Other Vehicle was {ov_text}.")

# =========================
# (6) 한 영상 처리
# =========================
@torch.no_grad()
def classify_video_and_make_sentence(vpath: str,
                                     clf_model,
                                     device: str = "cuda",
                                     num_frames: int = 16,
                                     size: int = 224,
                                     use_place: bool = False) -> Dict:
    # 🔴 여기서 반드시 build_classifier_input을 사용
    xin = build_classifier_input(vpath, clf_model, num_frames=num_frames, size=size)
    if isinstance(xin, tuple):
        xin = (xin[0].to(device), xin[1].to(device))   # (xg, xr)
    else:
        xin = xin.to(device)                           # single-branch fallback

    out = clf_model(xin)

    # 출력 형태 분기(두-브랜치=2헤드, 트리플=3헤드)
    if isinstance(out, (list, tuple)) and len(out) == 2:
        ld, lv = out
        lp = None
    elif isinstance(out, (list, tuple)) and len(out) == 3:
        ld, lp, lv = out
    else:
        raise RuntimeError("Unexpected classifier outputs")

    pd_dv = ld.softmax(-1)[0].detach().cpu().numpy()
    pd_ov = lv.softmax(-1)[0].detach().cpu().numpy()
    i_dv = int(pd_dv.argmax()); i_ov = int(pd_ov.argmax())
    dv_text = LABELS["dv"][i_dv]; ov_text = LABELS["ov"][i_ov]

    place_text = None
    if (lp is not None) and use_place:
        pd_pl = lp.softmax(-1)[0].detach().cpu().numpy()
        place_text = LABELS["pl"][int(pd_pl.argmax())]

    sent = render_template_from_labels(dv_text, ov_text, place_text if use_place else None)
    return {
        "sentence": sent,
        "dv_idx": i_dv, "ov_idx": i_ov,
        "dv_text": dv_text, "ov_text": ov_text,
        "pl_text": place_text
    }
# =========================
# (7) 메인 평가 루프
# =========================
def evaluate_on_json_cls2ratio(eval_json_path: str,
                               classifier_ckpt: str,
                               fault_ckpt: str,
                               out_json_path: str,
                               video_root: str,
                               backbone: str = "r3d18",
                               classifier_pretrained: bool = False,
                               model_name: str = "bert-base-uncased",
                               target_basis: float = 10.0,
                               num_frames: int = 16,
                               size: int = 224,
                               use_place_in_sentence: bool = False,
                               verbose: bool = True,
                               print_every: int = 1,
                               device_classify: str = "cuda:0",
                               device_fault: str = "cuda:0", 
                               force_two_branch: bool = False):
    # 체크
    if not os.path.exists(eval_json_path): raise FileNotFoundError(eval_json_path)
    if not os.path.exists(classifier_ckpt): raise FileNotFoundError(classifier_ckpt)
    if not os.path.exists(fault_ckpt): raise FileNotFoundError(fault_ckpt)
    if not os.path.isdir(video_root): raise NotADirectoryError(video_root)

    wandb.init(project="cls2ratio", config={
        "eval_json": eval_json_path,
        "classifier_ckpt": classifier_ckpt,
        "fault_ckpt": fault_ckpt,
        "video_root": video_root,
        "backbone": backbone,
        "classifier_pretrained": classifier_pretrained,
        "target_basis": target_basis,
        "num_frames": num_frames,
        "size": size,
        "use_place_in_sentence": use_place_in_sentence,
    }, job_type="evaluation")

    clf = load_classifier(classifier_ckpt, backbone=backbone, device=device_classify,
                          pretrained=classifier_pretrained, force_two_branch=force_two_branch)
    fr_model, fr_tok = load_fault_model(fault_ckpt, model_name=model_name, device=device_fault)

    data = json.load(open(eval_json_path, "r", encoding="utf-8"))
    if not isinstance(data, list): raise ValueError("eval_json must be a list")

    preds_basis, labels_basis = [], []
    results = []
    out_prefix = os.path.splitext(out_json_path)[0]
    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)

    for i, row in enumerate(tqdm(data, total=len(data), dynamic_ncols=True, desc="Evaluating")):
        # GT
        gt_basis = None; gt100 = None
        if "dashcam_vehicle_negligence" in row and "other_vehicle_negligence" in row:
            gt100 = normalize_pair_100([row["dashcam_vehicle_negligence"], row["other_vehicle_negligence"]])
            gt_basis = np.array(to_basis(gt100, target_basis), dtype=float)

        # 비디오 경로
        video_name = row.get("video_name") or row.get("video_path") or ""
        vpath = find_video_file(video_root, video_name) if video_name else None
        if not vpath:
            raw = row.get("video_path")
            if raw and os.path.exists(raw): vpath = raw
        if not vpath:
            results.append({"idx": i, "video_name": video_name, "error": "video_not_found"})
            if verbose: print(f"[{i+1}/{len(data)}] {video_name} | VIDEO NOT FOUND", flush=True)
            continue

        try:
            # 1) 분류 → 문장
            cls_out = classify_video_and_make_sentence(
                vpath, clf, device=device_classify, num_frames=num_frames, size=size,
                use_place=use_place_in_sentence
            )
            sentence = cls_out["sentence"]

            # 2) 문장 → 회귀
            y = predict_fault_ratio(fr_model, fr_tok, sentence, device=device_fault)
            pred_basis = project_pair_to_basis(y, total=target_basis)

            snap_total = int(round(target_basis))
            pred_basis_snapped = snap_pair_to_integer_basis(pred_basis, total=snap_total)

        except Exception as e:
            results.append({"idx": i, "video_name": video_name, "error": f"pipeline_failed: {e}"})
            if verbose: print(f"[{i+1}/{len(data)}] {video_name} | ERROR: {e}", flush=True)
            continue

        preds_basis.append(pred_basis)
        out_item = {
            "idx": i,
            "video_name": video_name,
            "sentence_from_classifier": sentence,
            "dv_pred_label": cls_out["dv_text"],
            "ov_pred_label": cls_out["ov_text"],
            "pl_pred_label": cls_out["pl_text"],
            "pred_basis_dashcam": float(pred_basis[0]),
            "pred_basis_other": float(pred_basis[1]),
            "pred_100_dashcam": float(pred_basis[0]*(100.0/target_basis)),
            "pred_100_other": float(pred_basis[1]*(100.0/target_basis)),
            "pred_basis_dashcam_int": float(pred_basis_snapped[0]),
            "pred_basis_other_int":   float(pred_basis_snapped[1]),
        }

        if gt_basis is not None:
            out_item["gt_basis_dashcam"] = float(gt_basis[0])
            out_item["gt_basis_other"]   = float(gt_basis[1])
            out_item["gt_100_dashcam"]   = float(gt100[0])
            out_item["gt_100_other"]     = float(gt100[1])
            out_item["abs_err_basis_dashcam"] = abs(out_item["gt_basis_dashcam"] - out_item["pred_basis_dashcam"])
            out_item["abs_err_basis_other"]   = abs(out_item["gt_basis_other"]   - out_item["pred_basis_other"])
            labels_basis.append(gt_basis)

        results.append(out_item)

        if verbose and (i==0 or (i+1)%print_every==0 or i+1==len(data)):
            msg = f"[{i+1}/{len(data)}] {video_name} | pred_basis=[{out_item['pred_basis_dashcam']:.2f}, {out_item['pred_basis_other']:.2f}]"
            msg = f"[{i+1}/{len(data)}] {video_name} | pred_basis=[{out_item['pred_basis_dashcam_int']:.2f}, {out_item['pred_basis_other_int']:.2f}]"
            if gt_basis is not None:
                msg += f" | gt_basis=[{gt_basis[0]:.2f}, {gt_basis[1]:.2f}]"
            msg += "\n" + f"📝 {sentence}"
            print(msg, flush=True)

        if (i+1) % 20 == 0:
            wandb.log({"eval_progress_samples": i+1})

    # =========================
    # 메트릭/저장
    # =========================
    metrics = {}
    pred_rows = [r for r in results if "pred_basis_dashcam" in r]
    if len(pred_rows) and len(labels_basis):
        yhat = np.array([[r["pred_basis_dashcam"], r["pred_basis_other"]] for r in pred_rows], dtype=float)
        y    = np.array(labels_basis, dtype=float)

        # 1) 보정 전 메트릭
        metrics_pre = compute_metrics(y, yhat)
        metrics_pre["target_basis"] = target_basis

        # 2) 등단조 보정
        from sklearn.isotonic import IsotonicRegression
        p_hat  = ratio_from_pairs(yhat)
        p_true = ratio_from_pairs(y)
        try:
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True).fit(p_hat, p_true)
            p_cal = iso.transform(p_hat)
        except Exception:
            bins = np.linspace(0,1,11)
            idx  = np.clip(np.digitize(p_hat, bins)-1, 0, 9)
            bin_mean = np.array([p_true[idx==b].mean() if np.any(idx==b) else (bins[b]+bins[b+1])/2 for b in range(10)])
            p_cal = bin_mean[idx]

        yhat_cal = pairs_from_ratio(p_cal, total=target_basis)
        yhat_cal = np.vstack([project_pair_to_basis(v, total=target_basis) for v in yhat_cal])

        # 3) 정수 스냅(반올/반내림) – 합이 snap_total(기본 target_basis 반올림)이 되도록
        snap_total = int(round(target_basis))
        yhat_int     = np.vstack([snap_pair_to_integer_basis(v, total=snap_total) for v in yhat])
        yhat_cal_int = np.vstack([snap_pair_to_integer_basis(v, total=snap_total) for v in yhat_cal])

        # 4) 정수 스냅 메트릭
        m_int     = compute_metrics(y, yhat_int)
        m_int_cal = compute_metrics(y, yhat_cal_int)

        # 5) 보정 후 메트릭 + 결과 합치기
        for k, r in enumerate(pred_rows):
            r["pred_basis_dashcam_cal"] = float(yhat_cal[k, 0])
            r["pred_basis_other_cal"]   = float(yhat_cal[k, 1])
            # (원하면 정수 스냅 결과도 저장)
            r["pred_basis_dashcam_int"] = float(yhat_int[k, 0])
            r["pred_basis_other_int"]   = float(yhat_int[k, 1])
            r["pred_basis_dashcam_cal_int"] = float(yhat_cal_int[k, 0])
            r["pred_basis_other_cal_int"]   = float(yhat_cal_int[k, 1])

        metrics_cal = compute_metrics(y, yhat_cal)
        metrics = {
            **metrics_pre,
            **{f"cal/{k}": v for k, v in metrics_cal.items()},
            # 정수 스냅 요약(필요한 것만 넣음)
            "int/MAE": m_int["MAE"],
            "int/R2":  m_int["R2"],
            "int/RMSE": m_int["RMSE"],
            "int_cal/MAE": m_int_cal["MAE"],
            "int_cal/R2":  m_int_cal["R2"],
            "int_cal/RMSE": m_int_cal["RMSE"],
        }

        # 6) 플롯 + 저장
        out_prefix = os.path.splitext(out_json_path)[0]
        plots_pre = save_plots(y, yhat, target_basis, out_prefix+"_precal")
        plots_cal = save_plots(y, yhat_cal, target_basis, out_prefix+"_cal")
        plots = {**plots_pre, **{f"cal_{k}": v for k,v in plots_cal.items()}}

        out_csv_path = f"{out_prefix}.csv"
        pd.DataFrame(results).to_csv(out_csv_path, index=False, encoding="utf-8")
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "results": results}, f, ensure_ascii=False, indent=2)

        # 7) W&B 로그 (조건부로 안전하게)
        log_payload = {
            "eval/MAE": metrics.get("MAE"),
            "eval/RMSE": metrics.get("RMSE"),
            "eval/R2": metrics.get("R2"),
            "eval/MAE_dashcam": metrics.get("MAE_dashcam"),
            "eval/MAE_other": metrics.get("MAE_other"),
            "eval_cal/MAE": metrics.get("cal/MAE"),
            "eval_cal/RMSE": metrics.get("cal/RMSE"),
            "eval_cal/R2": metrics.get("cal/R2"),
            "eval_cal/MAE_dashcam": metrics.get("cal/MAE_dashcam"),
            "eval_cal/MAE_other": metrics.get("cal/MAE_other"),
            "eval_int/MAE": metrics.get("int/MAE"),
            "eval_int/R2": metrics.get("int/R2"),
            "eval_int_cal/MAE": metrics.get("int_cal/MAE"),
            "eval_int_cal/R2": metrics.get("int_cal/R2"),
        }
        log_payload = {k: v for k, v in log_payload.items() if v is not None}
        if log_payload:
            wandb.log(log_payload)

        for k, pth in plots.items():
            if os.path.exists(pth):
                wandb.log({f"plots/{k}": wandb.Image(pth)})

        try:
            table = wandb.Table(dataframe=pd.DataFrame(results))
            wandb.log({"eval/table": table})
        except Exception:
            pass
    else:
        # GT가 없거나 예측이 전혀 없을 때
        out_prefix = os.path.splitext(out_json_path)[0]
        out_csv_path = f"{out_prefix}.csv"
        pd.DataFrame(results).to_csv(out_csv_path, index=False, encoding="utf-8")
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": {}, "results": results}, f, ensure_ascii=False, indent=2)


    print("=== CLS→TEXT→RATIO Evaluation Summary ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved JSON: {out_json_path}")
    print(f"Saved CSV : {os.path.splitext(out_json_path)[0]}.csv")
    wandb.finish()

# =========================
# (8) CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Video → Classify → Template sentence → Text Regression (fault ratio)")
    p.add_argument("--eval_json", type=str,
                   default=os.environ.get("EVAL_JSON", "/app/data/raw/json/text-evaluate/video_accident_ratio_training_results_trimmed_unsignalized_validation_0901.json"))
    p.add_argument("--video_root", type=str,
                   default=os.environ.get("VIDEO_ROOT", "/app/data/raw/videos/validation_reencoded"))
    p.add_argument("--classifier_ckpt", type=str,
                   default=os.environ.get("CLS_CKPT", "/app/checkpoints/best_f1_ep18_r3d18_lstm_two.pth"))
    # p.add_argument("--classifier_ckpt", type=str,
    #                default=os.environ.get("CLS_CKPT", "/app/checkpoints/best_exact_ep13_r3d18.pth"))
    # p.add_argument("--classifier_ckpt", type=str,
    #                default=os.environ.get("CLS_CKPT", "/app/checkpoints/best_exact_ep11_r3d18_mlp.pth"))
    p.add_argument("--fault_ckpt", type=str,
                   default=os.environ.get("FAULT_CKPT", "/app/text-train/fault_ratio_bert.pt"))
    p.add_argument("--out_json", type=str,
                   default=os.environ.get("OUT_JSON", "/app/text-train/result_lstm_two/cls2ratio_eval.json"))
    p.add_argument("--backbone", type=str, default=os.environ.get("BACKBONE", "r3d18"),
                   choices=["r3d18","timesformer","videomae"])
    p.add_argument("--classifier_pretrained", action="store_true",
                   help="r3d18일 때 Kinetics400 pretrained stem 사용(ckpt로 덮어씀)")
    p.add_argument("--model_name", type=str, default=os.environ.get("MODEL_NAME","bert-base-uncased"))
    p.add_argument("--target_basis", type=float, default=float(os.environ.get("TARGET_BASIS", 10.0)))
    p.add_argument("--num_frames", type=int, default=int(os.environ.get("NUM_FRAMES", 16)))
    p.add_argument("--size", type=int, default=int(os.environ.get("SIZE", 224)))
    p.add_argument("--use_place_in_sentence", action="store_true",
                   help="문장 앞부분에 place 라벨을 괄호로 포함")
    p.add_argument("--gpus", type=str, default=os.environ.get("GPUS","0"),
                   help="예: '0' 또는 '0,1' (첫번째는 분류, 두번째는 회귀)")
    p.add_argument("--print_every", type=int, default=int(os.environ.get("PRINT_EVERY", 1)))
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--force_two_branch", action="store_true",
               help="항상 Two-Branch 분류기를 사용하고 ROI 입력을 생성합니다.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # 디바이스 배치
    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip()!=""]
    if torch.cuda.is_available() and len(gpu_list)>=1:
        device_classify = f"cuda:{gpu_list[0]}"
        device_fault    = f"cuda:{gpu_list[1]}" if len(gpu_list)>1 else device_classify
    else:
        device_classify = device_fault = "cpu"

    evaluate_on_json_cls2ratio(
        eval_json_path=args.eval_json,
        classifier_ckpt=args.classifier_ckpt,
        fault_ckpt=args.fault_ckpt,
        out_json_path=args.out_json,
        video_root=args.video_root,
        backbone=args.backbone,
        classifier_pretrained=args.classifier_pretrained,
        model_name=args.model_name,
        target_basis=args.target_basis,
        num_frames=args.num_frames,
        size=args.size,
        use_place_in_sentence=args.use_place_in_sentence,
        verbose=not args.quiet,
        print_every=args.print_every,
        device_classify=device_classify,
        device_fault=device_fault,
        force_two_branch=args.force_two_branch,
    )
