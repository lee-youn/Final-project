# train_video_desc.py
# pip install torch torchvision opencv-python numpy pandas
import os, csv, json, math, argparse, random, re
from typing import List, Dict, Tuple
import cv2
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18
from torchvision.transforms import functional as VF
from dataclasses import dataclass
import pandas as pd
import pandas as pd
from pathlib import Path
from transformers import TimesformerModel, VideoMAEModel
import wandb
from tqdm.auto import tqdm
import os, sys, glob, json, time, math, random, argparse, traceback
from sklearn.metrics import (
    f1_score, classification_report, accuracy_score,
    precision_recall_fscore_support, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef, confusion_matrix,
    top_k_accuracy_score
)
import matplotlib
matplotlib.use("Agg")              # NEW: 헤드리스
import matplotlib.pyplot as plt

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

# train_eval_videos_no_csv.py
# pip install torch torchvision opencv-python scikit-learn numpy matplotlib

import os, re, json, argparse, random
from glob import glob
from typing import Dict, List, Tuple
import cv2, numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.transforms import functional as VF
from sklearn.metrics import f1_score, classification_report, accuracy_score
import matplotlib.pyplot as plt


def _topk_acc(y_true, logits, k=3):
    # y_true: List[int], logits: torch.Tensor [N, C]
    # top_k_accuracy_score는 numpy array 필요
    import numpy as np
    probs = logits.softmax(dim=-1).detach().cpu().numpy()
    y = np.array(y_true)
    try:
        return float(top_k_accuracy_score(y, probs, k=k, labels=list(range(probs.shape[1]))))
    except Exception:
        return 0.0


def plot_confmat(y_true, y_pred, labels, title, out_path, wandb_log_key=None):
    from sklearn.metrics import ConfusionMatrixDisplay
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # 절대경로 + 디렉토리 보장
    out_path = os.path.abspath(out_path)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(8,8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(len(labels))])
    disp.plot(include_values=False, cmap="Blues", ax=ax, colorbar=True)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved confusion matrix: {out_path}")

    # 저장 파일이 있을 때만 wandb.log
    if wandb_log_key is not None and os.path.exists(out_path):
        try:
            import wandb
            wandb.log({wandb_log_key: wandb.Image(out_path)})
        except Exception as e:
            print(f"[WARN] wandb image log failed: {e}")

    return out_path


# ===== (A) 라벨 사전: 네가 준 dict 그대로 붙여넣기 =====
# 2nd / 3rd / 4th (생략 없이 그대로 사용)

def _dict_to_list_by_id(d: dict): return [d[k] for k in sorted(d.keys())]
# LABELS = {
#     "dashcam_vehicle_info": _dict_to_list_by_id(dashcam_vehicle_info),  # DV
#     "accident_place_feature": _dict_to_list_by_id(real_categories_ids_2nd),# Place
#     "other_vehicle_info": _dict_to_list_by_id(other_vehicle_info),    # OV
# }
# L2I = {
#     "dashcam_vehicle_info": {s:i for i,s in enumerate(LABELS["dashcam_vehicle_info"])},
#     "accident_place_feature": {s:i for i,s in enumerate(LABELS["accident_place_feature"])},
#     "other_vehicle_info": {s:i for i,s in enumerate(LABELS["other_vehicle_info"])},
# }
# NCLS = {
#     "dashcam_vehicle_info": len(LABELS["dashcam_vehicle_info"]),
#     "accident_place_feature": len(LABELS["accident_place_feature"]),
#     "other_vehicle_info": len(LABELS["other_vehicle_info"]),
# }
LABELS = {
    "dashcam_vehicle_info": _dict_to_list_by_id(dashcam_vehicle_info),  # DV
    "other_vehicle_info": _dict_to_list_by_id(other_vehicle_info),    # OV
}
L2I = {
    "dashcam_vehicle_info": {s:i for i,s in enumerate(LABELS["dashcam_vehicle_info"])},
    "other_vehicle_info": {s:i for i,s in enumerate(LABELS["other_vehicle_info"])},
}
NCLS = {
    "dashcam_vehicle_info": len(LABELS["dashcam_vehicle_info"]),
    "other_vehicle_info": len(LABELS["other_vehicle_info"]),
}

# ===== (B) 데이터셋: video.mp4 + video.json (같은 파일명) =====
def _read_table_any(path: str) -> list[dict]:
    """
    1) CSV (자동 구분자 sniff, engine='python')
    2) JSON (list/dict 또는 {"data": [...]})
    3) JSONL
    순서로 시도해서 rows(list[dict])를 반환
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    # 1) CSV 우선 (구분자 자동 추정)
    try:
        df = pd.read_csv(p, engine="python", sep=None)
        return df.to_dict("records")
    except Exception:
        pass

    # 2) JSON 전체 로드
    try:
        txt = p.read_text(encoding="utf-8")
        data = json.loads(txt)
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # 3) JSONL 라인별 파싱
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_by_stem(stem: str, root: str, exts=(".mp4",".mov",".avi")) -> str:
    """video_name(확장자 없는 스템) → 실제 파일 경로 탐색"""
    # 1) root 바로 아래
    for ext in exts:
        cand = os.path.join(root, stem + ext)
        if os.path.exists(cand):
            return cand
    # 2) 재귀 탐색
    for ext in exts:
        hits = glob(os.path.join(root, "**", stem + ext), recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"Video not found for stem={stem} under {root}")


def _maybe_to_idx(val, mapping: Dict[str,int], ncls: int):
    if isinstance(val, (int, np.integer)): 
        assert 0 <= int(val) < ncls, f"label id out of range: {val}"
        return int(val)
    val = str(val).strip()
    if val in mapping: return mapping[val]
    raise KeyError(f"Unknown label text: {val}")

def load_video_tensor(path, num_frames=16, size=224):
    import torchvision.transforms.functional as VF
    import torch
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total-1, num_frames).astype(int)

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok: continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size))
        ten = torch.from_numpy(frame).permute(2,0,1).float()/255.0  # (C,H,W)
        frames.append(ten)
    cap.release()

    if not frames:
        raise RuntimeError(f"no frames in {path}")

    # (T,C,H,W)
    vid = torch.stack(frames, 0)

    # normalize 프레임 단위로
    vid = VF.normalize(vid, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    return vid  # shape: (T,C,H,W)

# class VideoJsonDataset(Dataset):
#     """
#     루트 아래의 모든 *.mp4/*.mov/*.avi 를 수집.
#     각 비디오와 같은 stem의 JSON 파일에서 라벨을 읽음.
#     JSON 예:
#     {
#       "dashcam_vehicle_info": 18,  # 또는 문자열 라벨
#       "accident_place_feature": 10,
#       "other_vehicle_info": 0
#     }
#     """
#     def __init__(self, root: str, exts=(".mp4",".mov",".avi"), num_frames=16, size=224):
#         self.videos = []
#         for ext in exts:
#             self.videos += sorted(glob(os.path.join(root, f"**/*{ext}"), recursive=True))
#         self.num_frames = num_frames; self.size = size

#     def __len__(self): return len(self.videos)

#     def __getitem__(self, i):
#         vpath = self.videos[i]
#         stem = os.path.splitext(vpath)[0]
#         jpath = stem + ".json"
#         if not os.path.exists(jpath):
#             raise FileNotFoundError(f"Label JSON not found for: {vpath}")
#         meta = json.load(open(jpath, "r", encoding="utf-8"))

#         x = load_video_tensor(vpath, self.num_frames, self.size)   # [C,T,H,W]
#         dv = _maybe_to_idx(meta["dashcam_vehicle_info"], L2I["dashcam_vehicle_info"], NCLS["dashcam_vehicle_info"])
#         pl = _maybe_to_idx(meta["accident_place_feature"], L2I["accident_place_feature"], NCLS["accident_place_feature"])
#         ov = _maybe_to_idx(meta["other_vehicle_info"], L2I["other_vehicle_info"], NCLS["other_vehicle_info"])
#         return x, (dv, pl, ov)
class SingleCsvDataset(Dataset):
    """
    하나의 표 파일(확장자 무관: CSV/JSON/JSONL)에서 비디오 경로와 라벨을 읽어오는 데이터셋.
    지원 컬럼:
      - (권장) video_name: 확장자 없는 스템 → root에서 파일 탐색
      - (대안) video_path: 상대/절대 경로 → root와 join 또는 그대로 사용
    라벨 컬럼:
      dashcam_vehicle_info, accident_place_feature, other_vehicle_info (문자/정수 모두 OK)
    """
    def __init__(self, table_path: str, video_root: str, num_frames=16, size=224):
        self.rows = _read_table_any(table_path)
        if not self.rows:
            raise RuntimeError(f"No rows loaded from {table_path}")

        self.video_root = video_root
        self.num_frames = num_frames
        self.size = size
        
        self.first_col = list(self.rows[0].keys())[0]

        # 라벨 컬럼 확인
        # need = {"dashcam_vehicle_info","accident_place_feature","other_vehicle_info"}
        need = {"dashcam_vehicle_info","other_vehicle_info"}
        missing = need - set(self.rows[0].keys())
        if missing:
            raise KeyError(f"Missing {missing} in {table_path}")

    def __len__(self):
        return len(self.rows)
# ---- Video file detection helpers ----
    def _resolve_video(self, row: dict) -> str:
        # 무조건 첫 번째 컬럼 값 사용
        VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
        val = str(row[self.first_col]).strip()
        # 확장자 있으면 그대로, 없으면 스템으로 탐색
        if any(val.lower().endswith(ext) for ext in VIDEO_EXTS):
            return val if os.path.isabs(val) else os.path.join(self.video_root, val)
        stem = os.path.splitext(val)[0]
        return _resolve_by_stem(stem, self.video_root)

    # def __getitem__(self, i):
    #     r = self.rows[i]
    #     try:
    #         vpath = self._resolve_video(r)
    #         vid = load_video_tensor(vpath, self.num_frames, self.size)  # (T,C,H,W)
    #         vid = vid.permute(1,0,2,3)  # (C,T,H,W)
    #     except FileNotFoundError:
    #         # 못 찾으면 None 반환
    #         return None

    #     dv = _maybe_to_idx(r["dashcam_vehicle_info"],   L2I["dashcam_vehicle_info"],   NCLS["dashcam_vehicle_info"])
    #     pl = _maybe_to_idx(r["accident_place_feature"], L2I["accident_place_feature"], NCLS["accident_place_feature"])
    #     ov = _maybe_to_idx(r["other_vehicle_info"],     L2I["other_vehicle_info"],     NCLS["other_vehicle_info"])
    #     return vid, (dv, pl, ov)
    def __getitem__(self, i):
        r = self.rows[i]
        try:
            vpath = self._resolve_video(r)
            vid = load_video_tensor(vpath, self.num_frames, self.size)  # (T,C,H,W)
            vid = vid.permute(1,0,2,3)  # (C,T,H,W)
        except FileNotFoundError:
            # 못 찾으면 None 반환
            return None

        dv = _maybe_to_idx(r["dashcam_vehicle_info"],   L2I["dashcam_vehicle_info"],   NCLS["dashcam_vehicle_info"])
        ov = _maybe_to_idx(r["other_vehicle_info"],     L2I["other_vehicle_info"],     NCLS["other_vehicle_info"])
        return vid, (dv, ov)

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)   
# ===== (C) 모델 =====
class TripleHeadVideoClassifier(nn.Module):
    # def __init__(self, n_dv, n_pl, n_ov, backbone="r3d18"):
    def __init__(self, n_dv, n_ov, backbone="r3d18"):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "r3d18":
            self.backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            feat = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone == "timesformer":
            self.backbone = TimesformerModel.from_pretrained(
                "facebook/timesformer-base-finetuned-k400",
                use_safetensors=True
            )
            feat = self.backbone.config.hidden_size

        elif backbone == "videomae":
            self.backbone = VideoMAEModel.from_pretrained(
                "MCG-NJU/videomae-base-finetuned-kinetics",
                use_safetensors=True
            )
            feat = self.backbone.config.hidden_size

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # # 공통 3개 classification head
        # self.dv = nn.Linear(feat, n_dv)
        # # self.pl = nn.Linear(feat, n_pl)
        # self.ov = nn.Linear(feat, n_ov)

        # self.dv = nn.Sequential(
        #     nn.LayerNorm(feat),
        #     nn.Linear(feat, feat//2),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(feat//2, n_dv)
        # )
        # self.ov = nn.Sequential(
        #     nn.LayerNorm(feat),
        #     nn.Linear(feat, feat//2),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(feat//2, n_ov)
        # )
        self.shared_proj = nn.Sequential(
            nn.LayerNorm(feat),
            nn.Linear(feat, feat),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.dv = nn.Linear(feat, n_dv)
        self.ov = nn.Linear(feat, n_ov)
        
    def forward(self, x):
        if self.backbone_name == "r3d18":
            # (B, C, T, H, W)
            z = self.backbone(x)

        elif self.backbone_name == "timesformer":
            # (B, C, T, H, W) → (B, T, C, H, W)
            x = x.permute(0, 2, 1, 3, 4)
            out = self.backbone(x)
            z = out.last_hidden_state.mean(1)

        elif self.backbone_name == "videomae":
            # (B, C, T, H, W) → (B, T, C, H, W)
            x = x.permute(0, 2, 1, 3, 4)
            out = self.backbone(x)
            z = out.last_hidden_state.mean(1)

        # --- Pass through shared projection ---
        h = self.shared_proj(z)

        return self.dv(h), self.ov(h)

    # def forward(self, x):
    #     if self.backbone_name == "r3d18":
    #         # (B, C, T, H, W)
    #         z = self.backbone(x)

    #     elif self.backbone_name == "timesformer":
    #         # (B, C, T, H, W) → (B, T, C, H, W)
    #         x = x.permute(0, 2, 1, 3, 4)
    #         out = self.backbone(x)
    #         z = out.last_hidden_state.mean(1)  # [CLS] or mean pooling

    #     elif self.backbone_name == "videomae":
    #         # (B, C, T, H, W) → (B, T, C, H, W)
    #         x = x.permute(0, 2, 1, 3, 4)
    #         out = self.backbone(x)
    #         z = out.last_hidden_state.mean(1)

    #     return self.dv(z), self.ov(z)

# ===== (D) 학습/평가 =====
# def train_one_epoch(model, loader, opt, device, fp16=False, epoch=1, epochs=1, log_interval=10, global_step_offset=0):
#     model.train()


#     ce = nn.CrossEntropyLoss()
#     scaler = torch.cuda.amp.GradScaler(enabled=fp16)
#     running, n = 0.0, 0
#     global_step = global_step_offset

#     pbar = tqdm(loader, desc=f"Train [{epoch}/{epochs}]", dynamic_ncols=True)
#     for step, (x, (yd, yp, yv)) in enumerate(pbar, start=1):
#         x = x.to(device, non_blocking=True)
#         yd = yd.to(device); yp = yp.to(device); yv = yv.to(device)

#         opt.zero_grad(set_to_none=True)
#         with torch.cuda.amp.autocast(enabled=fp16):
#             ld, lp, lv = model(x)
#             loss = ce(ld, yd) + ce(lp, yp) + ce(lv, yv)

#         scaler.scale(loss).backward()
#         scaler.step(opt); scaler.update()

#         bs = x.size(0)
#         running += loss.item() * bs; n += bs
#         avg_loss = running / max(1, n)

#         # 진행바 업데이트 (주기적으로만 상세 postfix 갱신)
#         if (step % log_interval) == 0 or step == 1:
#             pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

#         global_step += 1

#     epoch_loss = running / max(1, n)

#     return epoch_loss, global_step
def train_one_epoch(model, loader, opt, device, fp16=False, epoch=1, epochs=1, log_interval=10, global_step_offset=0):
    model.train()
    ce = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    running, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Train [{epoch}/{epochs}]", dynamic_ncols=True)

    for step, batch in enumerate(pbar, start=1):
        if batch is None: continue
        x, (yd, yv) = batch
        x = x.to(device, non_blocking=True)
        yd = yd.to(device); yv = yv.to(device)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=fp16):
            ld, lv = model(x)
            loss = ce(ld, yd) + ce(lv, yv)

        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()

        bs = x.size(0)
        running += loss.item() * bs; n += bs
        if (step % log_interval) == 0 or step == 1:
            pbar.set_postfix({"loss": f"{(running/max(1,n)):.4f}"})
    return (running / max(1, n)), step

# @torch.no_grad()
# def evaluate(model, loader, device, epoch=None, epochs=None, save_confmat=True, backbone="r3d18"):
#     model.eval()
#     # y_dv, y_pl, y_ov = [], [], []
#     # p_dv, p_pl, p_ov = [], [], []
#     # logits_dv, logits_pl, logits_ov = [], [], []
#     y_dv, y_ov = [], [], []
#     p_dv, p_ov = [], [], []
#     logits_dv, logits_ov = [], [], []

#     it = loader
#     if epoch is not None and epochs is not None:
#         it = tqdm(loader, desc=f"Val   [{epoch}/{epochs}]", dynamic_ncols=True)

#     for x,(yd,yp,yv) in it:
#         x = x.to(device, non_blocking=True)
#         ld, lp, lv = model(x)
#         # 저장
#         logits_dv.append(ld.cpu()); logits_pl.append(lp.cpu()); logits_ov.append(lv.cpu())
#         p_dv += ld.argmax(-1).cpu().tolist()
#         p_pl += lp.argmax(-1).cpu().tolist()
#         p_ov += lv.argmax(-1).cpu().tolist()
#         y_dv += yd.tolist(); y_pl += yp.tolist(); y_ov += yv.tolist()

#     # concat logits
#     logits_dv = torch.cat(logits_dv, dim=0) if logits_dv else torch.empty(0)
#     logits_pl = torch.cat(logits_pl, dim=0) if logits_pl else torch.empty(0)
#     logits_ov = torch.cat(logits_ov, dim=0) if logits_ov else torch.empty(0)

#     def _all_metrics(y, p, logits, label_key):
#         # 기본 지표
#         acc = accuracy_score(y, p)
#         f1_ma = f1_score(y, p, average="macro", zero_division=0)
#         f1_mi = f1_score(y, p, average="micro", zero_division=0)
#         bal_acc = balanced_accuracy_score(y, p)
#         kappa = cohen_kappa_score(y, p)
#         mcc = matthews_corrcoef(y, p)
#         prec_ma, rec_ma, _, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
#         prec_mi, rec_mi, _, _ = precision_recall_fscore_support(y, p, average="micro", zero_division=0)
#         prec_wt, rec_wt, _, _ = precision_recall_fscore_support(y, p, average="weighted", zero_division=0)

#         # top-k (k=3)
#         top3 = _topk_acc(y, logits, k=3) if logits.numel() else 0.0

#         rep = classification_report(y, p, digits=3, zero_division=0)

#         out = {
#             "acc": acc,
#             "balanced_acc": bal_acc,
#             "f1_macro": f1_ma,
#             "f1_micro": f1_mi,
#             "precision_macro": prec_ma,
#             "recall_macro": rec_ma,
#             "precision_micro": prec_mi,
#             "recall_micro": rec_mi,
#             "precision_weighted": prec_wt,
#             "recall_weighted": rec_wt,
#             "cohen_kappa": kappa,
#             "mcc": mcc,
#             "top3_acc": top3,
#             "report": rep
#         }
#         return out

#     # m_dv = _all_metrics(y_dv, p_dv, logits_dv, "dashcam_vehicle_info")
#     # m_pl = _all_metrics(y_pl, p_pl, logits_pl, "accident_place_feature")
#     # m_ov = _all_metrics(y_ov, p_ov, logits_ov, "other_vehicle_info")
#     m_dv = _all_metrics(y_dv, p_dv, logits_dv, "dashcam_vehicle_info")
#     m_ov = _all_metrics(y_ov, p_ov, logits_ov, "other_vehicle_info")

#     # exact = np.mean((np.array(y_dv)==np.array(p_dv)) &
#     #                 (np.array(y_pl)==np.array(p_pl)) &
#     #                 (np.array(y_ov)==np.array(p_ov)))
#     exact = np.mean((np.array(y_dv)==np.array(p_dv)) &
#                     (np.array(y_ov)==np.array(p_ov)))
#     # out = {"dv":m_dv, "place":m_pl, "ov":m_ov, "exact_match": float(exact)}
#     out = {"dv":m_dv, "ov":m_ov, "exact_match": float(exact)}

#     # 혼동행렬 저장 (옵션)
#     # if save_confmat:
#     #     plot_confmat(y_dv, p_dv, LABELS["dashcam_vehicle_info"], "DV Confusion Matrix", f"cm_dv_{backbone}.png","val/dv/confmat")
#     #     plot_confmat(y_pl, p_pl, LABELS["accident_place_feature"], "Place Confusion Matrix", f"cm_place_{backbone}.png","val/dv/confmat")
#     #     plot_confmat(y_ov, p_ov, LABELS["other_vehicle_info"], "OV Confusion Matrix", f"cm_ov_{backbone}.png","val/dv/confmat")
#     if save_confmat:
#             plot_confmat(y_dv, p_dv, LABELS["dashcam_vehicle_info"], "DV Confusion Matrix", f"cm_dv_{backbone}.png","val/dv/confmat")
#             plot_confmat(y_ov, p_ov, LABELS["other_vehicle_info"], "OV Confusion Matrix", f"cm_ov_{backbone}.png","val/dv/confmat")

#     return out
@torch.no_grad()
def evaluate(model, loader, device, epoch=None, epochs=None, save_confmat=True, backbone="r3d18"):
    model.eval()
    ce = nn.CrossEntropyLoss()

    y_dv, y_ov = [], []
    p_dv, p_ov = [], []
    logits_dv, logits_ov = [], []

    running_loss, n = 0.0, 0

    it = loader if (epoch is None or epochs is None) else tqdm(loader, desc=f"Val   [{epoch}/{epochs}]", dynamic_ncols=True)
    for batch in it:
        if batch is None: 
            continue
        x, (yd, yv) = batch
        x = x.to(device, non_blocking=True)
        yd = yd.to(device); yv = yv.to(device)

        ld, lv = model(x)
        # ----- loss 누적 -----
        loss = ce(ld, yd) + ce(lv, yv)
        bs = x.size(0)
        running_loss += loss.item() * bs
        n += bs

        # ----- 예측/지표용 누적 -----
        logits_dv.append(ld.cpu()); logits_ov.append(lv.cpu())
        p_dv += ld.argmax(-1).cpu().tolist()
        p_ov += lv.argmax(-1).cpu().tolist()
        y_dv += yd.cpu().tolist(); y_ov += yv.cpu().tolist()

    # concat logits
    logits_dv = torch.cat(logits_dv, dim=0) if logits_dv else torch.empty(0)
    logits_ov = torch.cat(logits_ov, dim=0) if logits_ov else torch.empty(0)

    def _all_metrics(y, p, logits):
        acc = accuracy_score(y, p)
        f1_ma = f1_score(y, p, average="macro", zero_division=0)
        f1_mi = f1_score(y, p, average="micro", zero_division=0)
        bal_acc = balanced_accuracy_score(y, p)
        kappa = cohen_kappa_score(y, p)
        mcc = matthews_corrcoef(y, p)
        prec_ma, rec_ma, _, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
        prec_mi, rec_mi, _, _ = precision_recall_fscore_support(y, p, average="micro", zero_division=0)
        top3 = _topk_acc(y, logits, k=3) if logits.numel() else 0.0
        rep = classification_report(y, p, digits=3, zero_division=0)
        return {
            "acc": acc, "balanced_acc": bal_acc,
            "f1_macro": f1_ma, "f1_micro": f1_mi,
            "precision_macro": prec_ma, "recall_macro": rec_ma,
            "precision_micro": prec_mi, "recall_micro": rec_mi,
            "cohen_kappa": kappa, "mcc": mcc, "top3_acc": top3,
            "report": rep
        }

    m_dv = _all_metrics(y_dv, p_dv, logits_dv)
    m_ov = _all_metrics(y_ov, p_ov, logits_ov)

    exact = float(np.mean((np.array(y_dv)==np.array(p_dv)) &
                          (np.array(y_ov)==np.array(p_ov))))
    val_loss = running_loss / max(1, n)

    if save_confmat:
        plot_confmat(y_dv, p_dv, LABELS["dashcam_vehicle_info"], "DV Confusion Matrix", f"cm_dv_{backbone}.png","val/dv/confmat")
        plot_confmat(y_ov, p_ov, LABELS["other_vehicle_info"], "OV Confusion Matrix", f"cm_ov_{backbone}.png","val/ov/confmat")

    return {"loss": val_loss, "dv": m_dv, "ov": m_ov, "exact_match": exact}

# def plot_summary(metrics: Dict, out_path="results_metrics.png"):
#     names = [
#         "DV f1_macro","Place f1_macro","OV f1_macro",
#         "DV bal_acc","Place bal_acc","OV bal_acc",
#         "Exact-Match"
#     ]
#     vals = [
#         metrics["dv"]["f1_macro"], metrics["place"]["f1_macro"], metrics["ov"]["f1_macro"],
#         metrics["dv"]["balanced_acc"], metrics["place"]["balanced_acc"], metrics["ov"]["balanced_acc"],
#         metrics["exact_match"]
#     ]
#     plt.figure(figsize=(10,5))
#     plt.bar(range(len(names)), vals)
#     plt.xticks(range(len(names)), names, rotation=20)
#     plt.ylim(0,1.0)
#     plt.ylabel("Score")
#     plt.title("Validation Summary")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=180)
#     print(f"Saved plot: {out_path}")
def plot_summary(metrics: Dict, out_path="results_metrics.png"):
    names = ["DV P_macro","DV R_macro","DV F1_macro",
             "OV P_macro","OV R_macro","OV F1_macro",
             "Exact-Match"]
    vals = [
        metrics["dv"]["precision_macro"], metrics["dv"]["recall_macro"], metrics["dv"]["f1_macro"],
        metrics["ov"]["precision_macro"], metrics["ov"]["recall_macro"], metrics["ov"]["f1_macro"],
        metrics["exact_match"]
    ]
    plt.figure(figsize=(10,5))
    plt.bar(range(len(names)), vals)
    plt.xticks(range(len(names)), names, rotation=20)
    plt.ylim(0,1.0)
    plt.ylabel("Score")
    plt.title("Validation Summary (Precision/Recall/F1)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    print(f"Saved plot: {out_path}")


def split_dataset(ds: SingleCsvDataset, val_ratio=0.2, seed=42):
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    n_val = int(len(idxs)*val_ratio)
    val_idx, tr_idx = idxs[:n_val], idxs[n_val:]
    return torch.utils.data.Subset(ds, tr_idx), torch.utils.data.Subset(ds, val_idx)

def main():
    best_exact = 0.0 
    ap = argparse.ArgumentParser(conflict_handler="resolve")
    ap.add_argument("--backbone", type=str, default="r3d18",
                choices=["r3d18", "timesformer", "videomae"],
                help="영상 백본 선택")
    ap.add_argument("--mode", choices=["train","validate","predict"], default="train")
    ap.add_argument(
        "--train_csv",
        type=str,
        help="CSV with columns: video_path,dashcam_vehicle_info,accident_place_feature,other_vehicle_info",
        default="/app/data/raw/json/video-train/video_accident_caption_results_unsignalized_0811.csv",
    )
    ap.add_argument(
        "--val_csv",
        type=str,
        help="CSV/JSON path for validation",
        default="/app/data/raw/json/video-evaluate/video_accident_caption_results_unsignalized_validation_0901.csv",
    )
    ap.add_argument("--log_interval", type=int, default=1, help="스텝 로그/진행바 postfix 업데이트 주기")
    ap.add_argument("--train_video_root", type=str,
               default=os.environ.get("TRAIN_VIDEO_ROOT", "/app/data/raw/videos/training_reencoded"),
               help="Root dir for training videos (env TRAIN_VIDEO_ROOT overrides)")
    ap.add_argument("--val_video_root", type=str,
                default=os.environ.get("VAL_VIDEO_ROOT", "/app/data/raw/videos/validation_reencoded"),
                help="Root dir for validation videos (env VAL_VIDEO_ROOT overrides)")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--val_batch_size", type=int, default=8) 
    ap.add_argument("--dataparallel", action="store_true")         # NEW
    ap.add_argument("--save_dir", type=str, default="./checkpoints")  # NEW
    ap.add_argument("--out_ckpt", type=str, default=f"./checkpoints/desc.pth")  # (원하면 유지)

    ap.add_argument("--video", type=str, help="video path for prediction")
    ap.add_argument("--ckpt", type=str, help="checkpoint for prediction")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    wandb.init(project="video-desc", config=vars(args))

    if args.mode == "train":
        train_ds = SingleCsvDataset(args.train_csv, video_root=args.train_video_root,
                                    num_frames=args.num_frames, size=args.size)
        val_ds   = SingleCsvDataset(args.val_csv,   video_root=args.val_video_root,
                                    num_frames=args.num_frames, size=args.size)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_skip_none)
        val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, collate_fn=collate_skip_none)

        # model = TripleHeadVideoClassifier(NCLS["dashcam_vehicle_info"],
        #                                   NCLS["accident_place_feature"],
        #                                   NCLS["other_vehicle_info"],backbone=args.backbone).to(device)
        model = TripleHeadVideoClassifier(NCLS["dashcam_vehicle_info"],
                                          NCLS["other_vehicle_info"],backbone=args.backbone).to(device)
        if torch.cuda.device_count() > 1 and args.dataparallel:
            model = nn.DataParallel(model)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

        start_time = time.time()

        for ep in range(1, args.epochs+1):
            # === 1) Train ===
            tr_loss, _ = train_one_epoch(
                model, train_loader, opt, device,
                fp16=args.fp16, epoch=ep, epochs=args.epochs,
                log_interval=args.log_interval
            )

            # === 2) Validate ===
            val = evaluate(model, val_loader, device, epoch=ep, epochs=args.epochs, backbone=args.backbone)
            log_dict = {
                "epoch": ep,
                "train/loss": tr_loss,
                "val/exact_match": val["exact_match"],
                "val/loss":  val["loss"],          # ★ 추가
                "val/exact_match": val["exact_match"],

                # DV
                "val/dv/acc": val["dv"]["acc"],
                "val/dv/f1_macro": val["dv"]["f1_macro"],
                "val/dv/precision_macro": val["dv"]["precision_macro"],
                "val/dv/recall_macro": val["dv"]["recall_macro"],
                "val/dv/balanced_acc": val["dv"]["balanced_acc"],
                "val/dv/top3_acc": val["dv"]["top3_acc"],

                # OV
                "val/ov/acc": val["ov"]["acc"],
                "val/ov/f1_macro": val["ov"]["f1_macro"],
                "val/ov/precision_macro": val["ov"]["precision_macro"],
                "val/ov/recall_macro": val["ov"]["recall_macro"],
                "val/ov/balanced_acc": val["ov"]["balanced_acc"],
            }
            wandb.log(log_dict)

            # === 3) 터미널 로그 ===
            print(f"[Epoch {ep}/{args.epochs}] train_loss={tr_loss:.4f} "
              f"exact={val['exact_match']:.3f}")

            # === 4) 체크포인트 저장 (best) ===
            if val["exact_match"] > best_exact:
                best_exact = val["exact_match"]
                os.makedirs(args.save_dir, exist_ok=True)
                ckpt_path = os.path.join(args.save_dir, f"best_exact_ep{ep}_{args.backbone}_joint.pth")
                torch.save({
                    "model": model.state_dict(),
                    "epoch": ep,
                    "best_exact": best_exact
                }, ckpt_path)
                wandb.run.summary["best_exact_match"] = best_exact
                print(f"[CKPT] Saved best model to {ckpt_path}")

        # 마지막 리포트 & 플롯
        print("\n=== Classification Report (DV) ===\n"+val["dv"]["report"])
        # print("\n=== Classification Report (PLACE) ===\n"+val["place"]["report"])
        print("\n=== Classification Report (OV) ===\n"+val["ov"]["report"])
        plot_summary(val, out_path=f"results_metrics_{args.backbone}.png")
        end_time = time.time()
        total_time = end_time - start_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"✅ Training {args.backbone} completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

    elif args.mode == "validate":
        assert args.ckpt, "--ckpt is required for --mode validate"

        val_ds =SingleCsvDataset(args.val_csv, video_root=args.val_video_root,
                                 num_frames=args.num_frames, size=args.size)
        val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

        # model = TripleHeadVideoClassifier(NCLS["dashcam_vehicle_info"],
        #                                   NCLS["accident_place_feature"],
        #                                   NCLS["other_vehicle_info"], backbone=args.backbone).to(device).eval()
        model = TripleHeadVideoClassifier(NCLS["dashcam_vehicle_info"],
                                          NCLS["other_vehicle_info"], backbone=args.backbone).to(device).eval()
        sd = torch.load(args.ckpt, map_location=device)
        if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
        model.load_state_dict(sd, strict=False)

        val = evaluate(model, val_loader, device)
        print("\n=== VALIDATION ===")
        print(f"DV    acc={val['dv']['acc']:.3f}  f1_macro={val['dv']['f1_macro']:.3f}")
        # print(f"PLACE acc={val['place']['acc']:.3f}  f1_macro={val['place']['f1_macro']:.3f}")
        print(f"OV    acc={val['ov']['acc']:.3f}  f1_macro={val['ov']['f1_macro']:.3f}")
        print(f"Exact-Match={val['exact_match']:.3f}")
        plot_summary(val, out_path=f"results_metrics_validation_{args.backbone}.png")


if __name__ == "__main__":
    main()