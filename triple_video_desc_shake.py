# train_video_desc.py
# pip install torch torchvision opencv-python numpy pandas
import os, csv, json, math, argparse, random, re
from typing import Optional, List, Dict, Tuple
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

torch.backends.cudnn.benchmark = True   # conv autotune
cv2.setNumThreads(0)                     # 워커 n개 * OpenCV n스레드 폭주 방지

# ---- 파일 상단에 붙여서 동일 파라미터로 사용 ----
SHAKE_MAX_CORNERS   = 800
SHAKE_QUALITY       = 0.01
SHAKE_MIN_DISTANCE  = 8
SHAKE_BLOCK_SIZE    = 7
SHAKE_MAD_K         = 3.2
SHAKE_MIN_MAG_PX    = 1.8
SHAKE_STRICT_RATIO  = 1.5  # 예전과 동일

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

# --------------------------
# 카메라 셰이크 중심 시점 + 인덱스 샘플링
# --------------------------
def _video_meta(path: str) -> Tuple[float, int]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.release()
    return float(fps), int(total)

def _estimate_motion(prev_gray, gray):
    pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=SHAKE_MAX_CORNERS,
        qualityLevel=SHAKE_QUALITY,
        minDistance=SHAKE_MIN_DISTANCE,
        blockSize=SHAKE_BLOCK_SIZE,
        useHarrisDetector=False
    )
    if pts is None or len(pts) < 8:
        return 0.0, 0.0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
    if nxt is None or st is None:
        return 0.0, 0.0
    good_prev = pts[st[:,0]==1]; good_next = nxt[st[:,0]==1]
    if len(good_prev) < 6:
        return 0.0, 0.0
    M, _ = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        flow = good_next - good_prev
        dx = float(np.median(flow[:,0,0])); dy = float(np.median(flow[:,0,1]))
        return dx, dy
    return float(M[0,2]), float(M[1,2])

def detect_camera_shake_center(video_path: str) -> Optional[float]:
    """예전 make_tracks_ego.py 방식과 동일한 로직로 피크 중심 시각을 반환."""
    cap = cv2.VideoCapture(video_path)
    ok, prev = cap.read()
    if not ok:
        cap.release()
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    mags, times = [], []
    idx = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dx, dy = _estimate_motion(prev_gray, g)
        mag = float(np.hypot(dx, dy))
        mags.append(mag)
        times.append(float(idx / fps))
        prev_gray = g
        idx += 1
    cap.release()
    if not mags:
        return None

    mags = np.asarray(mags, dtype=np.float32)
    med = float(np.median(mags))
    mad = float(np.median(np.abs(mags - med)) + 1e-6)
    thr = max(med + SHAKE_MAD_K * mad, SHAKE_MIN_MAG_PX)

    # 예전처럼 '충분히 큰' 로컬 피크만 후보로
    peaks = []
    for i in range(1, len(mags)-1):
        if (mags[i] > thr and mags[i] >= mags[i-1] and mags[i] >= mags[i+1]
            and mags[i] >= thr * SHAKE_STRICT_RATIO):
            peaks.append(i)

    if peaks:
        # 로컬 피크 중 최댓값 시각
        imax = peaks[int(np.argmax(mags[peaks]))]
    else:
        # 로컬 피크가 없으면 전역 최대
        imax = int(np.argmax(mags))

    return float(times[imax])

def _indices_around_center(center_t: float, fps: float, total: int,
                           num_frames: int=16, window_sec: float=0.5) -> List[int]:
    import math
    center_f = int(round(center_t * fps))
    half = int(round(window_sec * fps))
    s = max(0, center_f - half)
    e = min(total - 1, center_f + half)
    if e <= s:
        s = max(0, center_f - num_frames//2)
        e = min(total - 1, s + num_frames - 1)

    # 1차: 균등 샘플
    idxs = np.linspace(s, e, num_frames).round().astype(int)
    idxs = np.clip(idxs, 0, total - 1)
    uniq = np.unique(idxs)

    if uniq.size < num_frames:
        span = e - s + 1
        if span < num_frames:
            # 창이 너무 좁으면 전체 구간 균등
            idxs = np.linspace(0, total - 1, num_frames).round().astype(int)
        else:
            # stride 기반 채우기
            step = max(1, math.floor(span / num_frames))
            arr = np.arange(s, min(e + 1, s + step * num_frames), step, dtype=int)
            if arr.size >= num_frames:
                idxs = arr[:num_frames]
            else:
                extra = num_frames - arr.size
                tail = np.minimum(e, arr[-1] + np.arange(1, extra + 1, dtype=int))
                idxs = np.concatenate([arr, tail])

    idxs = np.clip(idxs, 0, total - 1)
    if idxs.size != num_frames:
        pad = np.full((num_frames - idxs.size,), idxs[-1], dtype=int)
        idxs = np.concatenate([idxs, pad])
    return idxs.tolist()

def load_video_tensor(path, num_frames=16, size=224, indices=None):
    import torchvision.transforms.functional as VF
    import torch, cv2, numpy as np

    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    if indices is None:
        idxs = np.linspace(0, total - 1, num_frames).astype(int)
    else:
        idxs = np.clip(np.array(indices, dtype=int), 0, total - 1)

    # ---- 한 번만 seek, 이후 순차 grab/retrieve ----
    start = int(idxs[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames = []
    want = set(int(i) for i in idxs.tolist())
    cur = start - 1
    last = None

    while len(frames) < len(idxs):
        ok = cap.grab()
        if not ok:
            break
        cur += 1
        if cur in want:
            ok, frame = cap.retrieve()
            if not ok:
                # 실패 시 마지막 성공 프레임 복제
                if last is not None:
                    frames.append(last.clone())
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (size, size))
            ten = torch.from_numpy(frame).permute(2,0,1).contiguous().float()/255.0
            frames.append(ten)
            last = ten

    cap.release()

    # ---- 길이 보장: 모자라면 마지막 프레임 복제해서 num_frames 맞춤 ----
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    while len(frames) < len(idxs):
        frames.append(frames[-1].clone())

    vid = torch.stack(frames, 0)  # (T,C,H,W)
    vid = VF.normalize(vid, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    return vid.contiguous()

# --------------------------
# (옵션) YOLO 검출 품질 확인
# --------------------------
_YOLO_SINGLETON = {"model": None}
def _ensure_yolo(model_path: str):
    if _YOLO_SINGLETON["model"] is None:
        from ultralytics import YOLO
        _YOLO_SINGLETON["model"] = YOLO(model_path)
    return _YOLO_SINGLETON["model"]

def yolo_count_on_tensor_TCHW(vid_TCHW: torch.Tensor, min_conf=0.25, model_path="yolov8n.pt") -> Tuple[int, float]:
    model = _ensure_yolo(model_path)
    T = vid_TCHW.shape[0]
    total, confs = 0, []
    for t in range(T):
        img = (vid_TCHW[t].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)  # HWC
        res = model.predict(img, verbose=False)
        if not res: continue
        r = res[0]
        if r.boxes is None or r.boxes.conf is None: continue
        conf = r.boxes.conf.cpu().numpy()
        sel = conf >= float(min_conf)
        total += int(sel.sum())
        if sel.any(): confs.append(float(conf[sel].mean()))
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return total, avg_conf

def auto_window_sec(num_frames, fps, min_margin=0.15, max_cap=0.5, safety=1.2):
    base = (num_frames - 1) / (2.0 * max(1.0, fps))  # 최소 반쪽 윈도
    return float(min(max(base * safety, min_margin), max_cap))
# --------------------------
# 데이터셋
# --------------------------
class SingleCsvDataset(Dataset):
    """
    하나의 표 파일(CSV/JSON/JSONL)에서 비디오 경로와 라벨을 읽음.
    - 첫 컬럼: video_path 또는 video_name(stem)
    - 라벨 컬럼: dashcam_vehicle_info, other_vehicle_info
    - 셰이크 중심 시점에서 ±shake_window_sec 사이 16장을 균등 추출
    - (옵션) YOLO 검출로 품질 확인/폴백
    """
    def __init__(self, table_path: str, video_root: str,
                 num_frames=16, size=224,
                 shake_window_sec: float=0.5, center_policy: str="shake_then_middle",
                 use_yolo: bool=False, yolo_model: str="yolov8s.pt", yolo_min_conf: float=0.25):
        self.rows = _read_table_any(table_path)
        if not self.rows:
            raise RuntimeError(f"No rows loaded from {table_path}")
        self.video_root = video_root
        self.num_frames = int(num_frames)
        self.size = int(size)
        self.shake_window_sec = float(shake_window_sec)
        self.center_policy = center_policy
        self.use_yolo = bool(use_yolo)
        self.yolo_model = yolo_model
        self.yolo_min_conf = float(yolo_min_conf)
        self.first_col = list(self.rows[0].keys())[0]
        need = {"dashcam_vehicle_info","other_vehicle_info"}
        missing = need - set(self.rows[0].keys())
        if missing:
            raise KeyError(f"Missing {missing} in {table_path}")
        self._center_cache: Dict[str, Tuple[float,int,float]] = {}  # vpath -> (fps,total,center_t)

    def __len__(self): 
        return len(self.rows)

    def _resolve_video(self, row: dict) -> str:
        VIDEO_EXTS = (".mp4",".mov",".avi",".mkv",".webm")
        val = str(row[self.first_col]).strip()
        if any(val.lower().endswith(ext) for ext in VIDEO_EXTS):
            return val if os.path.isabs(val) else os.path.join(self.video_root, val)
        stem = os.path.splitext(val)[0]
        return _resolve_by_stem(stem, self.video_root)
    
    def _center_cache_path(self, vpath: str):
        d = os.path.join(os.path.dirname(vpath), ".shake_cache")
        os.makedirs(d, exist_ok=True)
        base = os.path.splitext(os.path.basename(vpath))[0]
        return os.path.join(d, base + ".center.json")

    def _load_center_from_disk(self, vpath: str):
        p = self._center_cache_path(vpath)
        if os.path.exists(p):
            try:
                return json.load(open(p, "r")).get("center_t", None)
            except:
                return None
        return None

    def _save_center_to_disk(self, vpath: str, center_t: float):
        p = self._center_cache_path(vpath)
        try:
            json.dump({"center_t": float(center_t)}, open(p, "w"))
        except:
            pass
    

    def _center_time(self, vpath: str) -> tuple[float, int, float, bool]:
        """returns (fps, total_frames, center_t, has_shake)"""
        fps, total = _video_meta(vpath)
        t_shake = self._load_center_from_disk(vpath)
        if t_shake is None:
            t_shake = detect_camera_shake_center(vpath)
            if t_shake is not None:
                self._save_center_to_disk(vpath, t_shake)
        if t_shake is not None:
            return fps, total, float(t_shake), True
        t_mid = (total / fps) * 0.5
        return fps, total, float(t_mid), False

    def __getitem__(self, i: int):
        r = self.rows[i]
        try:
            vpath = self._resolve_video(r)
        except FileNotFoundError:
            return None

        fps, total, center_t, has_shake = self._center_time(vpath)
        indices = None
        if self.center_policy == "uniform":
            indices = None  # 전체 균등
        elif self.center_policy == "shake_only":
            if has_shake:
                win = auto_window_sec(self.num_frames, fps, min_margin=0.15,
                                    max_cap=self.shake_window_sec, safety=1.2)
                indices = _indices_around_center(center_t, fps, total,
                                                num_frames=self.num_frames, window_sec=win)
            else:
                indices = None  # 셰이크 없으면 전체 균등(요구사항)
        else:  # "shake_then_middle" (default)
            if has_shake:
                win = auto_window_sec(self.num_frames, fps, min_margin=0.15,
                                    max_cap=self.shake_window_sec, safety=1.2)
                indices = _indices_around_center(center_t, fps, total,
                                                num_frames=self.num_frames, window_sec=win)
            else:
                indices = None  # 전체 균등

        vid = load_video_tensor(vpath, self.num_frames, self.size, indices=indices)  # (T,C,H,W)

        if self.use_yolo:
            try:
                n, avgc = yolo_count_on_tensor_TCHW(vid, self.yolo_min_conf, self.yolo_model)
                # 품질이 너무 낮으면 uniform 폴백(예시 기준)
                if n == 0 or avgc < (self.yolo_min_conf + 0.05):
                    vid = load_video_tensor(vpath, self.num_frames, self.size, indices=None)
            except Exception:
                pass

        vid = vid.permute(1,0,2,3).contiguous()

        dv = _maybe_to_idx(r["dashcam_vehicle_info"], L2I["dashcam_vehicle_info"], NCLS["dashcam_vehicle_info"])
        ov = _maybe_to_idx(r["other_vehicle_info"],   L2I["other_vehicle_info"],   NCLS["other_vehicle_info"])
        return vid, (dv, ov)

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# --------------------------
# 모델
# --------------------------
class DualHeadVideoClassifier(nn.Module):
    def __init__(self, n_dv: int, n_ov: int, backbone="r3d18"):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "r3d18":
            self.backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
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
            raise ValueError(f"Unknown backbone: {backbone}")

        self.shared_proj = nn.Sequential(
            nn.LayerNorm(feat),
            nn.Linear(feat, feat),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.dv = nn.Linear(feat, n_dv)
        self.ov = nn.Linear(feat, n_ov)

    def forward(self, x):
        # x: (B,C,T,H,W)
        if self.backbone_name == "r3d18":
            z = self.backbone(x)  # (B, feat)
        elif self.backbone_name == "timesformer":
            x = x.permute(0,2,1,3,4)  # (B,T,C,H,W)
            out = self.backbone(x)
            # ✅ 분류용 CLS 토큰
            z = out.last_hidden_state[:, 0]  # (B, feat)

        elif self.backbone_name == "videomae":
            x = x.permute(0,2,1,3,4)  # (B,T,C,H,W)
            out = self.backbone(x)
            # 모델/체크포인트에 따라 pooler_output이 있을 수 있음
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                z = out.pooler_output                        # (B, feat)
            else:
                z = out.last_hidden_state[:, 0]   # (B, feat)  
        h = self.shared_proj(z)
        return self.dv(h), self.ov(h)

# --------------------------
# 학습/평가
# --------------------------
def _topk_acc(y_true, logits, k=3):
    probs = logits.softmax(dim=-1).detach().cpu().numpy()
    y = np.array(y_true)
    try:
        return float(top_k_accuracy_score(y, probs, k=k, labels=list(range(probs.shape[1]))))
    except Exception:
        return 0.0

def train_one_epoch(model, loader, opt, device, fp16=False, epoch=1, epochs=1, log_interval=10):
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
    return (running / max(1, n))

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
        loss = ce(ld, yd) + ce(lv, yv)
        bs = x.size(0)
        running_loss += loss.item() * bs
        n += bs

        logits_dv.append(ld.cpu()); logits_ov.append(lv.cpu())
        p_dv += ld.argmax(-1).cpu().tolist()
        p_ov += lv.argmax(-1).cpu().tolist()
        y_dv += yd.cpu().tolist(); y_ov += yv.cpu().tolist()

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
    exact = float(np.mean((np.array(y_dv)==np.array(p_dv)) & (np.array(y_ov)==np.array(p_ov))))
    val_loss = running_loss / max(1, n)

    if save_confmat:
        plot_confmat(y_dv, p_dv, LABELS["dashcam_vehicle_info"], "DV Confusion Matrix", f"cm_dv_{backbone}_shake_change_setting.png","val/dv/confmat")
        plot_confmat(y_ov, p_ov, LABELS["other_vehicle_info"], "OV Confusion Matrix", f"cm_ov_{backbone}_shake_change_setting.png","val/ov/confmat")

    return {"loss": val_loss, "dv": m_dv, "ov": m_ov, "exact_match": exact}

def plot_confmat(y_true, y_pred, labels, title, out_path, wandb_log_key=None):
    from sklearn.metrics import ConfusionMatrixDisplay
    import numpy as np
    import matplotlib.pyplot as plt
    import os

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
    if wandb_log_key is not None and os.path.exists(out_path):
        try:
            wandb.log({wandb_log_key: wandb.Image(out_path)})
        except Exception as e:
            print(f"[WARN] wandb image log failed: {e}")
    return out_path

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

def split_dataset(ds: Dataset, val_ratio=0.2, seed=42):
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    n_val = int(len(idxs)*val_ratio)
    val_idx, tr_idx = idxs[:n_val], idxs[n_val:]
    return torch.utils.data.Subset(ds, tr_idx), torch.utils.data.Subset(ds, val_idx)

# --------------------------
# 메인
# --------------------------
def main():
    ap = argparse.ArgumentParser(conflict_handler="resolve")
    ap.add_argument("--backbone", type=str, default="r3d18",
                    choices=["r3d18", "timesformer", "videomae"])
    ap.add_argument("--mode", choices=["train","validate","predict"], default="train")

    ap.add_argument("--train_csv", type=str, required=False,
                    default="/app/data/raw/json/video-train/video_accident_caption_results_unsignalized_0811.csv")
    ap.add_argument("--val_csv", type=str, required=False,
                    default="/app/data/raw/json/video-evaluate/video_accident_caption_results_unsignalized_validation_0901.csv")
    ap.add_argument("--train_video_root", type=str,
                    default=os.environ.get("TRAIN_VIDEO_ROOT", "/app/data/raw/videos/training_reencoded"))
    ap.add_argument("--val_video_root", type=str,
                    default=os.environ.get("VAL_VIDEO_ROOT", "/app/data/raw/videos/validation_reencoded"))

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--val_batch_size", type=int, default=8)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dataparallel", action="store_true")
    ap.add_argument("--save_dir", type=str, default="./checkpoints")
    ap.add_argument("--out_ckpt", type=str, default=f"./checkpoints/desc.pth")

    # 셰이크/YOLO 옵션
    ap.add_argument("--shake_window_sec", type=float, default=0.5,
                    help="center_t ± window_sec 범위에서 num_frames 균등 추출")
    ap.add_argument("--center_policy", type=str, default="shake_then_middle",
                    choices=["shake_only","shake_then_middle","uniform"])
    ap.add_argument("--use_yolo", action="store_true",
                    help="추출된 프레임(16장)에 YOLO 검출 수행")
    ap.add_argument("--yolo_model", type=str, default="yolov8s.pt")
    ap.add_argument("--yolo_min_conf", type=float, default=0.25)

    # predict 모드
    ap.add_argument("--video", type=str, help="single video for prediction")
    ap.add_argument("--ckpt", type=str, help="checkpoint path (validate/predict)")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    wandb.init(project="video-desc-shake", config=vars(args))

    if args.mode == "train":
        train_ds = SingleCsvDataset(
            args.train_csv, video_root=args.train_video_root,
            num_frames=args.num_frames, size=args.size,
            shake_window_sec=args.shake_window_sec, center_policy=args.center_policy,
            use_yolo=args.use_yolo, yolo_model=args.yolo_model, yolo_min_conf=args.yolo_min_conf
        )
        val_ds = SingleCsvDataset(
            args.val_csv, video_root=args.val_video_root,
            num_frames=args.num_frames, size=args.size,
            shake_window_sec=args.shake_window_sec, center_policy=args.center_policy,
            use_yolo=args.use_yolo, yolo_model=args.yolo_model, yolo_min_conf=args.yolo_min_conf
        )

        # train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
        #                           num_workers=args.num_workers, pin_memory=True, drop_last=True,
        #                           collate_fn=collate_skip_none)
        # val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False,
        #                         num_workers=args.num_workers, pin_memory=True,
        #                         collate_fn=collate_skip_none)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, prefetch_factor=4 if args.num_workers>0 else None,
            persistent_workers=True if args.num_workers>0 else False,
            pin_memory=True, drop_last=True, collate_fn=collate_skip_none
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.val_batch_size, shuffle=False,
            num_workers=args.num_workers, prefetch_factor=4 if args.num_workers>0 else None,
            persistent_workers=True if args.num_workers>0 else False,
            pin_memory=True, collate_fn=collate_skip_none
        )

        model = DualHeadVideoClassifier(NCLS["dashcam_vehicle_info"], NCLS["other_vehicle_info"],
                                        backbone=args.backbone).to(device)
        if torch.cuda.device_count() > 1 and args.dataparallel:
            model = nn.DataParallel(model)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

        best_exact = 0.0
        start_time = time.time()

        for ep in range(1, args.epochs+1):
            tr_loss = train_one_epoch(model, train_loader, opt, device,
                                      fp16=args.fp16, epoch=ep, epochs=args.epochs,
                                      log_interval=1)
            val = evaluate(model, val_loader, device, epoch=ep, epochs=args.epochs, backbone=args.backbone)
            log_dict = {
                "epoch": ep,
                "train/loss": tr_loss,
                "val/loss":  val["loss"],
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

            print(f"[Epoch {ep}/{args.epochs}] train_loss={tr_loss:.4f} exact={val['exact_match']:.3f}")

            if val["exact_match"] > best_exact:
                best_exact = val["exact_match"]
                os.makedirs(args.save_dir, exist_ok=True)
                ckpt_path = os.path.join(args.save_dir, f"best_exact_ep{ep}_{args.backbone}_shake16_change_setting.pth")
                state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save({"model": state, "epoch": ep, "best_exact": best_exact}, ckpt_path)
                wandb.run.summary["best_exact_match"] = best_exact
                print(f"[CKPT] Saved best model to {ckpt_path}")

        # 마지막 결과 출력/플롯
        print("\n=== Classification Report (DV) ===\n"+val["dv"]["report"])
        print("\n=== Classification Report (OV) ===\n"+val["ov"]["report"])
        plot_summary(val, out_path=f"results_metrics_{args.backbone}_shake_change_setting.png")
        end_time = time.time()
        total_time = end_time - start_time
        h, rem = divmod(total_time, 3600); m, s = divmod(rem, 60)
        print(f"✅ Training {args.backbone} completed in {int(h)}h {int(m)}m {int(s)}s")

    elif args.mode == "validate":
        assert args.ckpt, "--ckpt is required for --mode validate"
        val_ds = SingleCsvDataset(
            args.val_csv, video_root=args.val_video_root,
            num_frames=args.num_frames, size=args.size,
            shake_window_sec=args.shake_window_sec, center_policy=args.center_policy,
            use_yolo=args.use_yolo, yolo_model=args.yolo_model, yolo_min_conf=args.yolo_min_conf
        )
        val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True,
                                collate_fn=collate_skip_none)

        model = DualHeadVideoClassifier(NCLS["dashcam_vehicle_info"], NCLS["other_vehicle_info"],
                                        backbone=args.backbone).to(device).eval()
        sd = torch.load(args.ckpt, map_location=device)
        if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
        model.load_state_dict(sd, strict=False)

        val = evaluate(model, val_loader, device, backbone=args.backbone)
        print("\n=== VALIDATION ===")
        print(f"DV    acc={val['dv']['acc']:.3f}  f1_macro={val['dv']['f1_macro']:.3f}")
        print(f"OV    acc={val['ov']['acc']:.3f}  f1_macro={val['ov']['f1_macro']:.3f}")
        print(f"Exact-Match={val['exact_match']:.3f}")
        plot_summary(val, out_path=f"results_metrics_validation_{args.backbone}_shake_change_setting.png")

    elif args.mode == "predict":
        assert args.video and args.ckpt, "--video, --ckpt 모두 필요"
        # 단일 비디오에서 셰이크 중심 16장 뽑아 추론
        fps, total = _video_meta(args.video)
        t_mid = (total / fps) * 0.5
        t_shake = detect_camera_shake_center(args.video)
        center_t = t_shake if (t_shake is not None) else t_mid
        indices = _indices_around_center(center_t, fps, total, num_frames=args.num_frames, window_sec=args.shake_window_sec)
        vid = load_video_tensor(args.video, args.num_frames, args.size, indices=indices).permute(1,0,2,3).unsqueeze(0)  # (1,C,T,H,W)

        model = DualHeadVideoClassifier(NCLS["dashcam_vehicle_info"], NCLS["other_vehicle_info"],
                                        backbone=args.backbone).to(device).eval()
        sd = torch.load(args.ckpt, map_location=device)
        if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
        model.load_state_dict(sd, strict=False)

        with torch.no_grad():
            ld, lv = model(vid.to(device))
            pdv = ld.softmax(-1).argmax(-1).item()
            pov = lv.softmax(-1).argmax(-1).item()
        print(f"[Predict] center_t={center_t:.2f}s  DV={pdv}:{LABELS['dashcam_vehicle_info'][pdv]}  OV={pov}:{LABELS['other_vehicle_info'][pov]}")

if __name__ == "__main__":
    main()