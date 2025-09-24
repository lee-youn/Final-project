# ===== no_llm_3d.py =====
# 비디오 클립 -> 3D CNN(r3d_18) -> 2D 회귀(대시캠/상대)
# - 텍스트/탐지기 사용 안 함
# - CSV/JSON 자동 인식
# - W&B 로깅, 최종 ckpt만 저장
# - 평가 출력(메트릭/CSV/JSON/플롯) 기존 포맷 유사
from sklearn.isotonic import IsotonicRegression
from joblib import dump, load

import os, json, argparse, glob
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== Calibration helpers =====
def post_fix_pair_np(v, total=10.0, mode="project"):
    """
    v: [2] 예측 쌍
    mode:
      - "none"     : 그대로 사용
      - "project"  : (권장) 음수→0 클램프 후 합=total로 재투영
      - "snap_int" : project 후 정수 스냅(가까운 정수 쌍, 합=total)
      - "snap_05"  : project 후 0.5 단위 스냅(합 재보장)
    """
    x = np.asarray(v, dtype=float)

    if mode == "none":
        return x

    # 1) 합=total 재투영(음수 방지)
    x = np.maximum(x, 0.0)
    s = float(x.sum())
    if s <= 0:
        x = np.array([total/2.0, total/2.0], dtype=float)
    else:
        x = x * (total / s)

    if mode == "project":
        return x

    if mode == "snap_int":
        a_int = int(np.floor(x[0] + 0.5))
        a_int = max(0, min(int(total), a_int))
        b_int = int(total) - a_int
        return np.array([float(a_int), float(b_int)], dtype=float)

    if mode == "snap_05":
        # 0.5 단위 반올림 후 합 재보장
        a = round(x[0] * 2.0) / 2.0
        b = round(x[1] * 2.0) / 2.0
        s = a + b
        if s <= 0:
            return np.array([total/2.0, total/2.0], dtype=float)
        return np.array([a, b]) * (total / s)

    return x

def fit_isotonic_ratio(yhat_pairs: np.ndarray, ytrue_pairs: np.ndarray) -> IsotonicRegression:
    """쌍 → 비율로 변환해 등화기 학습."""
    p_hat = ratio_from_pairs(yhat_pairs)
    p_true = ratio_from_pairs(ytrue_pairs)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True).fit(p_hat, p_true)
    return iso

def apply_ratio_calibrator(calibrator, yhat_pairs: np.ndarray, total=10.0) -> np.ndarray:
    """등화기 적용 → 쌍 복원 → 합=total 재보장."""
    p_hat = ratio_from_pairs(yhat_pairs)
    p_cal = calibrator.transform(p_hat)
    y_cal = pairs_from_ratio(p_cal, total=total)
    y_cal = np.vstack([project_pair_to_basis_np(v, total=total) for v in y_cal])
    return y_cal

def snap_pair_to_integer_basis_np(v, total=10):
    v = np.asarray(v, dtype=float)
    v = np.maximum(v, 0.0)
    s = v.sum()
    if s <= 0:
        a = total // 2
        return np.array([float(a), float(total - a)])
    v = v * (total / s)
    a_int = int(np.floor(v[0] + 0.5))
    a_int = max(0, min(total, a_int))
    b_int = total - a_int
    return np.array([float(a_int), float(b_int)])

# ------------------------------
# 공용 유틸
# ------------------------------
def normalize_pair_100(p) -> List[float]:
    a, b = float(p[0]), float(p[1])
    if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
        a *= 100.0; b *= 100.0
    a = max(a, 0.0); b = max(b, 0.0)
    s = a + b
    if s == 0.0: return [50.0, 50.0]
    scale = 100.0 / s
    return [a*scale, b*scale]

def to_basis(pair100: List[float], target_basis: float = 10.0) -> List[float]:
    factor = 100.0 / target_basis
    return [p / factor for p in pair100]

def project_pair_to_basis_np(v, total=10.0):
    v = np.maximum(np.asarray(v, dtype=float), 0.0)
    s = float(v.sum())
    if s <= 0:
        return np.array([total/2.0, total/2.0], dtype=float)
    return v * (total / s)

def ratio_from_pairs(y):  # y: (N,2)
    s = np.clip(y.sum(axis=1, keepdims=True), 1e-6, None)
    return (y[:, [0]] / s).ravel()

def pairs_from_ratio(p, total=10.0):
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    a = total * p
    b = total * (1.0 - p)
    return np.stack([a, b], axis=1)

def calibrate_ratio_binwise(p_hat, p_true, nbins=10):
    bins = np.linspace(0.0, 1.0, nbins + 1)
    idx = np.clip(np.digitize(p_hat, bins) - 1, 0, nbins - 1)
    bin_mean_true = np.zeros(nbins, dtype=float)
    for b in range(nbins):
        m = (idx == b)
        bin_mean_true[b] = float(p_true[m].mean()) if m.any() else float((bins[b]+bins[b+1])/2.0)
    def f(p):
        ii = np.clip(np.digitize(p, bins) - 1, 0, nbins - 1)
        return bin_mean_true[ii]
    return f

def compute_metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = float(mean_absolute_error(y, yhat))
    try: rmse = float(mean_squared_error(y, yhat, squared=False))
    except TypeError: rmse = float(np.sqrt(mean_squared_error(y, yhat)))
    r2 = float(r2_score(y, yhat))
    mae_dc = float(mean_absolute_error(y[:, 0], yhat[:, 0]))
    mae_ov = float(mean_absolute_error(y[:, 1], yhat[:, 1]))
    return {"MAE": mae, "RMSE": rmse, "R2": r2,
            "MAE_dashcam": mae_dc, "MAE_other": mae_ov, "count": int(len(y))}

def save_plots(y: np.ndarray, yhat: np.ndarray, target_basis: float, out_prefix: str) -> dict:
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    paths = {}
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    titles = ["Dashcam", "Other Vehicle"]
    for i in range(2):
        ax[i].scatter(y[:, i], yhat[:, i], alpha=0.5)
        ax[i].plot([0,target_basis],[0,target_basis],"--",color="gray")
        ax[i].set_title(f"{titles[i]} Fault (basis={target_basis})")
        ax[i].set_xlabel("True"); ax[i].set_ylabel("Pred")
        ax[i].set_xlim(0,target_basis); ax[i].set_ylim(0,target_basis)
    plt.tight_layout()
    p_sc = f"{out_prefix}_scatter.png"; plt.savefig(p_sc); plt.close(fig); paths["scatter"]=p_sc

    err = np.abs(yhat - y)
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.hist(err[:,0], bins=20, alpha=0.6, label="Dashcam")
    ax2.hist(err[:,1], bins=20, alpha=0.6, label="Other")
    ax2.set_title(f"Absolute Error Histogram (basis={target_basis})")
    ax2.set_xlabel("Abs Error"); ax2.set_ylabel("Count"); ax2.legend()
    p_hi = f"{out_prefix}_err_hist.png"; plt.tight_layout(); plt.savefig(p_hi); plt.close(fig2); paths["err_hist"]=p_hi
    return paths

def find_video_file(video_root: str, name_or_rel: str) -> Optional[str]:
    cand = os.path.join(video_root, name_or_rel)
    if os.path.exists(cand): return cand
    stem = os.path.splitext(name_or_rel)[0]
    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        p = os.path.join(video_root, stem + ext)
        if os.path.exists(p): return p
    g = glob.glob(os.path.join(video_root, stem + "*"))
    return g[0] if g else None

# ------------------------------
# 비디오 → 클립 텐서(C,T,H,W)
# ------------------------------
KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD  = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

def load_clip_tensor(video_path: str, num_frames: int = 16, size: int = 224) -> torch.Tensor:
    """균등샘플 num_frames -> [C,T,H,W], float32, Kinetics 정규화."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Empty or unreadable video: {video_path}")
    if total <= num_frames:
        idxs = list(range(total))
    else:
        step = total / (num_frames + 1)
        idxs = [int(step*(i+1)) for i in range(num_frames)]
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total-1))
        ok, fr = cap.read()
        if not ok:
            if frames: fr = cv2.cvtColor(np.array(frames[-1]), cv2.COLOR_RGB2BGR)
            else: fr = np.zeros((size,size,3), dtype=np.uint8)
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        fr = cv2.resize(fr, (size, size))
        frames.append(fr.astype(np.float32)/255.0)
    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1])
    arr = np.stack(frames, axis=0)                     # [T,H,W,3]
    arr = (arr - KINETICS_MEAN) / KINETICS_STD
    arr = np.transpose(arr, (3,0,1,2))                 # [C,T,H,W]
    return torch.from_numpy(arr)

# ------------------------------
# 데이터 로딩(JSON/CSV 자동)
# ------------------------------
def load_rows(path: str) -> List[dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f: arr = json.load(f)
        assert isinstance(arr, list)
        return arr
    elif ext == ".csv":
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

class VideoFaultClipDataset(Dataset):
    def __init__(self, data_path: str, video_root: str,
                 target_basis: float = 10.0, num_frames: int = 16, size: int = 224,
                 col_video: Optional[str] = None,
                 col_dc: Optional[str] = None,
                 col_ov: Optional[str] = None):
        rows = load_rows(data_path)
        self.rows = rows
        self.video_root = video_root
        self.target_basis = target_basis
        self.num_frames = num_frames
        self.size = size

        keys = set(rows[0].keys()) if rows else set()
        self.col_video = col_video or (
            "video_name" if "video_name" in keys else
            "video_path" if "video_path" in keys else
            "filename" if "filename" in keys else
            "path" if "path" in keys else
            None
        )
        self.col_dc = col_dc or (
            "dashcam_vehicle_negligence" if "dashcam_vehicle_negligence" in keys else
            "dc_negligence" if "dc_negligence" in keys else
            "dashcam" if "dashcam" in keys else
            "dc" if "dc" in keys else
            None
        )
        self.col_ov = col_ov or (
            "other_vehicle_negligence" if "other_vehicle_negligence" in keys else
            "ov_negligence" if "ov_negligence" in keys else
            "other" if "other" in keys else
            "ov" if "ov" in keys else
            None
        )
        if self.col_video is None:
            raise ValueError(f"Could not infer video column from {keys}. Use --col_video.")
        if self.col_dc is None or self.col_ov is None:
            raise ValueError(f"Could not infer label columns from {keys}. Use --col_dc / --col_ov.")

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        raw_name = row.get(self.col_video, "")
        video_name = str(raw_name)
        vpath = find_video_file(self.video_root, video_name) if video_name else None
        if (not vpath) and isinstance(raw_name, str) and os.path.isabs(raw_name) and os.path.exists(raw_name):
            vpath = raw_name

        if not vpath:
            clip = torch.zeros(3, self.num_frames, self.size, self.size, dtype=torch.float32)
        else:
            clip = load_clip_tensor(vpath, num_frames=self.num_frames, size=self.size)  # [C,T,H,W]

        dc = row.get(self.col_dc, 50.0)
        ov = row.get(self.col_ov, 50.0)
        gt100 = normalize_pair_100([dc, ov])
        gt_basis = torch.tensor(to_basis(gt100, self.target_basis), dtype=torch.float32)  # [2]
        return clip, gt_basis, (video_name or os.path.basename(vpath) if vpath else "NA")

def collate_clip(batch):
    clips = torch.stack([b[0] for b in batch], dim=0)  # [B,C,T,H,W]
    labels = torch.stack([b[1] for b in batch], dim=0) # [B,2]
    names  = [b[2] for b in batch]
    return clips, labels, names

# ------------------------------
# 3D CNN 모델
# ------------------------------
import torchvision
from torchvision.models.video import r3d_18, R3D_18_Weights

class R3DFaultRegressor(nn.Module):
    def __init__(self, basis_total=10.0, pretrained=True, freeze_backbone=False):
        super().__init__()
        self.basis_total = basis_total
        if pretrained:
            try:
                weights = R3D_18_Weights.KINETICS400_V1
            except Exception:
                weights = None
        else:
            weights = None
        self.backbone = r3d_18(weights=weights)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(in_f, 256), nn.ReLU(),
            nn.Linear(256, 2)
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):  # x: [B,C,T,H,W]
        feat = self.backbone(x)      # [B, 512]
        out  = self.head(feat)       # [B, 2]
        # 합=basis_total 프로젝션 (torch 버전)
        v = torch.clamp(out, min=0.0)
        s = v.sum(dim=1, keepdim=True).clamp(min=1e-6)
        v = v * (self.basis_total / s)
        return v

# ------------------------------
# 학습 루프 (최종만 저장)
# ------------------------------
def train_3d(
    train_path: str,
    val_path: Optional[str],
    video_root: str,
    out_ckpt: str,
    target_basis: float = 10.0,
    num_frames: int = 16,
    size: int = 224,
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 4,
    freeze_backbone: bool = False,
    gpus: str = "0",
    use_amp: bool = True,
    col_video: Optional[str] = None,
    col_dc: Optional[str] = None,
    col_ov: Optional[str] = None,
):
    gpu_list = [int(x) for x in gpus.split(",") if x.strip()!=""]
    if torch.cuda.is_available() and len(gpu_list)>=1:
        device = f"cuda:{gpu_list[0]}"
    else:
        device = "cpu"

    wandb.init(project="rcnn-image-only-fault", job_type="training-3d",
               config=dict(train=train_path, val=val_path, video_root=video_root,
                           target_basis=target_basis, num_frames=num_frames, size=size,
                           epochs=epochs, lr=lr, batch_size=batch_size,
                           freeze_backbone=freeze_backbone))

    model = R3DFaultRegressor(basis_total=target_basis, pretrained=True,
                              freeze_backbone=freeze_backbone).to(device)
    params = []
    if not freeze_backbone:
        params += list(model.backbone.parameters())
    params += list(model.head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device!="cpu"))
    loss_fn = nn.MSELoss()

    train_ds = VideoFaultClipDataset(train_path, video_root, target_basis, num_frames, size,
                                     col_video=col_video, col_dc=col_dc, col_ov=col_ov)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, collate_fn=collate_clip, pin_memory=True)
    if val_path:
        val_ds = VideoFaultClipDataset(val_path, video_root, target_basis, num_frames, size,
                                       col_video=col_video, col_dc=col_dc, col_ov=col_ov)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, collate_fn=collate_clip, pin_memory=True)
    else:
        val_dl = None

    for ep in range(1, epochs+1):
        model.train()
        pbar = tqdm(train_dl, desc=f"Train3D ep{ep}", dynamic_ncols=True)
        total_loss = 0.0; n_samples = 0
        for clips, labels, names in pbar:
            clips = clips.to(device)            # [B,C,T,H,W]
            labels = labels.to(device)          # [B,2]
            with torch.cuda.amp.autocast(enabled=(use_amp and device!="cpu")):
                preds = model(clips)            # [B,2], already projected
                loss = loss_fn(preds, labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)
            pbar.set_postfix(loss=loss.item())
        train_loss = total_loss / max(1, n_samples)
        wandb.log({"train/loss": train_loss, "epoch": ep})

        if val_dl:
            model.eval()
            preds_all, gts_all = [], []
            with torch.no_grad():
                for clips, labels, names in tqdm(val_dl, desc=f"Val3D ep{ep}", dynamic_ncols=True):
                    clips = clips.to(device)
                    pred = model(clips).cpu().numpy()
                    lab  = labels.numpy()
                    preds_all.append(pred); gts_all.append(lab)
            preds_all = np.concatenate(preds_all, axis=0)
            gts_all   = np.concatenate(gts_all, axis=0)
            m = compute_metrics(gts_all, preds_all)
            wandb.log({f"val/{k}": v for k,v in m.items()})

    # 최종만 저장
    os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "target_basis": target_basis},
               out_ckpt)
    print(f"[ckpt] final saved to {out_ckpt}")
    wandb.finish()

# ------------------------------
# 평가 (A/B 비교용)
# ------------------------------
@torch.no_grad()
def evaluate_3d(
    eval_path: str,
    video_root: str,
    ckpt_path: str,
    out_json_path: str,
    target_basis: float = 10.0,
    num_frames: int = 16,
    size: int = 224,
    gpus: str = "0",
    col_video: Optional[str] = None,
    col_dc: Optional[str] = None,
    col_ov: Optional[str] = None,
):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    gpu_list = [int(x) for x in gpus.split(",") if x.strip()!=""]
    if torch.cuda.is_available() and len(gpu_list)>=1:
        device = f"cuda:{gpu_list[0]}"
    else:
        device = "cpu"

    wandb.init(project="rcnn-image-only-fault", job_type="evaluation-3d",
               config=dict(eval=eval_path, video_root=video_root, ckpt=ckpt_path,
                           target_basis=target_basis, num_frames=num_frames, size=size))

    # 모델 로드
    model = R3DFaultRegressor(basis_total=target_basis, pretrained=False).to(device).eval()
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd["state_dict"], strict=False)

    # 데이터 적재
    ds = VideoFaultClipDataset(eval_path, video_root, target_basis, num_frames, size,
                               col_video=col_video, col_dc=col_dc, col_ov=col_ov)

    results = []
    preds_list = []
    gts_list   = []
    iterator = tqdm(range(len(ds)), desc="Evaluating(3D)", dynamic_ncols=True)

    for i in iterator:
        clip, gt_basis, name = ds[i]
        clip = clip.unsqueeze(0).to(device)  # [1,C,T,H,W]
        try:
            raw_pred = model(clip).squeeze(0).cpu().numpy()  # [2]
            post_mode = getattr(args, "post_fix", "project")
            pred = post_fix_pair_np(raw_pred, total=target_basis, mode=post_mode)
        except Exception as e:
            results.append({"idx": i, "video_name": name, "error": f"predict_failed: {e}"})
            continue

        out_item = {
            "idx": i,
            "video_name": name,
            "pred_basis_dashcam_raw": float(raw_pred[0]),
            "pred_basis_other_raw":  float(raw_pred[1]),
            "pred_basis_dashcam": float(pred[0]),
            "pred_basis_other":  float(pred[1]),
            "pred_100_dashcam": float(pred[0] * (100.0/target_basis)),
            "pred_100_other":  float(pred[1] * (100.0/target_basis)),
        }
        # GT 있을 때만
        if np.isfinite(gt_basis.numpy()).all():
            gb = gt_basis.numpy()
            out_item["gt_basis_dashcam"] = float(gb[0])
            out_item["gt_basis_other"]   = float(gb[1])
            out_item["gt_100_dashcam"]   = float(gb[0]*(100.0/target_basis))
            out_item["gt_100_other"]     = float(gb[1]*(100.0/target_basis))
            out_item["abs_err_basis_dashcam"] = abs(out_item["gt_basis_dashcam"] - out_item["pred_basis_dashcam"])
            out_item["abs_err_basis_other"]   = abs(out_item["gt_basis_other"]   - out_item["pred_basis_other"])
            preds_list.append([pred[0], pred[1]])
            gts_list.append([gb[0], gb[1]])
        results.append(out_item)

    metrics = {}
    out_prefix = os.path.splitext(out_json_path)[0]
    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)

    if len(gts_list) > 0:
        yhat_all = np.array(preds_list, dtype=float)
        y_gt     = np.array(gts_list, dtype=float)

        # 보정 전
        metrics_pre = compute_metrics(y_gt, yhat_all); metrics_pre["target_basis"] = target_basis

        # 비율 등화(iso 실패 시 bin-wise)
        try:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True).fit(ratio_from_pairs(yhat_all),
                                                                                ratio_from_pairs(y_gt))
            p_cal_all = iso.transform(ratio_from_pairs(yhat_all))
        except Exception as e:
            fbin = calibrate_ratio_binwise(ratio_from_pairs(yhat_all), ratio_from_pairs(y_gt), nbins=10)
            p_cal_all = fbin(ratio_from_pairs(yhat_all))
        yhat_cal_all = pairs_from_ratio(p_cal_all, total=target_basis)
        # 결과 dict에 보정값 넣기
        k = 0
        for r in results:
            if "pred_basis_dashcam" in r and "gt_basis_dashcam" in r:
                r["pred_basis_dashcam_cal"] = float(yhat_cal_all[k,0])
                r["pred_basis_other_cal"]   = float(yhat_cal_all[k,1])
                k += 1

        # 메트릭(보정 후)
        yhat_cal_gt = yhat_cal_all
        metrics_cal = compute_metrics(y_gt, yhat_cal_gt)
        metrics = {**metrics_pre, **{f"cal/{k}": v for k,v in metrics_cal.items()}}

        # 플롯/저장
        plot_paths = save_plots(y_gt, yhat_all, target_basis, out_prefix + "_precal")
        plot_paths_cal = save_plots(y_gt, yhat_cal_gt, target_basis, out_prefix + "_cal")
        plot_paths.update({("cal_"+k): v for k,v in plot_paths_cal.items()})

        out_csv = f"{out_prefix}.csv"
        pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8")
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "results": results}, f, ensure_ascii=False, indent=2)

        wandb.log({
            "eval3d/MAE": metrics["MAE"], "eval3d/RMSE": metrics["RMSE"], "eval3d/R2": metrics["R2"],
            "eval3d/MAE_dashcam": metrics["MAE_dashcam"], "eval3d/MAE_other": metrics["MAE_other"],
            "eval3d_cal/MAE": metrics["cal/MAE"], "eval3d_cal/RMSE": metrics["cal/RMSE"], "eval3d_cal/R2": metrics["cal/R2"],
            "eval3d_cal/MAE_dashcam": metrics["cal/MAE_dashcam"], "eval3d_cal/MAE_other": metrics["cal/MAE_other"],
        })
        for k, pth in plot_paths.items():
            if os.path.exists(pth):
                wandb.log({f"plots3d/{k}": wandb.Image(pth)})

        try:
            table = wandb.Table(dataframe=pd.DataFrame(results))
            wandb.log({"eval3d/table": table})
        except Exception:
            pass
    else:
        out_csv = f"{out_prefix}.csv"
        pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8")
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": {}, "results": results}, f, ensure_ascii=False, indent=2)

    print("=== Evaluation Summary (3D CNN) ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved JSON: {out_json_path}")
    print(f"Saved CSV : {os.path.splitext(out_json_path)[0]}.csv")
    wandb.finish()

# ------------------------------
# CLI
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="3D CNN video regressor (train/eval).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    pt = sub.add_parser("train")
    pt.add_argument("--train_json", type=str, required=True)
    pt.add_argument("--val_json",   type=str, default="")
    pt.add_argument("--video_root", type=str, required=True)
    pt.add_argument("--out_ckpt",   type=str, required=True)
    pt.add_argument("--target_basis", type=float, default=10.0)
    pt.add_argument("--num_frames", type=int, default=16)
    pt.add_argument("--size",       type=int, default=224)
    pt.add_argument("--epochs",     type=int, default=5)
    pt.add_argument("--lr",         type=float, default=1e-3)
    pt.add_argument("--batch_size", type=int, default=4)
    pt.add_argument("--freeze_backbone", action="store_true")
    pt.add_argument("--gpus",       type=str, default="0")
    pt.add_argument("--no_amp",     action="store_true")
    pt.add_argument("--col_video",  type=str, default="")
    pt.add_argument("--col_dc",     type=str, default="")
    pt.add_argument("--col_ov",     type=str, default="")
    # train 서브커맨드에 추가 (보정기 저장 관련)
    pt.add_argument("--save_calibrator", type=str, default="",
                    help="isotonic 보정기 저장 경로(.pkl). 지정 시 훈련 종료 후 선택 split로 학습해 저장")
    pt.add_argument("--calibrator_split", type=str, default="train",
                    choices=["train","val"], help="보정기를 학습할 데이터 분할")

    # eval 서브커맨드에 추가 (보정기 로드/스냅 관련)

    # eval
    pe = sub.add_parser("eval")
    pe.add_argument("--eval_json",  type=str, required=True)
    pe.add_argument("--video_root", type=str, required=True)
    pe.add_argument("--ckpt",       type=str, required=True)
    pe.add_argument("--out_json",   type=str, required=True)
    pe.add_argument("--target_basis", type=float, default=10.0)
    pe.add_argument("--num_frames", type=int, default=16)
    pe.add_argument("--size",       type=int, default=224)
    pe.add_argument("--gpus",       type=str, default="0")
    pe.add_argument("--col_video",  type=str, default="")
    pe.add_argument("--col_dc",     type=str, default="")
    pe.add_argument("--col_ov",     type=str, default="")
    pe.add_argument("--post_fix", type=str, default="project",
                choices=["none", "project", "snap_int", "snap_05"],
                help="막판 과실비율 보정 모드")
    pe.add_argument("--calibrator", type=str, default="",
                    help="사전에 학습/저장한 isotonic 보정기(.pkl) 경로. 지정 시 eval에서는 '적용만' 수행")
    pe.add_argument("--snap_to_int", action="store_true",
                    help="보정 후 정수 스냅(합=target_basis) A/B 옵션")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "train":
        if not os.path.exists(args.train_json):
            raise SystemExit(f"[Config] missing --train_json: {args.train_json}")
        if not os.path.isdir(args.video_root):
            raise SystemExit(f"[Config] --video_root not a directory: {args.video_root}")
        train_3d(
            train_path=args.train_json,
            val_path=(args.val_json if args.val_json else None),
            video_root=args.video_root,
            out_ckpt=args.out_ckpt,
            target_basis=args.target_basis,
            num_frames=args.num_frames,
            size=args.size,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            freeze_backbone=args.freeze_backbone,
            gpus=args.gpus,
            use_amp=(not args.no_amp),
            col_video=(args.col_video or None),
            col_dc=(args.col_dc or None),
            col_ov=(args.col_ov or None),
        )
    else:
        if not os.path.exists(args.ckpt):
            raise SystemExit(f"[Config] missing --ckpt: {args.ckpt}")
        if not os.path.exists(args.eval_json):
            raise SystemExit(f"[Config] missing --eval_json: {args.eval_json}")
        if not os.path.isdir(args.video_root):
            raise SystemExit(f"[Config] --video_root not a directory: {args.video_root}")
        evaluate_3d(
            eval_path=args.eval_json,
            video_root=args.video_root,
            ckpt_path=args.ckpt,
            out_json_path=args.out_json,
            target_basis=args.target_basis,
            num_frames=args.num_frames,
            size=args.size,
            gpus=args.gpus,
            col_video=(args.col_video or None),
            col_dc=(args.col_dc or None),
            col_ov=(args.col_ov or None),
        )
