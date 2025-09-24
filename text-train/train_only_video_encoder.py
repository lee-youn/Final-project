# -*- coding: utf-8 -*-
"""
fault_pipeline_direct.py

비디오 → (텍스트 없이) → 과실비율(대시캠/상대) 직접 예측 파이프라인
- 백본: HuggingFace VideoMAE (기본: MCG-NJU/videomae-base)
- 출력: 두 값의 합이 항상 target_basis(기본 10)가 되도록 softmax 스케일
- 모드:
  * train  : JSON(gt)로 회귀 학습 (L1 Loss)
  * eval   : JSON(gt)로 평가 + CSV/그림/JSON 저장
  * single : 단일 비디오 파일 빠른 예측
- GPU: --gpus "1,2" 로 두 장(DataParallel) 사용
- W&B: --wandb_* 플래그로 완전 제어

필요 패키지:
  pip install "transformers>=4.41" torch opencv-python pillow numpy pandas scikit-learn matplotlib wandb

JSON 형식(리스트):
[
  {
    "video_name": "VID_0001.mp4",   # 또는 "video_path": "/abs/path/.."
    "dashcam_vehicle_negligence": 40,   # 0~100 또는 0~1 가능
    "other_vehicle_negligence":   60
  },
  ...
]
"""

import os, re, json, glob, argparse
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import wandb

# ------------------------------
# (0) W&B 유틸
# ------------------------------
def wandb_parse_tags(s: str) -> Optional[list]:
    if not s: return None
    tags = [t.strip() for t in s.split(",") if t.strip()]
    return tags or None

def wandb_init(project: str, job_type: str, entity: Optional[str], run_name: Optional[str],
               tags: Optional[list], group: Optional[str], mode: str, config: dict):
    # mode: "online" | "offline" | "disabled"
    os.environ["WANDB_MODE"] = mode
    return wandb.init(
        project=project, job_type=job_type, entity=entity,
        name=run_name, tags=tags, group=group, config=config
    )

# ------------------------------
# (A) 유틸: 프레임 샘플링
# ------------------------------
def sample_frames_pil(video_path: str, num_frames: int = 16, size: int = 224) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Empty or unreadable video: {video_path}")

    if total <= num_frames:
        idxs = list(range(total))
    else:
        step = total / (num_frames + 1)
        idxs = [int(step * (i + 1)) for i in range(num_frames)]

    frames: List[Image.Image] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total - 1))
        ok, fr = cap.read()
        if not ok:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(Image.new("RGB", (size, size), color=(0, 0, 0)))
            continue
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        fr = cv2.resize(fr, (size, size))
        frames.append(Image.fromarray(fr))
    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames[:num_frames]

# ------------------------------
# (B) 라벨/경로 유틸
# ------------------------------
def normalize_pair_100(p) -> List[float]:
    a, b = float(p[0]), float(p[1])
    s = a + b
    if 0 <= a <= 1 and 0 <= b <= 1:
        a *= 100.0; b *= 100.0; s = a + b
    if s > 0 and abs(s - 100.0) > 1e-6:
        a *= (100.0 / s); b *= (100.0 / s)
    return [max(0.0, min(100.0, a)), max(0.0, min(100.0, b))]

def to_basis(pair100: List[float], target_basis: float = 10.0) -> List[float]:
    factor = 100.0 / target_basis
    return [p / factor for p in pair100]

def find_video_file(video_root: str, name_or_rel: str) -> Optional[str]:
    cand = os.path.join(video_root, name_or_rel)
    if os.path.exists(cand):
        return cand
    stem = os.path.splitext(name_or_rel)[0]
    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        p = os.path.join(video_root, stem + ext)
        if os.path.exists(p):
            return p
    g = glob.glob(os.path.join(video_root, stem + "*"))
    return g[0] if g else None

# ------------------------------
# (C) 메트릭/그림/CSV
# ------------------------------
def compute_metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y, yhat)
    try:
        rmse = mean_squared_error(y, yhat, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y, yhat))
    r2 = r2_score(y, yhat)
    mae_dc = (np.abs(y[:, 0] - yhat[:, 0])).mean()
    mae_ov = (np.abs(y[:, 1] - yhat[:, 1])).mean()
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2),
            "MAE_dashcam": float(mae_dc), "MAE_other": float(mae_ov), "count": int(len(y))}

def save_plots(y: np.ndarray, yhat: np.ndarray, target_basis: float, out_prefix: str) -> dict:
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    paths = {}
    # 스캐터
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    titles = ["Dashcam", "Other Vehicle"]
    for i in range(2):
        ax[i].scatter(y[:, i], yhat[:, i], alpha=0.5)
        ax[i].plot([0, target_basis], [0, target_basis], "--", color="gray")
        ax[i].set_title(f"{titles[i]} Fault (basis={target_basis})")
        ax[i].set_xlabel("True"); ax[i].set_ylabel("Pred")
        ax[i].set_xlim(0, target_basis); ax[i].set_ylim(0, target_basis)
    plt.tight_layout()
    scatter_path = f"{out_prefix}_scatter.png"
    plt.savefig(scatter_path); plt.close(fig)
    paths["scatter"] = scatter_path

    # 절대오차 히스토그램
    err = np.abs(yhat - y)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(err[:, 0], bins=20, alpha=0.6, label="Dashcam")
    ax2.hist(err[:, 1], bins=20, alpha=0.6, label="Other")
    ax2.set_title(f"Absolute Error Histogram (basis={target_basis})")
    ax2.set_xlabel("Abs Error"); ax2.set_ylabel("Count"); ax2.legend()
    hist_path = f"{out_prefix}_err_hist.png"
    plt.tight_layout(); plt.savefig(hist_path); plt.close(fig2)
    paths["err_hist"] = hist_path
    return paths

def df_to_csv(rows: List[dict], out_csv_path: str):
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv_path, index=False, encoding="utf-8")

# ------------------------------
# (D) 비디오 인코더 → 회귀 헤드 (+DP 지원)
# ------------------------------
from transformers import AutoImageProcessor, VideoMAEModel

class VideoEncoderRegressor(nn.Module):
    def __init__(self,
                 backbone_name: str = "MCG-NJU/videomae-base",
                 hidden_dim: int = 768,
                 target_basis: float = 10.0,
                 freeze_backbone: bool = True):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(backbone_name)
        self.backbone = VideoMAEModel.from_pretrained(backbone_name)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2)
        )
        self.target_basis = target_basis

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values)   # last_hidden_state: [B, T*P+1, H]
        cls = out.last_hidden_state[:, 0]                # [B, hidden_dim]
        logits = self.head(cls)                          # [B, 2]
        probs = torch.softmax(logits, dim=-1)            # 합=1
        return probs * self.target_basis                 # 합=target_basis

def _maybe_wrap_dp(model: nn.Module, gpu_ids: List[int]):
    if torch.cuda.is_available() and gpu_ids and len(gpu_ids) > 1:
        return nn.DataParallel(model, device_ids=gpu_ids)
    return model

def _get_processor_from_model(model: nn.Module):
    # DP 래핑 시 model.module.processor 로 접근
    if isinstance(model, nn.DataParallel):
        return model.module.processor
    return model.processor

def load_video_regressor(ckpt_path: str,
                         backbone_name: str = "MCG-NJU/videomae-base",
                         target_basis: float = 10.0,
                         device: str = "cuda:0",
                         gpu_ids: Optional[List[int]] = None):
    model = VideoEncoderRegressor(backbone_name=backbone_name,
                                  target_basis=target_basis)
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        model.load_state_dict(obj["state_dict"], strict=True)
    elif isinstance(obj, dict):
        model.load_state_dict(obj, strict=True)
    else:
        model = obj
    model.to(device)
    model = _maybe_wrap_dp(model, gpu_ids or [])
    model.eval()
    return model

@torch.no_grad()
def predict_fault_ratio_from_video_direct(model: nn.Module,
                                          video_path: str,
                                          num_frames: int = 16,
                                          size: int = 224,
                                          device: str = "cuda:0") -> np.ndarray:
    frames = sample_frames_pil(video_path, num_frames=num_frames, size=size)
    processor = _get_processor_from_model(model)
    inputs = processor(videos=[frames], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)  # [1, F, 3, H, W]
    pred_basis = model(pixel_values=pixel_values).squeeze(0).float().cpu().numpy()  # [2]
    return pred_basis

# ------------------------------
# (E) 학습
# ------------------------------
def train_video_regressor_from_json(train_json: str,
                                    save_path: str,
                                    video_root: str,
                                    backbone_name: str = "MCG-NJU/videomae-base",
                                    target_basis: float = 10.0,
                                    num_frames: int = 16,
                                    size: int = 224,
                                    epochs: int = 5,
                                    lr: float = 1e-4,
                                    batch_size: int = 2,
                                    device: str = "cuda:0",
                                    freeze_backbone: bool = True,
                                    gpu_ids: Optional[List[int]] = None,
                                    wandb_args: Optional[dict] = None):
    with open(train_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # W&B
    if wandb_args:
        wandb_init(
            project=wandb_args["project"], job_type="train",
            entity=wandb_args["entity"], run_name=wandb_args["run_name"],
            tags=wandb_args["tags"], group=wandb_args["group"],
            mode=wandb_args["mode"],
            config=dict(backbone=backbone_name, target_basis=target_basis,
                        num_frames=num_frames, size=size, epochs=epochs, lr=lr, bs=batch_size,
                        gpus=gpu_ids)
        )

    # 모델 + DP
    model = VideoEncoderRegressor(backbone_name=backbone_name,
                                  target_basis=target_basis,
                                  freeze_backbone=freeze_backbone).to(device)
    model = _maybe_wrap_dp(model, gpu_ids or [])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    def iter_batches(items, bs):
        for i in range(0, len(items), bs):
            yield items[i:i+bs]

    model.train()
    for ep in range(1, epochs + 1):
        losses = []
        for batch in iter_batches(data, batch_size):
            videos, y = [], []
            for row in batch:
                if "dashcam_vehicle_negligence" not in row or "other_vehicle_negligence" not in row:
                    continue
                video_name = row.get("video_name") or row.get("video_path") or ""
                vpath = find_video_file(video_root, video_name) if video_name else None
                if not vpath and row.get("video_path") and os.path.exists(row["video_path"]):
                    vpath = row["video_path"]
                if not vpath:
                    continue

                frames = sample_frames_pil(vpath, num_frames=num_frames, size=size)
                videos.append(frames)
                gt100 = normalize_pair_100([row["dashcam_vehicle_negligence"], row["other_vehicle_negligence"]])
                y.append(to_basis(gt100, target_basis))

            if not videos:
                continue

            processor = _get_processor_from_model(model)
            inputs = processor(videos=videos, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            target = torch.tensor(y, dtype=torch.float32, device=device)

            pred = model(pixel_values)                   # [B,2], 합=target_basis
            loss = torch.nn.functional.l1_loss(pred, target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"[ep {ep}] train L1 = {mean_loss:.4f}")
        if wandb_args:
            wandb.log({"train/l1": mean_loss, "epoch": ep})

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({"state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()},
               save_path)
    print("saved:", save_path)
    if wandb_args:
        wandb.finish()

# ------------------------------
# (F) 평가(JSON)
# ------------------------------
def evaluate_on_json_direct(eval_json_path: str,
                            direct_ckpt_path: str,
                            out_json_path: str,
                            video_root: str,
                            backbone_name: str = "MCG-NJU/videomae-base",
                            target_basis: float = 10.0,
                            num_frames: int = 16,
                            size: int = 224,
                            device: str = "cuda:0",
                            verbose: bool = True,
                            print_every: int = 1,
                            use_tqdm: bool = True,
                            shard_idx: int = -1,
                            num_shards: int = -1,
                            gpu_ids: Optional[List[int]] = None,
                            wandb_args: Optional[dict] = None):
    if not os.path.exists(eval_json_path):
        raise FileNotFoundError(eval_json_path)
    if not os.path.exists(direct_ckpt_path):
        raise FileNotFoundError(direct_ckpt_path)
    if not os.path.isdir(video_root):
        raise NotADirectoryError(video_root)

    if wandb_args:
        wandb_init(
            project=wandb_args["project"], job_type="evaluation",
            entity=wandb_args["entity"], run_name=wandb_args["run_name"],
            tags=wandb_args["tags"], group=wandb_args["group"],
            mode=wandb_args["mode"],
            config={
                "eval_json": eval_json_path,
                "direct_ckpt": direct_ckpt_path,
                "video_root": video_root,
                "backbone": backbone_name,
                "target_basis": target_basis,
                "num_frames": num_frames,
                "size": size,
                "device": device,
                "gpus": gpu_ids,
            },
        )

    model = load_video_regressor(direct_ckpt_path,
                                 backbone_name=backbone_name,
                                 target_basis=target_basis,
                                 device=device,
                                 gpu_ids=gpu_ids)

    with open(eval_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    import numpy as _np
    if shard_idx >= 0 and num_shards > 0:
        idxs = _np.array_split(_np.arange(len(data)), num_shards)[shard_idx].tolist()
        data = [data[i] for i in idxs]

    iterator = range(len(data))
    if use_tqdm:
        try:
            from tqdm.auto import tqdm as _tqdm
            iterator = _tqdm(iterator, total=len(data), desc="Evaluating(Direct)", dynamic_ncols=True)
        except Exception:
            pass

    preds_basis, labels_basis, results = [], [], []
    N = len(data)
    for i in iterator:
        row = data[i]
        # GT
        gt_basis = None
        gt100 = None
        if "dashcam_vehicle_negligence" in row and "other_vehicle_negligence" in row:
            gt100 = normalize_pair_100([row["dashcam_vehicle_negligence"], row["other_vehicle_negligence"]])
            gt_basis = np.array(to_basis(gt100, target_basis), dtype=float)

        # 비디오 경로
        video_name = row.get("video_name") or row.get("video_path") or ""
        vpath = find_video_file(video_root, video_name) if video_name else None
        if not vpath and row.get("video_path") and os.path.exists(row["video_path"]):
            vpath = row["video_path"]
        if not vpath:
            results.append({"idx": i, "video_name": video_name, "error": "video_not_found"})
            if verbose:
                print(f"[{i+1}/{N}] {video_name} | VIDEO NOT FOUND", flush=True)
            continue

        # 직접 예측
        try:
            pred_basis = predict_fault_ratio_from_video_direct(
                model, vpath, num_frames=num_frames, size=size, device=device
            )
        except Exception as e:
            results.append({"idx": i, "video_name": video_name, "error": f"direct_pred_failed: {e}"})
            if verbose:
                print(f"[{i+1}/{N}] {video_name} | DIRECT PRED ERROR: {e}", flush=True)
            continue

        preds_basis.append(pred_basis)
        pred_100 = [float(x) * (100.0 / target_basis) for x in pred_basis]

        out_item = {
            "idx": i,
            "video_name": video_name,
            "pred_basis_dashcam": float(pred_basis[0]),
            "pred_basis_other": float(pred_basis[1]),
            "pred_100_dashcam": float(pred_100[0]),
            "pred_100_other": float(pred_100[1]),
        }
        if gt_basis is not None:
            out_item["gt_basis_dashcam"] = float(gt_basis[0])
            out_item["gt_basis_other"] = float(gt_basis[1])
            out_item["gt_100_dashcam"] = float(gt100[0])
            out_item["gt_100_other"] = float(gt100[1])
            out_item["abs_err_basis_dashcam"] = abs(out_item["gt_basis_dashcam"] - out_item["pred_basis_dashcam"])
            out_item["abs_err_basis_other"] = abs(out_item["gt_basis_other"] - out_item["pred_basis_other"])
            labels_basis.append(gt_basis)

        results.append(out_item)

        if verbose and ((i == 0) or ((i + 1) % print_every == 0) or (i + 1 == N)):
            msg = f"[{i+1}/{N}] {video_name} | pred_basis=[{out_item['pred_basis_dashcam']:.2f}, {out_item['pred_basis_other']:.2f}]"
            if gt_basis is not None:
                msg += f" | gt_basis=[{out_item['gt_basis_dashcam']:.2f}, {out_item['gt_basis_other']:.2f}]"
            print(msg, flush=True)

        if wandb_args and ((i + 1) % 20 == 0):
            wandb.log({"eval_direct/progress_samples": i + 1})

    # 저장/메트릭
    out_prefix = os.path.splitext(out_json_path)[0]
    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
    metrics = {}
    if labels_basis:
        y = np.vstack(labels_basis)
        yhat = np.vstack(preds_basis)
        metrics = compute_metrics(y, yhat)
        metrics["target_basis"] = target_basis
        plot_paths = save_plots(y, yhat, target_basis, out_prefix)
        df_to_csv(results, f"{out_prefix}.csv")
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "results": results}, f, ensure_ascii=False, indent=2)

        if wandb_args:
            wandb.log({
                "eval_direct/MAE": metrics["MAE"],
                "eval_direct/RMSE": metrics["RMSE"],
                "eval_direct/R2": metrics["R2"],
                "eval_direct/MAE_dashcam": metrics["MAE_dashcam"],
                "eval_direct/MAE_other": metrics["MAE_other"],
            })
            for k, p in plot_paths.items():
                if os.path.exists(p):
                    wandb.log({f"plots_direct/{k}": wandb.Image(p)})
            # 결과물 아티팩트
            try:
                art = wandb.Artifact("eval_results_direct", type="evaluation")
                art.add_file(out_json_path)
                art.add_file(f"{out_prefix}.csv")
                for p in plot_paths.values():
                    if os.path.exists(p):
                        art.add_file(p)
                wandb.log_artifact(art)
            except Exception:
                pass
    else:
        df_to_csv(results, f"{out_prefix}.csv")
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": {}, "results": results}, f, ensure_ascii=False, indent=2)

    print("=== Evaluation Summary (Direct video→fault) ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved JSON: {out_json_path}")
    print(f"Saved CSV : {out_prefix}.csv")
    if wandb_args:
        wandb.finish()

# ------------------------------
# (G) 단일 비디오 빠른 예측
# ------------------------------
def run_fault_from_video_direct(video_path: str,
                                direct_ckpt_path: str,
                                backbone_name: str = "MCG-NJU/videomae-base",
                                target_basis: float = 10.0,
                                num_frames: int = 16,
                                size: int = 224,
                                device: str = "cuda:0",
                                gpu_ids: Optional[List[int]] = None):
    model = load_video_regressor(direct_ckpt_path,
                                 backbone_name=backbone_name,
                                 target_basis=target_basis,
                                 device=device,
                                 gpu_ids=gpu_ids)
    basis = predict_fault_ratio_from_video_direct(model, video_path, num_frames=num_frames, size=size, device=device)
    ratio_100 = basis * (100.0 / target_basis)
    dash, other = ratio_100.tolist()
    print("----- [Direct video→fault] -----")
    print(f"📊 Predicted Fault Ratio → Dashcam: {dash:.1f}%, Other: {other:.1f}%")
    return {"dashcam_100": float(dash), "other_100": float(other), "basis": basis.tolist()}

# ------------------------------
# (H) CLI
# ------------------------------
def parse_gpus(s: str) -> List[int]:
    if not s: return []
    ids = []
    for x in s.split(","):
        x = x.strip()
        if x == "": continue
        try:
            ids.append(int(x))
        except ValueError:
            pass
    return ids

def parse_args():
    p = argparse.ArgumentParser(description="Direct video→fault pipeline (no text).")
    p.add_argument("--mode", type=str, default=os.environ.get("MODE", "eval"),
                   choices=["train", "eval", "single"])

    # 공통
    p.add_argument("--video_backbone", type=str,
                   default=os.environ.get("VIDEO_BACKBONE", "MCG-NJU/videomae-base"))
    p.add_argument("--target_basis", type=float,
                   default=float(os.environ.get("TARGET_BASIS", 10.0)))
    p.add_argument("--num_frames", type=int, default=int(os.environ.get("NUM_FRAMES", 16)))
    p.add_argument("--size", type=int, default=int(os.environ.get("SIZE", 224)))
    p.add_argument("--device", type=str, default=os.environ.get("DEVICE", "cuda:0"))
    p.add_argument("--gpus", type=str, default=os.environ.get("GPUS", "0"),
                   help="쉼표로 구분된 GPU 인덱스 (예: '1,2')")

    # W&B
    p.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "video-fault-direct"))
    p.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY", None))
    p.add_argument("--wandb_run", type=str, default=os.environ.get("WANDB_RUN", None))
    p.add_argument("--wandb_group", type=str, default=os.environ.get("WANDB_GROUP", None))
    p.add_argument("--wandb_tags", type=str, default=os.environ.get("WANDB_TAGS", "direct,video"))
    p.add_argument("--wandb_mode", type=str, default=os.environ.get("WANDB_MODE", "online"), choices=["online", "offline", "disabled"])

    # 학습
    p.add_argument("--train_json", type=str, default=os.environ.get("TRAIN_JSON", ""))
    p.add_argument("--video_root", type=str, default=os.environ.get("VIDEO_ROOT", ""))
    p.add_argument("--save_ckpt", type=str, default=os.environ.get("SAVE_CKPT", "./video_fault_regressor.pt"))
    p.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", 5)))
    p.add_argument("--lr", type=float, default=float(os.environ.get("LR", 1e-4)))
    p.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", 2)))
    p.add_argument("--freeze_backbone", action="store_true")

    # 평가
    p.add_argument("--eval_json", type=str, default=os.environ.get("EVAL_JSON", ""))
    p.add_argument("--out_json", type=str, default=os.environ.get("OUT_JSON", "./eval_results_direct.json"))
    p.add_argument("--direct_ckpt", type=str, default=os.environ.get("DIRECT_CKPT", "./video_fault_regressor.pt"))
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--print_every", type=int, default=int(os.environ.get("PRINT_EVERY", 1)))
    p.add_argument("--no_tqdm", action="store_true")
    p.add_argument("--shard_idx", type=int, default=int(os.environ.get("SHARD_IDX", -1)))
    p.add_argument("--num_shards", type=int, default=int(os.environ.get("NUM_SHARDS", -1)))

    # 단일
    p.add_argument("--video_path", type=str, default=os.environ.get("VIDEO_PATH", ""))
    return p.parse_args()

# ------------------------------
# (I) Main
# ------------------------------
if __name__ == "__main__":
    args = parse_args()

    gpu_ids = parse_gpus(args.gpus)
    # 기본 디바이스 자동 보정 (예: --gpus 1,2 이면 device= cuda:1 로 설정)
    if torch.cuda.is_available() and gpu_ids:
        args.device = f"cuda:{gpu_ids[0]}"
    elif not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("[Warn] CUDA unavailable. Falling back to CPU.")
        args.device = "cpu"

    wandb_args = dict(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run,
        tags=wandb_parse_tags(args.wandb_tags),
        group=args.wandb_group,
        mode=args.wandb_mode,
    )

    if args.mode == "train":
        if not args.train_json or not os.path.exists(args.train_json):
            raise SystemExit(f"--train_json not found: {args.train_json}")
        if not args.video_root or not os.path.isdir(args.video_root):
            raise SystemExit(f"--video_root not a directory: {args.video_root}")

        train_video_regressor_from_json(
            train_json=args.train_json,
            save_path=args.save_ckpt,
            video_root=args.video_root,
            backbone_name=args.video_backbone,
            target_basis=args.target_basis,
            num_frames=args.num_frames,
            size=args.size,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            freeze_backbone=args.freeze_backbone,
            gpu_ids=gpu_ids,
            wandb_args=wandb_args if args.wandb_mode != "disabled" else None,
        )

    elif args.mode == "eval":
        if not args.eval_json or not os.path.exists(args.eval_json):
            raise SystemExit(f"--eval_json not found: {args.eval_json}")
        if not args.video_root or not os.path.isdir(args.video_root):
            raise SystemExit(f"--video_root not a directory: {args.video_root}")
        if not args.direct_ckpt or not os.path.exists(args.direct_ckpt):
            raise SystemExit(f"--direct_ckpt not found: {args.direct_ckpt}")

        evaluate_on_json_direct(
            eval_json_path=args.eval_json,
            direct_ckpt_path=args.direct_ckpt,
            out_json_path=args.out_json,
            video_root=args.video_root,
            backbone_name=args.video_backbone,
            target_basis=args.target_basis,
            num_frames=args.num_frames,
            size=args.size,
            device=args.device,
            verbose=not args.quiet,
            print_every=args.print_every,
            use_tqdm=not args.no_tqdm,
            shard_idx=args.shard_idx,
            num_shards=args.num_shards,
            gpu_ids=gpu_ids,
            wandb_args=wandb_args if args.wandb_mode != "disabled" else None,
        )

    else:  # single
        if not args.video_path or not os.path.exists(args.video_path):
            raise SystemExit(f"--video_path not found: {args.video_path}")
        if not args.direct_ckpt or not os.path.exists(args.direct_ckpt):
            raise SystemExit(f"--direct_ckpt not found: {args.direct_ckpt}")

        run_fault_from_video_direct(
            video_path=args.video_path,
            direct_ckpt_path=args.direct_ckpt,
            backbone_name=args.video_backbone,
            target_basis=args.target_basis,
            num_frames=args.num_frames,
            size=args.size,
            device=args.device,
            gpu_ids=gpu_ids,
        )
