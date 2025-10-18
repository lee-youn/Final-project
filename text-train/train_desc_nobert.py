# eval_clsprob2ratio.py
# pip install torch torchvision transformers opencv-python numpy pandas scikit-learn matplotlib wandb
import os, json, argparse, glob
from typing import Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
from tqdm.auto import tqdm
import wandb
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F

from transformers import TimesformerModel, VideoMAEModel
from torchvision.models.video import r3d_18, R3D_18_Weights

# ----- 고정 클래스 개수 -----
N_DV = 30
N_OV = 28

# ===== 비디오 로딩 =====
@torch.no_grad()
def load_video_tensor(path, num_frames=16, size=224):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); raise RuntimeError(f"no frames: {path}")
    idxs = np.linspace(0, max(total-1,0), num_frames).astype(int)
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
    if not frames: raise RuntimeError(f"no decoded frames: {path}")
    vid = torch.stack(frames, 0)  # (T,C,H,W)
    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    vid = (vid - mean)/std
    return vid.permute(1,0,2,3)  # (C,T,H,W)

def snap_pair_to_integer_basis_np(v, total=10):
    v = np.asarray(v, dtype=float)
    v = np.maximum(v, 0.0)
    s = v.sum()
    if s <= 0:
        a = total // 2
        return np.array([float(a), float(total - a)])
    v = v * (total / s)  # 합 total로 정규화
    a_int = int(np.floor(v[0] + 0.5))
    a_int = max(0, min(total, a_int))
    b_int = total - a_int
    return np.array([float(a_int), float(b_int)])

def find_video_file(video_root: str, name_or_rel: str) -> Optional[str]:
    if not name_or_rel: return None
    cand = os.path.join(video_root, name_or_rel)
    if os.path.exists(cand): return cand
    stem = os.path.splitext(name_or_rel)[0]
    for ext in (".mp4",".avi",".mov",".mkv"):
        p = os.path.join(video_root, stem + ext)
        if os.path.exists(p): return p
    g = glob.glob(os.path.join(video_root, stem + "*"))
    return g[0] if g else None

# ===== 분류기: DV/OV 두 헤드 =====
class DVOVClassifier(nn.Module):
    def __init__(self, n_dv=N_DV, n_ov=N_OV, backbone="r3d18", pretrained=False):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "r3d18":
            self.backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1 if pretrained else None)
            feat = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "timesformer":
            self.backbone = TimesformerModel.from_pretrained(
                "facebook/timesformer-base-finetuned-k400", use_safetensors=True)
            feat = self.backbone.config.hidden_size
        elif backbone == "videomae":
            self.backbone = VideoMAEModel.from_pretrained(
                "MCG-NJU/videomae-base-finetuned-kinetics", use_safetensors=True)
            feat = self.backbone.config.hidden_size
        else:
            raise ValueError(backbone)
        self.dv = nn.Linear(feat, n_dv)
        self.ov = nn.Linear(feat, n_ov)

    def forward(self, x):
        if self.backbone_name == "r3d18":
            z = self.backbone(x)
        else:
            x = x.permute(0,2,1,3,4)
            out = self.backbone(x)
            z = out.last_hidden_state.mean(1)
        return self.dv(z), self.ov(z)

def load_classifier(ckpt_path: str, backbone="r3d18", device="cuda", pretrained=False):
    model = DVOVClassifier(backbone=backbone, pretrained=pretrained)
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            new_sd[k[7:]] = v
        else:
            new_sd[k] = v
    model.load_state_dict(new_sd, strict=False)
    model.to(device).eval()
    return model

# ===== 회귀기 1) 확률 → 비율 =====
class ProbToRatioRegressor(nn.Module):
    def __init__(self, in_dim, hidden=256, dropout=0.2):
        super().__init__()
        # self.mlp = nn.Sequential(
        #     nn.LayerNorm(in_dim),
        #     nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
        #     nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
        #     nn.Linear(hidden//2, 2),
        # )
        self.mlp = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, 2),
            )
    def forward(self, x):             # x: [B, 58]
        # softmax 대신 softplus + 합으로 정규화
        y = self.mlp(x)                            # [B,2]
        v = F.softplus(y)                          # >= 0
        s = v.sum(dim=1, keepdim=True) + 1e-8
        probs = v / s                               # 합 = 1
        return probs


# ===== 회귀기 2) (임베딩 기반) 확률 → 가중임베딩 → 비율 =====
# class EmbedToRatioRegressor(nn.Module):
#     def __init__(self, dv_vocab=N_DV, ov_vocab=N_OV, emb_dim=64, hidden=128, dropout=0.2):
#         super().__init__()
#         self.dv_emb = nn.Embedding(dv_vocab, emb_dim)
#         self.ov_emb = nn.Embedding(ov_vocab, emb_dim)
#         self.mlp = nn.Sequential(
#             nn.LayerNorm(emb_dim*2),
#             nn.Linear(emb_dim*2, hidden), nn.ReLU(), nn.Dropout(dropout),
#             nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
#             nn.Linear(hidden//2, 2),
#         )
#     def forward(self, p_dv, p_ov):    # p_*: [B, vocab]
#         dv = p_dv @ self.dv_emb.weight   # [B, emb_dim]
#         ov = p_ov @ self.ov_emb.weight   # [B, emb_dim]
#         h = torch.cat([dv, ov], dim=-1)  # [B, 2*emb_dim]
#         y = self.mlp(h)                  # [B,2]
#         v = F.softplus(y)                # >= 0
#         s = v.sum(dim=1, keepdim=True) + 1e-8
#         probs = v / s                    # 합 = 1
#         return probs
class EmbedToRatioRegressor(nn.Module):
    def __init__(self, dv_vocab=N_DV, ov_vocab=N_OV, emb_dim=64,
                 hidden=128, dropout=0.2, depth="shallow"):  # "shallow" | "deep"
        super().__init__()
        self.dv_emb = nn.Embedding(dv_vocab, emb_dim)
        self.ov_emb = nn.Embedding(ov_vocab, emb_dim)

        if depth == "shallow":
            # LayerNorm -> Linear(hidden) -> ReLU -> Dropout -> Linear(2)
            self.mlp = nn.Sequential(
                nn.LayerNorm(emb_dim*2),
                nn.Linear(emb_dim*2, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, 2),
            )
        else:  # deep (기존 2개 hidden)
            self.mlp = nn.Sequential(
                nn.LayerNorm(emb_dim*2),
                nn.Linear(emb_dim*2, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden//2, 2),
            )

    # def forward(self, p_dv, p_ov):          # <<< 여기! (p_dv, p_ov)로 변경
    #     dv = p_dv @ self.dv_emb.weight      # [B, emb_dim]
    #     ov = p_ov @ self.ov_emb.weight      # [B, emb_dim]
    #     h  = torch.cat([dv, ov], dim=-1)    # [B, 2*emb_dim]
    #     y  = self.mlp(h)                    # [B, 2]
    #     v  = F.softplus(y)                  # >= 0
    #     probs = v / (v.sum(dim=1, keepdim=True) + 1e-8)  # 합=1
    #     return probs
    def forward(self, p_dv, p_ov):
        dv = p_dv @ self.dv_emb.weight      # [B, emb_dim]
        ov = p_ov @ self.ov_emb.weight      # [B, emb_dim]
        h  = torch.cat([dv, ov], dim=-1)    # [B, 2*emb_dim]
        y  = self.mlp(h)                    # [B, 2]
        probs = F.softmax(y, dim=-1)        # 합=1
        return probs
    
def load_regressor_auto(ckpt_path: str, device="cuda"):
    sd_raw = torch.load(ckpt_path, map_location="cpu")
    sd = sd_raw["model"] if (isinstance(sd_raw, dict) and "model" in sd_raw) else sd_raw

    # 1) 임베딩 기반 체크포인트 감지
    if any(k.startswith("dv_emb.weight") for k in sd.keys()):
        # emb_dim 자동 추정
        emb_dim = sd["dv_emb.weight"].shape[1]
        dv_vocab = sd["dv_emb.weight"].shape[0]
        ov_vocab = sd["ov_emb.weight"].shape[0]
        model = EmbedToRatioRegressor(dv_vocab=dv_vocab, ov_vocab=ov_vocab, emb_dim=emb_dim)
        model.load_state_dict(sd, strict=True)
        model.to(device).eval()
        return model, "embed"

    # 2) 확률→비율 MLP (입력 58)
    in_dim = N_DV + N_OV
    model = ProbToRatioRegressor(in_dim=in_dim)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model, "prob"

# ===== 메트릭/플롯 =====
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics(y_basis: np.ndarray, yhat_basis: np.ndarray) -> dict:
    # MAE
    mae = mean_absolute_error(y_basis, yhat_basis)

    # RMSE: 구버전 호환(= 직접 루트)
    mse = mean_squared_error(y_basis, yhat_basis)
    rmse = float(np.sqrt(mse))

    # 전체(다중출력) R2 및 per-head
    r2  = r2_score(y_basis, yhat_basis)
    r2_dc = r2_score(y_basis[:,0], yhat_basis[:,0])
    r2_ov = r2_score(y_basis[:,1], yhat_basis[:,1])

    mae_dc = mean_absolute_error(y_basis[:,0], yhat_basis[:,0])
    mae_ov = mean_absolute_error(y_basis[:,1], yhat_basis[:,1])

    return {
        "MAE": float(mae),
        "RMSE": rmse,
        "R2": float(r2),
        "R2_dashcam": float(r2_dc),
        "R2_other": float(r2_ov),
        "MAE_dashcam": float(mae_dc),
        "MAE_other": float(mae_ov),
    }

def save_plots(y: np.ndarray, yhat: np.ndarray, target_basis: float, out_prefix: str) -> Dict[str,str]:
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    paths = {}
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    for i,title in enumerate(["Dashcam","Other Vehicle"]):
        ax[i].scatter(y[:,i], yhat[:,i], alpha=0.5)
        ax[i].plot([0,target_basis],[0,target_basis],"--")
        ax[i].set_title(f"{title} Fault (basis={target_basis})")
        ax[i].set_xlabel("True"); ax[i].set_ylabel("Pred")
        ax[i].set_xlim(0,target_basis); ax[i].set_ylim(0,target_basis)
    plt.tight_layout()
    p_sc = f"{out_prefix}_scatter.png"; plt.savefig(p_sc); plt.close(fig)
    paths["scatter"] = p_sc
    err = np.abs(yhat - y)
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.hist(err[:,0], bins=20, alpha=0.6, label="Dashcam")
    ax2.hist(err[:,1], bins=20, alpha=0.6, label="Other")
    ax2.set_title(f"Absolute Error Histogram (basis={target_basis})"); ax2.legend()
    plt.tight_layout(); p_hist = f"{out_prefix}_err_hist.png"; plt.savefig(p_hist); plt.close(fig2)
    paths["err_hist"] = p_hist
    return paths

# ===== 메인 평가 루프 =====
@torch.no_grad()
def evaluate(eval_json: str, video_root: str,
             cls_ckpt: str, reg_ckpt: str,
             backbone="r3d18", classifier_pretrained=False,
             out_json: str = "./eval_prob2ratio.json",
             num_frames=16, size=224, target_basis=10.0,
             gpus="0", project="prob2ratio-eval", run_name=None, print_every=1, use_hard_labels=False):

    gpu_list = [int(x) for x in gpus.split(",") if x.strip()!=""]
    if torch.cuda.is_available() and len(gpu_list)>=1:
        device = f"cuda:{gpu_list[0]}"
    else:
        device = "cpu"

    wandb.init(project=project, name=run_name, job_type="evaluation", config={
        "eval_json": eval_json, "video_root": video_root,
        "cls_ckpt": cls_ckpt, "reg_ckpt": reg_ckpt,
        "backbone": backbone, "classifier_pretrained": classifier_pretrained,
        "num_frames": num_frames, "size": size, "target_basis": target_basis
    })

    # 1) 모델 로드
    clf = load_classifier(cls_ckpt, backbone=backbone, device=device, pretrained=classifier_pretrained)
    reg, reg_type = load_regressor_auto(reg_ckpt, device=device)  # 'prob' or 'embed'
    wandb.log({"regressor/type": reg_type})

    # 2) 데이터 로드
    data = json.load(open(eval_json, "r", encoding="utf-8"))
    assert isinstance(data, list), "eval_json must be a list of dicts"

    preds_basis, labels_basis, rows_out = [], [], []
    preds_basis_int, labels_basis_int = [], []
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    out_prefix = os.path.splitext(out_json)[0]

    for i, row in enumerate(tqdm(data, dynamic_ncols=True, desc="Evaluating")):
        vname = row.get("video_name") or row.get("video_path") or ""
        vpath = find_video_file(video_root, vname) if vname else None
        if (not vpath) and row.get("video_path") and os.path.exists(row["video_path"]):
            vpath = row["video_path"]
        if not vpath:
            rows_out.append({"idx": i, "video_name": vname, "error": "video_not_found"})
            continue

        # GT (있으면 0~1 정규화 후 기준변환)
        gt_basis = None
        if "dashcam_vehicle_negligence" in row and "other_vehicle_negligence" in row:
            a, b = float(row["dashcam_vehicle_negligence"]), float(row["other_vehicle_negligence"])
            if a > 1.5 or b > 1.5:  # 0~100 스케일 처리
                a /= 100.0; b /= 100.0
            s = max(a+b, 1e-6)
            a, b = a/s, b/s
            gt_basis = np.array([a*target_basis, b*target_basis], dtype=float)

        try:
            x = load_video_tensor(vpath, num_frames=num_frames, size=size).unsqueeze(0).to(device)
            ld, lv = clf(x)
            pd_dv = ld.softmax(-1)
            pd_ov = lv.softmax(-1)
            if reg_type == "embed" and use_hard_labels:
                # DV 하드 라벨 → 원-핫
                dv_idx = pd_dv.argmax(dim=-1)           # [B]
                ov_idx = pd_ov.argmax(dim=-1)           # [B]
                pd_dv_hard = torch.zeros_like(pd_dv)
                pd_ov_hard = torch.zeros_like(pd_ov)
                pd_dv_hard.scatter_(1, dv_idx.unsqueeze(1), 1.0)
                pd_ov_hard.scatter_(1, ov_idx.unsqueeze(1), 1.0)
                pd_dv, pd_ov = pd_dv_hard, pd_ov_hard

            if reg_type == "prob":
                feat = torch.cat([pd_dv, pd_ov], dim=-1)           # [1, 58]
                yp = reg(feat).squeeze(0).cpu().numpy()            # [2], sum=1
            else:  # embed
                yp = reg(pd_dv, pd_ov).squeeze(0).cpu().numpy()    # [2], sum=1

            pred_basis = (yp * float(target_basis)).astype(float)

            # ★ 정수 스냅 (합 = round(target_basis))
            snap_total = int(round(target_basis))
            pred_basis_int = snap_pair_to_integer_basis_np(pred_basis, total=snap_total)

            out = {
                "idx": i, "video_name": vname,
                "pred_basis_dashcam": float(pred_basis[0]),
                "pred_basis_other":   float(pred_basis[1]),
                "pred_100_dashcam":   float(pred_basis[0]*(100.0/target_basis)),
                "pred_100_other":     float(pred_basis[1]*(100.0/target_basis)),

                # ★ 정수 스냅 값도 저장
                "pred_basis_dashcam_int": float(pred_basis_int[0]),
                "pred_basis_other_int":   float(pred_basis_int[1]),
                "pred_100_dashcam_int":   float(pred_basis_int[0]*(100.0/target_basis)),
                "pred_100_other_int":     float(pred_basis_int[1]*(100.0/target_basis)),
            }


            if gt_basis is not None:
                out["gt_basis_dashcam"] = float(gt_basis[0])
                out["gt_basis_other"]   = float(gt_basis[1])
                preds_basis.append(pred_basis)
                labels_basis.append(gt_basis)
                preds_basis_int.append(pred_basis_int)
                labels_basis_int.append(gt_basis)

            rows_out.append(out)

            if (i+1) % print_every == 0:
                msg = (
                    f"[{i+1}/{len(data)}] {vname} | "
                    f"pred_basis=[{pred_basis[0]:.2f}, {pred_basis[1]:.2f}] "
                    f"pred_100=[{pred_basis[0]*(100.0/target_basis):.1f}%, {pred_basis[1]*(100.0/target_basis):.1f}%]"
                )
                if gt_basis is not None:
                    msg += f" | gt=[{gt_basis[0]:.2f}, {gt_basis[1]:.2f}]"
                tqdm.write(msg)

        except Exception as e:
            rows_out.append({"idx": i, "video_name": vname, "error": f"run_failed: {e}"})

        if (i+1) % 20 == 0:
            wandb.log({"eval_progress_samples": i+1})

    # 3) 메트릭/저장
    metrics = {}
    if len(preds_basis) and len(labels_basis):
        yhat = np.vstack(preds_basis)
        y    = np.vstack(labels_basis)
        metrics = compute_metrics(y, yhat)

        if len(preds_basis_int) == len(labels_basis_int):
            yhat_int = np.vstack(preds_basis_int)
            y_int_m  = compute_metrics(y, yhat_int)
            # 키 충돌 피하려고 접두사 부여
            metrics.update({
                "int/MAE":        y_int_m["MAE"],
                "int/RMSE":       y_int_m["RMSE"],
                "int/R2":         y_int_m["R2"],
                "int/R2_dashcam": y_int_m["R2_dashcam"],
                "int/R2_other":   y_int_m["R2_other"],
                "int/MAE_dashcam":y_int_m["MAE_dashcam"],
                "int/MAE_other":  y_int_m["MAE_other"],
            })

        # 플롯(연속값 기준)
        plots = save_plots(y, yhat, float(target_basis), out_prefix)
        for k, pth in plots.items():
            if os.path.exists(pth):
                wandb.log({f"plots/{k}": wandb.Image(pth)})

        # W&B 로그
        log_payload = {
            "eval/MAE": metrics.get("MAE"),
            "eval/RMSE": metrics.get("RMSE"),
            "eval/R2": metrics.get("R2"),
            "eval/R2_dashcam": metrics.get("R2_dashcam"),
            "eval/R2_other": metrics.get("R2_other"),
            "eval/MAE_dashcam": metrics.get("MAE_dashcam"),
            "eval/MAE_other": metrics.get("MAE_other"),
            "count": len(y),

            # ★ 정수 스냅 로그
            "eval_int/MAE": metrics.get("int/MAE"),
            "eval_int/RMSE": metrics.get("int/RMSE"),
            "eval_int/R2": metrics.get("int/R2"),
            "eval_int/R2_dashcam": metrics.get("int/R2_dashcam"),
            "eval_int/R2_other": metrics.get("int/R2_other"),
            "eval_int/MAE_dashcam": metrics.get("int/MAE_dashcam"),
            "eval_int/MAE_other": metrics.get("int/MAE_other"),
        }
        wandb.log({k: v for k, v in log_payload.items() if v is not None})

    # CSV/JSON 저장
    pd.DataFrame(rows_out).to_csv(f"{out_prefix}.csv", index=False, encoding="utf-8")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": rows_out}, f, ensure_ascii=False, indent=2)

    print("=== PROB→RATIO Evaluation Summary ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved JSON: {out_json}")
    print(f"Saved CSV : {out_prefix}.csv")
    wandb.finish()

# ===== CLI =====
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate: Classifier(DV/OV probs) → Regression head → Fault ratio")
    p.add_argument("--eval_json", type=str,
                   default=os.environ.get("EVAL_JSON", "/app/data/raw/json/text-evaluate/video_accident_ratio_training_results_trimmed_unsignalized_validation_0901.json"))
    p.add_argument("--video_root", type=str,
                   default=os.environ.get("VIDEO_ROOT", "/app/data/raw/videos/validation_reencoded"))
    p.add_argument("--classifier_ckpt", type=str,
                   default=os.environ.get("CLS_CKPT", "/app/checkpoints/best_exact_ep13_r3d18.pth"))
    # 확률-기반/임베딩-기반 ckpt 아무 거나 넣어도 자동감지됨
    p.add_argument("--regressor_ckpt", type=str, default="/app/text-train/checkpoints/best_multi_emb_onelayer_softmax.pt")
    p.add_argument("--backbone", type=str, default="r3d18", choices=["r3d18","timesformer","videomae"])
    p.add_argument("--classifier_pretrained", action="store_true")
    p.add_argument("--target_basis", type=float, default=10.0)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--out_json", type=str, default="./eval_prob2ratio.json")
    p.add_argument("--project", type=str, default="prob2ratio-eval")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--print_every", type=int, default=1)
    p.add_argument("--use_hard_labels", action="store_true",
               help="embed형 회귀 ckpt일 때 DV/OV를 soft 확률 대신 argmax 원-핫으로 넣기")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate(
        eval_json=args.eval_json, video_root=args.video_root,
        cls_ckpt=args.classifier_ckpt, reg_ckpt=args.regressor_ckpt,
        backbone=args.backbone, classifier_pretrained=args.classifier_pretrained,
        out_json=args.out_json, num_frames=args.num_frames, size=args.size,
        target_basis=args.target_basis, gpus=args.gpus,
        project=args.project, run_name=args.run_name, print_every=args.print_every,
        use_hard_labels=args.use_hard_labels
    )
