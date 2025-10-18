import os, argparse, json, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import wandb

# -----------------------
# Dataset
# -----------------------
class InfoOnlyDataset(Dataset):
    def __init__(self, df: pd.DataFrame,
                 dv_col="dashcam_vehicle_info",
                 ov_col="other_vehicle_info",
                 target_cols=("dashcam_vehicle_negligence","other_vehicle_negligence"),
                 cat_vocab=None,
                 scale_to_01=True):
        self.df = df.reset_index(drop=True)
        self.dv_col = dv_col
        self.ov_col = ov_col
        self.target_cols = target_cols
        assert all(c in df.columns for c in [dv_col, ov_col]), "Missing DV/OV info columns"
        assert all(t in df.columns for t in target_cols), f"Missing target columns {target_cols}"

        # --- build vocab (per column) ---
        if cat_vocab is None:
            dv_vals = sorted(df[dv_col].astype(str).unique().tolist())
            ov_vals = sorted(df[ov_col].astype(str).unique().tolist())
            self.cat_vocab = {
                "dv": {v:i for i,v in enumerate(dv_vals)},
                "ov": {v:i for i,v in enumerate(ov_vals)},
            }
        else:
            self.cat_vocab = cat_vocab

        # --- targets & scale detection ---
        y = df[list(target_cols)].astype(float).values  # (N,2)
        # 자동 스케일: 값이 1.5보다 크면 0~100로 보고 0~1로 스케일
        self.scale_to_01 = scale_to_01
        self.scale_div = np.ones(2, dtype=np.float32)
        if self.scale_to_01:
            for j in range(2):
                col = y[:,j]
                mx = np.nanmax(col)
                self.scale_div[j] = 100.0 if mx > 1.5 else 1.0
            y = y / self.scale_div

        self.targets = y.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        dv_idx = self.cat_vocab["dv"][str(row[self.dv_col])]
        ov_idx = self.cat_vocab["ov"][str(row[self.ov_col])]
        y = self.targets[i]  # (2,)
        return torch.tensor([dv_idx, ov_idx], dtype=torch.long), torch.tensor(y, dtype=torch.float32)

# -----------------------
# Model: embeddings -> MLP -> 2 outputs
# -----------------------
class CatEmbedRegressor(nn.Module):
    def __init__(self, dv_vocab_size, ov_vocab_size, emb_dim=64, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.dv_emb = nn.Embedding(dv_vocab_size, emb_dim)
        self.ov_emb = nn.Embedding(ov_vocab_size, emb_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(emb_dim*2),
            nn.Linear(emb_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 2),  # 2 targets
        )

    def forward(self, x_idx):  # x_idx: [B,2] -> (dv_idx, ov_idx)
        dv = self.dv_emb(x_idx[:,0])
        ov = self.ov_emb(x_idx[:,1])
        h = torch.cat([dv, ov], dim=-1)
        y = self.mlp(h)  # [B,2], each in [unbounded]
        # Sigmoid로 0~1 범위 보장 (비율 문제이므로)
        return torch.sigmoid(y)

# -----------------------
# Train/Eval
# -----------------------
def eval_metrics(y_true, y_pred, scale_div):
    """
    y_* are numpy (N,2) in 0~1 scale; convert back to original scale using scale_div (shape (2,))
    """
    y_true_raw = y_true * scale_div
    y_pred_raw = y_pred * scale_div

    out = {}
    names = ["dashcam_vehicle_negligence","other_vehicle_negligence"]
    for j, name in enumerate(names):
        mae = mean_absolute_error(y_true_raw[:,j], y_pred_raw[:,j])
        mse = mean_squared_error(y_true_raw[:,j], y_pred_raw[:,j])
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_raw[:,j], y_pred_raw[:,j])
        out[f"{name}/MAE"] = float(mae)
        out[f"{name}/MSE"] = float(mse)
        out[f"{name}/RMSE"] = float(rmse)
        out[f"{name}/R2"] = float(r2)

    # macro averages
    out["macro/MAE"]  = float(np.mean([out["dashcam_vehicle_negligence/MAE"], out["other_vehicle_negligence/MAE"]]))
    out["macro/MSE"]  = float(np.mean([out["dashcam_vehicle_negligence/MSE"], out["other_vehicle_negligence/MSE"]]))
    out["macro/RMSE"] = float(np.mean([out["dashcam_vehicle_negligence/RMSE"], out["other_vehicle_negligence/RMSE"]]))
    out["macro/R2"]   = float(np.mean([out["dashcam_vehicle_negligence/R2"], out["other_vehicle_negligence/R2"]]))
    return out, y_true_raw, y_pred_raw

def run_epoch(model, loader, opt=None, device="cuda"):
    is_train = opt is not None
    model.train(mode=is_train)
    losses = []
    y_all, p_all = [], []
    crit = nn.L1Loss()  # MAE on 0~1 space

    for x_idx, y in loader:
        x_idx = x_idx.to(device); y = y.to(device)
        pred = model(x_idx)
        loss = crit(pred, y)  # simple MAE

        if is_train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        losses.append(loss.item())
        y_all.append(y.detach().cpu().numpy())
        p_all.append(pred.detach().cpu().numpy())

    y_all = np.concatenate(y_all, axis=0) if y_all else np.zeros((0,2))
    p_all = np.concatenate(p_all, axis=0) if p_all else np.zeros((0,2))
    return float(np.mean(losses)), y_all, p_all

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train_csv",
        type=str,
        help="학습 CSV (columns: dashcam_vehicle_info, other_vehicle_info, ...negligence)",
        default="/app/data/raw/json/video-train/video_accident_caption_results_unsignalized_0811.csv",
    )
    ap.add_argument(
        "--val_csv",
        type=str,
        help="검증 CSV",
        default="/app/data/raw/json/video-evaluate/video_accident_caption_results_unsignalized_validation_0901.csv",
    )
    ap.add_argument("--dv_col", type=str, default="dashcam_vehicle_info")
    ap.add_argument("--ov_col", type=str, default="other_vehicle_info")
    ap.add_argument("--target_cols", type=str, nargs="+",
                    default=["dashcam_vehicle_negligence", "other_vehicle_negligence"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--project", type=str, default="video-fault-regression-nobert")
    ap.add_argument("--run_name", type=str, default=None)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # --- read train/val separately
    df_tr = pd.read_csv(args.train_csv)
    df_val = pd.read_csv(args.val_csv)

    need = [args.dv_col, args.ov_col] + args.target_cols
    miss_tr = [c for c in need if c not in df_tr.columns]
    miss_vl = [c for c in need if c not in df_val.columns]
    if miss_tr: raise KeyError(f"Missing columns in TRAIN CSV: {miss_tr}")
    if miss_vl: raise KeyError(f"Missing columns in VAL CSV: {miss_vl}")

    # drop rows with missing targets
    df_tr = df_tr.dropna(subset=args.target_cols).reset_index(drop=True)
    df_val = df_val.dropna(subset=args.target_cols).reset_index(drop=True)

    # --- build vocab & scale from (train+val) to avoid KeyError on unseen categories in val
    df_all_for_vocab = pd.concat([df_tr[[args.dv_col, args.ov_col]],
                                  df_val[[args.dv_col, args.ov_col]]], axis=0, ignore_index=True)
    tmp_vocab_builder = InfoOnlyDataset(
        pd.concat([df_tr[need], df_val[need]], axis=0, ignore_index=True),
        dv_col=args.dv_col, ov_col=args.ov_col, target_cols=tuple(args.target_cols)
    )
    cat_vocab = tmp_vocab_builder.cat_vocab
    scale_div = tmp_vocab_builder.scale_div  # np.array([div_dv, div_ov])

    # --- final datasets/loaders
    dtr = InfoOnlyDataset(df_tr, args.dv_col, args.ov_col, tuple(args.target_cols), cat_vocab=cat_vocab)
    dvl = InfoOnlyDataset(df_val, args.dv_col, args.ov_col, tuple(args.target_cols), cat_vocab=cat_vocab)
    tr_loader = DataLoader(dtr, batch_size=args.batch_size, shuffle=True, drop_last=False)
    vl_loader = DataLoader(dvl, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # --- model
    device = args.device if torch.cuda.is_available() else "cpu"
    model = CatEmbedRegressor(
        dv_vocab_size=len(cat_vocab["dv"]),
        ov_vocab_size=len(cat_vocab["ov"]),
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # --- wandb
    wandb.init(project=args.project, name=args.run_name, config=vars(args))
    wandb.config.update({
        "dv_vocab_size": len(cat_vocab["dv"]),
        "ov_vocab_size": len(cat_vocab["ov"]),
        "scale_div": scale_div.tolist(),
        "targets": args.target_cols
    })

    best_macro_r2 = -1e9
    for ep in range(1, args.epochs+1):
        tr_loss, _, _ = run_epoch(model, tr_loader, opt, device=device)
        vl_loss, y_true_01, y_pred_01 = run_epoch(model, vl_loader, opt=None, device=device)

        mts, y_true_raw, y_pred_raw = eval_metrics(y_true_01, y_pred_01, scale_div)
        log = {"epoch": ep, "train/MAE_01": tr_loss, "val/MAE_01": vl_loss}
        log.update({
            "val/dv/MAE": mts["dashcam_vehicle_negligence/MAE"],
            "val/dv/RMSE": mts["dashcam_vehicle_negligence/RMSE"],
            "val/dv/R2": mts["dashcam_vehicle_negligence/R2"],
            "val/ov/MAE": mts["other_vehicle_negligence/MAE"],
            "val/ov/RMSE": mts["other_vehicle_negligence/RMSE"],
            "val/ov/R2": mts["other_vehicle_negligence/R2"],
            "val/macro/MAE": mts["macro/MAE"],
            "val/macro/RMSE": mts["macro/RMSE"],
            "val/macro/R2": mts["macro/R2"],
        })
        wandb.log(log)

        if mts["macro/R2"] > best_macro_r2:
            best_macro_r2 = mts["macro/R2"]
            os.makedirs("checkpoints", exist_ok=True)
            path = "checkpoints/best_multi_emb.pt"
            torch.save({
                "model": model.state_dict(),
                "cat_vocab": cat_vocab,
                "scale_div": scale_div,
                "targets": args.target_cols,
                "epoch": ep,
                "macro_R2": best_macro_r2
            }, path)
            wandb.run.summary["best_macro_R2"] = best_macro_r2
            print(f"[CKPT] saved: {path} (macro R2={best_macro_r2:.4f})")

    print("Done.")

if __name__ == "__main__":
    main()
