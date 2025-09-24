# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoModel, AutoTokenizer
# from torch.optim import AdamW
# import pandas as pd
# import numpy as np
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import matplotlib.pyplot as plt
# import wandb
# import os

# df = pd.read_json("../data/raw/json/text-train/video_accident_ratio_unsignalized.json")

# # 🧠 모델 정의
# class TextToFaultRatio(nn.Module):
#     def __init__(self, model_name="bert-base-uncased", hidden_dim=768):
#         super().__init__()
#         self.encoder = AutoModel.from_pretrained(model_name)
#         self.regressor = nn.Sequential(
#             nn.Linear(hidden_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 2)
#         )

#     def forward(self, input_ids, attention_mask):
#         output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         cls_token = output.last_hidden_state[:, 0]
#         return self.regressor(cls_token)


# # 📦 Dataset 정의
# class FaultRatioDataset(Dataset):
#     def __init__(self, dataframe, tokenizer, max_length=256):
#         self.data = dataframe
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         text = row["generated_caption"]
#         ratio = torch.tensor(row["ratio"], dtype=torch.float)

#         encoding = self.tokenizer(
#             text,
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt"
#         )

#         return {
#             "input_ids": encoding["input_ids"].squeeze(0),
#             "attention_mask": encoding["attention_mask"].squeeze(0),
#             "labels": ratio
#         }


# # 🏋️‍♂️ 학습 함수
# def train_model(train_df, eval_df, model_name="bert-base-uncased", epochs=5, batch_size=8, lr=2e-5):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     wandb.init(project="0914_fault_ratio_regression", config={"lr": lr, "batch_size": batch_size, "epochs": epochs})

#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     train_dataset = FaultRatioDataset(train_df, tokenizer)
#     eval_dataset = FaultRatioDataset(eval_df, tokenizer)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

#     model = TextToFaultRatio(model_name=model_name).to(device)
#     optimizer = AdamW(model.parameters(), lr=lr)
#     loss_fn = nn.L1Loss()  # MAE 기반 L1 Loss

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0

#         for batch in train_loader:
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)

#             preds = model(input_ids, attention_mask)
#             loss = loss_fn(preds, labels)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         avg_loss = total_loss / len(train_loader)
#         wandb.log({"train_loss": avg_loss, "epoch": epoch + 1})
#         print(f"✅ Epoch {epoch + 1} | Avg L1 Loss: {avg_loss:.4f}")

#         # 💡 Epoch 끝나면 평가 및 로그
#         mae, rmse, r2, preds, labels = evaluate_model(model, eval_loader, device)
#         wandb.log({"eval_mae": mae, "eval_rmse": rmse, "eval_r2": r2, "epoch": epoch + 1})

#     # 모델 저장
#     torch.save(model.state_dict(), "fault_ratio_bert.pt")
#     print("📦 모델 저장 완료: fault_ratio_bert.pt")

#     # 시각화
#     visualize_predictions(preds, labels)


# # 🧪 평가 함수
# def evaluate_model(model, dataloader, device):
#     model.eval()
#     all_preds, all_labels = [], []

#     with torch.no_grad():
#         for batch in dataloader:
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].cpu().numpy()

#             preds = model(input_ids, attention_mask).cpu().numpy()

#             all_preds.extend(preds)
#             all_labels.extend(labels)

#     all_preds = np.array(all_preds)
#     all_labels = np.array(all_labels)

#     mae = mean_absolute_error(all_labels, all_preds)
#     rmse = mean_squared_error(all_labels, all_preds, squared=False)
#     r2 = r2_score(all_labels, all_preds)

#     print(f"📊 MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
#     return mae, rmse, r2, all_preds, all_labels


# # 📊 시각화 함수
# def visualize_predictions(preds, labels):
#     preds = np.array(preds)
#     labels = np.array(labels)

#     fig, ax = plt.subplots(1, 2, figsize=(12, 5))
#     for i, role in enumerate(["Dashcam", "Other Vehicle"]):
#         ax[i].scatter(labels[:, i], preds[:, i], alpha=0.5)
#         ax[i].plot([0, 100], [0, 100], "--", color="gray")
#         ax[i].set_title(f"{role} Fault Prediction")
#         ax[i].set_xlabel("True Ratio")
#         ax[i].set_ylabel("Predicted Ratio")
#         ax[i].set_xlim(0, 100)
#         ax[i].set_ylim(0, 100)
#     plt.tight_layout()
#     plt.savefig("prediction_scatter.png")
#     print("📈 산점도 저장 완료: prediction_scatter.png")
#     wandb.log({"prediction_scatter": wandb.Image("prediction_scatter.png")})

# from sklearn.model_selection import train_test_split

# # df는 `generated_caption`, `ratio` ([dashcam, other]) 컬럼 포함
# train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# train_model(train_df, eval_df, model_name="bert-base-uncased", epochs=5, batch_size=8)
# -*- coding: utf-8 -*-
"""
Fault-ratio regression from generated captions
- Input JSON columns (required):
    - generated_caption : str
    - dashcam_vehicle_negligence : number (0~100 or 0~1)
    - other_vehicle_negligence   : number (0~100 or 0~1)
- Output:
    - Trains BERT encoder + small MLP head to predict [dashcam, other] negligence
    - Saves model weights: fault_ratio_bert.pt
    - Saves scatter plot : prediction_scatter.png

Run examples:
    python train_fault_ratio.py \
        --data /app/data/raw/json/text-train/video_accident_ratio_unsignalized.json \
        --gpu 0 --epochs 5 --batch_size 8 --lr 2e-5

    # Only-GPU-3 (no code change way)
    CUDA_VISIBLE_DEVICES=3 python train_fault_ratio.py --gpu 0
"""
# -*- coding: utf-8 -*-
"""
Fault-ratio regression (default basis=10)
- Input JSON columns (required):
    - generated_caption : str
    - dashcam_vehicle_negligence : number (0~100 or 0~1)
    - other_vehicle_negligence   : number (0~100 or 0~1)

Default behavior:
- Labels are scaled so that 40/60 -> [4.0, 6.0] (sum = 10).
"""

import os
import math
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import wandb


# ---------------------------
# Utils
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_pair(p):
    """Normalize negligence pair to [0,100] each and sum==100."""
    a, b = float(p[0]), float(p[1])
    s = a + b
    # if probs (0~1), scale
    if 0 <= a <= 1 and 0 <= b <= 1:
        a *= 100.0
        b *= 100.0
        s = a + b
    # if sum differs from 100, renormalize
    if s != 0 and abs(s - 100.0) > 1e-6:
        a = a * (100.0 / s)
        b = b * (100.0 / s)
    a = max(0.0, min(100.0, a))
    b = max(0.0, min(100.0, b))
    return [a, b]


def load_and_prepare_dataframe(data_path: str) -> pd.DataFrame:
    df = pd.read_json(data_path)

    # caption column fallback
    if "generated_caption" not in df.columns:
        for c in ["caption", "pred_caption", "prediction", "text", "generated_text"]:
            if c in df.columns:
                df["generated_caption"] = df[c]
                break
    if "generated_caption" not in df.columns:
        raise KeyError("No caption column found. Need 'generated_caption' or one of "
                       "['caption','pred_caption','prediction','text','generated_text'].")

    # primary columns
    if {"dashcam_vehicle_negligence", "other_vehicle_negligence"}.issubset(df.columns):
        pair = df[["dashcam_vehicle_negligence", "other_vehicle_negligence"]].astype(float).values.tolist()
        df["ratio_pair"] = [normalize_pair(p) for p in pair]
    else:
        # fallback: ratio array/dict etc.
        if "ratio" in df.columns:
            vals = []
            for v in df["ratio"].tolist():
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    vals.append(normalize_pair(v))
                elif isinstance(v, dict):
                    for ka, kb in [("dashcam", "other"), ("a", "b")]:
                        if ka in v and kb in v:
                            vals.append(normalize_pair([v[ka], v[kb]]))
                            break
                    else:
                        raise KeyError(f"Unsupported ratio dict keys: {list(v.keys())}")
                else:
                    raise KeyError(f"Unsupported ratio type: {type(v)}")
            df["ratio_pair"] = vals
        else:
            raise KeyError("Missing negligence columns: "
                           "'dashcam_vehicle_negligence' & 'other_vehicle_negligence' (or 'ratio').")

    df = df.dropna(subset=["generated_caption", "ratio_pair"]).reset_index(drop=True)
    return df


# ---------------------------
# Model & Dataset
# ---------------------------
class TextToFaultRatio(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", hidden_dim: int = 768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(512, 128),
            nn.GELU(),

            nn.Linear(128, 2)  # logits(2)
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = out.last_hidden_state[:, 0]
        logits = self.regressor(cls_token)          # [B,2]
        probs  = self.softmax(logits)               # [B,2], sum=1
        pair   = probs * self.target_basis          # [B,2], sum=target_basis
        return pair


class FaultRatioDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: AutoTokenizer,
                 max_length: int = 256, scale_divisor: float = 10.0):
        """
        scale_divisor: how much to divide 0~100 labels by.
          - default 10.0 → 40/60 -> 4.0/6.0
          - if you want 0.4/0.6, use 100.0 (basis=1)
        """
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.scale_divisor = scale_divisor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        text = row["generated_caption"]
        # ratio_pair is 0~100 basis; divide by scale_divisor (default 10 -> [4,6])
        ratio = torch.tensor(row["ratio_pair"], dtype=torch.float) / self.scale_divisor

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": ratio,
        }


# ---------------------------
# Train / Eval / Viz
# ---------------------------
def evaluate_model(model, dataloader, device, target_basis: float, print_samples: int = 5):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            y = batch["labels"].cpu().numpy()      # already scaled to basis
            y_hat = model(input_ids, attention_mask).cpu().numpy()
            preds.extend(y_hat); labels.extend(y)

    preds = np.array(preds)
    labels = np.array(labels)

    mae = mean_absolute_error(labels, preds)
    try:
        rmse = mean_squared_error(labels, preds, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(labels, preds))
    r2 = r2_score(labels, preds)

    print(f"📊 (basis={target_basis}) MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")

    n_show = min(print_samples, len(labels))
    print(f"\n🔎 Sample predictions (총 {len(labels)}개 중 {n_show}개):")
    for i in range(n_show):
        gt = labels[i]
        pr = preds[i]
        print(f"[{i}] GT: Dashcam={gt[0]:.2f}, Other={gt[1]:.2f} "
              f"| Pred: Dashcam={pr[0]:.2f}, Other={pr[1]:.2f}")

    return mae, rmse, r2, preds, labels


def visualize_predictions(preds, labels, target_basis: float, out_path="prediction_scatter.png"):
    preds = np.array(preds)
    labels = np.array(labels)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for i, role in enumerate(["Dashcam", "Other Vehicle"]):
        ax[i].scatter(labels[:, i], preds[:, i], alpha=0.5)
        ax[i].plot([0, target_basis], [0, target_basis], "--", color="gray")
        ax[i].set_title(f"{role} Fault Prediction (basis={target_basis})")
        ax[i].set_xlabel("True")
        ax[i].set_ylabel("Pred")
        ax[i].set_xlim(0, target_basis)
        ax[i].set_ylim(0, target_basis)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"📈 산점도 저장 완료: {out_path}")
    try:
        wandb.log({"prediction_scatter": wandb.Image(out_path)})
    except Exception:
        pass


def train_model(
    df: pd.DataFrame,
    model_name: str = "bert-base-uncased",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 2e-5,
    device: torch.device = torch.device("cpu"),
    num_workers: int = 2,
    fp16: bool = False,
    grad_clip: float = 5.0,
    # ▼ 기본을 10으로: 40/60 -> 4/6
    target_basis: float = 10.0,
):
    """
    target_basis: labels range [0, target_basis], default 10.0
    """
    wandb.init(project=os.environ.get("WANDB_PROJECT", "fault_ratio_regression"),
               config={"lr": lr, "batch_size": batch_size, "epochs": epochs,
                       "model": model_name, "target_basis": target_basis})

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

    # ratio_pair is 0~100; to get basis=10, divide by 10
    scale_divisor = 100.0 / target_basis  # basis=10 -> 10, basis=100 -> 1, basis=1 -> 100

    train_ds = FaultRatioDataset(train_df, tokenizer, max_length=256, scale_divisor=scale_divisor)
    eval_ds  = FaultRatioDataset(eval_df,  tokenizer, max_length=256, scale_divisor=scale_divisor)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    model = TextToFaultRatio(model_name=model_name, target_basis=target_basis).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = nn.HuberLoss(delta=1.0)

    scaler = torch.cuda.amp.GradScaler(enabled=(fp16 and device.type == "cuda"))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    preds = model(input_ids, attention_mask)
                    loss = loss_fn(preds, labels)
                scaler.scale(loss).backward()
                if grad_clip is not None and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(input_ids, attention_mask)
                loss = loss_fn(preds, labels)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        wandb.log({"train_loss": avg_loss, "epoch": epoch})
        print(f"✅ Epoch {epoch} | Avg L1 Loss: {avg_loss:.4f}")

        mae, rmse, r2, preds, ytrue = evaluate_model(model, eval_loader, device, target_basis)
        wandb.log({"eval_mae": mae, "eval_rmse": rmse, "eval_r2": r2, "epoch": epoch})

    torch.save(model.state_dict(), "fault_ratio_bert.pt")
    print("📦 모델 저장 완료: fault_ratio_bert.pt")

    visualize_predictions(preds, ytrue, target_basis=target_basis, out_path="prediction_scatter.png")


# ---------------------------
# Main
# ---------------------------
# -*- coding: utf-8 -*-
"""
Fault-ratio regression from generated captions
기본 라벨 스케일: basis=10 (예: 40/60 -> [4.0, 6.0])

지원 입력:
- generated_caption (str)
- dashcam_vehicle_negligence (0~100 또는 0~1)
- other_vehicle_negligence   (0~100 또는 0~1)
또는 ratio=[a,b] / {"dashcam":a,"other":b}

사용 예:
1) 분리 파일/디렉토리/글롭:
   python train.py --train_data "/app/data/raw/json/text-train/train/*.json" \
                   --eval_data  "/app/data/raw/json/text-train/eval/*.json"

2) 단일 파일에서 자동 split:
   python train.py --data /app/data/raw/json/text-train/video_accident_ratio_unsignalized.json

3) 특정 GPU:
   CUDA_VISIBLE_DEVICES=3 python train.py --gpu 0
"""

import os
import glob
import math
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import wandb


# =========================
# Utils
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_pair(p):
    """Normalize negligence pair to [0,100] each and sum==100."""
    a, b = float(p[0]), float(p[1])
    s = a + b
    # 0~1 확률형이면 100 스케일
    if 0 <= a <= 1 and 0 <= b <= 1:
        a *= 100.0
        b *= 100.0
        s = a + b
    # 합 100으로 정규화
    if s != 0 and abs(s - 100.0) > 1e-6:
        a = a * (100.0 / s)
        b = b * (100.0 / s)
    a = max(0.0, min(100.0, a))
    b = max(0.0, min(100.0, b))
    return [a, b]


def _load_one_json(path: str) -> pd.DataFrame:
    df = pd.read_json(path)

    # caption fallback
    if "generated_caption" not in df.columns:
        for c in ["caption", "pred_caption", "prediction", "text", "generated_text"]:
            if c in df.columns:
                df["generated_caption"] = df[c]
                break
    if "generated_caption" not in df.columns:
        raise KeyError(f"[{path}] 'generated_caption' 컬럼이 없습니다.")

    # negligence → ratio_pair(0~100, 합100)
    if {"dashcam_vehicle_negligence", "other_vehicle_negligence"}.issubset(df.columns):
        pair = df[["dashcam_vehicle_negligence", "other_vehicle_negligence"]].astype(float).values.tolist()
        df["ratio_pair"] = [normalize_pair(p) for p in pair]
    elif "ratio" in df.columns:
        vals = []
        for v in df["ratio"].tolist():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                vals.append(normalize_pair(v))
            elif isinstance(v, dict):
                ok = False
                for ka, kb in [("dashcam", "other"), ("a", "b")]:
                    if ka in v and kb in v:
                        vals.append(normalize_pair([v[ka], v[kb]]))
                        ok = True
                        break
                if not ok:
                    raise KeyError(f"[{path}] ratio dict 키 미지원: {list(v.keys())}")
            else:
                raise KeyError(f"[{path}] ratio 타입 미지원: {type(v)}")
        df["ratio_pair"] = vals
    else:
        raise KeyError(f"[{path}] negligence 컬럼이 없습니다.")
    return df.dropna(subset=["generated_caption", "ratio_pair"]).reset_index(drop=True)


def load_and_prepare_dataframe(paths: str | list[str]) -> pd.DataFrame:
    """
    paths: 파일 경로(or 글롭 패턴) 문자열 또는 문자열 리스트.
           디렉터리를 주면 *.json 로드.
    """
    files = []
    def expand(p: str):
        if os.path.isdir(p):
            return sorted(glob.glob(os.path.join(p, "*.json")))
        g = glob.glob(p)
        return g if g else [p]

    if isinstance(paths, str):
        files = expand(paths)
    else:
        for p in paths:
            files += expand(p)

    if not files:
        raise FileNotFoundError(f"데이터 파일을 찾지 못했습니다: {paths}")

    dfs = [_load_one_json(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


# =========================
# Model & Dataset
# =========================
class TextToFaultRatio(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", hidden_dim: int = 768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = out.last_hidden_state[:, 0]
        return self.regressor(cls_token)


class FaultRatioDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: AutoTokenizer,
                 max_length: int = 256, scale_factor: float = 10.0):
        """
        scale_factor: 100/basis. (기본 basis=10 → scale_factor=10)
        0~100 라벨을 basis로 축소하여 학습 타깃으로 사용.
        """
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        text = row["generated_caption"]
        # 0~100 기준 라벨을 scale_factor로 나눔 (기본 100→10으로 축소: 40→4)
        ratio = torch.tensor(row["ratio_pair"], dtype=torch.float) / self.scale_factor

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": ratio,
        }


# =========================
# Train / Eval / Viz
# =========================
def evaluate_model(model, dataloader, device, target_basis: float, print_samples: int = 5):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            y = batch["labels"].cpu().numpy()          # 이미 basis 스케일
            y_hat = model(input_ids, attention_mask).cpu().numpy()
            preds.extend(y_hat); labels.extend(y)

    preds = np.array(preds)
    labels = np.array(labels)

    mae = mean_absolute_error(labels, preds)
    # 버전-무관 RMSE
    try:
        rmse = mean_squared_error(labels, preds, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(labels, preds))
    r2 = r2_score(labels, preds)

    print(f"📊 (basis={target_basis}) MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
    n_show = min(print_samples, len(labels))
    print(f"\n🔎 Sample predictions (총 {len(labels)}개 중 {n_show}개):")
    for i in range(n_show):
        gt = labels[i]
        pr = preds[i]
        print(f"[{i}] GT: Dashcam={gt[0]:.2f}, Other={gt[1]:.2f} "
              f"| Pred: Dashcam={pr[0]:.2f}, Other={pr[1]:.2f}")

    return mae, rmse, r2, preds, labels


def visualize_predictions(preds, labels, target_basis: float, out_path="prediction_scatter.png"):
    preds = np.array(preds)
    labels = np.array(labels)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for i, role in enumerate(["Dashcam", "Other Vehicle"]):
        ax[i].scatter(labels[:, i], preds[:, i], alpha=0.5)
        ax[i].plot([0, target_basis], [0, target_basis], "--", color="gray")  # 대각선
        ax[i].set_title(f"{role} Fault Prediction (basis={target_basis})")
        ax[i].set_xlabel("True")
        ax[i].set_ylabel("Pred")
        ax[i].set_xlim(0, target_basis)
        ax[i].set_ylim(0, target_basis)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"📈 산점도 저장 완료: {out_path}")
    try:
        wandb.log({"prediction_scatter": wandb.Image(out_path)})
    except Exception:
        pass


def train_model(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    model_name: str = "bert-base-uncased",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 2e-5,
    device: torch.device = torch.device("cpu"),
    num_workers: int = 2,
    fp16: bool = False,
    grad_clip: float = 5.0,
    target_basis: float = 10.0,  # 기본 10 → 40/60 -> 4/6
):
    wandb.init(project=os.environ.get("WANDB_PROJECT", "fault_ratio_regression"),
               config={"lr": lr, "batch_size": batch_size, "epochs": epochs,
                       "model": model_name, "target_basis": target_basis,
                       "train_rows": len(train_df), "eval_rows": len(eval_df)})

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 100 → basis 축소를 위한 factor
    scale_factor = 100.0 / target_basis  # basis=10 → factor=10
    train_ds = FaultRatioDataset(train_df, tokenizer, max_length=256, scale_factor=scale_factor)
    eval_ds  = FaultRatioDataset(eval_df,  tokenizer, max_length=256, scale_factor=scale_factor)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    model = TextToFaultRatio(model_name=model_name).to(device)
    # optimizer = AdamW(model.parameters(), lr=lr)
    head_lr_mul = 5.0
    optimizer = AdamW([
        {"params": model.encoder.parameters(),   "lr": lr,              "weight_decay": 0.01},
        {"params": model.regressor.parameters(), "lr": lr * head_lr_mul, "weight_decay": 0.01},
    ])
    loss_fn = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=(fp16 and device.type == "cuda"))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    preds = model(input_ids, attention_mask)
                    loss = loss_fn(preds, labels)
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(input_ids, attention_mask)
                loss = loss_fn(preds, labels)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        wandb.log({"train_loss": avg_loss, "epoch": epoch})
        print(f"✅ Epoch {epoch} | Avg L1 Loss: {avg_loss:.4f}")

        mae, rmse, r2, preds, ytrue = evaluate_model(model, eval_loader, device, target_basis)
        wandb.log({"eval_mae": mae, "eval_rmse": rmse, "eval_r2": r2, "epoch": epoch})

    torch.save(model.state_dict(), "fault_ratio_bert_3.pt")
    print("📦 모델 저장 완료: fault_ratio_bert.pt")
    visualize_predictions(preds, ytrue, target_basis=target_basis, out_path="prediction_scatter.png")


# =========================
# Args & Main
# =========================
def parse_args():
    p = argparse.ArgumentParser()

    # 분리 데이터 경로(파일/디렉토리/글롭)
    p.add_argument("--train_data", type=str, 
                   default="/app/data/raw/json/text-train/video_accident_ratio_unsignalized.json",
                   help="학습 데이터 파일/디렉토리/글롭 패턴")
    p.add_argument("--eval_data", type=str, 
                   default="/app/data/raw/json/text-evaluate/video_accident_ratio_training_results_trimmed_unsignalized_validation_0901.json",
                   help="평가 데이터 파일/디렉토리/글롭 패턴")

    # 백워드 호환: 단일 파일에서 split
    p.add_argument("--data", type=str,
                   default="/app/data/raw/json/text-train/video_accident_ratio_unsignalized.json",
                   help="단일 데이터 파일 (train/eval 미지정 시 사용)")

    p.add_argument("--model_name", type=str, default="bert-base-uncased")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index (after CUDA_VISIBLE_DEVICES)")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # 기본 10.0 → 40/60 -> 4/6
    p.add_argument("--target_basis", type=float, default=10.0,
                   help="Target label basis (default 10.0; 40/60 -> 4/6)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    # 데이터 로드
    if args.train_data and args.eval_data:
        train_df = load_and_prepare_dataframe(args.train_data)
        eval_df  = load_and_prepare_dataframe(args.eval_data)
    else:
        df = load_and_prepare_dataframe(args.data)
        train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

    print("Train rows:", len(train_df), "Eval rows:", len(eval_df))
    print(train_df[["generated_caption", "ratio_pair"]].head(2).to_dict(orient="records"))

    train_model(
        train_df=train_df,
        eval_df=eval_df,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        num_workers=args.num_workers,
        fp16=args.fp16,
        target_basis=args.target_basis,  # 기본 10.0
    )

# # -*- coding: utf-8 -*-
# """
# Fault-ratio regression from generated captions (BERT + 2D head + softmax + Huber + joint calibration)
# - Inputs (JSON):
#     - generated_caption : str
#     - dashcam_vehicle_negligence : number (0~100 or 0~1)
#     - other_vehicle_negligence   : number (0~100 or 0~1)
#   or ratio=[a,b] / {"dashcam":a,"other":b}

# Core choices:
# - 2D head -> logits -> (vector scaling: cal_w, cal_b) -> softmax -> p in [0,1]^2 (sum=1) -> *target_basis*
# - HuberLoss(delta≈1.2) on basis scale
# - Sum constraint & non-negativity are enforced by design

# Run examples:
#     python train_fault_ratio.py \
#       --train_data "/app/data/raw/json/text-train/video_accident_ratio_unsignalized.json" \
#       --eval_data  "/app/data/raw/json/text-evaluate/video_accident_ratio_training_results_trimmed_unsignalized_validation_0901.json" \
#       --epochs 10 --batch_size 16 --lr 2e-5 --target_basis 10

#     # single file w/ internal split
#     python train_fault_ratio.py --data /app/data/raw/json/text-train/video_accident_ratio_unsignalized.json
# """

# import os, re, glob, math, argparse, random, unicodedata
# from typing import List, Dict, Any

# import numpy as np
# import pandas as pd

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

# from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
# from torch.optim import AdamW

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# import matplotlib.pyplot as plt
# import wandb


# # =========================
# # Utils
# # =========================
# def set_seed(seed: int = 42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# def normalize_caption(t: str) -> str:
#     # 유니코드/공백 정규화 -> 토크나이저 노이즈 최소화
#     t = unicodedata.normalize("NFKC", str(t))
#     t = re.sub(r"\s+", " ", t).strip()
#     return t


# def normalize_pair(p):
#     """Normalize negligence pair to [0,100] each and sum==100."""
#     a, b = float(p[0]), float(p[1])
#     s = a + b
#     if 0 <= a <= 1 and 0 <= b <= 1:
#         a *= 100.0
#         b *= 100.0
#         s = a + b
#     if s != 0 and abs(s - 100.0) > 1e-6:
#         a = a * (100.0 / s)
#         b = b * (100.0 / s)
#     a = max(0.0, min(100.0, a))
#     b = max(0.0, min(100.0, b))
#     return [a, b]


# def _load_one_json(path: str) -> pd.DataFrame:
#     df = pd.read_json(path)

#     # caption fallback
#     if "generated_caption" not in df.columns:
#         for c in ["caption", "pred_caption", "prediction", "text", "generated_text"]:
#             if c in df.columns:
#                 df["generated_caption"] = df[c]
#                 break
#     if "generated_caption" not in df.columns:
#         raise KeyError(f"[{path}] 'generated_caption' column is missing.")

#     # negligence → ratio_pair(0~100, 합100)
#     if {"dashcam_vehicle_negligence", "other_vehicle_negligence"}.issubset(df.columns):
#         pair = df[["dashcam_vehicle_negligence", "other_vehicle_negligence"]].astype(float).values.tolist()
#         df["ratio_pair"] = [normalize_pair(p) for p in pair]
#     elif "ratio" in df.columns:
#         vals = []
#         for v in df["ratio"].tolist():
#             if isinstance(v, (list, tuple)) and len(v) == 2:
#                 vals.append(normalize_pair(v))
#             elif isinstance(v, dict):
#                 ok = False
#                 for ka, kb in [("dashcam", "other"), ("a", "b")]:
#                     if ka in v and kb in v:
#                         vals.append(normalize_pair([v[ka], v[kb]]))
#                         ok = True
#                         break
#                 if not ok:
#                     raise KeyError(f"[{path}] unsupported ratio dict keys: {list(v.keys())}")
#             else:
#                 raise KeyError(f"[{path}] unsupported ratio type: {type(v)}")
#         df["ratio_pair"] = vals
#     else:
#         raise KeyError(f"[{path}] negligence/ratio columns are missing.")

#     # 정규화된 캡션 컬럼
#     df["generated_caption"] = df["generated_caption"].map(normalize_caption)
#     return df.dropna(subset=["generated_caption", "ratio_pair"]).reset_index(drop=True)


# def load_and_prepare_dataframe(paths: str | List[str]) -> pd.DataFrame:
#     def expand(p: str):
#         if os.path.isdir(p):
#             return sorted(glob.glob(os.path.join(p, "*.json")))
#         g = glob.glob(p)
#         return g if g else [p]

#     files = []
#     if isinstance(paths, str):
#         files = expand(paths)
#     else:
#         for p in paths:
#             files += expand(p)

#     if not files:
#         raise FileNotFoundError(f"No data files: {paths}")

#     dfs = [_load_one_json(f) for f in files]
#     return pd.concat(dfs, ignore_index=True)


# # =========================
# # Model & Dataset
# # =========================
# class Softmax2Head(nn.Module):
#     """
#     2D logits -> (vector scaling) -> softmax -> p -> *target_basis
#     - Sum constraint & non-negativity enforced by design.
#     - cal_w, cal_b : joint calibration parameters learned together.
#     """
#     def __init__(self, model_name: str = "bert-base-uncased", hidden_dim: int = 768, target_basis: float = 10.0):
#         super().__init__()
#         self.encoder = AutoModel.from_pretrained(model_name)
#         self.head = nn.Linear(hidden_dim, 2)      # raw logits
#         self.cal_w = nn.Parameter(torch.ones(2))  # vector scaling (joint calibration)
#         self.cal_b = nn.Parameter(torch.zeros(2))
#         self.target_basis = float(target_basis)

#     def forward(self, input_ids, attention_mask, return_all: bool = False):
#         out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         cls = out.last_hidden_state[:, 0]             # [B, H]
#         logits = self.head(cls)                       # [B, 2]
#         logits = logits * self.cal_w + self.cal_b     # joint calibration
#         p = torch.softmax(logits, dim=-1)             # [B, 2], sum=1
#         pair = p * self.target_basis                  # [B, 2], sum=target_basis
#         if return_all:
#             return pair, p, logits
#         return pair


# class FaultRatioDataset(Dataset):
#     """
#     ratio_pair: [dashcam, other] in 0~100 with sum=100
#     We train on basis scale labels: label_basis = ratio_pair / (100 / target_basis)
#     """
#     def __init__(self, dataframe: pd.DataFrame, tokenizer: AutoTokenizer,
#                  max_length: int = 256, target_basis: float = 10.0):
#         self.data = dataframe
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.scale_factor = 100.0 / float(target_basis)  # e.g. basis=10 -> factor=10

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx: int):
#         row = self.data.iloc[idx]
#         text = row["generated_caption"]
#         ratio_basis = (torch.tensor(row["ratio_pair"], dtype=torch.float) / self.scale_factor)  # [2], sum=target_basis

#         enc = self.tokenizer(
#             text,
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt",
#         )
#         return {
#             "input_ids": enc["input_ids"].squeeze(0),
#             "attention_mask": enc["attention_mask"].squeeze(0),
#             "labels": ratio_basis,
#         }


# # =========================
# # Train / Eval / Viz
# # =========================
# def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device, target_basis: float):
#     model.eval()
#     preds, labels = [], []
#     with torch.no_grad():
#         for batch in dataloader:
#             input_ids = batch["input_ids"].to(device, non_blocking=True)
#             attention_mask = batch["attention_mask"].to(device, non_blocking=True)
#             y = batch["labels"].cpu().numpy()
#             y_hat = model(input_ids, attention_mask).cpu().numpy()
#             preds.extend(y_hat); labels.extend(y)

#     preds = np.array(preds)
#     labels = np.array(labels)

#     mae = mean_absolute_error(labels, preds)
#     try:
#         rmse = mean_squared_error(labels, preds, squared=False)
#     except TypeError:
#         rmse = np.sqrt(mean_squared_error(labels, preds))
#     r2 = r2_score(labels, preds)

#     # sum check (diagnostics)
#     sum_err = float(np.abs(preds.sum(axis=1) - target_basis).mean())

#     print(f"📊 (basis={target_basis}) MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f} | mean |sum-"
#           f"basis|: {sum_err:.4f}")
#     return mae, rmse, r2, preds, labels, sum_err


# def visualize_predictions(preds, labels, target_basis: float, out_path="prediction_scatter.png"):
#     preds = np.array(preds)
#     labels = np.array(labels)
#     fig, ax = plt.subplots(1, 2, figsize=(12, 5))
#     for i, role in enumerate(["Dashcam", "Other Vehicle"]):
#         ax[i].scatter(labels[:, i], preds[:, i], alpha=0.5)
#         ax[i].plot([0, target_basis], [0, target_basis], "--", color="gray")
#         ax[i].set_title(f"{role} Fault Prediction (basis={target_basis})")
#         ax[i].set_xlabel("True")
#         ax[i].set_ylabel("Pred")
#         ax[i].set_xlim(0, target_basis)
#         ax[i].set_ylim(0, target_basis)
#     plt.tight_layout()
#     plt.savefig(out_path)
#     print(f"📈 산점도 저장: {out_path}")
#     try:
#         wandb.log({"prediction_scatter": wandb.Image(out_path)})
#     except Exception:
#         pass


# def train_model(
#     train_df: pd.DataFrame,
#     eval_df: pd.DataFrame,
#     model_name: str = "bert-base-uncased",
#     epochs: int = 5,
#     batch_size: int = 8,
#     lr: float = 2e-5,
#     device: torch.device = torch.device("cpu"),
#     num_workers: int = 2,
#     fp16: bool = False,
#     grad_clip: float = 1.0,
#     target_basis: float = 10.0,
#     warmup_ratio: float = 0.1,
#     freeze_layers: int = 0,   # e.g., 0~6 (freeze first N encoder layers)
# ):
#     wandb.init(project=os.environ.get("WANDB_PROJECT", "fault_ratio_regression"),
#                config={"lr": lr, "batch_size": batch_size, "epochs": epochs,
#                        "model": model_name, "target_basis": target_basis,
#                        "train_rows": len(train_df), "eval_rows": len(eval_df)})

#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     train_ds = FaultRatioDataset(train_df, tokenizer, max_length=256, target_basis=target_basis)
#     eval_ds  = FaultRatioDataset(eval_df,  tokenizer, max_length=256, target_basis=target_basis)

#     train_loader = DataLoader(
#         train_ds, batch_size=batch_size, shuffle=True,
#         num_workers=num_workers, pin_memory=True, drop_last=False
#     )
#     eval_loader = DataLoader(
#         eval_ds, batch_size=batch_size, shuffle=False,
#         num_workers=num_workers, pin_memory=True, drop_last=False
#     )

#     model = Softmax2Head(model_name=model_name, target_basis=target_basis).to(device)

#     # (선택) 일부 레이어 freeze
#     if freeze_layers > 0 and hasattr(model.encoder, "encoder"):
#         try:
#             for i, layer in enumerate(model.encoder.encoder.layer):
#                 if i < freeze_layers:
#                     for p in layer.parameters():
#                         p.requires_grad = False
#             print(f"🔒 Freeze first {freeze_layers} encoder layers.")
#         except Exception:
#             pass

#     optimizer = AdamW(model.parameters(), lr=lr)
#     steps_total = len(train_loader) * epochs
#     num_warmup_steps = int(steps_total * warmup_ratio)
#     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, steps_total)

#     loss_fn = nn.HuberLoss(delta=1.2)   # 극단 오차 완화
#     scaler = torch.cuda.amp.GradScaler(enabled=(fp16 and device.type == "cuda"))

#     # sanity log: parameter counts
#     n_all = sum(p.numel() for p in model.parameters())
#     n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Params: total={n_all/1e6:.2f}M / trainable={n_trainable/1e6:.2f}M")

#     best_mae = float("inf")
#     last_preds, last_labels = None, None

#     for epoch in range(1, epochs + 1):
#         model.train()
#         total_loss = 0.0

#         for batch in train_loader:
#             input_ids = batch["input_ids"].to(device, non_blocking=True)
#             attention_mask = batch["attention_mask"].to(device, non_blocking=True)
#             labels = batch["labels"].to(device, non_blocking=True)  # basis scale

#             optimizer.zero_grad(set_to_none=True)
#             if scaler.is_enabled():
#                 with torch.cuda.amp.autocast():
#                     preds = model(input_ids, attention_mask)  # [B,2], sum=basis
#                     loss = loss_fn(preds, labels)
#                 scaler.scale(loss).backward()
#                 if grad_clip and grad_clip > 0:
#                     scaler.unscale_(optimizer)
#                     nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 preds = model(input_ids, attention_mask)
#                 loss = loss_fn(preds, labels)
#                 loss.backward()
#                 if grad_clip and grad_clip > 0:
#                     nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
#                 optimizer.step()

#             scheduler.step()
#             total_loss += loss.item()

#         avg_loss = total_loss / max(1, len(train_loader))
#         wandb.log({"train_loss": avg_loss, "epoch": epoch})
#         print(f"✅ Epoch {epoch}/{epochs} | Avg Huber Loss: {avg_loss:.4f}")

#         mae, rmse, r2, preds, ytrue, sum_err = evaluate_model(model, eval_loader, device, target_basis)
#         wandb.log({"eval_mae": mae, "eval_rmse": rmse, "eval_r2": r2,
#                    "eval_mean_sum_err": sum_err, "epoch": epoch})

#         last_preds, last_labels = preds, ytrue
#         if mae < best_mae:
#             best_mae = mae
#             torch.save(model.state_dict(), "fault_ratio_bert_best.pt")
#             print("💾 Saved best: fault_ratio_bert_best.pt (MAE={:.4f})".format(best_mae))

#     # 최종 저장
#     torch.save(model.state_dict(), "fault_ratio_bert_2.pt")
#     print("📦 Saved final: fault_ratio_bert.pt")

#     # 시각화
#     if last_preds is not None and last_labels is not None:
#         visualize_predictions(last_preds, last_labels, target_basis=target_basis, out_path="prediction_scatter.png")


# # =========================
# # Args & Main
# # =========================
# def parse_args():
#     p = argparse.ArgumentParser()
#     # 분리 데이터 경로(파일/디렉토리/글롭)
#     p.add_argument("--train_data", type=str,
#                    default="/app/data/raw/json/text-train/video_accident_ratio_unsignalized.json",
#                    help="train file/dir/glob")
#     p.add_argument("--eval_data", type=str,
#                    default="/app/data/raw/json/text-evaluate/video_accident_ratio_training_results_trimmed_unsignalized_validation_0901.json",
#                    help="eval file/dir/glob")

#     # 단일 파일에서 split (백워드 호환)
#     p.add_argument("--data", type=str,
#                    default="",
#                    help="single dataset file (used if train/eval not provided)")

#     p.add_argument("--model_name", type=str, default="bert-base-uncased")
#     p.add_argument("--epochs", type=int, default=50)
#     p.add_argument("--batch_size", type=int, default=16)
#     p.add_argument("--lr", type=float, default=2e-5)
#     p.add_argument("--gpu", type=int, default=0)
#     p.add_argument("--num_workers", type=int, default=1)
#     p.add_argument("--fp16", action="store_true")
#     p.add_argument("--seed", type=int, default=42)

#     # 문제 스케일
#     p.add_argument("--target_basis", type=float, default=10.0,
#                    help="Target label basis (default 10.0; 40/60 -> 4/6)")

#     # 훈련 안정화 옵션
#     p.add_argument("--warmup_ratio", type=float, default=0.1)
#     p.add_argument("--freeze_layers", type=int, default=0,
#                    help="Freeze first N encoder layers (0 to disable)")
#     return p.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     set_seed(args.seed)

#     if torch.cuda.is_available():
#         torch.cuda.set_device(args.gpu)
#         device = torch.device(f"cuda:{args.gpu}")
#     else:
#         device = torch.device("cpu")

#     # 데이터 로드
#     if args.train_data and args.eval_data and args.train_data != "" and args.eval_data != "":
#         train_df = load_and_prepare_dataframe(args.train_data)
#         eval_df  = load_and_prepare_dataframe(args.eval_data)
#     else:
#         if not args.data:
#             raise SystemExit("Provide --data or both --train_data and --eval_data.")
#         df = load_and_prepare_dataframe(args.data)
#         train_df, eval_df = train_test_split(df, test_size=0.2, random_state=args.seed)

#     print("Train rows:", len(train_df), "Eval rows:", len(eval_df))
#     print(train_df[["generated_caption", "ratio_pair"]].head(2).to_dict(orient="records"))

#     train_model(
#         train_df=train_df,
#         eval_df=eval_df,
#         model_name=args.model_name,
#         epochs=args.epochs,
#         batch_size=args.batch_size,
#         lr=args.lr,
#         device=device,
#         num_workers=args.num_workers,
#         fp16=args.fp16,
#         target_basis=args.target_basis,
#         warmup_ratio=args.warmup_ratio,
#         freeze_layers=args.freeze_layers,
#     )
