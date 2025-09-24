# -*- coding: utf-8 -*-
"""
학습: JSON 라벨(과실비율 [dashcam, other]) 그대로 회귀 학습
설명: 평가/추론 시 RAG로 법적 근거(규정/판례/표준 과실표) + prior 비율/신뢰도(conf) + guardrail 경고를 첨부

필요 패키지:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # (CUDA 환경 예시)
  pip install transformers scikit-learn pandas numpy matplotlib
  pip install wandb  # (선택) 로깅

실행 예시:
  python train_rag_fault_ratio.py \
    --json_path ./data/raw/json/video_accident_ratio_unsignalized.json \
    --kb_path   ./data/kb/kb.csv \
    --out_dir   ./outputs \
    --epochs 5 --batch_size 8 --lr 2e-5 --model_name bert-base-uncased
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# --------------------------
# utils
# --------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def normalize_ratio_vec(x):
    """입력: 길이 2 배열/리스트. 합이 0이면 [50,50] 반환. 그 외 합=100으로 정규화."""
    x = np.array(x, dtype=np.float32)
    s = float(np.sum(x))
    if s <= 1e-6:
        return np.array([50.0, 50.0], dtype=np.float32)
    x = x / s * 100.0
    return np.clip(x, 0.0, 100.0)

# --------------------------
# Dataset
# --------------------------
class FaultRatioDataset(Dataset):
    """
    required columns:
      - generated_caption: str (설명 텍스트)
      - ratio: [dashcam, other] (합=100)
    """
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 보정: ratio 합이 100이 되도록 정규화
        self.df["ratio"] = self.df["ratio"].apply(normalize_ratio_vec)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        caption = row["generated_caption"]
        ratio = row["ratio"]  # numpy array length 2

        enc = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(ratio, dtype=torch.float32),
            "caption": caption  # 평가 시 RAG에 사용
        }
        return item

# --------------------------
# Model (회귀 전용. 라벨은 JSON 그대로)
# --------------------------
class TextToFaultRatio(nn.Module):
    def __init__(self, model_name="bert-base-uncased", hidden_dim=768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # [dash, other]
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # [CLS]
        return self.regressor(cls)

# --------------------------
# RAG (평가/추론 전용: 근거/표준비율/신뢰도)
# --------------------------
class LegalKB:
    def __init__(self, kb_df: pd.DataFrame):
        """
        kb_df columns (필수):
          - rule_id
          - scenario
          - evidence_text
          - base_ratio_dash
          - base_ratio_other
        """
        self.kb = kb_df.copy()
        for col in ["base_ratio_dash", "base_ratio_other"]:
            self.kb[col] = self.kb[col].astype(float)
        self.kb["doc"] = self.kb["scenario"].astype(str) + " " + self.kb["evidence_text"].astype(str)

        self.vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,2))
        self.X = self.vectorizer.fit_transform(self.kb["doc"].tolist())

    def explain(self, caption: str, k=3):
        if not isinstance(caption, str):
            caption = "" if caption is None else str(caption)

        q = self.vectorizer.transform([caption])
        sims = cosine_similarity(q, self.X).ravel()
        order = sims.argsort()[::-1]
        top_idx = order[:k]
        top = self.kb.iloc[top_idx]
        top_sims = sims[top_idx]

        conf = float(top_sims.max()) if len(top_sims) else 0.0
        w = top_sims / (top_sims.sum() + 1e-8) if len(top_sims) else np.array([1.0])

        prior_dash = float(np.sum(w * top["base_ratio_dash"].values)) if len(top) else 50.0
        prior_other = float(np.sum(w * top["base_ratio_other"].values)) if len(top) else 50.0
        prior = np.array([prior_dash, prior_other], dtype=np.float32)

        evidence = []
        rule_ids = []
        for _, r in top.iterrows():
            rule_ids.append(str(r["rule_id"]))
            evidence.append(f"[{r['rule_id']}] {r['scenario']} — {r['evidence_text']}")

        return {
            "evidence": evidence,         # 리스트
            "prior": prior,               # np.array([dash, other])
            "conf": conf,                 # 0~1 근사
            "rule_ids": rule_ids          # 리스트
        }

def legal_guardrail(pred, prior, tol=25.0):
    """
    pred/prior: 길이 2, 합≈100 가정
    tol 이상 벌어지면 경고 플래그 True
    """
    pred = normalize_ratio_vec(pred)
    prior = normalize_ratio_vec(prior)
    gap = float(np.max(np.abs(pred - prior)))
    return (gap >= tol), gap

# --------------------------
# Train / Evaluate
# --------------------------
def evaluate(model, dataloader, device, kb: LegalKB, out_dir, k=3, tol=25.0, use_wandb=False):
    model.eval()
    all_preds, all_labels = [], []
    rows = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            captions = batch["caption"]  # list of str

            preds = model(input_ids, attn).cpu().numpy()

            for i in range(len(preds)):
                cap = captions[i]
                ev = kb.explain(cap, k=k) if kb is not None else {"evidence": [], "prior": np.array([50,50]), "conf": 0.0, "rule_ids": []}
                prior = ev["prior"]
                conf = ev["conf"]
                rules = ev["rule_ids"]
                evidence_txt = ev["evidence"]

                flag, gap = legal_guardrail(preds[i], prior, tol=tol)

                rows.append({
                    "caption": cap,
                    "pred_dash": float(preds[i][0]),
                    "pred_other": float(preds[i][1]),
                    "gt_dash": float(labels[i][0]),
                    "gt_other": float(labels[i][1]),
                    "prior_dash": float(prior[0]),
                    "prior_other": float(prior[1]),
                    "conf": float(conf),
                    "gap_max": float(gap),
                    "guardrail_flag": bool(flag),
                    "rule_ids": "|".join(rules),
                    "evidence": " || ".join(evidence_txt)
                })

            all_preds.extend(preds)
            all_labels.extend(labels)

    all_preds = np.array(all_preds, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.float32)

    mae = mean_absolute_error(all_labels, all_preds)
    rmse = mean_squared_error(all_labels, all_preds, squared=False)
    r2 = r2_score(all_labels, all_preds)

    # 결과 테이블 저장
    df_out = pd.DataFrame(rows)
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "eval_with_evidence.csv")
    df_out.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"📄 RAG 근거 포함 평가 테이블 저장: {csv_path}")

    if use_wandb:
        try:
            import wandb
            table = wandb.Table(dataframe=df_out)
            wandb.log({"eval_table": table})
        except Exception:
            pass

    print(f"📊 Eval | MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
    return mae, rmse, r2, all_preds, all_labels

def plot_scatter(preds, labels, out_dir, use_wandb=False):
    preds = np.array(preds); labels = np.array(labels)
    ensure_dir(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    roles = ["Dashcam", "Other Vehicle"]
    for i, role in enumerate(roles):
        axes[i].scatter(labels[:, i], preds[:, i], alpha=0.5)
        axes[i].plot([0,100],[0,100],"--")
        axes[i].set_title(f"{role} Fault Prediction")
        axes[i].set_xlabel("True Ratio")
        axes[i].set_ylabel("Predicted Ratio")
        axes[i].set_xlim(0,100); axes[i].set_ylim(0,100)
    plt.tight_layout()
    png_path = os.path.join(out_dir, "prediction_scatter.png")
    plt.savefig(png_path, dpi=150)
    print(f"📈 산점도 저장: {png_path}")

    if use_wandb:
        try:
            import wandb
            wandb.log({"prediction_scatter": wandb.Image(png_path)})
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--kb_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--kb_topk", type=int, default=3)
    parser.add_argument("--guardrail_tol", type=float, default=25.0)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    # 데이터 로드 (JSON: record-orient 추천. 열: generated_caption(str), ratio([dash,other]))
    df = pd.read_json(args.json_path)
    if "generated_caption" not in df.columns or "ratio" not in df.columns:
        raise ValueError("JSON에는 'generated_caption'와 'ratio' 컬럼이 필요합니다.")

    # KB 로드
    kb_df = pd.read_csv(args.kb_path)
    kb = LegalKB(kb_df)

    # 데이터 분할
    train_df, eval_df = train_test_split(df, test_size=args.test_size, random_state=args.random_state)

    # 토크나이저/데이터로더
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = FaultRatioDataset(train_df, tokenizer, max_length=args.max_length)
    eval_ds  = FaultRatioDataset(eval_df,  tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=args.batch_size, shuffle=False)

    # 모델/옵티마이저/손실
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextToFaultRatio(model_name=args.model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()  # 회귀 라벨은 JSON 그대로

    # wandb 옵션
    use_wandb = False
    if args.use_wandb:
        try:
            import wandb
            wandb.init(project="fault_ratio_regression_with_RAG", config=vars(args))
            use_wandb = True
        except Exception:
            print("wandb 초기화 실패. wandb 미사용으로 진행합니다.")
            use_wandb = False

    # 학습
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)  # (B,2)

            preds = model(input_ids, attn)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += float(loss.item())

        avg = total / max(1, len(train_loader))
        print(f"✅ Epoch {epoch}/{args.epochs} | Train L1: {avg:.4f}")
        if use_wandb:
            try:
                import wandb
                wandb.log({"train_l1": avg, "epoch": epoch})
            except Exception:
                pass

        # 에폭 끝 평가 + RAG 근거 첨부
        mae, rmse, r2, preds, labels = evaluate(
            model, eval_loader, device, kb=kb,
            out_dir=args.out_dir, k=args.kb_topk,
            tol=args.guardrail_tol, use_wandb=use_wandb
        )
        if use_wandb:
            try:
                import wandb
                wandb.log({"eval_mae": mae, "eval_rmse": rmse, "eval_r2": r2, "epoch": epoch})
            except Exception:
                pass

    # 저장
    ckpt = os.path.join(args.out_dir, "fault_ratio_bert.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"💾 모델 저장: {ckpt}")

    # 시각화
    plot_scatter(preds, labels, out_dir=args.out_dir, use_wandb=use_wandb)

if __name__ == "__main__":
    main()
