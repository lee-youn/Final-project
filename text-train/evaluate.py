# ===== fault_pipeline.py =====
# 평가 파이프라인: JSON의 캡션은 절대 쓰지 않고, 항상 비디오 -> Video-LLaVA 캡션 생성 -> 텍스트→과실 예측
# - W&B 로깅
# - 각 샘플 결과 CSV 저장
# - 스캐터/히스토그램 이미지 저장(+ W&B 업로드)
# - tqdm 진행 표시 + per-sample 터미널 로그
# - GPU 2개 분담(캡션/과실 모델 디바이스 분리) + 선택적 샤딩

import os, re, json, glob, argparse
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 서버/컨테이너 환경에서 그림 저장
import matplotlib.pyplot as plt
from PIL import Image
import wandb
from tqdm.auto import tqdm
import cv2  # 비디오 프레임 추출용
from transformers import AutoModel, AutoTokenizer

# 네 기존 모듈 (Video-LLaVA 래퍼)
from train_tracking_video_llava import VideoLLaVACaptioner


# ------------------------------
# (A) Video frame sampling
# ------------------------------
# def sample_frames_pil(video_path: str, num_frames: int = 8, size: int = 224) -> List[Image.Image]:
#     cap = cv2.VideoCapture(video_path)
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
#     if total <= 0:
#         cap.release()
#         raise RuntimeError(f"Empty or unreadable video: {video_path}")

#     if total <= num_frames:
#         idxs = list(range(total))
#     else:
#         step = total / (num_frames + 1)
#         idxs = [int(step * (i + 1)) for i in range(num_frames)]

#     frames: List[Image.Image] = []
#     for idx in idxs:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total - 1))
#         ok, fr = cap.read()
#         if not ok:
#             # fallback: 마지막 프레임 복제 또는 검은 화면
#             if frames:
#                 frames.append(frames[-1])
#             else:
#                 frames.append(Image.new("RGB", (size, size), color=(0, 0, 0)))
#             continue
#         fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
#         fr = cv2.resize(fr, (size, size))
#         frames.append(Image.fromarray(fr))
#     cap.release()

#     while len(frames) < num_frames:
#         frames.append(frames[-1])
#     return frames[:num_frames]


# ------------------------------
# (B) Video-LLaVA load & caption
# ------------------------------
# def load_videollava_from_ckpt(ckpt_path: str,
#                               model_id: str = "LanguageBind/Video-LLaVA-7B-hf",
#                               dtype=torch.float16,
#                               device: str = "cuda"):
#     model = VideoLLaVACaptioner(model_id=model_id, lora=False, force_torch_dtype=dtype)
#     ckpt = torch.load(ckpt_path, map_location="cpu")
#     if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
#         missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
#         if missing or unexpected:
#             print("[Video-LLaVA] load_state_dict note:",
#                   f"missing={len(missing)} unexpected={len(unexpected)}")
#     model.to(device).eval()
#     return model


def postprocess_one_sentence(text: str) -> str:
    t = text.strip()
    parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', t) if p.strip()]
    seen, cleaned = set(), []
    for p in parts:
        key = re.sub(r'\s+', ' ', p.lower())
        if key not in seen:
            cleaned.append(p); seen.add(key)
    t = ' '.join(cleaned)
    idx = t.lower().find("seconds.")
    if idx != -1:
        return t[:idx + len("seconds.")].strip()
    m = re.split(r'(?<=[.!?])\s+', t)
    return (m[0].strip() if m else t)


# @torch.no_grad()
# def generate_caption_from_video(vl_model,
#                                 video_path: str,
#                                 prompt: str,
#                                 num_frames: int = 8,
#                                 size: int = 224,
#                                 max_new_tokens: int = 128) -> str:
#     frames_pil = sample_frames_pil(video_path, num_frames=num_frames, size=size)
#     out = vl_model(frames_pil=frames_pil, src_text=prompt,
#                    generate=True, max_new_tokens=max_new_tokens)
#     raw = (out.get("text", "") if isinstance(out, dict) else str(out))
#     return postprocess_one_sentence(raw)


# ------------------------------
# (C) Text -> Fault model
# ------------------------------
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
        cls = out.last_hidden_state[:, 0]
        return self.regressor(cls)


def load_fault_model(path: str,
                     model_name: str = "bert-base-uncased",
                     device: str = "cuda"):
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
def predict_fault_ratio(model, tokenizer, text: str, device: str = "cuda", max_length: int = 256) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt",
                       padding="max_length", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pred = model(input_ids=input_ids, attention_mask=attention_mask)  # [1,2]
    return pred.squeeze(0).float().cpu().numpy()  # basis 스케일(기본 10)


# ------------------------------
# (D) Label utils & video resolver
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
    factor = 100.0 / target_basis  # 10이면 /10
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
# (E) Metrics, plots, saving
# ------------------------------
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

    # 1) 스캐터 (대시캠/상대차량)
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

    # 2) 절대오차 히스토그램
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


def df_to_csv(results: List[dict], out_csv_path: str):
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True
               )
    df = pd.DataFrame(results)
    df.to_csv(out_csv_path, index=False, encoding="utf-8")


# ------------------------------
# (F) Evaluation (JSON 캡션 절대 사용 X)
# ------------------------------
def evaluate_on_json(eval_json_path: str,
                     fault_model_path: str,
                     out_json_path: str,
                     model_name: str = "bert-base-uncased",
                     target_basis: float = 10.0,
                     video_root: str = None,
                     vl_ckpt_path: str = None,
                     max_new_tokens: int = 128,
                     num_frames: int = 8,
                     size: int = 224,
                     verbose: bool = True,
                     print_every: int = 1,
                     use_tqdm: bool = True,
                     device_caption: str = "cuda:0",
                     device_fault: str = "cuda:0",
                     shard_idx: int = -1,
                     num_shards: int = -1):
    # 입력 검증
    if not os.path.exists(eval_json_path):
        raise FileNotFoundError(eval_json_path)
    if not os.path.exists(vl_ckpt_path or ""):
        raise FileNotFoundError(vl_ckpt_path)
    if not os.path.isdir(video_root or ""):
        raise NotADirectoryError(video_root)

    # W&B 세팅
    wandb.init(
        project="only-bert-accident-caption",
        config={
            "eval_json": eval_json_path,
            "fault_ckpt": fault_model_path,
            "video_root": video_root,
            "vl_ckpt": vl_ckpt_path,
            "model_name": model_name,
            "target_basis": target_basis,
            "num_frames": num_frames,
            "size": size,
            "max_new_tokens": max_new_tokens,
            "device_caption": device_caption,
            "device_fault": device_fault,
            "shard_idx": shard_idx,
            "num_shards": num_shards,
        },
        job_type="evaluation",
    )

    # # 모델 로드 (GPU 분담)
    # vl_model = load_videollava_from_ckpt(vl_ckpt_path, device=device_caption)
    fr_model, fr_tok = load_fault_model(fault_model_path, model_name=model_name, device=device_fault)

    # 데이터 적재 (+샤딩)
    with open(eval_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("eval_json must be a list of objects.")
    if shard_idx >= 0 and num_shards > 0:
        idxs = np.array_split(np.arange(len(data)), num_shards)[shard_idx].tolist()
        data = [data[i] for i in idxs]

    caption_prompt = (
        "Task: Describe the accident scene in one sentence.\n"
        "You MUST include:\n"
        "- intersection type\n"
        "- the movement and direction of both vehicles (Dashcam and Other)\n"
        "- entry order (who entered first)\n"
        "Avoid adding traffic signals or unrelated details."
    )

    preds_basis, labels_basis = [], []
    results = []

    N = len(data)
    iterator = tqdm(range(N), total=N, desc="Evaluating", dynamic_ncols=True) if use_tqdm else range(N)

    for i in iterator:
        row = data[i]

        # GT (metrics용)
        gt_basis = None
        gt100 = None
        if "dashcam_vehicle_negligence" in row and "other_vehicle_negligence" in row:
            gt100 = normalize_pair_100([row["dashcam_vehicle_negligence"], row["other_vehicle_negligence"]])
            gt_basis = np.array(to_basis(gt100, target_basis), dtype=float)

        # 비디오 찾기
        video_name = row.get("video_name") or row.get("video_path") or ""
        vpath = find_video_file(video_root, video_name) if video_name else None
        if not vpath:
            raw = row.get("video_path")
            if raw and os.path.exists(raw):
                vpath = raw
        if not vpath:
            out_item = {"idx": i, "video_name": video_name, "error": "video_not_found"}
            results.append(out_item)
            if verbose:
                print(f"[{i+1}/{N}] {video_name} | VIDEO NOT FOUND", flush=True)
            continue

        # 캡션 생성(항상 비디오에서)
        # try:
        #     caption = generate_caption_from_video(
        #         vl_model, vpath, prompt=caption_prompt,
        #         num_frames=num_frames, size=size, max_new_tokens=max_new_tokens
        #     )
        # except Exception as e:
        #     out_item = {"idx": i, "video_name": video_name, "error": f"caption_generation_failed: {e}"}
        #     results.append(out_item)
        #     if verbose:
        #         print(f"[{i+1}/{N}] {video_name} | CAPTION ERROR: {e}", flush=True)
        #     continue

        # 예측

        tgt_text = str(row.get('generated_caption' or "No description available").strip())
        m = re.search(r"\bthe collision\b", tgt_text, flags=re.IGNORECASE)
        if m:
                tgt_text = tgt_text[:m.start()].rstrip()
                # 끝이 .!? 로 안 끝나면 마침표 하나 붙여 깔끔하게
                if not tgt_text.endswith(('.', '!', '?')):
                    tgt_text += '.'
        
        pred_basis = predict_fault_ratio(fr_model, fr_tok, tgt_text, device=device_fault)  # basis 스케일
        caption = tgt_text
        preds_basis.append(pred_basis)
        if gt_basis is not None:
            labels_basis.append(gt_basis)

        # 보기 편하게 0~100도 함께 기록
        pred_100 = [float(x) * (100.0 / target_basis) for x in pred_basis]

        out_item = {
            "idx": i,
            "video_name": video_name,
            "caption": caption,  # 생성된 캡션
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

        results.append(out_item)

        # per-sample 로그
        if verbose and ( (i == 0) or ((i + 1) % print_every == 0) or (i + 1 == N) ):
            msg = f"[{i+1}/{N}] {video_name} | pred_basis=[{out_item['pred_basis_dashcam']:.2f}, {out_item['pred_basis_other']:.2f}]"
            msg += "\n"
            msg += f"⭐️gt_caption: {tgt_text}"
            if gt_basis is not None:
                ae_dc = out_item["abs_err_basis_dashcam"]
                ae_ov = out_item["abs_err_basis_other"]
                msg += f" | gt_basis=[{out_item['gt_basis_dashcam']:.2f}, {out_item['gt_basis_other']:.2f}] | abs_err=[{ae_dc:.2f}, {ae_ov:.2f}]"

            print(msg, flush=True)

        # 배치/스텝 로깅(선택)
        if (i + 1) % 20 == 0:
            wandb.log({"eval_progress_samples": i + 1})

    # 메트릭 & 저장
    metrics = {}
    out_prefix = os.path.splitext(out_json_path)[0]
    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)

    if labels_basis:
        y = np.vstack(labels_basis)
        yhat = np.vstack(preds_basis)
        metrics = compute_metrics(y, yhat)
        metrics["target_basis"] = target_basis

        # 플롯 저장
        plot_paths = save_plots(y, yhat, target_basis, out_prefix)

        # CSV 저장
        out_csv_path = f"{out_prefix}.csv"
        df_to_csv(results, out_csv_path)

        # JSON 저장
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "results": results}, f, ensure_ascii=False, indent=2)

        # W&B 로깅
        wandb.log({
            "eval/MAE": metrics["MAE"],
            "eval/RMSE": metrics["RMSE"],
            "eval/R2": metrics["R2"],
            "eval/MAE_dashcam": metrics["MAE_dashcam"],
            "eval/MAE_other": metrics["MAE_other"],
        })
        # 이미지 업로드
        for k, pth in plot_paths.items():
            if os.path.exists(pth):
                wandb.log({f"plots/{k}": wandb.Image(pth)})

        # CSV/JSON/이미지 아티팩트 업로드
        try:
            table = wandb.Table(dataframe=pd.DataFrame(results))
            wandb.log({"eval/table": table})
        except Exception:
            pass
        try:
            art = wandb.Artifact("eval_results", type="evaluation")
            art.add_file(out_json_path)
            art.add_file(out_csv_path)
            for p in plot_paths.values():
                if os.path.exists(p):
                    art.add_file(p)
            wandb.log_artifact(art)
        except Exception as e:
            print("W&B artifact upload failed:", e)
    else:
        # GT가 전혀 없으면 결과만 CSV/JSON 저장(메트릭 없음)
        out_csv_path = f"{out_prefix}.csv"
        df_to_csv(results, out_csv_path)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": {}, "results": results}, f, ensure_ascii=False, indent=2)

    print("=== Evaluation Summary (JSON captions NOT used) ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved JSON: {out_json_path}")
    print(f"Saved CSV : {os.path.splitext(out_json_path)[0]}.csv")
    wandb.finish()


# ------------------------------
# (G) CLI
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="End-to-end evaluation from video via Video-LLaVA.")
    p.add_argument("--eval_json", type=str,
                   default=os.environ.get("EVAL_JSON", "/app/data/raw/json/text-evaluate/video_accident_ratio_training_results_trimmed_unsignalized_validation_0901.json"))
    p.add_argument("--fault_ckpt", type=str,
                   default=os.environ.get("FAULT_CKPT", "/app/text-train/fault_ratio_bert.pt"))
    p.add_argument("--out_json", type=str,
                   default=os.environ.get("OUT_JSON", "/app/text-train/text-results/eval_results.json"))
    p.add_argument("--video_root", type=str,
                   default=os.environ.get("VIDEO_ROOT", "/app/data/raw/videos/validation_reencoded"))
    p.add_argument("--vl_ckpt", type=str,
                   default=os.environ.get("VL_CKPT", "/app/checkpoints/last_videollava_epoch_4.pt"))
    p.add_argument("--model_name", type=str, default=os.environ.get("MODEL_NAME", "bert-base-uncased"))
    p.add_argument("--target_basis", type=float, default=float(os.environ.get("TARGET_BASIS", 10.0)))
    p.add_argument("--max_new_tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", 128)))
    p.add_argument("--num_frames", type=int, default=int(os.environ.get("NUM_FRAMES", 8)))
    p.add_argument("--size", type=int, default=int(os.environ.get("SIZE", 224)))
    # 터미널/진행 표시
    p.add_argument("--no_tqdm", action="store_true")
    p.add_argument("--print_every", type=int, default=int(os.environ.get("PRINT_EVERY", 1)))
    p.add_argument("--quiet", action="store_true", help="샘플별 프린트 끄기")
    # GPU 분담
    p.add_argument("--gpus", type=str, default=os.environ.get("GPUS", "0"),
                   help="쉼표로 구분된 GPU 인덱스 (예: '0' 또는 '0,1')")
    # 샤딩(병렬 실행시)
    p.add_argument("--shard_idx", type=int, default=int(os.environ.get("SHARD_IDX", -1)))
    p.add_argument("--num_shards", type=int, default=int(os.environ.get("NUM_SHARDS", -1)))
    return p.parse_args()


# ------------------------------
# (H) Single-video quick test (원형 유지)
# ------------------------------
# def run_fault_from_video(video_path: str, vl_ckpt_path: str, fault_model_path: str,
#                          target_basis: float = 10.0,
#                          device_caption: str = "cuda:0",
#                          device_fault: str = "cuda:0"):
#     vl = load_videollava_from_ckpt(vl_ckpt_path, dtype=torch.float16, device=device_caption)
#     caption = generate_caption_from_video(
#         vl, video_path,
#         prompt=("Describe the accident scene in one sentence. "
#                 "Include intersection type, both vehicles' movements, and entry order."),
#         max_new_tokens=128
#     )
#     fr_model, fr_tok = load_fault_model(fault_model_path, device=device_fault)
#     ratios_basis = predict_fault_ratio(fr_model, fr_tok, caption, device=device_fault)
#     ratios_100 = ratios_basis * (100.0 / target_basis)
#     dashcam, other = ratios_100.tolist()
#     print("-----")
#     print("Caption:", caption)
#     print(f"📊 Predicted Fault Ratio → Dashcam: {dashcam:.1f}%, Other: {other:.1f}%")
#     return {"caption": caption, "dashcam_100": float(dashcam), "other_100": float(other)}


# ------------------------------
# (I) Main
# ------------------------------
if __name__ == "__main__":
    args = parse_args()

    # 경로 점검
    for k, v in {
        "eval_json": args.eval_json,
        "fault_ckpt": args.fault_ckpt,
        "vl_ckpt": args.vl_ckpt,
    }.items():
        if not os.path.exists(v):
            raise SystemExit(f"[Config] Missing file for --{k}: {v}")
    if not os.path.isdir(args.video_root):
        raise SystemExit(f"[Config] --video_root not a directory: {args.video_root}")

    # GPU 분담 설정
    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    if torch.cuda.is_available() and len(gpu_list) >= 1:
        device_caption = f"cuda:{gpu_list[0]}"
        device_fault = f"cuda:{gpu_list[1]}" if len(gpu_list) > 1 else device_caption
    else:
        device_caption = device_fault = "cpu"

    # 메인 평가 실행
    evaluate_on_json(
        eval_json_path=args.eval_json,
        fault_model_path=args.fault_ckpt,
        out_json_path=args.out_json,
        model_name=args.model_name,
        target_basis=args.target_basis,
        video_root=args.video_root,
        vl_ckpt_path=args.vl_ckpt,
        max_new_tokens=args.max_new_tokens,
        num_frames=args.num_frames,
        size=args.size,
        verbose=not args.quiet,
        print_every=args.print_every,
        use_tqdm=not args.no_tqdm,
        device_caption=device_caption,
        device_fault=device_fault,
        shard_idx=args.shard_idx,
        num_shards=args.num_shards,
    )