# # ===== fault_pipeline.py =====
# # 평가 파이프라인: JSON의 캡션은 절대 쓰지 않고, 항상 비디오 -> Video-LLaVA 캡션 생성 -> 텍스트→과실 예측
# # - W&B 로깅
# # - 각 샘플 결과 CSV 저장
# # - 스캐터/히스토그램 이미지 저장(+ W&B 업로드)
# # - tqdm 진행 표시 + per-sample 터미널 로그
# # - GPU 2개 분담(캡션/과실 모델 디바이스 분리) + 선택적 샤딩

# import os, re, json, glob, argparse
# from typing import List, Optional

# import numpy as np
# import torch
# import torch.nn as nn
# import pandas as pd
# import matplotlib
# matplotlib.use("Agg")  # 서버/컨테이너 환경에서 그림 저장
# import matplotlib.pyplot as plt
# from PIL import Image
# import wandb
# from tqdm.auto import tqdm
# import cv2  # 비디오 프레임 추출용
# from transformers import AutoModel, AutoTokenizer

# # 네 기존 모듈 (Video-LLaVA 래퍼)
# from train_tracking_video_llava import VideoLLaVACaptioner

# def snap_pair_to_integer_basis(v, total=10):
#     """
#     v: np.array([a, b]) (합=total 근처)
#     반환: 합=total, 두 값 모두 '정수'인 쌍
#     """
#     v = np.asarray(v, dtype=float)
#     # 음수 방지 + 합 재정규화
#     v = np.maximum(v, 0.0)
#     s = float(v.sum())
#     if s <= 0:
#         a = total // 2
#         return np.array([a, total - a], dtype=float)
#     v = v * (total / s)

#     # 대시캠을 반올림 정수로, 나머지는 보전
#     a_int = int(np.floor(v[0] + 0.5))
#     a_int = max(0, min(total, a_int))
#     b_int = total - a_int
#     return np.array([float(a_int), float(b_int)], dtype=float)

# def project_pair_to_basis(v, total=10.0):
#     """임의의 (a,b)를 합=total로 투영. a,b<0 방지."""
#     v = np.maximum(np.asarray(v, dtype=float), 0.0)
#     s = float(v.sum())
#     if s <= 0:
#         return np.array([total/2.0, total/2.0], dtype=float)
#     return v * (total / s)

# def ratio_from_pairs(y):  # y: (N,2) basis 스케일
#     s = np.clip(y.sum(axis=1, keepdims=True), 1e-6, None)
#     return (y[:, [0]] / s).ravel()  # 대시캠 비율 ∈ (0,1)

# def pairs_from_ratio(p, total=10.0):
#     p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
#     a = total * p
#     b = total * (1.0 - p)
#     return np.stack([a, b], axis=1)

# def calibrate_ratio_isotonic(p_hat, p_true):
#     """단조(증가) 등화. scikit-learn 필요."""
#     try:
#         from sklearn.isotonic import IsotonicRegression
#     except Exception as e:
#         raise RuntimeError("IsotonicRegression을 쓸 수 없습니다(sklearn 미설치?).") from e
#     iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True)
#     iso.fit(p_hat, p_true)
#     return iso  # .transform(x)로 사용

# def calibrate_ratio_binwise(p_hat, p_true, nbins=10):
#     """등간격 bin-wise 평균 매핑. 리턴: 함수 f(p)->보정p"""
#     bins = np.linspace(0.0, 1.0, nbins + 1)
#     idx = np.clip(np.digitize(p_hat, bins) - 1, 0, nbins - 1)
#     bin_mean_true = np.zeros(nbins, dtype=float)
#     for b in range(nbins):
#         m = (idx == b)
#         if m.any():
#             bin_mean_true[b] = float(p_true[m].mean())
#         else:
#             bin_mean_true[b] = float((bins[b] + bins[b+1]) / 2.0)

#     def f(p):
#         ii = np.clip(np.digitize(p, bins) - 1, 0, nbins - 1)
#         return bin_mean_true[ii]
#     return f

# # def project_pair_to_basis(v, total=10.0):
# #     v = np.asarray(v, dtype=float)
# #     v = np.maximum(v, 0.0)
# #     s = float(v.sum())
# #     if s == 0.0:
# #         return np.array([total/2.0, total/2.0], dtype=float)
# #     return v * (total / s)

# # ------------------------------
# # (A) Video frame sampling
# # ------------------------------
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


# # ------------------------------
# # (B) Video-LLaVA load & caption
# # ------------------------------
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


# def postprocess_one_sentence(text: str) -> str:
#     t = text.strip()
#     parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', t) if p.strip()]
#     seen, cleaned = set(), []
#     for p in parts:
#         key = re.sub(r'\s+', ' ', p.lower())
#         if key not in seen:
#             cleaned.append(p); seen.add(key)
#     t = ' '.join(cleaned)
#     idx = t.lower().find("seconds.")
#     if idx != -1:
#         return t[:idx + len("seconds.")].strip()
#     m = re.split(r'(?<=[.!?])\s+', t)
#     return (m[0].strip() if m else t)


# # @torch.no_grad()
# # def generate_caption_from_video(vl_model,
# #                                 video_path: str,
# #                                 prompt: str,
# #                                 num_frames: int = 8,
# #                                 size: int = 224,
# #                                 max_new_tokens: int = 128) -> str:
# #     frames_pil = sample_frames_pil(video_path, num_frames=num_frames, size=size)
# #     out = vl_model(frames_pil=frames_pil, src_text=prompt,
# #                    generate=True, max_new_tokens=max_new_tokens)
# #     raw = (out.get("text", "") if isinstance(out, dict) else str(out))
# #     return postprocess_one_sentence(raw)
# import re

# TEMPLATE_NEEDS = [
#     "At an unsignalized intersection, the Dashcam Vehicle was",
#     "while the Other Vehicle was",
#     "Vehicle entered",
# ]

# ALLOWED_MV   = {"going straight", "left turn", "right turn"}
# ALLOWED_SIDE = {"from right road", "from left road", "from main road", "from side road left", "from side road right"}
# ALLOWED_WHO  = {"Dashcam", "Other"}
# ALLOWED_EARL = {"earlier", "later"}

# def _looks_complete_template(txt: str) -> bool:
#     t = " ".join(txt.split())
#     if not t.endswith("."):
#         return False
#     # 필수 구간 존재?
#     if not all(k in t for k in TEMPLATE_NEEDS):
#         return False
#     # 슬롯 값이 하나라도 들어갔는지 대충 검사(엄격 매칭 아님)
#     has_mv   = any(k in t for k in ALLOWED_MV)
#     has_side = any(k in t for k in ALLOWED_SIDE)
#     has_who  = any(k in t for k in ALLOWED_WHO)
#     has_e    = any(k in t for k in ALLOWED_EARL)
#     num_slots = sum([has_mv, has_side, has_who, has_e])
#     return num_slots >= 2

# def _clean_sentence(txt: str) -> str:
#     t = " ".join(txt.split())
#     # 복수형/오타 흔들림 정리
#     t = t.replace("Other Vehicles were", "Other Vehicle was")
#     t = re.sub(r"\s+", " ", t).strip()
#     if not t.endswith("."):
#         t += "."
#     return t

# @torch.no_grad()
# def generate_caption_from_video(
#     vl_model, video_path, prompt, num_frames=8, size=224, max_new_tokens=128
# ):
#     frames_pil = sample_frames_pil(video_path, num_frames=num_frames, size=size)

#     attempts = []
#     # 서로 다른 디코딩 전략을 실제로 전달
#     # schedule = [
#     #     # 1) 가장 가벼운 기본 (짧게)
#     #     dict(max_new_tokens=40, 
#     #         do_sample=False, num_beams=1),

#     #     # 2) 조금 더 길고 빔서치
#     #     dict(max_new_tokens=40,
#     #         do_sample=False, num_beams=1),

#     #     # 3) 더 길게, 빔서치=5
#     #     dict(max_new_tokens=40,
#     #         do_sample=False, num_beams=1),

#     #     # 4) 샘플링 (탐색 강화)
#     #     dict(max_new_tokens=40,
#     #         do_sample=True, num_beams=1, top_p=0.9, temperature=0.7),

#     #     # 5) 샘플링 + 빔서치 혼합
#     #     dict(max_new_tokens=40,
#     #         do_sample=True, num_beams=1, top_p=0.9, temperature=0.8),
#     # ]
#     schedule = [
#         # 짧고 보수적으로 시작
#         dict(max_new_tokens=48,  do_sample=False, num_beams=1,  no_repeat_ngram_size=3),
#         dict(max_new_tokens=64,  do_sample=False, num_beams=3,  no_repeat_ngram_size=3),
#         dict(max_new_tokens=80,  do_sample=False, num_beams=5,  no_repeat_ngram_size=3),

#         # 탐색 강화(샘플링)
#         dict(max_new_tokens=96,  do_sample=True,  num_beams=1,  top_p=0.9, temperature=0.7, no_repeat_ngram_size=3),
#         dict(max_new_tokens=112, do_sample=True,  num_beams=1,  top_p=0.9, temperature=0.9, no_repeat_ngram_size=3),

#         # 마지막 안전망(더 길게)
#         dict(max_new_tokens=128, do_sample=False, num_beams=5,  no_repeat_ngram_size=3),
#     ]

#     for i, kwargs in enumerate(schedule, 1):
#         out = vl_model(frames_pil=frames_pil, src_text=prompt, generate=True, **kwargs)
#         raw = (out.get("text", "") if isinstance(out, dict) else str(out)).strip()
#         attempts.append(raw)

#         # 완결성 검사: 마침표 + 필수 구간 + 슬롯 단어가 하나씩이라도 들어갔는지
#         if _looks_complete_template(raw):
#             return _clean_sentence(raw)

#         print(f"[caption retry] step={i} need_retry: text='{raw[:120]}...'")

#     # 그래도 실패 → 가장 긴 후보를 정리해서 반환
#     best = max(attempts, key=lambda s: len(s) if isinstance(s, str) else 0)
#     return _clean_sentence(best or "")

# # ------------------------------
# # (C) Text -> Fault model
# # ------------------------------
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
#         out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         cls = out.last_hidden_state[:, 0]
#         return self.regressor(cls)


# def load_fault_model(path: str,
#                      model_name: str = "bert-base-uncased",
#                      device: str = "cuda"):
#     obj = torch.load(path, map_location="cpu")
#     model = TextToFaultRatio(model_name=model_name)
#     if isinstance(obj, dict) and "state_dict" in obj:
#         model.load_state_dict(obj["state_dict"], strict=True)
#     elif isinstance(obj, dict):
#         model.load_state_dict(obj, strict=False)
#     elif hasattr(obj, "state_dict"):
#         model = obj
#     model.to(device).eval()
#     tok = AutoTokenizer.from_pretrained(model_name)
#     return model, tok


# @torch.no_grad()
# def predict_fault_ratio(model, tokenizer, text: str, device: str = "cuda", max_length: int = 256) -> np.ndarray:
#     inputs = tokenizer(text, return_tensors="pt",
#                        padding="max_length", truncation=True, max_length=max_length)
#     input_ids = inputs["input_ids"].to(device)
#     attention_mask = inputs["attention_mask"].to(device)
#     pred = model(input_ids=input_ids, attention_mask=attention_mask)  # [1,2]
#     return pred.squeeze(0).float().cpu().numpy()  # basis 스케일(기본 10)


# # ------------------------------
# # (D) Label utils & video resolver
# # ------------------------------
# # def normalize_pair_100(p) -> List[float]:
# #     a, b = float(p[0]), float(p[1])
# #     s = a + b
# #     if 0 <= a <= 1 and 0 <= b <= 1:
# #         a *= 100.0; b *= 100.0; s = a + b
# #     if s > 0 and abs(s - 100.0) > 1e-6:
# #         a *= (100.0 / s); b *= (100.0 / s)
# #     return [max(0.0, min(100.0, a)), max(0.0, min(100.0, b))]
# def normalize_pair_100(p) -> List[float]:
#     a, b = float(p[0]), float(p[1])
#     # [0,1] 스케일이면 100으로 확장
#     if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
#         a *= 100.0; b *= 100.0
#     # 음수 컷 → 그 뒤 반드시 재정규화
#     a = max(a, 0.0); b = max(b, 0.0)
#     s = a + b
#     if s == 0.0:
#         return [50.0, 50.0]
#     scale = 100.0 / s
#     return [a * scale, b * scale]


# def to_basis(pair100: List[float], target_basis: float = 10.0) -> List[float]:
#     factor = 100.0 / target_basis  # 10이면 /10
#     return [p / factor for p in pair100]


# def find_video_file(video_root: str, name_or_rel: str) -> Optional[str]:
#     cand = os.path.join(video_root, name_or_rel)
#     if os.path.exists(cand):
#         return cand
#     stem = os.path.splitext(name_or_rel)[0]
#     for ext in (".mp4", ".avi", ".mov", ".mkv"):
#         p = os.path.join(video_root, stem + ext)
#         if os.path.exists(p):
#             return p
#     g = glob.glob(os.path.join(video_root, stem + "*"))
#     return g[0] if g else None


# # ------------------------------
# # (E) Metrics, plots, saving
# # ------------------------------
# def compute_metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
#     from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#     mae = mean_absolute_error(y, yhat)
#     try:
#         rmse = mean_squared_error(y, yhat, squared=False)
#     except TypeError:
#         rmse = np.sqrt(mean_squared_error(y, yhat))
#     r2 = r2_score(y, yhat)
#     mae_dc = mean_absolute_error(y[:, 0], yhat[:, 0])
#     mae_ov = mean_absolute_error(y[:, 1], yhat[:, 1])
#     return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2),
#             "MAE_dashcam": float(mae_dc), "MAE_other": float(mae_ov), "count": int(len(y))}


# def save_plots(y: np.ndarray, yhat: np.ndarray, target_basis: float, out_prefix: str) -> dict:
#     os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
#     paths = {}

#     # 1) 스캐터 (대시캠/상대차량)
#     fig, ax = plt.subplots(1, 2, figsize=(12, 5))
#     titles = ["Dashcam", "Other Vehicle"]
#     for i in range(2):
#         ax[i].scatter(y[:, i], yhat[:, i], alpha=0.5)
#         ax[i].plot([0, target_basis], [0, target_basis], "--", color="gray")
#         ax[i].set_title(f"{titles[i]} Fault (basis={target_basis})")
#         ax[i].set_xlabel("True"); ax[i].set_ylabel("Pred")
#         ax[i].set_xlim(0, target_basis); ax[i].set_ylim(0, target_basis)
#     plt.tight_layout()
#     scatter_path = f"{out_prefix}_scatter.png"
#     plt.savefig(scatter_path); plt.close(fig)
#     paths["scatter"] = scatter_path

#     # 2) 절대오차 히스토그램
#     err = np.abs(yhat - y)
#     fig2, ax2 = plt.subplots(figsize=(6, 4))
#     ax2.hist(err[:, 0], bins=20, alpha=0.6, label="Dashcam")
#     ax2.hist(err[:, 1], bins=20, alpha=0.6, label="Other")
#     ax2.set_title(f"Absolute Error Histogram (basis={target_basis})")
#     ax2.set_xlabel("Abs Error"); ax2.set_ylabel("Count"); ax2.legend()
#     hist_path = f"{out_prefix}_err_hist.png"
#     plt.tight_layout(); plt.savefig(hist_path); plt.close(fig2)
#     paths["err_hist"] = hist_path

#     return paths


# def df_to_csv(results: List[dict], out_csv_path: str):
#     os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True
#                )
#     df = pd.DataFrame(results)
#     df.to_csv(out_csv_path, index=False, encoding="utf-8")


# # ------------------------------
# # (F) Evaluation (JSON 캡션 절대 사용 X)
# # ------------------------------
# def evaluate_on_json(eval_json_path: str,
#                      fault_model_path: str,
#                      out_json_path: str,
#                      model_name: str = "bert-base-uncased",
#                      target_basis: float = 10.0,
#                      video_root: str = None,
#                      vl_ckpt_path: str = None,
#                      max_new_tokens: int = 128,
#                      num_frames: int = 8,
#                      size: int = 224,
#                      verbose: bool = True,
#                      print_every: int = 1,
#                      use_tqdm: bool = True,
#                      device_caption: str = "cuda:0",
#                      device_fault: str = "cuda:0",
#                      shard_idx: int = -1,
#                      num_shards: int = -1):
#     # 입력 검증
#     if not os.path.exists(eval_json_path):
#         raise FileNotFoundError(eval_json_path)
#     if not os.path.exists(vl_ckpt_path or ""):
#         raise FileNotFoundError(vl_ckpt_path)
#     if not os.path.isdir(video_root or ""):
#         raise NotADirectoryError(video_root)

#     # W&B 세팅
#     wandb.init(
#         project="end-to-end-accident-caption",
#         config={
#             "eval_json": eval_json_path,
#             "fault_ckpt": fault_model_path,
#             "video_root": video_root,
#             "vl_ckpt": vl_ckpt_path,
#             "model_name": model_name,
#             "target_basis": target_basis,
#             "num_frames": num_frames,
#             "size": size,
#             "max_new_tokens": max_new_tokens,
#             "device_caption": device_caption,
#             "device_fault": device_fault,
#             "shard_idx": shard_idx,
#             "num_shards": num_shards,
#         },
#         job_type="evaluation",
#     )

#     # 모델 로드 (GPU 분담)
#     vl_model = load_videollava_from_ckpt(vl_ckpt_path, device=device_caption)
#     fr_model, fr_tok = load_fault_model(fault_model_path, model_name=model_name, device=device_fault)

#     # 데이터 적재 (+샤딩)
#     with open(eval_json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     if not isinstance(data, list):
#         raise ValueError("eval_json must be a list of objects.")
#     if shard_idx >= 0 and num_shards > 0:
#         idxs = np.array_split(np.arange(len(data)), num_shards)[shard_idx].tolist()
#         data = [data[i] for i in idxs]

#     # caption_prompt = (
#     #     "Task: Describe the accident scene in one sentence.\n"
#     #     "You MUST include:\n"
#     #     "- intersection type\n"
#     #     "- the movement and direction of both vehicles (Dashcam and Other)\n"
#     #     "- entry order (who entered first)\n"
#     #     "Avoid adding traffic signals or unrelated details."
#     # )
#     # caption_prompt = (
#     #         "This is a first-person dashcam (ego-view) video. 'Dashcam Vehicle' is the recording car (ego).\n"
#     #         "ONE sentence. Use this EXACT template:\n"
#     #         "At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, "
#     #         "while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>.\n"
#     #         "Values: <mv_*>={going straight,left turn,right turn}; "
#     #         "<side_*>={from the right,from left road,from main road,from side road left,from side road right}; "
#     #         "<who_entered>={Dashcam,Other}; <earlier_or_later>={earlier,later}.\n"
#     #         "No extra words. Use singular 'Other Vehicle'. Do NOT mention camera/ego/first-person in the sentence."
#     # )
#     # caption_prompt = (
#     #         "This is a first-person dashcam (ego-view) video. 'Dashcam Vehicle' = the recording car.\n"
#     #         "OUTPUT: ONE sentence in THIS EXACT template (replace <> only):\n"
#     #         "At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, "
#     #         "while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>.\n"
#     #         "Allowed: <mv>={going straight,left turn,right turn}; "
#     #         "<side>={from the right,from left road,from main road,from side road left,from side road right}; "
#     #         "<who_entered>={Dashcam,Other}; <earlier_or_later>={earlier,later}.\n"
#     #         "If '<SCENE_HINTS>' appears below, copy values EXACTLY.\n"
#     #         "Return ONLY the sentence. Do NOT mention camera/ego/first-person."
#     # )
#     # caption_prompt = (
#     #     "You are extracting 6 categorical slots for an unsignalized intersection scene.\n"
#     #     "Return ONLY a single-line JSON object with these EXACT keys:\n"
#     #     '{"mv_dv": "...", "side_dv": "...", "mv_ov": "...", "side_ov": "...", "who_entered": "...", "earlier_or_later": "..."}\n\n'
#     #     "Allowed values (copy verbatim):\n"
#     #     '- \"mv_dv\", \"mv_ov\" ∈ {\"going straight\",\"left turn\",\"right turn\"}\n'
#     #     '- \"side_dv\", \"side_ov\" ∈ {\"from right road\",\"from left road\",\"from main road\",\"from side road left\",\"from side road right\"}\n'
#     #     '- \"who_entered\" ∈ {\"Dashcam\",\"Other\"}\n'
#     #     '- \"earlier_or_later\" ∈ {\"earlier\",\"later\"}\n\n'
#     #     "Hard rules:\n"
#     #     "- Output MUST be valid JSON on ONE line. No code fences, no explanations.\n"
#     #     "- Use ONLY the allowed values (verbatim). No synonyms, no extra words.\n"
#     #     "- Do NOT mention lights, signals, lanes, time, date, crash/collision.\n"
#     #     "- Do NOT output any sentence. Only the JSON object.\n\n"
#     #     "Good example:\n"
#     #     '{"mv_dv":"going straight","side_dv":"from main road","mv_ov":"right turn","side_ov":"from side road left","who_entered":"Other","earlier_or_later":"earlier"}'
#     # )
#     # 이게 성능 가장 좋음
#     # caption_prompt = (
#     #     "This is a first-person dashcam (ego-view) video. 'Dashcam Vehicle' = the recording car.\n"
#     #     "OUTPUT EXACTLY ONE sentence by filling ONLY the 4 slots in the TEMPLATE below.\n"
#     #     "TEMPLATE (copy every word EXACTLY, replace only the bracketed slots):\n"
#     #     "At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, "
#     #     "while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>.\n\n"
#     #     "ALLOWED VALUES (copy verbatim, NOTHING ELSE):\n"
#     #     "- <mv_*> = {going straight, left turn, right turn}\n"
#     #     "- <side_*> = {from right road, from left road, from main road, from side road left, from side road right}\n"
#     #     "- <who_entered> = {Dashcam, Other}\n"
#     #     "- <earlier_or_later> = {earlier, later}\n\n"
#     #     "HARD RULES:\n"
#     #     "- Do NOT invent words like 'following vehicle', 'lanes', 'light', 'signal', 'time', etc.\n"
#     #     "- Do NOT change anything outside the 4 slots. Copy the TEMPLATE text exactly.\n"
#     #     "- Use singular 'Other Vehicle' exactly. NEVER write Vegetable, Vehitcle, Vehicel, Vehov, Vehicles, Vehi, Veh, Ve etc.\n"
#     #     "- Do NOT add or remove commas/words/punctuation.\n"
#     #     "- Final output must be ONE sentence ending with a period.\n\n"
#     #     "GOOD EXAMPLE (valid):\n"
#     #     "At an unsignalized intersection, the Dashcam Vehicle was going straight from main road, "
#     #     "while the Other Vehicle was right turn from side road left, and the Other Vehicle entered earlier.\n\n"
#     #     "BAD EXAMPLE (invalid):\n"
#     #     "At an unsignalized intersection, the Dashcam Vehicle was following vehicle of the other vehicle.  <-- WRONG: not in allowed values\n"
#     #     "At an uncontrolled intersection, the dashcam vehicle was going ahead.  <-- WRONG: template not copied, invalid words\n"
#     # )
#     # caption_prompt = (
#     #     "NEVER write Vegetable, Vehitcle, Vehicel, Vehov, Vehicles, Vehi, Veh, Ve etc.\n"
#     #     "This is a first-person dashcam (ego-view) video. 'Dashcam Vehicle' = the recording car.\n"
#     #     "OUTPUT EXACTLY ONE sentence by filling ONLY the 4 slots in the TEMPLATE below.\n"
#     #     "TEMPLATE (copy every word EXACTLY, replace only the bracketed slots):\n"
#     #     "At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, "
#     #     "while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>.\n\n"
#     #     "ALLOWED VALUES (copy verbatim, NOTHING ELSE):\n"
#     #     "- <mv_*> = {going straight, left turn, right turn}\n"
#     #     "- <side_*> = {from right road, from left road, from side road left, from side road right}\n"
#     #     "- <who_entered> = {Dashcam, Other}\n"
#     #     "- <earlier_or_later> = {earlier, later}\n\n"
#     #     "HARD RULES:\n"
#     #     "- Do NOT invent words like 'following vehicle', 'lanes', 'light', 'signal', 'time', etc.\n"
#     #     "- Do NOT change anything outside the 4 slots. Copy the TEMPLATE text exactly.\n"
#     #     "- Use singular 'Other Vehicle' exactly.\n"
#     #     "- Do NOT add or remove commas/words/punctuation.\n"
#     #     "- Final output must be ONE sentence ending with a period.\n\n"
#     # )
#     caption_prompt = (
#         "STRICT INSTRUCTIONS: Follow ALL rules below without exception.\n"
#         "NEVER write misspelled or partial words such as: Vegetable, Vehitcle, Vehicel, Vehov, Vehicles, Vehi, Veh, Ve, etc.\n"
#         "NEVER output truncated tokens (e.g., 'Vehi').\n\n"

#         "This is a first-person dashcam (ego-view) video. 'Dashcam Vehicle' = the recording car.\n\n"

#         "OUTPUT EXACTLY ONE sentence by filling ONLY the 4 slots in the TEMPLATE below.\n"
#         "TEMPLATE (copy every word EXACTLY, replace only the bracketed slots):\n"
#         "At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, "
#         "while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>.\n\n"

#         "ALLOWED VALUES (copy verbatim, NOTHING ELSE):\n"
#         "- <mv_*> = {going straight, left turn, right turn}\n"
#         "- <side_*> = {from right road, from left road, from side road left, from side road right}\n"
#         "- <who_entered> = {Dashcam, Other}\n"
#         "- <earlier_or_later> = {earlier, later}\n\n"

#         "HARD RULES:\n"
#         "- All 4 slots MUST be filled. Never leave a slot blank.\n"
#         "- Do NOT invent or add any words (e.g., 'following vehicle', 'lanes', 'light', 'signal', 'time').\n"
#         "- Do NOT change, drop, or paraphrase any words outside the slots. Copy the TEMPLATE text exactly.\n"
#         "- Always use singular form 'Other Vehicle' (NEVER plural).\n"
#         "- Do NOT add or remove commas, words, or punctuation.\n"
#         "- The final output MUST be exactly ONE sentence and MUST end with a period.\n"
#     )

#     # caption_prompt = (
#     #     "This is a first-person dashcam (ego-view) video. 'Dashcam Vehicle' = the recording car.\n"
#     #     "OUTPUT EXACTLY ONE sentence by filling ONLY the 4 slots in the TEMPLATE below.\n"
#     #     "TEMPLATE (copy every word EXACTLY, replace only the bracketed slots):\n"
#     #     "At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, "
#     #     "while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>.\n\n"
#     #     "ALLOWED VALUES (copy verbatim, NOTHING ELSE):\n"
#     #     "- <mv_*> = {going straight, left turn, right turn}\n"
#     #     "- <side_*> = {from right road, from left road, from main road, from side road left, from side road right}\n"
#     #     "- <who_entered> = {Dashcam, Other}\n"
#     #     "- <earlier_or_later> = {earlier, later}\n\n"
#     #     "HARD RULES:\n"
#     #     "- Do NOT invent words like 'following vehicle', 'lanes', 'light', 'signal', 'time', etc.\n"
#     #     "- Do NOT change anything outside the 4 slots. Copy the TEMPLATE text exactly.\n"
#     #     "- Use singular 'Other Vehicle' exactly. NEVER write Vegetable, Vehitcle, Vehicel, Vehov etc.\n"
#     #     "- Do NOT add or remove commas/words/punctuation.\n"
#     #     "- Final output must be ONE sentence ending with a period.\n\n"
#     #     "GOOD EXAMPLE (valid):\n"
#     #     "At an unsignalized intersection, the Dashcam Vehicle was going straight from main road, while the Other Vehicle was right turn from side road left, and the Other Vehicle entered earlier.\n\n"
#     #     "BAD EXAMPLE (invalid):\n"
#     #     "At an unsignalized intersection, the Dashcam Vehicle was following vehicle of the other vehicle.  <-- WRONG: not in allowed values\n"
#     #     "At an uncontrolled intersection, the dashcam vehicle was going ahead.  <-- WRONG: template not copied, invalid words\n"
#     # )
#     # caption_prompt = (
#     #     "This is a first-person dashcam (ego-view) video. 'Dashcam Vehicle' = the recording car.\n"
#     #     "RETURN EXACTLY ONE SENTENCE, wrapped in <CAPTION> ... </CAPTION>.\n"
#     #     "Fill ONLY the 4 slots in the TEMPLATE below. Copy every non-bracket character EXACTLY.\n\n"
#     #     "TEMPLATE:\n"
#     #     "<CAPTION>At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, "
#     #     "while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>.</CAPTION>\n\n"
#     #     "ALLOWED VALUES (copy verbatim; choose ONE from each line):\n"
#     #     "<mv_dv> = {going straight, left turn, right turn}\n"
#     #     "<side_dv> = {from right road, from left road, from main road, from side road left, from side road right}\n"
#     #     "<mv_ov> = {going straight, left turn, right turn}\n"
#     #     "<side_ov> = {from right road, from left road, from main road, from side road left, from side road right}\n"
#     #     "<who_entered> = {Dashcam, Other}\n"
#     #     "<earlier_or_later> = {earlier, later}\n\n"
#     #     "HARD RULES:\n"
#     #     "- Output ONLY the <CAPTION> sentence. No explanations, no extra text.\n"
#     #     "- Do NOT invent words like lanes, signal, traffic light, time, speed, camera, ego.\n"
#     #     "- Do NOT change any punctuation. Exactly one comma after each <side_*>, and exactly one space after each comma.\n"
#     #     "- Use singular 'Other Vehicle' exactly. Do NOT use 'Vehicles'\n"
#     #     "- Final output must end with a period inside </CAPTION>.\n\n"
#     #     "GOOD EXAMPLE:\n"
#     #     "<CAPTION>At an unsignalized intersection, the Dashcam Vehicle was going straight from main road, "
#     #     "while the Other Vehicle was right turn from side road left, and the Other Vehicle entered earlier.</CAPTION>\n"
#     # )
#     # caption_prompt = (
#     #     "This is a first-person dashcam (ego-view) video. 'Dashcam Vehicle' = the recording car.\n"
#     #     "OUTPUT EXACTLY ONE sentence by filling ONLY the 4 slots in the TEMPLATE below.\n"
#     #     "TEMPLATE (copy every word EXACTLY, replace only the bracketed slots):\n"
#     #     "At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, "
#     #     "while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>.\n\n"
#     #     "ALLOWED VALUES (copy verbatim, NOTHING ELSE):\n"
#     #     "- <mv_*> = {going straight, left turn, right turn}\n"
#     #     "- <side_*> = {from right road, from left road, from main road, from side road left, from side road right}\n"
#     #     "- <who_entered> = {Dashcam, Other}\n"
#     #     "- <earlier_or_later> = {earlier, later}\n\n"
#     #     "HARD RULES:\n"
#     #     "- Do NOT invent words like 'following vehicle', 'lanes', 'light', 'signal', 'time', etc.\n"
#     #     "- Do NOT change anything outside the 4 slots. Copy the TEMPLATE text exactly.\n"
#     #     "- Use singular 'Other Vehicle' exactly. NEVER write Vegetable, Vehitcle, Vehicel, Vehov etc.\n"
#     #     "- Do NOT add or remove commas/words/punctuation.\n"
#     #     "- Final output must be ONE sentence ending with a period.\n\n"
#     # )
#     # caption_prompt = (
#     #         "This is a first-person dashcam (ego-view) video. 'Dashcam Vehicle' = the recording car.\n"
#     #         "OUTPUT: ONE sentence in THIS EXACT template (replace <> only):\n"
#     #         "At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, "
#     #         "while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>.\n"
#     #         "Allowed: <mv>={going straight,left turn,right turn}; "
#     #         "<side>={from the right,from left road,from main road,from side road left,from side road right}; "
#     #         "<who_entered>={Dashcam,Other}; <earlier_or_later>={earlier,later}.\n"
#     #         "If '<SCENE_HINTS>' appears below, copy values EXACTLY.\n"
#     #         "Return ONLY the sentence. Do NOT mention camera/ego/first-person."

#     #         "GOOD EXAMPLE (valid):\n"
#     #         "At an unsignalized intersection, the Dashcam Vehicle was going straight from main road, "
#     #         "while the Other Vehicle was right turn from side road left, and the Other Vehicle entered earlier.\n\n"
#     #         "BAD EXAMPLE (invalid):\n"
#     #         "At an unsignalized intersection, the Dashcam Vehicle was following vehicle of the other vehicle.  <-- WRONG: not in allowed values\n"
#     #         "At an uncontrolled intersection, the dashcam vehicle was going ahead.  <-- WRONG: template not copied, invalid words\n"
#     # )
#     # caption_prompt = (
#     #     "Return ONLY one-line JSON with EXACT keys and values:\n"
#     #     '{"mv_dv":"...","side_dv":"...","mv_ov":"...","side_ov":"...","who_entered":"...","earlier_or_later":"..."}\n'
#     #     "Allowed values (copy verbatim; nothing else):\n"
#     #     '- "mv_dv","mv_ov" ∈ {"going straight","left turn","right turn"}\n'
#     #     '- "side_dv","side_ov" ∈ {"from right road","from left road","from main road","from side road left","from side road right"}\n'
#     #     '- "who_entered" ∈ {"Dashcam","Other"}\n'
#     #     '- "earlier_or_later" ∈ {"earlier","later"}\n'
#     #     "Hard rules: one line, valid JSON, ONLY allowed values; no other words.\n"
#     # )


#     preds_basis, labels_basis = [], []
#     results = []
#     check_list_video = []

#     N = len(data)
#     iterator = tqdm(range(N), total=N, desc="Evaluating", dynamic_ncols=True) if use_tqdm else range(N)

#     for i in iterator:
#         row = data[i]

#         # GT (metrics용)
#         gt_basis = None
#         gt100 = None
#         if "dashcam_vehicle_negligence" in row and "other_vehicle_negligence" in row:
#             gt100 = normalize_pair_100([row["dashcam_vehicle_negligence"], row["other_vehicle_negligence"]])
#             gt_basis = np.array(to_basis(gt100, target_basis), dtype=float)

#         # 비디오 찾기
#         video_name = row.get("video_name") or row.get("video_path") or ""
#         vpath = find_video_file(video_root, video_name) if video_name else None
#         if not vpath:
#             raw = row.get("video_path")
#             if raw and os.path.exists(raw):
#                 vpath = raw
#         if not vpath:
#             out_item = {"idx": i, "video_name": video_name, "error": "video_not_found"}
#             results.append(out_item)
#             if verbose:
#                 print(f"[{i+1}/{N}] {video_name} | VIDEO NOT FOUND", flush=True)
#             continue

#         # 캡션 생성(항상 비디오에서)
#         try:
#             caption = generate_caption_from_video(
#                 vl_model, vpath, prompt=caption_prompt,
#                 num_frames=num_frames, size=size, max_new_tokens=max_new_tokens
#             )
#         except Exception as e:
#             out_item = {"idx": i, "video_name": video_name, "error": f"caption_generation_failed: {e}"}
#             results.append(out_item)
#             if verbose:
#                 print(f"[{i+1}/{N}] {video_name} | CAPTION ERROR: {e}", flush=True)
#             continue

#         # 예측
#         pred_basis = predict_fault_ratio(fr_model, fr_tok, caption, device=device_fault)  # basis 스케일
#         pred_basis = project_pair_to_basis(pred_basis, total=target_basis)

#         # pred_basis = snap_pair_to_integer_basis(pred_basis, total=int(target_basis))

#         preds_basis.append(pred_basis)
#         if gt_basis is not None:
#             labels_basis.append(gt_basis)

#         # 보기 편하게 0~100도 함께 기록
#         pred_100 = [float(x) * (100.0 / target_basis) for x in pred_basis]

#         out_item = {
#             "idx": i,
#             "video_name": video_name,
#             "caption_pred": caption,  # 생성된 캡션
#             "pred_basis_dashcam": float(pred_basis[0]),
#             "pred_basis_other": float(pred_basis[1]),
#             "pred_100_dashcam": float(pred_100[0]),
#             "pred_100_other": float(pred_100[1]),
#         }
#         if gt_basis is not None:
#             out_item["gt_basis_dashcam"] = float(gt_basis[0])
#             out_item["gt_basis_other"] = float(gt_basis[1])
#             out_item["gt_100_dashcam"] = float(gt100[0])
#             out_item["gt_100_other"] = float(gt100[1])
#             out_item["abs_err_basis_dashcam"] = abs(out_item["gt_basis_dashcam"] - out_item["pred_basis_dashcam"])
#             out_item["abs_err_basis_other"] = abs(out_item["gt_basis_other"] - out_item["pred_basis_other"])

#         results.append(out_item)

#         # per-sample 로그
#         if verbose and ( (i == 0) or ((i + 1) % print_every == 0) or (i + 1 == N) ):
#             msg = f"[{i+1}/{N}] {video_name} | pred_basis=[{out_item['pred_basis_dashcam']:.2f}, {out_item['pred_basis_other']:.2f}]"
#             tgt_text = str(row.get('generated_caption', "No description available")).strip()
#             m = re.search(r"\bthe collision\b", tgt_text, flags=re.IGNORECASE)
#             if m:
#                 tgt_text = tgt_text[:m.start()].rstrip()
#                 # 끝이 .!? 로 안 끝나면 마침표 하나 붙여 깔끔하게
#                 if not tgt_text.endswith(('.', '!', '?')):
#                     tgt_text += '.'

#             if gt_basis is not None:
#                 ae_dc = out_item["abs_err_basis_dashcam"]
#                 ae_ov = out_item["abs_err_basis_other"]
#                 if ae_dc >= 2.0 or ae_ov >= 2.0:
#                     check_list_video.append([video_name, caption, tgt_text])
#                 msg += f" | gt_basis=[{out_item['gt_basis_dashcam']:.2f}, {out_item['gt_basis_other']:.2f}] | abs_err=[{ae_dc:.2f}, {ae_ov:.2f}]"

#             msg += "\n"
#             msg += f"⭐️pred_caption: {caption}\n" + f"⭐️gt_caption: {tgt_text}"
#             print(msg, flush=True)

#         # 배치/스텝 로깅(선택)
#         if (i + 1) % 20 == 0:
#             wandb.log({"eval_progress_samples": i + 1})

#     # 메트릭 & 저장
#     metrics = {}
#     out_prefix = os.path.splitext(out_json_path)[0]
#     os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)

#     # 결과 중 "예측이 있는 행"만 모으기 (VIDEO NOT FOUND 같은 에러행 제외)
#     pred_rows = [r for r in results if "pred_basis_dashcam" in r]
#     if len(pred_rows) == 0:
#         # 예측 자체가 없으면 결과만 저장
#         out_csv_path = f"{out_prefix}.csv"
#         df_to_csv(results, out_csv_path)
#         with open(out_json_path, "w", encoding="utf-8") as f:
#             json.dump({"metrics": {}, "results": results}, f, ensure_ascii=False, indent=2)
#         print("=== Evaluation Summary (JSON captions NOT used) ===")
#         print(json.dumps({}, indent=2, ensure_ascii=False))
#         print(f"Saved JSON: {out_json_path}")
#         print(f"Saved CSV : {out_csv_path}")
#         print("Check list (abs err basis >=2.0):", len(check_list_video))
#         print(check_list_video)
#         wandb.finish()
#         return

#     # (1) 배열로 정리
#     yhat_all = np.array([[r["pred_basis_dashcam"], r["pred_basis_other"]] for r in pred_rows], dtype=float)
#     mask_gt = np.array([("gt_basis_dashcam" in r) for r in pred_rows], dtype=bool)

#     # GT가 있는 행만 분리
#     if mask_gt.any():
#         y_gt    = np.array([[pred_rows[i]["gt_basis_dashcam"], pred_rows[i]["gt_basis_other"]] for i in range(len(pred_rows)) if mask_gt[i]], dtype=float)
#         yhat_gt = yhat_all[mask_gt]

#         # 보정 전
#         metrics_pre = compute_metrics(y_gt, yhat_gt)
#         metrics_pre["target_basis"] = target_basis

#         # 비율로 등화 학습
#         p_hat_gt = ratio_from_pairs(yhat_gt)
#         p_true_gt = ratio_from_pairs(y_gt)

#         try:
#             from sklearn.isotonic import IsotonicRegression
#             iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True).fit(p_hat_gt, p_true_gt)
#             p_cal_all = iso.transform(ratio_from_pairs(yhat_all))
#         except Exception as e:
#             print("[cal] isotonic failed; fallback to bin-wise:", e)
#             fbin = calibrate_ratio_binwise(p_hat_gt, p_true_gt, nbins=10)
#             p_cal_all = fbin(ratio_from_pairs(yhat_all))

#         # 비율→쌍으로 복원 + 합=target_basis 재보장
#         yhat_cal_all = pairs_from_ratio(p_cal_all, total=target_basis)
#         yhat_cal_all = np.vstack([project_pair_to_basis(v, total=target_basis) for v in yhat_cal_all])
#         # yhat_cal_all = np.vstack([snap_pair_to_integer_basis(v, total=int(target_basis)) for v in yhat_cal_all])

#         # 결과 dict에 보정값 반영
#         for k, r in enumerate(pred_rows):
#             r["pred_basis_dashcam_cal"] = float(yhat_cal_all[k, 0])
#             r["pred_basis_other_cal"]   = float(yhat_cal_all[k, 1])

#         # 보정 후 메트릭(GT 있는 서브셋)
#         yhat_cal_gt = yhat_cal_all[mask_gt]
#         metrics_cal = compute_metrics(y_gt, yhat_cal_gt)

#         metrics = {**metrics_pre, **{f"cal/{k}": v for k, v in metrics_cal.items()}}

#         # 플롯(전/후)
#         plot_paths = save_plots(y_gt, yhat_gt, target_basis, out_prefix + "_precal")
#         plot_paths_cal = save_plots(y_gt, yhat_cal_gt, target_basis, out_prefix + "_cal")
#         plot_paths.update({("cal_" + k): v for k, v in plot_paths_cal.items()})

#         # CSV/JSON 저장
#         out_csv_path = f"{out_prefix}.csv"
#         pd.DataFrame(results).to_csv(out_csv_path, index=False, encoding="utf-8")
#         with open(out_json_path, "w", encoding="utf-8") as f:
#             json.dump({"metrics": metrics, "results": results}, f, ensure_ascii=False, indent=2)

#         # W&B 로깅
#         wandb.log({
#             "eval/MAE": metrics["MAE"],
#             "eval/RMSE": metrics["RMSE"],
#             "eval/R2": metrics["R2"],
#             "eval/MAE_dashcam": metrics["MAE_dashcam"],
#             "eval/MAE_other": metrics["MAE_other"],
#             "eval_cal/MAE": metrics["cal/MAE"],
#             "eval_cal/RMSE": metrics["cal/RMSE"],
#             "eval_cal/R2": metrics["cal/R2"],
#             "eval_cal/MAE_dashcam": metrics["cal/MAE_dashcam"],
#             "eval_cal/MAE_other": metrics["cal/MAE_other"],
#         })
#         for k, pth in plot_paths.items():
#             if os.path.exists(pth):
#                 wandb.log({f"plots/{k}": wandb.Image(pth)})

#         # 아티팩트
#         try:
#             table = wandb.Table(dataframe=pd.DataFrame(results))
#             wandb.log({"eval/table": table})
#         except Exception:
#             pass
#         try:
#             art = wandb.Artifact("eval_results", type="evaluation")
#             art.add_file(out_json_path); art.add_file(out_csv_path)
#             for p in plot_paths.values():
#                 if os.path.exists(p):
#                     art.add_file(p)
#             wandb.log_artifact(art)
#         except Exception as e:
#             print("W&B artifact upload failed:", e)

#     else:
#         # GT가 전혀 없으면 메트릭 없이 저장
#         out_csv_path = f"{out_prefix}.csv"
#         df_to_csv(results, out_csv_path)
#         with open(out_json_path, "w", encoding="utf-8") as f:
#             json.dump({"metrics": {}, "results": results}, f, ensure_ascii=False, indent=2)

#     print("=== Evaluation Summary (JSON captions NOT used) ===")
#     print(json.dumps(metrics, indent=2, ensure_ascii=False))
#     print(f"Saved JSON: {out_json_path}")
#     print(f"Saved CSV : {os.path.splitext(out_json_path)[0]}.csv")
#     print("Check list (abs err basis >=2.0):", len(check_list_video))
#     print(check_list_video)
#     wandb.finish()


# # ------------------------------
# # (G) CLI
# # ------------------------------
# def parse_args():
#     p = argparse.ArgumentParser(description="End-to-end evaluation from video via Video-LLaVA.")
#     # p.add_argument("--eval_json", type=str,
#     #                default=os.environ.get("EVAL_JSON", "/app/data/raw/json/text-evaluate/video_accident_ratio_training_results_trimmed_unsignalized_validation_0901.json"))
#     p.add_argument("--eval_json", type=str,
#                    default=os.environ.get("EVAL_JSON", "/app/data/raw/json/text-evaluate/video_accident_caption_results_unsignalized_validation_0901_only_equal_road_delete_main_road.json"))
#     # p.add_argument("--eval_json", type=str,
#     #                default=os.environ.get("EVAL_JSON", "/app/data/raw/json/text-train/video_accident_ratio_unsignalized.json"))
#     p.add_argument("--fault_ckpt", type=str,
#                    default=os.environ.get("FAULT_CKPT", "/app/text-train/fault_ratio_bert.pt"))
#     p.add_argument("--out_json", type=str,
#                    default=os.environ.get("OUT_JSON", "/app/text-train/results_0922_change_prompt_2_2/eval_results_end2end.json"))
#     p.add_argument("--video_root", type=str,
#                    default=os.environ.get("VIDEO_ROOT", "/app/data/raw/videos/validation_reencoded"))
#     # p.add_argument("--video_root", type=str,
#     #                default=os.environ.get("VIDEO_ROOT", "/app/data/raw/videos/training_reencoded"))
#     p.add_argument("--vl_ckpt", type=str,
#                    default=os.environ.get("VL_CKPT", "/app/checkpoints/last_videollava_epoch_hint_drop_02_change_prompt_29_0923.pt"))
#     p.add_argument("--model_name", type=str, default=os.environ.get("MODEL_NAME", "bert-base-uncased"))
#     p.add_argument("--target_basis", type=float, default=float(os.environ.get("TARGET_BASIS", 10.0)))
#     p.add_argument("--max_new_tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", 512)))
#     p.add_argument("--num_frames", type=int, default=int(os.environ.get("NUM_FRAMES", 8)))
#     p.add_argument("--size", type=int, default=int(os.environ.get("SIZE", 224)))
#     # 터미널/진행 표시
#     p.add_argument("--no_tqdm", action="store_true")
#     p.add_argument("--print_every", type=int, default=int(os.environ.get("PRINT_EVERY", 1)))
#     p.add_argument("--quiet", action="store_true", help="샘플별 프린트 끄기")
#     # GPU 분담
#     p.add_argument("--gpus", type=str, default=os.environ.get("GPUS", "0"),
#                    help="쉼표로 구분된 GPU 인덱스 (예: '0' 또는 '0,1')")
#     # 샤딩(병렬 실행시)
#     p.add_argument("--shard_idx", type=int, default=int(os.environ.get("SHARD_IDX", -1)))
#     p.add_argument("--num_shards", type=int, default=int(os.environ.get("NUM_SHARDS", -1)))
#     p.add_argument("--n_captions", type=int, default=5, help="한 비디오당 생성할 캡션 수 (K)")
#     p.add_argument("--ensemble_mode", type=str, default="vote+mean",
#                 choices=["vote","mean","median","vote+mean","vote+median"],
#                 help="vote=문장 다수결만, mean/median=회귀 앙상블만, vote+*=둘 다")
#     return p.parse_args()


# # ------------------------------
# # (H) Single-video quick test (원형 유지)
# # ------------------------------
# def run_fault_from_video(video_path: str, vl_ckpt_path: str, fault_model_path: str,
#                          target_basis: float = 10.0,
#                          device_caption: str = "cuda:0",
#                          device_fault: str = "cuda:0"):
#     vl = load_videollava_from_ckpt(vl_ckpt_path, dtype=torch.float16, device=device_caption)
#     caption = generate_caption_from_video(
#         vl, video_path,
#         prompt=("Describe the accident scene in one sentence. "
#                 "Include intersection type, both vehicles' movements, and entry order."),
#         max_new_tokens=512
#     )
#     fr_model, fr_tok = load_fault_model(fault_model_path, device=device_fault)
#     ratios_basis = predict_fault_ratio(fr_model, fr_tok, caption, device=device_fault)
#     ratios_100 = ratios_basis * (100.0 / target_basis)
#     dashcam, other = ratios_100.tolist()
#     print("-----")
#     print("Caption:", caption)
#     print(f"📊 Predicted Fault Ratio → Dashcam: {dashcam:.1f}%, Other: {other:.1f}%")
#     return {"caption": caption, "dashcam_100": float(dashcam), "other_100": float(other)}


# # ------------------------------
# # (I) Main
# # ------------------------------
# if __name__ == "__main__":
#     args = parse_args()

#     # 경로 점검
#     for k, v in {
#         "eval_json": args.eval_json,
#         "fault_ckpt": args.fault_ckpt,
#         "vl_ckpt": args.vl_ckpt,
#     }.items():
#         if not os.path.exists(v):
#             raise SystemExit(f"[Config] Missing file for --{k}: {v}")
#     if not os.path.isdir(args.video_root):
#         raise SystemExit(f"[Config] --video_root not a directory: {args.video_root}")

#     # GPU 분담 설정
#     gpu_list = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
#     if torch.cuda.is_available() and len(gpu_list) >= 1:
#         device_caption = f"cuda:{gpu_list[0]}"
#         device_fault = f"cuda:{gpu_list[1]}" if len(gpu_list) > 1 else device_caption
#     else:
#         device_caption = device_fault = "cpu"

#     # 메인 평가 실행
#     evaluate_on_json(
#         eval_json_path=args.eval_json,
#         fault_model_path=args.fault_ckpt,
#         out_json_path=args.out_json,
#         model_name=args.model_name,
#         target_basis=args.target_basis,
#         video_root=args.video_root,
#         vl_ckpt_path=args.vl_ckpt,
#         max_new_tokens=args.max_new_tokens,
#         num_frames=args.num_frames,
#         size=args.size,
#         verbose=not args.quiet,
#         print_every=args.print_every,
#         use_tqdm=not args.no_tqdm,
#         device_caption=device_caption,
#         device_fault=device_fault,
#         shard_idx=args.shard_idx,
#         num_shards=args.num_shards,
#     )

# # ===== fault_pipeline.py =====
# # 평가 파이프라인: JSON의 캡션은 절대 쓰지 않고, 항상 비디오 -> Video-LLaVA 캡션 생성 -> 텍스트→과실 예측
# # - HuberLoss/Platt/0–1 정규화 기반 모델과 호환 (평가 시엔 학습된 체크포인트를 로드)
# # - W&B 로깅, CSV/JSON 저장, 스캐터/히스토그램 플롯, 샤딩 지원

# # ===== fault_pipeline.py =====
# # 평가 파이프라인: JSON의 캡션은 절대 쓰지 않고, 항상 비디오 -> Video-LLaVA 캡션 생성 -> 텍스트→과실 예측
# # - 2D 헤드 + softmax + Vector Platt(a,b) 공동학습 모델과 호환
# # - 0–1 정규화(확률)로 해석하고, 평가 시 basis(기본 10)로 재스케일
# # - W&B 로깅, CSV/JSON 저장, 스캐터/히스토그램 플롯, 샤딩 지원

# # import os, re, json, glob, argparse
# # from typing import List, Optional

# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import pandas as pd
# # import matplotlib
# # matplotlib.use("Agg")
# # import matplotlib.pyplot as plt
# # from PIL import Image
# # import wandb
# # from tqdm.auto import tqdm
# # import cv2
# # from transformers import AutoModel, AutoTokenizer

# # # 네 기존 모듈 (Video-LLaVA 래퍼)
# # from train_tracking_video_llava import VideoLLaVACaptioner


# # # ------------------------------
# # # (0) Ratio/Pair 유틸
# # # ------------------------------
# # def snap_pair_to_integer_basis(v, total=10):
# #     v = np.asarray(v, dtype=float)
# #     v = np.maximum(v, 0.0)
# #     s = float(v.sum())
# #     if s <= 0:
# #         a = total // 2
# #         return np.array([a, total - a], dtype=float)
# #     v = v * (total / s)
# #     a_int = int(np.floor(v[0] + 0.5))
# #     a_int = max(0, min(total, a_int))
# #     b_int = total - a_int
# #     return np.array([float(a_int), float(b_int)], dtype=float)

# # def project_pair_to_basis(v, total=10.0):
# #     v = np.maximum(np.asarray(v, dtype=float), 0.0)
# #     s = float(v.sum())
# #     if s <= 0:
# #         return np.array([total/2.0, total/2.0], dtype=float)
# #     return v * (total / s)

# # def ratio_from_pairs(y):  # y: (N,2) basis 스케일
# #     s = np.clip(y.sum(axis=1, keepdims=True), 1e-6, None)
# #     return (y[:, [0]] / s).ravel()  # 대시캠 비율 ∈ (0,1)

# # def pairs_from_ratio(p, total=10.0):
# #     p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
# #     a = total * p
# #     b = total * (1.0 - p)
# #     return np.stack([a, b], axis=1)

# # def calibrate_ratio_binwise(p_hat, p_true, nbins=10):
# #     bins = np.linspace(0.0, 1.0, nbins + 1)
# #     idx = np.clip(np.digitize(p_hat, bins) - 1, 0, nbins - 1)
# #     bin_mean_true = np.zeros(nbins, dtype=float)
# #     for b in range(nbins):
# #         m = (idx == b)
# #         if m.any():
# #             bin_mean_true[b] = float(p_true[m].mean())
# #         else:
# #             bin_mean_true[b] = float((bins[b] + bins[b+1]) / 2.0)
# #     def f(p):
# #         ii = np.clip(np.digitize(p, bins) - 1, 0, nbins - 1)
# #         return bin_mean_true[ii]
# #     return f


# # # ------------------------------
# # # (A) Video frame sampling
# # # ------------------------------
# # def sample_frames_pil(video_path: str, num_frames: int = 8, size: int = 224) -> List[Image.Image]:
# #     cap = cv2.VideoCapture(video_path)
# #     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
# #     if total <= 0:
# #         cap.release()
# #         raise RuntimeError(f"Empty or unreadable video: {video_path}")

# #     if total <= num_frames:
# #         idxs = list(range(total))
# #     else:
# #         step = total / (num_frames + 1)
# #         idxs = [int(step * (i + 1)) for i in range(num_frames)]

# #     frames: List[Image.Image] = []
# #     for idx in idxs:
# #         cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total - 1))
# #         ok, fr = cap.read()
# #         if not ok:
# #             if frames:
# #                 frames.append(frames[-1])
# #             else:
# #                 frames.append(Image.new("RGB", (size, size), color=(0, 0, 0)))
# #             continue
# #         fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
# #         fr = cv2.resize(fr, (size, size))
# #         frames.append(Image.fromarray(fr))
# #     cap.release()

# #     while len(frames) < num_frames:
# #         frames.append(frames[-1])
# #     return frames[:num_frames]


# # # ------------------------------
# # # (B) Video-LLaVA load & caption
# # # ------------------------------
# # def load_videollava_from_ckpt(ckpt_path: str,
# #                               model_id: str = "LanguageBind/Video-LLaVA-7B-hf",
# #                               dtype=torch.float16,
# #                               device: str = "cuda"):
# #     model = VideoLLaVACaptioner(model_id=model_id, lora=False, force_torch_dtype=dtype)
# #     ckpt = torch.load(ckpt_path, map_location="cpu")
# #     if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
# #         missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
# #         if missing or unexpected:
# #             print("[Video-LLaVA] load_state_dict note:",
# #                   f"missing={len(missing)} unexpected={len(unexpected)}")
# #     model.to(device).eval()
# #     return model

# # def postprocess_one_sentence(text: str) -> str:
# #     t = text.strip()
# #     parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', t) if p.strip()]
# #     seen, cleaned = set(), []
# #     for p in parts:
# #         key = re.sub(r'\s+', ' ', p.lower())
# #         if key not in seen:
# #             cleaned.append(p); seen.add(key)
# #     t = ' '.join(cleaned)
# #     idx = t.lower().find("seconds.")
# #     if idx != -1:
# #         return t[:idx + len("seconds.")].strip()
# #     m = re.split(r'(?<=[.!?])\s+', t)
# #     return (m[0].strip() if m else t)

# # @torch.no_grad()
# # def generate_caption_from_video(vl_model,
# #                                 video_path: str,
# #                                 prompt: str,
# #                                 num_frames: int = 8,
# #                                 size: int = 224,
# #                                 max_new_tokens: int = 128) -> str:
# #     frames_pil = sample_frames_pil(video_path, num_frames=num_frames, size=size)
# #     out = vl_model(frames_pil=frames_pil, src_text=prompt,
# #                    generate=True, max_new_tokens=max_new_tokens)
# #     raw = (out.get("text", "") if isinstance(out, dict) else str(out))
# #     return postprocess_one_sentence(raw)


# # # ------------------------------
# # # (C) Text -> Fault model (2D head + Vector Platt + softmax)
# # # ------------------------------
# # class CalibratedFaultPair2D(nn.Module):
# #     """
# #     encoder -> logits(2) -> (a⊙logits + b) -> softmax -> probs(2)
# #     - probs[0]: Dashcam, probs[1]: Other (합=1)
# #     - platt_a, platt_b는 클래스별 스케일/시프트(벡터 스케일링)
# #     """
# #     def __init__(self, model_name="bert-base-uncased", hidden_dim=768):
# #         super().__init__()
# #         self.encoder = AutoModel.from_pretrained(model_name)
# #         self.head = nn.Linear(hidden_dim, 2)          # 2D logits
# #         self.platt_a = nn.Parameter(torch.ones(2))    # [2]
# #         self.platt_b = nn.Parameter(torch.zeros(2))   # [2]

# #     def forward(self, input_ids, attention_mask, return_probs=False):
# #         out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
# #         cls = out.last_hidden_state[:, 0]             # [B, hidden]
# #         logits = self.head(cls)                       # [B, 2]
# #         cal_logits = logits * self.platt_a + self.platt_b
# #         probs = torch.softmax(cal_logits, dim=-1)     # [B, 2], sum=1
# #         if return_probs:
# #             return probs, logits, cal_logits
# #         return probs

# # def load_fault_model(path: str,
# #                      model_name: str = "bert-base-uncased",
# #                      device: str = "cuda"):
# #     obj = torch.load(path, map_location="cpu")
# #     model = CalibratedFaultPair2D(model_name=model_name)  # 2D 헤드 로드
# #     if isinstance(obj, dict) and "state_dict" in obj:
# #         model.load_state_dict(obj["state_dict"], strict=False)
# #     elif isinstance(obj, dict):
# #         model.load_state_dict(obj, strict=False)
# #     elif hasattr(obj, "state_dict"):
# #         model = obj
# #     model.to(device).eval()
# #     tok = AutoTokenizer.from_pretrained(model_name)
# #     return model, tok

# # @torch.no_grad()
# # def predict_fault_ratio(model, tokenizer, text: str, device: str = "cuda",
# #                         max_length: int = 256, target_basis: float = 10.0) -> np.ndarray:
# #     inputs = tokenizer(text, return_tensors="pt",
# #                        padding="max_length", truncation=True, max_length=max_length)
# #     input_ids = inputs["input_ids"].to(device)
# #     attention_mask = inputs["attention_mask"].to(device)
# #     probs = model(input_ids=input_ids, attention_mask=attention_mask)  # [1,2]
# #     probs = probs.squeeze(0).float().clamp(0, 1)                        # [2], sum≈1
# #     pair = (probs * target_basis).cpu().numpy()                         # [dashcam, other]
# #     return pair


# # # ------------------------------
# # # (D) Label utils & video resolver
# # # ------------------------------
# # def normalize_pair_100(p) -> List[float]:
# #     a, b = float(p[0]), float(p[1])
# #     if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
# #         a *= 100.0; b *= 100.0
# #     a = max(a, 0.0); b = max(b, 0.0)
# #     s = a + b
# #     if s == 0.0:
# #         return [50.0, 50.0]
# #     scale = 100.0 / s
# #     return [a * scale, b * scale]

# # def to_basis(pair100: List[float], target_basis: float = 10.0) -> List[float]:
# #     factor = 100.0 / target_basis
# #     return [p / factor for p in pair100]

# # def find_video_file(video_root: str, name_or_rel: str) -> Optional[str]:
# #     cand = os.path.join(video_root, name_or_rel)
# #     if os.path.exists(cand):
# #         return cand
# #     stem = os.path.splitext(name_or_rel)[0]
# #     for ext in (".mp4", ".avi", ".mov", ".mkv"):
# #         p = os.path.join(video_root, stem + ext)
# #         if os.path.exists(p):
# #             return p
# #     g = glob.glob(os.path.join(video_root, stem + "*"))
# #     return g[0] if g else None


# # # ------------------------------
# # # (E) Metrics, plots, saving
# # # ------------------------------
# # def compute_metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
# #     from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# #     mae = mean_absolute_error(y, yhat)
# #     try:
# #         rmse = mean_squared_error(y, yhat, squared=False)
# #     except TypeError:
# #         rmse = np.sqrt(mean_squared_error(y, yhat))
# #     r2 = r2_score(y, yhat)
# #     mae_dc = mean_absolute_error(y[:, 0], yhat[:, 0])
# #     mae_ov = mean_absolute_error(y[:, 1], yhat[:, 1])
# #     return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2),
# #             "MAE_dashcam": float(mae_dc), "MAE_other": float(mae_ov), "count": int(len(y))}

# # def save_plots(y: np.ndarray, yhat: np.ndarray, target_basis: float, out_prefix: str) -> dict:
# #     os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
# #     paths = {}
# #     fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# #     titles = ["Dashcam", "Other Vehicle"]
# #     for i in range(2):
# #         ax[i].scatter(y[:, i], yhat[:, i], alpha=0.5)
# #         ax[i].plot([0, target_basis], [0, target_basis], "--", color="gray")
# #         ax[i].set_title(f"{titles[i]} Fault (basis={target_basis})")
# #         ax[i].set_xlabel("True"); ax[i].set_ylabel("Pred")
# #         ax[i].set_xlim(0, target_basis); ax[i].set_ylim(0, target_basis)
# #     plt.tight_layout()
# #     scatter_path = f"{out_prefix}_scatter.png"
# #     plt.savefig(scatter_path); plt.close(fig)
# #     paths["scatter"] = scatter_path

# #     err = np.abs(yhat - y)
# #     fig2, ax2 = plt.subplots(figsize=(6, 4))
# #     ax2.hist(err[:, 0], bins=20, alpha=0.6, label="Dashcam")
# #     ax2.hist(err[:, 1], bins=20, alpha=0.6, label="Other")
# #     ax2.set_title(f"Absolute Error Histogram (basis={target_basis})")
# #     ax2.set_xlabel("Abs Error"); ax2.set_ylabel("Count"); ax2.legend()
# #     hist_path = f"{out_prefix}_err_hist.png"
# #     plt.tight_layout(); plt.savefig(hist_path); plt.close(fig2)
# #     paths["err_hist"] = hist_path
# #     return paths

# # def df_to_csv(results: List[dict], out_csv_path: str):
# #     os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
# #     df = pd.DataFrame(results)
# #     df.to_csv(out_csv_path, index=False, encoding="utf-8")


# # # ------------------------------
# # # (F) Evaluation (JSON 캡션 절대 사용 X)
# # # ------------------------------
# # def evaluate_on_json(eval_json_path: str,
# #                      fault_model_path: str,
# #                      out_json_path: str,
# #                      model_name: str = "bert-base-uncased",
# #                      target_basis: float = 10.0,
# #                      video_root: str = None,
# #                      vl_ckpt_path: str = None,
# #                      max_new_tokens: int = 128,
# #                      num_frames: int = 8,
# #                      size: int = 224,
# #                      verbose: bool = True,
# #                      print_every: int = 1,
# #                      use_tqdm: bool = True,
# #                      device_caption: str = "cuda:0",
# #                      device_fault: str = "cuda:0",
# #                      shard_idx: int = -1,
# #                      num_shards: int = -1):

# #     if not os.path.exists(eval_json_path):
# #         raise FileNotFoundError(eval_json_path)
# #     if not os.path.exists(vl_ckpt_path or ""):
# #         raise FileNotFoundError(vl_ckpt_path)
# #     if not os.path.isdir(video_root or ""):
# #         raise NotADirectoryError(video_root)

# #     wandb.init(
# #         project="end-to-end-accident-caption",
# #         config={
# #             "eval_json": eval_json_path,
# #             "fault_ckpt": fault_model_path,
# #             "video_root": video_root,
# #             "vl_ckpt": vl_ckpt_path,
# #             "model_name": model_name,
# #             "target_basis": target_basis,
# #             "num_frames": num_frames,
# #             "size": size,
# #             "max_new_tokens": max_new_tokens,
# #             "device_caption": device_caption,
# #             "device_fault": device_fault,
# #             "shard_idx": shard_idx,
# #             "num_shards": num_shards,
# #         },
# #         job_type="evaluation",
# #     )

# #     vl_model = load_videollava_from_ckpt(vl_ckpt_path, device=device_caption)
# #     fr_model, fr_tok = load_fault_model(fault_model_path, model_name=model_name, device=device_fault)

# #     with open(eval_json_path, "r", encoding="utf-8") as f:
# #         data = json.load(f)
# #     if not isinstance(data, list):
# #         raise ValueError("eval_json must be a list of objects.")
# #     if shard_idx >= 0 and num_shards > 0:
# #         idxs = np.array_split(np.arange(len(data)), num_shards)[shard_idx].tolist()
# #         data = [data[i] for i in idxs]

# #     caption_prompt = (
# #         "This is a first-person dashcam (ego-view) video. 'Dashcam Vehicle' = the recording car.\n"
# #         "OUTPUT EXACTLY ONE sentence by filling ONLY the 4 slots in the TEMPLATE below.\n"
# #         "TEMPLATE (copy every word EXACTLY, replace only the bracketed slots):\n"
# #         "At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, "
# #         "while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>.\n\n"
# #         "ALLOWED VALUES (copy verbatim, NOTHING ELSE):\n"
# #         "- <mv_*> = {going straight, left turn, right turn}\n"
# #         "- <side_*> = {from right road, from left road, from main road, from side road left, from side road right}\n"
# #         "- <who_entered> = {Dashcam, Other}\n"
# #         "- <earlier_or_later> = {earlier, later}\n\n"
# #         "HARD RULES:\n"
# #         "- Do NOT invent words like 'following vehicle', 'lanes', 'light', 'signal', 'time', etc.\n"
# #         "- Do NOT change anything outside the 4 slots. Copy the TEMPLATE text exactly.\n"
# #         "- Use singular 'Other Vehicle' exactly. NEVER write Vegetable, Vehitcle, Vehicel, Vehov etc.\n"
# #         "- Do NOT add or remove commas/words/punctuation.\n"
# #         "- Final output must be ONE sentence ending with a period.\n\n"
# #         "GOOD EXAMPLE (valid):\n"
# #         "At an unsignalized intersection, the Dashcam Vehicle was going straight from main road, "
# #         "while the Other Vehicle was right turn from side road left, and the Other Vehicle entered earlier.\n\n"
# #         "BAD EXAMPLE (invalid):\n"
# #         "At an unsignalized intersection, the Dashcam Vehicle was following vehicle of the other vehicle.  <-- WRONG: not in allowed values\n"
# #         "At an uncontrolled intersection, the dashcam vehicle was going ahead.  <-- WRONG: template not copied, invalid words\n"
# #     )

# #     preds_basis, labels_basis = [], []
# #     results = []
# #     check_list_video = []

# #     N = len(data)
# #     iterator = tqdm(range(N), total=N, desc="Evaluating", dynamic_ncols=True) if use_tqdm else range(N)

# #     for i in iterator:
# #         row = data[i]

# #         # GT
# #         gt_basis = None
# #         gt100 = None
# #         if "dashcam_vehicle_negligence" in row and "other_vehicle_negligence" in row:
# #             gt100 = normalize_pair_100([row["dashcam_vehicle_negligence"], row["other_vehicle_negligence"]])
# #             gt_basis = np.array(to_basis(gt100, target_basis), dtype=float)

# #         # 비디오
# #         video_name = row.get("video_name") or row.get("video_path") or ""
# #         vpath = find_video_file(video_root, video_name) if video_name else None
# #         if not vpath:
# #             raw = row.get("video_path")
# #             if raw and os.path.exists(raw):
# #                 vpath = raw
# #         if not vpath:
# #             out_item = {"idx": i, "video_name": video_name, "error": "video_not_found"}
# #             results.append(out_item)
# #             if verbose:
# #                 print(f"[{i+1}/{N}] {video_name} | VIDEO NOT FOUND", flush=True)
# #             continue

# #         # 캡션
# #         try:
# #             caption = generate_caption_from_video(
# #                 vl_model, vpath, prompt=caption_prompt,
# #                 num_frames=num_frames, size=size, max_new_tokens=max_new_tokens
# #             )
# #         except Exception as e:
# #             out_item = {"idx": i, "video_name": video_name, "error": f"caption_generation_failed: {e}"}
# #             results.append(out_item)
# #             if verbose:
# #                 print(f"[{i+1}/{N}] {video_name} | CAPTION ERROR: {e}", flush=True)
# #             continue

# #         # 예측 (probs -> basis pair)
# #         pred_basis = predict_fault_ratio(fr_model, fr_tok, caption, device=device_fault, target_basis=target_basis)
# #         pred_basis = project_pair_to_basis(pred_basis, total=target_basis)  # 안정 장치
# #         preds_basis.append(pred_basis)
# #         if gt_basis is not None:
# #             labels_basis.append(gt_basis)

# #         pred_100 = [float(x) * (100.0 / target_basis) for x in pred_basis]

# #         out_item = {
# #             "idx": i,
# #             "video_name": video_name,
# #             "caption_pred": caption,
# #             "pred_basis_dashcam": float(pred_basis[0]),
# #             "pred_basis_other": float(pred_basis[1]),
# #             "pred_100_dashcam": float(pred_100[0]),
# #             "pred_100_other": float(pred_100[1]),
# #         }
# #         if gt_basis is not None:
# #             out_item["gt_basis_dashcam"] = float(gt_basis[0])
# #             out_item["gt_basis_other"] = float(gt_basis[1])
# #             out_item["gt_100_dashcam"] = float(gt100[0])
# #             out_item["gt_100_other"] = float(gt100[1])
# #             out_item["abs_err_basis_dashcam"] = abs(out_item["gt_basis_dashcam"] - out_item["pred_basis_dashcam"])
# #             out_item["abs_err_basis_other"] = abs(out_item["gt_basis_other"] - out_item["pred_basis_other"])

# #         results.append(out_item)

# #         if verbose and ((i == 0) or ((i + 1) % print_every == 0) or (i + 1 == N)):
# #             msg = f"[{i+1}/{N}] {video_name} | pred_basis=[{out_item['pred_basis_dashcam']:.2f}, {out_item['pred_basis_other']:.2f}]"
# #             tgt_text = str(row.get('generated_caption', "No description available")).strip()
# #             m = re.search(r"\bthe collision\b", tgt_text, flags=re.IGNORECASE)
# #             if m:
# #                 tgt_text = tgt_text[:m.start()].rstrip()
# #                 if not tgt_text.endswith(('.', '!', '?')):
# #                     tgt_text += '.'

# #             if gt_basis is not None:
# #                 ae_dc = out_item["abs_err_basis_dashcam"]
# #                 ae_ov = out_item["abs_err_basis_other"]
# #                 if ae_dc >= 2.0 or ae_ov >= 2.0:
# #                     check_list_video.append([video_name, caption, tgt_text])
# #                 msg += f" | gt_basis=[{out_item['gt_basis_dashcam']:.2f}, {out_item['gt_basis_other']:.2f}] | abs_err=[{ae_dc:.2f}, {ae_ov:.2f}]"

# #             msg += "\n"
# #             msg += f"⭐️pred_caption: {caption}\n" + f"⭐️gt_caption: {tgt_text}"
# #             print(msg, flush=True)

# #         if (i + 1) % 20 == 0:
# #             wandb.log({"eval_progress_samples": i + 1})

# #     # 메트릭 & 저장
# #     metrics = {}
# #     out_prefix = os.path.splitext(out_json_path)[0]
# #     os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)

# #     pred_rows = [r for r in results if "pred_basis_dashcam" in r]
# #     if len(pred_rows) == 0:
# #         out_csv_path = f"{out_prefix}.csv"
# #         df_to_csv(results, out_csv_path)
# #         with open(out_json_path, "w", encoding="utf-8") as f:
# #             json.dump({"metrics": {}, "results": results}, f, ensure_ascii=False, indent=2)
# #         print("=== Evaluation Summary (JSON captions NOT used) ===")
# #         print(json.dumps({}, indent=2, ensure_ascii=False))
# #         print(f"Saved JSON: {out_json_path}")
# #         print(f"Saved CSV : {out_csv_path}")
# #         print("Check list (abs err basis >=2.0):", len(check_list_video))
# #         print(check_list_video)
# #         wandb.finish()
# #         return

# #     # (1) 배열로 정리
# #     yhat_all = np.array([[r["pred_basis_dashcam"], r["pred_basis_other"]] for r in pred_rows], dtype=float)
# #     mask_gt = np.array([("gt_basis_dashcam" in r) for r in pred_rows], dtype=bool)

# #     if mask_gt.any():
# #         y_gt    = np.array([[pred_rows[i]["gt_basis_dashcam"], pred_rows[i]["gt_basis_other"]] for i in range(len(pred_rows)) if mask_gt[i]], dtype=float)
# #         yhat_gt = yhat_all[mask_gt]

# #         # 보정 전
# #         metrics_pre = compute_metrics(y_gt, yhat_gt)
# #         metrics_pre["target_basis"] = target_basis

# #         # 비율 기반 등화 (joint-Platt 이후, optional isotonic 보정)
# #         p_hat_gt = ratio_from_pairs(yhat_gt)
# #         p_true_gt = ratio_from_pairs(y_gt)

# #         try:
# #             from sklearn.isotonic import IsotonicRegression
# #             iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True).fit(p_hat_gt, p_true_gt)
# #             p_cal_all = iso.transform(ratio_from_pairs(yhat_all))
# #         except Exception as e:
# #             print("[cal] isotonic failed; fallback to bin-wise:", e)
# #             fbin = calibrate_ratio_binwise(p_hat_gt, p_true_gt, nbins=10)
# #             p_cal_all = fbin(ratio_from_pairs(yhat_all))

# #         yhat_cal_all = pairs_from_ratio(p_cal_all, total=target_basis)
# #         yhat_cal_all = np.vstack([project_pair_to_basis(v, total=target_basis) for v in yhat_cal_all])

# #         for k, r in enumerate(pred_rows):
# #             r["pred_basis_dashcam_cal"] = float(yhat_cal_all[k, 0])
# #             r["pred_basis_other_cal"]   = float(yhat_cal_all[k, 1])

# #         yhat_cal_gt = yhat_cal_all[mask_gt]
# #         metrics_cal = compute_metrics(y_gt, yhat_cal_gt)

# #         metrics = {**metrics_pre, **{f"cal/{k}": v for k, v in metrics_cal.items()}}

# #         plot_paths = save_plots(y_gt, yhat_gt, target_basis, out_prefix + "_precal")
# #         plot_paths_cal = save_plots(y_gt, yhat_cal_gt, target_basis, out_prefix + "_cal")
# #         plot_paths.update({("cal_" + k): v for k, v in plot_paths_cal.items()})

# #         out_csv_path = f"{out_prefix}.csv"
# #         pd.DataFrame(results).to_csv(out_csv_path, index=False, encoding="utf-8")
# #         with open(out_json_path, "w", encoding="utf-8") as f:
# #             json.dump({"metrics": metrics, "results": results}, f, ensure_ascii=False, indent=2)

# #         wandb.log({
# #             "eval/MAE": metrics["MAE"],
# #             "eval/RMSE": metrics["RMSE"],
# #             "eval/R2": metrics["R2"],
# #             "eval/MAE_dashcam": metrics["MAE_dashcam"],
# #             "eval/MAE_other": metrics["MAE_other"],
# #             "eval_cal/MAE": metrics["cal/MAE"],
# #             "eval_cal/RMSE": metrics["cal/RMSE"],
# #             "eval_cal/R2": metrics["cal/R2"],
# #             "eval_cal/MAE_dashcam": metrics["cal/MAE_dashcam"],
# #             "eval_cal/MAE_other": metrics["cal/MAE_other"],
# #         })
# #         for k, pth in plot_paths.items():
# #             if os.path.exists(pth):
# #                 wandb.log({f"plots/{k}": wandb.Image(pth)})

# #         try:
# #             table = wandb.Table(dataframe=pd.DataFrame(results))
# #             wandb.log({"eval/table": table})
# #         except Exception:
# #             pass
# #         try:
# #             art = wandb.Artifact("eval_results", type="evaluation")
# #             art.add_file(out_json_path); art.add_file(out_csv_path)
# #             for p in plot_paths.values():
# #                 if os.path.exists(p):
# #                     art.add_file(p)
# #             wandb.log_artifact(art)
# #         except Exception as e:
# #             print("W&B artifact upload failed:", e)

# #     else:
# #         out_csv_path = f"{out_prefix}.csv"
# #         df_to_csv(results, out_csv_path)
# #         with open(out_json_path, "w", encoding="utf-8") as f:
# #             json.dump({"metrics": {}, "results": results}, f, ensure_ascii=False, indent=2)

# #     print("=== Evaluation Summary (JSON captions NOT used) ===")
# #     print(json.dumps(metrics, indent=2, ensure_ascii=False))
# #     print(f"Saved JSON: {out_json_path}")
# #     print(f"Saved CSV : {os.path.splitext(out_json_path)[0]}.csv")
# #     print("Check list (abs err basis >=2.0):", len(check_list_video))
# #     print(check_list_video)
# #     wandb.finish()


# # # ------------------------------
# # # (G) CLI
# # # ------------------------------
# # def parse_args():
# #     p = argparse.ArgumentParser(description="End-to-end evaluation from video via Video-LLaVA.")
# #     p.add_argument("--eval_json", type=str,
# #                    default=os.environ.get("EVAL_JSON", "/app/data/raw/json/text-evaluate/video_accident_ratio_training_results_trimmed_unsignalized_validation_0901.json"))
# #     p.add_argument("--fault_ckpt", type=str,
# #                    default=os.environ.get("FAULT_CKPT", "/app/text-train/fault_ratio_bert.pt"))
# #     p.add_argument("--out_json", type=str,
# #                    default=os.environ.get("OUT_JSON", "/app/text-train/results_0922_huber_platt_2d/eval_results_end2end.json"))
# #     p.add_argument("--video_root", type=str,
# #                    default=os.environ.get("VIDEO_ROOT", "/app/data/raw/videos/validation_reencoded"))
# #     p.add_argument("--vl_ckpt", type=str,
# #                    default=os.environ.get("VL_CKPT", "/app/checkpoints/last_videollava_epoch_hint_drop_02_change_prompt_23_0922.pt"))
# #     p.add_argument("--model_name", type=str, default=os.environ.get("MODEL_NAME", "bert-base-uncased"))
# #     p.add_argument("--target_basis", type=float, default=float(os.environ.get("TARGET_BASIS", 10.0)))
# #     p.add_argument("--max_new_tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", 128)))
# #     p.add_argument("--num_frames", type=int, default=int(os.environ.get("NUM_FRAMES", 8)))
# #     p.add_argument("--size", type=int, default=int(os.environ.get("SIZE", 224)))
# #     p.add_argument("--no_tqdm", action="store_true")
# #     p.add_argument("--print_every", type=int, default=int(os.environ.get("PRINT_EVERY", 1)))
# #     p.add_argument("--quiet", action="store_true", help="샘플별 프린트 끄기")
# #     p.add_argument("--gpus", type=str, default=os.environ.get("GPUS", "0"),
# #                    help="쉼표로 구분된 GPU 인덱스 (예: '0' 또는 '0,1')")
# #     p.add_argument("--shard_idx", type=int, default=int(os.environ.get("SHARD_IDX", -1)))
# #     p.add_argument("--num_shards", type=int, default=int(os.environ.get("NUM_SHARDS", -1)))
# #     return p.parse_args()


# # # ------------------------------
# # # (H) Single-video quick test
# # # ------------------------------
# # def run_fault_from_video(video_path: str, vl_ckpt_path: str, fault_model_path: str,
# #                          target_basis: float = 10.0,
# #                          device_caption: str = "cuda:0",
# #                          device_fault: str = "cuda:0"):
# #     vl = load_videollava_from_ckpt(vl_ckpt_path, dtype=torch.float16, device=device_caption)
# #     caption = generate_caption_from_video(
# #         vl, video_path,
# #         prompt=("Describe the accident scene in one sentence. "
# #                 "Include intersection type, both vehicles' movements, and entry order."),
# #         max_new_tokens=128
# #     )
# #     fr_model, fr_tok = load_fault_model(fault_model_path, device=device_fault)
# #     pair_basis = predict_fault_ratio(fr_model, fr_tok, caption, device=device_fault, target_basis=target_basis)
# #     ratios_100 = pair_basis * (100.0 / target_basis)
# #     dashcam, other = ratios_100.tolist()
# #     print("-----")
# #     print("Caption:", caption)
# #     print(f"📊 Predicted Fault Ratio → Dashcam: {dashcam:.1f}%, Other: {other:.1f}%")
# #     return {"caption": caption, "dashcam_100": float(dashcam), "other_100": float(other)}


# # # ------------------------------
# # # (I) Main
# # # ------------------------------
# # if __name__ == "__main__":
# #     args = parse_args()

# #     for k, v in {
# #         "eval_json": args.eval_json,
# #         "fault_ckpt": args.fault_ckpt,
# #         "vl_ckpt": args.vl_ckpt,
# #     }.items():
# #         if not os.path.exists(v):
# #             raise SystemExit(f"[Config] Missing file for --{k}: {v}")
# #     if not os.path.isdir(args.video_root):
# #         raise SystemExit(f"[Config] --video_root not a directory: {args.video_root}")

# #     gpu_list = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
# #     if torch.cuda.is_available() and len(gpu_list) >= 1:
# #         device_caption = f"cuda:{gpu_list[0]}"
# #         device_fault = f"cuda:{gpu_list[1]}" if len(gpu_list) > 1 else device_caption
# #     else:
# #         device_caption = device_fault = "cpu"

# #     evaluate_on_json(
# #         eval_json_path=args.eval_json,
# #         fault_model_path=args.fault_ckpt,
# #         out_json_path=args.out_json,
# #         model_name=args.model_name,
# #         target_basis=args.target_basis,
# #         video_root=args.video_root,
# #         vl_ckpt_path=args.vl_ckpt,
# #         max_new_tokens=args.max_new_tokens,
# #         num_frames=args.num_frames,
# #         size=args.size,
# #         verbose=not args.quiet,
# #         print_every=args.print_every,
# #         use_tqdm=not args.no_tqdm,
# #         device_caption=device_caption,
# #         device_fault=device_fault,
# #         shard_idx=args.shard_idx,
# #         num_shards=args.num_shards,
# #     )





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

def snap_pair_to_integer_basis(v, total=10):
    """
    v: np.array([a, b]) (합=total 근처)
    반환: 합=total, 두 값 모두 '정수'인 쌍
    """
    v = np.asarray(v, dtype=float)
    # 음수 방지 + 합 재정규화
    v = np.maximum(v, 0.0)
    s = float(v.sum())
    if s <= 0:
        a = total // 2
        return np.array([a, total - a], dtype=float)
    v = v * (total / s)

    # 대시캠을 반올림 정수로, 나머지는 보전
    a_int = int(np.floor(v[0] + 0.5))
    a_int = max(0, min(total, a_int))
    b_int = total - a_int
    return np.array([float(a_int), float(b_int)], dtype=float)

def project_pair_to_basis(v, total=10.0):
    """임의의 (a,b)를 합=total로 투영. a,b<0 방지."""
    v = np.maximum(np.asarray(v, dtype=float), 0.0)
    s = float(v.sum())
    if s <= 0:
        return np.array([total/2.0, total/2.0], dtype=float)
    return v * (total / s)

def ratio_from_pairs(y):  # y: (N,2) basis 스케일
    s = np.clip(y.sum(axis=1, keepdims=True), 1e-6, None)
    return (y[:, [0]] / s).ravel()  # 대시캠 비율 ∈ (0,1)

def pairs_from_ratio(p, total=10.0):
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    a = total * p
    b = total * (1.0 - p)
    return np.stack([a, b], axis=1)

def calibrate_ratio_isotonic(p_hat, p_true):
    """단조(증가) 등화. scikit-learn 필요."""
    try:
        from sklearn.isotonic import IsotonicRegression
    except Exception as e:
        raise RuntimeError("IsotonicRegression을 쓸 수 없습니다(sklearn 미설치?).") from e
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True)
    iso.fit(p_hat, p_true)
    return iso  # .transform(x)로 사용

def calibrate_ratio_binwise(p_hat, p_true, nbins=10):
    """등간격 bin-wise 평균 매핑. 리턴: 함수 f(p)->보정p"""
    bins = np.linspace(0.0, 1.0, nbins + 1)
    idx = np.clip(np.digitize(p_hat, bins) - 1, 0, nbins - 1)
    bin_mean_true = np.zeros(nbins, dtype=float)
    for b in range(nbins):
        m = (idx == b)
        if m.any():
            bin_mean_true[b] = float(p_true[m].mean())
        else:
            bin_mean_true[b] = float((bins[b] + bins[b+1]) / 2.0)

    def f(p):
        ii = np.clip(np.digitize(p, bins) - 1, 0, nbins - 1)
        return bin_mean_true[ii]
    return f

# def project_pair_to_basis(v, total=10.0):
#     v = np.asarray(v, dtype=float)
#     v = np.maximum(v, 0.0)
#     s = float(v.sum())
#     if s == 0.0:
#         return np.array([total/2.0, total/2.0], dtype=float)
#     return v * (total / s)

# ------------------------------
# (A) Video frame sampling
# ------------------------------
def sample_frames_pil(video_path: str, num_frames: int = 8, size: int = 224) -> List[Image.Image]:
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
            # fallback: 마지막 프레임 복제 또는 검은 화면
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
# (B) Video-LLaVA load & caption
# ------------------------------
def load_videollava_from_ckpt(ckpt_path: str,
                              model_id: str = "LanguageBind/Video-LLaVA-7B-hf",
                              dtype=torch.float16,
                              device: str = "cuda"):
    model = VideoLLaVACaptioner(model_id=model_id, lora=False, force_torch_dtype=dtype)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing or unexpected:
            print("[Video-LLaVA] load_state_dict note:",
                  f"missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()
    return model


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
import re

TEMPLATE_NEEDS = [
    "At an unsignalized intersection, the Dashcam Vehicle was",
    "while the Other Vehicle was",
    "Vehicle entered",
]

ALLOWED_MV   = {"going straight", "left turn", "right turn"}
ALLOWED_SIDE = {"from right road", "from left road", "from main road", "from side road left", "from side road right"}
ALLOWED_WHO  = {"Dashcam", "Other"}
ALLOWED_EARL = {"earlier", "later"}
import collections

def parse_slots_from_caption(sent: str) -> dict:
    """
    템플릿 문장에서 6개 슬롯 추출.
    실패하면 {} 반환.
    """
    s = " ".join(sent.strip().split())
    try:
        head, rest = s.split("the Dashcam Vehicle was", 1)
        dv_part, rest = rest.split(", while the Other Vehicle was", 1)
        ov_part, tail = rest.split(", and the", 1)
        who_part, end = tail.split("Vehicle entered", 1)

        dv_part = dv_part.strip()
        ov_part = ov_part.strip()
        who_part = who_part.strip()
        earl = end.strip(" .").strip()

        mv_dv = next((mv for mv in ALLOWED_MV   if dv_part.startswith(mv)), None)
        side_dv = next((sd for sd in ALLOWED_SIDE if sd in dv_part), None)
        mv_ov = next((mv for mv in ALLOWED_MV   if ov_part.startswith(mv)), None)
        side_ov = next((sd for sd in ALLOWED_SIDE if sd in ov_part), None)
        who_entered = "Dashcam" if "Dashcam" in who_part else ("Other" if "Other" in who_part else None)
        earlier_or_later = "earlier" if "earlier" in earl else ("later" if "later" in earl else None)

        slots = {
            "mv_dv": mv_dv, "side_dv": side_dv,
            "mv_ov": mv_ov, "side_ov": side_ov,
            "who_entered": who_entered, "earlier_or_later": earlier_or_later,
        }
        return {} if any(v is None for v in slots.values()) else slots
    except Exception:
        return {}

def vote_slots(slots_list: list[dict]) -> dict:
    """각 슬롯 최빈값 선택(동수면 첫 등장 우선)."""
    out = {}
    for key in ["mv_dv","side_dv","mv_ov","side_ov","who_entered","earlier_or_later"]:
        vals = [s[key] for s in slots_list if s.get(key)]
        if not vals:
            out[key] = None
            continue
        cnt = collections.Counter(vals)
        out[key] = cnt.most_common(1)[0][0]
    return out

def render_sentence(slots: dict) -> str:
    if any(slots.get(k) is None for k in ["mv_dv","side_dv","mv_ov","side_ov","who_entered","earlier_or_later"]):
        return ""
    return (f"At an unsignalized intersection, the Dashcam Vehicle was {slots['mv_dv']} {slots['side_dv']}, "
            f"while the Other Vehicle was {slots['mv_ov']} {slots['side_ov']}, "
            f"and the {slots['who_entered']} Vehicle entered {slots['earlier_or_later']}.")

def _looks_complete_template(txt: str) -> bool:
    t = " ".join(txt.split())
    if not t.endswith("."):
        return False
    # 필수 구간 존재?
    if not all(k in t for k in TEMPLATE_NEEDS):
        return False
    # 슬롯 값이 하나라도 들어갔는지 대충 검사(엄격 매칭 아님)
    has_mv   = any(k in t for k in ALLOWED_MV)
    has_side = any(k in t for k in ALLOWED_SIDE)
    has_who  = any(k in t for k in ALLOWED_WHO)
    has_e    = any(k in t for k in ALLOWED_EARL)
    num_slots = sum([has_mv, has_side, has_who, has_e])
    return num_slots >= 2

def _clean_sentence(txt: str) -> str:
    t = " ".join(txt.split())
    # 복수형/오타 흔들림 정리
    t = t.replace("Other Vehicles were", "Other Vehicle was")
    t = re.sub(r"\s+", " ", t).strip()
    if not t.endswith("."):
        t += "."
    return t

@torch.no_grad()
def generate_k_captions_from_video(vl_model, video_path, prompt,
                                   K=5, num_frames=8, size=224, max_new_tokens=128) -> list[str]:
    frames_pil = sample_frames_pil(video_path, num_frames=num_frames, size=size)
    captions = []

    # 빔서치/샘플링 혼합으로 다양성 확보
    schedules = [
        dict(do_sample=False, num_beams=3,  num_return_sequences=min(3, K), max_new_tokens=max_new_tokens, no_repeat_ngram_size=3),
        dict(do_sample=True,  top_p=0.9, temperature=0.7, num_beams=1, num_return_sequences=K, max_new_tokens=max_new_tokens, no_repeat_ngram_size=3),
    ]
    for sch in schedules:
        out = vl_model(frames_pil=frames_pil, src_text=prompt, generate=True, **sch)
        txt = out.get("text", None) if isinstance(out, dict) else None
        if isinstance(txt, list):
            candidates = txt
        elif isinstance(txt, str):
            candidates = [txt]
        else:
            candidates = []
        for t in candidates:
            t = postprocess_one_sentence((t or "").strip())
            if t and t not in captions:
                captions.append(t)
        if len(captions) >= K:
            break

    # 모자라면 단건 샘플링으로 채우기
    while len(captions) < K:
        out = vl_model(frames_pil=frames_pil, src_text=prompt, generate=True,
                       do_sample=True, top_p=0.9, temperature=0.7, num_beams=1, min_new_tokens=40,
                       max_new_tokens=max_new_tokens, no_repeat_ngram_size=3)
        t = postprocess_one_sentence((out.get("text", "") if isinstance(out, dict) else str(out)).strip())
        if t and t not in captions:
            captions.append(t)

    return captions[:K]

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
# def normalize_pair_100(p) -> List[float]:
#     a, b = float(p[0]), float(p[1])
#     s = a + b
#     if 0 <= a <= 1 and 0 <= b <= 1:
#         a *= 100.0; b *= 100.0; s = a + b
#     if s > 0 and abs(s - 100.0) > 1e-6:
#         a *= (100.0 / s); b *= (100.0 / s)
#     return [max(0.0, min(100.0, a)), max(0.0, min(100.0, b))]
def normalize_pair_100(p) -> List[float]:
    a, b = float(p[0]), float(p[1])
    # [0,1] 스케일이면 100으로 확장
    if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
        a *= 100.0; b *= 100.0
    # 음수 컷 → 그 뒤 반드시 재정규화
    a = max(a, 0.0); b = max(b, 0.0)
    s = a + b
    if s == 0.0:
        return [50.0, 50.0]
    scale = 100.0 / s
    return [a * scale, b * scale]


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
        project="end-to-end-accident-caption",
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

    # 모델 로드 (GPU 분담)
    vl_model = load_videollava_from_ckpt(vl_ckpt_path, device=device_caption)
    fr_model, fr_tok = load_fault_model(fault_model_path, model_name=model_name, device=device_fault)

    # 데이터 적재 (+샤딩)
    with open(eval_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("eval_json must be a list of objects.")
    if shard_idx >= 0 and num_shards > 0:
        idxs = np.array_split(np.arange(len(data)), num_shards)[shard_idx].tolist()
        data = [data[i] for i in idxs]

    # caption_prompt = (
    #     "Task: Describe the accident scene in one sentence.\n"
    #     "You MUST include:\n"
    #     "- intersection type\n"
    #     "- the movement and direction of both vehicles (Dashcam and Other)\n"
    #     "- entry order (who entered first)\n"
    #     "Avoid adding traffic signals or unrelated details."
    # )
    # caption_prompt = (
    #         "This is a first-person dashcam (ego-view) video. 'Dashcam Vehicle' is the recording car (ego).\n"
    #         "ONE sentence. Use this EXACT template:\n"
    #         "At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, "
    #         "while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>.\n"
    #         "Values: <mv_*>={going straight,left turn,right turn}; "
    #         "<side_*>={from the right,from left road,from main road,from side road left,from side road right}; "
    #         "<who_entered>={Dashcam,Other}; <earlier_or_later>={earlier,later}.\n"
    #         "No extra words. Use singular 'Other Vehicle'. Do NOT mention camera/ego/first-person in the sentence."
    # )
    # caption_prompt = (
    #         "This is a first-person dashcam (ego-view) video. 'Dashcam Vehicle' = the recording car.\n"
    #         "OUTPUT: ONE sentence in THIS EXACT template (replace <> only):\n"
    #         "At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, "
    #         "while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>.\n"
    #         "Allowed: <mv>={going straight,left turn,right turn}; "
    #         "<side>={from the right,from left road,from main road,from side road left,from side road right}; "
    #         "<who_entered>={Dashcam,Other}; <earlier_or_later>={earlier,later}.\n"
    #         "If '<SCENE_HINTS>' appears below, copy values EXACTLY.\n"
    #         "Return ONLY the sentence. Do NOT mention camera/ego/first-person."
    # )
    # caption_prompt = (
    #     "You are extracting 6 categorical slots for an unsignalized intersection scene.\n"
    #     "Return ONLY a single-line JSON object with these EXACT keys:\n"
    #     '{"mv_dv": "...", "side_dv": "...", "mv_ov": "...", "side_ov": "...", "who_entered": "...", "earlier_or_later": "..."}\n\n'
    #     "Allowed values (copy verbatim):\n"
    #     '- \"mv_dv\", \"mv_ov\" ∈ {\"going straight\",\"left turn\",\"right turn\"}\n'
    #     '- \"side_dv\", \"side_ov\" ∈ {\"from right road\",\"from left road\",\"from main road\",\"from side road left\",\"from side road right\"}\n'
    #     '- \"who_entered\" ∈ {\"Dashcam\",\"Other\"}\n'
    #     '- \"earlier_or_later\" ∈ {\"earlier\",\"later\"}\n\n'
    #     "Hard rules:\n"
    #     "- Output MUST be valid JSON on ONE line. No code fences, no explanations.\n"
    #     "- Use ONLY the allowed values (verbatim). No synonyms, no extra words.\n"
    #     "- Do NOT mention lights, signals, lanes, time, date, crash/collision.\n"
    #     "- Do NOT output any sentence. Only the JSON object.\n\n"
    #     "Good example:\n"
    #     '{"mv_dv":"going straight","side_dv":"from main road","mv_ov":"right turn","side_ov":"from side road left","who_entered":"Other","earlier_or_later":"earlier"}'
    # )
    # 이게 성능 가장 좋음
    # caption_prompt = (
    #     "This is a first-person dashcam (ego-view) video. 'Dashcam Vehicle' = the recording car.\n"
    #     "OUTPUT EXACTLY ONE sentence by filling ONLY the 4 slots in the TEMPLATE below.\n"
    #     "TEMPLATE (copy every word EXACTLY, replace only the bracketed slots):\n"
    #     "At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, "
    #     "while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>.\n\n"
    #     "ALLOWED VALUES (copy verbatim, NOTHING ELSE):\n"
    #     "- <mv_*> = {going straight, left turn, right turn}\n"
    #     "- <side_*> = {from right road, from left road, from main road, from side road left, from side road right}\n"
    #     "- <who_entered> = {Dashcam, Other}\n"
    #     "- <earlier_or_later> = {earlier, later}\n\n"
    #     "HARD RULES:\n"
    #     "- Do NOT invent words like 'following vehicle', 'lanes', 'light', 'signal', 'time', etc.\n"
    #     "- Do NOT change anything outside the 4 slots. Copy the TEMPLATE text exactly.\n"
    #     "- Use singular 'Other Vehicle' exactly. NEVER write Vegetable, Vehitcle, Vehicel, Vehov, Vehicles, Vehi, Veh, Ve etc.\n"
    #     "- Do NOT add or remove commas/words/punctuation.\n"
    #     "- Final output must be ONE sentence ending with a period.\n\n"
    #     "GOOD EXAMPLE (valid):\n"
    #     "At an unsignalized intersection, the Dashcam Vehicle was going straight from main road, "
    #     "while the Other Vehicle was right turn from side road left, and the Other Vehicle entered earlier.\n\n"
    #     "BAD EXAMPLE (invalid):\n"
    #     "At an unsignalized intersection, the Dashcam Vehicle was following vehicle of the other vehicle.  <-- WRONG: not in allowed values\n"
    #     "At an uncontrolled intersection, the dashcam vehicle was going ahead.  <-- WRONG: template not copied, invalid words\n"
    # )
    # caption_prompt = (
    #     "NEVER write Vegetable, Vehitcle, Vehicel, Vehov, Vehicles, Vehi, Veh, Ve etc.\n"
    #     "This is a first-person dashcam (ego-view) video. 'Dashcam Vehicle' = the recording car.\n"
    #     "OUTPUT EXACTLY ONE sentence by filling ONLY the 4 slots in the TEMPLATE below.\n"
    #     "TEMPLATE (copy every word EXACTLY, replace only the bracketed slots):\n"
    #     "At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, "
    #     "while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>.\n\n"
    #     "ALLOWED VALUES (copy verbatim, NOTHING ELSE):\n"
    #     "- <mv_*> = {going straight, left turn, right turn}\n"
    #     "- <side_*> = {from right road, from left road, from side road left, from side road right}\n"
    #     "- <who_entered> = {Dashcam, Other}\n"
    #     "- <earlier_or_later> = {earlier, later}\n\n"
    #     "HARD RULES:\n"
    #     "- Do NOT invent words like 'following vehicle', 'lanes', 'light', 'signal', 'time', etc.\n"
    #     "- Do NOT change anything outside the 4 slots. Copy the TEMPLATE text exactly.\n"
    #     "- Use singular 'Other Vehicle' exactly.\n"
    #     "- Do NOT add or remove commas/words/punctuation.\n"
    #     "- Final output must be ONE sentence ending with a period.\n\n"
    # )
    #이게 가장 최근
    caption_prompt = (
        "STRICT INSTRUCTIONS: Follow ALL rules below without exception.\n"
        "NEVER write misspelled or partial words such as: Vegetable, Vehitcle, Vehicel, Vehov, Vehicles, Vehi, Veh, Ve, etc.\n"
        "NEVER output truncated tokens (e.g., 'Vehi').\n\n"

        "This is a first-person dashcam (ego-view) video. 'Dashcam Vehicle' = the recording car.\n\n"

        "OUTPUT EXACTLY ONE sentence by filling ONLY the 4 slots in the TEMPLATE below.\n"
        "TEMPLATE (copy every word EXACTLY, replace only the bracketed slots):\n"
        "At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, "
        "while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>.\n\n"

        "ALLOWED VALUES (copy verbatim, NOTHING ELSE):\n"
        "- <mv_*> = {going straight, left turn, right turn}\n"
        "- <side_*> = {from right road, from left road, from side road left, from side road right}\n"
        "- <who_entered> = {Dashcam, Other}\n"
        "- <earlier_or_later> = {earlier, later}\n\n"

        "HARD RULES:\n"
        "- All 4 slots MUST be filled. Never leave a slot blank.\n"
        "- Do NOT invent or add any words (e.g., 'following vehicle', 'lanes', 'light', 'signal', 'time').\n"
        "- Do NOT change, drop, or paraphrase any words outside the slots. Copy the TEMPLATE text exactly.\n"
        "- Always use singular form 'Other Vehicle' (NEVER plural).\n"
        "- Do NOT add or remove commas, words, or punctuation.\n"
        "- The final output MUST be exactly ONE sentence and MUST end with a period.\n"
    )


    # 인코더 디코더 용
    # caption_prompt = (
    #         "Task: Describe an accident.\n"
    #         "Include: intersection type, both vehicles' movements.\n"
    #         "Output exactly one sentence and do not add any extra sentences or explanations."
    #     )



    # caption_prompt = (
    #     "This is a first-person dashcam (ego-view) video. 'Dashcam Vehicle' = the recording car.\n"
    #     "OUTPUT EXACTLY ONE sentence by filling ONLY the 4 slots in the TEMPLATE below.\n"
    #     "TEMPLATE (copy every word EXACTLY, replace only the bracketed slots):\n"
    #     "At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, "
    #     "while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>.\n\n"
    #     "ALLOWED VALUES (copy verbatim, NOTHING ELSE):\n"
    #     "- <mv_*> = {going straight, left turn, right turn}\n"
    #     "- <side_*> = {from right road, from left road, from main road, from side road left, from side road right}\n"
    #     "- <who_entered> = {Dashcam, Other}\n"
    #     "- <earlier_or_later> = {earlier, later}\n\n"
    #     "HARD RULES:\n"
    #     "- Do NOT invent words like 'following vehicle', 'lanes', 'light', 'signal', 'time', etc.\n"
    #     "- Do NOT change anything outside the 4 slots. Copy the TEMPLATE text exactly.\n"
    #     "- Use singular 'Other Vehicle' exactly. NEVER write Vegetable, Vehitcle, Vehicel, Vehov etc.\n"
    #     "- Do NOT add or remove commas/words/punctuation.\n"
    #     "- Final output must be ONE sentence ending with a period.\n\n"
    #     "GOOD EXAMPLE (valid):\n"
    #     "At an unsignalized intersection, the Dashcam Vehicle was going straight from main road, while the Other Vehicle was right turn from side road left, and the Other Vehicle entered earlier.\n\n"
    #     "BAD EXAMPLE (invalid):\n"
    #     "At an unsignalized intersection, the Dashcam Vehicle was following vehicle of the other vehicle.  <-- WRONG: not in allowed values\n"
    #     "At an uncontrolled intersection, the dashcam vehicle was going ahead.  <-- WRONG: template not copied, invalid words\n"
    # )
    # caption_prompt = (
    #     "This is a first-person dashcam (ego-view) video. 'Dashcam Vehicle' = the recording car.\n"
    #     "RETURN EXACTLY ONE SENTENCE, wrapped in <CAPTION> ... </CAPTION>.\n"
    #     "Fill ONLY the 4 slots in the TEMPLATE below. Copy every non-bracket character EXACTLY.\n\n"
    #     "TEMPLATE:\n"
    #     "<CAPTION>At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, "
    #     "while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>.</CAPTION>\n\n"
    #     "ALLOWED VALUES (copy verbatim; choose ONE from each line):\n"
    #     "<mv_dv> = {going straight, left turn, right turn}\n"
    #     "<side_dv> = {from right road, from left road, from main road, from side road left, from side road right}\n"
    #     "<mv_ov> = {going straight, left turn, right turn}\n"
    #     "<side_ov> = {from right road, from left road, from main road, from side road left, from side road right}\n"
    #     "<who_entered> = {Dashcam, Other}\n"
    #     "<earlier_or_later> = {earlier, later}\n\n"
    #     "HARD RULES:\n"
    #     "- Output ONLY the <CAPTION> sentence. No explanations, no extra text.\n"
    #     "- Do NOT invent words like lanes, signal, traffic light, time, speed, camera, ego.\n"
    #     "- Do NOT change any punctuation. Exactly one comma after each <side_*>, and exactly one space after each comma.\n"
    #     "- Use singular 'Other Vehicle' exactly. Do NOT use 'Vehicles'\n"
    #     "- Final output must end with a period inside </CAPTION>.\n\n"
    #     "GOOD EXAMPLE:\n"
    #     "<CAPTION>At an unsignalized intersection, the Dashcam Vehicle was going straight from main road, "
    #     "while the Other Vehicle was right turn from side road left, and the Other Vehicle entered earlier.</CAPTION>\n"
    # )
    # caption_prompt = (
    #     "This is a first-person dashcam (ego-view) video. 'Dashcam Vehicle' = the recording car.\n"
    #     "OUTPUT EXACTLY ONE sentence by filling ONLY the 4 slots in the TEMPLATE below.\n"
    #     "TEMPLATE (copy every word EXACTLY, replace only the bracketed slots):\n"
    #     "At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, "
    #     "while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>.\n\n"
    #     "ALLOWED VALUES (copy verbatim, NOTHING ELSE):\n"
    #     "- <mv_*> = {going straight, left turn, right turn}\n"
    #     "- <side_*> = {from right road, from left road, from main road, from side road left, from side road right}\n"
    #     "- <who_entered> = {Dashcam, Other}\n"
    #     "- <earlier_or_later> = {earlier, later}\n\n"
    #     "HARD RULES:\n"
    #     "- Do NOT invent words like 'following vehicle', 'lanes', 'light', 'signal', 'time', etc.\n"
    #     "- Do NOT change anything outside the 4 slots. Copy the TEMPLATE text exactly.\n"
    #     "- Use singular 'Other Vehicle' exactly. NEVER write Vegetable, Vehitcle, Vehicel, Vehov etc.\n"
    #     "- Do NOT add or remove commas/words/punctuation.\n"
    #     "- Final output must be ONE sentence ending with a period.\n\n"
    # )
    # caption_prompt = (
    #         "This is a first-person dashcam (ego-view) video. 'Dashcam Vehicle' = the recording car.\n"
    #         "OUTPUT: ONE sentence in THIS EXACT template (replace <> only):\n"
    #         "At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, "
    #         "while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>.\n"
    #         "Allowed: <mv>={going straight,left turn,right turn}; "
    #         "<side>={from the right,from left road,from main road,from side road left,from side road right}; "
    #         "<who_entered>={Dashcam,Other}; <earlier_or_later>={earlier,later}.\n"
    #         "If '<SCENE_HINTS>' appears below, copy values EXACTLY.\n"
    #         "Return ONLY the sentence. Do NOT mention camera/ego/first-person."

    #         "GOOD EXAMPLE (valid):\n"
    #         "At an unsignalized intersection, the Dashcam Vehicle was going straight from main road, "
    #         "while the Other Vehicle was right turn from side road left, and the Other Vehicle entered earlier.\n\n"
    #         "BAD EXAMPLE (invalid):\n"
    #         "At an unsignalized intersection, the Dashcam Vehicle was following vehicle of the other vehicle.  <-- WRONG: not in allowed values\n"
    #         "At an uncontrolled intersection, the dashcam vehicle was going ahead.  <-- WRONG: template not copied, invalid words\n"
    # )
    # caption_prompt = (
    #     "Return ONLY one-line JSON with EXACT keys and values:\n"
    #     '{"mv_dv":"...","side_dv":"...","mv_ov":"...","side_ov":"...","who_entered":"...","earlier_or_later":"..."}\n'
    #     "Allowed values (copy verbatim; nothing else):\n"
    #     '- "mv_dv","mv_ov" ∈ {"going straight","left turn","right turn"}\n'
    #     '- "side_dv","side_ov" ∈ {"from right road","from left road","from main road","from side road left","from side road right"}\n'
    #     '- "who_entered" ∈ {"Dashcam","Other"}\n'
    #     '- "earlier_or_later" ∈ {"earlier","later"}\n'
    #     "Hard rules: one line, valid JSON, ONLY allowed values; no other words.\n"
    # )


    preds_basis, labels_basis = [], []
    results = []
    check_list_video = []

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
        try:
            K = getattr(args, "n_captions", 5)
            mode = getattr(args, "ensemble_mode", "vote+mean")
        except NameError:
            K, mode = 5, "vote+mean"

        try:
            cands = generate_k_captions_from_video(
                vl_model, vpath, prompt=caption_prompt,
                K=K, num_frames=num_frames, size=size, max_new_tokens=max_new_tokens
            )
        except Exception as e:
            out_item = {"idx": i, "video_name": video_name, "error": f"caption_generation_failed: {e}"}
            results.append(out_item)
            if verbose:
                print(f"[{i+1}/{N}] {video_name} | CAPTION ERROR: {e}", flush=True)
            continue

        # 슬롯 다수결 문장 (선택)
        caption_voted = ""
        parsed = [parse_slots_from_caption(c) for c in cands]
        parsed = [p for p in parsed if p]
        if "vote" in mode and len(parsed) > 0:
            voted = vote_slots(parsed)
            caption_voted = render_sentence(voted)

        # 회귀 예측 앙상블
        pred_list = []
        caps_for_pred = ([caption_voted] if caption_voted else []) + cands
        for cap in caps_for_pred:
            y = predict_fault_ratio(fr_model, fr_tok, cap, device=device_fault)
            y = project_pair_to_basis(y, total=target_basis)
            pred_list.append(y)

        if len(pred_list) == 0:
            # 비상: 후보 하나라도 집어넣기
            fallback = cands[-1] if cands else ""
            y = predict_fault_ratio(fr_model, fr_tok, fallback, device=device_fault)
            pred_basis = project_pair_to_basis(y, total=target_basis)
            caption = fallback
        else:
            Y = np.vstack(pred_list)
            pred_basis = np.median(Y, axis=0) if "median" in mode else np.mean(Y, axis=0)
            caption = caption_voted if caption_voted else cands[0]

        # pred_basis = snap_pair_to_integer_basis(pred_basis, total=int(target_basis))

        preds_basis.append(pred_basis)
        if gt_basis is not None:
            labels_basis.append(gt_basis)

        # 보기 편하게 0~100도 함께 기록
        pred_100 = [float(x) * (100.0 / target_basis) for x in pred_basis]

        out_item = {
            "idx": i,
            "video_name": video_name,
            "caption_pred": caption,  # 생성된 캡션
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
            tgt_text = str(row.get('generated_caption', "No description available")).strip()
            m = re.search(r"\bthe collision\b", tgt_text, flags=re.IGNORECASE)
            if m:
                tgt_text = tgt_text[:m.start()].rstrip()
                # 끝이 .!? 로 안 끝나면 마침표 하나 붙여 깔끔하게
                if not tgt_text.endswith(('.', '!', '?')):
                    tgt_text += '.'

            if gt_basis is not None:
                ae_dc = out_item["abs_err_basis_dashcam"]
                ae_ov = out_item["abs_err_basis_other"]
                if ae_dc >= 2.0 or ae_ov >= 2.0:
                    check_list_video.append([video_name, caption, tgt_text])
                msg += f" | gt_basis=[{out_item['gt_basis_dashcam']:.2f}, {out_item['gt_basis_other']:.2f}] | abs_err=[{ae_dc:.2f}, {ae_ov:.2f}]"

            msg += "\n"
            msg += f"⭐️pred_caption: {caption}\n" + f"⭐️gt_caption: {tgt_text}"
            print(msg, flush=True)

        # 배치/스텝 로깅(선택)
        if (i + 1) % 20 == 0:
            wandb.log({"eval_progress_samples": i + 1})

    # 메트릭 & 저장
    metrics = {}
    out_prefix = os.path.splitext(out_json_path)[0]
    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)

    # 결과 중 "예측이 있는 행"만 모으기 (VIDEO NOT FOUND 같은 에러행 제외)
    pred_rows = [r for r in results if "pred_basis_dashcam" in r]
    if len(pred_rows) == 0:
        # 예측 자체가 없으면 결과만 저장
        out_csv_path = f"{out_prefix}.csv"
        df_to_csv(results, out_csv_path)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": {}, "results": results}, f, ensure_ascii=False, indent=2)
        print("=== Evaluation Summary (JSON captions NOT used) ===")
        print(json.dumps({}, indent=2, ensure_ascii=False))
        print(f"Saved JSON: {out_json_path}")
        print(f"Saved CSV : {out_csv_path}")
        print("Check list (abs err basis >=2.0):", len(check_list_video))
        print(check_list_video)
        wandb.finish()
        return

    # (1) 배열로 정리
    yhat_all = np.array([[r["pred_basis_dashcam"], r["pred_basis_other"]] for r in pred_rows], dtype=float)
    mask_gt = np.array([("gt_basis_dashcam" in r) for r in pred_rows], dtype=bool)

    # GT가 있는 행만 분리
    if mask_gt.any():
        y_gt    = np.array([[pred_rows[i]["gt_basis_dashcam"], pred_rows[i]["gt_basis_other"]] for i in range(len(pred_rows)) if mask_gt[i]], dtype=float)
        yhat_gt = yhat_all[mask_gt]

        # 보정 전
        metrics_pre = compute_metrics(y_gt, yhat_gt)
        metrics_pre["target_basis"] = target_basis

        # 비율로 등화 학습
        p_hat_gt = ratio_from_pairs(yhat_gt)
        p_true_gt = ratio_from_pairs(y_gt)

        try:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True).fit(p_hat_gt, p_true_gt)
            p_cal_all = iso.transform(ratio_from_pairs(yhat_all))
        except Exception as e:
            print("[cal] isotonic failed; fallback to bin-wise:", e)
            fbin = calibrate_ratio_binwise(p_hat_gt, p_true_gt, nbins=10)
            p_cal_all = fbin(ratio_from_pairs(yhat_all))

        # 비율→쌍으로 복원 + 합=target_basis 재보장
        yhat_cal_all = pairs_from_ratio(p_cal_all, total=target_basis)
        yhat_cal_all = np.vstack([project_pair_to_basis(v, total=target_basis) for v in yhat_cal_all])
        # yhat_cal_all = np.vstack([snap_pair_to_integer_basis(v, total=int(target_basis)) for v in yhat_cal_all])

        # 결과 dict에 보정값 반영
        for k, r in enumerate(pred_rows):
            r["pred_basis_dashcam_cal"] = float(yhat_cal_all[k, 0])
            r["pred_basis_other_cal"]   = float(yhat_cal_all[k, 1])

        # 보정 후 메트릭(GT 있는 서브셋)
        yhat_cal_gt = yhat_cal_all[mask_gt]
        metrics_cal = compute_metrics(y_gt, yhat_cal_gt)

        metrics = {**metrics_pre, **{f"cal/{k}": v for k, v in metrics_cal.items()}}

        # 플롯(전/후)
        plot_paths = save_plots(y_gt, yhat_gt, target_basis, out_prefix + "_precal")
        plot_paths_cal = save_plots(y_gt, yhat_cal_gt, target_basis, out_prefix + "_cal")
        plot_paths.update({("cal_" + k): v for k, v in plot_paths_cal.items()})

        # CSV/JSON 저장
        out_csv_path = f"{out_prefix}.csv"
        pd.DataFrame(results).to_csv(out_csv_path, index=False, encoding="utf-8")
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "results": results}, f, ensure_ascii=False, indent=2)

        # W&B 로깅
        wandb.log({
            "eval/MAE": metrics["MAE"],
            "eval/RMSE": metrics["RMSE"],
            "eval/R2": metrics["R2"],
            "eval/MAE_dashcam": metrics["MAE_dashcam"],
            "eval/MAE_other": metrics["MAE_other"],
            "eval_cal/MAE": metrics["cal/MAE"],
            "eval_cal/RMSE": metrics["cal/RMSE"],
            "eval_cal/R2": metrics["cal/R2"],
            "eval_cal/MAE_dashcam": metrics["cal/MAE_dashcam"],
            "eval_cal/MAE_other": metrics["cal/MAE_other"],
        })
        for k, pth in plot_paths.items():
            if os.path.exists(pth):
                wandb.log({f"plots/{k}": wandb.Image(pth)})

        # 아티팩트
        try:
            table = wandb.Table(dataframe=pd.DataFrame(results))
            wandb.log({"eval/table": table})
        except Exception:
            pass
        try:
            art = wandb.Artifact("eval_results", type="evaluation")
            art.add_file(out_json_path); art.add_file(out_csv_path)
            for p in plot_paths.values():
                if os.path.exists(p):
                    art.add_file(p)
            wandb.log_artifact(art)
        except Exception as e:
            print("W&B artifact upload failed:", e)

    else:
        # GT가 전혀 없으면 메트릭 없이 저장
        out_csv_path = f"{out_prefix}.csv"
        df_to_csv(results, out_csv_path)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": {}, "results": results}, f, ensure_ascii=False, indent=2)

    print("=== Evaluation Summary (JSON captions NOT used) ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved JSON: {out_json_path}")
    print(f"Saved CSV : {os.path.splitext(out_json_path)[0]}.csv")
    print("Check list (abs err basis >=2.0):", len(check_list_video))
    print(check_list_video)
    wandb.finish()


# ------------------------------
# (G) CLI
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="End-to-end evaluation from video via Video-LLaVA.")
    # p.add_argument("--eval_json", type=str,
    #                default=os.environ.get("EVAL_JSON", "/app/data/raw/json/text-evaluate/video_accident_ratio_training_results_trimmed_unsignalized_validation_0901.json"))
    p.add_argument("--eval_json", type=str,
                   default=os.environ.get("EVAL_JSON", "/app/data/raw/json/text-evaluate/video_accident_caption_results_unsignalized_validation_0901_only_equal_road_delete_main_road.json"))
    # p.add_argument("--eval_json", type=str,
    #                default=os.environ.get("EVAL_JSON", "/app/data/raw/json/text-train/video_accident_ratio_unsignalized.json"))
    p.add_argument("--fault_ckpt", type=str,
                   default=os.environ.get("FAULT_CKPT", "/app/text-train/fault_ratio_bert.pt"))
    p.add_argument("--out_json", type=str,
                   default=os.environ.get("OUT_JSON", "/app/text-train/results_0924_change_prompt_2_2_ensemble_no_yolo_no_main/eval_results_end2end.json"))
    p.add_argument("--video_root", type=str,
                   default=os.environ.get("VIDEO_ROOT", "/app/data/raw/videos/validation_reencoded"))
    # p.add_argument("--video_root", type=str,
    #                default=os.environ.get("VIDEO_ROOT", "/app/data/raw/videos/training_reencoded"))
    # p.add_argument("--vl_ckpt", type=str,
    #                default=os.environ.get("VL_CKPT", "/app/checkpoints/last_videollava_epoch_hint_drop_02_no_yolo_19_0923.pt"))
    p.add_argument("--vl_ckpt", type=str,
                   default=os.environ.get("VL_CKPT", "/app/checkpoints/last_model_epoch_29_20250923_160602.pt"))
    p.add_argument("--model_name", type=str, default=os.environ.get("MODEL_NAME", "bert-base-uncased"))
    p.add_argument("--target_basis", type=float, default=float(os.environ.get("TARGET_BASIS", 10.0)))
    p.add_argument("--max_new_tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", 512)))
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
    p.add_argument("--n_captions", type=int, default=5, help="한 비디오당 생성할 캡션 수 (K)")
    p.add_argument("--ensemble_mode", type=str, default="vote+mean",
                choices=["vote","mean","median","vote+mean","vote+median"],
                help="vote=문장 다수결만, mean/median=회귀 앙상블만, vote+*=둘 다")
    return p.parse_args()


# ------------------------------
# (H) Single-video quick test (원형 유지)
# ------------------------------
def run_fault_from_video(video_path: str, vl_ckpt_path: str, fault_model_path: str,
                         target_basis: float = 10.0,
                         device_caption: str = "cuda:0",
                         device_fault: str = "cuda:0"):
    vl = load_videollava_from_ckpt(vl_ckpt_path, dtype=torch.float16, device=device_caption)
    caption = generate_caption_from_video(
        vl, video_path,
        prompt=("Describe the accident scene in one sentence. "
                "Include intersection type, both vehicles' movements, and entry order."),
        max_new_tokens=512
    )
    fr_model, fr_tok = load_fault_model(fault_model_path, device=device_fault)
    ratios_basis = predict_fault_ratio(fr_model, fr_tok, caption, device=device_fault)
    ratios_100 = ratios_basis * (100.0 / target_basis)
    dashcam, other = ratios_100.tolist()
    print("-----")
    print("Caption:", caption)
    print(f"📊 Predicted Fault Ratio → Dashcam: {dashcam:.1f}%, Other: {other:.1f}%")
    return {"caption": caption, "dashcam_100": float(dashcam), "other_100": float(other)}


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