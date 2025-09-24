# -*- coding: utf-8 -*-
"""
Video-LLaVA 기반 사고 캡션 학습/평가 전체 스크립트
- 기존 T5 + VideoMAE 파이프라인을 Video-LLaVA(비디오 특화 LMM)로 교체
- 핵심 변경점:
  * 프레임 리스트(PIL.Image[])를 Video-LLaVA Processor에 직접 입력
  * 프롬프트(+track_hint)를 대화형 템플릿으로 구성하고, teacher forcing 시 프롬프트 토큰을 -100 마스킹
  * 소량 데이터(≈1k)에 적합한 LoRA 미세튜닝 옵션 포함

필요 패키지(예시):
  pip install "transformers>=4.41" accelerate peft einops pillow
  pip install videollava  # 가능하면 레포 전용 Processor/Model 사용
  pip install decord      # (옵션) 빠른 프레임 IO

작성일: 2025-09-04
"""

import os, sys, glob, json, time, math, random, argparse, traceback
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:native,max_split_size_mb:256,garbage_collection_threshold:0.7"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from typing import List, Dict, Tuple
from datetime import datetime


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import contextlib

# 시각화/메트릭(선택)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error

# 멀티모달/LLM
from PIL import Image
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import AdamW
import re
import nltk
# 로깅(선택)
try:
    import wandb
except Exception:
    wandb = None

# LoRA
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

# Video-LLaVA 전용(가능하면 사용)
try:
    from videollava import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
    VIDEOLLAVA_BACKEND = "repo"
except Exception:
    VideoLlavaProcessor = None
    VideoLlavaForConditionalGeneration = None
    VIDEOLLAVA_BACKEND = "hf"

# ---------------- CONFIG ----------------


# 데이터 경로
VIDEO_DIR = "data/raw/videos/training_reencoded"
VAL_VIDEO_DIR = "data/raw/videos/validation_reencoded"
TRAIN_META = "data/raw/json/video-train/video_accident_caption_results_unsignalized_0811.csv"
VAL_META = "data/raw/json/video-evaluate/video_accident_caption_results_unsignalized_validation_0901.csv"
# TRACK_JSON_DIR_TRAIN = "data/tracks/raw/train"
TRACK_JSON_DIR_TRAIN = "data/tracks/0916"
TRACK_JSON_DIR_VAL   = "data/tracks/raw/val"

# 출력 경로
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
PLOT_DIR = "plots"

# 비디오/프레임
NUM_FRAMES = 8        # (데이터셋 내부 기본 샘플 수 — PIL 리스트로도 동일 적용)
FRAME_SIZE = 224
VIS_MAX_FRAMES = 8     # Video-LLaVA Processor로 보낼 최대 프레임 수

# 학습 하이퍼
BATCH_SIZE = 1
EPOCHS = 6
LR = 2e-4
WARMUP_STEPS = 300
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 8
EVAL_EVERY_N_EPOCHS = 2
LOG_EVERY = 1

# 모델 선택: Video-LLaVA (필수)
USE_VIDEOLLAVA = True
VIDEOLLAVA_ID = "LanguageBind/Video-LLaVA-7B-hf"  # 또는 "PKU-YuanGroup/Video-LLaVA-7B-Chat"
MAX_NEW_TOKENS = 512

# LoRA 옵션(권장)
LORA_ENABLE = True
LORA_R, LORA_ALPHA = 16, 32
LORA_DROPOUT = 0.05

# 힌트 주입/드랍아웃
HINT_DROPOUT_P = 0.2

# 장치
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# W&B 설정(옵션)
WANDB_PROJECT = "accident-caption-videollava"
WANDB_ENTITY = None

# 시드 고정
SEED = 42

# ---------------- 유틸 ----------------
def is_rank0():
    return os.environ.get("LOCAL_RANK", "0") == "0"

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------- Metrics Calculator ----------------
class MetricsCalculator:
    def __init__(self):
        self.reset()
    def reset(self):
        self.predictions, self.references = [], []
        self.losses = []
    def add_batch(self, preds, refs, loss=None):
        self.predictions.extend(preds)
        self.references.extend(refs)
        if loss is not None:
            self.losses.append(loss)
    def compute_text_metrics(self):
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from rouge_score import rouge_scorer
            from sentence_transformers import SentenceTransformer
            # from nltk.translate.meteor_score import meteor_score

            smoothing = SmoothingFunction().method1
            bleu_scores = [sentence_bleu([r.split()], p.split(), smoothing_function=smoothing)
                           for p, r in zip(self.predictions, self.references)]
            scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
            rouge = {k: [] for k in ['rouge1','rouge2','rougeL']}

            for p, r in zip(self.predictions, self.references):
                s = scorer.score(r, p)
                for k in rouge: rouge[k].append(s[k].fmeasure)
            
            # METEOR ⭐️
            # try:
            #     from nltk.translate.meteor_score import meteor_score
            #     meteor_scores = [meteor_score([r], p) for p, r in zip(preds, refs)]
            #     out['meteor'] = float(np.mean(meteor_scores))
            # except Exception as e:
            #     print(f"[metrics] METEOR error: {e}")

            try:
                st = SentenceTransformer('all-MiniLM-L6-v2')
                pe = st.encode(self.predictions); re = st.encode(self.references)
                sims = [float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8)) for a,b in zip(pe,re)]
                sem = float(np.mean(sims))
            except Exception:
                sem = 0.0
            return {
                'bleu': float(np.mean(bleu_scores) if bleu_scores else 0.0),
                'rouge1': float(np.mean(rouge['rouge1']) if rouge['rouge1'] else 0.0),
                'rouge2': float(np.mean(rouge['rouge2']) if rouge['rouge2'] else 0.0),
                'rougeL': float(np.mean(rouge['rougeL']) if rouge['rougeL'] else 0.0),
                'semantic_similarity': sem,
                # 'meteor': meteor
            }
        except Exception as e:
            print(f"Text metrics unavailable: {e}")
            return {'bleu':0.0,'rouge1':0.0,'rouge2':0.0,'rougeL':0.0,'semantic_similarity':0.0}
    def compute_regression_metrics(self, pred_values, true_values):
        pred_values = np.array(pred_values); true_values = np.array(true_values)
        mae = mean_absolute_error(true_values, pred_values)
        rmse = math.sqrt(mean_squared_error(true_values, pred_values))
        def dcg(scores,k):
            s = scores[:k]; return float(np.sum([v/np.log2(i+2) for i,v in enumerate(s)]))
        def ndcg(true,pred,k):
            idx = np.argsort(pred)[::-1]; ts = true[idx]
            return (dcg(ts,k)/max(dcg(np.sort(true)[::-1],k),1e-8))
        nd = ndcg(true_values, pred_values, min(len(pred_values),10))
        return {'mae':float(mae),'rmse':float(rmse),'ndcg':float(nd)}
    def compute_classification_metrics(self, pred_labels, true_labels):
        return {
            'f1_macro': float(f1_score(true_labels, pred_labels, average='macro', zero_division=0)),
            'f1_micro': float(f1_score(true_labels, pred_labels, average='micro', zero_division=0)),
            'f1_weighted': float(f1_score(true_labels, pred_labels, average='weighted', zero_division=0)),
        }

# ---------------- Visualization ----------------
class MetricsVisualizer:
    def __init__(self, save_dir=PLOT_DIR):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
    def plot_training_curves(self, train_losses, val_losses, epoch):
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(train_losses, label='Train Loss')
        ax.plot(val_losses, label='Val Loss')
        ax.set_title(f'Training Progress (epoch={epoch})'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend()
        path = os.path.join(self.save_dir, f'loss_curves_epoch_{epoch}.png')
        plt.tight_layout(); plt.savefig(path, dpi=200); plt.close(); return path
import re
# ---------------- Dataset ----------------
class AccidentVideoDataset(Dataset):
    def __init__(self, meta_path, video_dir, tokenizer, num_frames=NUM_FRAMES, frame_size=FRAME_SIZE,
                 meta_drop_prob=0.0, track_json_dir=None, split="train"):
        import csv
        self.rows = []
        self.split = split
        self.video_dir = video_dir
        self.tok = tokenizer
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.meta_drop_prob = meta_drop_prob
        self.track_json_dir = track_json_dir

        try:
            ext = os.path.splitext(meta_path)[1].lower()
            if ext == ".csv":
                with open(meta_path, 'r', encoding='utf-8-sig') as f:
                    for r in csv.DictReader(f):
                        self.rows.append(r)
            elif ext in (".json", ".jsonl"):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    if ext == ".jsonl":
                        for line in f:
                            if line.strip(): self.rows.append(json.loads(line))
                    else:
                        data = json.load(f); self.rows = data if isinstance(data, list) else []
        except Exception as e:
            print(f"Error loading metadata from {meta_path}: {e}")
            self.rows = []

    def __len__(self):
        return len(self.rows)

    def _load_track_json(self, vid: str):
        if not self.track_json_dir: return None
        pattern = os.path.join(self.track_json_dir, vid + "*.json")
        cands = sorted(glob.glob(pattern), key=lambda x: (not x.endswith(".tracks.json"), x))
        if not cands: return None
        try:
            with open(cands[0], 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print("track json load error:", e); return None
        
    @staticmethod    
    def _std_mov(s: str | None) -> str | None:
        """자유표현 → {going straight,left turn,right turn}"""
        if not s: return None
        t = s.lower()
        if re.search(r"\b(left\s*turn|turn(ing)?\s+left)\b", t):  return "left turn"
        if re.search(r"\b(right\s*turn|turn(ing)?\s+right)\b", t): return "right turn"
        if re.search(r"\b(going|go|went|drive|driving)\s+straight\b|\bstraight\b", t):
            return "going straight"
        if "facing each other" in t and "straight" in t:        # 예외: "Facing Each Other Going Straight"
            return "going straight"
        return None
    
    @staticmethod
    def _std_side(s: str | None) -> str | None:
        """CSV 표기 → 제어어휘 {from right/left/main/side-left/side-right}"""
        if not s: return None
        t = s.lower()
        # CSV 예: "From The Right" / "From Left Road" / "From Main Road" / "From Side Road Right"
        if re.search(r"\bfrom\s+the?\s*right\b|\bright\s+road\b", t):  return "from right road"
        if re.search(r"\bfrom\s+the?\s*left\b|\bleft\s+road\b", t):    return "from left road"
        if re.search(r"\bfrom\s+main\s+road\b", t):                    return "from main road"
        if re.search(r"\bfrom\s+side\s+road\s+left\b", t):             return "from side road left"
        if re.search(r"\bfrom\s+side\s+road\s+right\b", t):            return "from side road right"
        return None
    
    @staticmethod
    def _split_mv_side_entered(info: str | None):
        """
        예) "Going Straight From The Right Entered Earlier"
        → mv="going straight", side="from right road", entered="earlier"
        예) "Right Turn" → mv="right turn", side=None, entered=None
        """
        if not info: return (None, None, None)
        t = info.strip()

        # entered tag
        entered = None
        if re.search(r"entered\s+earlier$", t, flags=re.I):
            entered = "earlier"; t = re.sub(r"\s*entered\s+earlier\s*$", "", t, flags=re.I)
        elif re.search(r"entered\s+later$", t, flags=re.I):
            entered = "later";   t = re.sub(r"\s*entered\s+later\s*$", "", t, flags=re.I)

        # movement / side
        # "Going Straight From The Right" 같은 합성
        mv  = AccidentVideoDataset._std_mov(t)
        # side는 "From ..." 구절만 보며, 없으면 None
        side = AccidentVideoDataset._std_side(t)

        return (mv, side, entered)
    
    @staticmethod
    def _build_hint_from_csv_infos(row: dict):
        """
        CSV 두 열(dashcam_vehicle_info, other_vehicle_info)만으로 힌트 구성.
        - who_entered / earlier_or_later는 'Entered Earlier/Later'가 붙은 쪽에서 바로 얻음.
        - 둘 다 'Entered' 표기가 있으면 Dashcam 기준을 우선 사용(둘 다 있으면 모순이므로 Dashcam→who로 둠).
        """
        dc_info = (row.get("dashcam_vehicle_info") or "").strip()
        ov_info = (row.get("other_vehicle_info") or "").strip()

        mv_dv, side_dv, ent_dv = AccidentVideoDataset._split_mv_side_entered(dc_info)
        mv_ov, side_ov, ent_ov = AccidentVideoDataset._split_mv_side_entered(ov_info)

        # 누가 먼저/나중?
        who_entered = None
        earlier_or_later = None
        if ent_dv in {"earlier", "later"}:
            who_entered = "Dashcam"; earlier_or_later = ent_dv
        elif ent_ov in {"earlier", "later"}:
            who_entered = "Other";   earlier_or_later = ent_ov

        # 문구 결합(없으면 생략)
        def join_mv_side(mv, side):
            if mv and side: return f"{mv} {side}"
            if mv:          return mv
            if side:        return side
            return "unknown"

        dc = join_mv_side(mv_dv, side_dv)
        ov = join_mv_side(mv_ov, side_ov)

        parts = [f"dashcam_vehicle_info={dc}", f"other_vehicle_info={ov}"]
        if who_entered and earlier_or_later:
            parts.append(f"the {who_entered} Vehicle entered {earlier_or_later}")

        return ", ".join(parts)

    def _vectorize_tracks(self, tj):
        vec = np.zeros(10, dtype=np.float32)
        slot = {"movement": -1, "side": -1, "entry": -1}
        
        if not isinstance(tj, dict) or not (tj.get("tracks") or []):
            return vec, slot, "none"

        # 1) 방어코드: tracks 없으면 바로 리턴
        tracks = tj.get("tracks") or []
        if not tj or not tracks:
            return vec, slot, "none"

        # 2) other(상대차) 선택: ego_other_id > primary_pair > 최장 트랙
        tbid = {t.get("id"): t for t in tracks}
        other = None
        other_id = tj.get("ego_other_id", None)
        if other_id in tbid:
            other = tbid[other_id]
        else:
            pp = tj.get("primary_pair") or []
            A = tbid.get(pp[0]) if len(pp) >= 1 else None
            B = tbid.get(pp[1]) if len(pp) >= 2 else None
            cands = [x for x in (A, B) if x]
            if cands:
                other = max(cands, key=lambda t: len(t.get("frames", [])))
            elif tbid:
                other = max(tbid.values(), key=lambda t: len(t.get("frames", [])))

        # other가 끝내 없으면 기본 리턴
        if not other:
            return vec, slot, "none"

        # 3) movement / side 원핫
        mv_map = {"straight": 0, "left_turn": 1, "right_turn": 2}
        sd_map = {"side_left": 0, "main": 1, "side_right": 2}

        mv_id = mv_map.get(other.get("move", "straight"), 0)
        vec[mv_id] = 1.0
        slot["movement"] = mv_id

        sd_id = sd_map.get(other.get("entry_side", "main"), 1)
        vec[3 + sd_id] = 1.0
        slot["side"] = sd_id

        # 4) entry_order 해석 (-1은 자차)
        ent_id = -1
        entry = tj.get("entry_order")
        if entry and ("earlier_id" in entry or "later_id" in entry):
            oid = other.get("id")
            if entry.get("earlier_id") == oid:
                ent_id = 0  # other가 먼저
            elif entry.get("later_id") == oid:
                ent_id = 1  # other가 나중
        if ent_id != -1:
            vec[6 + ent_id] = 1.0
            slot["entry"] = ent_id

        # 5) 충돌(t1,t2) 윈도우 추론
        def _infer_collision_window(tjson):
            col = tjson.get("collision") or {}
            t1 = col.get("t1", None)
            t2 = col.get("t2", None)
            if t1 is not None and t2 is not None and float(t2) > float(t1):
                return float(t1), float(t2), "collision"

            eco = tjson.get("ego_collision") or {}
            if eco:
                # shake가 있으면 우선 사용
                if eco.get("t_shake") is not None:
                    center = float(eco["t_shake"])
                    half = 0.15 if eco.get("shake_near_collision", False) else 0.30
                    return max(0.0, center - half), center + half, "shake"
                # shake가 없으면 t_at 기반
                if eco.get("t_at") is not None:
                    center = float(eco["t_at"])
                    half = 0.15
                    return max(0.0, center - half), center + half, "t_at"

            # 없으면 (0,0)
            return 0.0, 0.0, "none"

        t1, t2, src = _infer_collision_window(tj)

        # 6) 정규화 구간: 영상 길이 근사(모든 트랙 last_ts의 최댓값)
        lasts = [float(t.get("last_ts", 0.0) or 0.0) for t in tracks]
        dur = max(1e-3, max(lasts + [t2]))
        vec[8] = float(np.clip((t1 / dur) if dur > 0 else 0.0, 0.0, 1.0))
        vec[9] = float(np.clip((t2 / dur) if dur > 0 else 0.0, 0.0, 1.0))

        # 7) 힌트 문자열
        mv_txt = ["straight", "left_turn", "right_turn"][mv_id]
        sd_txt = ["left", "main", "right"][sd_id]
        ent_txt = ["other_earlier", "other_later"][ent_id] if ent_id != -1 else "unknown"
        hint = AccidentVideoDataset._build_hint_from_csv_infos(r)
        if self.split == "train" and hint:
            src_text = f"{src_text}\nScene hints: {hint}"

        return vec, slot, hint


    def sample_frames_efficient(self, video_path):
        import cv2
        try:
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total == 0:
                cap.release(); raise RuntimeError(f"Empty video: {video_path}")
            if total <= self.num_frames:
                idxs=list(range(total))
            else:
                step=total/(self.num_frames+1); idxs=[int(step*(i+1)) for i in range(self.num_frames)]
            frames=[]
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx,total-1))
                ret, fr = cap.read()
                if ret:
                    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                    fr = cv2.resize(fr, (FRAME_SIZE, FRAME_SIZE))
                    frames.append(fr)
                else:
                    frames.append(frames[-1] if frames else np.zeros((FRAME_SIZE,FRAME_SIZE,3),dtype=np.uint8))
            cap.release()
            while len(frames) < self.num_frames:
                frames.append(frames[-1])
            arr = np.stack(frames[:self.num_frames], axis=0)  # (T,H,W,3)
            ten = torch.from_numpy(arr).permute(0,3,1,2).float()/255.0  # (T,3,H,W)
            return ten
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return torch.zeros((self.num_frames,3,FRAME_SIZE,FRAME_SIZE))

    def __getitem__(self, idx):
        r = self.rows[idx]
        vid = r.get("video_name", f"unknown_{idx}")

        # 비디오 파일 경로 탐색
        video_path=None
        for ext in (".mp4",".mkv",".mov",".avi",".webm"):
            p=os.path.join(self.video_dir, vid+ext)
            if os.path.exists(p): video_path=p; break

        frames = self.sample_frames_efficient(video_path) if video_path else torch.zeros((self.num_frames,3,FRAME_SIZE,FRAME_SIZE))

        # 프롬프트 구성 + 힌트
        # src_text = (
        #     "Task: Describe an accident.\n"
        #     "Include: intersection type, both vehicles' movements.\n"
        #     "Output exactly one sentence and do not add any extra sentences or explanations."
        # )
        # src_text = (
        #     "Task: Describe the accident scene in one sentence.\n"
        #     "You MUST include:\n"
        #     "- intersection type\n"
        #     "- the movement and direction of both vehicles (Dashcam and Other)\n"
        #     "- entry order (who entered first)\n"
        #     "Avoid adding traffic signals or unrelated details."
        # )
        # src_text = (
        #     "Task: Describe the accident scene in one sentence.\n"
        #     "You MUST include:\n"
        #     "- intersection type\n"
        #     "- the movement and direction of both vehicles (Dashcam and Other)\n"
        #     "- entry order (who entered first)\n"
        #     "Avoid adding traffic signals or unrelated details."
        # )
        # src_text = (
        #     "Task: Describe the accident scene in EXACTLY ONE sentence using the FIXED TEMPLATE.\n"
        #     "TEMPLATE:\n"
        #     "At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, "
        #     "while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>.\n"
        #     "Controlled vocabulary ONLY (no paraphrasing):\n"
        #     "- <mv_*> ∈ {going straight, left turn, right turn}\n"
        #     "- <side_*> ∈ {from the right, from left road, from main road, from side road left, from side road right}\n"
        #     "- <who_entered> ∈ {Dashcam, Other}\n"
        #     "- <earlier_or_later> ∈ {earlier, later}\n"
        #     "STRICT RULES:\n"
        #     "- Use 'Other Vehicle' (singular). Do NOT use: light, signal, lane, facing each other, opposite, vehicles.\n"
        #     "- Do NOT add extra words beyond the template. Keep exactly ONE sentence.\n"
        # )
        # src_text = (
        #     "This is a first-person dashcam (ego-view) video. 'Dashcam Vehicle' is the recording car (ego).\n"
        #     "ONE sentence. Use this EXACT template:\n"
        #     "At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, "
        #     "while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>.\n"
        #     "Values: <mv_*>={going straight,left turn,right turn}; "
        #     "<side_*>={from the right,from left road,from main road,from side road left,from side road right}; "
        #     "<who_entered>={Dashcam,Other}; <earlier_or_later>={earlier,later}.\n"
        #     "No extra words. Use singular 'Other Vehicle'. Do NOT mention camera/ego/first-person in the sentence."
        # )
        src_text = (
            "This is a first-person dashcam (ego-view) video. 'Dashcam Vehicle' = the recording car.\n"
            "OUTPUT EXACTLY ONE sentence by filling ONLY the 4 slots in the TEMPLATE below.\n"
            "TEMPLATE (copy every word EXACTLY, replace only the bracketed slots):\n"
            "At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, "
            "while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>.\n\n"
            "ALLOWED VALUES (copy verbatim, NOTHING ELSE):\n"
            "- <mv_*> = {going straight, left turn, right turn}\n"
            "- <side_*> = {from right road, from left road, from main road, from side road left, from side road right}\n"
            "- <who_entered> = {Dashcam, Other}\n"
            "- <earlier_or_later> = {earlier, later}\n\n"
            "HARD RULES:\n"
            "- Do NOT invent words like 'following vehicle', 'lanes', 'light', 'signal', 'time', etc.\n"
            "- Do NOT change anything outside the 4 slots. Copy the TEMPLATE text exactly.\n"
            "- Use singular 'Other Vehicle' exactly. NEVER write Vegetable, Vehitcle, Vehicel, Vehov etc.\n"
            "- Do NOT add or remove commas/words/punctuation.\n"
            "- Final output must be ONE sentence ending with a period.\n\n"
        )
        
        # track_json = self._load_track_json(vid)
        # track_vec_np, slot_labels, track_hint = self._vectorize_tracks(track_json)
        # if self.split == "train" and track_hint and track_hint != "none":
        #     src_text = f"{src_text}\nScene hints: {track_hint}"
        hint = AccidentVideoDataset._build_hint_from_csv_infos(r)
        if self.split == "train" and hint:
            src_text = f"{src_text}\nScene hints: {hint}"
        # 힌트 드랍아웃
        if self.split == "train" and random.random() < HINT_DROPOUT_P:
            src_text = src_text.split("\nScene hints:")[0].strip()

        # 타깃 캡션
        tgt_text = str(r.get("generated_caption") or "No description available").strip()
        # cutoff = "seconds."
        # if cutoff in tgt_text: tgt_text = tgt_text.split(cutoff)[0] + cutoff
        m = re.search(r"\bthe collision\b", tgt_text, flags=re.IGNORECASE)
        if m:
            tgt_text = tgt_text[:m.start()].rstrip()
            # 끝이 .!? 로 안 끝나면 마침표 하나 붙여 깔끔하게
            if not tgt_text.endswith(('.', '!', '?')):
                tgt_text += '.'

        # PIL 프레임 리스트 (Video-LLaVA 입력)
        frames_np = (frames.clamp(0,1).permute(0,2,3,1).cpu().numpy()*255).astype("uint8")
        frames_pil = [Image.fromarray(frames_np[t]) for t in range(min(frames_np.shape[0], VIS_MAX_FRAMES))]

        # 텍스트 토큰(기존 파이프라인 호환용; collate_fn에서 pad id 필요 시 사용)
        try:
            enc = self.tok(src_text, truncation=True, max_length=512, padding=False, return_tensors="pt")
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            tgt = self.tok(tgt_text, truncation=True, max_length=512, padding=False, return_tensors="pt")
            labels = tgt["input_ids"].squeeze(0)
        except Exception:
            pad = getattr(self.tok, 'pad_token_id', 0)
            input_ids = torch.tensor([pad]); attention_mask = torch.tensor([1]); labels = torch.tensor([-100])

        return {
            "frames": frames,                       # (T,3,H,W) — (기존 호환)
            "frames_pil": frames_pil,               # ✅ Video-LLaVA 입력
            "src_text": src_text,                   # ✅ 프롬프트(+힌트)
            "input_ids": input_ids,                 # (기존 collate_fn 호환)
            "attention_mask": attention_mask,
            "labels": labels,
            "video_name": vid,
            "caption": tgt_text,
            "caption_length": len(tgt_text.split()),
            "_tok": self.tok,
        }

# collate_fn (기존 호환 — 일부 로더에서 사용될 수 있음)
def collate_fn(batch):
    tok = batch[0]["_tok"]
    PAD_ID = tok.pad_token_id if getattr(tok, 'pad_token_id', None) is not None else 0
    frames = torch.stack([item["frames"] for item in batch])
    video_names = [item["video_name"] for item in batch]
    captions = [item["caption"] for item in batch]
    caption_lengths = [item["caption_length"] for item in batch]
    frames_pil = [item["frames_pil"] for item in batch]
    src_texts = [item["src_text"] for item in batch]

    max_in = max(item["input_ids"].size(0) for item in batch)
    max_lb = max(item["labels"].size(0) for item in batch)
    input_ids, attention, labels = [], [], []
    for it in batch:
        pi = F.pad(it["input_ids"], (0, max_in - it["input_ids"].size(0)), value=PAD_ID)
        am = F.pad(it["attention_mask"], (0, max_in - it["attention_mask"].size(0)), value=0)
        lb = F.pad(it["labels"], (0, max_lb - it["labels"].size(0)), value=PAD_ID); lb[lb==PAD_ID] = -100
        input_ids.append(pi); attention.append(am); labels.append(lb)

    return {
        "frames": frames,
        "frames_pil": frames_pil,
        "src_text": src_texts,
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention),
        "labels": torch.stack(labels),
        "video_names": video_names,
        "captions": captions,
        "caption_lengths": caption_lengths,
    }

# ---------------- Video-LLaVA Wrapper ----------------
# class VideoLLaVACaptioner(nn.Module):
#     """Video-LLaVA 학습/추론 래퍼 (레포 백엔드 우선, HF 폴백)."""
#     def __init__(self, model_id=VIDEOLLAVA_ID, lora=LORA_ENABLE, lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT):
#         super().__init__()
#         self.model_id = model_id
#         self.backend = VIDEOLLAVA_BACKEND
#         if self.backend == "repo" and VideoLlavaProcessor is not None:
#             self.processor = VideoLlavaProcessor.from_pretrained(model_id)
#             self.processor.tokenizer.padding_side = "left"
#             self.model = VideoLlavaForConditionalGeneration.from_pretrained(
#                 "LanguageBind/Video-LLaVA-7B-hf",
#                 dtype=torch.float16,                 # torch_dtype 대신 dtype 사용 (경고 해결)
#                 device_map="auto",                   # 멀티 GPU면 자동 배치
#                 # attn_implementation="flash_attention_2",  # (옵션) flash-attn 설치 시 속도 ↑
#             )
#             self.tokenizer = self.processor.tokenizer
#         else:
#             self.processor = AutoProcessor.from_pretrained(model_id)
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 model_id,
#                 torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32)
#             )
#             self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
#             if self.tokenizer.pad_token is None:
#                 self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.model.config.use_cache = False

#         # LoRA
#         if lora and get_peft_model is not None and LoraConfig is not None:
#             try:
#                 if prepare_model_for_kbit_training is not None:
#                     self.model = prepare_model_for_kbit_training(self.model)
#             except Exception:
#                 pass
#             peft_cfg = LoraConfig(
#                 r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
#                 target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
#                 bias="none", task_type="CAUSAL_LM"
#             )
#             self.model = get_peft_model(self.model, peft_cfg)
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
import torch
import torch.nn as nn
# (선택) LoRA 쓰는 경우:
# from peft import get_peft_model, LoraConfig, TaskType

class VideoLLaVACaptioner(nn.Module):
    """Video-LLaVA 학습/추론 래퍼 (HF 전용)."""
    def __init__(
        self,
        model_id=VIDEOLLAVA_ID,
        lora=LORA_ENABLE,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        force_torch_dtype=None,
    ):
        super().__init__()
        self.model_id = model_id

        # ✅ 항상 전용 Processor/Model 사용 (AutoModel/AutoProcessor 금지)
        self.processor = VideoLlavaProcessor.from_pretrained(self.model_id)
        self.processor.tokenizer.padding_side = "left"  # 권장

        if force_torch_dtype is None:
            amp = os.environ.get("ACCELERATE_MIXED_PRECISION", "no").lower()
            if amp == "bf16":
                force_torch_dtype = torch.bfloat16
            elif amp == "fp16":
                force_torch_dtype = torch.float16
            else:
                force_torch_dtype = torch.float32

        self.model = VideoLlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=force_torch_dtype,  # ✅ torch_dtype 사용
            device_map=None,
            low_cpu_mem_usage=True,
        )

        # pad 토큰 안전장치
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id

        self.tokenizer = self.processor.tokenizer
        self.model.config.use_cache = False

        # (선택) LoRA 적용
        if lora:
            from peft import get_peft_model, LoraConfig, TaskType
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            )
            self.model = get_peft_model(self.model, lora_cfg)


    # def _build_chat_prompt(self, user_text: str, add_generation_prompt: bool):
    #     # tokenizer에 존재하는 플레이스홀더 우선순위: <video> → <image>
    #     placeholder = None
    #     if self.tokenizer is not None:
    #         for tok in ("<video>", "<image>"):
    #             tid = self.tokenizer.convert_tokens_to_ids(tok)
    #             if tid is not None and tid != self.tokenizer.unk_token_id:
    #                 placeholder = tok
    #                 break
    #     if placeholder is None:
    #         placeholder = "<image>"  # 최후 보루

    #     # 간단 프롬프트(모델이 placeholder 토큰을 input_ids에서 찾을 수 있게)
    #     return f"{placeholder}\n{user_text}"
    def _build_chat_prompt(self, user_text: str, add_generation_prompt: bool):
        # placeholder 우선순위는 유지
        placeholder = None
        if self.tokenizer is not None:
            for tok in ("<video>", "<image>"):
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                if tid is not None and tid != self.tokenizer.unk_token_id:
                    placeholder = tok
                    break
        if placeholder is None:
            placeholder = "<image>"

        # ✅ 대화형 템플릿로 변경 (생성 구간을 명확히 여는 게 핵심)
        prompt = f"{placeholder}\nUSER: {user_text}\nASSISTANT:"

        # ✅ 안전장치: placeholder 토큰이 실제로 인코딩됐는지 1회 확인(디버그용)
        if not hasattr(self, "_checked_video_token"):
            ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
            vid_id = self.tokenizer.convert_tokens_to_ids("<video>")
            img_id = self.tokenizer.convert_tokens_to_ids("<image>")
            if not ((ids == vid_id).any() or (ids == img_id).any()):
                print("[warn] video/image placeholder token not found in prompt ids. Vision path may be skipped.")
            self._checked_video_token = True

        return prompt
    
    def freeze_modules(self, *, train_vision: bool = False, train_projector: bool = True, freeze_lora: bool = True):
        """
        LLM은 전부 동결하고, (옵션) 비전 타워/프로젝터만 학습.
        - train_vision=False  : 비전 타워도 동결(추천, VRAM 절약)
          train_vision=True   : 비전 타워 미세튜닝 허용(무겁습니다)
        - train_projector=True: projector 계열만 학습(권장)
        - freeze_lora=True    : lora_* 파라미터도 동결
        """
        # 0) 전부 동결
        for n, p in self.model.named_parameters():
            p.requires_grad_(False)

        # 1) LLM(언어 모델)은 계속 동결 (아무 것도 해제하지 않음)

        # 2) 학습 허용 모듈들만 선택 해제
        UNFREEZE_KEYS = []
        if train_projector:
            UNFREEZE_KEYS += [
                "mm_projector",
                "multi_modal_projector",
                "visual_projector",
                "video_projector",
                "projector",
            ]
        if train_vision:
            UNFREEZE_KEYS += [
                "vision_tower",   # HF Video-LLaVA
                "video_tower",    # 일부 구현체
                # 하위 실제 백본까지 열고 싶으면 아래 주석 해제
                # "vision_model", "visual_backbone", "clip", "siglip"
            ]

        for n, p in self.model.named_parameters():
            if any(k in n for k in UNFREEZE_KEYS):
                p.requires_grad_(True)

        # 3) LoRA 파라미터는 요청대로 동결(혹시 로드되어 있더라도)
        if freeze_lora:
            for n, p in self.model.named_parameters():
                if "lora_" in n.lower():
                    p.requires_grad_(False)

        # 4) 디버그 프린트
        trainables = [n for n, p in self.model.named_parameters() if p.requires_grad]
        print(f"[Trainable tensors] {len(trainables)}")
        for n in trainables[:60]:
            print("  •", n)
        if len(trainables) > 60:
            print("  ...")
    
    def _make_prompt(self, src_text: str) -> str:
        # 간단 대화 템플릿
        return f"USER: {src_text}\nASSISTANT:"

    def _build_labels_for_concat(self, input_ids_prompt, input_ids_answer):
        input_ids = torch.cat([input_ids_prompt, input_ids_answer], dim=1)
        attn_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[:, :input_ids_prompt.size(1)] = -100
        return input_ids, attn_mask, labels

    def forward(self, frames_pil: List[Image.Image], src_text: str, answer_text: str = None,
            generate: bool = False, max_new_tokens: int = 512, **generate_kwargs):
        device = next(self.model.parameters()).device
        vision_dtype = next(self.model.parameters()).dtype 

        # 1) 프롬프트 만들기(placeholder 포함)
        prompt = self._build_chat_prompt(src_text, add_generation_prompt=not generate)

        # 2) processor가 비디오 + 텍스트를 함께 처리하도록 (여기서 토크나이징까지 진행)
        # proc = self.processor(
        #     videos=[frames_pil],
        #     text=[prompt],
        #     padding=True,
        #     return_tensors="pt",
        # )
        proc = self.processor(
            videos=[frames_pil],
            text=[prompt],
            padding="longest",
            truncation=True,          # ✅ 중요: MM 토큰 불일치 방지
            return_tensors="pt",
        )

        # 비전 텐서 꺼내기
        if "pixel_values_videos" in proc:
            vision = {"pixel_values_videos": proc["pixel_values_videos"].to(device=device, dtype=vision_dtype, non_blocking=True)}
        elif "pixel_values" in proc:
            vision = {"pixel_values_videos": proc["pixel_values"].to(device=device, dtype=vision_dtype, non_blocking=True)}
        else:
            raise KeyError(f"processor output has no pixel values. keys={list(proc.keys())}")

        # 텍스트 텐서(프롬프트) 꺼내기
        prompt_ids = proc["input_ids"].to(device)            # [1, Lp]
        prompt_attn = proc["attention_mask"].to(device)      # [1, Lp]

        # (안전장치) placeholder 토큰이 정말 들어갔는지 점검, 없으면 <image>/<video> 교체 시도 권장
        # vid_id = self.tokenizer.convert_tokens_to_ids("<video>")
        # img_id = self.tokenizer.convert_tokens_to_ids("<image>")
        # assert ((prompt_ids == vid_id).any() or (prompt_ids == img_id).any()), "No video/image placeholder in input_ids"

        if not generate:
            # 정답 토큰 준비
            # ans_ids = self.tokenizer(
            #     answer_text or "",
            #     add_special_tokens=False,
            #     truncation=True,     # ✅ 추가
            #     max_length=64,       # ✅ 추가
            #     return_tensors="pt"
            # ).input_ids.to(device)

            # # 프롬프트 + 정답 이어붙이고, 프롬프트 구간은 -100 마스킹
            # pad_id = self.tokenizer.pad_token_id or 0
            # inp = torch.cat([prompt_ids, ans_ids], dim=1)      # [1, Lp+La]
            # attn = torch.cat(
            #     [prompt_attn, torch.ones_like(ans_ids)], dim=1
            # )
            # labels = inp.clone()
            # labels[:, :prompt_ids.size(1)] = -100
            # labels[labels == pad_id] = -100

            ans_ids = self.tokenizer(
                answer_text or "",
                add_special_tokens=True,
                truncation=True, max_length=256,
                return_tensors="pt"
            ).input_ids.to(device)

            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None and (ans_ids[0, -1].item() != eos_id):
                ans_ids = torch.cat([ans_ids, torch.tensor([[eos_id]], device=device)], dim=1)

            # if ans_ids.numel() == 0:
            #     eos = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id or 0
            #     ans_ids = torch.tensor([[eos]], device=device)

            inp   = torch.cat([prompt_ids, ans_ids], dim=1)
            attn  = torch.cat([prompt_attn, torch.ones_like(ans_ids)], dim=1)
            labels = inp.clone()
            labels[:, :prompt_ids.size(1)] = -100
            pad_id = self.tokenizer.pad_token_id or 0
            labels[labels == pad_id] = -100

            out = self.model(
                **vision,
                input_ids=inp,
                attention_mask=attn,
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
            return {"loss": out.loss, "logits": out.logits}

        else:
            force_prefix = "At an unsignalized intersection, the Dashcam Vehicle was "
            prefix_ids = self.tokenizer(force_prefix, add_special_tokens=False, return_tensors="pt").input_ids.to(prompt_ids.device)

            # 금지어 → bad_words_ids (토큰 ID 리스트의 리스트)
            bad_words = [
                "light","lights","signal","signals","signalized","lane","lanes",
                "facing","opposite","collision","collided","collide","crash",
                "January","February","March","April","May","June","July","August","September","October","November","December",
                'room',"At an un signalized","At an un signals","unsignaled"
                "AM","PM",":","/","-",  # 시간/날짜 기호
            ] + [str(d) for d in range(10)]  # 숫자 전체 금지

            bad_words_ids = []
            for w in bad_words:
                ids = self.tokenizer(w, add_special_tokens=False).input_ids
                if ids:  # 빈 토큰열은 제외
                    bad_words_ids.append(ids)

            # 생성
            # gen_ids = self.model.generate(
            #     **vision,
            #     input_ids=prompt_ids,
            #     attention_mask=prompt_attn,
            #     max_new_tokens=max_new_tokens,
            #     do_sample=False, num_beams=2,
            #     no_repeat_ngram_size=3,
            #     repetition_penalty=1.2,
            #     length_penalty=1.1,
            #     early_stopping=True,
            # )
            # 자주 새는 금지어(신호/차선/충돌/메타문장 등)

            # banned_phrases = [
            #     "traffic light", "traffic lights", "traffic signal", "traffic signals",
            #     "signalized",                 # OK (unsignalized는 제외)
            #     "lane", "lanes",
            #     "collision", "collided", "detected",
            #     "facing each other", "opposite",
            #     "ATTENTION", "ATTORNEY", "USERS", "STATEMENT"
            # ]

            # 토크나이저로 서브워드 시퀀스로 변환(빈 시퀀스는 제외)
            # tok = self.tokenizer(banned_phrases, add_special_tokens=False)
            # bad_ids = [ids for ids in tok.input_ids if len(ids) > 0]

            default_kwargs = dict(
                min_new_tokens=32,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=3,
                repetition_penalty=1.01,
                bad_words_ids=bad_words_ids,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            default_kwargs.update(generate_kwargs)
            
            gen_ids = self.model.generate(
                **vision,
                input_ids=torch.cat([prompt_ids, prefix_ids], dim=1), 
                attention_mask=torch.cat([prompt_attn, torch.ones_like(prefix_ids)], dim=1),
                **default_kwargs,
                # max_new_tokens=128,

                # ✅ 빈 문자열 방지 및 탐색 강화
                # min_new_tokens=32,
                # do_sample=False,
                # temperature=0.7,
                # top_p=1.0,
                # num_beams=1,
                # no_repeat_ngram_size=3,    # ✅ 반복 줄이기
                # repetition_penalty=1.01,
                # bad_words_ids=bad_words_ids,
                # # 초기엔 제약 끄기

                # eos_token_id=self.tokenizer.eos_token_id,
                # pad_token_id=self.tokenizer.pad_token_id,
            )
            # 프롬프트 길이 이후만 디코딩
            gen_only = gen_ids[:, prompt_ids.size(1):]
            text = self.tokenizer.decode(gen_only[0], skip_special_tokens=True)
            return {"text": text}

def prepare_train_llm_only(wrapper_model,
                           *,
                           freeze_vision=True,
                           train_projector=True,
                           freeze_lora=True,
                           merge_lora=True):
    """
    wrapper_model: VideoLLaVACaptioner 인스턴스 (nn.Module)
    - LoRA를 학습에서 제외(동결). merge_lora=True면 base에 흡수해 경로 자체 제거(권장).
    - language_model(LLM)만 requires_grad=True.
    - projector(mm_projector / video_projector)는 train_projector=True일 때만 학습.
    - vision_tower(및 vision 관련)는 freeze_vision=True면 동결.
    """
    import re
    base = wrapper_model  # nn.Module (VideoLLaVACaptioner)
    # 내부 HF 모델
    hf = getattr(base, "model", None)
    if hf is None:
        raise RuntimeError("VideoLLaVACaptioner.model 을 찾지 못했습니다.")

    # 0) LoRA 제거/비활성
    try:
        # peft가 적용돼 있으면 merge_and_unload 또는 disable_adapter 지원
        if freeze_lora:
            if hasattr(hf, "merge_and_unload") and merge_lora:
                wrapper_model.model = hf.merge_and_unload()
                hf = wrapper_model.model
                print("[LoRA] merged & unloaded")
            elif hasattr(hf, "disable_adapter"):
                hf.disable_adapter()
                print("[LoRA] adapter disabled")
    except Exception as e:
        print("[LoRA] freeze/merge 실패:", e)

    # 1) 전부 동결 후 필요한 모듈만 풀기
    for _, p in hf.named_parameters():
        p.requires_grad_(False)

    # 2) language_model만 활성화(+ 옵션: projector)
    # Video-LLaVA의 일반적인 파라미터 경로: "model.language_model."
    for n, p in hf.named_parameters():
        if n.startswith("model.language_model."):
            p.requires_grad_(True)
        elif train_projector and re.search(r"(mm_projector|video_projector|multimodal_projector)", n):
            p.requires_grad_(True)
        elif freeze_vision and ("vision_tower" in n or "vision_model" in n or "vision_proj" in n):
            p.requires_grad_(False)  # 명시 동결

    # 3) LLM 쪽 gradient checkpointing / use_cache 끄기
    try:
        lm = hf.model.language_model
        if hasattr(lm, "gradient_checkpointing_enable"):
            lm.gradient_checkpointing_enable()
        if hasattr(lm, "config"):
            lm.config.use_cache = False
    except Exception:
        pass

    # 4) 확인용 로그
    trainable = [n for n, p in hf.named_parameters() if p.requires_grad]
    print(f"[Trainable tensors] {len(trainable)}")
    for n in trainable[:30]:
        print("  •", n)
    if len(trainable) > 30:
        print("  ...")

#train_vision랑 also_unfreeze_vision_ln 학습 옵션에 추가함
# def freeze_llm_train_vision_only(
#     model,
#     train_vision: bool = True,                 # 기본: 비전 타워 동결
#     train_projector: bool = True,               # 기본: 프로젝터만 학습
#     also_unfreeze_vision_ln: bool = True,      # True면 비전 타워의 LN/Norm만 풀기
#     force_dtype=torch.float16,                           # fp16/bf16 등 강제 dtype
# ):
#     """
#     LLM 전체 동결 + (옵션) 비전 타워/프로젝터만 학습.
#     """
#     import re
#     # 1) 전부 동결
#     for p in model.parameters():
#         p.requires_grad = False

#     # 2) 학습시킬 모듈 선택
#     #   - Video-LLaVA에서 프로젝터는 보통 mm_projector/visual_projector/connector 등 이름을 가짐
#     projector_keys = ("mm_projector", "multi_modal_projector", "visual_projector", "projector", "connector")
#     vision_keys    = ("video_tower", "vision_tower", "image_tower")

#     def is_vision_norm(name: str):
#         # 비전 타워 안의 LayerNorm만 풀고 싶을 때 사용
#         # common: "layer_norm", "ln", "norm", "layernorm"
#         return any(tok in name.lower() for tok in ("layer_norm", "layernorm", ".ln", "norm"))

#     for n, p in model.named_parameters():
#         # 프로젝터
#         if train_projector and any(k in n for k in projector_keys):
#             p.requires_grad = True
#             continue
#         # 비전 타워 (전체/또는 LN만)
#         if train_vision and any(k in n for k in vision_keys):
#             if also_unfreeze_vision_ln:
#                 if is_vision_norm(n):
#                     p.requires_grad = True
#             else:
#                 p.requires_grad = True

#     # 3) fp16/bf16 강제 캐스팅 (학습 파라미터만)
#     if force_dtype is not None:
#         with torch.no_grad():
#             for n, p in model.named_parameters():
#                 if p.requires_grad and p.dtype != force_dtype:
#                     p.data = p.data.to(force_dtype)

#     # 4) 디버그 출력
#     trainables = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
#     total_params = sum(p.numel() for _, p in trainables)
#     print(f"[Freeze] trainable tensors: {len(trainables)}, params: {total_params/1e6:.2f}M")
#     for n, _ in trainables[:50]:
#         print("  •", n)

#     return model
def freeze_llm_train_vision_only(
    model,
    train_vision: bool = False,
    train_projector: bool = True,
    also_unfreeze_vision_ln: bool = False,
    vision_last_k_blocks: int = 0,
    force_dtype=torch.float16,
):
    """
    LLM 동결 + projector + (선택) video_tower 일부만 학습.
    ※ image_tower/vision_tower 등 '비사용' 타워는 절대 안 풉니다.
    """
    import re
    # 1) 전부 동결
    for p in model.parameters():
        p.requires_grad = False

    projector_keys = ("mm_projector", "multi_modal_projector", "visual_projector",
                      "projector", "connector")

    # --- ✅ 실제 사용 타워 자동 감지: video > vision > image 순 ---
    all_names = [n for n, _ in model.named_parameters()]
    if any(".video_tower." in n for n in all_names):
        active_vision_prefixes = ("video_tower",)           # ★ 비디오만
    elif any(".vision_tower." in n for n in all_names):
        active_vision_prefixes = ("vision_tower",)
    else:
        active_vision_prefixes = tuple()  # 못찾으면 아무 것도 안 풉니다.

    def is_vision_norm(name: str):
        n = name.lower()
        return any(k in n for k in (
            "layer_norm", "layernorm", "rms_norm", "rmsnorm",
            ".ln", "ln_", "_ln", "lnpre", "ln_post", "ln_pre", "ln_f",
            "norm", "norm1", "norm2"
        ))

    # 2-a) projector 학습 허용
    for n, p in model.named_parameters():
        if train_projector and any(k in n for k in projector_keys):
            p.requires_grad = True

    # 2-b) vision 타워의 레이어 인덱스 최대값 추정 (layers.N)
    layer_pattern = re.compile(r"\.layers\.(\d+)\.")
    max_layer_idx = -1
    for n, _ in model.named_parameters():
        if any(f".{pref}." in n for pref in active_vision_prefixes):
            m = layer_pattern.search(n)
            if m:
                max_layer_idx = max(max_layer_idx, int(m.group(1)))

    # 2-c) vision 일부 해제
    if train_vision and active_vision_prefixes:
        for n, p in model.named_parameters():
            if not any(f".{pref}." in n for pref in active_vision_prefixes):
                continue  # 🔒 비활성 타워는 건드리지 않음

            # 마지막 K블록만
            if vision_last_k_blocks > 0 and max_layer_idx >= 0:
                m = layer_pattern.search(n)
                if not m:
                    continue
                idx = int(m.group(1))
                if idx < max_layer_idx - (vision_last_k_blocks - 1):
                    continue

            # LN만 풀지/전체 풀지
            if also_unfreeze_vision_ln:
                if is_vision_norm(n):
                    p.requires_grad = True
            else:
                p.requires_grad = True

    # 3) dtype: Norm은 fp32, 나머지는 force_dtype
    with torch.no_grad():
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if is_vision_norm(n):
                if p.dtype != torch.float32:
                    p.data = p.data.to(torch.float32)
            else:
                if force_dtype is not None and p.dtype != force_dtype:
                    p.data = p.data.to(force_dtype)

    # 4) 디버그 출력
    trainables = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    total_params = sum(p.numel() for _, p in trainables)
    print(f"[Freeze] trainable tensors: {len(trainables)}, params: {total_params/1e6:.2f}M")
    for n, _ in trainables[:50]:
        print("  •", n)
    if vision_last_k_blocks > 0 and max_layer_idx >= 0:
        print(f"[Freeze] vision_last_k_blocks={vision_last_k_blocks} (max_layer_idx={max_layer_idx})")

    return model
# ---------------- Optim/Sched & Logger ----------------
def create_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

def create_scheduler(optimizer, num_training_steps, warmup_steps=WARMUP_STEPS):
    from transformers import get_linear_schedule_with_warmup
    return get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)

class WandBLogger:
    def __init__(self, project_name, entity=None, config=None):
        self.run=None
        if wandb is None: return
        try:
            self.run = wandb.init(project=project_name, entity=entity, config=config or {}, tags=['Video-LLaVA','LoRA'])
        except Exception:
            self.run=None
    def log_metrics(self, d, step=None, commit=True):
        if self.run: 
            try: self.run.log(d, step=step, commit=commit)
            except Exception: pass
    def log_images(self, images_dict, step=None):
        if self.run:
            try:
                wi = {k: wandb.Image(v) for k,v in images_dict.items() if os.path.exists(v)}
                if wi: self.run.log(wi, step=step)
            except Exception: pass
    def finish(self):
        if self.run:
            try: self.run.finish()
            except Exception: pass

# ---------------- Train / Validate ----------------
# def train_epoch(model: VideoLLaVACaptioner, train_loader, optimizer, scheduler, accelerator, wandb_logger, epoch):
#     model.train()
#     total_loss = 0.0

#     pbar = train_loader
#     if accelerator.is_local_main_process:
#         from tqdm import tqdm
#         pbar = tqdm(train_loader, desc=f"Train {epoch}")

#     for step, batch in enumerate(pbar):
#         frames_pil_batch: List[List[Image.Image]] = batch["frames_pil"]
#         prompts: List[str] = batch["src_text"]
#         answers: List[str] = batch["captions"]

#         # 배치 안에서 여러 샘플을 순회하는 현재 구조 유지
#         micro_losses = []
#         for i in range(len(prompts)):
#             with accelerator.accumulate(model):
#                 out = model(frames_pil=frames_pil_batch[i], src_text=prompts[i],
#                             answer_text=answers[i], generate=False)
#                 loss = out["loss"]
#                 accelerator.backward(loss)

#                 if accelerator.sync_gradients:
#                     torch.nn.utils.clip_grad_norm_(
#                         [p for p in model.parameters() if p.requires_grad], 1.0
#                     )

#                 optimizer.step()
#                 scheduler.step()
#                 optimizer.zero_grad()

#                 micro_losses.append(float(loss.item()))

#         # 배치의 평균 loss
#         step_loss = float(np.mean(micro_losses)) if micro_losses else 0.0
#         total_loss += step_loss

#         # ✅ tqdm 진행바에 현재 배치 loss 표시
#         if accelerator.is_local_main_process:
#             try:
#                 pbar.set_postfix({"step_loss": f"{step_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
#             except Exception:
#                 pass

#         # ✅ W&B: 매 배치(혹은 LOG_EVERY 간격)마다 기록
#         if wandb_logger and accelerator.is_main_process:
#             if (step % LOG_EVERY == 0) or (step == len(train_loader) - 1):
#                 global_step = epoch * len(train_loader) + step
#                 wandb_logger.log_metrics(
#                     {"train/step_loss": step_loss, "lr": scheduler.get_last_lr()[0]},
#                     step=global_step
#                 )

#         # (선택) 콘솔에도 한 줄씩 찍고 싶으면:
#         if accelerator.is_main_process and (step % LOG_EVERY == 0):
#             accelerator.print(
#                 f"[epoch {epoch+1}/{EPOCHS}] step {step+1}/{len(train_loader)} "
#                 f"loss={step_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
#             )

#     return total_loss / max(1, len(train_loader))

# @torch.no_grad()
# def validate_epoch(model: VideoLLaVACaptioner, val_loader, accelerator, wandb_logger, epoch):
#     model.eval(); total_loss=0.0
#     metrics = MetricsCalculator()
#     pbar = val_loader
#     if accelerator.is_local_main_process:
#         from tqdm import tqdm; pbar = tqdm(val_loader, desc=f"Val {epoch}")

#     for batch in pbar:
#         frames_pil_batch: List[List[Image.Image]] = batch["frames_pil"]
#         prompts: List[str] = batch["src_text"]
#         answers: List[str] = batch["captions"]

#         preds=[]; refs=[]; losses=[]
#         for i in range(len(prompts)):
#             out = model(frames_pil=frames_pil_batch[i], src_text=prompts[i], answer_text=answers[i], generate=False)
#             losses.append(float(out["loss"].item()))
#             gen = model(frames_pil=frames_pil_batch[i], src_text=prompts[i], generate=True, max_new_tokens=MAX_NEW_TOKENS)
#             preds.append(gen["text"]); refs.append(answers[i])

#         if losses:
#             mean_l = float(np.mean(losses))
#             total_loss += mean_l
#             metrics.add_batch(preds, refs, mean_l)

#     avg_loss = total_loss / max(1, len(val_loader))
#     text = metrics.compute_text_metrics()

#     # 보조 메트릭들 추가
#     pred_lengths = [len(p.split()) for p in metrics.predictions]
#     true_lengths = [len(r.split()) for r in metrics.references]
#     if pred_lengths and true_lengths:
#         text.update(metrics.compute_regression_metrics(pred_lengths, true_lengths))
#         bins=[0,5,10,20,float('inf')]
#         pc=np.digitize(pred_lengths,bins)-1; tc=np.digitize(true_lengths,bins)-1
#         text.update(metrics.compute_classification_metrics(pc, tc))
#     text.update({"loss": avg_loss})

#     # ✅ 검증도 W&B에 즉시 반영 (epoch 단위 step로)
#     if wandb_logger and accelerator.is_main_process:
#         wandb_logger.log_metrics({f"val/{k}": v for k, v in text.items()}, step=(epoch+1)*len(val_loader))

#     return text

# 전역 상수(파일 상단 아무 곳)
LOG_EVERY = 1  # 매 스텝마다 로그. 간격을 늘리고 싶으면 10, 20 등으로 변경


# def train_epoch(model: VideoLLaVACaptioner, train_loader, optimizer, scheduler, accelerator, wandb_logger, epoch, start_step=0):
#     model.train()
#     total_loss = 0.0
#     global_step = start_step

#     from tqdm import tqdm
#     pbar = tqdm(
#         train_loader,
#         desc=f"Train {epoch}",
#         disable=not accelerator.is_local_main_process,
#         dynamic_ncols=True,
#         leave=False
#     )

#     ema = None
#     for _, batch in enumerate(pbar):
#         frames_pil_batch: List[List[Image.Image]] = batch["frames_pil"]
#         prompts: List[str] = batch["src_text"]
#         answers: List[str] = batch["captions"]

#         micro_losses = []

#         for i in range(len(prompts)):
#             with accelerator.accumulate(model):
#                 prompts[i] = prompts[i][:256]
#                 answers[i] = answers[i][:64]

#                 # with accelerator.autocast():  # ✅ 반드시 추가
#                 out = model(frames_pil=frames_pil_batch[i], src_text=prompts[i], answer_text=answers[i], generate=False)
#                 loss = out["loss"]

#                 # NaN 가드: 이 샘플 스킵
#                 if torch.isnan(loss) or torch.isinf(loss):
#                     if accelerator.is_local_main_process and (global_step % LOG_EVERY == 0):
#                         # tqdm.write는 진행바를 깨지 않음
#                         from tqdm import tqdm as _tqdm
#                         _tqdm.write(f"[ep {epoch} | step {global_step}] ⚠️ NaN/Inf loss, skip sample {i}")
#                     optimizer.zero_grad(set_to_none=True)
#                     continue

#                 accelerator.backward(loss)

#                 if accelerator.sync_gradients:
#                     accelerator.clip_grad_norm_(
#                         [p for p in model.parameters() if p.requires_grad], 1.0
#                     )
#                 optimizer.step()
#                 scheduler.step()
#                 optimizer.zero_grad(set_to_none=True)

#                 micro_losses.append(float(loss.item()))

#         torch.cuda.empty_cache()
#             # torch.cuda.ipc_collect()

#         # 배치 평균 손실(유효 샘플만)
#         step_loss = float(np.mean(micro_losses)) if micro_losses else float("nan")
#         if np.isfinite(step_loss):
#             total_loss += step_loss
#             ema = step_loss if ema is None else (0.9 * ema + 0.1 * step_loss)

#         # ── 진행바: 한 줄만 갱신
#         lr_val = scheduler.get_last_lr()[0]
#         if ema is not None:
#             pbar.set_postfix_str(f"loss={ema:.4f} lr={lr_val:.2e}")
#         else:
#             pbar.set_postfix_str(f"loss=nan lr={lr_val:.2e}")

#         # ── 터미널 한 줄 로그(원하면 남기기)
#         if accelerator.is_local_main_process and (global_step % LOG_EVERY == 0):
#             print(f"[ep {epoch} | step {global_step}] loss={(step_loss if np.isfinite(step_loss) else float('nan')):.4f} lr={lr_val:.6f}", flush=True)

#         # ── W&B: 매 스텝 기록
#         if wandb_logger:
#             wandb_logger.log_metrics(
#                 {"train/step_loss": (step_loss if np.isfinite(step_loss) else None),
#                  "train/ema_loss": (ema if ema is not None else None),
#                  "lr": lr_val,
#                  "epoch": epoch},
#                 step=global_step,
#                 commit=True
#             )

#         global_step += 1

#     avg = total_loss / max(1, len(train_loader))
#     return avg, global_step

def train_epoch(model: VideoLLaVACaptioner, train_loader, optimizer, scheduler, accelerator, wandb_logger, epoch, start_step=0):
    model.train()
    total_loss = 0.0
    global_step = start_step

    from tqdm import tqdm
    pbar = tqdm(
        train_loader,
        desc=f"Train {epoch}",
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True,
        leave=False
    )

    ema = None
    for _, batch in enumerate(pbar):
        frames_pil_batch: List[List[Image.Image]] = batch["frames_pil"]
        prompts: List[str] = batch["src_text"]
        answers: List[str] = batch["captions"]
        
        with accelerator.accumulate(model):
            micro_losses = []
            B = len(prompts)
            if B == 0:
                # 빈 배치 방어
                continue
            for i in range(B):
                try:
                    with accelerator.autocast():
                        out = model(
                            frames_pil=frames_pil_batch[i],
                            src_text=prompts[i],
                            answer_text=answers[i],
                            generate=False
                        )
                        raw_loss = out["loss"]  # ← 로깅/평균용 (나누기 전)
                except RuntimeError as e:
                    # OOM 등은 캐시 비우고 샘플만 스킵
                    if "out of memory" in str(e).lower():
                        if accelerator.is_local_main_process:
                            print(f"[OOM] skip sample {i}: {e}")
                        torch.cuda.empty_cache()
                        continue
                    raise  # 다른 예외는 그대로 올려보내기

                # NaN/Inf 방어: 역전파/로그 둘 다 스킵
                if not torch.isfinite(raw_loss):
                    if accelerator.is_local_main_process:
                        print(f"[NaN] skip sample {i}")
                    continue

                # 역전파는 배치 평균이 되도록 1/B로 스케일
                scaled_loss = raw_loss / B
                accelerator.backward(scaled_loss)

                # 로그엔 '나누기 전' 손실을 기록
                micro_losses.append(float(raw_loss.detach().item()))

                # 메모리 청소 (raw_loss/Scaled는 자동 해제되지만 out만 명시 삭제)
                del out

            if accelerator.sync_gradients:
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                optimizer.step(); scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        # ── 로그/프리뷰 (accumulate 블록 밖)
        step_loss = float(np.mean(micro_losses)) if micro_losses else float("nan")
        if np.isfinite(step_loss):
            total_loss += step_loss
            ema = step_loss if ema is None else (0.9 * ema + 0.1 * step_loss)

        lr_val = optimizer.param_groups[0]["lr"]
        if ema is not None:
            pbar.set_postfix_str(f"loss={ema:.4f} lr={lr_val:.2e}")
        else:
            pbar.set_postfix_str(f"loss=nan lr={lr_val:.2e}")

        hint_flags = [("Scene hints:" in s) for s in prompts]
        hint_rate  = float(sum(hint_flags)) / max(1, len(hint_flags))
        if accelerator.is_local_main_process and (global_step % LOG_EVERY == 0):
            print(f"[ep {epoch} | step {global_step}] loss={(step_loss if np.isfinite(step_loss) else float('nan')):.4f} lr={lr_val:.6f}", flush=True)
            print(f"[hint] rate={hint_rate:.2f} (dropout≈{HINT_DROPOUT_P:.2f})")

        if wandb_logger:
            wandb_logger.log_metrics(
                {"train/step_loss": (step_loss if np.isfinite(step_loss) else None),
                 "train/ema_loss": (ema if ema is not None else None),
                 "lr": lr_val,
                 "epoch": epoch,
                 "train/hint_rate": hint_rate},
                step=global_step,
                commit=True
            )

        PREVIEW_EVERY_STEPS = 10
        # PREVIEW_EVERY_STEPS = 0 if accelerator.num_processes > 1 else 10
        if (global_step % PREVIEW_EVERY_STEPS == 0) and accelerator.is_local_main_process:
            try:
                model.eval()
                i0 = 0
                p = prompts[i0]
                a = answers[i0]
                with torch.inference_mode():
                    gen = model(
                        frames_pil=frames_pil_batch[i0],
                        src_text=p,
                        generate=True
                    )
                pred = gen.get("text", "")
                print(f"\n[Preview @ step {global_step}]")
                print("PROMPT    :", p)
                print("REFERENCE :", a)
                print("PREDICTION:", pred if pred.strip() != "" else "<EMPTY>")
                print(f"step_loss={step_loss if np.isfinite(step_loss) else float('nan'):.4f}")
            except Exception as e:
                print(f"[Preview error] {e}")
            finally:
                model.train()

        accelerator.wait_for_everyone()  # ✅ 프리뷰 이후 간단 동기화
        global_step += 1

    avg = total_loss / max(1, len(train_loader))
    return avg, global_step


@torch.no_grad()
def validate_epoch(model: VideoLLaVACaptioner, val_loader, accelerator, wandb_logger, epoch, start_step=0, show_val_examples: int = 3):
    model.eval()
    total_loss = 0.0
    metrics = MetricsCalculator()
    global_step = start_step
    shown = 0 

    pbar = val_loader
    if accelerator.is_local_main_process:
        from tqdm import tqdm
        pbar = tqdm(val_loader, desc=f"Val {epoch}")

    # for step, batch in enumerate(pbar):

    #     frames_pil_batch: List[List[Image.Image]] = batch["frames_pil"]
    #     prompts: List[str] = batch["src_text"]
    #     answers: List[str] = batch["captions"]
        

    #     preds, refs, losses = [], [], []
    #     for i in range(len(prompts)):
    #         prompts[i] = prompts[i][:256]
    #         answers[i] = answers[i][:64]
    for step, batch in enumerate(pbar):
        # 배치에서 먼저 꺼낸 다음에만 접근!
        frames_pil_batch = batch.get("frames_pil", [])
        prompts = batch.get("src_text", [])
        answers = batch.get("captions", [])
        if not (frames_pil_batch and prompts and answers):
            if accelerator.is_local_main_process:
                from tqdm import tqdm as _tqdm
                _tqdm.write(f"[VAL] empty/invalid batch at step {step}, skip")
            global_step += 1
            continue

        preds, refs, losses = [], [], []
        for i in range(len(prompts)):
            p = prompts[i]
            a = answers[i]

            # ⬇️ 여기서 예시 출력 (rank0 & 처음 N개만)
            if accelerator.is_local_main_process and shown < show_val_examples:
                from tqdm import tqdm as _tqdm
                _tqdm.write("[VAL EXAMPLE]")
                _tqdm.write(f"PROMPT: {p}")
                _tqdm.write(f"GT    : {a}")

            try:
                out = model(frames_pil=frames_pil_batch[i], src_text=p, answer_text=a, generate=False)
                losses.append(float(out["loss"].item()))
                gen = model(frames_pil=frames_pil_batch[i], src_text=p, generate=True, max_new_tokens=MAX_NEW_TOKENS)
                pred = gen["text"]
                preds.append(pred); refs.append(a)

                if accelerator.is_local_main_process and shown < show_val_examples:
                    from tqdm import tqdm as _tqdm
                    _tqdm.write(f"PRED  : {pred}\n")
                    shown += 1
            except Exception as e:
                if accelerator.is_local_main_process:
                    from tqdm import tqdm as _tqdm
                    _tqdm.write(f"[VAL] sample {i} error: {e}; skip")
                continue

            # out = model(frames_pil=frames_pil_batch[i], src_text=prompts[i], answer_text=answers[i], generate=False)
            # losses.append(float(out["loss"].item()))
            # gen = model(frames_pil=frames_pil_batch[i], src_text=prompts[i], generate=True, max_new_tokens=MAX_NEW_TOKENS)
            # preds.append(gen["text"])
            # refs.append(answers[i])

        if losses:
            batch_val_loss = float(np.mean(losses))
            total_loss += batch_val_loss
            metrics.add_batch(preds, refs, batch_val_loss)

            if accelerator.is_local_main_process:
                pbar.set_postfix({"val_loss": f"{batch_val_loss:.4f}"})

            if wandb_logger:
                # wandb_logger.log_metrics({"val/step_loss": batch_val_loss, "epoch": epoch}, step=global_step, commit=True)
                wandb_logger.log_metrics({"val/step_loss": batch_val_loss, "epoch": epoch}, step=None, commit=True)

        global_step += 1

    avg_loss = total_loss / max(1, len(val_loader))
    text = metrics.compute_text_metrics()

    # 길이/분류 보조 메트릭
    pred_lengths = [len(p.split()) for p in metrics.predictions]
    true_lengths = [len(r.split()) for r in metrics.references]
    if pred_lengths and true_lengths:
        text.update(metrics.compute_regression_metrics(pred_lengths, true_lengths))
        bins=[0,5,10,20,float('inf')]
        pc=np.digitize(pred_lengths,bins)-1; tc=np.digitize(true_lengths,bins)-1
        text.update(metrics.compute_classification_metrics(pc, tc))
    text.update({"loss": avg_loss})

    return text

# ---------------- Main Train Function ----------------
def train_model():
    torch.cuda.empty_cache()
    start_time = time.time()
    
    global_step = 0
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,
        broadcast_buffers=False,         # ✅
        gradient_as_bucket_view=True     # (가능하면) 성능·안정성 개선
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        kwargs_handlers=[ddp_kwargs],
    )

    print("🚨 AMP 확인:", accelerator.mixed_precision)  # ← fp16 출력되어야 정상
    print("🚨 autocast context:", accelerator.autocast())  # ← context info
    # 1) 모델/프로세서/토크나이저 (한 번만 생성)
    # load_dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32

    load_dtype = (
        torch.bfloat16 if accelerator.mixed_precision == "bf16"
        else torch.float16 if accelerator.mixed_precision == "fp16"
        else torch.float32
    )

    captioner = VideoLLaVACaptioner(
        model_id=VIDEOLLAVA_ID,
        lora=False,
        force_torch_dtype=load_dtype,   # ✅ 여기!
    )

    processor = captioner.processor
    tokenizer = processor.tokenizer
    model = captioner

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # 데이터셋/로더
    train_ds = AccidentVideoDataset(TRAIN_META, VIDEO_DIR, tokenizer,
                                    num_frames=NUM_FRAMES, frame_size=FRAME_SIZE,
                                    meta_drop_prob=0.0, track_json_dir=TRACK_JSON_DIR_TRAIN, split="train")
    val_ds = AccidentVideoDataset(VAL_META, VAL_VIDEO_DIR, tokenizer,
                                  num_frames=NUM_FRAMES, frame_size=FRAME_SIZE,
                                  meta_drop_prob=0.0, track_json_dir=TRACK_JSON_DIR_VAL, split="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=collate_fn, num_workers=0)

    # 3) 🔒 동결 + (필요 시) dtype 맞춤
    if accelerator.mixed_precision in ("fp16", "bf16"):
        freeze_llm_train_vision_only(
            model, train_vision=True, train_projector=True,
            also_unfreeze_vision_ln=True, force_dtype=torch.float32
        )
    else:
        freeze_llm_train_vision_only(
            model, train_vision=False, train_projector=True,
            also_unfreeze_vision_ln=False, force_dtype=torch.float32
        )
        model = model.float()


    optimizer = create_optimizer(model, lr=LR)

    # ✅ accumulation을 반영한 실제 optimizer step 수로 계산
    updates_per_epoch = math.ceil(len(train_loader) / accelerator.gradient_accumulation_steps)
    num_training_steps = updates_per_epoch * EPOCHS

    scheduler = create_scheduler(optimizer, num_training_steps, WARMUP_STEPS)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    unwrapped = accelerator.unwrap_model(model)
    try:
        core = getattr(unwrapped, "model", None)
        base = getattr(core, "model", None) if core is not None else None
        vt = None
        if base is not None:
            vt = getattr(base, "video_tower", None) or getattr(base, "vision_tower", None)
        if vt is not None:
            # FP16/BF16로 로드된 dtype을 유지해야 메모리 절약
            vt.to(device=accelerator.device)    # dtype 캐스팅 금지
            print("[cast] vision tower -> keep dtype")
    except Exception as e:
        print("vision tower cast failed:", e)

    # 로거/비주얼라이저
    wb = WandBLogger(WANDB_PROJECT, WANDB_ENTITY, config={
        'model': VIDEOLLAVA_ID, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
        'lr': LR, 'num_frames': NUM_FRAMES, 'vis_max_frames': VIS_MAX_FRAMES,
        'lora': LORA_ENABLE, 'hint_dropout': HINT_DROPOUT_P,
    }) if accelerator.is_main_process else None
    viz = MetricsVisualizer(PLOT_DIR) if accelerator.is_main_process else None

    best_bleu = -1.0; best_ckpt=None
    train_losses=[]; val_losses=[]

    accelerator.print(f"Starting training on {accelerator.device}")
    accelerator.print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    for epoch in range(EPOCHS):
        accelerator.print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
        tr_loss, global_step = train_epoch(model, train_loader, optimizer, scheduler, accelerator, wb, epoch, start_step=global_step)
        train_losses.append(tr_loss)

        if (epoch+1) % EVAL_EVERY_N_EPOCHS == 0 or (epoch+1)==EPOCHS:
            val_metrics = validate_epoch(model, val_loader, accelerator, wb, epoch, start_step=global_step)
            val_losses.append(val_metrics.get('loss', float('nan')))
            accelerator.print(f"Train Loss: {tr_loss:.4f}")
            accelerator.print(f"Val   Loss: {val_metrics.get('loss',0):.4f}")
            accelerator.print(f"Val   BLEU: {val_metrics.get('bleu',0):.4f}")
            accelerator.print(f"Val   ROUGE-L: {val_metrics.get('rougeL',0):.4f}")
            # accelerator.print(f"Val   METEOR: {val_metrics.get('meteor',0):.4f}")  # ✅ 추가
            if wb:
                # wb.log_metrics({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step, commit=True)
                wb.log_metrics({f"val/{k}": v for k, v in val_metrics.items()}, step=None, commit=True)
                wb.log_metrics({"train/epoch_loss": tr_loss, "epoch": epoch}, step=global_step, commit=True)
            if viz:
                try:
                    figp = viz.plot_training_curves(train_losses, val_losses, epoch)
                    if wb: wb.log_images({"loss_curves": figp}, step=epoch)
                except Exception as e:
                    print(f"Viz error: {e}")

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        ckpt = {
                        'epoch': epoch,
                        'model_state_dict': unwrapped.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_metrics': val_metrics,
                        'train_loss': tr_loss,
                        'config': {
                            'VIDEOLLAVA_ID': VIDEOLLAVA_ID,
                            'EPOCHS': EPOCHS,
                            'BATCH_SIZE': BATCH_SIZE,
                            'LR': LR,
                            'NUM_FRAMES': NUM_FRAMES,
                            'VIS_MAX_FRAMES': VIS_MAX_FRAMES,
                            'LORA': LORA_ENABLE,
                        }
            }
        path = os.path.join(CHECKPOINT_DIR, f"last_videollava_epoch_hint_drop_02_no_yolo_{epoch}_0924.pt")
        torch.save(ckpt, path); best_ckpt=path
        accelerator.print(f"[Saved last checkpoint] {path} (epoch={epoch+1})")
    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"✅ Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    if accelerator.is_main_process and wb: wb.finish()
    accelerator.print("Training finished.")
    return {"best_bleu": best_bleu, "best_ckpt": best_ckpt}

# ---------------- Inference ----------------
@torch.no_grad()
def inference_single_video(model_ckpt: str, video_path: str, context_text: str = ""):
    """단일 비디오 추론 (학습된 Video-LLaVA 모델 체크포인트 사용)"""
    # 모델 로드
    device = DEVICE
    if os.path.isfile(model_ckpt):
        state = torch.load(model_ckpt, map_location=device)
    else:
        raise FileNotFoundError(model_ckpt)

    model = VideoLLaVACaptioner(model_id=VIDEOLLAVA_ID, lora=LORA_ENABLE)
    model.load_state_dict(state['model_state_dict'])
    model.to(device); model.eval()

    # 프레임 추출 (간단 균등 샘플)
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0: return "Error: empty video"
    step = max(1, total // VIS_MAX_FRAMES)
    frames=[]
    for i in range(0, min(total, VIS_MAX_FRAMES*step), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, fr = cap.read()
        if not ret: break
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        fr = cv2.resize(fr, (FRAME_SIZE, FRAME_SIZE))
        frames.append(Image.fromarray(fr))
    cap.release()
    if not frames:
        return "Error: could not read frames"

    # 프롬프트(힌트 없이도 동작)
    # src_text = (
    #         "Task: Describe an accident.\n"
    #         "Include: intersection type, both vehicles' movements.\n"
    #         "Output exactly one sentence and do not add any extra sentences or explanations."
    #     )
    # src_text = (
    #     "Task: Describe the accident scene in one sentence.\n"
    #     "You MUST include:\n"
    #     "- intersection type\n"
    #     "- the movement and direction of both vehicles (Dashcam and Other)\n"
    #     "- entry order (who entered first)\n"
    #     "Avoid adding traffic signals or unrelated details."
    # )
    # src_text = (
    #     "Task: Describe the accident scene in EXACTLY ONE sentence using the FIXED TEMPLATE.\n"
    #     "TEMPLATE:\n"
    #     "At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, "
    #     "while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>.\n"
    #     "Controlled vocabulary ONLY (no paraphrasing):\n"
    #     "- <mv_*> ∈ {going straight, left turn, right turn}\n"
    #     "- <side_*> ∈ {from the right, from left road, from main road, from side road left, from side road right}\n"
    #     "- <who_entered> ∈ {Dashcam, Other}\n"
    #     "- <earlier_or_later> ∈ {earlier, later}\n"
    #     "STRICT RULES:\n"
    #     "- Use 'Other Vehicle' (singular). Do NOT use: light, signal, lane, facing each other, opposite, vehicles.\n"
    #     "- Do NOT add extra words beyond the template. Keep exactly ONE sentence.\n"
    # )
    src_text = (
        "This is a first-person dashcam (ego-view) video. 'Dashcam Vehicle' = the recording car.\n"
        "OUTPUT EXACTLY ONE sentence by filling ONLY the 4 slots in the TEMPLATE below.\n"
        "TEMPLATE (copy every word EXACTLY, replace only the bracketed slots):\n"
        "At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, "
        "while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>.\n\n"
        "ALLOWED VALUES (copy verbatim, NOTHING ELSE):\n"
        "- <mv_*> = {going straight, left turn, right turn}\n"
        "- <side_*> = {from right road, from left road, from main road, from side road left, from side road right}\n"
        "- <who_entered> = {Dashcam, Other}\n"
        "- <earlier_or_later> = {earlier, later}\n\n"
        "HARD RULES:\n"
        "- Do NOT invent words like 'following vehicle', 'lanes', 'light', 'signal', 'time', etc.\n"
        "- Do NOT change anything outside the 4 slots. Copy the TEMPLATE text exactly.\n"
        "- Use singular 'Other Vehicle' exactly. NEVER write Vegetable, Vehitcle, Vehicel, Vehov etc.\n"
        "- Do NOT add or remove commas/words/punctuation.\n"
        "- Final output must be ONE sentence ending with a period.\n\n"
        "REGEX the final answer must match (copy verbatim):\n"
        "^At an unsignalized intersection, the Dashcam Vehicle was (going straight|left turn|right turn) "
        "(from right road|from left road|from main road|from side road left|from side road right), "
        "while the Other Vehicle was (going straight|left turn|right turn) "
        "(from right road|from left road|from main road|from side road left|from side road right), "
        "and the (Dashcam|Other) Vehicle entered (earlier|later)\\.$\n\n"
        "ONE correct example:\n"
        "At an unsignalized intersection, the Dashcam Vehicle was right turn from main road, "
        "while the Other Vehicle was going straight from left road, and the Dashcam Vehicle entered later.\n"
    )
    # src_text = (
    #     "Task: Output EXACTLY ONE sentence in THIS pattern (no extra words):\n"
    #     "At an unsignalized intersection, the Dashcam Vehicle was <mv_dv> <side_dv>, while the Other Vehicle was <mv_ov> <side_ov>, and the <who_entered> Vehicle entered <earlier_or_later>."
    #     "Allowed values:\n"
    #     "<mv_*> = {going straight, left turn, right turn}"
    #     "<side_*> = {from the right, from left road, from main road, from side road left, from side road right}"
    #    " <who_entered> = {Dashcam, Other}"
    #    " <earlier_or_later> = {earlier, later}"
    # )

    if context_text:
        src_text += f"\nContext: {context_text}"

    out = model(frames_pil=frames, src_text=src_text, generate=True, max_new_tokens=MAX_NEW_TOKENS)
    return out["text"]

# ---------------- CLI ----------------
def main():
    global BATCH_SIZE, EPOCHS, LR, NUM_FRAMES, FRAME_SIZE
    parser = argparse.ArgumentParser(description='Video-LLaVA Accident Caption Training')
    parser.add_argument('--mode', choices=['train','infer'], default='train')
    parser.add_argument('--model_path', type=str, help='Path to trained checkpoint (.pt) for inference')
    parser.add_argument('--video_path', type=str, help='Path to video for inference')
    parser.add_argument('--context', type=str, default='', help='Extra context string for inference')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--num_frames', type=int, default=NUM_FRAMES)
    parser.add_argument('--frame_size', type=int, default=FRAME_SIZE)
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size; EPOCHS=args.epochs; LR=args.lr
    NUM_FRAMES=args.num_frames; FRAME_SIZE=args.frame_size

    if args.mode == 'train':
        if is_rank0(): print("Starting training (Video-LLaVA)...")
        res = train_model()
        if is_rank0(): print(f"Done. Best BLEU: {res['best_bleu']:.4f}, ckpt: {res['best_ckpt']}")
    elif args.mode == 'infer':
        if not args.model_path or not args.video_path:
            print("Error: --model_path and --video_path required for inference"); return
        print("Running inference...")
        text = inference_single_video(args.model_path, args.video_path, args.context)
        print("Generated caption:\n", text)

        import torch
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # 멀티프로세싱 spawn 강제 (fork로 인한 cuDNN 초기화 레이스 회피)
        import torch.multiprocessing as mp
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    if is_rank0():
        print("Video-LLaVA Accident Captioner")
        print("1) Train with LoRA + Hints")
        print("2) Inference on a single video")
    for d in [CHECKPOINT_DIR, LOG_DIR, PLOT_DIR]: os.makedirs(d, exist_ok=True)
    set_seed(SEED)
    try:
        main()
    except KeyboardInterrupt:
        if is_rank0(): print("\nInterrupted by user")
    except Exception as e:
        if is_rank0():
            print(f"Error: {e}")
            traceback.print_exc()
        raise
