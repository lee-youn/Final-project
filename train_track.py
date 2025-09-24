import os, random, math, argparse
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, VideoMAEModel, VideoMAEImageProcessor
import torchvision
from torchvision import transforms
from accelerate import Accelerator
from ultralytics import YOLO
import wandb
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
import pandas as pd
from datetime import datetime
import json
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from accelerate.utils import DistributedDataParallelKwargs
import os
import os, sys, traceback, random, numpy as np, torch
import time
import os, json, glob
import re
# ---------------- CONFIG ----------------
VIDEO_DIR = "data/raw/videos/training_reencoded"
VAL_VIDEO_DIR = "data/raw/videos/validation_reencoded"
# TRAIN_META = "data/Preprocessing/json/video-train/video_accident_validation_caption_results_top6.csv"
# VAL_META = "data/Preprocessing/json/video-evaluate/video_accident_caption_results_validation_top6.csv"
TRAIN_META = "data/raw/json/video-train/video_accident_caption_results_unsignalized_0811.csv"
VAL_META = "data/raw/json/video-evaluate/video_accident_caption_results_unsignalized_validation_0901.csv"

TRACK_JSON_DIR_TRAIN = "data/tracks/raw/train"
TRACK_JSON_DIR_VAL   = "data/tracks/raw/val"

CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
PLOT_DIR = "plots"

NUM_FRAMES = 16
FRAME_SIZE = 224
VIDEO_PREFIX_TOKENS = 8
MAX_OBJECTS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

T5_NAME = "t5-base"
VIDEO_MODEL = "MCG-NJU/videomae-base"
YOLO_MODEL = "yolov8s.pt"

BATCH_SIZE = 8
EPOCHS = 30
LR = 2e-4
META_DROP_PROB = 0.2
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 4
EVAL_EVERY_N_EPOCHS = 5

# W&B 설정
WANDB_PROJECT = "accident-caption-multimodal-tracking"
WANDB_ENTITY = None  # 팀 이름이 있다면 설정



# ---------------- Vehicle Classes ----------------
VEHICLE_YOLO_CLASSES = {
    1: 'bicycle',
    2: 'car', 
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}
VEHICLE_CLASS_IDS = set(VEHICLE_YOLO_CLASSES.keys())

def is_rank0():
    # accelerate가 넣어주는 LOCAL_RANK 사용 (없으면 0으로 취급)
    return os.environ.get("LOCAL_RANK", "0") == "0"

def truncate_caption(text: str) -> str:
    cutoff = "seconds."
    if cutoff in text:
        return text.split(cutoff)[0] + cutoff
    return text

# ---------------- Metrics Calculator ----------------
class MetricsCalculator:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.references = []
        self.losses = []
        self.detection_scores = []
        self.semantic_similarities = []
    
    def add_batch(self, preds, refs, loss=None, det_scores=None):
        self.predictions.extend(preds)
        self.references.extend(refs)
        if loss is not None:
            self.losses.append(loss)
        if det_scores is not None:
            self.detection_scores.extend(det_scores)
    
    def compute_text_metrics(self):
        """텍스트 생성 메트릭 계산"""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from rouge_score import rouge_scorer
            from sentence_transformers import SentenceTransformer
            
            # BLEU 점수
            smoothing = SmoothingFunction().method1
            bleu_scores = []
            for pred, ref in zip(self.predictions, self.references):
                bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothing)
                bleu_scores.append(bleu)
            
            # ROUGE 점수
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            
            for pred, ref in zip(self.predictions, self.references):
                scores = scorer.score(ref, pred)
                for key in rouge_scores:
                    rouge_scores[key].append(scores[key].fmeasure)
            
            # 의미적 유사도 (Sentence-BERT)
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2')
                pred_embeddings = model.encode(self.predictions)
                ref_embeddings = model.encode(self.references)
                
                cosine_similarities = []
                for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
                    cos_sim = np.dot(pred_emb, ref_emb) / (np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb))
                    cosine_similarities.append(cos_sim)
                
                semantic_similarity = np.mean(cosine_similarities)
            except:
                semantic_similarity = 0.0
            
            return {
                'bleu': np.mean(bleu_scores),
                'rouge1': np.mean(rouge_scores['rouge1']),
                'rouge2': np.mean(rouge_scores['rouge2']),
                'rougeL': np.mean(rouge_scores['rougeL']),
                'semantic_similarity': semantic_similarity,
                'bleu_std': np.std(bleu_scores),
                'rouge1_std': np.std(rouge_scores['rouge1'])
            }
        except ImportError as e:
            print(f"Some text metrics libraries not available: {e}")
            return {'bleu': 0.0, 'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def compute_regression_metrics(self, pred_values, true_values):
        """회귀 메트릭 계산 (길이, 단어 수 등)"""
        pred_values = np.array(pred_values)
        true_values = np.array(true_values)
        
        mae = mean_absolute_error(true_values, pred_values)
        rmse = np.sqrt(mean_squared_error(true_values, pred_values))
        
        # nDCG 계산 (길이 기준 랭킹)
        def dcg_at_k(scores, k):
            scores = scores[:k]
            return np.sum([score / np.log2(i + 2) for i, score in enumerate(scores)])
        
        def ndcg_at_k(true_scores, pred_scores, k):
            # 예측값 기준으로 정렬
            sorted_indices = np.argsort(pred_scores)[::-1]
            true_sorted = true_scores[sorted_indices]
            
            dcg = dcg_at_k(true_sorted, k)
            ideal_dcg = dcg_at_k(np.sort(true_scores)[::-1], k)
            
            return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        
        ndcg = ndcg_at_k(true_values, pred_values, min(len(pred_values), 10))
        
        return {'mae': mae, 'rmse': rmse, 'ndcg': ndcg}
    
    def compute_classification_metrics(self, pred_labels, true_labels):
        """분류 메트릭 계산"""
        f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
        f1_micro = f1_score(true_labels, pred_labels, average='micro', zero_division=0)
        f1_weighted = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
        
        return {
            'f1_macro': f1_macro,
            'f1_micro': f1_micro, 
            'f1_weighted': f1_weighted
        }

# ---------------- Visualization ----------------
class MetricsVisualizer:
    def __init__(self, save_dir=PLOT_DIR):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
    
    def plot_training_curves(self, train_losses, val_losses, train_metrics, val_metrics, epoch):
        """훈련 곡선 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Training Progress - Epoch {epoch}', fontsize=16, fontweight='bold')
        
        # Loss 곡선
        axes[0, 0].plot(train_losses, label='Train Loss', color='blue', alpha=0.7)
        axes[0, 0].plot(val_losses, label='Val Loss', color='red', alpha=0.7)
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # BLEU 점수
        if 'bleu' in train_metrics and 'bleu' in val_metrics:
            axes[0, 1].plot([m['bleu'] for m in train_metrics], label='Train BLEU', color='green')
            axes[0, 1].plot([m['bleu'] for m in val_metrics], label='Val BLEU', color='orange')
            axes[0, 1].set_title('BLEU Scores')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('BLEU')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # ROUGE 점수들
        if 'rouge1' in train_metrics and 'rouge1' in val_metrics:
            for i, rouge_type in enumerate(['rouge1', 'rouge2', 'rougeL']):
                row, col = (0, 2) if i == 0 else (1, i-1)
                axes[row, col].plot([m.get(rouge_type, 0) for m in train_metrics], 
                                  label=f'Train {rouge_type.upper()}', alpha=0.8)
                axes[row, col].plot([m.get(rouge_type, 0) for m in val_metrics], 
                                  label=f'Val {rouge_type.upper()}', alpha=0.8)
                axes[row, col].set_title(f'{rouge_type.upper()} Scores')
                axes[row, col].set_xlabel('Epoch')
                axes[row, col].set_ylabel(f'{rouge_type.upper()}')
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'training_curves_epoch_3e3_250{epoch}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_metrics_comparison(self, all_metrics, epoch):
        """다양한 메트릭 비교 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Comprehensive Metrics - Epoch {epoch}', fontsize=16, fontweight='bold')
        
        # 메트릭 데이터 준비
        metrics_names = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'semantic_similarity']
        train_values = [all_metrics['train'].get(m, 0) for m in metrics_names]
        val_values = [all_metrics['val'].get(m, 0) for m in metrics_names]
        
        # 바 차트
        x = np.arange(len(metrics_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, train_values, width, label='Train', alpha=0.8, color='skyblue')
        axes[0, 0].bar(x + width/2, val_values, width, label='Validation', alpha=0.8, color='lightcoral')
        axes[0, 0].set_title('Text Generation Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 회귀 메트릭 (있는 경우)
        if 'mae' in all_metrics['val']:
            reg_metrics = ['mae', 'rmse', 'ndcg']
            reg_values = [all_metrics['val'].get(m, 0) for m in reg_metrics]
            
            axes[0, 1].bar(reg_metrics, reg_values, alpha=0.8, color='lightgreen')
            axes[0, 1].set_title('Regression Metrics')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].grid(True, alpha=0.3)
        
        # F1 점수들 (있는 경우)
        if 'f1_macro' in all_metrics['val']:
            f1_metrics = ['f1_macro', 'f1_micro', 'f1_weighted']
            f1_values = [all_metrics['val'].get(m, 0) for m in f1_metrics]
            
            axes[1, 0].bar(f1_metrics, f1_values, alpha=0.8, color='gold')
            axes[1, 0].set_title('F1 Scores')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 손실 분포
        if 'loss_distribution' in all_metrics:
            axes[1, 1].hist(all_metrics['loss_distribution'], bins=30, alpha=0.7, color='purple')
            axes[1, 1].set_title('Loss Distribution')
            axes[1, 1].set_xlabel('Loss Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'metrics_comparison_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_attention_heatmap(self, attention_weights, tokens, save_name):
        """어텐션 히트맵 시각화"""
        plt.figure(figsize=(12, 8))
        sns.heatmap(attention_weights, 
                   xticklabels=tokens[:attention_weights.shape[1]], 
                   yticklabels=tokens[:attention_weights.shape[0]],
                   cmap='Blues', annot=False, cbar=True)
        plt.title('Attention Weights Heatmap')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        
        save_path = os.path.join(self.save_dir, f'{save_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_summary_dashboard(self, final_metrics, training_history):
        """최종 대시보드 생성"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 전체 제목
        fig.suptitle('Accident Caption Model - Training Summary Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # 1. 손실 곡선 (전체 기간)
        ax1 = fig.add_subplot(gs[0, :2])
        epochs = range(len(training_history['train_loss']))
        ax1.plot(epochs, training_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, training_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 텍스트 메트릭 진행
        ax2 = fig.add_subplot(gs[0, 2:])
        metrics_to_plot = ['bleu', 'rouge1', 'rougeL']
        for metric in metrics_to_plot:
            if metric in training_history:
                ax2.plot(epochs, training_history[metric], label=metric.upper(), linewidth=2)
        ax2.set_title('Text Generation Metrics Progress', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 최종 메트릭 비교
        ax3 = fig.add_subplot(gs[1, :2])
        final_text_metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'semantic_similarity']
        final_values = [final_metrics.get(m, 0) for m in final_text_metrics]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(final_text_metrics)))
        bars = ax3.bar(final_text_metrics, final_values, color=colors, alpha=0.8)
        ax3.set_title('Final Text Generation Metrics', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Score')
        ax3.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, val in zip(bars, final_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. 회귀 메트릭
        ax4 = fig.add_subplot(gs[1, 2:])
        if 'mae' in final_metrics:
            reg_metrics = ['mae', 'rmse', 'ndcg']
            reg_values = [final_metrics.get(m, 0) for m in reg_metrics]
            
            bars = ax4.bar(reg_metrics, reg_values, color=['orange', 'red', 'green'], alpha=0.8)
            ax4.set_title('Regression Metrics', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Score')
            
            for bar, val in zip(bars, reg_values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(reg_values)*0.01, 
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. F1 점수들
        ax5 = fig.add_subplot(gs[2, :2])
        if 'f1_macro' in final_metrics:
            f1_metrics = ['f1_macro', 'f1_micro', 'f1_weighted']
            f1_values = [final_metrics.get(m, 0) for m in f1_metrics]
            
            bars = ax5.bar(f1_metrics, f1_values, color=['purple', 'magenta', 'cyan'], alpha=0.8)
            ax5.set_title('F1 Scores', fontsize=14, fontweight='bold')
            ax5.set_ylabel('F1 Score')
            
            for bar, val in zip(bars, f1_values):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. 손실 분포
        ax6 = fig.add_subplot(gs[2, 2:])
        if 'loss_distribution' in final_metrics:
            ax6.hist(final_metrics['loss_distribution'], bins=30, alpha=0.7, color='brown', edgecolor='black')
            ax6.set_title('Final Loss Distribution', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Loss Value')
            ax6.set_ylabel('Frequency')
            ax6.axvline(np.mean(final_metrics['loss_distribution']), color='red', 
                       linestyle='--', linewidth=2, label=f'Mean: {np.mean(final_metrics["loss_distribution"]):.3f}')
            ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. 메트릭 요약 테이블
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('tight')
        ax7.axis('off')
        
        summary_data = []
        for metric, value in final_metrics.items():
            if isinstance(value, (int, float)) and not metric.endswith('_distribution'):
                summary_data.append([metric.upper(), f'{value:.4f}'])
        
        if summary_data:
            table = ax7.table(cellText=summary_data, 
                             colLabels=['Metric', 'Value'],
                             cellLoc='center',
                             loc='center',
                             bbox=[0.0, 0.0, 1.0, 1.0])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 2)
            
            # 테이블 스타일링
            for i in range(len(summary_data) + 1):
                for j in range(2):
                    cell = table[(i, j)]
                    if i == 0:  # 헤더
                        cell.set_facecolor('#4472C4')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.save_dir, f'training_dashboard_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

# ---------------- Enhanced Dataset ----------------
class AccidentVideoDataset(Dataset):
    def __init__(self, meta_path, video_dir, tokenizer, num_frames=NUM_FRAMES, 
                 frame_size=FRAME_SIZE, meta_drop_prob=0.0, track_json_dir=None, split="train"):
        import csv, json
        self.rows = []
        ext = os.path.splitext(meta_path)[1].lower()
        self.split = split

        self.track_json_dir = track_json_dir
        
        try:
            if ext == ".csv":
                with open(meta_path, "r", encoding="utf-8-sig") as f:
                    rdr = csv.DictReader(f)
                    for r in rdr: 
                        self.rows.append(r)
            elif ext in (".json", ".jsonl"):
                with open(meta_path, "r", encoding="utf-8") as f:
                    if ext == ".jsonl":
                        for line in f:
                            if line.strip(): 
                                self.rows.append(json.loads(line))
                    else:
                        data = json.load(f)
                        if isinstance(data, list): 
                            self.rows = data
        except Exception as e:
            print(f"Error loading metadata from {meta_path}: {e}")
            self.rows = []
            
        self.video_dir = video_dir
        self.tok = tokenizer
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.meta_drop_prob = meta_drop_prob

    def __len__(self): 
        return len(self.rows)

    def sample_frames_efficient(self, video_path):
        """메모리 효율적인 프레임 샘플링"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                cap.release()
                raise RuntimeError(f"Empty video: {video_path}")
            
            # 시간적으로 균등하게 분포된 프레임 선택
            if total_frames <= self.num_frames:
                idxs = list(range(total_frames))
            else:
                # 전체 비디오에서 균등 샘플링
                step = total_frames / (self.num_frames + 1)
                idxs = [int(step * (i + 1)) for i in range(self.num_frames)]
            
            frames = []
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total_frames - 1))
                ret, frame = cap.read()
                if ret:
                    # BGR to RGB, resize
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (self.frame_size, self.frame_size))
                    frames.append(frame)
                else:
                    if frames:
                        frames.append(frames[-1])
                    else:
                        frames.append(np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8))
            
            cap.release()
            
            # 패딩
            while len(frames) < self.num_frames:
                frames.append(frames[-1] if frames else np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8))
            
            # tensor 변환
            frames = np.stack(frames[:self.num_frames], axis=0)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
            
            return frames
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return torch.zeros((self.num_frames, 3, self.frame_size, self.frame_size))

    def _load_track_json(self, vid: str):
        if not self.track_json_dir:
            return None

        # 후보 패턴: "vid.*json"
        pattern = os.path.join(self.track_json_dir, vid + "*.json")
        candidates = glob.glob(pattern)
        if not candidates:
            return None
        
        # 우선순위: .tracks.json > .json
        candidates.sort(key=lambda x: (not x.endswith(".tracks.json"), x))
        try:
            with open(candidates[0], "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print("track json load error:", e)
            return None

    def _vectorize_tracks(self, tj):
        """movement(3) + side(3) + entry(2) + collision t1,t2(2) = 10-dim 벡터"""
        vec = np.zeros(10, dtype=np.float32)
        slot_labels = {"movement": -1, "side": -1, "entry": -1}

        if not tj or not tj.get("primary_pair") or not tj.get("tracks"):
            return vec, slot_labels, "none"

        pid_a, pid_b = tj["primary_pair"]
        tracks_by_id = {t.get("id"): t for t in tj.get("tracks", [])}
        A, B = tracks_by_id.get(pid_a), tracks_by_id.get(pid_b)
        other = max([x for x in (A, B) if x], key=lambda t: len(t.get("frames", [])), default=None)

        # 1) movement
        mv_map = {"straight": 0, "left_turn": 1, "right_turn": 2}
        mv_id = mv_map.get((other or {}).get("move", "straight"), 0)
        vec[mv_id] = 1.0
        slot_labels["movement"] = mv_id

        # 2) side
        side_map = {"side_left": 0, "main": 1, "side_right": 2}
        sd_id = side_map.get((other or {}).get("entry_side", "main"), 1)
        vec[3 + sd_id] = 1.0
        slot_labels["side"] = sd_id

        # 3) entry order
        ent_id = -1
        entry = tj.get("entry_order")
        if entry and other and ("earlier_id" in entry):
            ent_id = 0 if other.get("id") == entry.get("earlier_id") else 1
            vec[6 + ent_id] = 1.0
            slot_labels["entry"] = ent_id

        # 4) collision window 정규화
        t1 = float(tj.get("collision", {}).get("t1", 0.0) or 0.0)
        t2 = float(tj.get("collision", {}).get("t2", 0.0) or 0.0)
        # last_ts가 없을 수도 있으니 안전하게
        last_candidates = [float(t.get("last_ts", 0.0) or 0.0) for t in tj.get("tracks", [])]
        dur = max(1e-3, max(last_candidates + [t2]))
        vec[8] = float(np.clip(t1 / dur, 0, 1))
        vec[9] = float(np.clip(t2 / dur, 0, 1))

        # hint 텍스트
        mv_txt = ["straight", "left_turn", "right_turn"][mv_id]
        sd_txt = ["left", "main", "right"][sd_id]
        ent_txt = ["other_earlier", "other_later"][ent_id] if ent_id != -1 else "unknown"
        # hint = f"other_move={mv_txt}, other_side={sd_txt}, entry_order={ent_txt}, collision_norm=[{vec[8]:.2f},{vec[9]:.2f}]"
        hint = f"other_move={mv_txt}, other_side={sd_txt}, entry_order={ent_txt}"
        return vec, slot_labels, hint
    # def __getitem__(self, idx):
    #     r = self.rows[idx]
    #     vid = r["video_name"]
        
    #     # 비디오 파일 찾기
    #     video_path = None
    #     for ext in (".mp4", ".mkv", ".mov", ".avi", ".webm"):
    #         p = os.path.join(self.video_dir, vid + ext)
    #         if os.path.exists(p):
    #             video_path = p
    #             break
        
    #     if video_path is None:
    #         print(f"Video not found: {vid}")
    #         frames = torch.zeros((self.num_frames, 3, self.frame_size, self.frame_size))
    #     else:
    #         frames = self.sample_frames_efficient(video_path)
        
    #     # T5용 입력 구성 (task prefix 방식)
    #     src_text = ""
    #     if random.random() >= self.meta_drop_prob:
    #         place = r.get('accident_place_feature', '').strip()
    #         dashcam = r.get('dashcam_vehicle_info', '').strip()
    #         other = r.get('other_vehicle_info', '').strip()
            
    #         context_parts = []
    #         if place: context_parts.append(f"location: {place}")
    #         if dashcam: context_parts.append(f"dashcam vehicle: {dashcam}")
    #         if other: context_parts.append(f"other vehicle: {other}")
            
    #         if context_parts:
    #             src_text = f"Generate accident description with context: {'; '.join(context_parts)}"
        
    #     if not src_text:
    #         src_text = "Generate accident description from video"
        
    #     tgt_text = r.get("generated_caption", "").strip()
    #     if not tgt_text:
    #         tgt_text = "No description available"
        
    #     # T5 토크나이징
    #     try:
    #         src_encoding = self.tok(
    #             src_text, 
    #             truncation=True, 
    #             max_length=512, 
    #             padding=False, 
    #             return_tensors=None
    #         )
            
    #         # T5는 target에도 task prefix 없이 직접 텍스트
    #         with self.tok.as_target_tokenizer():
    #             tgt_encoding = self.tok(
    #                 tgt_text, 
    #                 truncation=True, 
    #                 max_length=256, 
    #                 padding=False, 
    #                 return_tensors=None
    #             )
            
    #         input_ids = torch.tensor(src_encoding["input_ids"], dtype=torch.long)
    #         attention_mask = torch.tensor(src_encoding["attention_mask"], dtype=torch.long)
    #         # labels = torch.tensor(tgt_encoding["input_ids"], dtype=torch.long)
            
    #         #9/1 수정 위에 주석
    #         labels = torch.tensor(tgt_encoding["input_ids"], dtype=torch.long)
    #         labels = labels.clamp(0, self.tok.vocab_size - 1)
    #         labels[labels == self.tok.pad_token_id] = -100
    #         # 9/1 수정 위에 주석

    #         print(input_ids.shape, attention_mask.shape, labels.shape)
    #         print(input_ids.max(), labels.max())
    #         print(input_ids.min(), labels.min())
            
    #     except Exception as e:
    #         print(f"Tokenization error for sample {idx}: {e}")
    #         input_ids = torch.tensor([self.tok.pad_token_id], dtype=torch.long)
    #         attention_mask = torch.tensor([1], dtype=torch.long)
    #         labels = torch.tensor([self.tok.pad_token_id], dtype=torch.long)
        
    #     return {
    #         "frames": frames, 
    #         "input_ids": input_ids, 
    #         "attention_mask": attention_mask, 
    #         "labels": labels,
    #         "video_name": vid,
    #         "caption": tgt_text,
    #         "caption_length": len(tgt_text.split())
    #     }

    # 9/1 수정 위에 주석
    def __getitem__(self, idx):
        r = self.rows[idx]
        vid = r.get("video_name", f"unknown_{idx}")
        
        # ----------------- 비디오 처리 -----------------
        video_path = None
        for ext in (".mp4", ".mkv", ".mov", ".avi", ".webm"):
            p = os.path.join(self.video_dir, vid + ext)
            if os.path.exists(p):
                video_path = p
                break
        
        if video_path is None:
            print(f"Video not found: {vid}")
            frames = torch.zeros((self.num_frames, 3, self.frame_size, self.frame_size))
        else:
            frames = self.sample_frames_efficient(video_path)
        
        # ----------------- 입력 텍스트(src_text) 구성 -----------------
        # src_text = (
        #     "Task: Describe an accident.\n"
        #     "Include: intersection type, both vehicles' movements, and the collision time range in seconds.\n"
        #     "Output one sentence ending with 'seconds.'."
        # )
        src_text = (
            "Task: Describe an accident.\n"
            "Include: intersection type, both vehicles' movements.\n"
            "Output exactly one sentence and do not add any extra sentences or explanations."
        )

        # === NEW: 트래킹 JSON -> 벡터/라벨/힌트 ===
        track_json = self._load_track_json(vid)                      # NEW
        track_vec_np, slot_labels, track_hint = self._vectorize_tracks(track_json)  # NEW
        if self.split == "train" and track_hint and track_hint != "none":
                src_text = f"{src_text}\nScene hints: {track_hint}"      # NEW
        
        # (필요하면 메타 기반 context 추가 — 현재는 주석)
        # place = str(r.get('accident_place_feature') or "").strip()
        # dashcam = str(r.get('dashcam_vehicle_info') or "").strip()
        # other = str(r.get('other_vehicle_info') or "").strip()

        # ----------------- 정답 텍스트(tgt_text) -----------------
        tgt_text = r.get("generated_caption")
        tgt_text = str(tgt_text).strip() if tgt_text is not None else "No description available"
        
        # cutoff = "seconds."
        # if cutoff in tgt_text:
        #     tgt_text = tgt_text.split(cutoff)[0] + cutoff
        
        m = re.search(r"\bthe collision\b", tgt_text, flags=re.IGNORECASE)
        if m:
            tgt_text = tgt_text[:m.start()].rstrip()
            # 끝이 .!? 로 안 끝나면 마침표 하나 붙여 깔끔하게
            if not tgt_text.endswith(('.', '!', '?')):
                tgt_text += '.'

        # ----------------- 토크나이징 -----------------
        try:
            # ✅ 소스 입력: 파이토치 텐서 반환 + squeeze(0)로 1D
            src_encoding = self.tok(
                src_text,
                truncation=True,
                max_length=512,
                padding=False,
                return_tensors="pt",
            )
            input_ids = src_encoding["input_ids"].squeeze(0)           # [L]
            attention_mask = src_encoding["attention_mask"].squeeze(0) # [L]

            # ✅ 타깃(라벨): 여기서는 패딩하지 말고 1D로만 반환 (패딩은 collate_fn에서)
            tgt_encoding = self.tok(
                tgt_text,
                truncation=True,
                max_length=64,
                padding=False,
                return_tensors="pt",
            )
            labels = tgt_encoding["input_ids"].squeeze(0)              # [L_tgt]
            
        except Exception as e:
            print(f"Tokenization error for sample {idx}: {e}")
            input_ids = torch.tensor([self.tok.pad_token_id], dtype=torch.long)
            attention_mask = torch.tensor([1], dtype=torch.long)
            labels = torch.tensor([-100], dtype=torch.long)
            # 안전하게 track_vec도 기본값 세팅
            track_vec_np = np.zeros(10, dtype=np.float32)              # NEW
            slot_labels = {"movement": -1, "side": -1, "entry": -1}    # NEW
        
        return {
            "frames": frames,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "video_name": vid,
            "caption": tgt_text,
            "caption_length": len(tgt_text.split()),
            # === NEW: 보조 입력/라벨 ===
            "track_vec": torch.from_numpy(track_vec_np),   # (10,) float32
            "slot_labels": slot_labels                     # dict(int)
        }
    # 9/1 수정 위에 주석


def collate_fn(batch):
    """배치 콜레이트 함수"""
    PAD_ID = 0 

    frames = torch.stack([item["frames"] for item in batch])
    video_names = [item["video_name"] for item in batch]
    captions = [item["caption"] for item in batch]
    caption_lengths = [item["caption_length"] for item in batch]
    
    # 텍스트 패딩
    max_input_len = max(item["input_ids"].size(0) for item in batch)
    max_label_len = max(item["labels"].size(0) for item in batch)
    
    input_ids = []
    attention_mask = []
    labels = []
    
    for item in batch:
        # Input padding (pad_id=0)
        pad_len = max_input_len - item["input_ids"].size(0)
        input_ids.append(F.pad(item["input_ids"], (0, pad_len), value=PAD_ID))
        attention_mask.append(F.pad(item["attention_mask"], (0, pad_len), value=0))

        # Label padding: 일단 pad_id로 채워놓고 → -100으로 바꿔서 무시
        lpad = max_label_len - item["labels"].size(0)
        lab = F.pad(item["labels"], (0, lpad), value=PAD_ID)
        lab[lab == PAD_ID] = -100   # ✅ 라벨 패딩은 무시
        labels.append(lab)

    return {
        "frames": frames,
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
        "video_names": video_names,
        "captions": captions,
        "caption_lengths": caption_lengths,
    }

# ---------------- Enhanced Model ----------------
def shift_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int = 0):
    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[:, 0] = decoder_start_token_id
    shifted[:, 1:] = input_ids[:, :-1].clone()
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted
    
class MultimodalAccidentCaptioner(nn.Module):
    #9/1 수정 num_frames=16 추가
    def __init__(self, t5_name=T5_NAME, video_model=VIDEO_MODEL, vocab_size=None, num_frames=16, warmup_video_steps=50):
        super().__init__()
        
        #9/1 수정 num_frames=16 추가
        self.num_frames = num_frames
        self.warmup_video_steps = warmup_video_steps
        #9/1 수정 num_frames=16 추가

        # T5 모델 로드
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_name)
        self.t5.config.use_cache = False   
        if vocab_size:
            self.t5.resize_token_embeddings(vocab_size)
        
        # VideoMAE 인코더
        self.video_encoder = VideoMAEModel.from_pretrained(video_model)
        self.videoprocessor = VideoMAEImageProcessor.from_pretrained(video_model)  # ✅ 추가: processor
        video_dim = self.video_encoder.config.hidden_size
        
        # 크로스 어텐션 모듈
        t5_dim = self.t5.config.d_model
        self.video_projection = nn.Sequential(
            nn.Linear(video_dim, t5_dim),
            nn.LayerNorm(t5_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 어댑터 레이어
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=t5_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.video_gate = nn.Parameter(torch.tensor(1.5)) 
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(t5_dim * 2, t5_dim),
            nn.LayerNorm(t5_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 객체 검출 헤드 (보조 태스크)
        self.object_classifier = nn.Sequential(
            nn.Linear(t5_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, len(VEHICLE_CLASS_IDS))
        )
        
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
    # def encode_video(self, frames):
    #     """비디오 인코딩"""
    #     B, T, C, H, W = frames.shape
    #     frames = frames.view(-1, C, H, W)
        
    #     # VideoMAE에 맞는 전처리
    #     frames = torch.nn.functional.interpolate(frames, size=(224, 224), mode='bilinear')
    #     frames = frames.view(B, T, C, 224, 224)
        
    #     with torch.no_grad():
    #         video_outputs = self.video_encoder(pixel_values=frames)
        
    #     video_features = video_outputs.last_hidden_state
    #     video_features = self.video_projection(video_features)
        
    #     return video_features

    #9/1 수정 (위에 주석처리했음)
    def sample_frames(self, frames):
        T = frames.shape[0]
        num_frames = self.num_frames
        if T >= num_frames:
            indices = torch.linspace(0, T-1, num_frames).long()
            return frames[indices]
        else:
            pad = num_frames - T
            last_frame = frames[-1:].repeat(pad, 1, 1, 1)
            return torch.cat([frames, last_frame], dim=0)

    def encode_video(self, frames):
        """
        frames: (B, T_var, C, H, W), float in [0,1]
        returns: (B, N_patches, d_model)
        """
        device = frames.device
        B, T_var, C, H, W = frames.shape

        videos = []
        for b in range(B):
            # 1) T 고정 샘플링: (T, C, H, W)
            f = self.sample_frames(frames[b])

            # 2) [0,1] -> uint8, 각 프레임을 HWC로 변환
            f_uint8 = (f.clamp(0, 1) * 255).to(torch.uint8)  # (T, C, H, W)
            frames_hwc = [f_uint8[t].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
                        for t in range(f_uint8.size(0))]

            # 3) "비디오 = 프레임 리스트"
            videos.append(frames_hwc)

        # 4) processor로 리사이즈/크롭/정규화
        #    (videos: List[List[np.ndarray(H,W,C,uint8)]])
        processed = self.videoprocessor(
            videos,
            return_tensors="pt",          # pixel_values: (B, T, C, H, W)
            # do_rescale=True 가 기본이라 uint8 -> float/255 자동 처리됩니다.
        )
        pixel_values = processed["pixel_values"].to(device)

        # 5) VideoMAE forward
        video_outputs = self.video_encoder(pixel_values=pixel_values)  # (B, N_patches, hidden)
        video_features = video_outputs.last_hidden_state

        # 6) T5 차원으로 사영
        video_features = self.video_projection(video_features)         # (B, N_patches, d_model)
        return video_features
    #9/1 수정
    

    def forward(self, frames, input_ids, attention_mask, labels=None, global_step: int = 0, force_use_video: bool = False):
        """순전파"""
        # 1) T5 인코더만 먼저
        encoder_outputs = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = encoder_outputs.last_hidden_state  # [B, L_txt, d_model]

        # 2) warmup 지나면 그 때만 비디오 인코딩 (중복 호출 X)
        use_video = (global_step >= self.warmup_video_steps) or force_use_video

        if use_video:
            # 비디오 인코딩: 여기서만 호출 (중복 X)
            video_features = self.encode_video(frames)  # [B, T', d]
            enhanced_text, _ = self.cross_attention(
                query=text_features,
                key=video_features,
                value=video_features
            )
            alpha = torch.sigmoid(self.video_gate)  # learnable scalar in [0,1]
            fused_features = self.fusion_layer(
                torch.cat([text_features, alpha * enhanced_text], dim=-1)
            )
        else:
            enhanced_text = torch.zeros_like(text_features)
            fused_features = self.fusion_layer(
                torch.cat([text_features, enhanced_text], dim=-1)
            )

        # 4) 디코딩 / 손실
        if labels is not None:
            if labels.dim() == 3:
                labels = labels.squeeze(1)  # [B, L]

            # collate_fn에서 이미 -100 마스킹 했으면 아래 두 줄은 생략 가능
            # pad_id = self.t5.config.pad_token_id
            # labels = labels.clone(); labels[labels == pad_id] = -100

            enc_out = BaseModelOutput(last_hidden_state=fused_features)  # [B, L_enc, d_model]

            outputs = self.t5(
                input_ids=None,
                encoder_outputs=enc_out,
                attention_mask=attention_mask,  # encoder attention mask
                labels=labels,
                use_cache=False,
                return_dict=True,
            )

            logits = outputs.logits  # [B, L_tgt, V]

            # 라벨 스무딩(0.1)로 메인 로스 재계산 (초기 loss 낮추기)
            loss_mask = labels.ne(-100)
            main_loss = F.cross_entropy(
                logits[loss_mask], labels[loss_mask],
                label_smoothing=0.05
            )

            # 보조손실은 당분간 OFF (원하면 아주 작게)
            aux_loss = torch.tensor(0.0, device=frames.device)
            total_loss = main_loss

            return {
                'loss': total_loss,
                'main_loss': main_loss,
                'aux_loss': aux_loss,
                'logits': logits,
                'obj_logits': None
            }
        else:
            encoder_outputs = BaseModelOutput(last_hidden_state=fused_features)
            return self.t5.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                max_length=64,
                do_sample=False,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                length_penalty=1.1,
                early_stopping=True,
                min_length=30
            )

# ---------------- Training Utils ----------------
class WandBLogger:
    def __init__(self, project_name, entity=None, config=None):
        self.project_name = project_name
        self.entity = entity
        self.run = None
        
        if config is None:
            config = {
                'model': 'T5-VideoMAE-Multimodal',
                'dataset': 'Accident Videos',
                'batch_size': BATCH_SIZE,
                'learning_rate': LR,
                'epochs': EPOCHS,
                'num_frames': NUM_FRAMES,
                'frame_size': FRAME_SIZE
            }
        
        try:
            self.run = wandb.init(
                project=project_name,
                entity=entity,
                config=config,
                tags=['multimodal', 'video-captioning', 'accident-detection']
            )
            print("WandB initialized successfully")
        except Exception as e:
            print(f"Failed to initialize WandB: {e}")
            self.run = None
    
    def log_metrics(self, metrics_dict, step=None, commit=True):
        """메트릭 로깅"""
        if self.run:
            try:
                self.run.log(metrics_dict, step=step, commit=commit)
            except Exception as e:
                print(f"Failed to log to WandB: {e}")
    
    def log_images(self, images_dict, step=None):
        """이미지 로깅"""
        if self.run:
            try:
                wandb_images = {}
                for key, path in images_dict.items():
                    if os.path.exists(path):
                        wandb_images[key] = wandb.Image(path)
                
                if wandb_images:
                    self.run.log(wandb_images, step=step)
            except Exception as e:
                print(f"Failed to log images to WandB: {e}")
    
    def finish(self):
        """WandB 실행 종료"""
        if self.run:
            self.run.finish()

def create_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY):
    """최적화기 생성"""
    # 레이어별 학습률 설정
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and "video_encoder" not in n],
            "weight_decay": weight_decay,
            "lr": lr
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and "video_encoder" not in n],
            "weight_decay": 0.0,
            "lr": lr
        },
        {
            "params": [p for n, p in model.named_parameters() if "video_encoder" in n],
            "weight_decay": weight_decay,
            "lr": lr * 0.1  # 비디오 인코더는 낮은 학습률
        }
    ]
    
    return torch.optim.AdamW(optimizer_grouped_parameters)

def create_scheduler(optimizer, num_training_steps, warmup_steps=WARMUP_STEPS):
    """학습률 스케줄러 생성"""
    from transformers import get_linear_schedule_with_warmup
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

# ---------------- Training Loop ----------------
def train_epoch(model, train_loader, optimizer, scheduler, accelerator, wandb_logger, epoch, visualizer):
    """한 에포크 훈련"""
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_aux_loss = 0
    step = 0
    
    
    metrics_calc = MetricsCalculator()
    all_pred_lengths, all_true_lengths = [], []
    
    pbar = tqdm(
                train_loader,
                desc=f"Training Epoch {epoch}",
                disable=not accelerator.is_local_main_process
            )
    
    for batch_idx, batch in enumerate(pbar):

        global_step = epoch * len(train_loader) + batch_idx

        with accelerator.accumulate(model):
            outputs = model(
                frames=batch["frames"],
                input_ids=batch["input_ids"], 
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                global_step=global_step 
            )
            
            loss = outputs['loss']
            main_loss = outputs['main_loss']
            aux_loss = outputs['aux_loss']
            
            # 역전파
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # 손실 누적
            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_aux_loss += aux_loss.item()
            
            # 메트릭 계산 (매 N 스텝마다)
            if batch_idx % 10 == 0:
                # 현재 배치에 대한 예측 생성
                model.eval()
                with torch.no_grad():
                    generated_ids = model(
                        frames=batch["frames"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        force_use_video=True
                    )
                    
                    tokenizer = train_loader.dataset.tok
                    pred_texts = [tokenizer.decode(ids, skip_special_tokens=True) 
                                for ids in generated_ids]
                    ref_texts = batch["captions"]
                    
                    metrics_calc.add_batch(pred_texts, ref_texts, loss.item())
                    all_pred_lengths.extend([len(p.split()) for p in pred_texts])
                    all_true_lengths.extend(batch["caption_lengths"])

                    # 👇 여기에 출력 추가
                    if accelerator.is_local_main_process:
                         print(f"\n[Step {batch_idx}]")
                         print(f"Prediction: {pred_texts[0]}")
                         print(f"Reference : {ref_texts[0]}")
                         print(f"Loss: {loss.item():.4f}, Main: {main_loss.item():.4f}, Aux: {aux_loss.item():.4f}")
                model.train()
            
            # 진행률 업데이트
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Main': f'{main_loss.item():.4f}',
                'Aux': f'{aux_loss.item():.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # WandB 로깅 (매 스텝)
            if wandb_logger:
                wandb_logger.log_metrics({
                    'train/step_loss': loss.item(),
                    'train/main_loss': main_loss.item(),
                    'train/aux_loss': aux_loss.item(),
                    'train/learning_rate': scheduler.get_last_lr()[0],
                    'train/step': step
                }, commit=True)
            
            step += 1
    
    # 에포크 평균 계산
    avg_loss = total_loss / len(train_loader)
    avg_main_loss = total_main_loss / len(train_loader)
    avg_aux_loss = total_aux_loss / len(train_loader)
    
    # 텍스트 메트릭 계산
    text_metrics = metrics_calc.compute_text_metrics()
    
    # 회귀 메트릭 (캡션 길이 예측)
    # if metrics_calc.predictions and batch["caption_lengths"]:
    #     pred_lengths = [len(pred.split()) for pred in metrics_calc.predictions[-len(batch["caption_lengths"]):]]
    #     true_lengths = batch["caption_lengths"]
    #     reg_metrics = metrics_calc.compute_regression_metrics(pred_lengths, true_lengths)
    #     text_metrics.update(reg_metrics)
    if all_pred_lengths and all_true_lengths:
            reg_metrics = metrics_calc.compute_regression_metrics(all_pred_lengths, all_true_lengths)
            text_metrics.update(reg_metrics)
    
    all_metrics = {
        'loss': avg_loss,
        'main_loss': avg_main_loss,
        'aux_loss': avg_aux_loss,
        **text_metrics,
        'loss_distribution': metrics_calc.losses
    }
    
    return all_metrics

def validate_epoch(model, val_loader, accelerator, wandb_logger, epoch, visualizer):
    """검증"""
    model.eval()
    total_loss = 0
    total_main_loss = 0
    total_aux_loss = 0
    
    metrics_calc = MetricsCalculator()
    all_predictions = []
    all_references = []
    all_losses = []
    all_pred_lengths = []
    all_true_lengths = []
    all_video_names = []

    skipped_batches_local = 0
    
    pbar = tqdm(
                val_loader,
                desc=f"Validation Epoch {epoch}",
                disable=not accelerator.is_local_main_process
            )
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # 손실 계산
            outputs = model(
                frames=batch["frames"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"], 
                labels=batch["labels"]
            )
            
            loss = outputs['loss']
            main_loss = outputs['main_loss']
            aux_loss = outputs['aux_loss']
            
            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_aux_loss += aux_loss.item()
            all_losses.append(loss.item())
            all_video_names.extend(batch["video_names"])
            
            # 텍스트 생성
            generated_ids = model(
                frames=batch["frames"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                force_use_video=True
            )
            
            tokenizer = val_loader.dataset.tok
            pred_texts = [tokenizer.decode(ids, skip_special_tokens=True) 
                         for ids in generated_ids]
            ref_texts = batch["captions"]
            
            all_predictions.extend(pred_texts)
            all_references.extend(ref_texts)
            
            # 길이 메트릭을 위한 데이터
            pred_lengths = [len(pred.split()) for pred in pred_texts]
            true_lengths = batch["caption_lengths"]
            all_pred_lengths.extend(pred_lengths)
            all_true_lengths.extend(true_lengths)
            
            # 배치 메트릭 추가
            metrics_calc.add_batch(pred_texts, ref_texts, loss.item())
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Main': f'{main_loss.item():.4f}',
                'Aux': f'{aux_loss.item():.4f}'
            })
    
    # 평균 계산
    avg_loss = total_loss / len(val_loader)
    avg_main_loss = total_main_loss / len(val_loader) 
    avg_aux_loss = total_aux_loss / len(val_loader)
    
    # 전체 메트릭 계산
    metrics_calc.predictions = all_predictions
    metrics_calc.references = all_references
    metrics_calc.losses = all_losses
    
    text_metrics = metrics_calc.compute_text_metrics()
    reg_metrics = metrics_calc.compute_regression_metrics(all_pred_lengths, all_true_lengths)
    
    # F1 점수 계산 (길이 기반 분류)
    length_bins = [0, 5, 10, 20, float('inf')]
    pred_length_classes = np.digitize(all_pred_lengths, length_bins) - 1
    true_length_classes = np.digitize(all_true_lengths, length_bins) - 1
    f1_metrics = metrics_calc.compute_classification_metrics(pred_length_classes, true_length_classes)
    
    all_metrics = {
        'loss': avg_loss,
        'main_loss': avg_main_loss,
        'aux_loss': avg_aux_loss,
        'loss_distribution': all_losses,
        **text_metrics,
        **reg_metrics,
        **f1_metrics
    }
    
    return all_metrics, all_predictions, all_references, all_video_names

# ---------------- Main Training Function ----------------
def train_model():
    """메인 훈련 함수"""

    final_results = {} 
    start_time = time.time()
    print("⏳ Training started...")

    # 디렉터리 생성
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    # Accelerator 초기화
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision="fp16" if torch.cuda.is_available() else "no",
        kwargs_handlers=[ddp_kwargs]
    )
    
    # 토크나이저 & 모델
    tokenizer = T5Tokenizer.from_pretrained(T5_NAME)
    
    # 특수 토큰 추가
    special_tokens = ["<video>", "<object>", "<location>", "<vehicle>"]
    tokenizer.add_tokens(special_tokens)
    
    model = MultimodalAccidentCaptioner(
        t5_name=T5_NAME,
        video_model=VIDEO_MODEL, 
        vocab_size=len(tokenizer)
    )
    
    # 데이터셋 & 로더
    train_dataset = AccidentVideoDataset(
        TRAIN_META, VIDEO_DIR, tokenizer, 
        num_frames=NUM_FRAMES, frame_size=FRAME_SIZE, 
        meta_drop_prob=META_DROP_PROB,
        track_json_dir=TRACK_JSON_DIR_TRAIN,
        split="train"
    )
    
    val_dataset = AccidentVideoDataset(
        VAL_META, VAL_VIDEO_DIR, tokenizer,
        num_frames=NUM_FRAMES, frame_size=FRAME_SIZE,
        meta_drop_prob=0.0,
        track_json_dir=TRACK_JSON_DIR_VAL,
        split="val"
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, 
        shuffle=True, drop_last=False, collate_fn=collate_fn, num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, drop_last=False, collate_fn=collate_fn, num_workers=2
    )
    
    # 옵티마이저 & 스케줄러
    optimizer = create_optimizer(model)
    num_training_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
    scheduler = create_scheduler(optimizer, num_training_steps)
    
    # Accelerator로 준비
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    # WandB & 시각화 초기화
    if accelerator.is_main_process:
        wandb_logger = WandBLogger(WANDB_PROJECT, WANDB_ENTITY)
        visualizer = MetricsVisualizer(PLOT_DIR)
    else:
        wandb_logger = None
        visualizer = None
    
    # 훈련 기록
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'bleu': [],
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'mae': [],
        'rmse': [],
        'ndcg': [],
        'f1_macro': [],
        'f1_micro': [],
        'f1_weighted': []
    }
    
    best_val_bleu = 0.0
    patience_counter = 0
    patience = 999999
    
    print(f"Starting training on {accelerator.device}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    for epoch in range(EPOCHS):
        accelerator.print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        # ------- Train -------
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            accelerator, wandb_logger, epoch, visualizer
        )

        # 이번 에폭에 검증을 할지? (마지막 에폭은 강제 검증)
        do_validate = ((epoch + 1) % EVAL_EVERY_N_EPOCHS == 0) or (epoch == EPOCHS - 1)

        if do_validate:
            # ------- Validate -------
            val_metrics, val_predictions, val_references, val_video_names = validate_epoch(
                model, val_loader, accelerator, wandb_logger, epoch, visualizer
            )

            # 기록 업데이트
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['val_loss'].append(val_metrics['loss'])
            for metric in ['bleu', 'rouge1', 'rouge2', 'rougeL',
                        'mae', 'rmse', 'ndcg', 'f1_macro', 'f1_micro', 'f1_weighted']:
                if metric in val_metrics:
                    training_history[metric].append(val_metrics[metric])

            # 결과 출력
            accelerator.print(f"Train Loss: {train_metrics['loss']:.4f}")
            accelerator.print(f"Val   Loss: {val_metrics['loss']:.4f}")
            accelerator.print(f"Val   BLEU: {val_metrics.get('bleu', 0):.4f}")
            accelerator.print(f"Val   ROUGE-L: {val_metrics.get('rougeL', 0):.4f}")
            accelerator.print(f"Val   MAE: {val_metrics.get('mae', 0):.4f}")
            accelerator.print(f"Val   F1-Macro: {val_metrics.get('f1_macro', 0):.4f}")

            # WandB 로깅 (검증 있는 에폭에만)
            if accelerator.is_main_process and wandb_logger:
                combined_metrics = {}
                for k, v in train_metrics.items():
                    if k != 'loss_distribution':
                        combined_metrics[f'train/{k}'] = v
                for k, v in val_metrics.items():
                    if k != 'loss_distribution':
                        combined_metrics[f'val/{k}'] = v
                global_step = epoch * len(train_loader)
                combined_metrics['epoch'] = epoch
                wandb_logger.log_metrics(combined_metrics, step=global_step)

            # 시각화도 검증 있는 에폭에서만 (원래 2에폭마다였다면 그대로 유지 가능)
            if accelerator.is_main_process and visualizer:
                try:
                    curves_path = visualizer.plot_training_curves(
                        training_history['train_loss'],
                        training_history['val_loss'],
                        [{'bleu': b, 'rouge1': r1, 'rougeL': rl}
                        for b, r1, rl in zip(training_history.get('bleu', []),
                                            training_history.get('rouge1', []),
                                            training_history.get('rougeL', []))],
                        [{'bleu': val_metrics.get('bleu', 0),
                        'rouge1': val_metrics.get('rouge1', 0),
                        'rougeL': val_metrics.get('rougeL', 0)}],
                        epoch
                    )
                    comparison_path = visualizer.plot_metrics_comparison({
                        'train': train_metrics,
                        'val': val_metrics
                    }, epoch)
                    if wandb_logger:
                        wandb_logger.log_images({
                            'training_curves': curves_path,
                            'metrics_comparison': comparison_path
                        }, step=epoch)
                except Exception as e:
                    print(f"Visualization error: {e}")

            # 체크포인트 & 얼리스탑 (검증 있는 에폭에서만)
            current_val_bleu = val_metrics.get('bleu', 0.0)
            if current_val_bleu > best_val_bleu:
                best_val_bleu = current_val_bleu
                patience_counter = 0

                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': unwrapped_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_bleu': best_val_bleu,
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics,
                        'training_history': training_history
                    }
                    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'best_model_epoch_{epoch}.pt')
                    torch.save(checkpoint, checkpoint_path)
                    accelerator.print(f"New best model saved: BLEU {best_val_bleu:.4f}")

                    # 예측 샘플 저장
                    samples_df = pd.DataFrame({
                        'video_name': val_video_names[:min(5, len(val_video_names))],
                        'prediction': val_predictions[:5],
                        'reference': val_references[:5]
                    })
                    samples_df.to_csv(os.path.join(LOG_DIR, f'samples_epoch_{epoch}.csv'), index=False)
            else:
                patience_counter += 1

            # if patience_counter >= patience:
            #     accelerator.print(f"Early stopping at epoch {epoch+1}")
            #     break

        else:
            # 검증을 하지 않는 에폭에서는 기록만 남김 (val은 NaN으로 패딩해 길이 정렬)
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['val_loss'].append(float('nan'))
            for metric in ['bleu', 'rouge1', 'rouge2', 'rougeL',
                        'mae', 'rmse', 'ndcg', 'f1_macro', 'f1_micro', 'f1_weighted']:
                training_history.setdefault(metric, []).append(float('nan'))

            # 간단 출력 & WandB(train만)
            accelerator.print(f"Train Loss: {train_metrics['loss']:.4f} (no validation this epoch)")
            if accelerator.is_main_process and wandb_logger:
                log_only_train = {f"train/{k}": v for k, v in train_metrics.items() if k != 'loss_distribution'}
                log_only_train['epoch'] = epoch
                wandb_logger.log_metrics(log_only_train, step=epoch * len(train_loader))

        accelerator.wait_for_everyone()
    
    # 훈련 완료 후 최종 대시보드 생성
    if accelerator.is_main_process and visualizer:
        try:
            dashboard_path = visualizer.create_summary_dashboard(val_metrics, training_history)
            accelerator.print(f"Final dashboard saved: {dashboard_path}")
            
            if wandb_logger:
                wandb_logger.log_images({'final_dashboard': dashboard_path})
                
        except Exception as e:
            accelerator.print(f"Dashboard creation error: {e}")

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_bleu': best_val_bleu,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_history': training_history
        }

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        last_ckpt_path = os.path.join(CHECKPOINT_DIR, f'last_model_epoch_{epoch}_{ts}.pt')
        torch.save(checkpoint, last_ckpt_path)
        accelerator.print(f"Final model saved: {last_ckpt_path}")

    # 최종 결과 저장
    if accelerator.is_main_process:
        final_results = {
            'best_val_bleu': best_val_bleu,
            'final_train_metrics': train_metrics,
            'final_val_metrics': val_metrics,
            'training_history': training_history,
            'config': {
                'model': 'T5-VideoMAE-Multimodal',
                'epochs': epoch + 1,
                'batch_size': BATCH_SIZE,
                'learning_rate': LR,
                'num_frames': NUM_FRAMES,
                'frame_size': FRAME_SIZE
            }
        }
        
        results_path = os.path.join(LOG_DIR, f'final_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(results_path, 'w') as f:
            # NumPy arrays를 리스트로 변환
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                return obj
            
            json.dump(convert_numpy(final_results), f, indent=2)
        
        accelerator.print(f"Final results saved: {results_path}")
        
        # 최종 예측 샘플 출력
        # 최종 예측 샘플 출력
        print("\n=== Sample Predictions ===")
        if 'val_predictions' in locals() and len(val_predictions) > 0:
            n_show = min(3, len(val_predictions))
            for i in range(n_show):
                vname = val_video_names[i] if i < len(val_video_names) else "N/A"
                print(f"Video: {vname}")
                print(f"Prediction: {val_predictions[i]}")
                print(f"Reference:  {val_references[i]}")
                print("-" * 50)
    
    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"✅ Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

    # WandB 종료
    if accelerator.is_main_process and wandb_logger:
        wandb_logger.finish()
    
    
    return final_results

# ---------------- Inference Function ----------------
def inference_single_video(model, tokenizer, video_path, context_text="", device=DEVICE):
    """단일 비디오 추론"""
    model.eval()
    
    # 비디오 프레임 로드
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return "Error: Could not read video"
    
    # 프레임 샘플링
    step = max(1, total_frames // NUM_FRAMES)
    frames = []
    
    for i in range(0, min(total_frames, NUM_FRAMES * step), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
            frames.append(frame)
    
    cap.release()
    
    # 패딩
    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1] if frames else np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8))
    
    frames = np.stack(frames[:NUM_FRAMES])
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    frames = frames.unsqueeze(0).to(device)  # 배치 차원 추가
    
    # 텍스트 입력 준비
    if context_text:
        input_text = f"Generate accident description with context: {context_text}"
    else:
        input_text = "Generate accident description from video"
    
    encoding = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    ).to(device)
    
    # 추론
    with torch.no_grad():
        generated_ids = model(
            frames=frames,
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            force_use_video=True
        )
    
    # 디코딩
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

# ---------------- Evaluation Function ----------------
def comprehensive_evaluation(model_path, test_data_path, output_dir="evaluation_results"):
    """포괄적 평가 함수"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 모델 로드
    checkpoint = torch.load(model_path, map_location=DEVICE)
    tokenizer = T5Tokenizer.from_pretrained(T5_NAME)
    
    # 특수 토큰 추가 (훈련과 동일)
    special_tokens = ["<video>", "<object>", "<location>", "<vehicle>"]
    tokenizer.add_tokens(special_tokens)
    
    model = MultimodalAccidentCaptioner(vocab_size=len(tokenizer))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    # 테스트 데이터셋
    test_dataset = AccidentVideoDataset(test_data_path, VIDEO_DIR, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # 평가 실행
    metrics_calc = MetricsCalculator()
    visualizer = MetricsVisualizer(output_dir)
    
    all_results = []
    
    print("Running comprehensive evaluation...")
    for batch in tqdm(test_loader):
        with torch.no_grad():
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            generated_ids = model(
                frames=batch["frames"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                force_use_video=True
            )
            
            pred_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            ref_text = batch["captions"][0]
            
            metrics_calc.add_batch([pred_text], [ref_text])
            
            all_results.append({
                'video_name': batch["video_names"][0],
                'prediction': pred_text,
                'reference': ref_text,
                'pred_length': len(pred_text.split()),
                'ref_length': len(ref_text.split())
            })
    
    # 메트릭 계산
    text_metrics = metrics_calc.compute_text_metrics()
    
    pred_lengths = [r['pred_length'] for r in all_results]
    ref_lengths = [r['ref_length'] for r in all_results]
    reg_metrics = metrics_calc.compute_regression_metrics(pred_lengths, ref_lengths)
    
    # F1 메트릭
    length_bins = [0, 5, 10, 20, float('inf')]
    pred_classes = np.digitize(pred_lengths, length_bins) - 1
    ref_classes = np.digitize(ref_lengths, length_bins) - 1
    f1_metrics = metrics_calc.compute_classification_metrics(pred_classes, ref_classes)
    
    final_metrics = {**text_metrics, **reg_metrics, **f1_metrics}
    
    # 결과 저장
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)
    
    # 메트릭 저장
    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    # 대시보드 생성
    dashboard_path = visualizer.create_summary_dashboard(final_metrics, {'train_loss': [], 'val_loss': []})
    
    print(f"Evaluation completed. Results saved in {output_dir}")
    print(f"Final BLEU: {final_metrics.get('bleu', 0):.4f}")
    print(f"Final ROUGE-L: {final_metrics.get('rougeL', 0):.4f}")
    print(f"Final MAE: {final_metrics.get('mae', 0):.4f}")
    print(f"Final F1-Macro: {final_metrics.get('f1_macro', 0):.4f}")
    
    return final_metrics, all_results

# ---------------- CLI Interface ----------------
def main():
    global BATCH_SIZE, EPOCHS, LR, NUM_FRAMES, FRAME_SIZE

    parser = argparse.ArgumentParser(description='Accident Video Caption Training')
    parser.add_argument('--mode', choices=['train', 'eval', 'infer'], default='train',
                       help='Mode: train, eval, or infer')
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--video_path', type=str, help='Path to video for inference')
    parser.add_argument('--test_data', type=str, help='Path to test data for evaluation')
    parser.add_argument('--context', type=str, default='', help='Context for inference')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    
    # 하이퍼파라미터 오버라이드
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--num_frames', type=int, default=NUM_FRAMES)
    parser.add_argument('--frame_size', type=int, default=FRAME_SIZE)
    
    args = parser.parse_args()
    
    # 글로벌 설정 업데이트
    # global BATCH_SIZE, EPOCHS, LR, NUM_FRAMES, FRAME_SIZE
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    NUM_FRAMES = args.num_frames
    FRAME_SIZE = args.frame_size
    
    if args.mode == 'train':
        if is_rank0():
            print("Starting training...")
        results = train_model()
        if is_rank0():
            print(f"Training completed. Best BLEU: {results.get('best_val_bleu', 0):.4f}")

    elif args.mode == 'eval':
        if not args.model_path or not args.test_data:
            if is_rank0():
                print("Error: --model_path and --test_data required for evaluation")
            return
        if is_rank0():
            print("Starting evaluation...")
        metrics, results = comprehensive_evaluation(args.model_path, args.test_data, args.output_dir)
        if is_rank0():
            print("Evaluation completed.")

    elif args.mode == 'infer':
        if not args.model_path or not args.video_path:
            if is_rank0():
                print("Error: --model_path and --video_path required for inference")
            return
        
        # 모델 로드
        checkpoint = torch.load(args.model_path, map_location=DEVICE)
        tokenizer = T5Tokenizer.from_pretrained(T5_NAME)
        special_tokens = ["<video>", "<object>", "<location>", "<vehicle>"]
        tokenizer.add_tokens(special_tokens)
        
        model = MultimodalAccidentCaptioner(vocab_size=len(tokenizer))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        
        # 추론 실행
        result = inference_single_video(model, tokenizer, args.video_path, args.context)
        print(f"Generated caption: {result}")
        
        # 결과 저장
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'inference_result.txt'), 'w') as f:
            f.write(f"Video: {args.video_path}\n")
            f.write(f"Context: {args.context}\n")
            f.write(f"Generated Caption: {result}\n")

# if __name__ == "__main__":
#     # 필요한 디렉터리 생성
#     for dir_path in [CHECKPOINT_DIR, LOG_DIR, PLOT_DIR]:
#         os.makedirs(dir_path, exist_ok=True)
    
#     # 시드 설정
#     torch.manual_seed(42)
#     np.random.seed(42)
#     random.seed(42)
    
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\nTraining interrupted by user")
#     except Exception as e:
#         print(f"Error during execution: {e}")
#         import traceback
#         traceback.print_exc()

# ---------------- Additional Utility Functions ----------------

def analyze_model_performance(results_path):
    """모델 성능 심층 분석"""
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    training_history = results['training_history']
    final_metrics = results['final_val_metrics']
    
    print("=== Model Performance Analysis ===")
    print(f"Training completed after {results['config']['epochs']} epochs")
    print(f"Best validation BLEU: {results['best_val_bleu']:.4f}")
    
    print("\n--- Text Generation Metrics ---")
    for metric in ['bleu', 'rouge1', 'rouge2', 'rougeL', 'semantic_similarity']:
        if metric in final_metrics:
            print(f"{metric.upper()}: {final_metrics[metric]:.4f}")
    
    print("\n--- Regression Metrics (Length Prediction) ---")
    for metric in ['mae', 'rmse', 'ndcg']:
        if metric in final_metrics:
            print(f"{metric.upper()}: {final_metrics[metric]:.4f}")
    
    print("\n--- Classification Metrics (Length Categories) ---")
    for metric in ['f1_macro', 'f1_micro', 'f1_weighted']:
        if metric in final_metrics:
            print(f"{metric.upper()}: {final_metrics[metric]:.4f}")
    
    # 학습 안정성 분석
    train_losses = training_history['train_loss']
    val_losses = training_history['val_loss']
    
    if len(train_losses) > 5:
        loss_variance = np.var(train_losses[-5:])  # 마지막 5 에포크 분산
        convergence_rate = (train_losses[0] - train_losses[-1]) / len(train_losses)
        
        print(f"\n--- Training Stability ---")
        print(f"Final loss variance: {loss_variance:.6f}")
        print(f"Convergence rate: {convergence_rate:.6f}")
        
        if loss_variance < 0.001:
            print("✓ Training appears stable")
        else:
            print("⚠ Training may be unstable")

def create_inference_demo():
    """추론 데모 생성"""
    print("=== Accident Caption Inference Demo ===")
    
    # 사용자 입력
    model_path = input("Model checkpoint path: ").strip()
    video_path = input("Video path: ").strip()
    context = input("Context (optional): ").strip()
    
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        return
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
    
    try:
        # 모델 로드
        checkpoint = torch.load(model_path, map_location=DEVICE)
        tokenizer = T5Tokenizer.from_pretrained(T5_NAME)
        special_tokens = ["<video>", "<object>", "<location>", "<vehicle>"]
        tokenizer.add_tokens(special_tokens)
        
        model = MultimodalAccidentCaptioner(vocab_size=len(tokenizer))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        
        print("Model loaded successfully!")
        print("Generating caption...")
        
        # 추론 실행
        result = inference_single_video(model, tokenizer, video_path, context)
        
        print(f"\n=== Result ===")
        print(f"Video: {os.path.basename(video_path)}")
        if context:
            print(f"Context: {context}")
        print(f"Generated Caption: {result}")
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"inference_demo_{timestamp}.txt"
        
        with open(result_file, 'w') as f:
            f.write(f"Accident Caption Inference Demo\n")
            f.write(f"Generated at: {datetime.now()}\n\n")
            f.write(f"Video: {video_path}\n")
            f.write(f"Context: {context}\n")
            f.write(f"Generated Caption: {result}\n")
        
        print(f"Result saved to: {result_file}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

# ---------------- Batch Processing ----------------
def batch_inference(model_path, video_dir, output_file="batch_results.csv", context_dict=None):
    """배치 추론"""
    # 모델 로드
    checkpoint = torch.load(model_path, map_location=DEVICE)
    tokenizer = T5Tokenizer.from_pretrained(T5_NAME)
    special_tokens = ["<video>", "<object>", "<location>", "<vehicle>"]
    tokenizer.add_tokens(special_tokens)
    
    model = MultimodalAccidentCaptioner(vocab_size=len(tokenizer))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    
    # 비디오 파일 찾기
    video_files = []
    for ext in ['.mp4', '.mkv', '.mov', '.avi', '.webm']:
        video_files.extend([f for f in os.listdir(video_dir) if f.lower().endswith(ext)])
    
    results = []
    
    print(f"Processing {len(video_files)} videos...")
    for video_file in tqdm(video_files):
        video_path = os.path.join(video_dir, video_file)
        context = context_dict.get(video_file, "") if context_dict else ""
        
        try:
            caption = inference_single_video(model, tokenizer, video_path, context)
            results.append({
                'video_name': video_file,
                'generated_caption': caption,
                'context': context,
                'caption_length': len(caption.split()),
                'processing_time': datetime.now().isoformat()
            })
        except Exception as e:
            print(f"Error processing {video_file}: {e}")
            results.append({
                'video_name': video_file,
                'generated_caption': f"Error: {str(e)}",
                'context': context,
                'caption_length': 0,
                'processing_time': datetime.now().isoformat()
            })
    
    # 결과 저장
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Batch inference completed. Results saved to {output_file}")
    
    return results

# ---------------- Model Analysis Tools ----------------
def analyze_attention_patterns(model_path, sample_video_path, output_dir="attention_analysis"):
    """어텐션 패턴 분석"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 모델 로드
    checkpoint = torch.load(model_path, map_location=DEVICE)
    tokenizer = T5Tokenizer.from_pretrained(T5_NAME)
    special_tokens = ["<video>", "<object>", "<location>", "<vehicle>"]
    tokenizer.add_tokens(special_tokens)
    
    # 어텐션 추출을 위한 훅 설정
    attention_weights = {}
    
    def attention_hook(name):
        def hook(module, input, output):
            if hasattr(output, 'attentions') and output.attentions is not None:
                attention_weights[name] = output.attentions.detach().cpu().numpy()
        return hook
    
    model = MultimodalAccidentCaptioner(vocab_size=len(tokenizer))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    
    # 훅 등록
    model.cross_attention.register_forward_hook(attention_hook('cross_attention'))
    
    # 샘플 추론
    context = "location: highway intersection"
    result = inference_single_video(model, tokenizer, sample_video_path, context)
    
    # 어텐션 시각화
    visualizer = MetricsVisualizer(output_dir)
    
    if 'cross_attention' in attention_weights:
        attn = attention_weights['cross_attention'][0, 0]  # 첫 번째 헤드
        tokens = tokenizer.tokenize(f"Generate accident description with context: {context}")
        
        visualizer.plot_attention_heatmap(attn, tokens, 'cross_attention_analysis')
        print(f"Attention analysis saved in {output_dir}")
    
    return result, attention_weights

# ---------------- Quick Start Functions ----------------
def quick_train():
    """빠른 훈련 시작"""
    print("Quick training setup...")
    
    # 기본 설정으로 훈련 시작
    if not os.path.exists(TRAIN_META):
        print(f"Warning: Training data not found at {TRAIN_META}")
        print("Please prepare your training data first.")
        return
    
    print(f"Training configuration:")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Epochs: {EPOCHS}")
    print(f"- Learning rate: {LR}")
    print(f"- Number of frames: {NUM_FRAMES}")
    print(f"- Frame size: {FRAME_SIZE}")
    
    confirm = input("Start training with these settings? (y/n): ").strip().lower()
    if confirm == 'y':
        return train_model()
    else:
        print("Training cancelled.")
        return None

def quick_eval():
    """빠른 평가"""
    model_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')]
    
    if not model_files:
        print("No model checkpoints found in checkpoint directory.")
        return
    
    print("Available model checkpoints:")
    for i, file in enumerate(model_files):
        print(f"{i+1}. {file}")
    
    try:
        choice = int(input("Select model (number): ")) - 1
        model_path = os.path.join(CHECKPOINT_DIR, model_files[choice])
        
        test_data = input("Test data path (default: val.csv): ").strip()
        if not test_data:
            test_data = VAL_META
            
        return comprehensive_evaluation(model_path, test_data)
        
    except (ValueError, IndexError):
        print("Invalid selection.")
        return None

if __name__ == "__main__":
    # 랭크0에서만 배너 출력
    if is_rank0():
        print("Accident Video Caption Model")
        print("1. Full training with all metrics and visualization")
        print("2. Comprehensive evaluation with detailed analysis")
        print("3. Interactive inference demo")
        print("4. Batch processing")
        print("5. Attention pattern analysis")

    # 필요한 디렉터리 생성
    for dir_path in [CHECKPOINT_DIR, LOG_DIR, PLOT_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    # 시드 설정
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    try:
        main()
    except KeyboardInterrupt:
        if is_rank0():
            print("\nTraining interrupted by user")
    except Exception as e:
        # ❗ 절대 interactive 모드로 전환하지 말고 여기서 종료
        if is_rank0():
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()
        # 멀티프로세스에서 한 rank가 죽으면 전체가 죽도록 re-raise
        raise