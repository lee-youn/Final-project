# # -*- coding: utf-8 -*-
# """
# Video-LLaVA Chatbot (multi-turn)
# - Load Video-LLaVA model (HF Hub or local)
# - Upload a video + hints
# - DV/OV classifier + Fault-BERT(.pth) 사용
# - Fault-BERT encoder의 잠재벡터로 soft token을 만들어 Video-LLaVA 텍스트 임베딩 시퀀스에 삽입 (inputs_embeds)
# """

# import os, glob
# from typing import List, Optional, Tuple
# import numpy as np
# from PIL import Image
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import gradio as gr
# import cv2

# from torchvision.models.video import r3d_18, R3D_18_Weights
# from transformers import (
#     AutoModel, AutoTokenizer,
#     TimesformerModel, VideoMAEModel,
#     VideoLlavaProcessor, VideoLlavaForConditionalGeneration
# )

# # -------------------------
# # 검색 경로 / 확장자
# # -------------------------
# SEARCH_ROOTS = [
#     "/mnt/data/videos",
#     "/app/data/raw/videos/training_reencoded",
#     "/app/data/raw/videos/validation_reencoded",
#     "/data"
# ]
# VIDEO_EXTS = [".mp4", ".mkv", ".mov", ".avi", ".webm"]

# # =========================================================
# # Fault-BERT (.pth) - 네가 학습한 모델로부터 임베딩을 뽑는다
# # =========================================================
# class TextToFaultRatio(nn.Module):
#     def __init__(self, model_name="bert-base-uncased", hidden_dim=768):
#         super().__init__()
#         self.encoder = AutoModel.from_pretrained(model_name)
#         self.regressor = nn.Sequential(
#             nn.Linear(hidden_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 2)
#         )

#     def forward(self, input_ids=None, attention_mask=None, **kwargs):
#         out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         cls = out.last_hidden_state[:, 0]
#         return self.regressor(cls)

# @torch.no_grad()
# def load_fault_model(path: str, model_name="bert-base-uncased", device="cuda"):
#     """네가 학습한 .pth를 로드하여 encoder까지 가중치가 반영된 Fault-BERT를 리턴"""
#     obj = torch.load(path, map_location="cpu")
#     model = TextToFaultRatio(model_name=model_name)
#     if isinstance(obj, dict) and "state_dict" in obj:
#         sd = obj["state_dict"]
#         # 필요시 prefix 정리
#         new_sd = { (k.replace("module.", "")): v for k, v in sd.items() }
#         model.load_state_dict(new_sd, strict=False)
#     elif isinstance(obj, dict):
#         new_sd = { (k.replace("module.", "")): v for k, v in obj.items() }
#         model.load_state_dict(new_sd, strict=False)
#     elif hasattr(obj, "state_dict"):
#         model = obj
#     model.to(device).eval()
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     return model, tokenizer

# @torch.no_grad()
# def predict_fault_ratio(model, tokenizer, text: str, device="cuda", max_length=256):
#     inputs = tokenizer(text, return_tensors="pt",
#                        padding="max_length", truncation=True, max_length=max_length).to(device)
#     pred = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
#     pred = pred.squeeze(0).float().cpu().numpy()  # [2]
#     return pred  # (dashcam, other) 임의 스케일

# def project_pair_to_basis(v, total=10.0):
#     v = np.maximum(np.asarray(v, dtype=float), 0.0)
#     s = float(v.sum())
#     if s <= 0:
#         return np.array([total/2.0, total/2.0], dtype=float)
#     return v * (total / s)


# # -------------------------
# # Label 텍스트
# # -------------------------
# real_categories_ids_2nd = {
#   1 : "Changing Lanes Within Intersection",
#   2 : "Lane Change Path Change",
#   3 : "Lanes Wide Enough For Two Vehicles Side By Side",
#   4 : "Main Road And Side Road",
#   5 : "Other Vehicle Enters From The Side",
#   6 : "Roads Of Equal Width",
#   7 : "Start After Stopping Accident",
#   8 : "Turning Angle Less Than 90 Degrees",
#   9 : "Two Vehicles Turning Right Simultaneously"
# }
# dashcam_vehicle_info = {
#     1 : "Facing Each Other Going Straight",
#     2 : "Following Vehicle Going Straight After Leaving Safety Zone",
#     3 : "Following Vehicle Going Straight Right Side Of Lane",
#     4 : "Going Straight From Left Road",
#     5 : "Going Straight From Main Road",
#     6 : "Going Straight From Main Road Entered Earlier",
#     7 : "Going Straight From Main Road Entered Later",
#     8 : "Going Straight From Right Road",
#     9 : "Going Straight From Side Road Left",
#     10 : "Going Straight From Side Road Right",
#     11 : "Going Straight From The Right",
#     12 : "Going Straight From The Right Entered Earlier",
#     13 : "Going Straight From The Right Entered Later",
#     14 : "Going Straight Lane Change Inside Intersection",
#     15 : "Green Light Going Straight",
#     16 : "Green Light Going Straight Collided With Red Light Vehicle",
#     17 : "Left Turn From Right Road",
#     18 : "Left Turn From Side Road",
#     19 : "Right Turn",
#     20 : "Right Turn Entered Earlier",
#     21 : "Right Turn Entered Later",
#     22 : "Right Turn From Main Road",
#     23 : "Right Turn From Main Road Entered Later",
#     24 : "Right Turn From Side Road",
#     25 : "Right Turn From Side Road Entered Earlier",
#     26 : "Right Turn From Side Road Entered Later",
#     27 : "Right Turn Right Lane",
#     28 : "Simultaneous Lane Change",
#     29 : "Yellow Light Going Straight",
#     30 : "Yellow Light Left Turn Collided With Red Light Vehicle"
# }
# other_vehicle_info = {
#   1 : "Departing After Stop",
#   2 : "Facing Each Other Going Straight",
#   3 : "Following Vehicle Going Straight",
#   4 : "Following Vehicle Going Straight Left Side Of Lane",
#   5 : "Following Vehicle Going Straight Right Side Of Lane",
#   6 : "Going Straight From Left Road",
#   7 : "Going Straight From Main Road",
#   8 : "Going Straight From Main Road Entered Earlier",
#   9 : "Going Straight From Main Road Entered Later",
#   10 : "Going Straight From Right Road",
#   11 : "Going Straight From Side Road Left",
#   12 : "Going Straight From Side Road Right",
#   13 : "Going Straight From The Right",
#   14 : "Going Straight From The Right Entered Earlier",
#   15 : "Going Straight From The Right Entered Later",
#   16 : "Green Left Turn Signal Left Turn Collided With Red Light Vehicle",
#   17 : "Green Light Going Straight",
#   18 : "Left Turn From Right Road",
#   19 : "Left Turn From Side Road",
#   20 : "Right Turn",
#   21 : "Right Turn Entered Earlier",
#   22 : "Right Turn Entered Later",
#   23 : "Right Turn From Main Road",
#   24 : "Right Turn From Main Road Entered Earlier",
#   25 : "Right Turn From Main Road Entered Later",
#   26 : "Right Turn From Side Road",
#   27 : "Right Turn From Side Road Entered Later",
#   28 : "Right Turn Right Lane"
# }
# def _dict_to_list_by_id(d: dict): return [d[k] for k in sorted(d.keys())]
# LABELS = {
#     "dv": _dict_to_list_by_id(dashcam_vehicle_info),
#     "pl": _dict_to_list_by_id(real_categories_ids_2nd),
#     "ov": _dict_to_list_by_id(other_vehicle_info),
# }

# def render_template_from_labels(dv_text: str, ov_text: str) -> str:
#     return (f"At an unsignalized intersection, the Dashcam Vehicle was {dv_text}, "
#             f"while the Other Vehicle was {ov_text}.")

# # =========================================================
# # 비디오 분류기 (DV/OV 라벨 예측)
# # =========================================================
# class TripleHeadVideoClassifier(nn.Module):
#     def __init__(self, n_dv, n_ov, backbone="r3d18", pretrained=False):
#         super().__init__()
#         self.backbone_name = backbone

#         if backbone == "r3d18":
#             self.backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1 if pretrained else None)
#             feat = self.backbone.fc.in_features
#             self.backbone.fc = nn.Identity()
#         elif backbone == "timesformer":
#             self.backbone = TimesformerModel.from_pretrained(
#                 "facebook/timesformer-base-finetuned-k400", use_safetensors=True
#             )
#             feat = self.backbone.config.hidden_size
#         elif backbone == "videomae":
#             self.backbone = VideoMAEModel.from_pretrained(
#                 "MCG-NJU/videomae-base-finetuned-kinetics", use_safetensors=True
#             )
#             feat = self.backbone.config.hidden_size
#         else:
#             raise ValueError(backbone)

#         self.dv = nn.Linear(feat, n_dv)
#         self.ov = nn.Linear(feat, n_ov)

#     def forward(self, x):
#         if self.backbone_name == "r3d18":
#             z = self.backbone(x)              # (B, feat)
#         else:
#             x = x.permute(0,2,1,3,4)          # (B,T,C,H,W)
#             out = self.backbone(x)
#             z = out.last_hidden_state.mean(1) # (B, feat)
#         return self.dv(z), self.ov(z)

# @torch.no_grad()
# def load_video_tensor_for_clf(path, num_frames=16, size=224, device="cuda"):
#     cap = cv2.VideoCapture(path)
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     if total <= 0:
#         cap.release()
#         raise RuntimeError(f"no frames in {path}")
#     idxs = np.linspace(0, total-1, num_frames).astype(int)

#     frames = []
#     for i in idxs:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#         ok, fr = cap.read()
#         if not ok: continue
#         fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
#         fr = cv2.resize(fr, (size, size))
#         fr = torch.from_numpy(fr).permute(2,0,1).float()/255.0  # (C,H,W)
#         frames.append(fr)
#     cap.release()
#     if not frames:
#         raise RuntimeError(f"no readable frames in {path}")

#     vid = torch.stack(frames, 0)                        # (T,C,H,W)
#     mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
#     std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
#     vid = (vid - mean) / std
#     vid = vid.permute(1,0,2,3).unsqueeze(0).to(device)  # (1,C,T,H,W)
#     return vid

# def load_classifier(ckpt_path: str, backbone="r3d18", device="cuda", pretrained=False):
#     model = TripleHeadVideoClassifier(
#         n_dv=len(LABELS["dv"]), n_ov=len(LABELS["ov"]),
#         backbone=backbone, pretrained=pretrained
#     )
#     sd = torch.load(ckpt_path, map_location="cpu")
#     if isinstance(sd, dict) and "model" in sd:
#         sd = sd["model"]
#     new_sd = { (k[len("module."):] if k.startswith("module.") else k): v for k,v in sd.items() }
#     model.load_state_dict(new_sd, strict=False)
#     model.to(device).eval()
#     return model

# @torch.no_grad()
# def predict_classifier(video_path, clf_model, device="cuda", topk=3):
#     x = load_video_tensor_for_clf(video_path, num_frames=16, size=224, device=device)
#     ld, lv = clf_model(x)   # dv, ov

#     def _topk_text(logits, label_list, k):
#         probs = logits.softmax(dim=-1)[0]
#         val, idx = probs.topk(k=k, dim=-1)
#         pairs = [f"{label_list[i]}:{val[j].item():.2f}" for j,i in enumerate(idx.tolist())]
#         return label_list[idx[0].item()], ", ".join(pairs)

#     dv1, dv_topk = _topk_text(ld, LABELS["dv"], topk)
#     ov1, ov_topk = _topk_text(lv, LABELS["ov"], topk)

#     dashcam_info   = dv1
#     other_info     = ov1
#     classifier_top = f"DV[{dv_topk}] | OV[{ov_topk}]"
#     return dashcam_info, other_info, classifier_top

# # =========================================================
# # 파일/비디오 유틸
# # =========================================================
# def list_candidates(limit=50):
#     items = []
#     for root in SEARCH_ROOTS:
#         if not os.path.isdir(root):
#             continue
#         for f in os.listdir(root):
#             fname = f.lower()
#             for ext in VIDEO_EXTS:
#                 if fname.endswith(ext):
#                     items.append(os.path.join(root, f))
#     items = [p for p in items if os.path.isfile(p)]
#     items.sort(key=lambda p: os.path.getmtime(p), reverse=True)
#     return items[:limit]

# def find_video_path_by_name(name: str) -> Optional[str]:
#     name = name.strip().strip('"').strip("'")
#     if not name:
#         return None

#     has_ext = os.path.splitext(name)[1].lower() in VIDEO_EXTS

#     for root in SEARCH_ROOTS:
#         if not os.path.isdir(root):
#             continue

#         if any(ch in name for ch in "*?[]"):
#             patterns = [os.path.join(root, name)]
#         else:
#             patterns = [os.path.join(root, name)] if has_ext else [os.path.join(root, name + ext) for ext in VIDEO_EXTS]

#         for pat in patterns:
#             hits = sorted(glob.glob(pat))
#             if hits:
#                 hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
#                 return hits[0]
#     return None

# def _resolve_video_path(video, video_name: Optional[str]) -> Optional[str]:
#     path = None
#     if isinstance(video, str) and os.path.exists(video):
#         path = video
#     elif isinstance(video, dict) and "name" in video and os.path.exists(video["name"]):
#         path = video["name"]
#     if not path and video_name:
#         cand = find_video_path_by_name(video_name)
#         if cand and os.path.exists(cand):
#             path = cand
#     if not path and video_name and os.path.exists(video_name):
#         path = video_name
#     return path

# # ---------------- Frame Sampler ----------------
# def sample_frames(video_path: str, num_frames: int = 8, size: int = 224) -> List[Image.Image]:
#     frames = []
#     try:
#         from decord import VideoReader, cpu
#         vr = VideoReader(video_path, ctx=cpu(0))
#         total = len(vr)
#         if total <= 0:
#             raise RuntimeError("Empty video (decord).")
#         idxs = (np.linspace(0, total - 1, num_frames)).astype(int).tolist()
#         for i in idxs:
#             arr = vr[i].asnumpy()[:, :, ::-1]  # BGR->RGB
#             frames.append(Image.fromarray(arr).resize((size, size)))
#     except Exception:
#         cap = cv2.VideoCapture(video_path)
#         total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         if total <= 0:
#             cap.release()
#             raise RuntimeError("Empty/unreadable video (OpenCV).")
#         idxs = (np.linspace(0, total - 1, num_frames)).astype(int).tolist()
#         for i in idxs:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#             ok, fr = cap.read()
#             if not ok:
#                 if frames:
#                     frames.append(frames[-1])
#                 else:
#                     frames.append(Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8)))
#                 continue
#             fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
#             frames.append(Image.fromarray(fr).resize((size, size)))
#         cap.release()
#     while len(frames) < num_frames:
#         frames.append(frames[-1])
#     return frames[:num_frames]

# # ---------------- Prompt Builder ----------------
# def build_prompt(
#     user_message: str,
#     dashcam_info: Optional[str],
#     other_info: Optional[str],
#     classifier_topk: Optional[str],
#     style: str = "brief",
#     history: Optional[list] = None,
#     history_max_turns: int = 6,
#     fault_hint: Optional[str] = None,
# ) -> str:
#     system_header = (
#         "You are an expert at analyzing dashcam accident videos. "
#         "Base your description strictly on what is visible. "
#         "If any hint conflicts with visible evidence in the frames, ignore the hint and prioritize the video evidence. State the conflict briefly."
#         "Do not invent traffic lights, lane markings, numbers, timestamps, or unseen objects.\n"
#     )
#     if style == "short":
#         length_rule = "Write exactly one concise sentence (under ~30 words)."
#     elif style == "detailed":
#         length_rule = "Write 4–6 factual sentences (~80–120 words)."
#     else:
#         style = "brief"
#         length_rule = "Write 2–3 factual sentences (under ~90 words)."

#     hint_lines = []
#     if dashcam_info:
#         hint_lines.append(f"- Ego Vehicle: {dashcam_info}")
#         print(f"[Hint] Ego Vehicle: {dashcam_info}")
#     if other_info:
#         hint_lines.append(f"- Other Vehicle: {other_info}")
#         print(f"[Hint] Other Vehicle: {other_info}")
#     if classifier_topk:
#         hint_lines.append(f"- Classifier outputs (top-k): {classifier_topk}")
#         print(f"[Hint] Classifier top-k: {classifier_topk}")
#     if fault_hint:
#         hint_lines.append(f"- Fault analysis[Ego Vehicle:Other Vehicle]: {fault_hint}")
#         print(f"[Hint] Fault analysis: {fault_hint}")
#     hint_block = ("Always include the following hints exactly once in natural English:\n" + "\n".join(hint_lines) + "\n") if hint_lines else ""

#     hist_txt = ""
#     if history:
#         turns = history[-history_max_turns:]
#         for role, msg in turns:
#             if not msg: continue
#             hist_txt += f"{role.UPPER()}:\n{msg}\n\n" if hasattr(role, "UPPER") else f"{role}:\n{msg}\n\n"

#     # user_header = (
#     #     f"{hist_txt}"
#     #     "USER:\n"
#     #     "Describe the accident scene focusing on visible motion, entry order, and relative positions.\n"
#     #     f"{length_rule}\n"
#     #     f"{hint_block}"
#     #     "You MUST apply the provided hints in your answer (reflect them at least once).\n"
#     #     "Then add: the predicted fault ratio (basis 10), identify the likely victim (lower fault share), "
#     #     "and one-sentence cause reasoning grounded in visible evidence and the classification hints.\n"
#     #     "Avoid: the words 'traffic light', numbers unrelated to fault ratio, dates, or invented objects.\n"
#     #     "Use 'Ego Vehicle' and 'Other Vehicle' exactly once each.\n"
#     # )
#     user_header = (
#         f"{hist_txt}"
#         "USER:\n"
#         "Describe the accident scene focusing on visible motion, entry order, and relative positions.\n"
#         f"{length_rule}\n"
#         f"{hint_block}"
#         "Apply the provided hints exactly once if they agree with visible evidence.\n"
#         "Then add: the predicted fault ratio (basis 10), identify the likely victim (lower share), "
#         "and one-sentence cause reasoning grounded in visible evidence and the hints.\n"
#         "Avoid: the words 'traffic light', numbers unrelated to the fault ratio, dates, or invented objects. "
#         "Do not mention any element that is not clearly visible.\n"
#         "if any hint conflicts with the frames, briefly note the conflict and prioritize the video evidence.\n"
#     )
#     if user_message and user_message.strip():
#         user_header += f"\nAdditional instruction from user: {user_message.strip()}\n"

#     placeholder = "<video>"
#     return f"{placeholder}\n" + system_header + user_header + "ASSISTANT:"

# # =========================================================
# # Video-LLaVA 엔진 + soft token 삽입 경로
# # =========================================================
# class VideoLLaVAChatEngine:
#     def __init__(self):
#         self.model_id = None
#         self.processor = None
#         self.model = None
#         self.device = self._pick_device()

#         # BERT-soft 토큰 투영기 (Fault-BERT hidden -> LLaVA hidden)
#         self._hint_proj: Optional[nn.Module] = None
#         self._hint_proj_dim: Optional[int] = None

#     def _pick_device(self):
#         if torch.cuda.is_available():
#             return "cuda"
#         if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#             return "mps"
#         return "cpu"

#     def load_model(self, model_id: str):
#         if self.model_id == model_id and self.model is not None:
#             return f"Model already loaded: {model_id}"
#         self.processor = VideoLlavaProcessor.from_pretrained(model_id)
#         self.model = VideoLlavaForConditionalGeneration.from_pretrained(
#             model_id,
#             torch_dtype=torch.float16,
#             low_cpu_mem_usage=True,
#             device_map=None,
#         ).to(self.device)
#         self.model.config.use_cache = False
#         self.model_id = model_id
#         return f"Loaded model: {model_id}"

#     def _ensure_hint_proj(self, llm_hidden_size: int, bert_hidden_size: int):
#         if (self._hint_proj is None) or (self._hint_proj_dim != llm_hidden_size):
#             self._hint_proj = nn.Sequential(
#                 nn.Linear(bert_hidden_size, llm_hidden_size),
#                 nn.Tanh(),
#                 nn.LayerNorm(llm_hidden_size)
#             ).to(self.device).eval()
#             self._hint_proj_dim = llm_hidden_size
#         return self._hint_proj

#     @torch.no_grad()
#     def make_soft_tokens_from_trained_fault(
#         self,
#         fault_m: TextToFaultRatio,
#         fault_tok: AutoTokenizer,
#         text: str,
#         take: str = "cls",
#         max_tokens: int = 4
#     ) -> torch.Tensor:
#         """네가 학습한 Fault-BERT의 encoder에서 히든을 뽑아 LLaVA hidden으로 투영 -> soft tokens 반환 [1,M,H]"""
#         tok_inputs = fault_tok(text, return_tensors="pt", truncation=True, max_length=256).to(self.device)
#         # encoder 히든: [1, Lb, Hb]
#         hs = fault_m.encoder(**tok_inputs).last_hidden_state         # [1, Lb, Hb] (fp32일 가능성 높음)
#         Hb = hs.size(-1)
#         H  = self.model.get_input_embeddings().embedding_dim
#         proj = self._ensure_hint_proj(H, Hb)

#         # proj는 보통 fp32 파라미터 → 입력도 proj와 같은 dtype으로
#         hs = hs.to(next(proj.parameters()).dtype)

#         if take == "cls":
#             z = hs[:, 0, :]
#             s = proj(z).unsqueeze(1)                                 # [1,1,H]
#         elif take == "mean":
#             z = hs.mean(dim=1)
#             s = proj(z).unsqueeze(1)
#         else:
#             K = min(hs.size(1), max_tokens)
#             z = hs[:, :K, :]
#             s = proj(z)

#         max_norm = 2.0
#         s_norm = s.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # [1,1,1]
#         s = s * (max_norm / s_norm).clamp_max(1.0)

#         # ★ 모델 dtype으로 최종 캐스팅 (half/bfloat16 등)
#         s = s.to(self.model.dtype)
#         return s

#     @torch.no_grad()
#     # (기존 함수 대체)
#     def generate_with_soft_tokens_from_fault(
#         self,
#         frames: List[Image.Image],
#         chat_prompt: str,
#         soft_tokens: Optional[torch.Tensor],   # <-- Optional 로 변경
#         insert_pos: int = 1,
#         temperature: float = 0.2,
#         do_sample: bool = False,
#         max_new_tokens: int = 256,
#     ) -> str:
#         model = self.model
#         processor = self.processor
#         device = self.device
#         tok = processor.tokenizer

#         proc = processor(videos=[frames], text=[chat_prompt],
#                         padding="longest", truncation=False, return_tensors="pt")

#         # 비전 텐서
#         if "pixel_values_videos" in proc:
#             vision = {"pixel_values_videos": proc["pixel_values_videos"].to(device, dtype=model.dtype)}
#         elif "pixel_values" in proc:
#             vision = {"pixel_values_videos": proc["pixel_values"].to(device, dtype=model.dtype)}

#         input_ids = proc["input_ids"].to(device)
#         attn      = proc["attention_mask"].to(device)

#         # 금지어
#         bad_words = ["traffic light","lights","signal","signals","signalized","timestamp","AM","PM"] + [str(d) for d in range(10)]
#         bad_ids = [tok(w, add_special_tokens=False).input_ids for w in bad_words if tok(w, add_special_tokens=False).input_ids]

#         # === (A) 소프트 토큰 미사용 경로 ===
#         if soft_tokens is None:
#             gen_ids = model.generate(
#                 **vision,
#                 input_ids=input_ids,
#                 attention_mask=attn,
#                 max_new_tokens=max_new_tokens,
#                 temperature=temperature,
#                 do_sample=do_sample,
#                 num_beams=1,
#                 no_repeat_ngram_size=3,
#                 repetition_penalty=1.05,
#                 bad_words_ids=bad_ids,
#                 eos_token_id=tok.eos_token_id,
#                 pad_token_id=tok.pad_token_id,
#             )
#             return tok.decode(gen_ids[0], skip_special_tokens=True).strip()

#         # === (B) 소프트 토큰 사용 경로 (기존 그대로) ===
#         text_embeds = model.get_input_embeddings()(input_ids)
#         text_embeds = text_embeds.to(device, dtype=model.dtype)

#         S = soft_tokens.to(device, dtype=model.dtype)
#         L = text_embeds.size(1)
#         pos = max(1, min(4, L - 2))

#         inputs_embeds = torch.cat([text_embeds[:, :pos, :], S, text_embeds[:, pos:, :]], dim=1)
#         inputs_embeds = inputs_embeds.to(device, dtype=model.dtype)

#         new_mask = torch.cat([
#             attn[:, :pos],
#             torch.ones(attn.size(0), S.size(1), device=device, dtype=attn.dtype),
#             attn[:, pos:]
#         ], dim=1)

#         gen_ids = model.generate(
#             **vision,
#             inputs_embeds=inputs_embeds,
#             attention_mask=new_mask,
#             max_new_tokens=max_new_tokens,
#             temperature=temperature,
#             do_sample=do_sample,
#             num_beams=1,
#             no_repeat_ngram_size=3,
#             repetition_penalty=1.05,
#             bad_words_ids=bad_ids,
#             eos_token_id=tok.eos_token_id,
#             pad_token_id=tok.pad_token_id,
#         )
#         return tok.decode(gen_ids[0], skip_special_tokens=True).strip()


# ENGINE = VideoLLaVAChatEngine()

# # =========================================================
# # Gradio UI
# # =========================================================
# def ui_load_model(model_id):
#     try:
#         msg = ENGINE.load_model(model_id)
#     except Exception as e:
#         msg = f"Load error: {e}"
#     return msg

# _CLF_CACHE = {}
# _FAULT_CACHE = {}

# def _ensure_classifier(ckpt_path, backbone, pretrained, device):
#     key = (ckpt_path, backbone, bool(pretrained), device)
#     if key not in _CLF_CACHE:
#         mdl = load_classifier(ckpt_path, backbone=backbone, device=device, pretrained=pretrained)
#         _CLF_CACHE[key] = mdl
#     return _CLF_CACHE[key]

# def _ensure_fault(ckpt_path, model_name, device):
#     key = (ckpt_path, model_name, device)
#     if key not in _FAULT_CACHE:
#         mdl, tok = load_fault_model(ckpt_path, model_name=model_name, device=device)
#         _FAULT_CACHE[key] = (mdl, tok)
#     return _FAULT_CACHE[key]

# def ui_generate(
#         history, user_msg, video, video_name, model_id,
#         clf_ckpt, clf_backbone, clf_pretrained, clf_topk,
#         fault_ckpt, fault_model, fault_basis,
#         style, num_frames, frame_size, temperature, sampling, use_soft
#     ):
#     history = history or []

#     # 0) 비디오 찾기
#     video_path = _resolve_video_path(video, video_name)
#     print(f"[Chat] video_path={video_path}, video_name={video_name}")
#     if not video_path:
#         history = history + [{"role":"assistant","content": f"❌ Video not found. name='{video_name}'\n검색 루트: {SEARCH_ROOTS}"}]
#         return history
#     if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
#         history = history + [{"role":"assistant","content": f"❌ Video not readable: {video_path}"}]
#         return history

#     device = ENGINE.device

#     # 1) DV/OV 분류 → top-k 텍스트
#     try:
#         clf = _ensure_classifier(clf_ckpt, clf_backbone, clf_pretrained, device)
#         dv_hint, ov_hint, topk_text = predict_classifier(
#             video_path, clf, device=device, topk=int(clf_topk)
#         )
#     except Exception as e:
#         dv_hint, ov_hint = "UNKNOWN_DV", "UNKNOWN_OV"
#         topk_text = f"[classifier error: {e}]"
#     print(f"[Hint] DV={dv_hint} | OV={ov_hint} | TOPK={topk_text}")

#     # 2) 템플릿 문장
#     sentence = render_template_from_labels(dv_hint, ov_hint)

#     # 3) Fault-BERT 추론 (숫자는 금지어로 막지만, 벡터는 soft token으로 전달)
#     try:
#         fr_m, fr_tok = _ensure_fault(fault_ckpt, fault_model, device)
#         y = predict_fault_ratio(fr_m, fr_tok, sentence, device=device)  # [2]
#         pred = project_pair_to_basis(y, total=float(fault_basis))       # 합 = basis
#         dc_f, ov_f = float(pred[0]), float(pred[1])
#         if abs(dc_f - ov_f) < 1e-6:
#             victim = "Undetermined"
#         else:
#             victim = "Ego Vehicle" if dc_f < ov_f else "Other Vehicle"
#         fault_hint = f"Predicted fault (basis {int(fault_basis)}): ego={dc_f:.1f}, Other={ov_f:.1f}; Likely victim: {victim}; Template: {sentence}"
#     except Exception as e:
#         fault_hint = f"[fault error: {e}]; Template: {sentence}"
#         dc_f, ov_f, victim = 5.0, 5.0, "Undetermined"

#     # 4) Video-LLaVA 로드
#     load_msg = ENGINE.load_model(model_id)
#     print(load_msg)

#     # 5) 프롬프트 + 프레임 준비
#     frames = sample_frames(video_path, num_frames=int(num_frames), size=int(frame_size))
#     prompt_text = build_prompt(
#         user_message=user_msg or "",
#         dashcam_info=dv_hint,
#         other_info=ov_hint,
#         classifier_topk=topk_text,
#         style=style,
#         history=history,
#         fault_hint=fault_hint
#     )

#     # 6) 네가 학습한 Fault-BERT encoder 임베딩으로 soft token 생성
#     #    DV/OV/fault 정보를 한 줄로 모아 의미를 압축 (네가 학습한 표현공간을 활용)
#     bert_hint_text = f"Ego={dv_hint}; Other={ov_hint}; {fault_hint}"
#     try:
#         if use_soft:
#             fr_m, fr_tok = _ensure_fault(fault_ckpt, fault_model, device)
#             soft = ENGINE.make_soft_tokens_from_trained_fault(
#                 fault_m=fr_m, fault_tok=fr_tok, text=bert_hint_text, take="cls", max_tokens=4
#             )  # [1,1,H]
#         else:
#             soft = None  # <-- 소프트 토큰 비활성화

#         text = ENGINE.generate_with_soft_tokens_from_fault(
#             frames=frames,
#             chat_prompt=prompt_text,
#             soft_tokens=soft,                 # <-- None 이면 일반 경로로
#             insert_pos=1,
#             temperature=float(temperature),
#             do_sample=bool(sampling),
#             max_new_tokens=180,
#         )
#     except Exception as e:
#         text = f"Generation error (soft-token path): {e}"

#     # 7) 채팅창 출력
#     aux = f"⚖️ Fault(basis {int(fault_basis)}): DC {dc_f:.1f} / OV {ov_f:.1f} | Victim: {victim}\n📝 {sentence}"
#     if user_msg:
#         history = history + [{"role": "user", "content": user_msg}]
#     history = history + [{"role": "assistant", "content": text + "\n\n" + aux}]
#     return history

# # =========================================================
# # Gradio App
# # =========================================================
# with gr.Blocks(title="Video-LLaVA Chatbot") as demo:
#     gr.Markdown("## 🎥 Video-LLaVA Chatbot — Multi-turn video-first accident analysis")
#     with gr.Row():
#         with gr.Column(scale=1):
#             model_id = gr.Textbox(label="HF Model ID / local path", value="LanguageBind/Video-LLaVA-7B-hf")
#             load_btn = gr.Button("Load / Reload Model")
#             use_soft = gr.Checkbox(value=True, label="Use Fault-BERT Soft Token")  # <-- 추가
#             load_status = gr.Textbox(label="Load status", value="", interactive=False)

#             video_name = gr.Textbox(label="Video Name (search in folders)",
#                                     placeholder="예) crash_000123.mp4 또는 2024-09-*.mp4")
#             scan_btn = gr.Button("Scan Videos")
#             video_picker = gr.Dropdown(label="Pick found video (fills Video Name)", choices=[], value=None)

#             video = gr.Video(label="Video (mp4/mov/webm)", interactive=True)

#             clf_ckpt     = gr.Textbox(label="Classifier CKPT path", value="/app/checkpoints/best_exact_ep13_r3d18.pth")
#             clf_backbone = gr.Dropdown(label="Classifier backbone", choices=["r3d18","timesformer","videomae"], value="r3d18")
#             clf_pretrained = gr.Checkbox(label="Use pretrained backbone", value=False)
#             clf_topk     = gr.Slider(1,5,value=3,step=1,label="Classifier Top-K")

#             fault_ckpt   = gr.Textbox(label="Fault-BERT CKPT path", value="/app/text-train/fault_ratio_bert.pt")
#             fault_model  = gr.Textbox(label="Fault-BERT model_name", value="bert-base-uncased")
#             fault_basis  = gr.Slider(2, 20, value=10, step=1, label="Fault ratio basis")

#             style = gr.Dropdown(label="Output style", choices=["short", "brief", "detailed"], value="brief")
#             num_frames = gr.Slider(4, 16, value=8, step=1, label="Frames")
#             frame_size = gr.Slider(160, 336, value=224, step=16, label="Frame size")
#             temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")
#             sampling = gr.Checkbox(value=False, label="Enable sampling")

#         with gr.Column(scale=2):
#             chatbot = gr.Chatbot(height=520, show_copy_button=True, type="messages")
#             state = gr.State([])  # 대화 히스토리
#             user_msg = gr.Textbox(label="Your message", placeholder="추가 지시 입력")
#             send_btn = gr.Button("Send ▶️")

#     load_btn.click(ui_load_model, inputs=[model_id], outputs=[load_status])
#     scan_btn.click(lambda: list_candidates(), inputs=[], outputs=[video_picker])
#     video_picker.change(lambda p: os.path.basename(p), inputs=[video_picker], outputs=[video_name])
#     send_btn.click(
#         ui_generate,
#         inputs=[
#             state, user_msg, video, video_name, model_id,
#             clf_ckpt, clf_backbone, clf_pretrained, clf_topk,
#             fault_ckpt, fault_model, fault_basis,
#             style, num_frames, frame_size, temperature, sampling, use_soft
#         ],
#         outputs=[chatbot],
#     ).then(lambda h: h, inputs=[chatbot], outputs=[state])

# if __name__ == "__main__":
#     try:
#         torch.backends.cudnn.enabled = True
#         torch.backends.cudnn.benchmark = False
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True
#     except Exception:
#         pass

#     server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
#     server_port = int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", 7860)))
#     share = os.environ.get("GRADIO_SHARE", "1") == "1"

#     demo.queue()
#     demo.launch(
#         server_name=server_name,
#         server_port=server_port,
#         share=share,
#         show_error=True,
#         inbrowser=False,
#         prevent_thread_lock=False,
#     )
#     print(f"\n[Gradio] listening on http://{server_name}:{server_port}  (share={share})")


# -*- coding: utf-8 -*-
"""
Video-LLaVA Chatbot (multi-turn)
- Load Video-LLaVA model (HF Hub or local)
- Upload a video + hints
- DV/OV classifier + Fault-BERT(.pth) 사용
- (옵션) Fault-BERT encoder의 잠재벡터로 soft token을 만들어 Video-LLaVA 텍스트 임베딩 시퀀스에 삽입 (inputs_embeds)
- (옵션) 해석가능성 모드: 비디오만으로 초안 생성 → 소프트 토큰으로 짧게 수정(보정)
"""

import os, glob
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import cv2

from torchvision.models.video import r3d_18, R3D_18_Weights
from transformers import (
    AutoModel, AutoTokenizer,
    TimesformerModel, VideoMAEModel,
    VideoLlavaProcessor, VideoLlavaForConditionalGeneration
)

# -------------------------
# 검색 경로 / 확장자
# -------------------------
SEARCH_ROOTS = [
    "/mnt/data/videos",
    "/app/data/raw/videos/training_reencoded",
    "/app/data/raw/videos/validation_reencoded",
    "/data"
]
VIDEO_EXTS = [".mp4", ".mkv", ".mov", ".avi", ".webm"]

import re

def _clean_generated(text: str) -> str:
    if not text:
        return ""
    # Drop anything that looks like an instruction block
    text = re.sub(r"(?is)^you are a .*?elements\.\s*", "", text).strip()
    text = re.sub(r"(?is)^describe .*?positions\.\s*", "", text).strip()
    text = re.sub(r"(?is)\b(USER|ASSISTANT):.*", "", text).strip()
    # Keep only first 2–3 sentences max
    sents = re.split(r"(?<=[.!?])\s+", text)
    text = " ".join(sents[:3]).strip()
    # Ensure terminal punctuation
    if text and text[-1] not in ".!?":
        text += "."
    return text

def _enforce_two_line_revise(text: str) -> str:
    # Keep only the first “Description:” and “Evidence:” lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    desc = next((l for l in lines if l.lower().startswith("description:")), "")
    evid = next((l for l in lines if l.lower().startswith("evidence:")), "")
    # Fallbacks if model didn’t prefix correctly
    if not desc:
        desc = "Description: A collision scenario is visible based on motion and entry order."
    if not evid:
        evid = "Evidence: Paths cross near the center of the intersection."
    # Ensure no truncation mid-word
    if not re.search(r"[.!?]$", desc):
        desc += "."
    if not re.search(r"[.!?]$", evid):
        evid += "."
    return desc + "\n" + evid
# =========================================================
# Fault-BERT (.pth) - 네가 학습한 모델로부터 임베딩을 뽑는다
# =========================================================
class TextToFaultRatio(nn.Module):
    def __init__(self, model_name="bert-base-uncased", hidden_dim=768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.regressor(cls)

@torch.no_grad()
def load_fault_model(path: str, model_name="bert-base-uncased", device="cuda"):
    """네가 학습한 .pth를 로드하여 encoder까지 가중치가 반영된 Fault-BERT를 리턴"""
    obj = torch.load(path, map_location="cpu")
    model = TextToFaultRatio(model_name=model_name)
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
        # 필요시 prefix 정리
        new_sd = { (k.replace("module.", "")): v for k, v in sd.items() }
        model.load_state_dict(new_sd, strict=False)
    elif isinstance(obj, dict):
        new_sd = { (k.replace("module.", "")): v for k, v in obj.items() }
        model.load_state_dict(new_sd, strict=False)
    elif hasattr(obj, "state_dict"):
        model = obj
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

@torch.no_grad()
def predict_fault_ratio(model, tokenizer, text: str, device="cuda", max_length=256, total=10.0, temperature: float = 1.0):
    inputs = tokenizer(text, return_tensors="pt",
                       padding="max_length", truncation=True, max_length=max_length).to(device)
    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])  # [1,2]
    probs = F.softmax(logits / temperature, dim=-1).squeeze(0).float().cpu().numpy()        # [2], sum=1
    return probs * float(total)  # sum=total

def project_pair_to_basis(v, total=10.0):
    v = np.maximum(np.asarray(v, dtype=float), 0.0)
    s = float(v.sum())
    if s <= 0:
        return np.array([total/2.0, total/2.0], dtype=float)
    return v * (total / s)


# -------------------------
# Label 텍스트
# -------------------------
real_categories_ids_2nd = {
  1 : "Changing Lanes Within Intersection",
  2 : "Lane Change Path Change",
  3 : "Lanes Wide Enough For Two Vehicles Side By Side",
  4 : "Main Road And Side Road",
  5 : "Other Vehicle Enters From The Side",
  6 : "Roads Of Equal Width",
  7 : "Start After Stopping Accident",
  8 : "Turning Angle Less Than 90 Degrees",
  9 : "Two Vehicles Turning Right Simultaneously"
}
dashcam_vehicle_info = {
    1 : "Facing Each Other Going Straight",
    2 : "Following Vehicle Going Straight After Leaving Safety Zone",
    3 : "Following Vehicle Going Straight Right Side Of Lane",
    4 : "Going Straight From Left Road",
    5 : "Going Straight From Main Road",
    6 : "Going Straight From Main Road Entered Earlier",
    7 : "Going Straight From Main Road Entered Later",
    8 : "Going Straight From Right Road",
    9 : "Going Straight From Side Road Left",
    10 : "Going Straight From Side Road Right",
    11 : "Going Straight From The Right",
    12 : "Going Straight From The Right Entered Earlier",
    13 : "Going Straight From The Right Entered Later",
    14 : "Going Straight Lane Change Inside Intersection",
    15 : "Green Light Going Straight",
    16 : "Green Light Going Straight Collided With Red Light Vehicle",
    17 : "Left Turn From Right Road",
    18 : "Left Turn From Side Road",
    19 : "Right Turn",
    20 : "Right Turn Entered Earlier",
    21 : "Right Turn Entered Later",
    22 : "Right Turn From Main Road",
    23 : "Right Turn From Main Road Entered Later",
    24 : "Right Turn From Side Road",
    25 : "Right Turn From Side Road Entered Earlier",
    26 : "Right Turn From Side Road Entered Later",
    27 : "Right Turn Right Lane",
    28 : "Simultaneous Lane Change",
    29 : "Yellow Light Going Straight",
    30 : "Yellow Light Left Turn Collided With Red Light Vehicle"
}
other_vehicle_info = {
  1 : "Departing After Stop",
  2 : "Facing Each Other Going Straight",
  3 : "Following Vehicle Going Straight",
  4 : "Following Vehicle Going Straight Left Side Of Lane",
  5 : "Following Vehicle Going Straight Right Side Of Lane",
  6 : "Going Straight From Left Road",
  7 : "Going Straight From Main Road",
  8 : "Going Straight From Main Road Entered Earlier",
  9 : "Going Straight From Main Road Entered Later",
  10 : "Going Straight From Right Road",
  11 : "Going Straight From Side Road Left",
  12 : "Going Straight From Side Road Right",
  13 : "Going Straight From The Right",
  14 : "Going Straight From The Right Entered Earlier",
  15 : "Going Straight From The Right Entered Later",
  16 : "Green Left Turn Signal Left Turn Collided With Red Light Vehicle",
  17 : "Green Light Going Straight",
  18 : "Left Turn From Right Road",
  19 : "Left Turn From Side Road",
  20 : "Right Turn",
  21 : "Right Turn Entered Earlier",
  22 : "Right Turn Entered Later",
  23 : "Right Turn From Main Road",
  24 : "Right Turn From Main Road Entered Earlier",
  25 : "Right Turn From Main Road Entered Later",
  26 : "Right Turn From Side Road",
  27 : "Right Turn From Side Road Entered Later",
  28 : "Right Turn Right Lane"
}
def _dict_to_list_by_id(d: dict): return [d[k] for k in sorted(d.keys())]
LABELS = {
    "dv": _dict_to_list_by_id(dashcam_vehicle_info),
    "pl": _dict_to_list_by_id(real_categories_ids_2nd),
    "ov": _dict_to_list_by_id(other_vehicle_info),
}
def render_template_from_labels(dv_text: str, ov_text: str) -> str:
    return (f"At an unsignalized intersection, the Dashcam Vehicle was {dv_text}, "
            f"while the Other Vehicle was {ov_text}.")

# =========================================================
# 비디오 분류기 (DV/OV 라벨 예측)
# =========================================================
def build_video_only_prompt():
    return (
        "You are a traffic-accident legal analyst. "
        "Describe ONLY what is VISIBLE in the frames. "
        "Do NOT mention traffic lights, numbers, timestamps, speeds, or any unseen elements.\n"
        "Write 2–3 short factual sentences focusing on:\n"
        "- motion (straight/left/right turns), entry order, and relative positions.\n"
        "- No speculation or hidden context."
    )

def build_prompt(
    user_message: str,
    dashcam_info: Optional[str],
    other_info: Optional[str],
    classifier_topk: Optional[str],
    style: str = "brief",
    history: Optional[list] = None,
    history_max_turns: int = 6,
    fault_hint: Optional[str] = None,
) -> str:
    system_header = (
        "You are a **traffic-accident legal analyst**. "
        "Always prioritize **what is visible in the video**. "
        "If any hint (classification/fault estimate) conflicts with the frames, briefly note the conflict and follow the visual evidence. "
        "Do NOT invent traffic lights, numbers, timestamps, lane counts, speeds, or any unseen objects.\n"
    )

    if style == "short":
        length_rule = "Write exactly one concise sentence (~30 words)."
    elif style == "detailed":
        length_rule = "Write 4–6 factual sentences (~80–120 words)."
    else:
        style = "brief"
        length_rule = "Write 2–3 factual sentences (under ~90 words)."

    hint_lines = []
    if dashcam_info:
        hint_lines.append(f"- Dashcam Vehicle: {dashcam_info}")
    if other_info:
        hint_lines.append(f"- Other Vehicle: {other_info}")
    if classifier_topk:
        hint_lines.append(f"- Classifier outputs (top-k): {classifier_topk}")
    if fault_hint:
        hint_lines.append(f"- Fault analysis [Dashcam:Other]: {fault_hint}")
    hint_block = ("Include the following hints **exactly once** if they agree with visible evidence:\n" + "\n".join(hint_lines) + "\n") if hint_lines else ""

    # Include recent chat context (if any)
    hist_txt = ""
    if history:
        turns = history[-history_max_turns:]
        for m in turns:
            role = (m.get("role") or "").upper()
            msg  = m.get("content") or ""
            if msg:
                hist_txt += f"{role}:\n{msg}\n\n"

    user_header = (
        f"{hist_txt}"
        "USER:\n"
        "Describe the accident focusing on **motion, entry order, and relative positions** visible in the frames.\n"
        f"{length_rule}\n"
        f"{hint_block}"
        "Output requirements:\n"
        "1) **Description**: 2–3 factual sentences based ONLY on visible evidence.\n"
        "2) **Fault (basis N)**: predicted fault ratio (basis N, one decimal), and identify **Victim** (lower share) and **At-fault** vehicle.\n"
        "3) **Cause**: one-sentence cause grounded in visible evidence (and the hints if they agree).\n"
        "Avoid: traffic lights/signals, arbitrary numbers (except the fault ratio), dates/timestamps, unseen objects/lanes/speeds. "
        "If a hint conflicts with the frames, state the conflict briefly and follow the video evidence.\n"
    )
    if user_message and user_message.strip():
        user_header += f"\nAdditional instruction: {user_message.strip()}\n"

    placeholder = "<video>"
    return f"{placeholder}\n" + system_header + user_header + "ASSISTANT:"

# def build_revise_prompt(draft_text: str, user_message: str, fault_hint_line: str) -> str:
#     return (
#         "You are a traffic-accident legal analyst. Revise the DRAFT based ONLY on visible evidence in the frames. "
#         "If the hint agrees, reflect it ONCE; if it conflicts, briefly note the conflict and keep the visual evidence.\n"
#         "Return EXACTLY TWO LINES:\n"
#         "Description: <one or two concise factual sentences>\n"
#         "Evidence: <the single most decisive visible cue>\n"
#         f"DRAFT:\n{draft_text}\n"
#         f"HINT:\n{fault_hint_line}\n"
#         f"{('Additional instruction: ' + user_message.strip()) if user_message else ''}"
#     )
def build_revise_prompt(draft_text: str, user_message: str, fault_hint_line: str) -> str:
    return (
        "You are a traffic-accident legal analyst. Revise the DRAFT based ONLY on visible evidence in the frames. "
        "If the hint agrees, reflect it ONCE; if it conflicts, briefly note the conflict and keep the visual evidence.\n"
        "Description: <one or two concise factual sentences>\n"
        "Evidence: <the single most decisive visible cue>\n"
        f"DRAFT:\n{draft_text}\n"
        f"HINT:\n{fault_hint_line}\n"
        f"{('Additional instruction: ' + user_message.strip()) if user_message else ''}"
    )

def _ensure_video_token(processor, text: str) -> str:
    # 모델/프로세서에 따라 토큰 이름이 다를 수 있어 안전하게 처리
    video_token = getattr(processor, "video_token", None)
    if not video_token:
        video_token = getattr(processor, "image_token", None) or "<video>"
    if video_token not in text:
        return f"{video_token}\n{text}"
    return text

# =========================================================
# 분류기
# =========================================================
class TripleHeadVideoClassifier(nn.Module):
    def __init__(self, n_dv, n_ov, backbone="r3d18", pretrained=False):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "r3d18":
            self.backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1 if pretrained else None)
            feat = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "timesformer":
            self.backbone = TimesformerModel.from_pretrained(
                "facebook/timesformer-base-finetuned-k400", use_safetensors=True
            )
            feat = self.backbone.config.hidden_size
        elif backbone == "videomae":
            self.backbone = VideoMAEModel.from_pretrained(
                "MCG-NJU/videomae-base-finetuned-kinetics", use_safetensors=True
            )
            feat = self.backbone.config.hidden_size
        else:
            raise ValueError(backbone)

        self.dv = nn.Linear(feat, n_dv)
        self.ov = nn.Linear(feat, n_ov)

    def forward(self, x):
        if self.backbone_name == "r3d18":
            z = self.backbone(x)              # (B, feat)
        else:
            x = x.permute(0,2,1,3,4)          # (B,T,C,H,W)
            out = self.backbone(x)
            z = out.last_hidden_state.mean(1) # (B, feat)
        return self.dv(z), self.ov(z)

@torch.no_grad()
def load_video_tensor_for_clf(path, num_frames=16, size=224, device="cuda"):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"no frames in {path}")
    idxs = np.linspace(0, total-1, num_frames).astype(int)

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok: continue
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        fr = cv2.resize(fr, (size, size))
        fr = torch.from_numpy(fr).permute(2,0,1).float()/255.0  # (C,H,W)
        frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError(f"no readable frames in {path}")

    vid = torch.stack(frames, 0)                        # (T,C,H,W)
    vid = vid.permute(1,0,2,3).unsqueeze(0).to(device)  # (1,C,T,H,W)

    # 디바이스/정규화 안정화
    # mean = torch.tensor([0.485, 0.456, 0.406], device=vid.device).view(1,3,1,1)
    # std  = torch.tensor([0.229, 0.224, 0.225], device=vid.device).view(1,3,1,1)
    # vid = (vid - mean) / std
    mean = torch.tensor([0.485, 0.456, 0.406], device=vid.device).view(1, 3, 1, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=vid.device).view(1, 3, 1, 1, 1)
    vid = (vid - mean) / std
    return vid

def load_classifier(ckpt_path: str, backbone="r3d18", device="cuda", pretrained=False):
    model = TripleHeadVideoClassifier(
        n_dv=len(LABELS["dv"]), n_ov=len(LABELS["ov"]),
        backbone=backbone, pretrained=pretrained
    )
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    new_sd = { (k[len("module."):] if k.startswith("module.") else k): v for k,v in sd.items() }
    model.load_state_dict(new_sd, strict=False)
    model.to(device).eval()
    return model

@torch.no_grad()
def predict_classifier(video_path, clf_model, device="cuda", topk=3):
    x = load_video_tensor_for_clf(video_path, num_frames=16, size=224, device=device)
    ld, lv = clf_model(x)   # dv, ov

    def _topk_text(logits, label_list, k):
        probs = logits.softmax(dim=-1)[0]
        val, idx = probs.topk(k=k, dim=-1)
        names = [label_list[i] for i in idx.tolist()]
        top1 = names[0]
        return top1, ", ".join(names)

    dv1, dv_topk = _topk_text(ld, LABELS["dv"], topk)
    ov1, ov_topk = _topk_text(lv, LABELS["ov"], topk)

    dashcam_info   = dv1
    other_info     = ov1
    classifier_top = f"DV[{dv_topk}] | OV[{ov_topk}]"
    return dashcam_info, other_info, classifier_top

# =========================================================
# 파일/비디오 유틸
# =========================================================
def list_candidates(limit=50):
    items = []
    for root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        for f in os.listdir(root):
            fname = f.lower()
            for ext in VIDEO_EXTS:
                if fname.endswith(ext):
                    items.append(os.path.join(root, f))
    items = [p for p in items if os.path.isfile(p)]
    items.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return items[:limit]

def find_video_path_by_name(name: str) -> Optional[str]:
    name = name.strip().strip('"').strip("'")
    if not name:
        return None
    has_ext = os.path.splitext(name)[1].lower() in VIDEO_EXTS

    for root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        if any(ch in name for ch in "*?[]"):
            patterns = [os.path.join(root, name)]
        else:
            patterns = [os.path.join(root, name)] if has_ext else [os.path.join(root, name + ext) for ext in VIDEO_EXTS]
        for pat in patterns:
            hits = sorted(glob.glob(pat))
            if hits:
                hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return hits[0]
    return None

def _resolve_video_path(video, video_name: Optional[str]) -> Optional[str]:
    # gr.Video는 dict가 들어오기도 함
    path = None
    if isinstance(video, str) and os.path.exists(video):
        path = video
    elif isinstance(video, dict):
        cand = video.get("name") or video.get("video") or video.get("path")
        if cand and os.path.exists(cand):
            path = cand
    if not path and video_name:
        cand = find_video_path_by_name(video_name)
        if cand and os.path.exists(cand):
            path = cand
    if not path and video_name and os.path.exists(video_name):
        path = video_name
    return path

# ---------------- Frame Sampler ----------------
def sample_frames(video_path: str, num_frames: int = 8, size: int = 224) -> List[Image.Image]:
    frames = []
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        if total <= 0:
            raise RuntimeError("Empty video (decord).")
        idxs = (np.linspace(0, total - 1, num_frames)).astype(int).tolist()
        for i in idxs:
            arr = vr[i].asnumpy()  # Decord는 RGB 반환 (BGR 변환 금지)
            frames.append(Image.fromarray(arr).resize((size, size)))
    except Exception:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise RuntimeError("Empty/unreadable video (OpenCV).")
        idxs = (np.linspace(0, total - 1, num_frames)).astype(int).tolist()
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if not ok:
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8)))
                continue
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(fr).resize((size, size)))
        cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames[:num_frames]

# ---------------- Prompt Builders ----------------
def _ensure_pad_token(tok, model):
    eos_id = tok.eos_token_id or getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if tok.pad_token_id is None:
        if eos_id is not None:
            tok.pad_token_id = eos_id
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tok))
    return tok.pad_token_id, (tok.eos_token_id or eos_id)

def _apply_chat_template(processor, user_text: str) -> str:
    """
    VideoLLaVA Processor/Tokenizer에 chat template가 없으면 조용히 폴백한다.
    """
    # 1) 메시지 포맷
    messages = [{
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text": user_text},
        ],
    }]

    # 2) 템플릿 함수 탐색
    apply_fn = getattr(processor, "apply_chat_template", None)
    if apply_fn is None:
        apply_fn = getattr(getattr(processor, "tokenizer", None), "apply_chat_template", None)

    # 3) 없으면 폴백: <video> 토큰만 보장
    if apply_fn is None:
        return _ensure_video_token(processor, user_text)

    # 4) 있으면 사용 (여기서도 예외 터지면 폴백)
    try:
        return apply_fn(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return _ensure_video_token(processor, user_text)

def _find_video_token_insert_pos(processor, input_ids: torch.Tensor) -> int:
    """비디오 토큰이 있으면 그 바로 뒤에 soft token 삽입"""
    vid_token = getattr(processor, "video_token", None) or getattr(processor, "image_token", None)
    if not vid_token:
        return 2
    tok = processor.tokenizer
    vid_ids = tok(vid_token, add_special_tokens=False).input_ids
    ids = input_ids[0].tolist()
    # 간단 탐색(단일 토큰 가정)
    if len(vid_ids) == 1:
        tid = vid_ids[0]
        for i, t in enumerate(ids):
            if t == tid:
                return i + 1
    # 못 찾으면 기본값
    return 2

# =========================================================
# Video-LLaVA 엔진 + soft token 삽입 경로
# =========================================================
class VideoLLaVAChatEngine:
    def __init__(self):
        self.model_id = None
        self.processor = None
        self.model = None
        self.device = self._pick_device()

        # BERT-soft 토큰 투영기 (Fault-BERT hidden -> LLaVA hidden)
        self._hint_proj: Optional[nn.Module] = None
        self._hint_proj_dim: Optional[int] = None

    def _pick_device(self):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self, model_id: str):
        if self.model_id == model_id and self.model is not None:
            return f"Model already loaded: {model_id}"
        self.processor = VideoLlavaProcessor.from_pretrained(model_id)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = VideoLlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=None,
        ).to(self.device)
        self.model.config.use_cache = False
        self.model_id = model_id
        return f"Loaded model: {model_id}"

    def _ensure_hint_proj(self, llm_hidden_size: int, bert_hidden_size: int):
        if (self._hint_proj is None) or (self._hint_proj_dim != llm_hidden_size):
            self._hint_proj = nn.Sequential(
                nn.Linear(bert_hidden_size, llm_hidden_size),
                nn.Tanh(),
                nn.LayerNorm(llm_hidden_size)
            ).to(self.device).eval()
            self._hint_proj_dim = llm_hidden_size
        return self._hint_proj

    @torch.no_grad()
    def make_soft_tokens_from_trained_fault(
        self,
        fault_m: TextToFaultRatio,
        fault_tok: AutoTokenizer,
        text: str,
        take: str = "cls",
        max_tokens: int = 4
    ) -> torch.Tensor:
        """Fault-BERT encoder 히든을 LLaVA 차원으로 투영 → soft tokens [1,M,H]"""
        tok_inputs = fault_tok(text, return_tensors="pt", truncation=True, max_length=256).to(self.device)
        hs = fault_m.encoder(**tok_inputs).last_hidden_state         # [1, Lb, Hb]
        Hb = hs.size(-1)
        H  = self.model.get_input_embeddings().embedding_dim
        proj = self._ensure_hint_proj(H, Hb)

        hs = hs.to(next(proj.parameters()).dtype)

        if take == "cls":
            z = hs[:, 0, :]
            s = proj(z).unsqueeze(1)                                 # [1,1,H]
        elif take == "mean":
            z = hs.mean(dim=1)
            s = proj(z).unsqueeze(1)
        else:
            K = min(hs.size(1), max_tokens)
            z = hs[:, :K, :]
            s = proj(z)

        max_norm = 2.0
        s_norm = s.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        s = s * (max_norm / s_norm).clamp_max(1.0)

        s = s.to(self.model.dtype)
        return s

    @torch.no_grad()
    def generate_video_only(self, frames: List[Image.Image], chat_prompt: str,
                            temperature: float = 0.2, do_sample: bool = False,
                            max_new_tokens: int = 256) -> str:
        model = self.model
        processor = self.processor
        device = self.device
        tok = processor.tokenizer

        # 비디오 토큰 보장
        safe_prompt = _apply_chat_template(processor, chat_prompt)

        # tokenizer pad/eos 안전장치
        pad_id, eos_id = _ensure_pad_token(tok, model)

        # 템플릿 없이 평문 + 비디오만
        proc = processor(videos=[frames], text=[safe_prompt],
                         padding="longest", truncation=False, return_tensors="pt")

        if "pixel_values_videos" in proc:
            vision = {"pixel_values_videos": proc["pixel_values_videos"].to(device, dtype=model.dtype)}
        else:
            vision = {"pixel_values_videos": proc["pixel_values"].to(device, dtype=model.dtype)}

        input_ids = proc["input_ids"].to(device)
        attn      = proc["attention_mask"].to(device)

        bad_words = ["traffic light","signalized","timestamp","AM","PM"]
        bad_ids = []
        for w in bad_words:
            ids = tok(w, add_special_tokens=False).input_ids
            if ids:  # 빈 리스트 방지
                bad_ids.append(ids)

        gen_ids = model.generate(
            **vision,
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=1,
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
            bad_words_ids=bad_ids if bad_ids else None,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
        gen_only = gen_ids[0, input_ids.shape[1]:]
        text_out = tok.decode(gen_only, skip_special_tokens=True).strip()

        # >>> NEW: never fall back to full decode (that re-inserts the prompt)
        text_out = _clean_generated(text_out)
        if not text_out:
            # one retry with mild sampling to avoid empty output
            gen_ids = model.generate(
                **vision,
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                temperature=max(0.3, temperature),
                do_sample=True,
                num_beams=1,
                no_repeat_ngram_size=3,
                repetition_penalty=1.05,
                bad_words_ids=bad_ids if bad_ids else None,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )
            gen_only = gen_ids[0, input_ids.shape[1]:]
            text_out = _clean_generated(tok.decode(gen_only, skip_special_tokens=True).strip())
        return text_out or "The vehicles converge in the intersection based on their visible paths."

    @torch.no_grad()
    def generate_with_soft_tokens_from_fault(
        self,
        frames: List[Image.Image],
        chat_prompt: str,
        soft_tokens: Optional[torch.Tensor],   # Optional
        insert_pos: int = 2,
        temperature: float = 0.2,
        do_sample: bool = False,
        max_new_tokens: int = 256,
    ) -> str:
        model = self.model
        processor = self.processor
        device = self.device
        tok = processor.tokenizer

        # 안전한 프롬프트(비디오 토큰 보장)
        safe_prompt = _ensure_video_token(processor, chat_prompt)

        pad_id, eos_id = _ensure_pad_token(tok, model)

        proc = processor(videos=[frames], text=[safe_prompt],
                         padding="longest", truncation=False, return_tensors="pt")

        if "pixel_values_videos" in proc:
            vision = {"pixel_values_videos": proc["pixel_values_videos"].to(device, dtype=model.dtype)}
        else:
            vision = {"pixel_values_videos": proc["pixel_values"].to(device, dtype=model.dtype)}

        input_ids = proc["input_ids"].to(device)
        attn      = proc["attention_mask"].to(device)

        bad_words = ["traffic light","signalized","timestamp","AM","PM"]
        bad_ids = [tok(w, add_special_tokens=False).input_ids
                   for w in bad_words if tok(w, add_special_tokens=False).input_ids]

        # (A) 소프트 토큰 미사용
        if soft_tokens is None:
            gen_ids = model.generate(
                **vision,
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                num_beams=1,
                no_repeat_ngram_size=3,
                repetition_penalty=1.05,
                bad_words_ids=bad_ids if bad_ids else None,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )
            gen_only = gen_ids[0, input_ids.shape[1]:]               # ✨
            text_out = tok.decode(gen_only, skip_special_tokens=True).strip()
            if not text_out:
                text_out = tok.decode(gen_ids[0], skip_special_tokens=True).strip()
            return text_out

        # (B) 소프트 토큰 사용
        text_embeds = model.get_input_embeddings()(input_ids).to(device, dtype=model.dtype)
        S = soft_tokens.to(device, dtype=model.dtype)

        # 비디오 토큰 바로 뒤에 삽입 시도
        pos = _find_video_token_insert_pos(processor, input_ids)
        L = text_embeds.size(1)
        pos = max(1, min(pos, L - 1))

        inputs_embeds = torch.cat([text_embeds[:, :pos, :], S, text_embeds[:, pos:, :]], dim=1)
        new_mask = torch.cat([
            attn[:, :pos],
            torch.ones(attn.size(0), S.size(1), device=device, dtype=attn.dtype),
            attn[:, pos:]
        ], dim=1)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            min_new_tokens=32,          # ✨ 최소 생성 길이
            early_stopping=False,
            num_beams=1,
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
        if do_sample:
            gen_kwargs.update(dict(do_sample=True, temperature=temperature))

        # ❗ inputs_embeds 사용 시에는 보통 출력이 "생성 토큰만" 반환됩니다.
        gen_ids = model.generate(**vision, inputs_embeds=inputs_embeds, attention_mask=new_mask, **gen_kwargs)

        # ❗❗ 슬라이스 하지 말고 바로 디코딩
        text_out = tok.decode(gen_ids[0], skip_special_tokens=True).strip()
        if not text_out:
            text_out = tok.decode(gen_ids[0], skip_special_tokens=False).strip()
        return text_out

    @torch.no_grad()
    def revise_with_soft_tokens(self, frames: List[Image.Image], chat_prompt: str,
                                soft_tokens: Optional[torch.Tensor], gain: float = 0.5,
                                temperature: float = 0.2, do_sample: bool = False,
                                max_new_tokens: int = 180) -> str:
        # gain으로 세기 조절
        if soft_tokens is not None:
            soft_tokens = soft_tokens * float(gain)
        return self.generate_with_soft_tokens_from_fault(
            frames=frames,
            chat_prompt=chat_prompt,
            soft_tokens=soft_tokens,
            insert_pos=2,
            temperature=temperature,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
        )

ENGINE = VideoLLaVAChatEngine()

# =========================================================
# Gradio UI
# =========================================================
def ui_load_model(model_id):
    try:
        msg = ENGINE.load_model(model_id)
    except Exception as e:
        msg = f"Load error: {e}"
    return msg

_CLF_CACHE = {}
_FAULT_CACHE = {}

def _ensure_classifier(ckpt_path, backbone, pretrained, device):
    key = (ckpt_path, backbone, bool(pretrained), device)
    if key not in _CLF_CACHE:
        mdl = load_classifier(ckpt_path, backbone=backbone, device=device, pretrained=pretrained)
        _CLF_CACHE[key] = mdl
    return _CLF_CACHE[key]

def _ensure_fault(ckpt_path, model_name, device):
    key = (ckpt_path, model_name, device)
    if key not in _FAULT_CACHE:
        mdl, tok = load_fault_model(ckpt_path, model_name=model_name, device=device)
        _FAULT_CACHE[key] = (mdl, tok)
    return _FAULT_CACHE[key]

def ui_generate(
        history, user_msg, video, video_name, model_id,
        clf_ckpt, clf_backbone, clf_pretrained, clf_topk,
        fault_ckpt, fault_model, fault_basis,
        style, num_frames, frame_size, temperature, sampling,
        use_soft, interp_mode, soft_gain
    ):
    history = history or []

    # 0) 비디오 찾기
    video_path = _resolve_video_path(video, video_name)
    print(f"[Chat] video_path={video_path}, video_name={video_name}")
    if not video_path:
        history = history + [{"role":"assistant","content": f"❌ Video not found. name='{video_name}'\n검색 루트: {SEARCH_ROOTS}"}]
        return history
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        history = history + [{"role":"assistant","content": f"❌ Video not readable: {video_path}"}]
        return history

    device = ENGINE.device

    # 1) DV/OV 분류 → top-k 텍스트
    try:
        clf = _ensure_classifier(clf_ckpt, clf_backbone, clf_pretrained, device)
        dv_hint, ov_hint, topk_text = predict_classifier(
            video_path, clf, device=device, topk=int(clf_topk)
        )
    except Exception as e:
        dv_hint, ov_hint = "UNKNOWN_DV", "UNKNOWN_OV"
        topk_text = f"[classifier error: {e}]"
    print(f"[Hint] DV={dv_hint} | OV={ov_hint} | TOPK={topk_text}")

    # 2) 템플릿 문장
    sentence = render_template_from_labels(dv_hint, ov_hint)

    # 3) Fault-BERT 추론 (숫자는 금지어로 막지만, 벡터는 soft token으로 전달)
    try:
        fr_m, fr_tok = _ensure_fault(fault_ckpt, fault_model, device)
        y = predict_fault_ratio(fr_m, fr_tok, sentence, device=device)  # [2]
        # pred = project_pair_to_basis(y, total=float(fault_basis))       # 합 = basis
        # dc_f, ov_f = float(pred[0]), float(pred[1])
        dc_f, ov_f = float(y[0]), float(y[1])   
        if abs(dc_f - ov_f) < 1e-6:
            victim = "Undetermined"
        else:
            victim = "Ego Vehicle" if dc_f < ov_f else "Other Vehicle"
        fault_hint = f"Predicted fault (basis {int(fault_basis)}): dashcam={dc_f:.1f}, Other={ov_f:.1f}; Likely victim: {victim}; Template: {sentence}"
    except Exception as e:
        fault_hint = f"[fault error: {e}]; Template: {sentence}"
        dc_f, ov_f, victim = 5.0, 5.0, "Undetermined"

    # 4) Video-LLaVA 로드
    load_msg = ENGINE.load_model(model_id)
    print(load_msg)

    # 5) 프롬프트 + 프레임 준비
    frames = sample_frames(video_path, num_frames=int(num_frames), size=int(frame_size))
    prompt_text = build_prompt(
        user_message=user_msg or "",
        dashcam_info=dv_hint,
        other_info=ov_hint,
        classifier_topk=(None if interp_mode else topk_text),  # ← 이렇게 변경
        style=style,
        history=history,
        fault_hint=fault_hint
    )

    # 6) Soft-token 준비(옵션)
    bert_hint_text = f"dashcam={dv_hint}; Other={ov_hint}; {fault_hint}"
    soft = None
    if use_soft:
        try:
            fr_m, fr_tok = _ensure_fault(fault_ckpt, fault_model, device)
            soft = ENGINE.make_soft_tokens_from_trained_fault(
                fault_m=fr_m, fault_tok=fr_tok, text=bert_hint_text, take="cls", max_tokens=4
            )  # [1,1,H]
        except Exception as e:
            soft = None
            print(f"[SoftToken] error: {e}")

    # 7) 생성 경로: (A) 해석가능성 2패스 or (B) 단일패스
    try:
        if interp_mode:
            # 1) 비디오만으로 초안
            draft_prompt = build_video_only_prompt()
            draft = ENGINE.generate_video_only(
                frames=frames, chat_prompt=draft_prompt,
                temperature=float(temperature), do_sample=bool(sampling), max_new_tokens=150
            )
            # 2) 소프트 토큰으로 '수정'(use_soft=False면 초안 그대로)
            revise_prompt = build_revise_prompt(draft, user_msg or "", fault_hint)
            text = ENGINE.revise_with_soft_tokens(
                frames=frames,
                chat_prompt=revise_prompt,
                soft_tokens=soft,
                gain=float(soft_gain),
                temperature=float(temperature),
                do_sample=bool(sampling),
                max_new_tokens=120,
            )
            # text = _enforce_two_line_revise(text)
            aux_header = f"📝 DRAFT (video-only): {draft}\n✏️ REVISED: {text}\n⚙️ soft_gain={float(soft_gain):.2f}, use_soft={bool(use_soft)}"
        else:
            # 단일 패스 (소프트 토큰 켜짐/꺼짐 반영)
            text = ENGINE.generate_with_soft_tokens_from_fault(
                frames=frames,
                chat_prompt=prompt_text,
                soft_tokens=soft,
                insert_pos=2,
                temperature=float(temperature),
                do_sample=bool(sampling),
                max_new_tokens=180,
            )
            aux_header = f"📝 {sentence}"
    except Exception as e:
        text = f"Generation error: {e}"
        aux_header = ""

    # 8) 채팅창 출력
    aux_tail = f"\n⚖️ Fault(basis {int(fault_basis)}): DC {dc_f:.1f} / OV {ov_f:.1f} | Victim: {victim}\n"
    if user_msg:
        history = history + [{"role": "user", "content": user_msg}]
    main = (text or "").strip()
    if not main:
        main = "⚠️ No description generated."
    meta = (aux_tail + aux_header).strip()
    bubble = main if not meta else f"{main}\n\n{meta}"
    history = history + [{"role": "assistant", "content": bubble}]
    return history

# =========================================================
# Gradio App
# =========================================================
def _fill_dropdown():
    items = list_candidates()
    # choices와 value를 동시에 업데이트
    return gr.Dropdown.update(choices=items, value=(items[0] if items else None))

with gr.Blocks(title="Video-LLaVA Chatbot") as demo:
    gr.Markdown("## 🎥 Video-LLaVA Chatbot — Multi-turn video-first accident analysis")

    with gr.Row():
        with gr.Column(scale=1):
            model_id = gr.Textbox(label="HF Model ID / local path", value="LanguageBind/Video-LLaVA-7B-hf")
            load_btn = gr.Button("Load / Reload Model")
            load_status = gr.Textbox(label="Load status", value="", interactive=False)

            # --- 소프트 토큰/해석가능성 옵션 ---
            use_soft   = gr.Checkbox(value=True,  label="Use Fault-BERT Soft Token")
            interp_mode= gr.Checkbox(value=True,  label="Interpretability: two-pass (video-first → soft revise)")
            soft_gain  = gr.Slider(0.0, 1.5, value=0.5, step=0.05, label="Soft token gain")

            video_name = gr.Textbox(label="Video Name (search in folders)",
                                    placeholder="예) crash_000123.mp4 또는 2024-09-*.mp4")
            scan_btn = gr.Button("Scan Videos")
            video_picker = gr.Dropdown(label="Pick found video (fills Video Name)", choices=[], value=None)

            video = gr.Video(label="Video (mp4/mov/webm)", interactive=True)

            clf_ckpt     = gr.Textbox(label="Classifier CKPT path", value="/app/checkpoints/best_exact_ep13_r3d18.pth")
            clf_backbone = gr.Dropdown(label="Classifier backbone", choices=["r3d18","timesformer","videomae"], value="r3d18")
            clf_pretrained = gr.Checkbox(label="Use pretrained backbone", value=False)
            clf_topk     = gr.Slider(1,5,value=3,step=1,label="Classifier Top-K")

            fault_ckpt   = gr.Textbox(label="Fault-BERT CKPT path", value="/app/text-train/fault_ratio_bert_modify_softmax.pt")
            fault_model  = gr.Textbox(label="Fault-BERT model_name", value="bert-base-uncased")
            fault_basis  = gr.Slider(2, 20, value=10, step=1, label="Fault ratio basis")

            style = gr.Dropdown(label="Output style", choices=["short", "brief", "detailed"], value="brief")
            num_frames = gr.Slider(4, 16, value=8, step=1, label="Frames")
            frame_size = gr.Slider(160, 336, value=224, step=16, label="Frame size")
            temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")
            sampling = gr.Checkbox(value=False, label="Enable sampling")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=520, show_copy_button=True, type="messages")
            state = gr.State([])  # 대화 히스토리
            user_msg = gr.Textbox(label="Your message", placeholder="추가 지시 입력")
            send_btn = gr.Button("Send ▶️")

    load_btn.click(ui_load_model, inputs=[model_id], outputs=[load_status])
    scan_btn.click(_fill_dropdown, inputs=[], outputs=[video_picker])
    video_picker.change(lambda p: os.path.basename(p) if p else None, inputs=[video_picker], outputs=[video_name])

    send_btn.click(
        ui_generate,
        inputs=[
            state, user_msg, video, video_name, model_id,
            clf_ckpt, clf_backbone, clf_pretrained, clf_topk,
            fault_ckpt, fault_model, fault_basis,
            style, num_frames, frame_size, temperature, sampling,
            use_soft, interp_mode, soft_gain
        ],
        outputs=[chatbot],
    ).then(lambda h: h, inputs=[chatbot], outputs=[state])

if __name__ == "__main__":
    try:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", 7861)))
    share = os.environ.get("GRADIO_SHARE", "1") == "1"

    demo.queue()
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_error=True,
        inbrowser=False,
        prevent_thread_lock=False,
    )
    print(f"\n[Gradio] listening on http://{server_name}:{server_port}  (share={share})")