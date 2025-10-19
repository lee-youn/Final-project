#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_paper_plots.py
- Top-K context cosine bar (v vs template-conditioned phrase vectors)
- Local PCA map of {v} ∪ Top-K
- Prints P@K, Δ(mean_imp - mean_dist), permutation p-value
- Saves CSVs and PNG/PDFs (300dpi)

Usage (예시):
python make_paper_plots.py \
  --ckpt_pth fault_ratio_bert_modify_softmax.pt \
  --important "going straight,left turn,right turn,u turn,lane change,simultaneous lane change,following vehicle,facing each other,departing after stop,safety zone,main road,side road,left road,right road,left lane,right lane,left side of lane,right side of lane,green light,yellow light,red light,green left turn signal,pedestrian crossing,stop sign,entered earlier,entered later" \
  --distractor "banana,unicorn,microwave,weather,education,computer,coffee,universe,abstract art,random token,river,castle,airplane,galaxy,software,biology,cooking,basketball,waterfall,mountain,smartphone,jazz music,knitting,astronomy" \
  --out_dir figs_paper --save_prefix paper --K 15 --context_mode delta \
  --templates "In traffic, the vehicle performed {PHRASE} at an intersection.;Dashcam incident: {PHRASE} by the ego vehicle.;Traffic rule violation: {PHRASE}.;The other vehicle made a {PHRASE} near the crosswalk.;Safety report states {PHRASE} during a green signal."
"""

import os, argparse, random, csv
from typing import List, Dict, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- utils ----------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_E(sd: dict) -> torch.Tensor:
    key = "encoder.embeddings.word_embeddings.weight"
    if key not in sd:
        raise KeyError(f"'{key}' missing in checkpoint.")
    return sd[key].detach().clone()

def derive_v_from_regressor(sd: dict) -> torch.Tensor:
    if ("regressor.0.weight" not in sd) or ("regressor.2.weight" not in sd):
        raise KeyError("Need 'regressor.0.weight' and 'regressor.2.weight' to derive v.")
    W0 = sd["regressor.0.weight"]     # [H,D]
    W2 = sd["regressor.2.weight"]     # [2,H] or [H,2]
    if W2.dim()==2 and W2.shape[0] != 2 and W2.shape[1] == 2:
        W2 = W2.t()
    v = W0.t() @ (W2[0] - W2[1])      # [D]
    return v / (v.norm() + 1e-9)

def get_v(sd: dict) -> torch.Tensor:
    for k in ["soft_token","v","latent","soft","vector"]:
        if k in sd and isinstance(sd[k], torch.Tensor):
            vv = sd[k].reshape(-1)
            return vv / (vv.norm() + 1e-9)
    return derive_v_from_regressor(sd)

def ids_from_text(tok, text: str, vocab_size: int) -> List[int]:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    unk = getattr(tok, "unk_token_id", None)
    return [i for i in ids if 0 <= i < vocab_size and (unk is None or i != unk)]

def mean_emb(E: torch.Tensor, ids: List[int]) -> torch.Tensor:
    if len(ids)==0: return None
    return E[ids].mean(dim=0)

def context_vec_for_phrase(phrase: str, templates: List[str], tok, E, vocab_size: int, mode="delta"):
    vecs = []
    for tpl in templates:
        sent = tpl.replace("{PHRASE}", phrase)
        mw = mean_emb(E, ids_from_text(tok, sent, vocab_size))
        if mw is None: 
            continue
        if mode == "delta":
            base = tpl.replace("{PHRASE}", "").replace("  "," ").strip()
            m0 = mean_emb(E, ids_from_text(tok, base, vocab_size))
            vecs.append(mw if m0 is None else (mw - m0))
        else:
            vecs.append(mw)
    if not vecs: return None
    v = torch.stack(vecs, 0).mean(0)
    return v / (v.norm() + 1e-9)

def permutation_pvalue(a, b, n_perm=20000, seed=42):
    rng = np.random.RandomState(seed)
    a = np.asarray(a); b = np.asarray(b)
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a,b])
    cnt = 0; nA = len(a)
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(pool[:nA].mean() - pool[nA:].mean()) >= obs - 1e-12:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)

def set_style():
    plt.rcParams.update({
        "figure.dpi": 180, "savefig.dpi": 300,
        "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 14,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False
    })

# ---------- plotting ----------
BLUE = "#1f77b4"; RED = "#d62728"; GREEN = "#2ca02c"

def plot_bar_topk(phrases, scores, groups, out_png, out_pdf, title):
    order = np.argsort(-np.array(scores))
    phrases = [phrases[i] for i in order]
    scores  = [scores[i]  for i in order]
    groups  = [groups[i]  for i in order]
    colors  = [BLUE if g=="important" else RED for g in groups]
    plt.figure(figsize=(max(6, 0.35*len(phrases)), 4))
    x = np.arange(len(phrases))
    plt.bar(x, scores, color=colors, edgecolor="black", linewidth=0.3)
    plt.xticks(x, phrases, rotation=60, ha="right")
    plt.ylabel("cosine(v, context phrase)")
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_png); plt.savefig(out_pdf); plt.close()

def plot_local_pca(v, means, labels, groups, out_png, out_pdf, title):
    # PCA via SVD
    X = torch.vstack([v.unsqueeze(0)] + [m.unsqueeze(0) for m in means]).cpu().numpy()
    Xc = X - X.mean(0, keepdims=True)
    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    XY = Xc @ Vt[:2].T     # (1+K) x 2
    plt.figure(figsize=(5.8,4.2))
    # phrases
    for i,(lab,grp) in enumerate(zip(labels, groups), start=1):
        c = BLUE if grp=="important" else RED
        plt.scatter(XY[i,0], XY[i,1], s=36, c=c)
        plt.text(XY[i,0], XY[i,1], lab, fontsize=9, ha="center", va="bottom")
    # v
    plt.scatter(XY[0,0], XY[0,1], s=160, marker="*", edgecolor="k", c=GREEN, zorder=5)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png); plt.savefig(out_pdf); plt.close()
    # also save coords as CSV (논문 부록용)
    return XY

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_pth", required=True)
    ap.add_argument("--important", required=True, help="comma-separated phrases")
    ap.add_argument("--distractor", default="", help="comma-separated phrases")
    ap.add_argument("--tokenizer_model", default="bert-base-uncased")
    ap.add_argument("--templates", default="The vehicle made a {PHRASE} at the intersection.;Traffic violation: {PHRASE}.;Dashcam incident: {PHRASE} by the ego vehicle.;The other vehicle made a {PHRASE} near the crosswalk.;Safety report states {PHRASE} during a green signal.")
    ap.add_argument("--context_mode", choices=["delta","plain"], default="delta")
    ap.add_argument("--whiten", action="store_true", help="mean-center embeddings E")
    ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--out_dir", default="figs_paper")
    ap.add_argument("--save_prefix", default="paper")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    set_style(); ensure_dir(args.out_dir)

    # load
    sd = torch.load(args.ckpt_pth, map_location="cpu")
    if not isinstance(sd, dict):
        raise ValueError("Checkpoint must be a state_dict dict.")
    E = load_E(sd)                       # [V,D]
    if args.whiten:
        E = E - E.mean(dim=0, keepdim=True)
    v = get_v(sd)                        # [D]
    if v.numel() != E.shape[1]:
        raise ValueError(f"Dim mismatch: v={v.numel()} vs E_dim={E.shape[1]}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer_model)
    vocab_size = E.shape[0]

    # phrases
    imp = [p.strip() for p in args.important.split(",") if p.strip()]
    dis = [p.strip() for p in args.distractor.split(",") if p.strip()]
    phrases = imp + dis
    groups  = ["important"]*len(imp) + ["distractor"]*len(dis)
    templates = [t.strip() for t in args.templates.split(";") if t.strip()]

    # compute context vectors & scores
    Nv = v / (v.norm() + 1e-9)
    ctx_vecs = []
    ctx_scores = []
    for p in phrases:
        C = context_vec_for_phrase(p, templates, tok, E, vocab_size, mode=args.context_mode)
        ctx_vecs.append(C)
        if C is None or torch.isnan(C).any():
            ctx_scores.append(np.nan)
        else:
            ctx_scores.append(float(torch.dot(C, Nv)))

    # keep valid
    valid = [i for i,c in enumerate(ctx_scores) if np.isfinite(c) and ctx_vecs[i] is not None]
    phrases = [phrases[i] for i in valid]
    groups  = [groups[i]  for i in valid]
    ctx_vecs= [ctx_vecs[i] for i in valid]
    ctx_scores = [ctx_scores[i] for i in valid]

    # save CSV of context scores
    csv_path = os.path.join(args.out_dir, f"{args.save_prefix}_context_scores.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["phrase","group","context_cos"])
        for p,g,s in zip(phrases, groups, ctx_scores):
            w.writerow([p,g,s])
    print(f"[Saved] {csv_path}")

    # metrics for caption
    order = np.argsort(-np.array(ctx_scores))
    K = min(args.K, len(order))
    top_groups = [groups[i] for i in order[:K]]
    p_at_k = sum(g=="important" for g in top_groups) / K if K>0 else float("nan")
    imp_scores = np.array([s for s,g in zip(ctx_scores, groups) if g=="important"])
    dis_scores = np.array([s for s,g in zip(ctx_scores, groups) if g=="distractor"])
    delta = imp_scores.mean() - dis_scores.mean() if len(imp_scores)>0 and len(dis_scores)>0 else float("nan")
    pval = permutation_pvalue(imp_scores, dis_scores) if np.isfinite(delta) else float("nan")
    print(f"[Summary] P@{K} (context cosine): {p_at_k:.3f}")
    print(f"[Permutation test] mean_imp={imp_scores.mean():.4f}, mean_dist={dis_scores.mean():.4f}, Δ={delta:.4f}, p={pval:.5g}")

    # dump summary text for copy-paste to paper
    with open(os.path.join(args.out_dir, f"{args.save_prefix}_caption_stats.txt"), "w") as f:
        f.write(f"P@{K} = {p_at_k:.3f}\n")
        f.write(f"mean_imp = {imp_scores.mean():.4f}\nmean_dist = {dis_scores.mean():.4f}\nΔ = {delta:.4f}\n")
        f.write(f"permutation p-value = {pval:.5g}\n")
    print("[Saved] caption stats")

    # Top-K selection
    top_idx = order[:K]
    top_phr = [phrases[i] for i in top_idx]
    top_grp = [groups[i]  for i in top_idx]
    top_scr = [ctx_scores[i] for i in top_idx]
    top_vec = [ctx_vecs[i]   for i in top_idx]

    # Fig A: bar
    plot_bar_topk(
        top_phr, top_scr, top_grp,
        out_png=os.path.join(args.out_dir, f"{args.save_prefix}_knn_bar_context.png"),
        out_pdf=os.path.join(args.out_dir, f"{args.save_prefix}_knn_bar_context.pdf"),
        title=f"Top-{K} nearest (context cosine)"
    )
    print("[Saved] Top-K bar (context)")

    # Fig B: local PCA
    XY = plot_local_pca(
        v=v, means=top_vec, labels=top_phr, groups=top_grp,
        out_png=os.path.join(args.out_dir, f"{args.save_prefix}_knn_pca.png"),
        out_pdf=os.path.join(args.out_dir, f"{args.save_prefix}_knn_pca.pdf"),
        title="Local 2D map (PCA on {v ∪ Top-K})"
    )
    # save PCA coords (부록/재현용)
    with open(os.path.join(args.out_dir, f"{args.save_prefix}_knn_pca_coords.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["label","group","x","y"])
        w.writerow(["<v>","soft", XY[0,0], XY[0,1]])
        for lab,grp,pt in zip(top_phr, top_grp, XY[1:]):
            w.writerow([lab, grp, float(pt[0]), float(pt[1])])
    print("[Saved] PCA coords CSV")

if __name__ == "__main__":
    main()
