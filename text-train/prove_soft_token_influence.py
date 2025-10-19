import os, argparse, random, csv
from typing import List, Dict, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer
from sklearn.manifold import TSNE
from sklearn.utils import check_random_state
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------- helpers -----------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_E_from_ckpt(sd: dict) -> torch.Tensor:
    key = "encoder.embeddings.word_embeddings.weight"
    if key not in sd:
        raise ValueError("No 'encoder.embeddings.word_embeddings.weight' in checkpoint.")
    return sd[key].detach().clone()

def derive_v_from_regressor(sd: dict) -> torch.Tensor:
    if ("regressor.0.weight" not in sd) or ("regressor.2.weight" not in sd):
        raise ValueError("Need 'regressor.0.weight' and 'regressor.2.weight' to derive v.")
    W0 = sd["regressor.0.weight"]  # [H,D]
    W2 = sd["regressor.2.weight"]  # [2,H] or [H,2]
    if W2.dim() == 2 and W2.shape[0] != 2 and W2.shape[1] == 2:
        W2 = W2.t()
    dv_minus_ov = W2[0] - W2[1]    # [H]
    v = W0.t() @ dv_minus_ov       # [D]
    return v / (v.norm() + 1e-9)

def get_v_from_ckpt(sd: dict) -> torch.Tensor:
    for k in ["soft_token", "v", "latent", "soft", "vector"]:
        if k in sd and isinstance(sd[k], torch.Tensor) and sd[k].ndim in (1,2):
            v = sd[k].squeeze()
            return v / (v.norm() + 1e-9)
    return derive_v_from_regressor(sd)

def phrase_token_ids(tok, text: str, vocab_size: int) -> List[int]:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    unk = getattr(tok, "unk_token_id", None)
    return [i for i in ids if 0 <= i < vocab_size and (unk is None or i != unk)]

def phrase_mean(E: torch.Tensor, ids: List[int]) -> torch.Tensor:
    if len(ids) == 0: return None
    return E[ids].mean(dim=0)

def permutation_pvalue(a_vals: np.ndarray, b_vals: np.ndarray, n_perm: int = 50000, seed: int = 42) -> float:
    rng = check_random_state(seed)
    a = np.asarray(a_vals); b = np.asarray(b_vals)
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b]); nA = len(a); count = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        a_, b_ = pool[:nA], pool[nA:]
        if abs(a_.mean() - b_.mean()) >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)

def precision_at_k(scores: List[float], groups: List[str], k: int) -> float:
    order = np.argsort(-np.array(scores))
    top = [groups[i] for i in order[:min(k, len(order))]]
    return float(sum(g == "important" for g in top)) / max(1, len(top))

# ----------------- context features -----------------
def sentence_ids(tok, text: str, vocab_size: int) -> List[int]:
    return phrase_token_ids(tok, text, vocab_size)

def sentence_mean(E: torch.Tensor, ids: List[int]) -> torch.Tensor:
    if len(ids) == 0: return None
    return E[ids].mean(dim=0)

def context_vector_for_phrase(phrase: str, templates: List[str], tok, E: torch.Tensor,
                              vocab_size: int, mode: str = "delta") -> torch.Tensor:
    vecs = []
    for tpl in templates:
        with_phrase = tpl.replace("{PHRASE}", phrase)
        ids_w = sentence_ids(tok, with_phrase, vocab_size)
        mw = sentence_mean(E, ids_w)
        if mw is None: 
            continue
        if mode == "delta":
            without = tpl.replace("{PHRASE}", "").replace("  ", " ").strip()
            ids_0 = sentence_ids(tok, without, vocab_size)
            m0 = sentence_mean(E, ids_0)
            vecs.append(mw if m0 is None else (mw - m0))
        else:
            vecs.append(mw)
    if not vecs: return None
    vctx = torch.stack(vecs, dim=0).mean(dim=0)
    return vctx / (vctx.norm() + 1e-9)

# ----------------- plotting -----------------
def set_paper_style():
    plt.rcParams.update({
        "figure.dpi": 180, "savefig.dpi": 300,
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "legend.fontsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42
    })

def plot_bar_ranked(data: List[Tuple[str,float,str]], title: str, out_png: str, out_pdf: str):
    phrases = [d[0] for d in data]; scores = [d[1] for d in data]; groups = [d[2] for d in data]
    colors = ["#1f77b4" if g=="important" else "#d62728" for g in groups]
    plt.figure(figsize=(max(6, min(18, 0.24*len(phrases))), 5))
    x = np.arange(len(phrases))
    plt.bar(x, scores, edgecolor="black", linewidth=0.3, color=colors)
    plt.xticks(x, phrases, rotation=75, ha="right"); plt.ylabel("cosine(v, phrase mean)"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png); plt.savefig(out_pdf); plt.close()

def plot_heatmap(phrases: List[str], cos_scores: List[float], dot_scores: List[float],
                 groups: List[str], title: str, out_png: str, out_pdf: str):
    order = np.argsort(-np.array(cos_scores))
    phrases_ord = [phrases[i] for i in order]; groups_ord = [groups[i] for i in order]
    M = np.vstack([np.array(cos_scores)[order], np.array(dot_scores)[order]]).T
    M = np.nan_to_num(M, nan=np.nanmedian(M))
    Mn = (M - M.min(0, keepdims=True)) / (M.max(0, keepdims=True) - M.min(0, keepdims=True) + 1e-9)
    plt.figure(figsize=(6.5, max(4, 0.26*len(phrases_ord))))
    plt.imshow(Mn, aspect="auto", interpolation="nearest")
    plt.yticks(np.arange(len(phrases_ord)), phrases_ord); plt.xticks([0,1], ["cosine", "dot"])
    for i,g in enumerate(groups_ord):
        c = "#1f77b4" if g=="important" else "#d62728"; plt.scatter([-0.6], [i], s=20, c=c)
    plt.colorbar(label="normalized score"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png); plt.savefig(out_pdf); plt.close()

def plot_tsne(v: torch.Tensor, phrase_means: List[torch.Tensor], labels: List[str],
              groups: List[str], title: str, out_png: str, out_pdf: str, seed: int = 42):
    if not phrase_means: return
    X = torch.vstack([v.unsqueeze(0)] + phrase_means).cpu().numpy()
    rs = np.random.RandomState(seed); perpl = max(5, min(35, max(5, len(labels)//2)))
    tsne = TSNE(n_components=2, init="pca", perplexity=perpl, random_state=rs)
    XY = tsne.fit_transform(X)
    plt.figure(figsize=(6.8,5.2))
    for i,(lab,grp) in enumerate(zip(labels, groups), start=1):
        c = "#1f77b4" if grp=="important" else "#d62728"; plt.scatter(XY[i,0], XY[i,1], s=26, alpha=0.85, c=c)
    plt.scatter(XY[0,0], XY[0,1], s=120, marker="*", edgecolor="k", facecolor="#2ca02c", label="<soft token v>")
    d2 = ((XY[1:,0]-XY[0,0])**2 + (XY[1:,1]-XY[0,1])**2); show_idx = np.argsort(d2)[:min(18, len(labels))]
    for j in show_idx: plt.text(XY[j+1,0], XY[j+1,1], labels[j], fontsize=8)
    plt.title(title); plt.tight_layout(); plt.savefig(out_png); plt.savefig(out_pdf); plt.close()

def plot_knn_bar(phrases, cos_scores, groups, out_png, out_pdf, title="Top-K nearest (cosine)"):
    order = np.argsort(-np.array(cos_scores))
    phrases = [phrases[i] for i in order]; scores  = [cos_scores[i] for i in order]; groups  = [groups[i] for i in order]
    colors  = ["#1f77b4" if g=="important" else "#d62728" for g in groups]
    plt.figure(figsize=(max(5, 0.28*len(phrases)), 3.8)); x = np.arange(len(phrases))
    plt.bar(x, scores, color=colors, edgecolor="black", linewidth=0.3)
    plt.xticks(x, phrases, rotation=65, ha="right"); plt.ylabel("cosine(v, phrase mean)"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png); plt.savefig(out_pdf); plt.close()

def plot_knn_pca(v, means, labels, groups, out_png, out_pdf, title="Local map (PCA)"):
    X = torch.vstack([v.unsqueeze(0)] + [m.unsqueeze(0) for m in means]).cpu().numpy()
    Xc = X - X.mean(0, keepdims=True); U,S,Vt = np.linalg.svd(Xc, full_matrices=False); XY = Xc @ Vt[:2].T
    plt.figure(figsize=(5.6,4.4))
    for i,(lab,grp) in enumerate(zip(labels, groups), start=1):
        c = "#1f77b4" if grp=="important" else "#d62728"; plt.scatter(XY[i,0], XY[i,1], s=34, c=c); plt.text(XY[i,0], XY[i,1], lab, fontsize=9, ha="center", va="bottom")
    plt.scatter(XY[0,0], XY[0,1], s=150, marker="*", edgecolor="k", c="#2ca02c", zorder=5)
    plt.title(title); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout(); plt.savefig(out_png); plt.savefig(out_pdf); plt.close()

def plot_knn_polar(phrases, cos_scores, groups, out_png, out_pdf, title="Similarity polar"):
    r = np.clip(np.array(cos_scores), 0, None); theta = np.linspace(0, 2*np.pi, len(r), endpoint=False)
    colors = ["#1f77b4" if g=="important" else "#d62728" for g in groups]
    fig = plt.figure(figsize=(5,5)); ax = fig.add_subplot(111, projection="polar")
    ax.scatter(theta, r, c=colors, s=48)
    for th, rr, lab in zip(theta, r, phrases): ax.text(th, rr+0.03, lab, fontsize=8, ha="center", va="center")
    ax.set_yticklabels([]); ax.set_xticklabels([]); ax.set_title(title); plt.tight_layout()
    plt.savefig(out_png); plt.savefig(out_pdf); plt.close()

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_pth", type=str, required=True)
    ap.add_argument("--important_tokens", type=str, required=True)
    ap.add_argument("--distractor_tokens", type=str, default="")
    ap.add_argument("--tokenizer_model", type=str, default="bert-base-uncased")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--out_dir", type=str, default="softtoken_figs")
    ap.add_argument("--save_prefix", type=str, default="softv")
    ap.add_argument("--knn_k", type=int, default=12)
    ap.add_argument("--context_templates", type=str,
        default="The vehicle made a {PHRASE} at the intersection.;Traffic violation: {PHRASE}.;The dashcam reports {PHRASE}.;Incident category: {PHRASE}.;A car performed {PHRASE} in traffic.")
    ap.add_argument("--context_mode", type=str, default="delta", choices=["delta","plain"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    set_paper_style(); ensure_dir(args.out_dir)

    # 1) Load checkpoint, E, v
    sd = torch.load(args.ckpt_pth, map_location="cpu")
    if not isinstance(sd, dict): raise ValueError("ckpt must be a state_dict (dict).")
    E = load_E_from_ckpt(sd)   # [V,D]
    v = get_v_from_ckpt(sd)    # [D]
    if v.shape[0] != E.shape[1]: raise ValueError(f"Dim mismatch: v_dim={v.shape[0]} vs E_dim={E.shape[1]}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer_model)
    if hasattr(tok, "vocab") and len(tok.vocab) != E.shape[0]:
        print(f"[WARN] tokenizer vocab={len(tok.vocab)} vs embeddings V={E.shape[0]} → ckpt tokenizer mismatch risk")
    V, D = E.shape; vocab_size = V; Wout = E  # tie surrogate

    # 2) Pre-normalize once
    Nv = v / (v.norm() + 1e-9)
    NE = E / (E.norm(dim=1, keepdim=True) + 1e-9)

    # 3) Phrases
    phrases_imp = [p.strip() for p in args.important_tokens.split(",") if p.strip()]
    phrases_dis = [p.strip() for p in args.distractor_tokens.split(",") if p.strip()]
    phrases = phrases_imp + phrases_dis
    groups  = ["important"]*len(phrases_imp) + ["distractor"]*len(phrases_dis)
    if not phrases: raise ValueError("No phrases provided.")

    # 4) Tokenize & base scores
    phrase2ids: Dict[str, List[int]] = {p: phrase_token_ids(tok, p, vocab_size) for p in phrases}
    means, cos_scores, dot_scores = [], [], []
    for p in phrases:
        ids = phrase2ids[p]; m = phrase_mean(E, ids); means.append(m)
        if m is None:
            cos_scores.append(np.nan); dot_scores.append(np.nan)
        else:
            cos_scores.append(float((NE[ids] @ Nv).mean()))
            dot_scores.append(float((Wout[ids] @ v).mean()))

    # 5) Keep only valid phrases
    valid_idx = [i for i,(c,d,m) in enumerate(zip(cos_scores, dot_scores, means))
                 if np.isfinite(c) and np.isfinite(d) and m is not None]
    if len(valid_idx) == 0: raise RuntimeError("No valid phrases after tokenization.")
    phrases = [phrases[i] for i in valid_idx]
    groups  = [groups[i]  for i in valid_idx]
    means   = [means[i]   for i in valid_idx]
    cos_scores = [cos_scores[i] for i in valid_idx]
    dot_scores = [dot_scores[i] for i in valid_idx]

    # 6) Context scoring (after filtering for consistency)
    templates = [t.strip() for t in args.context_templates.split(";") if t.strip()]
    ctx_scores = []
    for p in phrases:
        C = context_vector_for_phrase(p, templates, tok, E, vocab_size, mode=args.context_mode)
        if C is None or torch.isnan(C).any(): ctx_scores.append(np.nan)
        else: ctx_scores.append(float(torch.dot(C, Nv)))
    ctx_valid = [i for i,s in enumerate(ctx_scores) if np.isfinite(s)]
    phrases_ctx = [phrases[i] for i in ctx_valid]
    groups_ctx  = [groups[i]  for i in ctx_valid]
    ctx_scores  = [ctx_scores[i] for i in ctx_valid]

    # 7) Save CSVs
    csv_path = os.path.join(args.out_dir, f"{args.save_prefix}_phrase_scores.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["phrase","group","n_subwords","cosine","dot"])
        for p,g in zip(phrases, groups):
            w.writerow([p, g, len(phrase2ids[p]), cos_scores[phrases.index(p)], dot_scores[phrases.index(p)]])
    print(f"[Saved] {csv_path}")

    if len(ctx_scores) > 0:
        ctx_csv = os.path.join(args.out_dir, f"{args.save_prefix}_context_scores.csv")
        with open(ctx_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["phrase","group","context_cos"])
            for p,g,s in zip(phrases_ctx, groups_ctx, ctx_scores): w.writerow([p,g,s])
        print(f"[Saved] {ctx_csv}")

    # 8) Tests & summaries
    if any(g=="important" for g in groups) and any(g=="distractor" for g in groups):
        imp_cos = np.array([c for c,g in zip(cos_scores, groups) if g=="important"])
        dis_cos = np.array([c for c,g in zip(cos_scores, groups) if g=="distractor"])
        imp_dot = np.array([d for d,g in zip(dot_scores, groups) if g=="important"])
        dis_dot = np.array([d for d,g in zip(dot_scores, groups) if g=="distractor"])
        p_cos = permutation_pvalue(imp_cos, dis_cos, n_perm=20000, seed=args.seed)
        p_dot = permutation_pvalue(imp_dot, dis_dot, n_perm=20000, seed=args.seed)
        print(f"[Permutation] cosine: mean_imp={imp_cos.mean():.4f}, mean_dist={dis_cos.mean():.4f}, p={p_cos:.5f}")
        print(f"[Permutation] dot   : mean_imp={imp_dot.mean():.4f}, mean_dist={dis_dot.mean():.4f}, p={p_dot:.5f}")
        print(f"[Summary] P@{args.knn_k} (phrase-avg cosine): {precision_at_k(cos_scores, groups, args.knn_k):.2f}")
    if len(ctx_scores) > 0:
        print(f"[Summary] P@{args.knn_k} (context cosine): {precision_at_k(ctx_scores, groups_ctx, args.knn_k):.2f}")

    # 9) Plots (global)
    ranked = sorted([(p,c,g) for p,c,g in zip(phrases, cos_scores, groups)], key=lambda x: -x[1])
    plot_bar_ranked(ranked, "Phrase relevance to soft token (cosine)",
                    os.path.join(args.out_dir, f"{args.save_prefix}_phrase_scores_bar_cosine.png"),
                    os.path.join(args.out_dir, f"{args.save_prefix}_phrase_scores_bar_cosine.pdf"))
    plot_heatmap(phrases, cos_scores, dot_scores, groups,
                 "Phrase scores (cosine & dot)",
                 os.path.join(args.out_dir, f"{args.save_prefix}_phrase_scores_heatmap.png"),
                 os.path.join(args.out_dir, f"{args.save_prefix}_phrase_scores_heatmap.pdf"))
    plot_tsne(v, [m.unsqueeze(0) for m in means], phrases, groups,
              "t-SNE of soft token and phrase means",
              os.path.join(args.out_dir, f"{args.save_prefix}_tsne.png"),
              os.path.join(args.out_dir, f"{args.save_prefix}_tsne.pdf"),
              seed=args.seed)

    # 10) Paper-ready Top-K (cosine)
    knn_idx = np.argsort(-np.array(cos_scores))[:min(args.knn_k, len(cos_scores))]
    knn_phrases = [phrases[i] for i in knn_idx]
    knn_groups  = [groups[i]  for i in knn_idx]
    knn_scores  = [cos_scores[i] for i in knn_idx]
    knn_means   = [means[i]   for i in knn_idx]
    plot_knn_bar(knn_phrases, knn_scores, knn_groups,
                 os.path.join(args.out_dir, f"{args.save_prefix}_knn_bar.png"),
                 os.path.join(args.out_dir, f"{args.save_prefix}_knn_bar.pdf"),
                 title=f"Top-{len(knn_idx)} nearest phrases (cosine)")
    plot_knn_pca(v, knn_means, knn_phrases, knn_groups,
                 os.path.join(args.out_dir, f"{args.save_prefix}_knn_pca.png"),
                 os.path.join(args.out_dir, f"{args.save_prefix}_knn_pca.pdf"),
                 title="Local 2D map (PCA on {v ∪ Top-K})")
    plot_knn_polar(knn_phrases, knn_scores, knn_groups,
                   os.path.join(args.out_dir, f"{args.save_prefix}_knn_polar.png"),
                   os.path.join(args.out_dir, f"{args.save_prefix}_knn_polar.pdf"),
                   title="Cosine similarity to v")
    knn_csv = os.path.join(args.out_dir, f"{args.save_prefix}_knn_top{len(knn_idx)}.csv")
    with open(knn_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["rank","phrase","group","cosine"])
        for r,(p,g,s) in enumerate(sorted(zip(knn_phrases, knn_groups, knn_scores), key=lambda x:-x[2]), start=1):
            w.writerow([r,p,g,s])
    print(f"[Saved] KNN plots & table → {knn_csv}")

    # 11) Paper-ready Top-K (context) if available
    if len(ctx_scores) > 0:
        order = np.argsort(-np.array(ctx_scores))
        k = min(args.knn_k, len(order))
        phrases_ord = [phrases_ctx[i] for i in order][:k]
        groups_ord  = [groups_ctx[i]  for i in order][:k]
        scores_ord  = [ctx_scores[i]  for i in order][:k]
        plot_knn_bar(phrases_ord, scores_ord, groups_ord,
                     os.path.join(args.out_dir, f"{args.save_prefix}_knn_bar_context.png"),
                     os.path.join(args.out_dir, f"{args.save_prefix}_knn_bar_context.pdf"),
                     title=f"Top-{k} nearest (context cosine)")

if __name__ == "__main__":
    main()
