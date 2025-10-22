#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from matplotlib.cm import get_cmap

def choose_best_k(Xz, Ks):
    sil_vals, db_vals, ch_vals, inertia = [], [], [], []
    for K in Ks:
        km = KMeans(n_clusters=K, n_init=50, random_state=42)
        y = km.fit_predict(Xz)
        inertia.append(km.inertia_)
        sil_vals.append(silhouette_score(Xz, y))
        db_vals.append(davies_bouldin_score(Xz, y))
        ch_vals.append(calinski_harabasz_score(Xz, y))

    # 저장용 표
    metrics = pd.DataFrame({
        "K": Ks,
        "Silhouette(↑)": np.round(sil_vals, 3),
        "Davies–Bouldin(↓)": np.round(db_vals, 3),
        "Calinski–Harabasz(↑)": np.round(ch_vals, 1),
        "Inertia": np.round(inertia, 1),
    })

    # 다수결: Silhouette 최대, CH 최대, DB 최소
    from collections import Counter
    k_sil = Ks[int(np.argmax(sil_vals))]
    k_ch  = Ks[int(np.argmax(ch_vals))]
    k_db  = Ks[int(np.argmin(db_vals))]
    bestK = sorted(Counter([k_sil, k_ch, k_db]).items(),
                   key=lambda kv: (-kv[1], kv[0]))[0][0]
    votes = (k_sil, k_ch, k_db)
    return bestK, votes, metrics, (sil_vals, db_vals, ch_vals, inertia)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords_csv", default="figs_paper_3/paper_delta_w_knn_pca_coords.csv")
    ap.add_argument("--out_dir", default="cluster_out")
    ap.add_argument("--include_distractor", action="store_true",
                    help="기본은 important만 클러스터링; 이 옵션을 켜면 distractor도 포함")
    ap.add_argument("--k", type=int, default=None,
                    help="강제로 K 지정 (미지정 시 2~6 스윕해 자동선택)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.coords_csv)

    # 대상 선택
    if args.include_distractor:
        data = df[df["group"].isin(["important","distractor"])].copy()
    else:
        data = df[df["group"]=="important"].copy()
    X = data[["x","y"]].to_numpy()
    labels_txt = data["label"].tolist()

    if len(X) < 3:
        raise SystemExit("클러스터링에 최소 3개 이상의 포인트가 필요합니다.")

    # 스케일링
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    # K 결정
    if args.k is None:
        Ks = list(range(2, min(6, len(X)) + 1))
        bestK, votes, metrics, curves = choose_best_k(Xz, Ks)
        metrics.to_csv(os.path.join(args.out_dir, "kmeans_metrics.csv"), index=False)

        # 지표 곡선 저장
        sil_vals, db_vals, ch_vals, inertia = curves
        def line(x, y, title, ylabel, fname, maximize=True):
            plt.figure(figsize=(6.6,4.6))
            plt.plot(x, y, marker="o", lw=2)
            idx = int(np.argmax(y) if maximize else np.argmin(y))
            plt.scatter([x[idx]],[y[idx]], zorder=3)
            plt.title(title); plt.xlabel("K"); plt.ylabel(ylabel)
            plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, fname), dpi=300)
            plt.close()

        line(Ks, sil_vals, "Silhouette vs K", "Silhouette", "silhouette_vs_k.png", True)
        line(Ks, ch_vals,  "Calinski–Harabasz vs K", "Calinski–Harabasz", "ch_vs_k.png", True)
        line(Ks, db_vals,  "Davies–Bouldin vs K", "Davies–Bouldin", "db_vs_k.png", False)
        line(Ks, inertia,  "Inertia (Elbow) vs K", "Inertia", "inertia_vs_k.png", False)
        K = bestK
        subtitle = f"(auto; votes Sil/CH/DB={votes})"
    else:
        K = int(args.k)
        subtitle = "(manual)"

    # 최종 k-means
    km = KMeans(n_clusters=K, n_init=50, random_state=42).fit(Xz)
    centers = scaler.inverse_transform(km.cluster_centers_)
    labs = km.labels_

    # 산점도 (레이블을 마커 우측에)
    cmap = get_cmap("tab10")
    plt.figure(figsize=(7.2, 5.8))
    for k in range(K):
        m = labs == k
        plt.scatter(X[m,0], X[m,1], s=52, color=cmap(k), label=f"cluster {k+1}")
    # plt.scatter(centers[:,0], centers[:,1], s=170, marker="X",
    #             color="black", edgecolor="white", lw=0.8, label="centroid")

    dx = 0.02 * (X[:,0].max() - X[:,0].min() if len(X)>0 else 1.0)
    # for (x, y, t) in zip(X[:,0], X[:,1], labels_txt):
    #     plt.text(x + dx, y, t, fontsize=9, va="center", ha="left")

    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"K-means on {'important(+distractor)' if args.include_distractor else 'important'} "
              f"(K={K}) {subtitle}")
    plt.legend(frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "kmeans_scatter.png"), dpi=300)
    plt.close()

    # 할당 결과 저장
    out_df = data.copy()
    out_df["kmeans_cluster"] = labs
    out_df.to_csv(os.path.join(args.out_dir, "kmeans_assignments.csv"), index=False)
    print(f"[Saved] {os.path.join(args.out_dir, 'kmeans_scatter.png')}")
    print(f"[Saved] {os.path.join(args.out_dir, 'kmeans_assignments.csv')}")

if __name__ == "__main__":
    main()
