#!/usr/bin/env python3
"""
run_clustering.py
------------------
Phase II: Unsupervised Clustering Analysis (K-Means & Agglomerative Hierarchical)

Author: Senior Data Scientist & Machine Learning Engineer
Project: Digital Device Usage Pattern Analysis Using Machine Learning
Strict Random Seed: 42

This script:
1. Evaluates clustering separately on:
   - Full Augmented Dataset (N=150: 54 genuine + 96 synthetic)
   - Genuine-Only Dataset (N=54 genuine student responses)
2. Compares Configuration A (Atomic, 18 features) and Configuration B (Composite, 14 features).
3. Strictly isolates features from targets (zero target leakage).
4. Standardizes feature vectors using StandardScaler prior to clustering.
5. Performs K-Means clustering across K in [2, 6] with K-Means++ and n_init=20.
6. Evaluates Agglomerative Hierarchical Clustering across Ward, Complete, and Average linkages.
7. Generates hierarchical dendrograms and silhouette profile plots.
8. Computes external validation metrics (Adjusted Rand Index, Normalized Mutual Information, Contingency Tables)
   against ground-truth impact_category ONLY post-hoc as an interpretability check.
9. Derives behavioral profiles and student archetypes for discovered clusters.
10. Generates publication-quality diagnostic visualizations saved in reports/phase_ii/.
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.metrics.cluster import contingency_matrix
from scipy.cluster.hierarchy import dendrogram, linkage

# Plot style setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "phase_ii"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

WORKSPACE_REPORTS_DIR = PROJECT_ROOT.parent / "reports" / "phase_ii"
WORKSPACE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_A = [
    'gender_encoded',
    'year_First year', 'year_Second year', 'year_Third year', 'year_Fourth year', 'year_Other',
    'daily_device_hours', 'check_frequency',
    'phone_with_family', 'conversation_interruption', 'social_complaints', 'stay_away_difficulty',
    'activity_Communication and messaging', 'activity_Educational activities',
    'activity_Online gaming', 'activity_Other', 'activity_Social media',
    'activity_Watching videos and entertainment'
]

CONFIG_B = [
    'gender_encoded',
    'year_First year', 'year_Second year', 'year_Third year', 'year_Fourth year', 'year_Other',
    'social_conflict_score', 'digital_dependency_score',
    'activity_Communication and messaging', 'activity_Educational activities',
    'activity_Online gaming', 'activity_Other', 'activity_Social media',
    'activity_Watching videos and entertainment'
]

CLASS_NAMES = ['Low Impact', 'Moderate Impact', 'High Impact']


def run_clustering_suite():
    print("[INFO] Loading datasets from data/processed/ ...")
    df_150 = pd.read_csv(DATA_PROCESSED / "full_150_corrected_features.csv")
    df_54 = pd.read_csv(DATA_PROCESSED / "genuine_54_corrected_features.csv")

    datasets = [
        ('Full 150 Augmented', df_150),
        ('Genuine 54 Only', df_54)
    ]

    configs = [
        ('Config A (Atomic, 18 feats)', CONFIG_A),
        ('Config B (Composite, 14 feats)', CONFIG_B)
    ]

    kmeans_records = []
    agg_records = []

    scaler = StandardScaler()

    # ==============================================================================
    # 1. K-MEANS EXPERIMENTS (K = 2 to 6)
    # ==============================================================================
    print("[INFO] Running K-Means parameter sweeps across K in [2, 6]...")
    for ds_name, df in datasets:
        for cfg_name, feats in configs:
            X_scaled = scaler.fit_transform(df[feats])
            y_true = df['impact_category'].values

            for k in range(2, 7):
                km = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=SEED)
                km.fit(X_scaled)
                labels = km.labels_

                sil = silhouette_score(X_scaled, labels)
                ari = adjusted_rand_score(y_true, labels)
                nmi = normalized_mutual_info_score(y_true, labels)

                kmeans_records.append({
                    'Dataset': ds_name,
                    'Configuration': cfg_name,
                    'K': k,
                    'Inertia': float(round(km.inertia_, 2)),
                    'Silhouette': float(round(sil, 4)),
                    'ARI': float(round(ari, 4)),
                    'NMI': float(round(nmi, 4))
                })

    df_kmeans_res = pd.DataFrame(kmeans_records)
    df_kmeans_res.to_csv(REPORTS_DIR / "kmeans_evaluation_summary.csv", index=False)
    print("Saved K-Means summary to reports/phase_ii/kmeans_evaluation_summary.csv")

    # ==============================================================================
    # 2. AGGLOMERATIVE CLUSTERING SWEEPS (Single, Complete, Average, Ward)
    # ==============================================================================
    print("[INFO] Running Agglomerative Hierarchical sweeps across all 4 linkages...")
    for ds_name, df in datasets:
        for cfg_name, feats in configs:
            X_scaled = scaler.fit_transform(df[feats])
            y_true = df['impact_category'].values

            for linkage_type in ['single', 'complete', 'average', 'ward']:
                for k in range(2, 7):
                    agg = AgglomerativeClustering(n_clusters=k, linkage=linkage_type)
                    agg.fit(X_scaled)
                    labels = agg.labels_

                    sil = silhouette_score(X_scaled, labels)
                    ari = adjusted_rand_score(y_true, labels)
                    nmi = normalized_mutual_info_score(y_true, labels)

                    agg_records.append({
                        'Dataset': ds_name,
                        'Configuration': cfg_name,
                        'Linkage': linkage_type,
                        'K': k,
                        'Silhouette': float(round(sil, 4)),
                        'ARI': float(round(ari, 4)),
                        'NMI': float(round(nmi, 4))
                    })

    df_agg_res = pd.DataFrame(agg_records)
    df_agg_res.to_csv(REPORTS_DIR / "agglomerative_evaluation_summary.csv", index=False)
    df_agg_res.to_csv(WORKSPACE_REPORTS_DIR / "agglomerative_evaluation_summary.csv", index=False)
    df_kmeans_res.to_csv(WORKSPACE_REPORTS_DIR / "kmeans_evaluation_summary.csv", index=False)
    print("Saved Agglomerative summary (Single, Complete, Average, Ward) to reports/phase_ii/agglomerative_evaluation_summary.csv")

    # ==============================================================================
    # FIGURE 5: K-MEANS ELBOW & SILHOUETTE DUAL PLOT
    # ==============================================================================
    print("[INFO] Generating Figure 5: K-Means Inertia Elbow & Silhouette Curves...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Top row: Inertia (Full 150 vs Genuine 54)
    # Bottom row: Silhouette (Full 150 vs Genuine 54)
    for col_idx, (ds_name, _) in enumerate(datasets):
        ax_inertia = axes[0, col_idx]
        ax_sil = axes[1, col_idx]

        sub = df_kmeans_res[df_kmeans_res['Dataset'] == ds_name]

        for cfg_name, color, marker in [
            ('Config A (Atomic, 18 feats)', '#1f77b4', 'o'),
            ('Config B (Composite, 14 feats)', '#2ca02c', 's')
        ]:
            cfg_sub = sub[sub['Configuration'] == cfg_name].sort_values('K')
            ax_inertia.plot(
                cfg_sub['K'], cfg_sub['Inertia'],
                label=cfg_name, color=color, marker=marker, linewidth=2.5, markersize=8
            )
            ax_sil.plot(
                cfg_sub['K'], cfg_sub['Silhouette'],
                label=cfg_name, color=color, marker=marker, linewidth=2.5, markersize=8
            )

        ax_inertia.set_title(f"{ds_name} - Inertia Elbow Curve", fontsize=12, fontweight='bold')
        ax_inertia.set_xlabel("Number of Clusters (K)")
        ax_inertia.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
        ax_inertia.set_xticks(range(2, 7))
        ax_inertia.legend(fontsize=9)
        ax_inertia.grid(True, linestyle=':', alpha=0.6)

        ax_sil.set_title(f"{ds_name} - Silhouette Score vs K", fontsize=12, fontweight='bold')
        ax_sil.set_xlabel("Number of Clusters (K)")
        ax_sil.set_ylabel("Mean Silhouette Coefficient")
        ax_sil.set_xticks(range(2, 7))
        ax_sil.set_ylim(0.08, 0.32)
        ax_sil.legend(fontsize=9)
        ax_sil.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "05_kmeans_elbow_and_silhouette.png", dpi=300, bbox_inches='tight')
    plt.savefig(WORKSPACE_REPORTS_DIR / "05_kmeans_elbow_and_silhouette.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ==============================================================================
    # FIGURE 6: HIERARCHICAL DENDROGRAMS (Full 150 & Genuine 54 across 4 Linkages)
    # ==============================================================================
    print("[INFO] Generating Figure 6: Hierarchical Clustering Dendrograms (Single, Complete, Average, Ward)...")
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    for row_idx, (ds_name, df) in enumerate(datasets):
        # Use Config B for clean dendrogram representation
        X_scaled = scaler.fit_transform(df[CONFIG_B])
        for col_idx, linkage_method in enumerate(['single', 'complete', 'average', 'ward']):
            ax = axes[row_idx, col_idx]
            Z = linkage(X_scaled, method=linkage_method, metric='euclidean')
            
            # Truncate dendrogram for Full 150 to maintain legibility
            truncate_mode = 'lastp' if len(df) > 60 else None
            p_val = 20 if len(df) > 60 else 30

            dendrogram(
                Z,
                ax=ax,
                truncate_mode=truncate_mode,
                p=p_val,
                show_leaf_counts=True,
                color_threshold=0.7 * max(Z[:, 2]),
                above_threshold_color='#7f7f7f'
            )
            ax.set_title(f"{ds_name} | {linkage_method.capitalize()} Linkage", fontsize=11, fontweight='bold')
            ax.set_xlabel("Sample Index / Cluster Size")
            ax.set_ylabel("Distance")
            ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "06_hierarchical_dendrograms.png", dpi=300, bbox_inches='tight')
    plt.savefig(WORKSPACE_REPORTS_DIR / "06_hierarchical_dendrograms.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ==============================================================================
    # FIGURE 7: SILHOUETTE SAMPLE PROFILES FOR OPTIMAL CLUSTERS (K=3)
    # ==============================================================================
    print("[INFO] Generating Figure 7: Silhouette Analysis Profiles...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (ds_name, df) in enumerate(datasets):
        ax = axes[idx]
        X_scaled = scaler.fit_transform(df[CONFIG_B])
        k = 3
        km = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=SEED)
        labels = km.fit_predict(X_scaled)
        
        sil_vals = silhouette_samples(X_scaled, labels)
        avg_sil = silhouette_score(X_scaled, labels)

        y_lower = 10
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for i in range(k):
            ith_sil_vals = sil_vals[labels == i]
            ith_sil_vals.sort()
            size_cluster_i = ith_sil_vals.shape[0]
            y_upper = y_lower + size_cluster_i

            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_sil_vals,
                facecolor=colors[i],
                edgecolor=colors[i],
                alpha=0.75,
                label=f"Cluster {i+1} (N={size_cluster_i})"
            )
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i + 1), fontweight='bold')
            y_lower = y_upper + 10

        ax.axvline(x=avg_sil, color="red", linestyle="--", linewidth=1.8, label=f"Mean Sil: {avg_sil:.3f}")
        ax.set_title(f"{ds_name} - Silhouette Plot (K=3, Config B)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Silhouette Coefficient Values")
        ax.set_ylabel("Cluster Assigned Samples")
        ax.set_xlim([-0.15, 0.65])
        ax.set_yticks([])
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "07_clustering_silhouette_analysis.png", dpi=300, bbox_inches='tight')
    plt.savefig(WORKSPACE_REPORTS_DIR / "07_clustering_silhouette_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ==============================================================================
    # FIGURE 8: CONTINGENCY HEATMAPS (CLUSTERS VS GROUND-TRUTH IMPACT CATEGORY)
    # ==============================================================================
    print("[INFO] Generating Figure 8: Cluster Contingency Heatmaps...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for idx, (ds_name, df) in enumerate(datasets):
        ax = axes[idx]
        X_scaled = scaler.fit_transform(df[CONFIG_B])
        k = 3
        km = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=SEED)
        labels = km.fit_predict(X_scaled)

        y_true = df['impact_category']
        contingency = pd.crosstab(
            pd.Series(labels, name="K-Means Discovered Cluster"),
            pd.Series(y_true, name="Ground Truth Impact Category")
        ).reindex(columns=CLASS_NAMES, fill_value=0)
        contingency.index = [f"Cluster {i+1}" for i in range(k)]

        ari = adjusted_rand_score(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels)

        sns.heatmap(
            contingency,
            annot=True,
            fmt='d',
            cmap='YlGnBu',
            cbar=False,
            ax=ax,
            annot_kws={'size': 13, 'weight': 'bold'}
        )
        ax.set_title(
            f"{ds_name} (K=3)\nARI: {ari:.4f} | NMI: {nmi:.4f}",
            fontsize=12, fontweight='bold'
        )
        ax.set_xlabel("True Academic Impact Category", fontweight='bold')
        ax.set_ylabel("Unsupervised Cluster Assignment", fontweight='bold')

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "08_cluster_contingency_heatmaps.png", dpi=300, bbox_inches='tight')
    plt.savefig(WORKSPACE_REPORTS_DIR / "08_cluster_contingency_heatmaps.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ==============================================================================
    # FIGURE 9: BEHAVIORAL PROFILING OF DISCOVERED CLUSTERS (Full 150, K=3)
    # ==============================================================================
    print("[INFO] Generating Figure 9: Discovered Cluster Behavioral Profiles...")
    X_scaled = scaler.fit_transform(df_150[CONFIG_B])
    km = KMeans(n_clusters=3, init='k-means++', n_init=20, random_state=SEED)
    df_150['cluster'] = km.fit_predict(X_scaled) + 1

    profile_vars = [
        'daily_device_hours',
        'check_frequency',
        'social_conflict_score',
        'digital_dependency_score',
        'stay_away_difficulty'
    ]
    
    cluster_means = df_150.groupby('cluster')[profile_vars].mean().round(2)
    print("\nCluster Behavioral Means (Full 150):\n", cluster_means)
    cluster_means.to_csv(REPORTS_DIR / "cluster_behavioral_profiles.csv")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(profile_vars))
    width = 0.25

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    neutral_labels = [
        "Cluster 1",
        "Cluster 2",
        "Cluster 3"
    ]

    for c_id in range(1, 4):
        vals = cluster_means.loc[c_id].values
        offset = (c_id - 2) * width
        bars = ax.bar(x + offset, vals, width, label=neutral_labels[c_id-1], color=colors[c_id-1], alpha=0.85)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold'
            )

    ax.set_title("Behavioral Characteristics Across Discovered Clusters (Full 150, K=3)", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([
        "Daily Device\nHours (1-5)",
        "Smartphone\nChecking (1-5)",
        "Social Conflict\nScore (1-5)",
        "Digital Dependency\nScore (1-5)",
        "Stay-Away\nDifficulty (1-5)"
    ], fontsize=10, fontweight='bold')
    ax.set_ylabel("Mean Behavioral Metric Score", fontweight='bold')
    ax.set_ylim(1.0, 5.0)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "09_cluster_behavioral_profiles.png", dpi=300, bbox_inches='tight')
    plt.savefig(WORKSPACE_REPORTS_DIR / "09_cluster_behavioral_profiles.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ==============================================================================
    # FIGURE 10: 2D PCA VISUALIZATION OF K-MEANS CLUSTERS VS GROUND-TRUTH LABELS
    # ==============================================================================
    print("[INFO] Generating Figure 10: 2D PCA Cluster Visualization & Unexpected Cluster Analysis...")
    X_scaled_150 = scaler.fit_transform(df_150[CONFIG_B])
    km_150 = KMeans(n_clusters=3, init='k-means++', n_init=20, random_state=SEED)
    km_labels_150 = km_150.fit_predict(X_scaled_150) + 1
    sil_samples_150 = silhouette_samples(X_scaled_150, km_labels_150 - 1)

    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled_150)
    centroids_pca = pca.transform(km_150.cluster_centers_)
    var_exp = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: K-Means Discovered Clusters
    cluster_colors = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c'}
    for cid in [1, 2, 3]:
        mask = km_labels_150 == cid
        axes[0].scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=cluster_colors[cid], label=f"Cluster {cid} (N={mask.sum()})",
            alpha=0.75, s=55, edgecolors='white', linewidth=0.5
        )
    axes[0].scatter(
        centroids_pca[:, 0], centroids_pca[:, 1],
        c='black', marker='X', s=200, label='Cluster Centroids', zorder=5
    )
    # Highlight boundary / negative-silhouette observations
    neg_mask = sil_samples_150 < 0
    if neg_mask.any():
        axes[0].scatter(
            X_pca[neg_mask, 0], X_pca[neg_mask, 1],
            facecolors='none', edgecolors='red', s=110, linewidths=1.5,
            label=f"Boundary / Neg. Silhouette (N={neg_mask.sum()})"
        )
    axes[0].set_title(f"K-Means Clusters in 2D PCA Space (Full 150, K=3, Config B)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)")
    axes[0].set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)")
    axes[0].legend(fontsize=9, loc='best')
    axes[0].grid(True, linestyle=':', alpha=0.5)

    # Right: Ground-Truth Academic Impact Category
    impact_colors = {'Low Impact': '#2ca02c', 'Moderate Impact': '#1f77b4', 'High Impact': '#d62728'}
    for cls in CLASS_NAMES:
        mask = df_150['impact_category'].values == cls
        axes[1].scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=impact_colors[cls], label=f"{cls} (N={mask.sum()})",
            alpha=0.75, s=55, edgecolors='white', linewidth=0.5
        )
    axes[1].set_title("Ground-Truth Academic Impact Labels in Same 2D PCA Space", fontsize=12, fontweight='bold')
    axes[1].set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)")
    axes[1].set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)")
    axes[1].legend(fontsize=9, loc='best')
    axes[1].grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "10_kmeans_pca_clusters.png", dpi=300, bbox_inches='tight')
    plt.savefig(WORKSPACE_REPORTS_DIR / "10_kmeans_pca_clusters.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save unexpected / boundary cluster observations for post-processing audit
    df_unexp = df_150[['data_source', 'daily_device_hours', 'check_frequency',
                       'social_conflict_score', 'digital_dependency_score', 'impact_category']].copy()
    df_unexp['row_id'] = df_150.index
    df_unexp['kmeans_cluster'] = km_labels_150
    df_unexp['silhouette_value'] = np.round(sil_samples_150, 4)
    df_unexp_flagged = df_unexp[df_unexp['silhouette_value'] < 0].sort_values('silhouette_value')
    df_unexp_flagged.to_csv(REPORTS_DIR / "unexpected_cluster_observations.csv", index=False)
    df_unexp_flagged.to_csv(WORKSPACE_REPORTS_DIR / "unexpected_cluster_observations.csv", index=False)
    print(f"[INFO] Flagged {len(df_unexp_flagged)} negative-silhouette boundary observations.")

    # ==============================================================================
    # AGGLOMERATIVE POST-PROCESSING BREAKDOWN (All 4 Linkages, K=3, Config B)
    # ==============================================================================
    print("[INFO] Computing Agglomerative cluster size & dominant class breakdown (K=3, Config B)...")
    agg_breakdown_rows = []
    for ds_name, df in datasets:
        X_scaled = scaler.fit_transform(df[CONFIG_B])
        y_true = df['impact_category'].values
        for linkage_type in ['single', 'complete', 'average', 'ward']:
            agg = AgglomerativeClustering(n_clusters=3, linkage=linkage_type)
            labels = agg.fit_predict(X_scaled) + 1
            for cid in [1, 2, 3]:
                mask = labels == cid
                sub_y = pd.Series(y_true[mask])
                counts = {cls: int((sub_y == cls).sum()) for cls in CLASS_NAMES}
                dom_class = max(counts, key=counts.get) if mask.sum() > 0 else 'N/A'
                dom_pct = round(100.0 * counts[dom_class] / mask.sum(), 1) if mask.sum() > 0 else 0.0
                agg_breakdown_rows.append({
                    'Dataset': ds_name,
                    'Linkage': linkage_type,
                    'Cluster': f"Cluster {cid}",
                    'Size': int(mask.sum()),
                    'Low Impact': counts['Low Impact'],
                    'Moderate Impact': counts['Moderate Impact'],
                    'High Impact': counts['High Impact'],
                    'Dominant Class': dom_class,
                    'Dominant %': dom_pct
                })

    df_agg_breakdown = pd.DataFrame(agg_breakdown_rows)
    df_agg_breakdown.to_csv(REPORTS_DIR / "agglomerative_cluster_breakdown.csv", index=False)
    df_agg_breakdown.to_csv(WORKSPACE_REPORTS_DIR / "agglomerative_cluster_breakdown.csv", index=False)
    cluster_means.to_csv(WORKSPACE_REPORTS_DIR / "cluster_behavioral_profiles.csv")
    print("[SUCCESS] Clustering suite, evaluation, and visualizations complete!")


if __name__ == "__main__":
    run_clustering_suite()
