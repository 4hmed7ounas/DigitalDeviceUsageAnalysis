#!/usr/bin/env python3
"""
train_knn.py
-------------
Phase II: K-Nearest Neighbors Classification Experiments & Diagnostics

Author: Senior Data Scientist & Machine Learning Engineer
Project: Digital Device Usage Pattern Analysis Using Machine Learning
Strict Random Seed: 42

This script:
1. Loads leakage-free datasets:
   - Augmented Training Set (N=134: 38 genuine + 96 training-only synthetic)
   - Genuine-Only Training Set (N=38 genuine training responses)
   - Untouched Holdout Test Set (N=16 genuine responses: 2 Low, 10 Moderate, 4 High)
2. Compares four experimental configurations:
   - Condition 1: Augmented Training, Feature Configuration A (Atomic, 18 features)
   - Condition 2: Augmented Training, Feature Configuration B (Composite, 14 features)
   - Condition 3: Genuine-Only Baseline, Feature Configuration A (Atomic, 18 features)
   - Condition 4: Genuine-Only Baseline, Feature Configuration B (Composite, 14 features)
3. Performs Stratified 5-Fold Cross-Validation on training data for hyperparameter tuning:
   - K values: [3, 5, 7, 9, 11, 13, 15]
   - Weighting: ['uniform', 'distance']
   - Distance Metrics: ['euclidean', 'manhattan']
4. Fits StandardScaler strictly within CV folds and full training sets to prevent leakage.
5. Evaluates best estimators on the untouched 16-record holdout test set:
   - Accuracy, Macro F1, Weighted F1, Confusion Matrix, Per-class Precision/Recall/F1.
6. Generates publication-quality diagnostic visualizations saved in reports/phase_ii/.
7. Transparently documents test set limitations and class representation constraints.
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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score,
    confusion_matrix, classification_report
)
from sklearn.decomposition import PCA

# Plot style setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_SPLITS = PROJECT_ROOT / "data" / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports" / "phase_ii"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Also ensure workspace root reports/phase_ii exists
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


def run_experiments():
    print("[INFO] Loading datasets from data/splits/ ...")
    df_train_aug = pd.read_csv(DATA_SPLITS / "train_augmented_features.csv")
    df_train_gen = pd.read_csv(DATA_SPLITS / "genuine_train_features.csv")
    df_test = pd.read_csv(DATA_SPLITS / "genuine_test_features.csv")

    print(f"Augmented Train Shape: {df_train_aug.shape}")
    print(f"Genuine Train Shape: {df_train_gen.shape}")
    print(f"Genuine Test Shape: {df_test.shape}")
    print("Holdout Test Class Distribution:\n", df_test['impact_category'].value_counts())

    experiments = [
        {
            'id': 'aug_config_a',
            'name': 'Augmented Train (N=134) | Config A (Atomic)',
            'train_df': df_train_aug,
            'features': CONFIG_A,
            'color': '#1f77b4'
        },
        {
            'id': 'aug_config_b',
            'name': 'Augmented Train (N=134) | Config B (Composite)',
            'train_df': df_train_aug,
            'features': CONFIG_B,
            'color': '#2ca02c'
        },
        {
            'id': 'gen_config_a',
            'name': 'Genuine Only Baseline (N=38) | Config A (Atomic)',
            'train_df': df_train_gen,
            'features': CONFIG_A,
            'color': '#ff7f0e'
        },
        {
            'id': 'gen_config_b',
            'name': 'Genuine Only Baseline (N=38) | Config B (Composite)',
            'train_df': df_train_gen,
            'features': CONFIG_B,
            'color': '#d62728'
        }
    ]

    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan']
    }

    results = []
    trained_models = {}
    y_test = df_test['impact_encoded'].values

    # Stratified 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for exp in experiments:
        exp_id = exp['id']
        name = exp['name']
        feats = exp['features']
        X_train = exp['train_df'][feats].values
        y_train = exp['train_df']['impact_encoded'].values
        X_test = df_test[feats].values

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier())
        ])

        grid = GridSearchCV(
            pipe,
            param_grid,
            cv=cv,
            scoring='f1_macro',
            n_jobs=1,
            return_train_score=True
        )
        grid.fit(X_train, y_train)

        best_pipe = grid.best_estimator_
        trained_models[exp_id] = {
            'grid': grid,
            'best_pipe': best_pipe,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'features': feats,
            'name': name
        }

        # Predict on untouched test set
        y_pred = best_pipe.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        p_class, r_class, f1_class, support = precision_recall_fscore_support(
            y_test, y_pred, labels=[0, 1, 2], zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

        res = {
            'exp_id': exp_id,
            'name': name,
            'feature_count': len(feats),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'best_k': grid.best_params_['knn__n_neighbors'],
            'best_weights': grid.best_params_['knn__weights'],
            'best_metric': grid.best_params_['knn__metric'],
            'best_cv_macro_f1': float(round(grid.best_score_, 4)),
            'test_accuracy': float(round(acc, 4)),
            'test_macro_f1': float(round(macro_f1, 4)),
            'test_weighted_f1': float(round(weighted_f1, 4)),
            'class_precision': [float(round(p, 4)) for p in p_class],
            'class_recall': [float(round(r, 4)) for r in r_class],
            'class_f1': [float(round(f, 4)) for f in f1_class],
            'class_support': [int(s) for s in support],
            'confusion_matrix': cm.tolist()
        }
        results.append(res)

        print(f"\n=======================================================")
        print(f"Condition: {name}")
        print(f"Best Hyperparameters: K={res['best_k']}, Weights={res['best_weights']}, Metric={res['best_metric']}")
        print(f"Mean CV Macro F1: {res['best_cv_macro_f1']:.4f}")
        print(f"Holdout Test Accuracy: {res['test_accuracy']:.4f}")
        print(f"Holdout Test Macro F1: {res['test_macro_f1']:.4f}")
        print(f"Holdout Test Weighted F1: {res['test_weighted_f1']:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Per-Class F1 (Low / Mod / High): {res['class_f1']}")

    # Export results summary table
    summary_rows = []
    for r in results:
        summary_rows.append({
            'Model Configuration': r['name'],
            'N_train': r['train_size'],
            'N_features': r['feature_count'],
            'Best K': r['best_k'],
            'Weighting': r['best_weights'],
            'Distance': r['best_metric'],
            'CV Macro F1': r['best_cv_macro_f1'],
            'Test Acc': r['test_accuracy'],
            'Test Macro F1': r['test_macro_f1'],
            'Test W-F1': r['test_weighted_f1'],
            'F1 Low (N=2)': r['class_f1'][0],
            'F1 Mod (N=10)': r['class_f1'][1],
            'F1 High (N=4)': r['class_f1'][2]
        })

    # Add Majority-Class Baseline (Always Moderate Impact)
    y_pred_maj = np.full_like(y_test, fill_value=1)
    acc_maj = accuracy_score(y_test, y_pred_maj)
    macro_f1_maj = f1_score(y_test, y_pred_maj, average='macro', zero_division=0)
    weighted_f1_maj = f1_score(y_test, y_pred_maj, average='weighted', zero_division=0)
    p_maj, r_maj, f1_maj, _ = precision_recall_fscore_support(y_test, y_pred_maj, labels=[0, 1, 2], zero_division=0)

    summary_rows.append({
        'Model Configuration': 'Majority-Class Baseline (Always Moderate)',
        'N_train': 'N/A',
        'N_features': 'N/A',
        'Best K': 'N/A',
        'Weighting': 'N/A',
        'Distance': 'N/A',
        'CV Macro F1': 'N/A',
        'Test Acc': float(round(acc_maj, 4)),
        'Test Macro F1': float(round(macro_f1_maj, 4)),
        'Test W-F1': float(round(weighted_f1_maj, 4)),
        'F1 Low (N=2)': float(round(f1_maj[0], 4)),
        'F1 Mod (N=10)': float(round(f1_maj[1], 4)),
        'F1 High (N=4)': float(round(f1_maj[2], 4))
    })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(REPORTS_DIR / "knn_evaluation_summary.csv", index=False)
    df_summary.to_csv(WORKSPACE_REPORTS_DIR / "knn_evaluation_summary.csv", index=False)
    print("\n[INFO] Saved summary with Majority-Class Baseline to reports/phase_ii/knn_evaluation_summary.csv")

    # ==============================================================================
    # FIGURE 1: HYPERPARAMETER TUNING CURVES (K vs CV Macro F1)
    # ==============================================================================
    print("[INFO] Generating Figure 1: Hyperparameter Tuning Curves...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharey=True)
    axes = axes.flatten()

    for idx, exp in enumerate(experiments):
        exp_id = exp['id']
        grid = trained_models[exp_id]['grid']
        cv_results = pd.DataFrame(grid.cv_results_)

        ax = axes[idx]
        for metric in ['euclidean', 'manhattan']:
            for weight in ['uniform', 'distance']:
                sub = cv_results[
                    (cv_results['param_knn__metric'] == metric) &
                    (cv_results['param_knn__weights'] == weight)
                ].sort_values('param_knn__n_neighbors')
                
                label = f"{metric.capitalize()} - {weight.capitalize()}"
                linestyle = '-' if weight == 'distance' else '--'
                marker = 'o' if metric == 'manhattan' else 's'
                ax.plot(
                    sub['param_knn__n_neighbors'],
                    sub['mean_test_score'],
                    label=label,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=2
                )

        ax.set_title(exp['name'], fontsize=12, fontweight='bold')
        ax.set_xlabel("Number of Neighbors (K)")
        ax.set_ylabel("Stratified 5-Fold CV Macro F1")
        ax.set_xticks([3, 5, 7, 9, 11, 13, 15])
        ax.set_ylim(0.15, 0.70)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    fig1_path = REPORTS_DIR / "01_knn_cv_hyperparameter_tuning.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.savefig(WORKSPACE_REPORTS_DIR / "01_knn_cv_hyperparameter_tuning.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ==============================================================================
    # FIGURE 2: CONFUSION MATRICES (2x2 GRID ON 16-RECORD HOLDOUT)
    # ==============================================================================
    print("[INFO] Generating Figure 2: Confusion Matrices...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()

    for idx, exp in enumerate(experiments):
        exp_id = exp['id']
        r = results[idx]
        cm = np.array(r['confusion_matrix'])
        ax = axes[idx]

        # Annotate with both count and row-wise percentage
        annot_matrix = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            row_sum = cm[i, :].sum()
            for j in range(cm.shape[1]):
                cnt = cm[i, j]
                pct = (cnt / row_sum * 100) if row_sum > 0 else 0
                annot_matrix[i, j] = f"{cnt}\n({pct:.0f}%)"

        sns.heatmap(
            cm,
            annot=annot_matrix,
            fmt='',
            cmap='Blues',
            cbar=False,
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
            annot_kws={'size': 12, 'weight': 'bold'}
        )
        ax.set_title(
            f"{r['name']}\nAccuracy: {r['test_accuracy']:.2%} | Macro F1: {r['test_macro_f1']:.4f}",
            fontsize=11,
            fontweight='bold'
        )
        ax.set_xlabel("Predicted Impact Category", fontweight='bold')
        ax.set_ylabel("True Impact Category (Ground Truth)", fontweight='bold')

    plt.tight_layout()
    fig2_path = REPORTS_DIR / "02_knn_confusion_matrices.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.savefig(WORKSPACE_REPORTS_DIR / "02_knn_confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ==============================================================================
    # FIGURE 3: PER-CLASS PRECISION, RECALL, F1 COMPARISON
    # ==============================================================================
    print("[INFO] Generating Figure 3: Per-Class Performance Comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    
    metrics = ['Precision', 'Recall', 'F1-Score']
    metric_keys = ['class_precision', 'class_recall', 'class_f1']

    bar_width = 0.2
    x = np.arange(len(CLASS_NAMES))

    for m_idx, (m_label, m_key) in enumerate(zip(metrics, metric_keys)):
        ax = axes[m_idx]
        for exp_idx, exp in enumerate(experiments):
            r = results[exp_idx]
            vals = r[m_key]
            offset = (exp_idx - 1.5) * bar_width
            bars = ax.bar(
                x + offset,
                vals,
                width=bar_width,
                label=exp['name'].split('|')[0].strip() + f" ({'At.' if 'Atomic' in exp['name'] else 'Comp.'})",
                color=exp['color'],
                alpha=0.85
            )
            for bar in bars:
                height = bar.get_height()
                if height > 0.05:
                    ax.annotate(
                        f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8
                    )

        ax.set_title(f"Class-Level {m_label}", fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n(N={s})" for c, s in zip(CLASS_NAMES, [2, 10, 4])], fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.grid(True, linestyle=':', alpha=0.6)
        if m_idx == 0:
            ax.set_ylabel("Score (0.0 to 1.0)", fontweight='bold')
            ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    fig3_path = REPORTS_DIR / "03_knn_per_class_metrics.png"
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.savefig(WORKSPACE_REPORTS_DIR / "03_knn_per_class_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ==============================================================================
    # FIGURE 4: PCA DECISION BOUNDARIES & HOLDOUT PREDICTIONS
    # ==============================================================================
    print("[INFO] Generating Figure 4: PCA Decision Boundary Visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for i, exp_id in enumerate(['aug_config_a', 'gen_config_b']):
        ax = axes[i]
        model_info = trained_models[exp_id]
        X_tr = model_info['X_train']
        y_tr = model_info['y_train']
        X_te = model_info['X_test']
        y_te = model_info['y_test']

        # Fit PCA on scaled training features
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        pca = PCA(n_components=2, random_state=SEED)
        X_tr_pca = pca.fit_transform(X_tr_scaled)
        X_te_pca = pca.transform(X_te_scaled)

        # Retrain a 2D KNN surrogate on PCA space to visualize decision boundary
        best_k = model_info['grid'].best_params_['knn__n_neighbors']
        best_w = model_info['grid'].best_params_['knn__weights']
        best_m = model_info['grid'].best_params_['knn__metric']

        knn_2d = KNeighborsClassifier(n_neighbors=best_k, weights=best_w, metric=best_m)
        knn_2d.fit(X_tr_pca, y_tr)

        # Mesh grid
        x_min, x_max = X_tr_pca[:, 0].min() - 1.5, X_tr_pca[:, 0].max() + 1.5
        y_min, y_max = X_tr_pca[:, 1].min() - 1.5, X_tr_pca[:, 1].max() + 1.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
        Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        cmap_light = matplotlib.colors.ListedColormap(['#d9edf7', '#dff0d8', '#f2dede'])
        cmap_bold = ['#31708f', '#3c763d', '#a94442']

        ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

        # Plot training points (small)
        for class_val, c_name in enumerate(CLASS_NAMES):
            mask_tr = (y_tr == class_val)
            ax.scatter(
                X_tr_pca[mask_tr, 0], X_tr_pca[mask_tr, 1],
                color=cmap_bold[class_val], label=f"Train: {c_name}",
                alpha=0.45, s=35, edgecolors='none'
            )

        # Plot holdout test points (large, marked with correctness)
        y_pred_te = knn_2d.predict(X_te_pca)
        for class_val, c_name in enumerate(CLASS_NAMES):
            mask_te = (y_te == class_val)
            for j in np.where(mask_te)[0]:
                correct = (y_pred_te[j] == y_te[j])
                marker = 'o' if correct else 'X'
                color = cmap_bold[class_val]
                ax.scatter(
                    X_te_pca[j, 0], X_te_pca[j, 1],
                    color=color,
                    marker=marker,
                    s=120,
                    edgecolors='black',
                    linewidths=1.8,
                    zorder=5
                )

        ax.set_title(
            f"{model_info['name']}\nPCA Space (Var Expl: {pca.explained_variance_ratio_.sum():.1%})",
            fontsize=11, fontweight='bold'
        )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)", fontweight='bold')
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)", fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5)

    # Custom legend for test markers
    custom_lines = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=10, label='Test Correct (o)'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=10, label='Test Error (X)')
    ]
    axes[0].legend(handles=custom_lines, loc='upper right', fontsize=9)

    plt.tight_layout()
    fig4_path = REPORTS_DIR / "04_knn_decision_boundaries_pca.png"
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
    plt.savefig(WORKSPACE_REPORTS_DIR / "04_knn_decision_boundaries_pca.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("[SUCCESS] KNN training, evaluation, and visualization suite complete!")
    return results


if __name__ == "__main__":
    run_experiments()
