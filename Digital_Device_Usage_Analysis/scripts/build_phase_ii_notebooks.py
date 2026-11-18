#!/usr/bin/env python3
"""
build_phase_ii_notebooks.py
----------------------------
Programmatic Notebook Construction and Execution Engine:
- notebooks/02_knn_classification.ipynb
- notebooks/03_clustering_analysis.ipynb
- notebooks/Assignment2_Complete_Submission.ipynb (single unified end-to-end submission notebook)

Author: Ahmed Younas (MS Artificial Intelligence, Semester 1)
Strict Random Seed: 42
"""

import os
import sys
import json
import io
import ast
import base64
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
WORKSPACE_NOTEBOOKS_DIR = PROJECT_ROOT.parent / "notebooks"

NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
WORKSPACE_NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


class NotebookExecutor:
    """Executes notebook cells in-process, capturing stdout, rich HTML/text, and matplotlib figures."""
    def __init__(self):
        import matplotlib
        matplotlib.use('Agg')

        def _display(*args, **kwargs):
            for arg in args:
                if hasattr(arg, 'to_string'):
                    print(arg.to_string())
                else:
                    print(arg)

        self.namespace = {
            '__name__': '__main__',
            '__builtins__': __builtins__,
            'display': _display
        }
        self.cells = []
        self.execution_count = 0

    def add_markdown(self, text):
        lines = [line + '\n' for line in text.strip().split('\n')]
        if lines:
            lines[-1] = lines[-1].rstrip('\n')
        self.cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": lines
        })

    def add_code(self, code_str):
        self.execution_count += 1
        lines = [line + '\n' for line in code_str.strip().split('\n')]
        if lines:
            lines[-1] = lines[-1].rstrip('\n')

        cell = {
            "cell_type": "code",
            "execution_count": self.execution_count,
            "metadata": {},
            "outputs": [],
            "source": lines
        }

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf

        eval_result = None
        eval_err = None

        try:
            tree = ast.parse(code_str)
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                exec_body = ast.Module(body=tree.body[:-1], type_ignores=[])
                exec(compile(exec_body, '<cell>', 'exec'), self.namespace)
                expr_code = compile(ast.Expression(body=tree.body[-1].value), '<cell>', 'eval')
                eval_result = eval(expr_code, self.namespace)
            else:
                exec(compile(tree, '<cell>', 'exec'), self.namespace)
        except Exception:
            import traceback
            eval_err = traceback.format_exc()
            print(f"Cell execution error:\n{eval_err}", file=sys.stderr)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        stdout_val = stdout_buf.getvalue()
        stderr_val = stderr_buf.getvalue()

        if stdout_val:
            cell["outputs"].append({
                "name": "stdout",
                "output_type": "stream",
                "text": [line + '\n' for line in stdout_val.splitlines()]
            })

        if stderr_val:
            cell["outputs"].append({
                "name": "stderr",
                "output_type": "stream",
                "text": [line + '\n' for line in stderr_val.splitlines()]
            })

        if eval_result is not None:
            data_dict = {}
            if hasattr(eval_result, '_repr_html_'):
                data_dict["text/html"] = [eval_result._repr_html_()]
                data_dict["text/plain"] = [repr(eval_result)]
            elif hasattr(eval_result, 'to_html'):
                data_dict["text/html"] = [eval_result.to_html(classes='table table-striped table-hover')]
                data_dict["text/plain"] = [repr(eval_result)]
            else:
                data_dict["text/plain"] = [repr(eval_result)]

            cell["outputs"].append({
                "data": data_dict,
                "execution_count": self.execution_count,
                "metadata": {},
                "output_type": "execute_result"
            })

        try:
            import matplotlib.pyplot as plt
            if plt.get_fignums():
                for fignum in plt.get_fignums():
                    fig = plt.figure(fignum)
                    img_buf = io.BytesIO()
                    fig.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
                    img_buf.seek(0)
                    b64_str = base64.b64encode(img_buf.read()).decode('utf-8')
                    cell["outputs"].append({
                        "data": {
                            "image/png": b64_str,
                            "text/plain": ["<Figure size ...>"]
                        },
                        "metadata": {},
                        "output_type": "display_data"
                    })
                plt.close('all')
        except Exception as e:
            print(f"[WARN] Failed to capture matplotlib figure: {e}")

        self.cells.append(cell)
        if eval_err:
            raise RuntimeError(f"Execution failed in cell {self.execution_count}:\n{eval_err}")

    def save(self, output_path: Path):
        nb_json = {
            "cells": self.cells,
            "metadata": {
                "language_info": {"name": "python", "version": "3.13"},
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
            },
            "nbformat": 4,
            "nbformat_minor": 5
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(nb_json, f, indent=1, ensure_ascii=False)
        print(f"[SUCCESS] Saved notebook ({len(self.cells)} cells): {output_path}")


# ==============================================================================
# NOTEBOOK 2: KNN CLASSIFICATION (CONCISE)
# ==============================================================================
def build_knn_notebook():
    print("\n" + "=" * 60)
    print("BUILDING NOTEBOOK 02: KNN CLASSIFICATION (CONCISE)")
    print("=" * 60)

    nb = NotebookExecutor()

    nb.add_markdown(r"""# Phase III: K-Nearest Neighbors (KNN) Classification
**Author:** Ahmed Younas — MS Artificial Intelligence, Semester 1  
**Protocol:** Stratified 38 genuine train (+96 training-only synthetic = 134 augmented) vs. 16 untouched genuine holdout records (`SEED=42`).""")

    nb.add_code("""from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix
from sklearn.decomposition import PCA

SEED = 42
np.random.seed(SEED)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
print(f"Environment ready | NumPy {np.__version__}, Pandas {pd.__version__}, Seed={SEED}")""")

    nb.add_markdown("## 1. Load Leakage-Free Splits & Define Feature Configurations\nCompare **Config A (18 atomic features)** and **Config B (14 composite features)**. Target columns (`academic_impact_avg`, `impact_category`, `impact_encoded`, Q6–Q8) are strictly excluded.")

    nb.add_code("""current_dir = Path.cwd()
if (current_dir / "data" / "splits").exists():
    data_dir = current_dir / "data" / "splits"
elif (current_dir / "Digital_Device_Usage_Analysis" / "data" / "splits").exists():
    data_dir = current_dir / "Digital_Device_Usage_Analysis" / "data" / "splits"
elif (current_dir / "train_augmented_features.csv").exists():
    data_dir = current_dir
else:
    data_dir = current_dir.parent / "data" / "splits"

df_train_aug = pd.read_csv(data_dir / "train_augmented_features.csv")
df_train_gen = pd.read_csv(data_dir / "genuine_train_features.csv")
df_test = pd.read_csv(data_dir / "genuine_test_features.csv")

config_a = [
    'gender_encoded',
    'year_First year', 'year_Second year', 'year_Third year', 'year_Fourth year', 'year_Other',
    'daily_device_hours', 'check_frequency',
    'phone_with_family', 'conversation_interruption', 'social_complaints', 'stay_away_difficulty',
    'activity_Communication and messaging', 'activity_Educational activities',
    'activity_Online gaming', 'activity_Other', 'activity_Social media',
    'activity_Watching videos and entertainment'
]
config_b = [
    'gender_encoded',
    'year_First year', 'year_Second year', 'year_Third year', 'year_Fourth year', 'year_Other',
    'social_conflict_score', 'digital_dependency_score',
    'activity_Communication and messaging', 'activity_Educational activities',
    'activity_Online gaming', 'activity_Other', 'activity_Social media',
    'activity_Watching videos and entertainment'
]

forbidden = {'academic_impact_avg', 'impact_category', 'impact_encoded', 'record_id', 'data_source', 'Timestamp'}
assert not forbidden.intersection(config_a) and not forbidden.intersection(config_b)

print(f"Augmented Train: {df_train_aug.shape} | Genuine Train: {df_train_gen.shape} | Genuine Holdout: {df_test.shape}")
print("Holdout class counts:", df_test['impact_category'].value_counts().to_dict())""")

    nb.add_markdown("## 2. Stratified 5-Fold Cross-Validation Grid Search\n`StandardScaler` is fitted inside the `Pipeline` per fold. Grid: $K \\in \\{3,5,7,9,11,13,15\\}$, weights $\\in \\{\\text{uniform}, \\text{distance}\\}$, metric $\\in \\{\\text{euclidean}, \\text{manhattan}\\}$, scored by Macro F1.")

    nb.add_code("""experiments = [
    {'id': 'aug_config_a', 'name': 'Augmented Train (N=134) | Config A (Atomic)', 'train_df': df_train_aug, 'features': config_a, 'color': '#1f77b4'},
    {'id': 'aug_config_b', 'name': 'Augmented Train (N=134) | Config B (Composite)', 'train_df': df_train_aug, 'features': config_b, 'color': '#2ca02c'},
    {'id': 'gen_config_a', 'name': 'Genuine Only Baseline (N=38) | Config A (Atomic)', 'train_df': df_train_gen, 'features': config_a, 'color': '#ff7f0e'},
    {'id': 'gen_config_b', 'name': 'Genuine Only Baseline (N=38) | Config B (Composite)', 'train_df': df_train_gen, 'features': config_b, 'color': '#d62728'}
]

param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan']
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
trained_models = {}
cv_rows = []

for exp in experiments:
    X_tr = exp['train_df'][exp['features']].values
    y_tr = exp['train_df']['impact_encoded'].values
    pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='f1_macro', n_jobs=1)
    grid.fit(X_tr, y_tr)
    trained_models[exp['id']] = {
        'grid': grid, 'best_pipe': grid.best_estimator_, 'best_params': grid.best_params_,
        'best_score': grid.best_score_, 'X_tr': X_tr, 'y_tr': y_tr,
        'features': exp['features'], 'name': exp['name']
    }
    cv_rows.append({
        'Condition': exp['name'], 'N_train': len(X_tr), 'Features': len(exp['features']),
        'Best K': grid.best_params_['knn__n_neighbors'],
        'Weights': grid.best_params_['knn__weights'],
        'Metric': grid.best_params_['knn__metric'],
        'CV Macro F1': round(grid.best_score_, 4)
    })

pd.DataFrame(cv_rows)""")

    nb.add_code("""fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
for idx, exp in enumerate(experiments):
    cv_res = pd.DataFrame(trained_models[exp['id']]['grid'].cv_results_)
    ax = axes.flatten()[idx]
    for metric in ['euclidean', 'manhattan']:
        for weight in ['uniform', 'distance']:
            sub = cv_res[(cv_res['param_knn__metric'] == metric) & (cv_res['param_knn__weights'] == weight)].sort_values('param_knn__n_neighbors')
            ax.plot(sub['param_knn__n_neighbors'], sub['mean_test_score'],
                    label=f"{metric.capitalize()} - {weight.capitalize()}",
                    linestyle='-' if weight == 'distance' else '--',
                    marker='o' if metric == 'manhattan' else 's', linewidth=2)
    ax.set_title(exp['name'], fontsize=11, fontweight='bold')
    ax.set_xlabel("K")
    ax.set_ylabel("5-Fold CV Macro F1")
    ax.set_xticks([3, 5, 7, 9, 11, 13, 15])
    ax.set_ylim(0.15, 0.70)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()""")

    nb.add_markdown("## 3. Holdout Evaluation ($N=16$) & Majority-Class Baseline\nEvaluate all 4 tuned pipelines on the untouched genuine holdout benchmark alongside a Majority-Class Baseline (always predicting `Moderate Impact`).")

    nb.add_code("""class_names = ['Low Impact', 'Moderate Impact', 'High Impact']
y_test = df_test['impact_encoded'].values
eval_records = []

for exp in experiments:
    m_info = trained_models[exp['id']]
    y_pred = m_info['best_pipe'].predict(df_test[m_info['features']].values)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    _, _, f1_cls, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1, 2], zero_division=0)
    m_info['y_pred'] = y_pred
    m_info['cm'] = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    eval_records.append({
        'Model': exp['name'], 'K': m_info['best_params']['knn__n_neighbors'],
        'Metric': m_info['best_params']['knn__metric'], 'Weights': m_info['best_params']['knn__weights'],
        'CV Macro F1': round(m_info['best_score'], 4),
        'Holdout Acc': round(acc, 4), 'Macro F1': round(macro_f1, 4), 'Weighted F1': round(weighted_f1, 4),
        'F1 Low': round(f1_cls[0], 4), 'F1 Mod': round(f1_cls[1], 4), 'F1 High': round(f1_cls[2], 4)
    })

y_pred_maj = np.full_like(y_test, fill_value=1)
_, _, f1_maj, _ = precision_recall_fscore_support(y_test, y_pred_maj, labels=[0, 1, 2], zero_division=0)
eval_records.append({
    'Model': 'Majority-Class Baseline (Always Moderate)', 'K': 'N/A', 'Metric': 'N/A', 'Weights': 'N/A',
    'CV Macro F1': 'N/A', 'Holdout Acc': round(accuracy_score(y_test, y_pred_maj), 4),
    'Macro F1': round(f1_score(y_test, y_pred_maj, average='macro', zero_division=0), 4),
    'Weighted F1': round(f1_score(y_test, y_pred_maj, average='weighted', zero_division=0), 4),
    'F1 Low': round(f1_maj[0], 4), 'F1 Mod': round(f1_maj[1], 4), 'F1 High': round(f1_maj[2], 4)
})
pd.DataFrame(eval_records)""")

    nb.add_code("""fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for idx, exp in enumerate(experiments):
    cm = trained_models[exp['id']]['cm']
    ax = axes.flatten()[idx]
    annot = np.array([[f"{cm[i,j]}\\n({(cm[i,j]/cm[i].sum()*100 if cm[i].sum()>0 else 0):.0f}%)" for j in range(3)] for i in range(3)])
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax, annot_kws={'size': 11, 'weight': 'bold'})
    rec = eval_records[idx]
    ax.set_title(f"{exp['name']}\\nAcc: {rec['Holdout Acc']:.1%} | Macro F1: {rec['Macro F1']:.4f}", fontsize=10, fontweight='bold')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.show()""")

    nb.add_code("""fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
x = np.arange(3)
bar_w = 0.2
for m_idx, m_name in enumerate(['Precision', 'Recall', 'F1-Score']):
    ax = axes[m_idx]
    for e_idx, exp in enumerate(experiments):
        p_c, r_c, f1_c, _ = precision_recall_fscore_support(y_test, trained_models[exp['id']]['y_pred'], labels=[0, 1, 2], zero_division=0)
        vals = [p_c, r_c, f1_c][m_idx]
        ax.bar(x + (e_idx - 1.5) * bar_w, vals, width=bar_w,
               label=exp['name'].split('|')[0].strip() + f" ({'A' if 'Atomic' in exp['name'] else 'B'})",
               color=exp['color'], alpha=0.85)
    ax.set_title(m_name, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\\n(N={s})" for c, s in zip(class_names, [2, 10, 4])])
    ax.set_ylim(0, 1.1)
    if m_idx == 0:
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
plt.tight_layout()
plt.show()""")

    nb.add_markdown("## 4. Illustrative 2D PCA Decision Boundaries\nScaler and PCA are fitted strictly on training data and applied to the 16 holdout points (`o` = correct, `X` = misclassified). Quantitative evaluation above uses the full feature space.")

    nb.add_code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for i, exp_id in enumerate(['aug_config_a', 'gen_config_b']):
    ax = axes[i]
    m_info = trained_models[exp_id]
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(m_info['X_tr'])
    X_te_sc = sc.transform(df_test[m_info['features']].values)
    pca = PCA(n_components=2, random_state=SEED)
    X_tr_p = pca.fit_transform(X_tr_sc)
    X_te_p = pca.transform(X_te_sc)
    knn_2d = KNeighborsClassifier(
        n_neighbors=m_info['best_params']['knn__n_neighbors'],
        weights=m_info['best_params']['knn__weights'],
        metric=m_info['best_params']['knn__metric']
    ).fit(X_tr_p, m_info['y_tr'])
    xx, yy = np.meshgrid(np.linspace(X_tr_p[:,0].min()-1.5, X_tr_p[:,0].max()+1.5, 200),
                         np.linspace(X_tr_p[:,1].min()-1.5, X_tr_p[:,1].max()+1.5, 200))
    Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=matplotlib.colors.ListedColormap(['#d9edf7', '#dff0d8', '#f2dede']), alpha=0.6)
    colors = ['#31708f', '#3c763d', '#a94442']
    for c_val, c_name in enumerate(class_names):
        m_tr = (m_info['y_tr'] == c_val)
        ax.scatter(X_tr_p[m_tr, 0], X_tr_p[m_tr, 1], color=colors[c_val], alpha=0.45, s=30, label=f"Train: {c_name}")
    y_pred_2d = knn_2d.predict(X_te_p)
    for j in range(len(y_test)):
        ax.scatter(X_te_p[j, 0], X_te_p[j, 1], color=colors[y_test[j]],
                   marker='o' if y_pred_2d[j] == y_test[j] else 'X', s=110, edgecolors='black', linewidths=1.5, zorder=5)
    ax.set_title(f"{m_info['name']}\\nPCA Var: {pca.explained_variance_ratio_.sum():.1%}", fontsize=10, fontweight='bold')
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.tight_layout()
plt.show()""")

    nb.add_markdown(r"""## 5. Key KNN Takeaways
- **CV vs. Holdout:** Synthetic augmentation boosted internal CV Macro F1 on Config A ($0.5976$ vs. $0.3014$), but did **not** consistently improve genuine holdout generalization. Genuine-Only Config B achieved the highest holdout accuracy ($68.75\%$, Macro F1 = $0.4000$), while Augmented Config A achieved $62.50\%$ (Macro F1 = $0.4091$) and Augmented Config B dropped to $31.25\%$.
- **Majority-Class Baseline:** Always predicting `Moderate Impact` yields $62.50\%$ accuracy (Macro F1 = $0.2564$), confirming accuracy alone is insufficient on this imbalanced 16-record test set.
- **Zero Low Impact Recall ($0.0\%$):** Both `Low Impact` holdout records were misclassified as `Moderate Impact` due to class imbalance and overlapping device-usage profiles once direct distraction items (Q6–Q8) are excluded.
- **Distance Metric:** Manhattan distance ($L_1$) performed best in Augmented Config A and Genuine Config B on this dataset, though this is an empirical finding specific to these configurations rather than a universal claim.""")

    output_path = NOTEBOOKS_DIR / "02_knn_classification.ipynb"
    nb.save(output_path)
    shutil.copy(output_path, WORKSPACE_NOTEBOOKS_DIR / "02_knn_classification.ipynb")
    print(f"[SUCCESS] Mirrored 02_knn_classification.ipynb to {WORKSPACE_NOTEBOOKS_DIR}")


# ==============================================================================
# NOTEBOOK 3: K-MEANS & AGGLOMERATIVE CLUSTERING + COMPARATIVE SUMMARY (CONCISE)
# ==============================================================================
def build_clustering_notebook():
    print("\n" + "=" * 60)
    print("BUILDING NOTEBOOK 03: CLUSTERING ANALYSIS (CONCISE + ALL 4 LINKAGES)")
    print("=" * 60)

    nb = NotebookExecutor()

    nb.add_markdown(r"""# Phases IV–VI: K-Means Clustering, Agglomerative Clustering & Comparative Analysis
**Author:** Ahmed Younas — MS Artificial Intelligence, Semester 1  
**Scope:** Full Augmented ($N=150$) & Genuine-Only ($N=54$), Config A (18 feats) vs. Config B (14 feats), `SEED=42`. Target columns (Q6–Q8, `impact_category`) are strictly excluded before fitting.""")

    nb.add_code("""from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage

SEED = 42
np.random.seed(SEED)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

current_dir = Path.cwd()
if (current_dir / "data" / "processed").exists():
    proc_dir = current_dir / "data" / "processed"
elif (current_dir / "Digital_Device_Usage_Analysis" / "data" / "processed").exists():
    proc_dir = current_dir / "Digital_Device_Usage_Analysis" / "data" / "processed"
elif (current_dir / "full_150_corrected_features.csv").exists():
    proc_dir = current_dir
else:
    proc_dir = current_dir.parent / "data" / "processed"

df_150 = pd.read_csv(proc_dir / "full_150_corrected_features.csv")
df_54 = pd.read_csv(proc_dir / "genuine_54_corrected_features.csv")

config_a = [
    'gender_encoded',
    'year_First year', 'year_Second year', 'year_Third year', 'year_Fourth year', 'year_Other',
    'daily_device_hours', 'check_frequency',
    'phone_with_family', 'conversation_interruption', 'social_complaints', 'stay_away_difficulty',
    'activity_Communication and messaging', 'activity_Educational activities',
    'activity_Online gaming', 'activity_Other', 'activity_Social media',
    'activity_Watching videos and entertainment'
]
config_b = [
    'gender_encoded',
    'year_First year', 'year_Second year', 'year_Third year', 'year_Fourth year', 'year_Other',
    'social_conflict_score', 'digital_dependency_score',
    'activity_Communication and messaging', 'activity_Educational activities',
    'activity_Online gaming', 'activity_Other', 'activity_Social media',
    'activity_Watching videos and entertainment'
]

forbidden = {'academic_impact_avg', 'impact_category', 'impact_encoded', 'record_id', 'data_source', 'Timestamp'}
assert not forbidden.intersection(config_a) and not forbidden.intersection(config_b)
print(f"Loaded Full 150: {df_150.shape} | Genuine 54: {df_54.shape} | Target isolation verified.")""")

    nb.add_markdown("## 1. K-Means Clustering ($K=2$ to $6$)\nStandardize features with `StandardScaler`, evaluate WCSS (Inertia) and Silhouette Score, and inspect Elbow curves.")

    nb.add_code("""datasets = [('Full 150 Augmented', df_150), ('Genuine 54 Only', df_54)]
configs = [('Config A (Atomic, 18 feats)', config_a), ('Config B (Composite, 14 feats)', config_b)]
scaler = StandardScaler()
class_names = ['Low Impact', 'Moderate Impact', 'High Impact']

km_rows = []
for ds_name, df in datasets:
    for cfg_name, feats in configs:
        X_sc = scaler.fit_transform(df[feats])
        y_true = df['impact_category'].values
        for k in range(2, 7):
            km = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=SEED).fit(X_sc)
            km_rows.append({
                'Dataset': ds_name, 'Config': cfg_name, 'K': k,
                'Inertia': round(km.inertia_, 2),
                'Silhouette': round(silhouette_score(X_sc, km.labels_), 4),
                'ARI': round(adjusted_rand_score(y_true, km.labels_), 4),
                'NMI': round(normalized_mutual_info_score(y_true, km.labels_), 4)
            })
df_km = pd.DataFrame(km_rows)
df_km""")

    nb.add_code("""fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for c_idx, (ds_name, _) in enumerate(datasets):
    sub = df_km[df_km['Dataset'] == ds_name]
    for cfg_name, color, mkr in [('Config A (Atomic, 18 feats)', '#1f77b4', 'o'), ('Config B (Composite, 14 feats)', '#2ca02c', 's')]:
        s = sub[sub['Config'] == cfg_name].sort_values('K')
        axes[0, c_idx].plot(s['K'], s['Inertia'], label=cfg_name, color=color, marker=mkr, linewidth=2)
        axes[1, c_idx].plot(s['K'], s['Silhouette'], label=cfg_name, color=color, marker=mkr, linewidth=2)
    axes[0, c_idx].set_title(f"{ds_name} — Elbow Curve (WCSS)", fontweight='bold')
    axes[0, c_idx].set_ylabel("Inertia")
    axes[1, c_idx].set_title(f"{ds_name} — Silhouette Score", fontweight='bold')
    axes[1, c_idx].set_ylabel("Silhouette")
    axes[1, c_idx].set_xlabel("K")
    for r in [0, 1]:
        axes[r, c_idx].set_xticks(range(2, 7))
        axes[r, c_idx].legend(fontsize=8)
plt.tight_layout()
plt.show()""")

    nb.add_markdown("## 2. K-Means Post-Processing: 2D PCA Clusters, Centroids & Unexpected Observations ($K=3$, Config B)\nVisualize discovered clusters in 2D PCA space, report descriptive cluster means (without circular $p$-values), dominant actual class per cluster, and negative-silhouette boundary observations.")

    nb.add_code("""X_sc_150 = scaler.fit_transform(df_150[config_b])
km3 = KMeans(n_clusters=3, init='k-means++', n_init=20, random_state=SEED)
km_lbls = km3.fit_predict(X_sc_150) + 1
sil_vals = silhouette_samples(X_sc_150, km_lbls - 1)

pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X_sc_150)
cent_pca = pca.transform(km3.cluster_centers_)
var_exp = pca.explained_variance_ratio_ * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
c_cols = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c'}
for cid in [1, 2, 3]:
    m = km_lbls == cid
    axes[0].scatter(X_pca[m, 0], X_pca[m, 1], c=c_cols[cid], label=f"Cluster {cid} (N={m.sum()})", alpha=0.75, s=50)
axes[0].scatter(cent_pca[:, 0], cent_pca[:, 1], c='black', marker='X', s=180, label='Centroids', zorder=5)
neg_m = sil_vals < 0
axes[0].scatter(X_pca[neg_m, 0], X_pca[neg_m, 1], facecolors='none', edgecolors='red', s=100, linewidths=1.4, label=f"Boundary/Neg Sil (N={neg_m.sum()})")
axes[0].set_title("K-Means Clusters (2D PCA, Full 150, K=3)", fontweight='bold')
axes[0].set_xlabel(f"PC1 ({var_exp[0]:.1f}%)"); axes[0].set_ylabel(f"PC2 ({var_exp[1]:.1f}%)"); axes[0].legend(fontsize=8)

imp_cols = {'Low Impact': '#2ca02c', 'Moderate Impact': '#1f77b4', 'High Impact': '#d62728'}
for cls in class_names:
    m = df_150['impact_category'].values == cls
    axes[1].scatter(X_pca[m, 0], X_pca[m, 1], c=imp_cols[cls], label=f"{cls} (N={m.sum()})", alpha=0.75, s=50)
axes[1].set_title("Ground-Truth Academic Impact Labels (Same 2D PCA)", fontweight='bold')
axes[1].set_xlabel(f"PC1 ({var_exp[0]:.1f}%)"); axes[1].set_ylabel(f"PC2 ({var_exp[1]:.1f}%)"); axes[1].legend(fontsize=8)
plt.tight_layout()
plt.show()""")

    nb.add_code("""df_prof = df_150.copy()
df_prof['Cluster'] = km_lbls
profile_vars = ['daily_device_hours', 'check_frequency', 'social_conflict_score', 'digital_dependency_score', 'stay_away_difficulty']
means_tbl = df_prof.groupby('Cluster')[profile_vars].mean().round(2)
means_tbl.insert(0, 'Count', df_prof['Cluster'].value_counts().sort_index())

ct = pd.crosstab(df_prof['Cluster'], df_prof['impact_category']).reindex(columns=class_names, fill_value=0)
means_tbl['Dominant Class'] = ct.idxmax(axis=1)
print("=== K-Means Cluster Counts, Descriptive Means & Dominant Actual Class (Full 150, K=3) ===")
display(means_tbl)
print("\\n=== Contingency Table (Cluster vs. Actual Impact) ===")
display(ct)

# Flag negative-silhouette / boundary observations
df_unexp = df_prof[['data_source', 'daily_device_hours', 'check_frequency', 'social_conflict_score', 'digital_dependency_score', 'impact_category', 'Cluster']].copy()
df_unexp['Silhouette'] = np.round(sil_vals, 4)
flagged = df_unexp[df_unexp['Silhouette'] < 0].sort_values('Silhouette')
print(f"\\n=== Sample Unexpected / Boundary Observations (Total Negative Silhouette: {len(flagged)}) ===")
display(flagged.head(6))""")

    nb.add_markdown("## 3. Agglomerative Hierarchical Clustering (Single, Complete, Average, Ward)\nEvaluate all **4 required linkage methods** across $K \\in [2, 6]$, plot dendrograms, and compare cluster size balance and dominant actual classes at $K=3$.")

    nb.add_code("""agg_rows = []
for ds_name, df in datasets:
    for cfg_name, feats in configs:
        X_sc = scaler.fit_transform(df[feats])
        y_true = df['impact_category'].values
        for link in ['single', 'complete', 'average', 'ward']:
            for k in range(2, 7):
                lbls = AgglomerativeClustering(n_clusters=k, linkage=link).fit_predict(X_sc)
                agg_rows.append({
                    'Dataset': ds_name, 'Config': cfg_name, 'Linkage': link, 'K': k,
                    'Silhouette': round(silhouette_score(X_sc, lbls), 4),
                    'ARI': round(adjusted_rand_score(y_true, lbls), 4),
                    'NMI': round(normalized_mutual_info_score(y_true, lbls), 4)
                })
df_agg = pd.DataFrame(agg_rows)
print("=== Agglomerative Comparison at K=3 (Config B, All 4 Linkages) ===")
display(df_agg[(df_agg['K'] == 3) & (df_agg['Config'].str.contains('Config B'))])""")

    nb.add_code("""fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for r_idx, (ds_name, df) in enumerate(datasets):
    X_sc = scaler.fit_transform(df[config_b])
    for c_idx, link in enumerate(['single', 'complete', 'average', 'ward']):
        ax = axes[r_idx, c_idx]
        Z = linkage(X_sc, method=link, metric='euclidean')
        dendrogram(Z, ax=ax, truncate_mode='lastp' if len(df) > 60 else None,
                   p=20 if len(df) > 60 else 30, show_leaf_counts=True,
                   color_threshold=0.7 * max(Z[:, 2]), above_threshold_color='#7f7f7f')
        ax.set_title(f"{ds_name} | {link.capitalize()}", fontsize=10, fontweight='bold')
        ax.set_ylabel("Distance")
plt.tight_layout()
plt.show()""")

    nb.add_code("""# Agglomerative cluster size & dominant class breakdown (Full 150, K=3, Config B)
X_sc = scaler.fit_transform(df_150[config_b])
y_true = df_150['impact_category'].values
brk_rows = []
for link in ['single', 'complete', 'average', 'ward']:
    lbls = AgglomerativeClustering(n_clusters=3, linkage=link).fit_predict(X_sc) + 1
    for cid in [1, 2, 3]:
        m = lbls == cid
        sub_y = pd.Series(y_true[m])
        cnts = {c: int((sub_y == c).sum()) for c in class_names}
        dom = max(cnts, key=cnts.get)
        brk_rows.append({
            'Linkage': link.capitalize(), 'Cluster': f"Cluster {cid}", 'Size': int(m.sum()),
            'Low': cnts['Low Impact'], 'Moderate': cnts['Moderate Impact'], 'High': cnts['High Impact'],
            'Dominant Class': dom, 'Dominant %': round(100 * cnts[dom] / m.sum(), 1)
        })
pd.DataFrame(brk_rows)""")

    nb.add_markdown(r"""## 4. Phase VI Comparative Summary (KNN vs. K-Means vs. Agglomerative)

| Evaluation Criteria | KNN | K-Means | Agglomerative |
| :--- | :--- | :--- | :--- |
| **Learning type** | Supervised | Unsupervised | Unsupervised |
| **Labels used during fitting** | Yes (`impact_encoded`) | No (strictly excluded) | No (strictly excluded) |
| **Accuracy on held-out labels ($N=16$)** | 68.75% (Genuine Config B) / 62.50% (Augmented Config A; Baseline = 62.50%) | Not directly applicable | Not directly applicable |
| **Silhouette Score ($K=3$, Config B)** | Not applicable | 0.1904 (Full 150) / 0.2194 (Genuine 54) | Ward: 0.1499 (150) / 0.2298 (54); Single/Avg: 0.2491 (chaining) |
| **ARI / NMI (Full 150, $K=3$, Config B)** | Not applicable | ARI = 0.0061 / NMI = 0.0043 | Ward: ARI = 0.0177 / NMI = 0.0516; Complete: ARI = 0.0398 / NMI = 0.0250 |
| **Main observations** | Predicts majority (`Moderate`) and some `High`, but 0% recall on `Low Impact`; Config B genuine baseline performed best. | Composite features (Config B) gave better separation; clusters reflect usage/platform habits rather than impact tiers. | Single & Average linkages suffer severe chaining (`136/9/5`), whereas Ward yields balanced clusters (`75/62/13`). |
| **Limitations** | Tiny holdout ($N=16$), class imbalance, synthetic augmentation did not improve holdout generalization. | Moderate silhouette (~0.19–0.22); 21 boundary observations with negative silhouette; near-zero alignment with impact labels. | Sensitive to linkage choice; Ward avoids chaining but yields lower raw silhouette; low external label alignment. |

*Note:* Near-zero ARI/NMI measures low cluster–label alignment; target isolation is verified independently via feature-column inspection. Whether unobserved factors (e.g., self-discipline or study habits) explain why similar usage profiles yield different impact labels is a hypothesis for future research.""")

    output_path = NOTEBOOKS_DIR / "03_clustering_analysis.ipynb"
    nb.save(output_path)
    shutil.copy(output_path, WORKSPACE_NOTEBOOKS_DIR / "03_clustering_analysis.ipynb")
    print(f"[SUCCESS] Mirrored 03_clustering_analysis.ipynb to {WORKSPACE_NOTEBOOKS_DIR}")


# ==============================================================================
# UNIFIED SUBMISSION NOTEBOOK (ALL PHASES IN ONE CLEAN SEQUENTIAL .IPYNB)
# ==============================================================================
def build_unified_submission_notebook():
    print("\n" + "=" * 60)
    print("BUILDING UNIFIED NOTEBOOK: Assignment2_Complete_Submission.ipynb")
    print("=" * 60)

    nb = NotebookExecutor()

    nb.add_markdown(r"""# Digital Device Usage Pattern Analysis Using Machine Learning
**Course:** Advanced Artificial Intelligence — Integrated Practical Assignment  
**Author:** Ahmed Younas — MS Artificial Intelligence, Semester 1  
**Pipeline:** Raw Survey ($N=54$) + Gaussian Copula Expansion ($N=150$) $\rightarrow$ EDA $\rightarrow$ Feature Engineering $\rightarrow$ KNN $\rightarrow$ K-Means $\rightarrow$ Agglomerative (Single, Complete, Average, Ward) $\rightarrow$ Comparative Analysis (`SEED=42`).""")

    nb.add_code("""from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix,
    silhouette_score, silhouette_samples, adjusted_rand_score, normalized_mutual_info_score
)

SEED = 42
np.random.seed(SEED)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

cwd = Path.cwd()
base = cwd / "Digital_Device_Usage_Analysis" if (cwd / "Digital_Device_Usage_Analysis").exists() else (cwd if (cwd / "data").exists() else cwd.parent)
def _load_csv(rel_path, flat_name):
    p = base / rel_path
    return pd.read_csv(p if p.exists() else (cwd / flat_name))

df_raw = _load_csv("data/raw/original_responses.csv", "original_responses.csv")
df_comb = _load_csv("data/processed/combined_dataset.csv", "combined_dataset.csv")
df_150 = _load_csv("data/processed/full_150_corrected_features.csv", "full_150_corrected_features.csv")
df_54 = _load_csv("data/processed/genuine_54_corrected_features.csv", "genuine_54_corrected_features.csv")
df_tr_aug = _load_csv("data/splits/train_augmented_features.csv", "train_augmented_features.csv")
df_tr_gen = _load_csv("data/splits/genuine_train_features.csv", "genuine_train_features.csv")
df_test = _load_csv("data/splits/genuine_test_features.csv", "genuine_test_features.csv")

print(f"Raw Genuine: {df_raw.shape} | Combined (54+96): {df_comb.shape} | Corrected Features: {df_150.shape}")
print(f"Leakage-Free Splits -> Genuine Train: {df_tr_gen.shape} | Augmented Train: {df_tr_aug.shape} | Genuine Holdout: {df_test.shape}")
print(f"Missing values (Combined): {df_comb.isna().sum().sum()} | Duplicate rows: {df_comb.duplicated().sum()}")""")

    nb.add_markdown("## Phase I: Exploratory Data Analysis (EDA)\nInspect representative records, target class balance, numerical distributions, box plots (outliers), correlations, and real-vs-synthetic statistical fidelity.")

    nb.add_code("""display(df_150[['daily_device_hours', 'check_frequency', 'social_conflict_score', 'digital_dependency_score', 'stay_away_difficulty', 'impact_category', 'data_source']].head(6))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
class_names = ['Low Impact', 'Moderate Impact', 'High Impact']
df_150['impact_category'].value_counts().reindex(class_names).plot(kind='bar', ax=axes[0], color=['#2ca02c', '#1f77b4', '#d62728'])
axes[0].set_title("Target Class Balance (N=150)", fontweight='bold')
axes[0].tick_params(axis='x', rotation=15)

num_cols = ['daily_device_hours', 'check_frequency', 'social_conflict_score', 'digital_dependency_score', 'stay_away_difficulty']
sns.boxplot(data=df_150[num_cols], ax=axes[1], palette='Set2')
axes[1].set_title("Feature Box Plots & Outlier Check", fontweight='bold')
axes[1].set_xticklabels(['Hours', 'Checks', 'Conflict', 'Depend.', 'StayAway'], rotation=20)

sns.heatmap(df_150[num_cols].corr(method='spearman'), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes[2])
axes[2].set_title("Spearman Correlation (Core Behavioral)", fontweight='bold')
plt.tight_layout()
plt.show()

# Real vs Synthetic KS fidelity check
real_sub = df_150[df_150['data_source'] == 'real']
syn_sub = df_150[df_150['data_source'] != 'real']
ks_rows = [{'Feature': c, 'KS Stat': round(ks_2samp(real_sub[c], syn_sub[c]).statistic, 4),
            'p-value': round(ks_2samp(real_sub[c], syn_sub[c]).pvalue, 4)} for c in num_cols]
pd.DataFrame(ks_rows)""")

    nb.add_markdown("## Phase II: Preprocessing & Feature Configurations\n- **Config A (18 Atomic Features):** Gender, 5 one-hot academic years, 6 Likert behavioral items, 6 one-hot primary activities.\n- **Config B (14 Composite Features):** Replaces raw behavioral items with normalized `social_conflict_score` $\\in [1, 5]$ and `digital_dependency_score` $\\in [1, 5]$.\n- **Target Isolation:** `academic_impact_avg`, `impact_category`, `impact_encoded`, and Q6–Q8 are strictly excluded.")

    nb.add_code("""config_a = [
    'gender_encoded',
    'year_First year', 'year_Second year', 'year_Third year', 'year_Fourth year', 'year_Other',
    'daily_device_hours', 'check_frequency',
    'phone_with_family', 'conversation_interruption', 'social_complaints', 'stay_away_difficulty',
    'activity_Communication and messaging', 'activity_Educational activities',
    'activity_Online gaming', 'activity_Other', 'activity_Social media',
    'activity_Watching videos and entertainment'
]
config_b = [
    'gender_encoded',
    'year_First year', 'year_Second year', 'year_Third year', 'year_Fourth year', 'year_Other',
    'social_conflict_score', 'digital_dependency_score',
    'activity_Communication and messaging', 'activity_Educational activities',
    'activity_Online gaming', 'activity_Other', 'activity_Social media',
    'activity_Watching videos and entertainment'
]
forbidden = {'academic_impact_avg', 'impact_category', 'impact_encoded', 'record_id', 'data_source', 'Timestamp'}
assert not forbidden.intersection(config_a) and not forbidden.intersection(config_b)
print(f"Verified zero target leakage | Config A: {len(config_a)} features | Config B: {len(config_b)} features")""")

    nb.add_markdown("## Phase III: Supervised KNN Classification\nStratified 5-Fold CV with `StandardScaler` inside `Pipeline` ($K \\in [3, 15]$, uniform/distance, Euclidean/Manhattan), evaluated on the untouched 16-record genuine holdout against a Majority-Class Baseline.")

    nb.add_code("""experiments = [
    ('Augmented Train (N=134) | Config A', df_tr_aug, config_a),
    ('Augmented Train (N=134) | Config B', df_tr_aug, config_b),
    ('Genuine Only (N=38) | Config A', df_tr_gen, config_a),
    ('Genuine Only (N=38) | Config B', df_tr_gen, config_b)
]
param_grid = {'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15], 'knn__weights': ['uniform', 'distance'], 'knn__metric': ['euclidean', 'manhattan']}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
y_test = df_test['impact_encoded'].values
knn_rows, cms = [], []

for name, tr_df, feats in experiments:
    grid = GridSearchCV(Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]),
                        param_grid, cv=cv, scoring='f1_macro', n_jobs=1)
    grid.fit(tr_df[feats].values, tr_df['impact_encoded'].values)
    y_pred = grid.best_estimator_.predict(df_test[feats].values)
    _, _, f1_c, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1, 2], zero_division=0)
    cms.append((name, confusion_matrix(y_test, y_pred, labels=[0, 1, 2]), accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro', zero_division=0)))
    knn_rows.append({
        'Model': name, 'K': grid.best_params_['knn__n_neighbors'],
        'Weights': grid.best_params_['knn__weights'], 'Metric': grid.best_params_['knn__metric'],
        'CV Macro F1': round(grid.best_score_, 4),
        'Test Acc': round(accuracy_score(y_test, y_pred), 4),
        'Test Macro F1': round(f1_score(y_test, y_pred, average='macro', zero_division=0), 4),
        'F1 Low': round(f1_c[0], 4), 'F1 Mod': round(f1_c[1], 4), 'F1 High': round(f1_c[2], 4)
    })

y_maj = np.full_like(y_test, 1)
_, _, f1_m, _ = precision_recall_fscore_support(y_test, y_maj, labels=[0, 1, 2], zero_division=0)
knn_rows.append({
    'Model': 'Majority-Class Baseline (Always Moderate)', 'K': 'N/A', 'Weights': 'N/A', 'Metric': 'N/A',
    'CV Macro F1': 'N/A', 'Test Acc': round(accuracy_score(y_test, y_maj), 4),
    'Test Macro F1': round(f1_score(y_test, y_maj, average='macro', zero_division=0), 4),
    'F1 Low': round(f1_m[0], 4), 'F1 Mod': round(f1_m[1], 4), 'F1 High': round(f1_m[2], 4)
})
display(pd.DataFrame(knn_rows))

fig, axes = plt.subplots(1, 4, figsize=(17, 4))
for ax, (name, cm, acc, mf1) in zip(axes, cms):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Low', 'Mod', 'High'], yticklabels=['Low', 'Mod', 'High'], ax=ax)
    ax.set_title(f"{name.split('|')[0].strip()}\\n{name.split('|')[1].strip()} | Acc {acc:.1%}", fontsize=9, fontweight='bold')
    ax.set_xlabel("Pred"); ax.set_ylabel("True")
plt.tight_layout()
plt.show()""")

    nb.add_markdown("## Phase IV: K-Means Clustering ($K \\in [2, 6]$)\nFit K-Means without target labels, inspect Elbow WCSS & Silhouette curves, visualize $K=3$ clusters in 2D PCA space, and inspect descriptive cluster profiles & boundary observations.")

    nb.add_code("""scaler = StandardScaler()
X_150_b = scaler.fit_transform(df_150[config_b])
X_150_a = scaler.fit_transform(df_150[config_a])
y_150 = df_150['impact_category'].values

km_sweep = []
for cfg_lbl, X_mat in [('Config A (18)', X_150_a), ('Config B (14)', X_150_b)]:
    for k in range(2, 7):
        km = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=SEED).fit(X_mat)
        km_sweep.append({'Config': cfg_lbl, 'K': k, 'Inertia': round(km.inertia_, 2),
                         'Silhouette': round(silhouette_score(X_mat, km.labels_), 4),
                         'ARI': round(adjusted_rand_score(y_150, km.labels_), 4),
                         'NMI': round(normalized_mutual_info_score(y_150, km.labels_), 4)})
df_km_sw = pd.DataFrame(km_sweep)
display(df_km_sw)

km3 = KMeans(n_clusters=3, init='k-means++', n_init=20, random_state=SEED)
km_lbls = km3.fit_predict(X_150_b) + 1
sil_s = silhouette_samples(X_150_b, km_lbls - 1)
pca = PCA(n_components=2, random_state=SEED)
Xp = pca.fit_transform(X_150_b)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for cfg_lbl, col in [('Config A (18)', '#1f77b4'), ('Config B (14)', '#2ca02c')]:
    s = df_km_sw[df_km_sw['Config'] == cfg_lbl]
    axes[0].plot(s['K'], s['Inertia'], marker='o', label=cfg_lbl, color=col)
    axes[1].plot(s['K'], s['Silhouette'], marker='s', label=cfg_lbl, color=col)
axes[0].set_title("K-Means Elbow (WCSS)", fontweight='bold'); axes[0].legend(fontsize=8)
axes[1].set_title("K-Means Silhouette vs K", fontweight='bold'); axes[1].legend(fontsize=8)

for cid, col in [(1, '#1f77b4'), (2, '#ff7f0e'), (3, '#2ca02c')]:
    m = km_lbls == cid
    axes[2].scatter(Xp[m, 0], Xp[m, 1], c=col, label=f"C{cid} (N={m.sum()})", alpha=0.75, s=40)
axes[2].scatter(Xp[sil_s < 0, 0], Xp[sil_s < 0, 1], facecolors='none', edgecolors='red', s=90, label=f"Boundary (N={(sil_s<0).sum()})")
axes[2].set_title("K-Means Clusters in 2D PCA (K=3, Config B)", fontweight='bold'); axes[2].legend(fontsize=8)
plt.tight_layout()
plt.show()

df_p = df_150.copy()
df_p['Cluster'] = km_lbls
prof_tbl = df_p.groupby('Cluster')[num_cols].mean().round(2)
prof_tbl.insert(0, 'Size', df_p['Cluster'].value_counts().sort_index())
prof_tbl['Dominant Class'] = pd.crosstab(df_p['Cluster'], df_p['impact_category']).idxmax(axis=1)
display(prof_tbl)""")

    nb.add_markdown("## Phase V: Agglomerative Hierarchical Clustering (Single, Complete, Average, Ward)\nCompare all 4 linkage methods at $K=3$ (Config B), visualize dendrograms, and inspect cluster counts and dominant classes.")

    nb.add_code("""agg_rows, brk_rows = [], []
for link in ['single', 'complete', 'average', 'ward']:
    lbls = AgglomerativeClustering(n_clusters=3, linkage=link).fit_predict(X_150_b) + 1
    agg_rows.append({
        'Linkage': link.capitalize(), 'K': 3,
        'Silhouette': round(silhouette_score(X_150_b, lbls - 1), 4),
        'ARI': round(adjusted_rand_score(y_150, lbls), 4),
        'NMI': round(normalized_mutual_info_score(y_150, lbls), 4),
        'Cluster Sizes': sorted([int((lbls == c).sum()) for c in [1, 2, 3]], reverse=True)
    })
    for cid in [1, 2, 3]:
        m = lbls == cid
        dom = pd.Series(y_150[m]).value_counts().idxmax()
        brk_rows.append({'Linkage': link.capitalize(), 'Cluster': f"C{cid}", 'Size': int(m.sum()), 'Dominant Class': dom})

display(pd.DataFrame(agg_rows))

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
for ax, link in zip(axes, ['single', 'complete', 'average', 'ward']):
    Z = linkage(X_150_b, method=link, metric='euclidean')
    dendrogram(Z, ax=ax, truncate_mode='lastp', p=20, show_leaf_counts=True, color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title(f"{link.capitalize()} Linkage", fontweight='bold')
plt.tight_layout()
plt.show()""")

    nb.add_markdown(r"""## Phase VI: Comparative Analysis (KNN vs. K-Means vs. Agglomerative)

| Evaluation Criteria | KNN | K-Means | Agglomerative |
| :--- | :--- | :--- | :--- |
| **Learning type** | Supervised | Unsupervised | Unsupervised |
| **Labels used during fitting** | Yes (`impact_encoded`) | No (strictly excluded) | No (strictly excluded) |
| **Accuracy on held-out labels ($N=16$)** | 68.75% (Genuine Config B) / 62.50% (Aug. Config A; Baseline = 62.50%) | Not directly applicable | Not directly applicable |
| **Silhouette Score ($K=3$, Config B)** | Not applicable | 0.1904 (Full 150) / 0.2194 (Genuine 54) | Ward: 0.1499 (150) / 0.2298 (54); Single/Avg: 0.2491 (chaining) |
| **ARI / NMI (Full 150, $K=3$, Config B)** | Not applicable | ARI = 0.0061 / NMI = 0.0043 | Ward: ARI = 0.0177 / NMI = 0.0516; Complete: ARI = 0.0398 / NMI = 0.0250 |
| **Main observations** | Captures majority (`Moderate`) and some `High`, but 0% `Low Impact` recall; genuine Config B generalized best. | Config B improves separation; clusters reflect usage/platform habits rather than impact tiers. | Single & Average linkages chain (`136/9/5`), while Ward creates balanced groupings (`75/62/13`). |
| **Limitations** | Small holdout ($N=16$), class imbalance, synthetic data did not improve holdout accuracy. | 21 negative-silhouette boundary points; low post-hoc alignment with subjective impact labels. | Highly sensitive to linkage rule; low post-hoc label alignment. |""")

    output_path = NOTEBOOKS_DIR / "Assignment2_Complete_Submission.ipynb"
    nb.save(output_path)
    shutil.copy(output_path, WORKSPACE_NOTEBOOKS_DIR / "Assignment2_Complete_Submission.ipynb")
    print(f"[SUCCESS] Mirrored Assignment2_Complete_Submission.ipynb to {WORKSPACE_NOTEBOOKS_DIR}")


if __name__ == "__main__":
    build_knn_notebook()
    build_clustering_notebook()
    build_unified_submission_notebook()
