#!/usr/bin/env python3
"""
generate_final_report.py
-------------------------
Generates the clean, black-and-white, Times New Roman assignment report for:
"Digital Device Usage Pattern Analysis Using Machine Learning"

Produces:
- reports/final_report.md
- reports/final_report.pdf
Mirrored to:
- Digital_Device_Usage_Analysis/reports/final_report.md
- Digital_Device_Usage_Analysis/reports/final_report.pdf

Author: Ahmed Younas (Roll No: 26I-7644, MS Artificial Intelligence, Semester 1)
Date: November 2026
"""

import os
import re
import sys
import shutil
import base64
import subprocess
from pathlib import Path
import mistune

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

REPORTS_DIR = PROJECT_ROOT / "reports"
WORKSPACE_REPORTS_DIR = WORKSPACE_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
WORKSPACE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_MD_PATH = REPORTS_DIR / "final_report.md"
WORKSPACE_MD_PATH = WORKSPACE_REPORTS_DIR / "final_report.md"

REPORT_PDF_PATH = REPORTS_DIR / "final_report.pdf"
WORKSPACE_PDF_PATH = WORKSPACE_REPORTS_DIR / "final_report.pdf"


def get_base64_image(image_path: Path) -> str:
    """Encodes an image to base64 for standalone HTML/PDF rendering."""
    if not image_path.exists():
        print(f"[WARN] Image not found: {image_path}", file=sys.stderr)
        return ""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = image_path.suffix.lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{encoded}"


def generate_markdown_content() -> str:
    """Constructs a concise, plain black-and-white assignment report in Markdown."""

    md = r"""# Digital Device Usage Pattern Analysis Using Machine Learning

Author: Ahmed Younas  
Roll No: 26I-7644  
Program: MS Artificial Intelligence, Semester 1  
Course: Advanced Artificial Intelligence (Integrated Practical Assignment)  
Date: November 2026  
Random Seed: 42  
Dataset: 54 Genuine + 96 Synthetic Student Survey Responses (N = 150)

### Google Colab Notebooks
- EDA: [https://colab.research.google.com/drive/1C0wd2HjejbnkiSiieOXYzO5M82DUOVZ5?usp=sharing](https://colab.research.google.com/drive/1C0wd2HjejbnkiSiieOXYzO5M82DUOVZ5?usp=sharing)
- KNN: [https://colab.research.google.com/drive/1L7zu46gCsgoSPekHj5bZPWXHnUFy7bUx?usp=sharing](https://colab.research.google.com/drive/1L7zu46gCsgoSPekHj5bZPWXHnUFy7bUx?usp=sharing)
- Clustering: [https://colab.research.google.com/drive/18UI1BT3uyC-G-71L7YeMT_tlE_6acZdg?usp=sharing](https://colab.research.google.com/drive/18UI1BT3uyC-G-71L7YeMT_tlE_6acZdg?usp=sharing)

---

### Executive Summary
This assignment investigates university students' digital device usage patterns and perceived academic impact through an end-to-end machine learning pipeline covering custom dataset expansion, exploratory data analysis (EDA), feature engineering, supervised classification (KNN), and unsupervised clustering (K-Means and Agglomerative Hierarchical Clustering). Starting from 54 genuine student survey responses, the dataset was expanded to 150 records using a correlation-preserving Multivariate Gaussian Copula. To prevent data leakage in supervised modeling, the 54 genuine records were stratified into 38 training and 16 untouched holdout records, with Phase II synthetic training data fitted strictly on the 38 genuine training records. On the 16-record holdout benchmark, Genuine-Only KNN with composite features achieved the highest accuracy (68.75%, Macro F1 = 0.4000) compared to a 62.50% Majority-Class Baseline (Macro F1 = 0.2564), while all models had 0.0% recall on the minority Low Impact class. In unsupervised clustering (K = 3, Config B), K-Means achieved Silhouette = 0.1904 (Full 150) and 0.2194 (Genuine 54); across all four Agglomerative linkages, Single and Average suffered severe chaining (136/9/5), whereas Ward produced balanced clusters (75/62/13). Post-hoc ARI and NMI values near zero (K-Means ARI = 0.0061; Ward ARI = 0.0177) show that objective device habits do not align linearly with subjective academic distress.

---

## 1. Dataset Collection and Documentation

### 1.1 Collection Protocol and Synthetic Expansion (Option C: Custom Tabular Dataset)
- Primary Source: A voluntary Google Forms survey of 54 genuine university students (Digital Device Usage and Its Impact on Students.csv), with 17 columns, 0 missing values, 0 duplicates, and 100% informed consent.
- Synthetic Expansion (N = 150): To satisfy Option C's minimum requirement of 100 records while preserving empirical distributions and inter-item correlations, 96 synthetic records were generated using a Multivariate Gaussian Copula (with covariance shrinkage alpha = 0.05) for ordinal and Likert items and Dirichlet-smoothed conditional sampling (gamma = 0.5) for nominal primary_activity.
- Anti-Leakage Separation (Phase I vs. Phase II):
  - Phase I Historical Dataset (data/raw/synthetic_responses.csv, N = 96): Generated from all 54 genuine records strictly for exploratory visualization and unsupervised full-dataset analysis (N = 150).
  - Phase II Leakage-Free Split (data/splits/): The 54 genuine responses were stratified 70/30 into 38 genuine training and 16 untouched genuine holdout records. A separate training-only synthetic set (N = 96) was fitted exclusively on the 38 genuine training records, forming an augmented training set of N = 134 while keeping the 16 holdout records completely isolated.

### 1.2 Data Dictionary and Target Variable Derivation

| Field | Survey Attribute | Type / Scale | Numerical Encoding |
| :--- | :--- | :--- | :--- |
| Q1 | Gender | Nominal (Female, Male) | Binary (0 = Female, 1 = Male) |
| Q2 | Academic Year | Nominal (1st, 2nd, 3rd, 4th, Other) | 5 One-Hot Binary Columns |
| Q3 | Daily Device Hours | Ordinal (5 brackets: <2h to >8h) | Integer 1 to 5 |
| Q4 | Primary Activity | Nominal (6 domains) | 6 One-Hot Binary Columns |
| Q5 | Phone Checking Frequency | Ordinal (<5/day to Continuous) | Integer 1 to 5 |
| Q6-Q8 | Academic Impact (Lectures, Study Time, Tasks) | 5-pt Likert (Strongly Disagree to Agree) | Target basis only (excluded from inputs) |
| Q9-Q10 | Phone with Family, Conversation Interruption | 5-pt Likert | Integer 1 to 5 |
| Q11 | Family Usage Complaints | 3-pt Ordinal (No, Sometimes, Yes) | 1, 2, 3 rescaled as 2*Q11 - 1 in {1, 3, 5} |
| Q12 | Difficulty Staying Away from Phone | 5-pt Ordinal (Very easy to Very difficult) | Integer 1 to 5 |
| Q13-Q15 | Open-Ended Qualitative Text | Free text | Excluded from numerical modeling |

Target Variable (impact_category): Computed as academic_impact_avg = (Q6 + Q7 + Q8) / 3 and discretized into Low Impact (<= 2.33), Moderate Impact (2.33 to 3.66), and High Impact (> 3.66). Genuine distribution (N = 54): Moderate = 32 (59.3%), High = 15 (27.8%), Low = 7 (13.0%). Full 150 distribution: Moderate = 79 (52.7%), High = 53 (35.3%), Low = 18 (12.0%).

---

## 2. Exploratory Data Analysis (EDA)

### 2.1 Data Quality, Univariate and Bivariate Findings
- Quality Audit: 0 missing values, 0 duplicate rows, and 100% valid categorical and Likert ranges across all 150 records.
- Usage Patterns: 84.0% of respondents use devices more than 4 hours/day (modal bracket: 4 to 6 hours, 42.7%), and 58.0% check smartphones more than 20 times/day or continuously. Social media is the top primary activity (44.0%), followed by Educational activities (20.7%) and Video entertainment (15.3%).
- Habit Correlations: Screen time correlates strongly with checking frequency (Spearman rho = 0.58, p < 0.001), and checking frequency correlates strongly with separation difficulty (rho = 0.62, p < 0.001). Primary activity is associated with impact tier (chi-squared = 24.8, p < 0.001), with higher High Impact share among social media and video users than educational users.
- Box Plots and Outlier Diagnostics: Box-plot IQR analysis (14_numerical_feature_boxplots.png) confirms that because all behavioral ratings are bounded on [1, 5] Likert scales, no invalid or extreme numerical outliers require removal.
- Real (N = 54) vs. Synthetic (N = 96) Fidelity: Two-sample Kolmogorov-Smirnov tests yielded p > 0.05 across all 11 ordinal dimensions (mean KS statistic D = 0.084 +/- 0.021), and the mean absolute rank-correlation residual was 0.068, confirming high statistical fidelity without record duplication.

![Figure 1: Daily device screen time and primary digital activities](eda/02_screentime_and_primary_activity.png)
![Figure 2: Target class balance across datasets](eda/06_target_impact_category_distribution.png)
![Figure 3: Numerical feature box plots and outlier check](eda/14_numerical_feature_boxplots.png)
![Figure 4: Real vs. synthetic distribution fidelity](eda/16_real_vs_synthetic_distributions.png)

---

## 3. Data Preprocessing and Feature Engineering

### 3.1 Preprocessing Decisions and Feature Extraction
1. Academic Year Encoding: Replaced arbitrary ordinal coding (which misordered "Other") with 5 one-hot binary indicators (year_First through year_Other) to avoid artificial distance assumptions.
2. Symmetric Social Conflict Score: Linearly mapped 3-point Q11 onto {1, 3, 5} via Q11_scaled = 2 * Q11 - 1, then computed social_conflict_score = (Q9 + Q10 + Q11_scaled) / 3 in [1.0, 5.0] (mean = 2.14 +/- 0.82).
3. Digital Dependency Score: Aggregated screen exposure, compulsive checking, and detachment difficulty as digital_dependency_score = (daily_device_hours + check_frequency + stay_away_difficulty) / 3 in [1.0, 5.0] (mean = 3.37 +/- 0.77).
4. Feature Scaling: Applied StandardScaler (zero mean, unit variance) so multi-level Likert features do not dominate 0/1 one-hot indicators in distance-based models. In supervised KNN, scaling is fitted strictly inside training folds via a scikit-learn Pipeline.

### 3.2 Audited Feature Configurations (Zero Target Leakage)
- Configuration A (Atomic, 18 features): gender_encoded, 5 one-hot academic years, 6 raw behavioral items (daily_device_hours, check_frequency, phone_with_family, conversation_interruption, social_complaints, stay_away_difficulty), and 6 one-hot primary activities.
- Configuration B (Composite, 14 features): Replaces the 6 raw behavioral items with the 2 normalized composite indices (social_conflict_score, digital_dependency_score) alongside gender, year, and activity indicators (no double-counting).
- Independent Target Isolation Verification: Direct inspection of both feature column lists confirms strict exclusion of academic_impact_avg, impact_category, impact_encoded, Q6, Q7, Q8, timestamps, and IDs.

![Figure 5: Spearman correlation heatmap across input features](eda/15_input_features_correlation_heatmap.png)

---

## 4. KNN Classification (Supervised Learning)

### 4.1 Experimental Setup and Cross-Validation
- Grid Search: Evaluated 28 combinations per condition across K in {3, 5, 7, 9, 11, 13, 15}, weighting in {uniform, distance}, and distance metric in {Euclidean (L2), Manhattan (L1)}, optimized for Macro F1 using Stratified 5-Fold CV (random_state = 42). Because the genuine training set (N = 38) has only 5 Low Impact records, 5 folds is the exact upper bound ensuring every fold contains at least one Low Impact sample.
- Holdout Benchmark (N = 16 genuine): Low = 2 (12.5%), Moderate = 10 (62.5%), High = 4 (25.0%). Evaluated alongside a Majority-Class Baseline (always predicting Moderate Impact).

### 4.2 Holdout Results and Majority-Class Baseline Comparison

| Experimental Condition | Train N | Feats | Best Params (K, Weight, Metric) | 5-Fold CV Macro F1 | Holdout Acc (N=16) | Holdout Macro F1 | Holdout W-F1 | F1 Low (N=2) | F1 Mod (N=10) | F1 High (N=4) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Augmented / Config A (Atomic) | 134 | 18 | K=15, Distance, Manhattan | 0.5976 | 62.50% | 0.4091 | 0.5795 | 0.0000 | 0.7273 | 0.5000 |
| Augmented / Config B (Composite) | 134 | 14 | K=9, Distance, Euclidean | 0.5754 | 31.25% | 0.1667 | 0.3125 | 0.0000 | 0.5000 | 0.0000 |
| Genuine Only / Config A (Atomic) | 38 | 18 | K=9, Uniform, Euclidean | 0.3014 | 56.25% | 0.2400 | 0.4500 | 0.0000 | 0.7200 | 0.0000 |
| Genuine Only / Config B (Composite) | 38 | 14 | K=11, Uniform, Manhattan | 0.3231 | 68.75% | 0.4000 | 0.6000 | 0.0000 | 0.8000 | 0.4000 |
| Majority-Class Baseline | N/A | N/A | Always Moderate | N/A | 62.50% | 0.2564 | 0.4808 | 0.0000 | 0.7692 | 0.0000 |

### 4.3 Evaluation, Error Analysis and Key Findings
- Synthetic Augmentation Did Not Improve Holdout Generalization: Synthetic augmentation increased internal 5-fold CV Macro F1 on Config A (0.5976 vs. 0.3014), but this did not consistently transfer to unseen genuine students. Genuine-Only Config B achieved the best holdout accuracy (68.75%, 11/16 correct, Weighted F1 = 0.6000), while Augmented Config B degraded to 31.25%.
- Insufficiency of Accuracy: The trivial Majority-Class Baseline achieves 62.50% accuracy (matching Augmented Config A) with only 0.2564 Macro F1, proving Macro F1 and per-class recall must be prioritized over raw accuracy.
- Misclassification Analysis (0.0% Recall on Low Impact): All models misclassified both Low Impact holdout students as Moderate Impact. Without target-derived distraction questions (Q6-Q8), Low Impact students overlap with Moderate Impact neighbors in screen-time and checking features, and the majority class (57.9% of training data) dominates local neighborhoods.
- Empirical Note on Manhattan Distance (L1): Manhattan distance was selected in the two strongest holdout models (Augmented Config A and Genuine Config B) for this dataset and feature encoding. This is an empirical result specific to these configurations and should not be generalized as universal superiority of L1 over L2 for one-hot encoded data.

![Figure 6: Stratified 5-Fold CV Macro F1 tuning curves](phase_ii/01_knn_cv_hyperparameter_tuning.png)
![Figure 7: Holdout confusion matrices across all 4 KNN conditions](phase_ii/02_knn_confusion_matrices.png)

---

## 5. K-Means Clustering (Unsupervised Learning)

### 5.1 Elbow Method and Silhouette Selection (K = 2 to 6)
K-Means (init = k-means++, n_init = 20, random_state = 42) minimizes within-cluster sum of squares (WCSS / Inertia) without access to target labels. Config B (Composite) outperformed Config A across every K on both datasets (+25.3% Silhouette at K = 3 on Full 150; +49.1% on Genuine 54) because condensing correlated items reduced high-dimensional noise. K = 3 was selected at the Elbow inflection point.

| Dataset Scope | Feature Config | K=2 Inertia (Sil) | K=3 Inertia (Sil) | K=4 Inertia (Sil) | K=5 Inertia (Sil) | K=6 Inertia (Sil) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Full 150 Augmented | Config A (18 feats) | 2439.37 (0.1201) | 2245.43 (0.1519) | 2072.65 (0.1495) | 1925.26 (0.1761) | 1724.23 (0.2020) |
| Full 150 Augmented | Config B (14 feats) | 1842.21 (0.1649) | 1644.67 (0.1904) | 1481.84 (0.2169) | 1332.88 (0.2232) | 1171.84 (0.2199) |
| Genuine 54 Only | Config A (18 feats) | 856.66 (0.1305) | 774.48 (0.1471) | 699.59 (0.1313) | 628.83 (0.1530) | 563.93 (0.1684) |
| Genuine 54 Only | Config B (14 feats) | 668.62 (0.1689) | 572.28 (0.2194) | 492.24 (0.2195) | 429.34 (0.2499) | 377.09 (0.2779) |

### 5.2 Cluster Profiles, Dominant Classes and Unexpected Observations (Full 150, K = 3, Config B)
Clusters are assigned neutral labels (Cluster 1 to 3) with purely descriptive centroid means (inferential ANOVA p-values are intentionally omitted to avoid circular double-dipping hypothesis testing):

| Cluster | Count (%) | Daily Hours | Check Freq | Social Conflict | Digital Depend. | Stay-Away Diff. | Top Primary Activities | Dominant Actual Class |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| Cluster 1 | 85 (56.7%) | 3.51 | 3.92 | 2.96 | 3.49 | 3.06 | Social Media (41.2%), Edu. (24.7%) | Moderate (46/85 = 54.1%) |
| Cluster 2 | 38 (25.3%) | 3.66 | 3.58 | 3.01 | 3.38 | 2.89 | Social Media (34.2%), Edu. (23.7%), Other (15.8%) | Moderate (19/38 = 50.0%) |
| Cluster 3 | 27 (18.0%) | 3.59 | 4.19 | 3.02 | 3.49 | 2.70 | Social Media (55.6%), Gaming (25.9%) | Moderate (14/27 = 51.9%) |

- Unexpected / Boundary Cluster Observations: Sample-level silhouette inspection flagged 21 of 150 observations (7 genuine) with negative silhouette scores (-0.0364 to -0.001), indicating boundary cases positioned closer to a neighboring centroid. For example, Row 93 reports > 8 hours/day (5) and > 20 checks/day (4) yet Low Impact (Silhouette = -0.0364), while genuine student Row 37 reports low phone-checking (2) yet elevated social conflict (4.33) and High Impact (Silhouette = -0.0223).
- External Alignment: Post-hoc comparison against impact_category yielded ARI = 0.0061, NMI = 0.0043 (Full 150) and ARI = 0.0708, NMI = 0.0652 (Genuine 54). All three clusters contain a mix of Low, Moderate, and High Impact students, showing that K-Means partitions students primarily by platform and demographic usage profiles rather than subjective academic distress.

![Figure 8: K-Means Inertia Elbow and Silhouette curves](phase_ii/05_kmeans_elbow_and_silhouette.png)
![Figure 9: K-Means clusters vs. ground-truth labels in 2D PCA space](phase_ii/10_kmeans_pca_clusters.png)

---

## 6. Agglomerative Hierarchical Clustering

### 6.1 Linkage Comparison: Single, Complete, Average and Ward (Config B, K = 3)
All four required linkage methods were evaluated across K in [2, 6] using Euclidean distance on standardized features:

| Dataset | Linkage | Silhouette (K=3) | ARI | NMI | Cluster Sizes (C1 / C2 / C3) | Dominant Actual Class per Cluster | Structural Observation |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| Full 150 | Single | 0.2491 | 0.0332 | 0.0351 | 5 / 9 / 136 | High (60%), Mod (56%), Mod (54%) | Severe chaining: 90.7% in 1 cluster |
| Full 150 | Complete | 0.2056 | 0.0398 | 0.0250 | 32 / 5 / 113 | Mod (50%), High (60%), Mod (55%) | Moderate chaining + 5-point outlier pocket |
| Full 150 | Average | 0.2491 | 0.0332 | 0.0351 | 136 / 5 / 9 | Mod (54%), High (60%), Mod (56%) | Same skewed split as Single linkage |
| Full 150 | Ward | 0.1499 | 0.0177 | 0.0516 | 75 / 62 / 13 | Mod (47%), Mod (63%), Mod/High tie (38.5%) | Most balanced and interpretable partition |
| Genuine 54 | Single | 0.3169 | 0.0389 | 0.0287 | 49 / 2 / 3 | Mod (61%), Mod/High (50%), Low/Mod/High | Extreme chaining (90.7% in 1 cluster) |
| Genuine 54 | Complete | 0.2210 | 0.0674 | 0.1173 | 14 / 38 / 2 | Mod (57%), Mod (61%), Mod/High (50%) | Two main groups + 2-student outlier |
| Genuine 54 | Average | 0.3169 | 0.0389 | 0.0287 | 49 / 2 / 3 | Mod (61%), Mod/High (50%), Low/Mod/High | Identical chaining to Single linkage |
| Genuine 54 | Ward | 0.2298 | 0.0285 | 0.0297 | 15 / 36 / 3 | Mod (60%), Mod (61%), Low/Mod/High | Balanced variance-minimizing merges |

### 6.2 Dendrogram Interpretation and Target-Isolation Clarification
- Impact of Linkage Choice: Different linkage methods produce substantially different groupings. Single and Average linkages isolate small outlier pockets of rare categorical combinations (5 to 9 students) while chaining 90.7% of records into one giant cluster, artificially inflating Silhouette (0.2491 to 0.3169) without yielding useful segments. Ward linkage minimizes within-cluster variance increase (delta ESS), producing balanced, interpretable branches (75 / 62 / 13 on Full 150).
- Target Isolation vs. Label Alignment: Near-zero ARI (0.0177 to 0.0674) and NMI (0.0250 to 0.1173) measure low external concordance between unsupervised clusters and impact_category; they do not serve as proof of target isolation. Complete target exclusion is verified independently by inspecting the input feature columns (which strictly omit Q6-Q8 and all target variables).

![Figure 10: Agglomerative dendrograms across all 4 linkages](phase_ii/06_hierarchical_dendrograms.png)
![Figure 11: Post-hoc contingency heatmaps](phase_ii/08_cluster_contingency_heatmaps.png)

---

## 7. Comparative Analysis and Final Interpretation

### 7.1 Algorithm Comparison Table (KNN vs. K-Means vs. Agglomerative)

| Evaluation Criteria | KNN | K-Means | Agglomerative |
| :--- | :--- | :--- | :--- |
| Learning type | Supervised classification | Unsupervised partitioning | Unsupervised hierarchical |
| Labels used during fitting | Yes (impact_encoded) | No (strictly excluded) | No (strictly excluded) |
| Accuracy on held-out labels (N = 16) | 68.75% (Genuine Config B) / 62.50% (Aug. Config A; Majority Baseline = 62.50%) | Not directly applicable | Not directly applicable |
| Silhouette Score (K = 3, Config B) | Not applicable | 0.1904 (Full 150) / 0.2194 (Genuine 54) | Ward: 0.1499 (150) / 0.2298 (54); Single/Avg: 0.2491 (chaining) |
| ARI / NMI (Full 150, K = 3, Config B) | Not applicable | ARI = 0.0061 / NMI = 0.0043 | Ward: ARI = 0.0177 / NMI = 0.0516; Complete: ARI = 0.0398 / NMI = 0.0250 |
| Main observations | Predicts Moderate and some High students, but 0% recall on Low Impact. Genuine Config B generalized best. | Config B improves cluster separation over Config A; clusters reflect platform and usage habits across impact tiers. | Single and Average suffer severe chaining (136/9/5); Ward yields balanced, interpretable groupings (75/62/13). |
| Limitations | Very small holdout (N = 16), class imbalance, synthetic data improved CV but not holdout accuracy. | Moderate silhouette (0.19 to 0.22); 21 negative-silhouette boundary points; low label alignment. | Highly sensitive to linkage choice; Ward avoids chaining at lower raw silhouette; low label alignment. |

### 7.2 Synthesis: Feature Representations and Behavioral Decoupling
- Atomic (Config A) vs. Composite (Config B): Condensing collinear items into normalized behavioral indices (Config B) consistently improved both Genuine-Only KNN holdout accuracy (68.75% vs. 56.25%) and unsupervised Silhouette separation (+25.3% to +49.1%).
- Observable Habits vs. Perceived Distress: Both supervised and unsupervised results show that objective device habits alone are insufficient to separate subjective academic impact tiers cleanly. Whether unobserved factors, such as individual self-discipline, study strategies, time management, or course workload, moderate the relationship between screen time and academic impact remains an open hypothesis for future empirical research, not a finding established by this cross-sectional experiment.

---

## 8. Limitations and Conclusion

### Key Limitations
1. Small Holdout Benchmark (N = 16): Each single student misclassification shifts holdout accuracy by 6.25%. Results reflect a pilot-scale genuine evaluation, not validation on 150 genuine students.
2. Self-Report and Cross-Sectional Bias: Survey responses rely on subjective recall rather than objective OS screen-time telemetry and capture a single point in time, preventing any causal claims.
3. Minority-Class Recall Failure: Zero recall on Low Impact (N = 2 holdout) underscores the difficulty of detecting minority profiles under class imbalance without direct academic distraction items.

### Conclusion
This project demonstrated a complete, leakage-free machine learning workflow from custom survey data expansion through EDA, feature engineering, KNN classification, K-Means clustering, and 4-linkage Agglomerative clustering. Key conclusions are: (1) synthetic copula augmentation improved internal CV scores but did not improve genuine holdout generalization; (2) Macro F1 and per-class recall are essential alongside a Majority-Class Baseline on imbalanced cohorts; (3) Ward linkage and Composite features (Config B) produced the most balanced unsupervised groupings; and (4) future studies should pair passive device telemetry with longitudinal academic records to test unobserved moderating variables.

---

## 9. References
1. Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. SODA, 1027-1035.
2. Hubert, L., & Arabie, P. (1985). Comparing partitions. Journal of Classification, 2(1), 193-218.
3. Lepp, A., Barkley, J. E., & Karpinski, A. C. (2014). The relationship between cell phone use, academic performance, anxiety, and satisfaction with life in college students. Computers in Human Behavior, 31, 343-350.
4. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. Proc. 5th Berkeley Symp., 1, 281-297.
5. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR, 12, 2825-2830.
6. Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. J. Comput. Appl. Math., 20, 53-65.
7. Sklar, A. (1959). Fonctions de repartition a n dimensions et leurs marges. Publ. Inst. Statist. Univ. Paris, 8, 229-231.
8. Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. JASA, 58(301), 236-244.
"""
    assert "—" not in md, "Em-dash found in markdown!"
    return md


def build_markdown_report() -> str:
    return generate_markdown_content()


def build_html_report(md_text: str) -> str:
    """Renders simple black-and-white Times New Roman HTML with embedded base64 figures and no headers/footers."""
    eda_dir = PROJECT_ROOT / "reports" / "eda"
    p2_dir = PROJECT_ROOT / "reports" / "phase_ii"

    figs = {
        "eda_act": get_base64_image(eda_dir / "02_screentime_and_primary_activity.png"),
        "eda_tgt": get_base64_image(eda_dir / "06_target_impact_category_distribution.png"),
        "eda_box": get_base64_image(eda_dir / "14_numerical_feature_boxplots.png"),
        "eda_fid": get_base64_image(eda_dir / "16_real_vs_synthetic_distributions.png"),
        "eda_cor": get_base64_image(eda_dir / "15_input_features_correlation_heatmap.png"),
        "knn_cv":  get_base64_image(p2_dir / "01_knn_cv_hyperparameter_tuning.png"),
        "knn_cm":  get_base64_image(p2_dir / "02_knn_confusion_matrices.png"),
        "km_elb":  get_base64_image(p2_dir / "05_kmeans_elbow_and_silhouette.png"),
        "km_pca":  get_base64_image(p2_dir / "10_kmeans_pca_clusters.png"),
        "agg_den": get_base64_image(p2_dir / "06_hierarchical_dendrograms.png"),
        "agg_ct":  get_base64_image(p2_dir / "08_cluster_contingency_heatmaps.png"),
    }

    def fig_single(b64, num, caption, max_h=245):
        if not b64:
            return ""
        return f"""
        <div class="figure-container">
            <img src="{b64}" alt="Figure {num}" style="max-height:{max_h}px;">
            <div class="figure-caption">Figure {num}: {caption}</div>
        </div>"""

    def fig_pair(b64_l, num_l, cap_l, b64_r, num_r, cap_r, max_h=215):
        return f"""
        <div class="fig-grid">
            <div class="fig-item">
                <img src="{b64_l}" alt="Figure {num_l}" style="max-height:{max_h}px;">
                <div class="figure-caption">Figure {num_l}: {cap_l}</div>
            </div>
            <div class="fig-item">
                <img src="{b64_r}" alt="Figure {num_r}" style="max-height:{max_h}px;">
                <div class="figure-caption">Figure {num_r}: {cap_r}</div>
            </div>
        </div>"""

    md_for_html = re.sub(r'^!\[.*?\]\(.*?\)\s*$', '', md_text, flags=re.MULTILINE)
    markdown_parser = mistune.create_markdown(plugins=['table', 'strikethrough'])
    html_body = markdown_parser(md_for_html)

    injections = {
        "<h2>3. Data Preprocessing": (
            fig_pair(
                figs["eda_act"], "1", "Daily device screen time and primary digital activities (N = 150).",
                figs["eda_tgt"], "2", "Target class balance (Low, Moderate, High Impact) across datasets.", max_h=205
            )
            + fig_pair(
                figs["eda_box"], "3", "Numerical feature box plots confirming bounded Likert ranges and zero extreme outliers.",
                figs["eda_fid"], "4", "Real (N = 54) vs. synthetic (N = 96) distribution fidelity (KS p > 0.05).", max_h=205
            )
            + "<h2>3. Data Preprocessing"
        ),
        "<h2>4. KNN Classification": (
            fig_single(
                figs["eda_cor"], "5", "Spearman rank correlation heatmap across input features (target items Q6-Q8 excluded).", max_h=245
            )
            + "<h2>4. KNN Classification"
        ),
        "<h2>5. K-Means Clustering": (
            fig_pair(
                figs["knn_cv"], "6", "Stratified 5-Fold CV Macro F1 tuning curves across K, weights, and metrics.",
                figs["knn_cm"], "7", "Holdout confusion matrices (N = 16 genuine) across all 4 KNN conditions.", max_h=225
            )
            + "<h2>5. K-Means Clustering"
        ),
        "<h2>6. Agglomerative Hierarchical Clustering": (
            fig_pair(
                figs["km_elb"], "8", "K-Means Inertia Elbow and Silhouette curves across K = 2 to 6 (Config A vs. B).",
                figs["km_pca"], "9", "K-Means clusters vs. ground-truth labels in 2D PCA space (Full 150, K = 3, Config B).", max_h=225
            )
            + "<h2>6. Agglomerative Hierarchical Clustering"
        ),
        "<h2>7. Comparative Analysis": (
            fig_single(
                figs["agg_den"], "10", "Agglomerative dendrograms across all 4 linkages (Single, Complete, Average, Ward; Config B).", max_h=245
            )
            + fig_single(
                figs["agg_ct"], "11", "Post-hoc contingency heatmaps comparing K = 3 unsupervised clusters to actual impact categories.", max_h=205
            )
            + "<h2>7. Comparative Analysis"
        ),
    }

    for marker, snippet in injections.items():
        assert marker in html_body, f"Missing HTML injection anchor: {marker}"
        html_body = html_body.replace(marker, snippet, 1)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Digital Device Usage Pattern Analysis</title>
<style>
    @page {{
        size: A4;
        margin: 18mm 16mm 18mm 16mm;
    }}
    body {{
        font-family: 'Times New Roman', Times, serif;
        font-size: 11pt;
        line-height: 1.45;
        color: #000000;
        background-color: #ffffff;
        margin: 0;
        padding: 0;
    }}
    h1 {{
        font-family: 'Times New Roman', Times, serif;
        font-size: 16pt;
        font-weight: bold;
        color: #000000;
        border-bottom: 1px solid #000000;
        padding-bottom: 4px;
        margin: 0 0 8px 0;
    }}
    h2 {{
        font-family: 'Times New Roman', Times, serif;
        font-size: 13pt;
        font-weight: bold;
        color: #000000;
        border-bottom: 1px solid #000000;
        padding-bottom: 2px;
        margin: 14px 0 6px 0;
        page-break-after: avoid;
    }}
    h3 {{
        font-family: 'Times New Roman', Times, serif;
        font-size: 11.5pt;
        font-weight: bold;
        color: #000000;
        margin: 10px 0 4px 0;
        page-break-after: avoid;
    }}
    p {{
        margin: 4px 0 6px 0;
        font-weight: normal;
        color: #000000;
        text-align: justify;
    }}
    ul, ol {{
        margin: 4px 0 6px 20px;
        padding: 0;
        font-weight: normal;
        color: #000000;
    }}
    li {{
        margin-bottom: 3px;
        font-weight: normal;
        color: #000000;
    }}
    strong, b {{
        font-weight: normal;
    }}
    a {{
        color: #000000;
        text-decoration: underline;
        font-weight: normal;
        font-size: 9.5pt;
    }}
    code {{
        font-family: 'Times New Roman', Times, serif;
        font-size: inherit;
        background: none;
        padding: 0;
        color: #000000;
    }}
    hr {{
        border: 0;
        border-top: 1px solid #000000;
        margin: 10px 0;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 8px 0 10px 0;
        font-family: 'Times New Roman', Times, serif;
        font-size: 9pt;
        color: #000000;
        background-color: #ffffff;
        page-break-inside: avoid;
    }}
    th {{
        background-color: #ffffff;
        border: 1px solid #000000;
        padding: 4px 5px;
        font-weight: bold;
        text-align: left;
        color: #000000;
    }}
    td {{
        background-color: #ffffff;
        border: 1px solid #000000;
        padding: 4px 5px;
        vertical-align: top;
        font-weight: normal;
        color: #000000;
    }}
    .figure-container {{
        text-align: center;
        margin: 8px auto 10px auto;
        page-break-inside: avoid;
    }}
    .figure-container img {{
        max-width: 96%;
        height: auto;
        border: 1px solid #000000;
    }}
    .fig-grid {{
        display: flex;
        justify-content: space-between;
        gap: 10px;
        margin: 8px 0 10px 0;
        page-break-inside: avoid;
    }}
    .fig-item {{
        flex: 1;
        text-align: center;
    }}
    .fig-item img {{
        max-width: 100%;
        height: auto;
        border: 1px solid #000000;
    }}
    .figure-caption {{
        font-family: 'Times New Roman', Times, serif;
        font-size: 9pt;
        font-weight: normal;
        color: #000000;
        margin-top: 3px;
        text-align: center;
    }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""
    assert "—" not in full_html, "Em-dash found in HTML!"
    return full_html


def compile_pdf_with_chrome(html_content: str, output_pdf_path: Path):
    """Compiles HTML to PDF via headless Chrome or Edge without browser headers or footers."""
    temp_html = output_pdf_path.parent / "temp_report.html"
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if not os.path.exists(chrome_path):
        chrome_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

    cmd = [
        chrome_path,
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--no-pdf-header-footer",
        f"--print-to-pdf={output_pdf_path.resolve()}",
        str(temp_html.resolve())
    ]

    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode != 0:
        raise RuntimeError(f"PDF compilation failed ({ret.returncode}): {ret.stderr}")

    if temp_html.exists():
        temp_html.unlink()

    print(f"[SUCCESS] Compiled PDF ({output_pdf_path.stat().st_size / 1024:.1f} KB): {output_pdf_path}")


def main():
    md_content = build_markdown_report()
    for p in [REPORT_MD_PATH, WORKSPACE_MD_PATH]:
        with open(p, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"[SUCCESS] Saved Markdown ({len(md_content):,} bytes): {p}")

    html_content = build_html_report(md_content)
    compile_pdf_with_chrome(html_content, REPORT_PDF_PATH)
    shutil.copy(REPORT_PDF_PATH, WORKSPACE_PDF_PATH)
    print(f"[SUCCESS] Mirrored PDF: {WORKSPACE_PDF_PATH}")


if __name__ == "__main__":
    main()
