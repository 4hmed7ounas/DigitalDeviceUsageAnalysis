# Phase I Review and Phase II Preparation Report

**Project Title:** Digital Device Usage Pattern Analysis Using Machine Learning  
**Author:** Senior Data Scientist & Machine Learning Engineer  
**Stage:** Phase I Audit & Phase II Preparation  
**Date:** November 2026  

---

## Executive Summary

Before implementing machine learning models (KNN classification, K-Means clustering, and Agglomerative hierarchical clustering), this report performs a rigorous audit of the completed Phase I Exploratory Data Analysis (EDA) notebook ([`01_exploratory_data_analysis.ipynb`](file:///c:/Ahmed/MSAI/MS%20Semester%201/Adv%20AI/Assignment/Assignment2/Digital_Device_Usage_Analysis/notebooks/01_exploratory_data_analysis.ipynb)), the underlying datasets, and the preprocessing scripts. 

The audit identified specific numerical discrepancies, methodological nuances in statistical tests, an encoding limitation in the academic-year attribute, and a minor scaling asymmetry in one composite metric. Furthermore, this document establishes the formal specifications for **two distinct modeling feature sets (Atomic vs. Composite)** and defines a **leakage-free, scientifically honest evaluation strategy** designed for small, imbalanced survey cohorts.

---

## 1. Audit of Phase I EDA Findings & Discovered Issues

### 1.1 Numerical & Percentage Discrepancies in EDA Markdown Commentary
A comparison between the programmatic execution outputs and the Markdown narrative in the EDA notebook revealed that several narrative cells contained hardcoded percentages derived from either the original genuine subset ($N=54$) or preliminary drafts, rather than dynamically reflecting the final combined dataset ($N=150$):

| Attribute / Question | Markdown Narrative Text | Actual Combined Dataset ($N=150$) | Discrepancy Cause & Impact |
|:---|:---:|:---:|:---|
| **Gender (Q1)** | 76 Male (50.7%) / 74 Female (49.3%) | **80 Male (53.33%) / 70 Female (46.67%)** | Narrative text used preliminary draft counts; plots rendered the correct data ($80/70$). |
| **Academic Year (Q2)** | Third year: 59.3% ($N=89$) | **Third year: 56.67% ($N=85$)** | 59.3% was the exact percentage in the genuine $N=54$ sample ($32/54=59.26\%$), incorrectly cited as the combined $N=150$ count. |
| **Primary Activity (Q4)** | Social Media: 44.7% ($N=67$) | **Social Media: 42.00% ($N=63$)** | Narrative text used draft numbers; actual count in combined data is 63 students. |
| **Checking Frequency (Q5)** | Almost continuously: 38.7% ($N=58$) | **Almost continuously: 43.33% ($N=65$)** | Narrative understated continuous checkers by 7 students; plots correctly display 65. |
| **Screen Time (Q3)** | $>8$ hours: 25.3%, 6–8 hours: 26.7% | **$>8$ hours: 30.00% ($N=45$), 6–8 hours: 24.67% ($N=37$)** | Heavy screen exposure is even more pronounced ($54.67\%$ spend $>6$ hours daily). |

*Correction Recommended:* In the final academic report, all summary tables and narrative paragraphs must strictly draw from the verified `combined_dataset.csv` execution tallies.

---

### 1.2 Evaluation of Real-versus-Synthetic Statistical Tests
In Section 6 of the EDA notebook, two-sample Kolmogorov-Smirnov (KS) tests (`stats.ks_2samp`) and Mann-Whitney U tests (`stats.mannwhitneyu`) were applied to evaluate whether synthetic features follow the same continuous probability distributions as genuine survey data.

- **Methodological Nuance:** The classical two-sample Kolmogorov-Smirnov test is mathematically formulated under the assumption of **continuous** underlying distributions. When applied to discrete ordinal survey items with heavy ties ($1, 2, 3, 4, 5$), the test statistic $D$ remains informative as the maximum vertical distance between empirical CDFs, but the asymptotic p-values become **conservative** (i.e., p-values tend to be higher than their true continuous equivalents).
- **Sample Size Constraints:** The genuine sample consists of $N_1 = 54$ records, while the synthetic sample has $N_2 = 96$ records. The unequal ratio ($1 : 1.78$) is well handled by rank-based non-parametric tests, but statistical power is naturally constrained for low-frequency classes (such as Low Impact, where only 7 genuine examples exist).
- *Correction Recommended:* In the final academic methodology report, explicitly clarify that:
  1. The **Mann-Whitney U test** (which explicitly handles ties via mid-rank adjustments) and **Chi-Square test of homogeneity** serve as the primary discrete non-parametric metrics.
  2. The KS test is reported as a supplementary metric measuring maximum cumulative distribution divergence ($\sup |F_{\text{real}}(x) - F_{\text{syn}}(x)| \le 0.15$).

---

### 1.3 Academic Year Encoding: The "Other" Category Problem
In `generate_dataset.py` and `feature_dataset.csv`, the academic year feature was encoded as:
$$\text{year\_map} = \{\text{'First year'}: 1, \text{'Second year'}: 2, \text{'Third year'}: 3, \text{'Fourth year'}: 4, \text{'Other'}: 3\}$$

- **Critique:** Mapping `'Other'` to integer `3` (Third year) was implemented based on mode/median assignment. However, conceptually, students selecting "Other" typically represent postgraduate students (Master's, PhD), exchange students, or diploma candidates.
- **Impact on Machine Learning Models:**
  - For distance-based algorithms like **KNN** and **K-Means**, assigning `'Other'` as `3` forces them to be equidistant to Second-year (distance $= 1$) and Fourth-year (distance $= 1$) undergraduates, falsely asserting an undergraduate junior progression level.
- *Correction Recommended:* For Phase II modeling:
  - In **Configuration A (Atomic)**, academic year should either be one-hot encoded or tested both ways (ordinal vs. one-hot) to confirm that the mode-imputed ordinal value does not distort KNN decision boundaries.
  - Document this encoding as an explicit constraint of the survey schema in the final report.

---

### 1.4 Calculation Audit of Composite Scores
Two composite metrics were engineered to summarize multi-question behavioral constructs:

#### A. `digital_dependency_score`
$$\text{digital\_dependency\_score} = \frac{\text{daily\_device\_hours} + \text{check\_frequency} + \text{stay\_away\_difficulty}}{3}$$
- **Audit Result:** **100% Correct.** All three constituent features operate on the identical $[1, 5]$ ordinal scale with equal psychological weighting. The composite metric yields a clean, symmetric distribution spanning $[1.33, 5.00]$ ($\mu = 3.46, \sigma = 0.88$).

#### B. `social_conflict_score`
$$\text{social\_conflict\_score} = \frac{\text{phone\_with\_family} + \text{conversation\_interruption} + (\text{social\_complaints} \times 1.67)}{3}$$
- **Audit Result:** **Scaling Asymmetry Discovered.**
  - `phone_with_family` (Q9) is on $[1, 5]$.
  - `conversation_interruption` (Q10) is on $[1, 5]$.
  - `social_complaints` (Q11) has three categorical values: $\text{'No'} = 1, \text{'Sometimes'} = 2, \text{'Yes'} = 3$.
  - The script multiplied Q11 by $1.67$ ($3 \times 1.67 \approx 5.01$). However, multiplying without centering shifts the minimum: for $\text{'No'} = 1$, the value becomes $1.67$ rather than $1.00$.
  - The mathematically rigorous linear min-max mapping from $[1, 3]$ to $[1, 5]$ is:
    $$Y_{[1, 5]} = 1 + (X_{[1, 3]} - 1) \times \frac{5 - 1}{3 - 1} = 2 X_{[1, 3]} - 1$$
    Under the true linear mapping, $\text{'No'} \to 1.0$, $\text{'Sometimes'} \to 3.0$, and $\text{'Yes'} \to 5.0$.
  - *Impact:* Because the current dataset cannot be modified, the existing `social_conflict_score` exhibits a theoretical minimum of $(1 + 1 + 1.67)/3 = 1.22$ rather than $1.00$. While this does not harm monotonic correlations ($r = 0.582$ with target), it represents an empirical scaling shift that must be transparently disclosed.

---

### 1.5 Target Leakage Verification
- **Verification:** An inspection of `feature_dataset.csv` confirmed that **zero target leakage exists in the feature columns**:
  - `Q6. Lecture distraction` $\to$ **Excluded**
  - `Q7. Study time reduction` $\to$ **Excluded**
  - `Q8. Task completion delays` $\to$ **Excluded**
  - `academic_impact_avg` $\to$ **Excluded**
- **Modeling Requirement:** The dataframe currently stores `impact_category` and `impact_encoded` in the final two columns for convenience. In Phase II modeling scripts, feature matrices must strictly drop both target columns:
  ```python
  X = df_feat.drop(columns=['impact_category', 'impact_encoded'])
  y = df_feat['impact_encoded']
  ```

---

### 1.6 Language Review: Causal vs. Observational Claims
- **Critique:** Cross-sectional survey data captures associations at a single cross-section in time; it cannot establish temporal precedence or causal direction.
- In earlier markdown text, terms like *"social media causes disruption"* or *"screen time drives high impact"* were occasionally used.
- *Correction Recommended:* In all Phase II deliverables, replace causal verbs with rigorous observational terminology:
  - Replace *"drives / causes"* with *"is strongly associated with / co-occurs with"*.
  - Replace *"protective shield"* with *"correlates with lower observed impairment"*.
  - Explicitly acknowledge that bidirectionality is plausible (e.g. students who struggle academically may turn to doom-scrolling as an avoidance coping mechanism).

---

## 2. Modeling Feature Set Configurations

To evaluate the impact of feature aggregation versus atomic survey items and prevent collinearity distortion, we define two formal feature configurations:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ALL AVAILABLE FEATURES                          │
│                                                                        │
│  [Demographics]           [Usage & Purpose]       [Social & Dependency]│
│  - gender_encoded         - daily_device_hours    - phone_with_family  │
│  - academic_year_encoded  - check_frequency       - conversation_interr│
│                           - activity_* (6 dummies)- social_complaints  │
│                                                   - stay_away_diff     │
│                                                                        │
│                      [Composite Indices]                               │
│                      - digital_dependency_score                        │
│                      - social_conflict_score                           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌─────────────────────────────────┐   ┌──────────────────────────────────┐
│   CONFIGURATION A: ATOMIC       │   │   CONFIGURATION B: COMPOSITE     │
│   (14 Input Features)           │   │   (10 Input Features)            │
│                                 │   │                                  │
│  - 2 Demographics               │   │  - 2 Demographics                │
│  - 2 Usage Habits (Q3, Q5)      │   │  - 6 Activity Dummies (Q4)       │
│  - 6 Activity Dummies (Q4)      │   │  - 2 Composite Scores            │
│  - 4 Social/Dep Items (Q9-Q12)  │   │                                  │
│                                 │   │  EXCLUDED (Prevent Collinearity):│
│  EXCLUDED (Prevent Redundancy): │   │  - Q3, Q5, Q12 (in Dependency)   │
│  - digital_dependency_score     │   │  - Q9, Q10, Q11 (in Conflict)    │
│  - social_conflict_score        │   │                                  │
└─────────────────────────────────┘   └──────────────────────────────────┘
```

### Configuration A: Atomic Features (14 Input Features)
Focuses on raw, unaggregated behavioral, demographic, and contextual survey responses. Composite scores are strictly excluded because their constituent items are already present.

| # | Feature Name | Representation / Scale | Conceptual Domain |
|:---:|:---|:---|:---|
| 1 | `gender_encoded` | Binary ($0 = \text{Female}, 1 = \text{Male}$) | Demographic baseline |
| 2 | `academic_year_encoded` | Ordinal ($1 = \text{1st}, 2 = \text{2nd}, 3 = \text{3rd}, 4 = \text{4th}, 3 = \text{Other}$) | Academic level |
| 3 | `daily_device_hours` | Ordinal ($1 = <2\text{h}$ to $5 = >8\text{h}$) | Usage volume (Q3) |
| 4 | `check_frequency` | Ordinal ($1 = <5/\text{day}$ to $5 = \text{Continuously}$) | Unlocking compulsion (Q5) |
| 5 | `phone_with_family` | Likert ($1 = \text{Str Disagree}$ to $5 = \text{Str Agree}$) | Social phubbing (Q9) |
| 6 | `conversation_interruption` | Likert ($1 = \text{Str Disagree}$ to $5 = \text{Str Agree}$) | Face-to-face interruption (Q10) |
| 7 | `social_complaints` | Ordinal ($1 = \text{No}, 2 = \text{Sometimes}, 3 = \text{Yes}$) | External social friction (Q11) |
| 8 | `stay_away_difficulty` | Likert ($1 = \text{Very easy}$ to $5 = \text{Very difficult}$) | Disconnection tolerance (Q12) |
| 9 | `activity_Social media` | Binary dummy ($0$ or $1$) | Primary application context (Q4) |
| 10 | `activity_Educational activities` | Binary dummy ($0$ or $1$) | Primary application context (Q4) |
| 11 | `activity_Watching videos and entertainment` | Binary dummy ($0$ or $1$) | Primary application context (Q4) |
| 12 | `activity_Communication and messaging` | Binary dummy ($0$ or $1$) | Primary application context (Q4) |
| 13 | `activity_Online gaming` | Binary dummy ($0$ or $1$) | Primary application context (Q4) |
| 14 | `activity_Other` | Binary dummy ($0$ or $1$) | Primary application context (Q4) |

- **Excluded from Config A:** `digital_dependency_score`, `social_conflict_score`, `impact_category`, `impact_encoded`, `record_id`, `data_source`, timestamps, consent, Q6, Q7, Q8, `academic_impact_avg`.

---

### Configuration B: Composite Features (10 Input Features)
Condenses multi-item behavioral traits into continuous composite indices to reduce dimensionality and mitigate noise. The 6 component survey items are strictly removed to prevent collinearity and dimension double-counting.

| # | Feature Name | Representation / Scale | Conceptual Domain |
|:---:|:---|:---|:---|
| 1 | `gender_encoded` | Binary ($0 = \text{Female}, 1 = \text{Male}$) | Demographic baseline |
| 2 | `academic_year_encoded` | Ordinal ($1 \dots 4$) | Academic level |
| 3 | `social_conflict_score` | Continuous composite ($1.22 \dots 5.00$) | Social disruption summary (Q9, Q10, Q11) |
| 4 | `digital_dependency_score` | Continuous composite ($1.33 \dots 5.00$) | Behavioral addiction summary (Q3, Q5, Q12) |
| 5 | `activity_Social media` | Binary dummy ($0$ or $1$) | Primary application context (Q4) |
| 6 | `activity_Educational activities` | Binary dummy ($0$ or $1$) | Primary application context (Q4) |
| 7 | `activity_Watching videos and entertainment` | Binary dummy ($0$ or $1$) | Primary application context (Q4) |
| 8 | `activity_Communication and messaging` | Binary dummy ($0$ or $1$) | Primary application context (Q4) |
| 9 | `activity_Online gaming` | Binary dummy ($0$ or $1$) | Primary application context (Q4) |
| 10 | `activity_Other` | Binary dummy ($0$ or $1$) | Primary application context (Q4) |

- **Component Features Removed from Config B to Prevent Double-Counting:**
  - `phone_with_family`, `conversation_interruption`, `social_complaints` (subsumed in `social_conflict_score`).
  - `daily_device_hours`, `check_frequency`, `stay_away_difficulty` (subsumed in `digital_dependency_score`).
- **Excluded Metadata & Targets:** `impact_category`, `impact_encoded`, `record_id`, `data_source`, timestamps, consent, Q6, Q7, Q8, `academic_impact_avg`.

---

## 3. Machine Learning Evaluation Strategy

### 3.1 Supervised Learning: K-Nearest Neighbors (KNN) Classification

```
┌────────────────────────────────────────────────────────────────────────┐
│                     DATASET SPLITTING ARCHITECTURE                     │
│                                                                        │
│  [Genuine Student Sample: N=54]             [Synthetic Sample: N=96]   │
│            │                                           │               │
│            ├─ Stratified Train Split (70%: N=38)       │               │
│            │         │                                 │               │
│            │         ▼                                 ▼               │
│            │   ┌──────────────────────────────────────────────┐        │
│            │   │              TRAINING SET                    │        │
│            │   │  N = 134 (96 Synthetic + 38 Genuine Train)   │        │
│            │   │  * Fit StandardScaler strictly here          │        │
│            │   │  * Perform 5-Fold Stratified Cross-Validation │       │
│            │   └──────────────────────┬───────────────────────┘        │
│            │                          ▼                                │
│            │                  [Trained KNN Model]                      │
│            │                          │                                │
│            ▼                          ▼                                │
│  ┌──────────────────────────────────────────────┐                      │
│  │         GENUINE HELD-OUT TEST BENCHMARK      │                      │
│  │         N = 16 (100% Genuine Students)       │                      │
│  │  * Transform using train-fitted scaler only  │                      │
│  │  * Independent ground-truth evaluation       │                      │
│  └──────────────────────────────────────────────┘                      │
└────────────────────────────────────────────────────────────────────────┘
```

#### A. Scientifically Honest Train / Test Partitioning
- **Core Principle:** A university ML model cannot claim genuine real-world validity if evaluated on synthetic data. Therefore, the **final benchmark test set must consist entirely of genuine student responses**.
- **Partitioning Design:**
  - Total genuine student records: $N_{\text{real}} = 54$.
  - A stratified split ($70\% / 30\%$) is performed on the genuine dataset:
    - **Genuine Test Benchmark ($30\%$):** Exactly **16 genuine students**, stratified by target class:
      - Low Impact: $2$ students ($12.5\%$)
      - Moderate Impact: $10$ students ($62.5\%$)
      - High Impact: $4$ students ($25.0\%$)
    - **Genuine Train Pool ($70\%$):** Exactly **38 genuine students** ($5$ Low, $22$ Moderate, $11$ High).
  - **Combined Training Set:** The 38 genuine training students are augmented with all **96 synthetic observations**, yielding a rich training pool of **$N_{\text{train}} = 134$ observations**.
  - **Alternative Benchmark:** A secondary evaluation will evaluate a model trained solely on the 96 synthetic records and tested on all 54 genuine students to measure pure cross-domain generalization.

#### B. Preprocessing Leakage Prevention
- **Strict Pipeline Scoping:** Feature scaling (`StandardScaler` or `MinMaxScaler`) must be `.fit()` **strictly and exclusively on the training set**.
- The fitted scaler parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) are subsequently used to `.transform()` the held-out genuine test set. Global scaling across all 150 instances is strictly forbidden.

#### C. Hyperparameter Optimization & Grid Search
Using 5-fold Stratified Cross-Validation on the training pool, the following hyperparameter grid will be evaluated:
- **Number of Neighbors ($K$):** $K \in [3, 5, 7, 9, 11, 13, 15]$.
- **Weighting Schemes:**
  - `weights='uniform'` (standard majority voting).
  - `weights='distance'` (inverse-distance weighting $w_i = 1 / d(x, x_i)$, critical for counteracting class imbalance against the majority Moderate class).
- **Distance Metrics:**
  - Euclidean distance ($p = 2$).
  - Manhattan distance ($p = 1$, more robust to high-dimensional dummy sparsity).

#### D. Comprehensive Performance Metrics
Given the multi-class ordinal structure and class imbalance, accuracy alone is insufficient:
1. **Overall Accuracy:** Percentage of correct classifications.
2. **Macro-Averaged F1-Score:** Unweighted mean of class F1-scores, ensuring the minority 'Low Impact' class is evaluated with equal importance.
3. **Weighted F1-Score:** F1-score weighted by class prevalence.
4. **Per-Class Metrics:** Precision, Recall, and F1-Score reported individually for Low, Moderate, and High Impact.
5. **Confusion Matrix:** Both absolute count and normalized recall confusion matrices to reveal boundary confusion (e.g. whether High Impact is mistakenly classified as Low Impact).

---

### 3.2 Unsupervised Learning: Clustering Strategy

#### A. Total Target Isolation
- **Rule:** All target columns (`impact_category`, `impact_encoded`) and academic impact evaluation items (Q6, Q7, Q8, `academic_impact_avg`) are **completely removed** before clustering.
- Clustering algorithms must identify natural student groupings based solely on behavioral usage, application context, and interpersonal friction.

#### B. Mandatory Distance Scaling
- Because clustering algorithms (K-Means and Agglomerative) compute spatial Euclidean distances, all input features must be standardized to zero mean and unit variance ($\mu=0, \sigma=1$) using `StandardScaler`. Without scaling, ordinal features with wider ranges (e.g. daily hours $1\dots 5$) would artificially dominate binary activity dummies ($0$ or $1$).

#### C. Evaluated Clustering Algorithms
1. **K-Means Clustering:**
   - Evaluated across cluster numbers $K \in [2, 6]$.
   - Initialization: $K$-Means++ with $50$ random restarts (`n_init=50`) to ensure global convergence.
   - Diagnostic Validation:
     - **Inertia Elbow Method:** Visual inspection of within-cluster sum of squared errors ($WSS$).
     - **Silhouette Analysis:** Mean silhouette coefficient and silhouette cluster thickness plots to identify the optimal $K$.
2. **Agglomerative Hierarchical Clustering:**
   - Evaluated across linkage criteria:
     - **Ward's Linkage:** Minimizes variance growth within merging clusters (Euclidean distance).
     - **Complete Linkage:** Maximizes pairwise distance between cluster members.
     - **Average Linkage:** Average pairwise distance.
   - Visualization: **Dendrogram** plotting tree merges to determine the natural cut height.

#### D. Post-Hoc Diagnostic Interpretation (External Validation Only)
- Clustering algorithms will operate 100% blind to academic labels.
- Only **after** clusters are formed, analysts will cross-tabulate cluster IDs against `impact_category` to assess:
  - Do behavioral usage clusters map naturally to academic impact severity?
  - Compute **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)** purely as post-hoc external alignment diagnostics.
  - Profile the behavioral centroids of each discovered cluster (e.g. "Cluster 0: Heavy Social Media Checkers", "Cluster 1: Academic Productivity Users", "Cluster 2: Balanced Moderate Users").

---

## 4. Academic Limitations & Disclosure Protocol

To maintain complete academic honesty in the university assignment report, the following constraints and limitations must be explicitly disclosed:

1. **Small Genuine Sample Baseline ($N=54$):**
   - The original empirical foundation represents a single-institution convenience sample collected via Google Forms.
   - The minority class **Low Impact** contains only **7 genuine student records** ($12.96\%$). Any test split allocating 2 or 3 examples to the test set has high sensitivity to individual misclassifications (a single error changes recall by $33\text{–}50\%$).
2. **Synthetic Data Role & Disclaimers:**
   - Synthetic data ($N=96$) was generated using a Multivariate Gaussian Copula to expand training capacity, stabilize covariance estimation, and facilitate algorithmic exploration.
   - The report must **never claim** that the machine learning models have been validated on 150 independent genuine human respondents.
3. **Cross-Sectional Self-Report Bias:**
   - Survey responses rely on subjective self-reporting (e.g. students estimating their own daily device hours or reporting whether family complained).
   - Actual screen time recorded by operating system trackers (e.g. Apple Screen Time or Android Digital Wellbeing) may deviate from self-perception.
4. **Non-Causality:**
   - All documented relationships are correlational associations. Causality cannot be inferred without longitudinal tracking or randomized intervention trials.

---

## 5. Summary Checklist for Phase II Implementation Approval

| Item | Status | Verification Detail |
|:---|:---:|:---|
| **Original Survey & Synthetic Data Untouched** | Verified | File modification timestamps and SHA256 hashes remain unchanged. |
| **EDA Numerical Audit Completed** | Verified | Exact combined dataset ($N=150$) tallies documented to replace draft text. |
| **Feature Configuration A (Atomic) Defined** | Ready | 14 input features; composite scores excluded to prevent redundancy. |
| **Feature Configuration B (Composite) Defined** | Ready | 10 input features; component questions removed to avoid double-counting. |
| **KNN Leakage & Scaler Protocol Established** | Ready | `StandardScaler` fitted strictly on train split; test set 100% genuine. |
| **Clustering Architecture Formulated** | Ready | Target strictly isolated; K-Means and Agglomerative evaluated with Silhouette. |
| **Academic Limitations Disclosed** | Documented | Small genuine class size ($N=7$ Low Impact) and synthetic role clearly framed. |

*This preparation plan is submitted for review and approval prior to executing model training and cluster experimentation in Phase II.*
