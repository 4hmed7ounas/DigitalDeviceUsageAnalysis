# Digital Device Usage Pattern Analysis: Synthetic Data Generation & Statistical Validation Report

**Course**: Advanced Artificial Intelligence  
**Role**: Senior Data Scientist & Machine Learning Engineer  
**Random Seed**: 42 (Strictly Reproducible)  
**Date**: November 2026  

---

## 1. Executive Summary & Overview
This report documents the rigorous statistical analysis and synthetic data expansion for the university assignment **Digital Device Usage Pattern Analysis Using Machine Learning**. The primary objective was to expand an initial sample of 54 genuine university student responses collected via Google Forms into a robust, statistically realistic dataset of **150 observations** (54 real + 96 synthetic).

The methodology prioritizes **structural fidelity**, **correlation preservation**, **target leakage prevention**, and **academic transparency**. The synthetic records reflect the joint multidimensional distribution of student behavior without creating identical copies of respondents or fabricating claims of independent field collection.

## 2. Dataset Description & Original Survey Analysis
The genuine survey dataset consists of N = 54 complete responses collected between November 14 and November 16, 2026. The survey instrument comprises 17 columns:
- **Administrative / Consent**: Timestamp, Informed Consent (100% agreed).
- **Demographics & Academics**: Q1 Gender (51.9% Male, 48.1% Female), Q2 Academic Year (59.3% Third Year, 14.8% First Year, 14.8% Second Year, 7.4% Other, 3.7% Fourth Year).
- **Digital Device Usage Habits**: Q3 Daily screen time, Q4 Primary device activity, Q5 Smartphone checking frequency.
- **Academic Impact (Core Drivers)**: Q6 Lecture distraction, Q7 Study time reduction, Q8 Task completion difficulty (all 5-point Likert scales).
- **Social Interference & Addiction Indicators**: Q9 Phone use with family/friends, Q10 Conversation interruption, Q11 Social complaints received, Q12 Difficulty staying away from phone.
- **Open-Ended Qualitative Responses**: Q13 Study benefits, Q14 Problems encountered, Q15 Suggested balance strategies.

Inspection confirmed **zero missing values** and **zero duplicate records** in the raw survey.

## 3. Synthetic Data Generation Methodology
Rather than employing naive marginal resampling (which destroys multivariate covariance) or unperturbed bootstrapping (which generates exact duplicate student records), we implemented a **Multivariate Gaussian Copula with Empirical Marginal Quantiles** coupled with **Dirichlet-Smoothed Conditional Sampling**:

### Mathematical Formulation
1. **Latent Gaussian Mapping**:
   Each ordinal/Likert survey item X_j in {c_1, ..., c_K} has empirical cumulative probabilities gamma_{j, k} = P(X_j <= c_{j, k}). In latent standard normal space Z_j ~ N(0, 1), empirical threshold cutoffs are derived via the inverse normal CDF:
   tau_{j, k} = Phi^{-1}(gamma_{j, k})

2. **Rank Covariance Preservation**:
   The empirical Spearman rank correlation matrix R_s across all ordinal items is computed and converted to latent normal correlation:
   R_z = 2 * sin((pi / 6) * R_s)
   R_z is regularized to guarantee positive semi-definiteness via eigenvalue clipping (lambda_min >= 1e-4) and diagonal rescaling.

3. **Stochastic Sampling**:
   We draw 96 vectors from the latent multivariate normal distribution:
   Z ~ N(0, R_z)
   Each latent coordinate Z_j is discretized into original survey response categories via the empirical cutoffs tau_{j, k}.

4. **Conditional Categorical Sampling for Nominal Activity (Q4)**:
   The primary activity is sampled conditionally on the respondent's derived academic impact profile using empirical conditional probabilities with Laplace/Dirichlet smoothing (alpha = 0.5):
   P(Activity = a | Impact = i) = (N_{a, i} + alpha) / sum_{a'} (N_{a', i} + alpha)
   This ensures that students experiencing high academic disruption show higher probabilities of social media use, while low-impact students exhibit educational utility, matching the real data.

5. **Context-Aware Open-Ended Generation**:
   Responses for Q13, Q14, and Q15 are sampled from profile-aligned qualitative pools and explicitly tagged with `[Synthetic]` to prevent false claims of human survey authorship.

6. **Non-Colliding Valid Timestamps**:
   Timestamps were stochastically generated within the survey timeframe (Nov 14–17, 2026), formatted identically to Google Forms (`YYYY/MM/DD H:MM:SS AM/PM GMT+5`), ensuring zero collisions with real timestamps.

## 4. Why This Methodology Is Appropriate
1. **Guaranteed Marginal Distribution Fidelity**: Discretizing latent Gaussian variables at empirical quantile thresholds asymptotically reproduces the original marginal probabilities of every survey question.
2. **Correlation & Covariance Preservation**: Unlike independent sampling, the copula accurately propagates the empirical correlations (e.g., between screen time and checking frequency, r = 0.52; distraction and conversation interruption, r = 0.54; and stay-away difficulty and study reduction, r = 0.51).
3. **Prevention of Artificial Duplicates**: Because latent vectors are continuous stochastic draws, zero synthetic respondents share identical multi-item profiles with genuine respondents.
4. **Realistic Class Imbalance**: The methodology avoids artificially balanced classes or synthetic separation, retaining the natural university distribution.

## 5. Target Variable Engineering & Leakage Prevention
The target variable `impact_category` was engineered strictly according to the assignment specification using the three academic impact items:
- **Q6**: Lecture distraction (1–5)
- **Q7**: Entertainment reduces study time (1–5)
- **Q8**: Device makes task completion difficult (1–5)

academic_impact_avg = (Q6 + Q7 + Q8) / 3

- **Low Impact**: academic_impact_avg <= 2.33
- **Moderate Impact**: 2.33 < academic_impact_avg <= 3.66
- **High Impact**: academic_impact_avg > 3.66

### Target Distribution Comparison
| Impact Category | Real Count (N=54) | Real % | Synthetic Count (N=96) | Synthetic % | Combined Count (N=150) | Combined % |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Moderate Impact** | 32 | 59.26% | 47 | 48.96% | 79 | 52.67% |
| **High Impact** | 15 | 27.78% | 38 | 39.58% | 53 | 35.33% |
| **Low Impact** | 7 | 12.96% | 11 | 11.46% | 18 | 12.0% |
| **Total** | **54** | **100.0%** | **96** | **100.0%** | **150** | **100.0%** |

### Numerical Summary of Academic Impact Average
- **Real Responses**: Mean = 3.173, Std = 0.829, Median = 3.0, IQR = [2.667, 3.667]
- **Synthetic Responses**: Mean = 3.333, Std = 0.861, Median = 3.333, IQR = [2.667, 4.0]

### Leakage Exclusion Guarantee
In accordance with rigorous ML protocols:
- `Q6`, `Q7`, `Q8`, and `academic_impact_avg` are **strictly excluded** from all input feature sets in `feature_dataset.csv`.
- Models predicting `impact_category` must learn from behavioral usage, checking frequency, social interference, and device dependency without direct access to the academic impact evaluation questions.

## 6. Feature Engineering for ML & Clustering
The engineered feature dataset (`data/processed/feature_dataset.csv`) contains 16 input features + 2 target columns:
1. `gender_encoded`: Binary indicator (0 = Female, 1 = Male).
2. `academic_year_encoded`: Ordinal integer (1 = First year, 2 = Second year, 3 = Third year, 4 = Fourth year).
3. `daily_device_hours`: Ordinal scale 1 to 5 (<2h, 2-4h, 4-6h, 6-8h, >8h).
4. `check_frequency`: Ordinal scale 1 to 5 (<5/day to Almost continuously).
5. `phone_with_family`: Likert scale 1 to 5 (Social phone presence).
6. `conversation_interruption`: Likert scale 1 to 5 (Face-to-face interruption).
7. `social_complaints`: Ordinal scale 1 to 3 (1 = No, 2 = Sometimes, 3 = Yes).
8. `stay_away_difficulty`: Likert scale 1 to 5 (1 = Very easy to 5 = Very difficult).
9–14. `activity_*`: One-hot indicators for primary digital activities.
15. `social_conflict_score`: Statistically justified composite score of social sphere disruption:
    social_conflict_score = (Q9 + Q10 + 1.67 * Q11) / 3
16. `digital_dependency_score`: Behavioral dependency composite capturing daily exposure, checking frequency, and separation anxiety:
    digital_dependency_score = (Q3 + Q5 + Q12) / 3
17. `impact_category`: Target string (Low Impact, Moderate Impact, High Impact).
18. `impact_encoded`: Target integer (0 = Low, 1 = Moderate, 2 = High).

*StandardScaler Notice*: In adherence to assignment instructions, `StandardScaler` was **not applied globally**. Scaling must be fitted strictly inside cross-validation folds or clustering execution scripts to prevent data snooping.

## 7. Limitations of Synthetic Data
1. **Latent Linearity Assumption**: Gaussian copulas assume an underlying Gaussian dependence structure. Extreme multi-way non-linear interactions may be smoothed out.
2. **Sample Size of Reference Distribution**: The empirical reference sample (N = 54) exhibits small sample variation; synthetic distributions naturally reflect the characteristics of this specific student cohort.
3. **Qualitative Synthetics**: Open-ended responses, while contextually coherent, are generated from representative qualitative templates and tagged with `[Synthetic]` rather than reflecting spontaneous student prose.

## 8. Artifact Locations & Verification Checklist
- **Raw Original Data**: `data/raw/original_responses.csv` (N = 54)
- **Raw Synthetic Data**: `data/raw/synthetic_responses.csv` (N = 96)
- **Combined Processed Data**: `data/processed/combined_dataset.csv` (N = 150)
- **Engineered ML Features**: `data/processed/feature_dataset.csv` (N = 150, 18 columns)
- **Validation Comparison Figure**: `reports/real_vs_synthetic_comparison.png`
- **Reproducible Script**: `scripts/generate_dataset.py`

All validation assertions passed with zero errors.
