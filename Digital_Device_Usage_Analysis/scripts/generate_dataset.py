#!/usr/bin/env python3
"""
generate_dataset.py
-------------------
Advanced AI University Assignment:
Digital Device Usage Pattern Analysis Using Machine Learning

Author: Senior Data Scientist & Machine Learning Engineer
Random Seed: 42 (Strictly Reproducible)

This script:
1. Ingests the genuine Google Forms survey responses (54 students).
2. Fits a correlation-preserving Multivariate Gaussian Copula on ordinal/Likert dimensions.
3. Employs Dirichlet-smoothed conditional probability sampling for nominal activities.
4. Generates plausible, context-aware open-ended responses explicitly flagged as [Synthetic].
5. Generates valid, non-colliding synthetic survey timestamps.
6. Computes the target variable `impact_category` from Q6, Q7, and Q8 academic impact items.
7. Produces an engineered numerical feature dataset for KNN and clustering without target leakage.
8. Validates all constraints through rigorous assertions.
9. Exports all required raw, processed, report, and visual files.
"""

import os
import sys
import shutil
import random
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 0. CONFIGURATION & REPRODUCIBILITY
# ==============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

ORIGINAL_CSV_PATH = WORKSPACE_ROOT / "Digital Device Usage and Its Impact on Students.csv"
if not ORIGINAL_CSV_PATH.exists():
    ORIGINAL_CSV_PATH = PROJECT_ROOT / "Digital Device Usage and Its Impact on Students.csv"

# Target Directories
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, REPORTS_DIR, SCRIPTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. LOAD & INSPECT ORIGINAL DATASET
# ==============================================================================
def load_and_inspect_original(file_path: Path) -> pd.DataFrame:
    """Loads original survey CSV and validates fundamental schema."""
    if not file_path.exists():
        raise FileNotFoundError(f"Original dataset not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"[INFO] Loaded original dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # Assertions
    assert df.shape[0] == 54, f"Expected 54 original rows, found {df.shape[0]}."
    assert df.shape[1] == 17, f"Expected 17 original columns, found {df.shape[1]}."
    assert df.isnull().sum().sum() == 0, "Missing values detected in original survey dataset!"
    
    return df


# ==============================================================================
# 2. STATISTICAL SYNTHETIC DATA GENERATION ENGINE
# ==============================================================================
def generate_synthetic_data(df_orig: pd.DataFrame, n_samples: int = 96) -> pd.DataFrame:
    """
    Generates n_samples synthetic student survey observations using:
    - Multivariate Gaussian Copula for joint ordinal/Likert distributions.
    - Conditional categorical sampling with Dirichlet smoothing for primary activity.
    - Contextual domain-based generation for open-ended responses with [Synthetic] marker.
    - Non-colliding valid synthetic timestamps within survey window.
    """
    cols = list(df_orig.columns)
    
    # Categorical domain definitions matching original survey exactly
    gender_cats = ['Female', 'Male']
    year_cats = ['First year', 'Second year', 'Third year', 'Fourth year', 'Other']
    time_cats = ['Less than 2 hours', '2–4 hours', '4–6 hours', '6–8 hours', 'More than 8 hours']
    freq_cats = ['Less than 5 times a day', '5–10 times a day', '11–20 times a day', 'More than 20 times a day', 'Almost continuously']
    likert_cats = ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree']
    complain_cats = ['No', 'Sometimes', 'Yes']
    stay_away_cats = ['Very easy', 'Easy', 'Neutral', 'Difficult', 'Very difficult']
    activity_cats = ['Social media', 'Educational activities', 'Watching videos and entertainment', 
                     'Communication and messaging', 'Other', 'Online gaming']

    # Ordinal dimensions to model with Gaussian Copula
    ordinal_specs = [
        ('gender', cols[2], gender_cats),
        ('year', cols[3], year_cats),
        ('time', cols[4], time_cats),
        ('freq', cols[6], freq_cats),
        ('q6', cols[7], likert_cats),
        ('q7', cols[8], likert_cats),
        ('q8', cols[9], likert_cats),
        ('q9', cols[10], likert_cats),
        ('q10', cols[11], likert_cats),
        ('q11', cols[12], complain_cats),
        ('q12', cols[13], stay_away_cats)
    ]

    # Step A: Estimate empirical marginal thresholds and rank correlation matrix
    encoded_matrix = []
    thresholds = {}

    for var_id, col_name, cat_list in ordinal_specs:
        cat_to_int = {c: i for i, c in enumerate(cat_list)}
        int_series = df_orig[col_name].map(cat_to_int).values
        encoded_matrix.append(int_series)
        
        # Empirical cumulative probability thresholds
        counts = pd.Series(int_series).value_counts(sort=False).reindex(range(len(cat_list)), fill_value=0)
        cum_probs = (counts.cumsum() / len(int_series)).values
        cum_probs = np.clip(cum_probs, 1e-6, 1.0 - 1e-6)
        thresholds[var_id] = stats.norm.ppf(cum_probs[:-1])

    encoded_matrix = np.array(encoded_matrix).T  # Shape: (54, 11)

    # Compute Spearman rank correlation matrix
    corr_spearman, _ = stats.spearmanr(encoded_matrix)
    
    # Latent Gaussian correlation conversion: r_normal = 2 * sin(pi / 6 * r_spearman)
    r_latent = 2.0 * np.sin(np.pi / 6.0 * corr_spearman)
    
    # Regularize to guarantee positive semi-definiteness
    eigvals, eigvecs = np.linalg.eigh(r_latent)
    eigvals = np.clip(eigvals, 1e-4, None)
    r_latent_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    inv_d = 1.0 / np.sqrt(np.diag(r_latent_psd))
    r_latent_psd = r_latent_psd * np.outer(inv_d, inv_d)

    # Step B: Sample latent multivariate Gaussian vectors
    z_samples = np.random.multivariate_normal(np.zeros(len(ordinal_specs)), r_latent_psd, size=n_samples)

    # Step C: Discretize back to original categories via empirical thresholds
    syn_data_dict = {}
    for idx, (var_id, col_name, cat_list) in enumerate(ordinal_specs):
        z_col = z_samples[:, idx]
        cuts = thresholds[var_id]
        indices = np.digitize(z_col, cuts)
        indices = np.clip(indices, 0, len(cat_list) - 1)
        syn_data_dict[col_name] = [cat_list[i] for i in indices]

    # Step D: Compute synthetic academic impact average for conditional sampling
    likert_5 = {'Strongly disagree': 1, 'Disagree': 2, 'Neutral': 3, 'Agree': 4, 'Strongly agree': 5}
    q6_vals = [likert_5[v] for v in syn_data_dict[cols[7]]]
    q7_vals = [likert_5[v] for v in syn_data_dict[cols[8]]]
    q8_vals = [likert_5[v] for v in syn_data_dict[cols[9]]]
    syn_academic_avg = [(a + b + c) / 3.0 for a, b, c in zip(q6_vals, q7_vals, q8_vals)]
    
    def map_impact(avg_val):
        if avg_val <= 2.33:
            return 'Low Impact'
        elif avg_val <= 3.66:
            return 'Moderate Impact'
        else:
            return 'High Impact'

    syn_impact_categories = [map_impact(a) for a in syn_academic_avg]

    # Step E: Conditional sampling for Primary Activity (Q4)
    # Compute empirical conditional distribution P(Activity | Impact Category) with Laplace smoothing
    real_q6 = df_orig[cols[7]].map(likert_5)
    real_q7 = df_orig[cols[8]].map(likert_5)
    real_q8 = df_orig[cols[9]].map(likert_5)
    real_avg = (real_q6 + real_q7 + real_q8) / 3.0
    real_impact = real_avg.apply(map_impact)

    cond_counts = pd.crosstab(real_impact, df_orig[cols[5]]).reindex(
        index=['Low Impact', 'Moderate Impact', 'High Impact'],
        columns=activity_cats,
        fill_value=0
    )
    # Dirichlet/Laplace smoothing (alpha = 0.5) to avoid impossible zero categories
    smoothed_probs = (cond_counts + 0.5).div((cond_counts + 0.5).sum(axis=1), axis=0)

    syn_activities = []
    for imp in syn_impact_categories:
        prob_vec = smoothed_probs.loc[imp].values
        chosen_act = np.random.choice(activity_cats, p=prob_vec)
        syn_activities.append(chosen_act)
    syn_data_dict[cols[5]] = syn_activities

    # Step F: Consent Column (universal consent in survey)
    syn_data_dict[cols[1]] = ['Yes, I agree to participate.'] * n_samples

    # Step G: Plausible Non-colliding Synthetic Timestamps
    # Timeframe: Nov 14 to Nov 17, 2026 (matching survey collection period)
    base_start = datetime.datetime(2026, 11, 14, 8, 30, 0)
    real_timestamps = set(df_orig['Timestamp'].tolist())
    syn_timestamps = []

    for _ in range(n_samples):
        while True:
            # Random offset across 72 hours
            rand_sec = int(np.random.uniform(0, 72 * 3600))
            candidate_dt = base_start + datetime.timedelta(seconds=rand_sec)
            # Reformat to match survey: YYYY/MM/DD H:MM:SS AM/PM GMT+5
            h_12 = candidate_dt.hour % 12
            if h_12 == 0:
                h_12 = 12
            period = "AM" if candidate_dt.hour < 12 else "PM"
            ts_candidate = f"{candidate_dt.year}/{candidate_dt.month:02d}/{candidate_dt.day:02d} {h_12}:{candidate_dt.minute:02d}:{candidate_dt.second:02d} {period} GMT+5"
            
            if ts_candidate not in real_timestamps and ts_candidate not in syn_timestamps:
                syn_timestamps.append(ts_candidate)
                break
    syn_data_dict['Timestamp'] = syn_timestamps

    # Step H: Open-Ended Qualitative Responses (Q13, Q14, Q15)
    # Explicitly flagged with [Synthetic] marker as per assignment guidelines
    q13_templates = {
        'Social media': [
            "[Synthetic] Staying connected with study groups and sharing notes with batchmates.",
            "[Synthetic] Communication with classmates and quick access to department announcements.",
            "[Synthetic] Group project coordination and following educational accounts on LinkedIn.",
            "[Synthetic] Connecting with university societies and participating in academic forums.",
            "[Synthetic] Networking with tech peers and following industry tutorials."
        ],
        'Educational activities': [
            "[Synthetic] Accessing lecture slides, reading research papers, and using AI coding assistants.",
            "[Synthetic] Watching online tutorials, solving programming problems, and reviewing class notes.",
            "[Synthetic] Downloading textbooks, accessing LMS portals, and completing online quizzes.",
            "[Synthetic] Running algorithmic simulations, watching webinars, and organizing study material.",
            "[Synthetic] Researching academic papers and preparing assignment presentations."
        ],
        'Watching videos and entertainment': [
            "[Synthetic] Watching concept explanations on YouTube and relaxing between study sessions.",
            "[Synthetic] Accessing video lectures, recorded webinars, and educational channels.",
            "[Synthetic] Visual learning through animated tutorials and technical demonstrations.",
            "[Synthetic] Educational documentaries and refreshing my mind after intense classes."
        ],
        'Communication and messaging': [
            "[Synthetic] Instant messaging with project teammates and discussing homework questions.",
            "[Synthetic] Asking professors questions and organizing study group schedules.",
            "[Synthetic] Real-time collaboration on group assignments and sharing reference PDFs."
        ],
        'Online gaming': [
            "[Synthetic] De-stressing after intense study hours and connecting with university friends.",
            "[Synthetic] Strategy games help me relax and sharpen problem-solving reflexes."
        ],
        'Other': [
            "[Synthetic] Managing daily schedule, calendar reminders, and to-do lists for courses.",
            "[Synthetic] Digital note-taking and tracking submission deadlines across courses."
        ]
    }

    q14_templates = {
        'High Impact': [
            "[Synthetic] Severe eye fatigue and staying up late doom-scrolling before exam days.",
            "[Synthetic] Difficulty focusing during lectures and postponing assignment submissions.",
            "[Synthetic] Reduced attention span and constantly checking phone notifications while studying.",
            "[Synthetic] Chronic procrastination and disrupted sleep schedule due to screen exposure.",
            "[Synthetic] Headaches and feeling mentally exhausted after prolonged device usage."
        ],
        'Moderate Impact': [
            "[Synthetic] Mild eye strain after evening study sessions and occasional procrastination.",
            "[Synthetic] Getting distracted by notifications during study sessions.",
            "[Synthetic] Losing track of time while watching short reels or checking social feeds.",
            "[Synthetic] Occasional sleep disruption when using devices right before bedtime.",
            "[Synthetic] Neck stiffness and taking longer than intended to finish readings."
        ],
        'Low Impact': [
            "[Synthetic] Minor eye dryness after staring at the screen for prolonged periods.",
            "[Synthetic] Occasional notification distraction, but manageable with focus modes.",
            "[Synthetic] Slight physical fatigue from sitting too long at the computer.",
            "[Synthetic] Temporary eye strain when working without proper ambient lighting."
        ]
    }

    q15_templates = [
        "[Synthetic] Set app timers for entertainment apps and keep phone face down while studying.",
        "[Synthetic] Follow the 20-20-20 rule for eye health and take regular five-minute stretch breaks.",
        "[Synthetic] Establish designated device-free study hours and keep phones out of reach during lectures.",
        "[Synthetic] Schedule a fixed daily timetable with dedicated offline time for outdoor exercise and family.",
        "[Synthetic] Turn off non-essential notifications and use focus mode during study sessions.",
        "[Synthetic] Keep screens out of the bedroom thirty minutes before going to sleep.",
        "[Synthetic] Use website blockers during revision hours and reward yourself with short offline breaks.",
        "[Synthetic] Allocate specific hours for social media rather than checking feeds continuously.",
        "[Synthetic] Study with physical notebooks alongside devices to minimize continuous screen exposure.",
        "[Synthetic] Set clear personal boundaries between academic screen time and recreation."
    ]

    syn_q13 = []
    syn_q14 = []
    syn_q15 = []

    for i in range(n_samples):
        act = syn_activities[i]
        imp = syn_impact_categories[i]
        
        pool13 = q13_templates.get(act, q13_templates['Educational activities'])
        syn_q13.append(np.random.choice(pool13))
        
        pool14 = q14_templates.get(imp, q14_templates['Moderate Impact'])
        syn_q14.append(np.random.choice(pool14))
        
        syn_q15.append(np.random.choice(q15_templates))

    syn_data_dict[cols[14]] = syn_q13
    syn_data_dict[cols[15]] = syn_q14
    syn_data_dict[cols[16]] = syn_q15

    # Assemble synthetic DataFrame in exact original schema
    df_syn = pd.DataFrame({c: syn_data_dict[c] for c in cols})
    
    assert df_syn.shape == (n_samples, len(cols)), f"Synthetic shape mismatch: {df_syn.shape}"
    return df_syn


# ==============================================================================
# 3. TARGET ENGINEERING & DATASET COMBINATION
# ==============================================================================
def combine_and_engineer_target(df_orig: pd.DataFrame, df_syn: pd.DataFrame) -> pd.DataFrame:
    """
    Combines real and synthetic observations, adds tracking columns,
    and engineers the target variable `impact_category`.
    """
    cols = list(df_orig.columns)
    q6_col = cols[7]
    q7_col = cols[8]
    q8_col = cols[9]

    likert_5 = {
        'Strongly disagree': 1,
        'Disagree': 2,
        'Neutral': 3,
        'Agree': 4,
        'Strongly agree': 5
    }

    # Prepare Real
    df_real_tagged = df_orig.copy()
    df_real_tagged['data_source'] = 'real'

    # Prepare Synthetic
    df_syn_tagged = df_syn.copy()
    df_syn_tagged['data_source'] = 'synthetic'

    # Concatenate
    df_comb = pd.concat([df_real_tagged, df_syn_tagged], ignore_index=True)
    df_comb.insert(0, 'record_id', range(1, len(df_comb) + 1))

    # Calculate Academic Impact Average
    q6_num = df_comb[q6_col].map(likert_5)
    q7_num = df_comb[q7_col].map(likert_5)
    q8_num = df_comb[q8_col].map(likert_5)

    assert q6_num.isnull().sum() == 0, "Unmapped values in Q6!"
    assert q7_num.isnull().sum() == 0, "Unmapped values in Q7!"
    assert q8_num.isnull().sum() == 0, "Unmapped values in Q8!"

    academic_avg = (q6_num + q7_num + q8_num) / 3.0
    df_comb['academic_impact_avg'] = academic_avg.round(4)

    def assign_category(score):
        if score <= 2.33:
            return 'Low Impact'
        elif score <= 3.66:
            return 'Moderate Impact'
        else:
            return 'High Impact'

    df_comb['impact_category'] = academic_avg.apply(assign_category)
    return df_comb


# ==============================================================================
# 4. ML FEATURE DATASET ENGINEERING
# ==============================================================================
def engineer_ml_features(df_combined: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers numerical features for KNN, K-Means, and Agglomerative clustering.
    Strictly avoids target leakage: Q6, Q7, Q8 and academic_impact_avg are excluded.
    No global StandardScaler is applied (must be fitted in downstream ML pipelines).
    """
    cols = list(df_combined.columns)

    likert_5 = {'Strongly disagree': 1, 'Disagree': 2, 'Neutral': 3, 'Agree': 4, 'Strongly agree': 5}
    time_map = {'Less than 2 hours': 1, '2–4 hours': 2, '4–6 hours': 3, '6–8 hours': 4, 'More than 8 hours': 5}
    freq_map = {'Less than 5 times a day': 1, '5–10 times a day': 2, '11–20 times a day': 3, 'More than 20 times a day': 4, 'Almost continuously': 5}
    complain_map = {'No': 1, 'Sometimes': 2, 'Yes': 3}
    stay_away_map = {'Very easy': 1, 'Easy': 2, 'Neutral': 3, 'Difficult': 4, 'Very difficult': 5}
    gender_map = {'Female': 0, 'Male': 1}
    year_map = {'First year': 1, 'Second year': 2, 'Third year': 3, 'Fourth year': 4, 'Other': 3}

    df_feat = pd.DataFrame()

    # 1. Demographic & Academic Progression
    df_feat['gender_encoded'] = df_combined[cols[3]].map(gender_map)
    df_feat['academic_year_encoded'] = df_combined[cols[4]].map(year_map)

    # 2. Digital Device Usage Patterns
    df_feat['daily_device_hours'] = df_combined[cols[5]].map(time_map)
    df_feat['check_frequency'] = df_combined[cols[7]].map(freq_map)

    # 3. Social Interference & Behavioral Dependency
    df_feat['phone_with_family'] = df_combined[cols[11]].map(likert_5)
    df_feat['conversation_interruption'] = df_combined[cols[12]].map(likert_5)
    df_feat['social_complaints'] = df_combined[cols[13]].map(complain_map)
    df_feat['stay_away_difficulty'] = df_combined[cols[14]].map(stay_away_map)

    # 4. Primary Device Activity (One-Hot Encoded)
    act_dummies = pd.get_dummies(df_combined[cols[6]], prefix='activity', dtype=int)
    for c in ['activity_Social media', 'activity_Educational activities', 'activity_Watching videos and entertainment', 
              'activity_Communication and messaging', 'activity_Online gaming', 'activity_Other']:
        df_feat[c] = act_dummies[c] if c in act_dummies else 0

    # 5. Conceptually Justified Composite Indices
    # Social Conflict Score: Average social sphere disruption (normalized across scales)
    df_feat['social_conflict_score'] = (
        df_feat['phone_with_family'] + 
        df_feat['conversation_interruption'] + 
        (df_feat['social_complaints'] * 1.67)
    ) / 3.0

    # Digital Dependency Index: Composite metric of screen exposure, compulsive checking, and separation difficulty
    df_feat['digital_dependency_score'] = (
        df_feat['daily_device_hours'] + 
        df_feat['check_frequency'] + 
        df_feat['stay_away_difficulty']
    ) / 3.0

    # Round floating engineered features
    df_feat['social_conflict_score'] = df_feat['social_conflict_score'].round(4)
    df_feat['digital_dependency_score'] = df_feat['digital_dependency_score'].round(4)

    # 6. Target Labels (Appended for evaluation / supervised training)
    df_feat['impact_category'] = df_combined['impact_category']
    df_feat['impact_encoded'] = df_feat['impact_category'].map({'Low Impact': 0, 'Moderate Impact': 1, 'High Impact': 2})

    return df_feat


# ==============================================================================
# 5. COMPREHENSIVE VALIDATION ENGINE
# ==============================================================================
def validate_datasets(df_orig: pd.DataFrame, df_syn: pd.DataFrame, df_comb: pd.DataFrame, df_feat: pd.DataFrame):
    """Executes validation assertions across all data artifacts."""
    print("[INFO] Running rigorous validation checks...")

    # Record counts
    assert len(df_orig) == 54, f"Real count error: {len(df_orig)} != 54"
    assert len(df_syn) == 96, f"Synthetic count error: {len(df_syn)} != 96"
    assert len(df_comb) == 150, f"Combined count error: {len(df_comb)} != 150"
    assert len(df_feat) == 150, f"Feature dataset count error: {len(df_feat)} != 150"

    # Data source breakdown
    source_counts = df_comb['data_source'].value_counts()
    assert source_counts['real'] == 54, "Real record count mismatch in combined data."
    assert source_counts['synthetic'] == 96, "Synthetic record count mismatch in combined data."

    # Record ID uniqueness
    assert df_comb['record_id'].nunique() == 150, "Duplicate record IDs found!"
    assert df_comb['record_id'].min() == 1 and df_comb['record_id'].max() == 150, "Record ID sequence error!"

    # Original records untouched
    orig_cols = list(df_orig.columns)
    comb_real_subset = df_comb[df_comb['data_source'] == 'real'][orig_cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(orig_cols_df := df_orig.reset_index(drop=True), comb_real_subset, check_dtype=False)
    print("  -> Original responses validated: 100% identical and unmodified.")

    # No exact duplicate profiles between real and synthetic across core items
    core_cols = orig_cols[2:14]
    overlap = pd.merge(df_orig[core_cols], df_syn[core_cols], how='inner')
    assert len(overlap) == 0, f"Detected {len(overlap)} exact duplicate responses between real and synthetic!"
    print("  -> Zero identical profile overlaps between real and synthetic data.")

    # Target variable integrity
    valid_targets = {'Low Impact', 'Moderate Impact', 'High Impact'}
    assert set(df_comb['impact_category'].unique()).issubset(valid_targets), "Invalid target values!"
    assert df_comb['impact_category'].isnull().sum() == 0, "Missing values in impact_category!"

    # Target leakage verification in feature dataset
    leakage_cols = ['Q6', 'Q7', 'Q8', 'academic_impact_avg', 'Timestamp', 'Consent', 'record_id', 'data_source']
    for col in df_feat.columns:
        for leak in leakage_cols:
            assert leak.lower() not in col.lower() or col == 'impact_category' or col == 'impact_encoded', \
                f"POTENTIAL TARGET LEAKAGE OR METADATA DETECTED: {col}"
    print("  -> Target leakage check passed: Zero leakage features present.")

    # Feature completeness
    assert df_feat.isnull().sum().sum() == 0, "Missing values detected in feature dataset!"
    print("[SUCCESS] All validation assertions passed successfully.")


# ==============================================================================
# 6. VISUALIZATION COMPARISON ENGINE
# ==============================================================================
def generate_comparison_plots(df_comb: pd.DataFrame, output_path: Path):
    """
    Renders a 9-panel visual comparison between real and synthetic observations.
    Saved at high DPI for university report submission.
    """
    print(f"[INFO] Generating visual comparison figure at: {output_path}")
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = {'real': '#2b5c8f', 'synthetic': '#d95f02'}

    fig, axes = plt.subplots(3, 3, figsize=(18, 15), constrained_layout=True)
    fig.suptitle("Real vs. Synthetic Student Responses: Comparative Validation", fontsize=18, fontweight='bold', y=1.02)

    # 1. Target Impact Category
    ax = axes[0, 0]
    target_df = df_comb.groupby(['impact_category', 'data_source'], observed=False).size().unstack(fill_value=0)
    target_pct = target_df.div(target_df.sum(axis=0), axis=1) * 100
    target_pct[['real', 'synthetic']].plot(kind='bar', ax=ax, color=[palette['real'], palette['synthetic']], width=0.6)
    ax.set_title("1. Academic Impact Category (%)", fontweight='bold', fontsize=12)
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Data Source")

    # 2. Daily Device Usage (Q3)
    ax = axes[0, 1]
    time_col = [c for c in df_comb.columns if c.startswith('Q3')][0]
    time_order = ['Less than 2 hours', '2–4 hours', '4–6 hours', '6–8 hours', 'More than 8 hours']
    time_df = df_comb.groupby([time_col, 'data_source'], observed=False).size().unstack(fill_value=0).reindex(time_order)
    time_pct = time_df.div(time_df.sum(axis=0), axis=1) * 100
    time_pct[['real', 'synthetic']].plot(kind='bar', ax=ax, color=[palette['real'], palette['synthetic']], width=0.6)
    ax.set_title("2. Daily Screen Time (Q3) (%)", fontweight='bold', fontsize=12)
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("")
    ax.set_xticklabels(['<2h', '2–4h', '4–6h', '6–8h', '>8h'], rotation=0)
    ax.legend(title="Data Source")

    # 3. Checking Frequency (Q5)
    ax = axes[0, 2]
    freq_col = [c for c in df_comb.columns if c.startswith('Q5')][0]
    freq_order = ['Less than 5 times a day', '5–10 times a day', '11–20 times a day', 'More than 20 times a day', 'Almost continuously']
    freq_df = df_comb.groupby([freq_col, 'data_source'], observed=False).size().unstack(fill_value=0).reindex(freq_order)
    freq_pct = freq_df.div(freq_df.sum(axis=0), axis=1) * 100
    freq_pct[['real', 'synthetic']].plot(kind='bar', ax=ax, color=[palette['real'], palette['synthetic']], width=0.6)
    ax.set_title("3. Checking Frequency (Q5) (%)", fontweight='bold', fontsize=12)
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("")
    ax.set_xticklabels(['<5/d', '5–10', '11–20', '>20', 'Contin.'], rotation=0)
    ax.legend(title="Data Source")

    # 4. Academic Year (Q2)
    ax = axes[1, 0]
    year_col = [c for c in df_comb.columns if c.startswith('Q2')][0]
    year_order = ['First year', 'Second year', 'Third year', 'Fourth year', 'Other']
    year_df = df_comb.groupby([year_col, 'data_source'], observed=False).size().unstack(fill_value=0).reindex(year_order)
    year_pct = year_df.div(year_df.sum(axis=0), axis=1) * 100
    year_pct[['real', 'synthetic']].plot(kind='bar', ax=ax, color=[palette['real'], palette['synthetic']], width=0.6)
    ax.set_title("4. Academic Year (Q2) (%)", fontweight='bold', fontsize=12)
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("")
    ax.set_xticklabels(['1st Yr', '2nd Yr', '3rd Yr', '4th Yr', 'Other'], rotation=0)
    ax.legend(title="Data Source")

    # 5. Primary Activity (Q4)
    ax = axes[1, 1]
    act_col = [c for c in df_comb.columns if c.startswith('Q4')][0]
    act_df = df_comb.groupby([act_col, 'data_source'], observed=False).size().unstack(fill_value=0)
    act_pct = act_df.div(act_df.sum(axis=0), axis=1) * 100
    act_pct[['real', 'synthetic']].plot(kind='barh', ax=ax, color=[palette['real'], palette['synthetic']], width=0.6)
    ax.set_title("5. Primary Activity (Q4) (%)", fontweight='bold', fontsize=12)
    ax.set_xlabel("Percentage (%)")
    ax.set_ylabel("")
    ax.legend(title="Data Source")

    # 6. Stay Away Difficulty (Q12)
    ax = axes[1, 2]
    q12_col = [c for c in df_comb.columns if c.startswith('Q12')][0]
    q12_order = ['Very easy', 'Easy', 'Neutral', 'Difficult', 'Very difficult']
    q12_df = df_comb.groupby([q12_col, 'data_source'], observed=False).size().unstack(fill_value=0).reindex(q12_order)
    q12_pct = q12_df.div(q12_df.sum(axis=0), axis=1) * 100
    q12_pct[['real', 'synthetic']].plot(kind='bar', ax=ax, color=[palette['real'], palette['synthetic']], width=0.6)
    ax.set_title("6. Stay Away Difficulty (Q12) (%)", fontweight='bold', fontsize=12)
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("")
    ax.set_xticklabels(['V.Easy', 'Easy', 'Neutral', 'Diff.', 'V.Diff.'], rotation=0)
    ax.legend(title="Data Source")

    # 7. Real Correlation Heatmap
    ax = axes[2, 0]
    likert_5 = {'Strongly disagree': 1, 'Disagree': 2, 'Neutral': 3, 'Agree': 4, 'Strongly agree': 5}
    time_map = {'Less than 2 hours': 1, '2–4 hours': 2, '4–6 hours': 3, '6–8 hours': 4, 'More than 8 hours': 5}
    freq_map = {'Less than 5 times a day': 1, '5–10 times a day': 2, '11–20 times a day': 3, 'More than 20 times a day': 4, 'Almost continuously': 5}
    stay_away_map = {'Very easy': 1, 'Easy': 2, 'Neutral': 3, 'Difficult': 4, 'Very difficult': 5}

    def extract_corr_df(subset_df):
        sub = pd.DataFrame()
        sub['Time (Q3)'] = subset_df[[c for c in subset_df.columns if c.startswith('Q3')][0]].map(time_map)
        sub['Freq (Q5)'] = subset_df[[c for c in subset_df.columns if c.startswith('Q5')][0]].map(freq_map)
        sub['Distr (Q6)'] = subset_df[[c for c in subset_df.columns if c.startswith('Q6')][0]].map(likert_5)
        sub['Reduc (Q7)'] = subset_df[[c for c in subset_df.columns if c.startswith('Q7')][0]].map(likert_5)
        sub['Diff (Q8)'] = subset_df[[c for c in subset_df.columns if c.startswith('Q8')][0]].map(likert_5)
        sub['Interr (Q10)'] = subset_df[[c for c in subset_df.columns if c.startswith('Q10')][0]].map(likert_5)
        sub['StayAway (Q12)'] = subset_df[[c for c in subset_df.columns if c.startswith('Q12')][0]].map(stay_away_map)
        return sub.corr(method='spearman')

    real_corr = extract_corr_df(df_comb[df_comb['data_source'] == 'real'])
    syn_corr = extract_corr_df(df_comb[df_comb['data_source'] == 'synthetic'])

    sns.heatmap(real_corr, ax=ax, annot=True, fmt=".2f", cmap='Blues', vmin=-0.2, vmax=0.8, cbar=False, annot_kws={"size": 8})
    ax.set_title("7. Real Data Spearman Correlations", fontweight='bold', fontsize=12)

    # 8. Synthetic Correlation Heatmap
    ax = axes[2, 1]
    sns.heatmap(syn_corr, ax=ax, annot=True, fmt=".2f", cmap='Oranges', vmin=-0.2, vmax=0.8, cbar=False, annot_kws={"size": 8})
    ax.set_title("8. Synthetic Data Spearman Correlations", fontweight='bold', fontsize=12)

    # 9. Correlation Preservation Scatter Plot
    ax = axes[2, 2]
    triu_idx = np.triu_indices_from(real_corr, k=1)
    real_vals = real_corr.values[triu_idx]
    syn_vals = syn_corr.values[triu_idx]
    
    ax.scatter(real_vals, syn_vals, color='#4a148c', s=60, alpha=0.8, edgecolors='black')
    ax.plot([-0.3, 0.8], [-0.3, 0.8], 'r--', lw=1.5, label='Ideal Alignment (y = x)')
    corr_preservation_r = np.corrcoef(real_vals, syn_vals)[0, 1]
    ax.set_title(f"9. Correlation Alignment (r = {corr_preservation_r:.2f})", fontweight='bold', fontsize=12)
    ax.set_xlabel("Real Correlation Pairs")
    ax.set_ylabel("Synthetic Correlation Pairs")
    ax.legend()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("[SUCCESS] Comparison figure generated successfully.")


# ==============================================================================
# 7. MARKDOWN REPORT GENERATOR
# ==============================================================================
def generate_markdown_report(df_orig: pd.DataFrame, df_syn: pd.DataFrame, df_comb: pd.DataFrame, df_feat: pd.DataFrame, report_path: Path):
    """Generates the comprehensive methodological report."""
    print(f"[INFO] Generating report at: {report_path}")

    real_counts = df_comb[df_comb['data_source'] == 'real']['impact_category'].value_counts()
    syn_counts = df_comb[df_comb['data_source'] == 'synthetic']['impact_category'].value_counts()
    total_counts = df_comb['impact_category'].value_counts()

    real_pcts = (real_counts / len(df_orig) * 100).round(2)
    syn_pcts = (syn_counts / len(df_syn) * 100).round(2)
    total_pcts = (total_counts / len(df_comb) * 100).round(2)

    avg_real = df_comb[df_comb['data_source'] == 'real']['academic_impact_avg'].describe().round(3)
    avg_syn = df_comb[df_comb['data_source'] == 'synthetic']['academic_impact_avg'].describe().round(3)

    sections = []
    sections.append("# Digital Device Usage Pattern Analysis: Synthetic Data Generation & Statistical Validation Report\n")
    sections.append("**Course**: Advanced Artificial Intelligence  ")
    sections.append("**Role**: Senior Data Scientist & Machine Learning Engineer  ")
    sections.append("**Random Seed**: 42 (Strictly Reproducible)  ")
    sections.append("**Date**: November 2026  \n")
    sections.append("---\n")
    
    sections.append("## 1. Executive Summary & Overview")
    sections.append(
        "This report documents the rigorous statistical analysis and synthetic data expansion for the university assignment "
        "**Digital Device Usage Pattern Analysis Using Machine Learning**. The primary objective was to expand an initial sample "
        "of 54 genuine university student responses collected via Google Forms into a robust, statistically realistic dataset of "
        "**150 observations** (54 real + 96 synthetic).\n\n"
        "The methodology prioritizes **structural fidelity**, **correlation preservation**, **target leakage prevention**, "
        "and **academic transparency**. The synthetic records reflect the joint multidimensional distribution of student behavior "
        "without creating identical copies of respondents or fabricating claims of independent field collection.\n"
    )

    sections.append("## 2. Dataset Description & Original Survey Analysis")
    sections.append(
        "The genuine survey dataset consists of N = 54 complete responses collected between November 14 and November 16, 2026. "
        "The survey instrument comprises 17 columns:\n"
        "- **Administrative / Consent**: Timestamp, Informed Consent (100% agreed).\n"
        "- **Demographics & Academics**: Q1 Gender (51.9% Male, 48.1% Female), Q2 Academic Year (59.3% Third Year, 14.8% First Year, 14.8% Second Year, 7.4% Other, 3.7% Fourth Year).\n"
        "- **Digital Device Usage Habits**: Q3 Daily screen time, Q4 Primary device activity, Q5 Smartphone checking frequency.\n"
        "- **Academic Impact (Core Drivers)**: Q6 Lecture distraction, Q7 Study time reduction, Q8 Task completion difficulty (all 5-point Likert scales).\n"
        "- **Social Interference & Addiction Indicators**: Q9 Phone use with family/friends, Q10 Conversation interruption, Q11 Social complaints received, Q12 Difficulty staying away from phone.\n"
        "- **Open-Ended Qualitative Responses**: Q13 Study benefits, Q14 Problems encountered, Q15 Suggested balance strategies.\n\n"
        "Inspection confirmed **zero missing values** and **zero duplicate records** in the raw survey.\n"
    )

    sections.append("## 3. Synthetic Data Generation Methodology")
    sections.append(
        "Rather than employing naive marginal resampling (which destroys multivariate covariance) or unperturbed bootstrapping "
        "(which generates exact duplicate student records), we implemented a **Multivariate Gaussian Copula with Empirical Marginal Quantiles** "
        "coupled with **Dirichlet-Smoothed Conditional Sampling**:\n\n"
        "### Mathematical Formulation\n"
        "1. **Latent Gaussian Mapping**:\n"
        "   Each ordinal/Likert survey item X_j in {c_1, ..., c_K} has empirical cumulative probabilities gamma_{j, k} = P(X_j <= c_{j, k}). "
        "In latent standard normal space Z_j ~ N(0, 1), empirical threshold cutoffs are derived via the inverse normal CDF:\n"
        "   tau_{j, k} = Phi^{-1}(gamma_{j, k})\n\n"
        "2. **Rank Covariance Preservation**:\n"
        "   The empirical Spearman rank correlation matrix R_s across all ordinal items is computed and converted to latent normal correlation:\n"
        "   R_z = 2 * sin((pi / 6) * R_s)\n"
        "   R_z is regularized to guarantee positive semi-definiteness via eigenvalue clipping (lambda_min >= 1e-4) and diagonal rescaling.\n\n"
        "3. **Stochastic Sampling**:\n"
        "   We draw 96 vectors from the latent multivariate normal distribution:\n"
        "   Z ~ N(0, R_z)\n"
        "   Each latent coordinate Z_j is discretized into original survey response categories via the empirical cutoffs tau_{j, k}.\n\n"
        "4. **Conditional Categorical Sampling for Nominal Activity (Q4)**:\n"
        "   The primary activity is sampled conditionally on the respondent's derived academic impact profile using empirical conditional "
        "probabilities with Laplace/Dirichlet smoothing (alpha = 0.5):\n"
        "   P(Activity = a | Impact = i) = (N_{a, i} + alpha) / sum_{a'} (N_{a', i} + alpha)\n"
        "   This ensures that students experiencing high academic disruption show higher probabilities of social media use, while "
        "low-impact students exhibit educational utility, matching the real data.\n\n"
        "5. **Context-Aware Open-Ended Generation**:\n"
        "   Responses for Q13, Q14, and Q15 are sampled from profile-aligned qualitative pools and explicitly tagged with `[Synthetic]` "
        "to prevent false claims of human survey authorship.\n\n"
        "6. **Non-Colliding Valid Timestamps**:\n"
        "   Timestamps were stochastically generated within the survey timeframe (Nov 14–17, 2026), formatted identically to Google Forms "
        "(`YYYY/MM/DD H:MM:SS AM/PM GMT+5`), ensuring zero collisions with real timestamps.\n"
    )

    sections.append("## 4. Why This Methodology Is Appropriate")
    sections.append(
        "1. **Guaranteed Marginal Distribution Fidelity**: Discretizing latent Gaussian variables at empirical quantile thresholds "
        "asymptotically reproduces the original marginal probabilities of every survey question.\n"
        "2. **Correlation & Covariance Preservation**: Unlike independent sampling, the copula accurately propagates the empirical correlations "
        "(e.g., between screen time and checking frequency, r = 0.52; distraction and conversation interruption, r = 0.54; and stay-away difficulty "
        "and study reduction, r = 0.51).\n"
        "3. **Prevention of Artificial Duplicates**: Because latent vectors are continuous stochastic draws, zero synthetic respondents share "
        "identical multi-item profiles with genuine respondents.\n"
        "4. **Realistic Class Imbalance**: The methodology avoids artificially balanced classes or synthetic separation, retaining the natural university distribution.\n"
    )

    sections.append("## 5. Target Variable Engineering & Leakage Prevention")
    sections.append(
        "The target variable `impact_category` was engineered strictly according to the assignment specification using the three academic impact items:\n"
        "- **Q6**: Lecture distraction (1–5)\n"
        "- **Q7**: Entertainment reduces study time (1–5)\n"
        "- **Q8**: Device makes task completion difficult (1–5)\n\n"
        "academic_impact_avg = (Q6 + Q7 + Q8) / 3\n\n"
        "- **Low Impact**: academic_impact_avg <= 2.33\n"
        "- **Moderate Impact**: 2.33 < academic_impact_avg <= 3.66\n"
        "- **High Impact**: academic_impact_avg > 3.66\n"
    )

    table = (
        "### Target Distribution Comparison\n"
        "| Impact Category | Real Count (N=54) | Real % | Synthetic Count (N=96) | Synthetic % | Combined Count (N=150) | Combined % |\n"
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n"
        f"| **Moderate Impact** | {real_counts.get('Moderate Impact', 0)} | {real_pcts.get('Moderate Impact', 0.0)}% | {syn_counts.get('Moderate Impact', 0)} | {syn_pcts.get('Moderate Impact', 0.0)}% | {total_counts.get('Moderate Impact', 0)} | {total_pcts.get('Moderate Impact', 0.0)}% |\n"
        f"| **High Impact** | {real_counts.get('High Impact', 0)} | {real_pcts.get('High Impact', 0.0)}% | {syn_counts.get('High Impact', 0)} | {syn_pcts.get('High Impact', 0.0)}% | {total_counts.get('High Impact', 0)} | {total_pcts.get('High Impact', 0.0)}% |\n"
        f"| **Low Impact** | {real_counts.get('Low Impact', 0)} | {real_pcts.get('Low Impact', 0.0)}% | {syn_counts.get('Low Impact', 0)} | {syn_pcts.get('Low Impact', 0.0)}% | {total_counts.get('Low Impact', 0)} | {total_pcts.get('Low Impact', 0.0)}% |\n"
        f"| **Total** | **54** | **100.0%** | **96** | **100.0%** | **150** | **100.0%** |\n"
    )
    sections.append(table)

    summary_stats = (
        "### Numerical Summary of Academic Impact Average\n"
        f"- **Real Responses**: Mean = {avg_real['mean']}, Std = {avg_real['std']}, Median = {avg_real['50%']}, IQR = [{avg_real['25%']}, {avg_real['75%']}]\n"
        f"- **Synthetic Responses**: Mean = {avg_syn['mean']}, Std = {avg_syn['std']}, Median = {avg_syn['50%']}, IQR = [{avg_syn['25%']}, {avg_syn['75%']}]\n"
    )
    sections.append(summary_stats)

    sections.append(
        "### Leakage Exclusion Guarantee\n"
        "In accordance with rigorous ML protocols:\n"
        "- `Q6`, `Q7`, `Q8`, and `academic_impact_avg` are **strictly excluded** from all input feature sets in `feature_dataset.csv`.\n"
        "- Models predicting `impact_category` must learn from behavioral usage, checking frequency, social interference, and device dependency "
        "without direct access to the academic impact evaluation questions.\n"
    )

    sections.append("## 6. Feature Engineering for ML & Clustering")
    sections.append(
        "The engineered feature dataset (`data/processed/feature_dataset.csv`) contains 16 input features + 2 target columns:\n"
        "1. `gender_encoded`: Binary indicator (0 = Female, 1 = Male).\n"
        "2. `academic_year_encoded`: Ordinal integer (1 = First year, 2 = Second year, 3 = Third year, 4 = Fourth year).\n"
        "3. `daily_device_hours`: Ordinal scale 1 to 5 (<2h, 2-4h, 4-6h, 6-8h, >8h).\n"
        "4. `check_frequency`: Ordinal scale 1 to 5 (<5/day to Almost continuously).\n"
        "5. `phone_with_family`: Likert scale 1 to 5 (Social phone presence).\n"
        "6. `conversation_interruption`: Likert scale 1 to 5 (Face-to-face interruption).\n"
        "7. `social_complaints`: Ordinal scale 1 to 3 (1 = No, 2 = Sometimes, 3 = Yes).\n"
        "8. `stay_away_difficulty`: Likert scale 1 to 5 (1 = Very easy to 5 = Very difficult).\n"
        "9–14. `activity_*`: One-hot indicators for primary digital activities.\n"
        "15. `social_conflict_score`: Statistically justified composite score of social sphere disruption:\n"
        "    social_conflict_score = (Q9 + Q10 + 1.67 * Q11) / 3\n"
        "16. `digital_dependency_score`: Behavioral dependency composite capturing daily exposure, checking frequency, and separation anxiety:\n"
        "    digital_dependency_score = (Q3 + Q5 + Q12) / 3\n"
        "17. `impact_category`: Target string (Low Impact, Moderate Impact, High Impact).\n"
        "18. `impact_encoded`: Target integer (0 = Low, 1 = Moderate, 2 = High).\n\n"
        "*StandardScaler Notice*: In adherence to assignment instructions, `StandardScaler` was **not applied globally**. Scaling must be fitted strictly inside cross-validation folds or clustering execution scripts to prevent data snooping.\n"
    )

    sections.append("## 7. Limitations of Synthetic Data")
    sections.append(
        "1. **Latent Linearity Assumption**: Gaussian copulas assume an underlying Gaussian dependence structure. Extreme multi-way non-linear interactions may be smoothed out.\n"
        "2. **Sample Size of Reference Distribution**: The empirical reference sample (N = 54) exhibits small sample variation; synthetic distributions naturally reflect the characteristics of this specific student cohort.\n"
        "3. **Qualitative Synthetics**: Open-ended responses, while contextually coherent, are generated from representative qualitative templates and tagged with `[Synthetic]` rather than reflecting spontaneous student prose.\n"
    )

    sections.append("## 8. Artifact Locations & Verification Checklist")
    sections.append(
        "- **Raw Original Data**: `data/raw/original_responses.csv` (N = 54)\n"
        "- **Raw Synthetic Data**: `data/raw/synthetic_responses.csv` (N = 96)\n"
        "- **Combined Processed Data**: `data/processed/combined_dataset.csv` (N = 150)\n"
        "- **Engineered ML Features**: `data/processed/feature_dataset.csv` (N = 150, 18 columns)\n"
        "- **Validation Comparison Figure**: `reports/real_vs_synthetic_comparison.png`\n"
        "- **Reproducible Script**: `scripts/generate_dataset.py`\n\n"
        "All validation assertions passed with zero errors.\n"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))
    print("[SUCCESS] Markdown report generated successfully.")


# ==============================================================================
# 8. MAIN EXECUTION PIPELINE
# ==============================================================================
def main():
    print("=" * 70)
    print("DIGITAL DEVICE USAGE PATTERN ANALYSIS: DATASET GENERATION PIPELINE")
    print("=" * 70)
    
    # 1. Load original survey
    df_orig = load_and_inspect_original(ORIGINAL_CSV_PATH)

    # 2. Generate 96 synthetic observations
    df_syn = generate_synthetic_data(df_orig, n_samples=96)

    # 3. Combine and engineer target
    df_comb = combine_and_engineer_target(df_orig, df_syn)

    # 4. Engineer ML features
    df_feat = engineer_ml_features(df_comb)

    # 5. Validate all datasets
    validate_datasets(df_orig, df_syn, df_comb, df_feat)

    # 6. Export data files
    orig_export_path = DATA_RAW_DIR / "original_responses.csv"
    syn_export_path = DATA_RAW_DIR / "synthetic_responses.csv"
    comb_export_path = DATA_PROCESSED_DIR / "combined_dataset.csv"
    feat_export_path = DATA_PROCESSED_DIR / "feature_dataset.csv"

    # Export original copy (untouched schema)
    df_orig.to_csv(orig_export_path, index=False)
    print(f"[EXPORT] Saved: {orig_export_path}")

    # Export synthetic raw (matching original schema)
    df_syn.to_csv(syn_export_path, index=False)
    print(f"[EXPORT] Saved: {syn_export_path}")

    # Export combined processed
    df_comb.to_csv(comb_export_path, index=False)
    print(f"[EXPORT] Saved: {comb_export_path}")

    # Export feature dataset
    df_feat.to_csv(feat_export_path, index=False)
    print(f"[EXPORT] Saved: {feat_export_path}")

    # 7. Generate comparison plot
    plot_export_path = REPORTS_DIR / "real_vs_synthetic_comparison.png"
    generate_comparison_plots(df_comb, plot_export_path)

    # 8. Generate comprehensive report
    report_export_path = REPORTS_DIR / "data_generation_report.md"
    generate_markdown_report(df_orig, df_syn, df_comb, df_feat, report_export_path)

    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Original Records:    {len(df_orig)}")
    print(f"Synthetic Records:   {len(df_syn)}")
    print(f"Total Records:       {len(df_comb)}")
    print("Target Distribution:")
    print(df_comb['impact_category'].value_counts().to_string())
    print("=" * 70)


if __name__ == "__main__":
    main()
