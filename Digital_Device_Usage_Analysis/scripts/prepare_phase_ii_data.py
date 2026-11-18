#!/usr/bin/env python3
"""
prepare_phase_ii_data.py
-------------------------
Phase II Data Preparation & Leakage-Free Stratified Splitting

Author: Senior Data Scientist & Machine Learning Engineer
Project: Digital Device Usage Pattern Analysis Using Machine Learning
Strict Random Seed: 42

This script implements the required anti-leakage protocol:
1. Ingests genuine student responses (N=54).
2. Stratifies genuine responses into:
   - 38 Training responses (genuine_train.csv)
   - 16 Untouched Holdout Test responses (genuine_test.csv)
3. Fits the statistical synthetic data generation model (Multivariate Gaussian Copula
   + Dirichlet-smoothed conditional activity sampling) STRICTLY on the 38 genuine
   training responses.
4. Generates 96 training-only synthetic observations (synthetic_train_augmented.csv).
5. Assembles the augmented training dataset: 38 real + 96 synthetic = 134 records (train_augmented.csv).
6. Preserves the original 96 synthetic records from Phase I without overwriting.
7. Performs corrected feature engineering:
   - One-hot academic year encoding (5 categories: First, Second, Third, Fourth, Other)
   - Corrected social conflict score: Q11_scaled = 2 * Q11 - 1, score = (Q9 + Q10 + Q11_scaled)/3
   - Digital dependency score: (daily_device_hours + check_frequency + stay_away_difficulty)/3
   - Configuration A (Atomic, 18 features)
   - Configuration B (Composite, 14 features)
8. Exports clean, validated datasets for modeling and clustering.
"""

import os
import sys
import random
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_SPLITS = PROJECT_ROOT / "data" / "splits"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

DATA_SPLITS.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# Survey Column Names
ORIGINAL_CSV = PROJECT_ROOT.parent / "Digital Device Usage and Its Impact on Students.csv"
if not ORIGINAL_CSV.exists():
    ORIGINAL_CSV = PROJECT_ROOT / "Digital Device Usage and Its Impact on Students.csv"
if not ORIGINAL_CSV.exists():
    ORIGINAL_CSV = DATA_RAW / "original_responses.csv"


def load_genuine_data():
    df = pd.read_csv(ORIGINAL_CSV)
    assert df.shape[0] == 54, f"Expected 54 genuine responses, got {df.shape[0]}"
    assert df.isnull().sum().sum() == 0, "Missing values in original data"
    
    # Compute academic impact average and impact category
    likert_5 = {'Strongly disagree': 1, 'Disagree': 2, 'Neutral': 3, 'Agree': 4, 'Strongly agree': 5}
    q6 = df.iloc[:, 7].map(likert_5)
    q7 = df.iloc[:, 8].map(likert_5)
    q8 = df.iloc[:, 9].map(likert_5)
    avg = (q6 + q7 + q8) / 3.0
    
    def assign_category(score):
        if score <= 2.33:
            return 'Low Impact'
        elif score <= 3.66:
            return 'Moderate Impact'
        else:
            return 'High Impact'
            
    df['academic_impact_avg'] = avg.round(4)
    df['impact_category'] = avg.apply(assign_category)
    return df


def split_genuine_data(df):
    """Stratified 38/16 split preserving impact_category distribution."""
    df_train, df_test = train_test_split(
        df,
        test_size=16,
        random_state=SEED,
        stratify=df['impact_category']
    )
    df_train = df_train.copy().reset_index(drop=True)
    df_test = df_test.copy().reset_index(drop=True)
    
    print(f"[INFO] Genuine train shape: {df_train.shape}")
    print(f"[INFO] Genuine train distribution:\n{df_train['impact_category'].value_counts()}")
    print(f"[INFO] Genuine test shape: {df_test.shape}")
    print(f"[INFO] Genuine test distribution:\n{df_test['impact_category'].value_counts()}")
    
    assert len(df_train) == 38
    assert len(df_test) == 16
    return df_train, df_test


def generate_training_only_synthetics(df_train_genuine, n_samples=96):
    """
    Fits Gaussian Copula and conditional activity generator STRICTLY
    on the 38 genuine training records.
    """
    cols = list(df_train_genuine.columns[:17])  # original 17 survey columns
    
    gender_cats = ['Female', 'Male']
    year_cats = ['First year', 'Second year', 'Third year', 'Fourth year', 'Other']
    time_cats = ['Less than 2 hours', '2–4 hours', '4–6 hours', '6–8 hours', 'More than 8 hours']
    freq_cats = ['Less than 5 times a day', '5–10 times a day', '11–20 times a day', 'More than 20 times a day', 'Almost continuously']
    likert_cats = ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree']
    complain_cats = ['No', 'Sometimes', 'Yes']
    stay_away_cats = ['Very easy', 'Easy', 'Neutral', 'Difficult', 'Very difficult']
    activity_cats = ['Social media', 'Educational activities', 'Watching videos and entertainment', 
                     'Communication and messaging', 'Other', 'Online gaming']

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

    # Empirical cumulative probability thresholds from 38 train records
    encoded_matrix = []
    thresholds = {}

    for var_id, col_name, cat_list in ordinal_specs:
        cat_to_int = {c: i for i, c in enumerate(cat_list)}
        int_series = df_train_genuine[col_name].map(cat_to_int).values
        encoded_matrix.append(int_series)
        
        counts = pd.Series(int_series).value_counts(sort=False).reindex(range(len(cat_list)), fill_value=0)
        cum_probs = (counts.cumsum() / len(int_series)).values
        cum_probs = np.clip(cum_probs, 1e-6, 1.0 - 1e-6)
        thresholds[var_id] = stats.norm.ppf(cum_probs[:-1])

    encoded_matrix = np.array(encoded_matrix).T  # Shape: (38, 11)

    corr_spearman, _ = stats.spearmanr(encoded_matrix)
    r_latent = 2.0 * np.sin(np.pi / 6.0 * corr_spearman)
    
    # Regularize to guarantee positive semi-definiteness
    eigvals, eigvecs = np.linalg.eigh(r_latent)
    eigvals = np.clip(eigvals, 1e-4, None)
    r_latent_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    inv_d = 1.0 / np.sqrt(np.diag(r_latent_psd))
    r_latent_psd = r_latent_psd * np.outer(inv_d, inv_d)

    # Sample latent normal vectors
    z_samples = np.random.multivariate_normal(np.zeros(len(ordinal_specs)), r_latent_psd, size=n_samples)

    # Discretize back to categories
    syn_data_dict = {}
    for idx, (var_id, col_name, cat_list) in enumerate(ordinal_specs):
        z_col = z_samples[:, idx]
        cuts = thresholds[var_id]
        indices = np.digitize(z_col, cuts)
        indices = np.clip(indices, 0, len(cat_list) - 1)
        syn_data_dict[col_name] = [cat_list[i] for i in indices]

    # Compute academic impact average and category for synthetic records
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

    # Conditional sampling for Primary Activity (Q4)
    # Computed strictly from the 38 genuine training records
    cond_counts = pd.crosstab(df_train_genuine['impact_category'], df_train_genuine[cols[5]]).reindex(
        index=['Low Impact', 'Moderate Impact', 'High Impact'],
        columns=activity_cats,
        fill_value=0
    )
    # Dirichlet/Laplace smoothing (alpha = 0.5)
    smoothed_probs = (cond_counts + 0.5).div((cond_counts + 0.5).sum(axis=1), axis=0)

    syn_activities = []
    for imp in syn_impact_categories:
        prob_vec = smoothed_probs.loc[imp].values
        chosen_act = np.random.choice(activity_cats, p=prob_vec)
        syn_activities.append(chosen_act)
    syn_data_dict[cols[5]] = syn_activities

    # Consent
    syn_data_dict[cols[1]] = ['Yes, I agree to participate.'] * n_samples

    # Non-colliding synthetic timestamps
    base_start = datetime.datetime(2026, 11, 14, 8, 30, 0)
    real_timestamps = set(df_train_genuine['Timestamp'].tolist())
    syn_timestamps = []

    for _ in range(n_samples):
        while True:
            rand_sec = int(np.random.uniform(0, 72 * 3600))
            candidate_dt = base_start + datetime.timedelta(seconds=rand_sec)
            h_12 = candidate_dt.hour % 12
            if h_12 == 0:
                h_12 = 12
            period = "AM" if candidate_dt.hour < 12 else "PM"
            ts_candidate = f"{candidate_dt.year}/{candidate_dt.month:02d}/{candidate_dt.day:02d} {h_12}:{candidate_dt.minute:02d}:{candidate_dt.second:02d} {period} GMT+5"
            if ts_candidate not in real_timestamps and ts_candidate not in syn_timestamps:
                syn_timestamps.append(ts_candidate)
                break
    syn_data_dict['Timestamp'] = syn_timestamps

    # Qualitative Responses with explicit [Synthetic] marker
    q13_templates = {
        'Social media': [
            "[Synthetic] Staying connected with study groups and sharing notes with batchmates.",
            "[Synthetic] Communication with classmates and quick access to department announcements.",
            "[Synthetic] Group project coordination and following educational accounts on LinkedIn.",
            "[Synthetic] Connecting with university societies and participating in academic forums."
        ],
        'Educational activities': [
            "[Synthetic] Accessing lecture slides, reading research papers, and using AI coding assistants.",
            "[Synthetic] Watching online tutorials, solving programming problems, and reviewing class notes.",
            "[Synthetic] Downloading textbooks, accessing LMS portals, and completing online quizzes.",
            "[Synthetic] Running algorithmic simulations, watching webinars, and organizing study material."
        ],
        'Watching videos and entertainment': [
            "[Synthetic] Watching concept explanations on YouTube and relaxing between study sessions.",
            "[Synthetic] Accessing video lectures, recorded webinars, and educational channels.",
            "[Synthetic] Educational documentaries and refreshing my mind after intense classes."
        ],
        'Communication and messaging': [
            "[Synthetic] Instant messaging with project partners and contacting professors via email.",
            "[Synthetic] Asking doubts in student WhatsApp groups and receiving study material.",
            "[Synthetic] Fast coordination for group presentations and university submissions."
        ],
        'Online gaming': [
            "[Synthetic] Quick gaming breaks between study sessions to relieve academic stress.",
            "[Synthetic] Problem-solving mindset training and unwinding after project deadlines."
        ],
        'Other': [
            "[Synthetic] Managing personal schedules, tracking assignment deadlines, and general productivity.",
            "[Synthetic] Cloud storage backups, checking timetable updates, and reading newsletters."
        ]
    }

    q14_templates = {
        'Low Impact': [
            "[Synthetic] Occasional mild eye fatigue during late-night reading assignments.",
            "[Synthetic] Minor neck stiffness when working on laptops for prolonged periods.",
            "[Synthetic] Slight delay in sleep schedule when responding to messages late at night."
        ],
        'Moderate Impact': [
            "[Synthetic] Losing track of time while browsing feeds instead of starting homework.",
            "[Synthetic] Procrastinating on challenging academic tasks by checking social apps.",
            "[Synthetic] Reduced deep focus during exam revision due to frequent notification pings.",
            "[Synthetic] Difficulty sustaining concentration during long two-hour lecture periods."
        ],
        'High Impact': [
            "[Synthetic] Severe procrastination leading to late assignment submissions and sleep deprivation.",
            "[Synthetic] Inability to concentrate on studying without compulsively checking my smartphone.",
            "[Synthetic] Chronic sleep disruption, eye strain, and acute exam anxiety due to excessive screen time.",
            "[Synthetic] Frequent family arguments regarding staying on my phone during family dinners."
        ]
    }

    q15_templates = [
        "[Synthetic] Set strict daily app limits and enable Do Not Disturb mode during study blocks.",
        "[Synthetic] Keep the phone in another room or out of arm's reach while attending lectures.",
        "[Synthetic] Use the Pomodoro technique (25 min study, 5 min screen break) to stay disciplined.",
        "[Synthetic] Turn off non-essential social notifications and schedule specific checking intervals.",
        "[Synthetic] Replace evening screen scrolling with physical exercise or reading physical books.",
        "[Synthetic] Use dedicated website and app blockers during exam preparation weeks."
    ]

    syn_q13 = []
    for act in syn_activities:
        pool = q13_templates.get(act, q13_templates['Other'])
        syn_q13.append(np.random.choice(pool))

    syn_q14 = []
    for imp in syn_impact_categories:
        pool = q14_templates[imp]
        syn_q14.append(np.random.choice(pool))

    syn_q15 = [np.random.choice(q15_templates) for _ in range(n_samples)]

    syn_data_dict[cols[14]] = syn_q13
    syn_data_dict[cols[15]] = syn_q14
    syn_data_dict[cols[16]] = syn_q15

    df_syn = pd.DataFrame(syn_data_dict)[cols]
    df_syn['academic_impact_avg'] = [round(x, 4) for x in syn_academic_avg]
    df_syn['impact_category'] = syn_impact_categories
    df_syn['data_source'] = 'synthetic_training_only'

    print(f"[INFO] Generated {len(df_syn)} training-only synthetic observations.")
    print(f"[INFO] Synthetic distribution:\n{df_syn['impact_category'].value_counts()}")
    return df_syn


def engineer_features(df):
    """
    Applies corrected feature engineering:
    - One-hot academic year: year_First year, year_Second year, year_Third year, year_Fourth year, year_Other
    - Corrected social conflict score: Q11_scaled = 2 * Q11 - 1, score = (Q9 + Q10 + Q11_scaled) / 3
    - Digital dependency score: (daily_device_hours + check_frequency + stay_away_difficulty) / 3
    - Activity one-hot encoding (6 classes)
    - Gender binary encoding (Female: 0, Male: 1)
    """
    cols = list(df.columns)
    
    col_gender = [c for c in cols if 'gender' in c.lower()][0]
    col_year = [c for c in cols if 'academic year' in c.lower()][0]
    col_time = [c for c in cols if 'spend using digital devices daily' in c.lower()][0]
    col_act = [c for c in cols if 'activity do you use digital devices' in c.lower()][0]
    col_freq = [c for c in cols if 'check your smartphone' in c.lower()][0]
    col_q9 = [c for c in cols if 'spending time with friends' in c.lower()][0]
    col_q10 = [c for c in cols if 'interrupts my face-to-face' in c.lower()][0]
    col_q11 = [c for c in cols if 'complained about your excessive' in c.lower()][0]
    col_q12 = [c for c in cols if 'stay away from your smartphone' in c.lower()][0]
    
    likert_5 = {'Strongly disagree': 1, 'Disagree': 2, 'Neutral': 3, 'Agree': 4, 'Strongly agree': 5}
    time_map = {'Less than 2 hours': 1, '2–4 hours': 2, '4–6 hours': 3, '6–8 hours': 4, 'More than 8 hours': 5}
    freq_map = {'Less than 5 times a day': 1, '5–10 times a day': 2, '11–20 times a day': 3, 'More than 20 times a day': 4, 'Almost continuously': 5}
    complain_map = {'No': 1, 'Sometimes': 2, 'Yes': 3}
    stay_away_map = {'Very easy': 1, 'Easy': 2, 'Neutral': 3, 'Difficult': 4, 'Very difficult': 5}
    gender_map = {'Female': 0, 'Male': 1}

    feat = pd.DataFrame()

    # 1. Gender
    feat['gender_encoded'] = df[col_gender].map(gender_map)

    # 2. Academic Year (One-Hot Encoded into 5 distinct categories)
    year_cats = ['First year', 'Second year', 'Third year', 'Fourth year', 'Other']
    for yr in year_cats:
        feat[f'year_{yr}'] = (df[col_year] == yr).astype(int)

    # 3. Behavioral Features (Atomic)
    feat['daily_device_hours'] = df[col_time].map(time_map)
    feat['check_frequency'] = df[col_freq].map(freq_map)
    feat['phone_with_family'] = df[col_q9].map(likert_5)
    feat['conversation_interruption'] = df[col_q10].map(likert_5)
    feat['social_complaints'] = df[col_q11].map(complain_map)
    feat['stay_away_difficulty'] = df[col_q12].map(stay_away_map)

    # 4. Activity (One-Hot Encoded into 6 distinct categories)
    activity_cats = [
        'Communication and messaging',
        'Educational activities',
        'Online gaming',
        'Other',
        'Social media',
        'Watching videos and entertainment'
    ]
    for act in activity_cats:
        feat[f'activity_{act}'] = (df[col_act] == act).astype(int)

    # 5. Corrected Composite Indices
    # Q11_scaled = 2 * Q11 - 1 (maps {1,2,3} to {1,3,5})
    # social_conflict_score = (Q9 + Q10 + Q11_scaled) / 3 -> Theoretical range: [1.0, 5.0]
    q11_scaled = 2 * feat['social_complaints'] - 1
    feat['social_conflict_score'] = (
        (feat['phone_with_family'] + feat['conversation_interruption'] + q11_scaled) / 3.0
    ).round(4)

    # digital_dependency_score = (daily_device_hours + check_frequency + stay_away_difficulty) / 3
    feat['digital_dependency_score'] = (
        (feat['daily_device_hours'] + feat['check_frequency'] + feat['stay_away_difficulty']) / 3.0
    ).round(4)

    # Validate score bounds
    assert feat['social_conflict_score'].min() >= 1.0 and feat['social_conflict_score'].max() <= 5.0, \
        f"social_conflict_score out of range: [{feat['social_conflict_score'].min()}, {feat['social_conflict_score'].max()}]"
    assert feat['digital_dependency_score'].min() >= 1.0 and feat['digital_dependency_score'].max() <= 5.0, \
        f"digital_dependency_score out of range: [{feat['digital_dependency_score'].min()}, {feat['digital_dependency_score'].max()}]"

    # Targets (if available in df)
    if 'academic_impact_avg' in df.columns:
        feat['academic_impact_avg'] = df['academic_impact_avg']
    if 'impact_category' in df.columns:
        feat['impact_category'] = df['impact_category']
        impact_map = {'Low Impact': 0, 'Moderate Impact': 1, 'High Impact': 2}
        feat['impact_encoded'] = feat['impact_category'].map(impact_map)
    if 'data_source' in df.columns:
        feat['data_source'] = df['data_source']

    return feat


def main():
    print("[STEP 1] Ingesting genuine survey dataset...")
    df_genuine = load_genuine_data()
    df_genuine['data_source'] = 'real'
    
    print("[STEP 2] Stratified split of 54 genuine responses (38 train, 16 test)...")
    df_train_gen, df_test_gen = split_genuine_data(df_genuine)

    print("[STEP 3] Fitting synthetic generator strictly on 38 genuine train records...")
    df_syn_train = generate_training_only_synthetics(df_train_gen, n_samples=96)

    # Assemble augmented training set
    df_train_aug = pd.concat([df_train_gen, df_syn_train], ignore_index=True)
    df_train_aug.insert(0, 'train_record_id', range(1, len(df_train_aug) + 1))
    
    df_train_gen_tagged = df_train_gen.copy()
    df_train_gen_tagged.insert(0, 'train_record_id', range(1, len(df_train_gen_tagged) + 1))

    df_test_gen_tagged = df_test_gen.copy()
    df_test_gen_tagged.insert(0, 'test_record_id', range(1, len(df_test_gen_tagged) + 1))

    print("[STEP 4] Exporting raw data splits...")
    df_train_gen_tagged.to_csv(DATA_SPLITS / "genuine_train.csv", index=False)
    df_test_gen_tagged.to_csv(DATA_SPLITS / "genuine_test.csv", index=False)
    df_syn_train.to_csv(DATA_SPLITS / "synthetic_train_augmented.csv", index=False)
    df_train_aug.to_csv(DATA_SPLITS / "train_augmented.csv", index=False)
    print("Saved genuine_train.csv, genuine_test.csv, synthetic_train_augmented.csv, train_augmented.csv")

    print("[STEP 5] Applying corrected feature engineering...")
    feat_train_aug = engineer_features(df_train_aug)
    feat_train_gen = engineer_features(df_train_gen_tagged)
    feat_test_gen = engineer_features(df_test_gen_tagged)
    
    # Also engineer features for the full 54 genuine responses (for genuine clustering)
    feat_genuine_54 = engineer_features(df_genuine)
    
    # Also engineer features for the full 150 augmented dataset (for 150-row clustering)
    # Using original historical synthetic dataset + genuine 54 with corrected features
    historical_syn_path = DATA_RAW / "synthetic_responses.csv"
    if historical_syn_path.exists():
        df_hist_syn = pd.read_csv(historical_syn_path)
        # Add academic_impact_avg & impact_category if needed
        likert_5 = {'Strongly disagree': 1, 'Disagree': 2, 'Neutral': 3, 'Agree': 4, 'Strongly agree': 5}
        q6 = df_hist_syn.iloc[:, 7].map(likert_5)
        q7 = df_hist_syn.iloc[:, 8].map(likert_5)
        q8 = df_hist_syn.iloc[:, 9].map(likert_5)
        avg = (q6 + q7 + q8) / 3.0
        def assign_category(score):
            if score <= 2.33: return 'Low Impact'
            elif score <= 3.66: return 'Moderate Impact'
            else: return 'High Impact'
        df_hist_syn['academic_impact_avg'] = avg.round(4)
        df_hist_syn['impact_category'] = avg.apply(assign_category)
        df_hist_syn['data_source'] = 'synthetic_historical'
        
        df_full_150 = pd.concat([df_genuine, df_hist_syn], ignore_index=True)
        feat_full_150 = engineer_features(df_full_150)
    else:
        feat_full_150 = None

    print("[STEP 6] Defining and validating Feature Configurations...")
    config_a_cols = [
        'gender_encoded',
        'year_First year', 'year_Second year', 'year_Third year', 'year_Fourth year', 'year_Other',
        'daily_device_hours', 'check_frequency',
        'phone_with_family', 'conversation_interruption', 'social_complaints', 'stay_away_difficulty',
        'activity_Communication and messaging', 'activity_Educational activities',
        'activity_Online gaming', 'activity_Other', 'activity_Social media',
        'activity_Watching videos and entertainment'
    ]
    
    config_b_cols = [
        'gender_encoded',
        'year_First year', 'year_Second year', 'year_Third year', 'year_Fourth year', 'year_Other',
        'social_conflict_score', 'digital_dependency_score',
        'activity_Communication and messaging', 'activity_Educational activities',
        'activity_Online gaming', 'activity_Other', 'activity_Social media',
        'activity_Watching videos and entertainment'
    ]

    print(f"Configuration A (Atomic) feature count: {len(config_a_cols)}")
    print(f"Configuration B (Composite) feature count: {len(config_b_cols)}")
    assert len(config_a_cols) == 18, f"Config A must have 18 features, got {len(config_a_cols)}"
    assert len(config_b_cols) == 14, f"Config B must have 14 features, got {len(config_b_cols)}"

    # Check for target leakage in both configs
    forbidden = {'academic_impact_avg', 'impact_category', 'impact_encoded', 'record_id', 'train_record_id', 'test_record_id', 'data_source', 'Timestamp'}
    assert not forbidden.intersection(set(config_a_cols)), "Target leakage detected in Config A!"
    assert not forbidden.intersection(set(config_b_cols)), "Target leakage detected in Config B!"

    # Export feature datasets
    feat_train_aug.to_csv(DATA_SPLITS / "train_augmented_features.csv", index=False)
    feat_train_gen.to_csv(DATA_SPLITS / "genuine_train_features.csv", index=False)
    feat_test_gen.to_csv(DATA_SPLITS / "genuine_test_features.csv", index=False)
    feat_genuine_54.to_csv(DATA_PROCESSED / "genuine_54_corrected_features.csv", index=False)
    if feat_full_150 is not None:
        feat_full_150.to_csv(DATA_PROCESSED / "full_150_corrected_features.csv", index=False)

    print("[SUCCESS] All Phase II datasets prepared, validated, and saved successfully!")
    print(f"  - Train Augmented (N=134): {DATA_SPLITS / 'train_augmented_features.csv'}")
    print(f"  - Genuine Train (N=38): {DATA_SPLITS / 'genuine_train_features.csv'}")
    print(f"  - Genuine Holdout Test (N=16): {DATA_SPLITS / 'genuine_test_features.csv'}")
    print(f"  - Genuine 54 (N=54): {DATA_PROCESSED / 'genuine_54_corrected_features.csv'}")
    if feat_full_150 is not None:
        print(f"  - Full 150 (N=150): {DATA_PROCESSED / 'full_150_corrected_features.csv'}")


if __name__ == "__main__":
    main()
