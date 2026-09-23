# -*- coding: utf-8 -*-
"""Training for PLASTICS-THz + Leave-one-day-out + alpha Norm """

import hashlib
import json
import os
import platform
import random
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from scipy.signal import savgol_filter

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

OUTDIR = os.path.join(REPO, 'results', 'exp_5_v2')
OUT_PREFIX = 'lodo_'  # output filename prefix
WORKERS = os.cpu_count() or 4  # parallel (fold, norm) processes
SMOKE = False  # True -> fold 0 only, K=[10, 3]
MODE = "perfold"  # "perfold" or "nested-shared"
SEED = 42  # fixed seed for reproducibility (random_state, np.random, random.seed)

K_LIST = [50, 20, 10, 5, 3, 1]

FREQS_ALL = list(range(100, 591, 10))

MODELS_ORDER = ['RF', 'NB', 'LR', 'GB', 'SVM']

THICKNESS_MM = {
    'A': 0.20,   # PE/tie/EVOH/tie/PE/Adhesive/PE/tie/EVOH/tie/PE
    'B': 0.57,   # PE/tie/EVOH/tie/PE (Admer AT1707E)
    'C': 2.05,   # ABS+PC
    'D': 3.00,   # ABS
    'E': 0.10,   # Ecovio/PVOH/Ecovio
    'F': 0.29,   # PP/tie/EVOH/tie/PP (tupper, 0.27-0.31 -> 0.29)
    'G': 0.10,   # PHB/PVOH/Ecovio
    'H': 0.07,   # PP/tie/EVOH/tie/PP
    'I': 0.36,   # PS
    'J': 0.07,   # LDPE
    'L': 1.85,   # PVC
    'O': 0.12,   # PET
}
ALPHA_REF_FLOOR_MV = 0.5  # |HG median| below this ~= dead-band noise
ALPHA_LG_FLOOR_MV = 3.0

WINDOW_S = 0.1

LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'O']
TEST_NORMS = ['alpha']  

NORM_LABELS = {'baseline': 'baseline T(f)', 'alpha': 'alpha(f) = -ln(T)/d'}
APPLY_SCALING = True
APPLY_SG = False      # AFTER windowing
SG_W = 3
SG_P = 2
APPLY_PRE_SG = False  # BEFORE windowing
PRE_SG_W = 7
PRE_SG_P = 2
APPLY_PCA = False
APPLY_LDA = False   
APPLY_QDA = False
APPLY_ICA = False

TRAIN_DIR = 'data/experiment_5_plastics/processed/'

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    return seed

def load_data_with_groups(input_path):
    """ Load data but keeps filename -> Day/SourceFile grouping keys."""
    frames = []
    for file in sorted(os.listdir(input_path)):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_path, file), delimiter=';', header=0)
            df['SourceFile'] = file
            m = re.match(r'[A-Z](\d+)_', file) # Day = leading number after polymer letter
            df['Day'] = int(m.group(1)) if m else -1
            frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data['Sample'] = data['Sample'].str[0]  # same cleaning as prepare_train_test_data
    return data

def load_grouped_data(notebook_dir, train_dir=TRAIN_DIR):
    """Load single-directory long-format frame (all days; windowing happens AFTER, once)."""
    data = load_data_with_groups(os.path.normpath(os.path.join(notebook_dir, '..', '..', train_dir)))
    keep = [c for c in ['Frequency (GHz)', 'LG (mV)', 'HG (mV)', 'Sample', 'Day', 'SourceFile'] if c in data.columns]
    return data[keep]

def _mode_first(s):
    m = s.mode()
    return m.iloc[0] if len(m) else s.iloc[0]

def grouped_window_averages(df, data_percentage):
    results = []
    for (sample, freq), group in df.groupby(['Sample', 'Frequency (GHz)']):
        window_size = max(1, int(len(group) * data_percentage / 100))
        for start in range(0, len(group), window_size):
            window_data = group.iloc[start:start + window_size]
            mean_values = window_data[['LG (mV)', 'HG (mV)']].mean()
            std_deviation_values = window_data[['LG (mV)', 'HG (mV)']].std()
            results.append({
                'Frequency (GHz)': freq,
                'LG (mV) mean': mean_values['LG (mV)'],
                'HG (mV) mean': mean_values['HG (mV)'],
                'LG (mV) std deviation': std_deviation_values['LG (mV)'],
                'HG (mV) std deviation': std_deviation_values['HG (mV)'],
                'Sample': sample,
                'Day': _mode_first(window_data['Day']),
                'SourceFile': _mode_first(window_data['SourceFile']),
            })
    return pd.DataFrame(results)

def grouped_pivot(df, data_percentage):
    """Same pivot as freq_as_variable + Day/SourceFile carried per (Sample, unique_id) row."""
    df_window = grouped_window_averages(df, data_percentage)
    df_window['unique_id'] = df_window.groupby(['Sample', 'Frequency (GHz)']).cumcount()
    feat = df_window.drop(columns=['Day', 'SourceFile'])
    df_pivot = feat.pivot(index=['Sample', 'unique_id'], columns='Frequency (GHz)')
    df_pivot.columns = [' '.join([str(col[1]), str(col[0])]) for col in df_pivot.columns]
    df_pivot = df_pivot.dropna(axis=1, how='all')
    df_pivot = df_pivot.reset_index()
    meta = (df_window.groupby(['Sample', 'unique_id'])
            .agg(Day=('Day', _mode_first), SourceFile=('SourceFile', _mode_first))
            .reset_index())
    df_pivot = df_pivot.merge(meta, on=['Sample', 'unique_id'], how='left')
    df_pivot = df_pivot.drop(columns=['unique_id'])
    feat_cols = sorted([c for c in df_pivot.columns if c not in ('Sample', 'Day', 'SourceFile')])
    return df_pivot[['Sample', 'Day', 'SourceFile'] + feat_cols]

def preprocess_data(df, labels, freqs, eliminate_std_dev=False, eliminate_LG=False, drop_sample=True):

    X_ = df[df['Sample'].isin(labels)]
    y_ = X_['Sample']

    if drop_sample:
        X_ = X_.drop(columns=['Sample'])

    if freqs:
        # Subset of specific frequencies and features to use as input 
        columns = [f'{freq}.0 HG (mV) mean' for freq in freqs] + \
                  [f'{freq}.0 LG (mV) mean' for freq in freqs] + \
                  [f'{freq}.0 HG (mV)' for freq in freqs] + \
                  [f'{freq}.0 LG (mV)' for freq in freqs] + \
                  [f'{freq}.0 HG (mV) std deviation' for freq in freqs] + \
                  [f'{freq}.0 LG (mV) std deviation' for freq in freqs] + \
                  ['Sample']
        existing_columns = [col for col in columns if col in X_.columns]
        if not existing_columns:
            print("No matching columns found in X_.")
        else:
            X_ = X_[existing_columns]
        X_ = X_.reindex(sorted(X_.columns), axis=1)

    if eliminate_std_dev:
        # Eliminate std dev columns from the input features
        X_ = X_.drop(columns=[col for col in X_.columns if 'std deviation' in col])

    if eliminate_LG:
        # Eliminate LG columns from the input features
        X_ = X_.drop(columns=[col for col in X_.columns if 'LG' in col])

    return X_, y_

def add_features(X, y, subset_freqs, HG_diff=True, LG_diff=True):

    X['Sample'] = y
    mean_std_dict = {}

    for freq in subset_freqs:
        # Calculate HG and LG mean values for each frequency
        agg_dict = {}
        if f'{freq}.0 LG (mV) mean' in X.columns:
            agg_dict['LG_mean'] = (f'{freq}.0 LG (mV) mean', 'mean')
        if f'{freq}.0 HG (mV) mean' in X.columns:
            agg_dict['HG_mean'] = (f'{freq}.0 HG (mV) mean', 'mean')
        mean_std_dict[freq] = X.groupby('Sample').agg(**agg_dict).reset_index()
        mean_std_dict[freq]['Frequency'] = freq

    # Concatenate all DataFrames in the dictionary
    mean_std_df = pd.concat(mean_std_dict.values(), ignore_index=True)

    # For each frequency after first one
    for i, freq in enumerate(subset_freqs[1:]):
        prev_freq = subset_freqs[i]  # Get previous frequency

        # For each row
        for idx, row in X.iterrows():
            sample = row['Sample']

            if HG_diff:
                # Get previous frequency's HG mean for this sample
                prev_hg = mean_std_df[
                    (mean_std_df['Frequency'] == prev_freq) &
                    (mean_std_df['Sample'] == sample)
                ]['HG_mean'].values[0]

                # 1) Inputs: xt - (xt-1) --First-order differences
                # 2) Inputs: (xt/(xt-1)) - 1 --Relative differences

                # Calculate and store difference
                X.loc[idx, f'{freq}.0 HG diff'] = X.loc[idx, f'{freq}.0 HG (mV) mean'] - prev_hg
                # X.loc[idx, f'{freq}.0 HG relative diff'] = (X.loc[idx, f'{freq}.0 HG (mV) mean'] / prev_hg) -1

            if LG_diff:
                prev_lg = mean_std_df[
                    (mean_std_df['Frequency'] == prev_freq) &
                    (mean_std_df['Sample'] == sample)
                ]['LG_mean'].values[0]

                # Calculate and store difference
                # X.loc[idx, f'{freq}.0 LG diff'] = X.loc[idx, f'{freq}.0 LG (mV) mean'] - prev_lg
                X.loc[idx, f'{freq}.0 LG relative diff'] = (X.loc[idx, f'{freq}.0 LG (mV) mean'] / prev_lg) -1

    X = X.drop(columns=['Sample'])
    return X

def _lr_coef(lr_model):
    """Return LR coef_ matrix whether lr_model is a bare estimator or a Pipeline."""
    if hasattr(lr_model, 'named_steps'):
        for _step in ('logisticregression', 'clf', 'lr'):
            if _step in lr_model.named_steps and hasattr(lr_model.named_steps[_step], 'coef_'):
                return lr_model.named_steps[_step].coef_
    return lr_model.coef_

def train_models(X_train, y_train, seed):
    training_times = []

    # RF-A: tuned depth/trees
    start_time = time.time()
    rf_model = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    rf_model.fit(X_train, y_train)
    training_times.append(time.time() - start_time)

    # Naive Bayes
    start_time = time.time()
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    training_times.append(time.time() - start_time)

    # Logistic Regression
    start_time = time.time()
    lr_model = make_pipeline(StandardScaler(),
                             LogisticRegression(random_state=seed, max_iter=5000))
    lr_model.fit(X_train, y_train)
    training_times.append(time.time() - start_time)

    # Gradient Boosting
    start_time = time.time()
    gb_model = GradientBoostingClassifier(random_state=seed)
    gb_model.fit(X_train, y_train)
    training_times.append(time.time() - start_time)

    # SVM
    start_time = time.time()
    svm_model = SVC(random_state=seed)
    svm_model.fit(X_train, y_train)
    training_times.append(time.time() - start_time)

    return rf_model, nb_model, lr_model, gb_model, svm_model, training_times

def get_feature_importances(rf_model, lr_model, gb_model, nb_model, svm_model, X_train, y_train, seed, plot=True, n=10):
    feature_names = X_train.columns

    # Random Forest feature importances
    rf_feature_importances = rf_model.feature_importances_
    rf_feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': rf_feature_importances})
    rf_feature_importances_df = rf_feature_importances_df.sort_values('Importance', ascending=False)

    # Logistic Regression feature importances (pipeline-aware: see _lr_coef)
    lr_feature_importances = _lr_coef(lr_model)[0]
    lr_feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': lr_feature_importances})
    lr_feature_importances_df = lr_feature_importances_df.sort_values('Importance', ascending=False)

    # Gradient Boosting feature importances
    gb_feature_importances = gb_model.feature_importances_
    gb_feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': gb_feature_importances})
    gb_feature_importances_df = gb_feature_importances_df.sort_values('Importance', ascending=False)

    # Naive Bayes permutation importance (n_jobs: deterministic with fixed random_state)
    result_nb = permutation_importance(nb_model, X_train, y_train, n_repeats=5, random_state=seed, n_jobs=1)
    sorted_idx_nb = result_nb.importances_mean.argsort()[::-1]
    nb_feature_importances_df = pd.DataFrame({'Feature': feature_names[sorted_idx_nb], 'Importance': result_nb.importances_mean[sorted_idx_nb]})

    # SVM permutation importance
    result_svm = permutation_importance(svm_model, X_train, y_train, n_repeats=5, random_state=seed, n_jobs=1)
    sorted_idx_svm = result_svm.importances_mean.argsort()[::-1]
    svm_feature_importances_df = pd.DataFrame({'Feature': feature_names[sorted_idx_svm], 'Importance': result_svm.importances_mean[sorted_idx_svm]})

    if plot:
        # Set standard font family
        plt.rcParams['font.family'] = 'Arial'  # or 'Arial', 'Times New Roman', etc.

        # Create directory for saving feature importance plots
        feature_imp_path = os.path.normpath(os.path.join(OUTDIR, 'feature_importance_detailed/'))
        if not os.path.exists(feature_imp_path):
            os.makedirs(feature_imp_path)

        # Define enhanced color schemes for each model
        colors = {
            'RF': plt.cm.viridis(np.linspace(0.2, 0.8, n)),
            'LR': plt.cm.plasma(np.linspace(0.2, 0.8, n)),
            'GB': plt.cm.inferno(np.linspace(0.2, 0.8, n)),
            'NB': plt.cm.cividis(np.linspace(0.2, 0.8, n)),
            'SVM': plt.cm.magma(np.linspace(0.2, 0.8, n))
        }

        # Random Forest Plot
        fig, ax = plt.subplots(figsize=(20, 10))
        bars = ax.barh(rf_feature_importances_df['Feature'][:n],
                      rf_feature_importances_df['Importance'][:n],
                      color=colors['RF'],
                      edgecolor='white',
                      linewidth=0.8,
                      alpha=0.85)

        # Add gradient effect to bars
        for i, bar in enumerate(bars):
            bar.set_facecolor(colors['RF'][i])

        ax.set_xlabel('Importance', fontsize=20, color='#2E2E2E', family='DejaVu Sans')
        ax.set_title('Random Forest Feature Importances', fontsize=22,
                    color='#2E2E2E', pad=20, family='DejaVu Sans')
        ax.tick_params(axis='x', labelsize=18, colors='#2E2E2E')
        ax.tick_params(axis='y', labelsize=18, colors='#2E2E2E')

        # Enhanced grid styling
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, color='gray')
        ax.set_axisbelow(True)

        # Subtle background gradient
        ax.patch.set_facecolor('#FAFAFA')

        plt.tight_layout()
        plt.savefig(os.path.join(feature_imp_path, 'RF_detailed_feature_importance.pdf'),
                   format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
        plt.show()

        # Logistic Regression Plot
        fig, ax = plt.subplots(figsize=(20, 10))
        bars = ax.barh(lr_feature_importances_df['Feature'][:n],
                      lr_feature_importances_df['Importance'][:n],
                      color=colors['LR'],
                      edgecolor='white',
                      linewidth=0.8,
                      alpha=0.85)

        for i, bar in enumerate(bars):
            bar.set_facecolor(colors['LR'][i])

        ax.set_xlabel('Importance', fontsize=18, color='#2E2E2E', family='DejaVu Sans')
        ax.set_title('Logistic Regression Feature Importances', fontsize=22,
                    color='#2E2E2E', pad=20, family='DejaVu Sans')
        ax.tick_params(axis='x', labelsize=16, colors='#2E2E2E')
        ax.tick_params(axis='y', labelsize=16, colors='#2E2E2E')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, color='gray')
        ax.set_axisbelow(True)
        ax.patch.set_facecolor('#FAFAFA')

        plt.tight_layout()
        plt.savefig(os.path.join(feature_imp_path, 'LR_detailed_feature_importance.pdf'),
                   format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
        plt.show()

        # Gradient Boosting Plot
        fig, ax = plt.subplots(figsize=(20, 10))
        bars = ax.barh(gb_feature_importances_df['Feature'][:n],
                      gb_feature_importances_df['Importance'][:n],
                      color=colors['GB'],
                      edgecolor='white',
                      linewidth=0.8,
                      alpha=0.85)

        for i, bar in enumerate(bars):
            bar.set_facecolor(colors['GB'][i])

        ax.set_xlabel('Importance', fontsize=18, color='#2E2E2E', family='DejaVu Sans')
        ax.set_title('Gradient Boosting Feature Importances', fontsize=22,
                    color='#2E2E2E', pad=20, family='DejaVu Sans')
        ax.tick_params(axis='x', labelsize=16, colors='#2E2E2E')
        ax.tick_params(axis='y', labelsize=16, colors='#2E2E2E')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, color='gray')
        ax.set_axisbelow(True)
        ax.patch.set_facecolor('#FAFAFA')

        plt.tight_layout()
        plt.savefig(os.path.join(feature_imp_path, 'GB_detailed_feature_importance.pdf'),
                   format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
        plt.show()

        # Naive Bayes Plot (Enhanced Boxplot)
        fig, ax = plt.subplots(figsize=(12, 8))
        bp = ax.boxplot(result_nb.importances[sorted_idx_nb][:n].T,
                       vert=False,
                       labels=X_train.columns[sorted_idx_nb][:n],
                       patch_artist=True,
                       boxprops=dict(facecolor='#8E44AD', alpha=0.8, linewidth=1.5),
                       whiskerprops=dict(color='#2E2E2E', linewidth=2),
                       capprops=dict(color='#2E2E2E', linewidth=2),
                       medianprops=dict(color='white', linewidth=3),
                       flierprops=dict(marker='o', markerfacecolor='#E74C3C', markersize=8, alpha=0.8, markeredgecolor='white'))

        ax.set_xlabel('Permutation Importance', fontsize=18, color='#2E2E2E', family='DejaVu Sans')
        ax.set_title('Naive Bayes Permutation Feature Importance', fontsize=22,
                    color='#2E2E2E', pad=20, family='DejaVu Sans')
        ax.tick_params(axis='x', labelsize=16, colors='#2E2E2E')
        ax.tick_params(axis='y', labelsize=16, colors='#2E2E2E')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, color='gray')
        ax.set_axisbelow(True)
        ax.patch.set_facecolor('#FAFAFA')

        plt.tight_layout()
        plt.savefig(os.path.join(feature_imp_path, 'NB_detailed_feature_importance.pdf'),
                   format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
        plt.show()

        # SVM Plot (Enhanced Boxplot)
        fig, ax = plt.subplots(figsize=(12, 8))
        bp = ax.boxplot(result_svm.importances[sorted_idx_svm][:n].T,
                       vert=False,
                       labels=X_train.columns[sorted_idx_svm][:n],
                       patch_artist=True,
                       boxprops=dict(facecolor='#E67E22', alpha=0.8, linewidth=1.5),
                       whiskerprops=dict(color='#2E2E2E', linewidth=2),
                       capprops=dict(color='#2E2E2E', linewidth=2),
                       medianprops=dict(color='white', linewidth=3),
                       flierprops=dict(marker='o', markerfacecolor='#E74C3C', markersize=8, alpha=0.8, markeredgecolor='white'))

        ax.set_xlabel('Permutation Importance', fontsize=18, color='#2E2E2E', family='DejaVu Sans')
        ax.set_title('SVM Permutation Feature Importance', fontsize=22,
                    color='#2E2E2E', pad=20, family='DejaVu Sans')
        ax.tick_params(axis='x', labelsize=16, colors='#2E2E2E')
        ax.tick_params(axis='y', labelsize=16, colors='#2E2E2E')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, color='gray')
        ax.set_axisbelow(True)
        ax.patch.set_facecolor('#FAFAFA')

        plt.tight_layout()
        plt.savefig(os.path.join(feature_imp_path, 'SVM_detailed_feature_importance.pdf'),
                   format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
        plt.show()

    return rf_feature_importances_df, lr_feature_importances_df, gb_feature_importances_df, nb_feature_importances_df, svm_feature_importances_df


def compute_band_reference(df_pivot_train, freqs_all=FREQS_ALL, floor_mv=ALPHA_REF_FLOOR_MV, channel='HG'):
    """Per-frequency reference from TRAIN fold only: median of '{f}.0 {channel} (mV) mean'.

    No empty-system sweep exists for experiment 5, so the reference is the train-fold
    median per band. Computed inside each outer fold from TRAIN rows only (no leakage);
    the same ref transforms both the train and the held-out fold.
    Bands with non-finite, non-positive, or sub-floor |median| (HG: empirically 110-190 GHz
    plus weak 210/220/260/270/280; LG: 100 GHz plus ~290 GHz and everything 300+ GHz)
    get NaN: no signal lives there, only noise around zero.
    """
    ref = {}
    for f in freqs_all:
        col = f'{f}.0 {channel} (mV) mean'
        vals = pd.to_numeric(df_pivot_train[col], errors='coerce').dropna() if col in df_pivot_train.columns else pd.Series([], dtype=float)
        m = float(vals.median()) if len(vals) else float('nan')
        ref[f] = m if (np.isfinite(m) and m > 0 and abs(m) >= floor_mv) else float('nan')
    return ref


def _row_thickness(df_pivot, thickness_map):
    """Per-row thickness from the Sample letter (one thickness per polymer)."""
    d = df_pivot['Sample'].map(thickness_map)
    if d.isna().any():
        raise ValueError(f"Missing thickness for samples: {df_pivot.loc[d.isna(), 'Sample'].unique()}")
    if bool((d <= 0).any()):
        raise ValueError("Non-positive thickness encountered")
    return d


def apply_alpha_pivoted(df_pivot, ref_hg, ref_lg, freqs_all=FREQS_ALL, thickness_map=THICKNESS_MM, eps=1e-6):
    """Beer-Lambert alpha(f) = -ln(T)/d with T = sample(f)/ref(f), per channel.

    HG/LG means with live refs -> alpha; dead-band means -> exact 0 (anti-leak:
    a scaled constant would encode 1/d = the label, one thickness per polymer).
    All std-deviation cols pass through RAW (identical in both arms: std carries
    polymer signal with no systematic thickness scaling, so dividing it by d
    would only inject a 1/d code that exists solely in the alpha arm).
    Column names are preserved so preprocess_data/add_features work unchanged; frames
    stay separate per norm_mode. 'Sample'/'Day'/'SourceFile' columns unchanged.
    """
    out = df_pivot.copy()
    d = _row_thickness(out, thickness_map)
    for f in freqs_all:
        hm = f'{f}.0 HG (mV) mean'
        if hm in out.columns:
            r = ref_hg.get(f, float('nan'))
            if r is not None and np.isfinite(r) and r > 0:
                T = pd.to_numeric(out[hm], errors='coerce') / r
                out[hm] = (-np.log(T.clip(lower=eps))) / d.values
            else:
                # Dead band: no signal, only noise. Constant 0 (NOT -ln(eps)/d):
                # a scaled constant would encode 1/d = the label (one thickness per
                # polymer) and let the alpha arm win via thickness, not spectroscopy.
                out[hm] = 0.0
        lm = f'{f}.0 LG (mV) mean'
        if lm in out.columns:
            r = ref_lg.get(f, float('nan'))
            if r is not None and np.isfinite(r) and r > 0:
                T = pd.to_numeric(out[lm], errors='coerce') / r
                out[lm] = (-np.log(T.clip(lower=eps))) / d.values
            else:
                # Dead LG band: same anti-leak rule as HG (constant 0, not 1/d).
                out[lm] = 0.0
    # NOTE: std-deviation cols intentionally untouched (raw in both arms).
    return out

def frequency_scores_from_importances(imp_by_model, feature_columns, freqs_all=FREQS_ALL):
    """Feature importances -> frequency scores.

    Each frequency votes with its HG-channel columns only (HG mean + HG std):
    HG is the primary spectroscopy channel (dead HG bands carry no signal, so
    they sink instead of being promoted by LG-magnitude votes), while LG/std
    columns still ride along untouched in training. Per model: z-score its
    importances (scales differ across models), mean over the freq's HG cols;
    then average across the 5 models. Returns a ranking table (best first).
    """
    z = {}
    for m, s in imp_by_model.items():
        # Label-aligned: imp Series arrive sorted by importance, so reindex
        s = pd.Series(s, dtype=float).reindex(list(feature_columns)).astype(float)
        sd = float(s.std(ddof=0))
        z[m] = (s - float(s.mean())) / (sd if sd > 0 else 1.0)
    rows = []
    for f in freqs_all:
        cols = [f'{f}.0 HG (mV) mean',
                f'{f}.0 HG (mV) std deviation']
        have = [c for c in cols if c in z['RF'].index]
        rows.append({'Frequency': f,
                     'score': float(np.mean([float(z[m][have].mean()) for m in z])) if have else float('nan'),
                     'n_cols': len(have)})
    tab = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)
    tab['rank'] = tab.index + 1
    return tab

def select_topK_for_fold(Xtr_df, ytr, seed, K_list=K_LIST, freqs_all=FREQS_ALL):
    """V20 ET-free systematic selection on TRAIN fold only -> ({K: top-K}, ranking, None).

    Single bootstrapped-forest voice: mean rank over 10 resample fits of a
    RandomForest (sqrt features) on the train-fit standardized frame
    (Xtr_df must be a DataFrame). Each frequency votes with its HG-channel
    columns only. No ExtraTrees, no NB/SVM/LR/GB votes, no permutation
    scoring, no test data. Provenance: search ledger V20-boot10-sqrt.
    Returns None for models (unused: scaling is ON so run_unit always refits).
    """
    _sc = StandardScaler()
    _Xs = pd.DataFrame(_sc.fit_transform(Xtr_df),
                       columns=[str(c) for c in Xtr_df.columns], index=Xtr_df.index)
    _cols = [str(c) for c in _Xs.columns]
    _yarr = np.asarray(ytr)
    _ranks = []
    for _b in range(10):
        _idx = np.random.RandomState(seed * 1000 + _b).choice(
            len(_yarr), size=len(_yarr), replace=True)
        _m = RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                    max_features="sqrt", n_jobs=-1,
                                    random_state=seed + _b)
        _m.fit(_Xs.iloc[_idx], _yarr[_idx])
        _ranks.append(pd.Series(np.asarray(_m.feature_importances_, dtype=float),
                                index=_cols).rank(ascending=False, method="average"))
    _cons = pd.concat(_ranks, axis=1).mean(axis=1)
    _rows = []
    for _f in freqs_all:
        _have = [c for c in (f'{_f}.0 HG (mV) mean',
                             f'{_f}.0 HG (mV) std deviation') if c in _cols]
        _rows.append({'Frequency': _f,
                      'score': float(_cons[_have].mean()) if _have else float('nan')})
    ranking = pd.DataFrame(_rows).sort_values('score', ascending=True).reset_index(drop=True)
    ranking['rank'] = ranking.index + 1
    topK = {int(K): ranking['Frequency'].head(int(K)).tolist() for K in K_list}
    return topK, ranking, None

def nested_emergent_sets(df_outer_tr, seed, K_list, labels, freqs_all):
    """Inner LOO over the 4 outer-train days -> ({K: emergent bands}, audit).

    Rule (fixed in advance): per K, order bands by (inner-count desc, mean inner
    rank asc), take top-K. Inner selections reuse select_topK_for_fold as-is.
    """
    days = sorted(df_outer_tr["Day"].unique().tolist())
    per_inner = []
    for h in days:
        d = df_outer_tr[df_outer_tr["Day"] != h].reset_index(drop=True)
        X, y = preprocess_data(d, labels, freqs_all, eliminate_std_dev=True)
        X = add_features(X, y, freqs_all, False, False)
        per_inner.append(select_topK_for_fold(X, y, seed, K_list, freqs_all))
    out, audit = {}, {}
    for K in K_list:
        cnt, ranks = Counter(), {}
        for topK, ranking, _ in per_inner:
            for f in topK[int(K)]:
                cnt[f] += 1
            for f, r in ranking.set_index("Frequency")["rank"].items():
                ranks.setdefault(int(f), []).append(int(r))
        ordered = sorted(freqs_all, key=lambda f: (-cnt.get(f, 0),
                         float(np.mean(ranks[f])) if f in ranks else 1e9))
        out[int(K)] = [int(f) for f in ordered[:int(K)]]
        audit[int(K)] = [{"freq": int(f), "n": int(cnt.get(f, 0)),
                          "mr": round(float(np.mean(ranks[f])), 2)} for f in ordered[:8]]
    return out, audit


def nested_emergent_sets_alpha(df_outer_tr, seed, K_list, labels, freqs_all):
    """Inner LOO on the alpha path -> ({K: emergent bands}, audit).

    Same rule as baseline version, but each inner-train computes its own
    alpha refs (train rows only) and selects on alpha-transformed data.
    """
    days = sorted(df_outer_tr["Day"].unique().tolist())
    per_inner = []
    for h in days:
        d = df_outer_tr[df_outer_tr["Day"] != h].reset_index(drop=True)
        _ref = compute_band_reference(d, freqs_all)
        _ref_lg = compute_band_reference(d, freqs_all, ALPHA_LG_FLOOR_MV, "LG")
        dn = apply_alpha_pivoted(d, _ref, _ref_lg, freqs_all)
        X, y = preprocess_data(dn, labels, freqs_all, eliminate_std_dev=True)
        X = add_features(X, y, freqs_all, False, False)
        per_inner.append(select_topK_for_fold(X, y, seed, K_list, freqs_all))
    out, audit = {}, {}
    for K in K_list:
        cnt, ranks = Counter(), {}
        for topK, ranking, _ in per_inner:
            for f in topK[int(K)]:
                cnt[f] += 1
            for f, r in ranking.set_index("Frequency")["rank"].items():
                ranks.setdefault(int(f), []).append(int(r))
        ordered = sorted(freqs_all, key=lambda f: (-cnt.get(f, 0),
                         float(np.mean(ranks[f])) if f in ranks else 1e9))
        out[int(K)] = [int(f) for f in ordered[:int(K)]]
        audit[int(K)] = [{"freq": int(f), "n": int(cnt.get(f, 0)),
                          "mr": round(float(np.mean(ranks[f])), 2)} for f in ordered[:8]]
    return out, audit


def run_unit(fold, held_day, norm, df_tr, df_te, seed, K_list, labels, freqs_all, flags,
             fixed_topK=None, option="sel"):
    """One (fold, norm) unit: transform, fit+score per K. fixed_topK bypasses selection."""
    set_seed(seed)
    if norm == "baseline":
        _tr_n, _te_n, _ref, _ref_lg = df_tr.copy(), df_te.copy(), None, None
    elif norm == "alpha":
        _ref = compute_band_reference(df_tr, freqs_all)
        _ref_lg = compute_band_reference(df_tr, freqs_all, ALPHA_LG_FLOOR_MV, "LG")
        _tr_n = apply_alpha_pivoted(df_tr, _ref, _ref_lg, freqs_all)
        _te_n = apply_alpha_pivoted(df_te, _ref, _ref_lg, freqs_all)
    else:
        raise ValueError(f"unknown norm: {norm} (TEST_NORMS must be a subset of {{'alpha'}})")
    _t0 = time.time()
    if fixed_topK is not None:
        _topK, _ranking, _sel_models = {int(K): list(v) for K, v in fixed_topK.items()}, None, None
        _sel_time = 0.0
    else:
        _Xtr50, _ytr50 = preprocess_data(_tr_n, labels, freqs_all, eliminate_std_dev=True)
        _Xtr50 = add_features(_Xtr50, _ytr50, freqs_all, False, False)
        _topK, _ranking, _sel_models = select_topK_for_fold(_Xtr50, _ytr50, seed, K_list, freqs_all)
        _sel_time = time.time() - _t0
    _raw_path = not (flags["scaling"] or flags["sg"] or flags["pca"]
                     or flags["lda"] or flags["qda"] or flags["ica"])
    records = []
    for _K in K_list:
        _fq = _topK[int(_K)]
        _Xtr, _ytr = preprocess_data(_tr_n, labels, _fq, eliminate_std_dev=True)
        _Xtr = add_features(_Xtr, _ytr, _fq, False, False)
        _Xte, _yte = preprocess_data(_te_n, labels, _fq, eliminate_std_dev=True)
        _Xte = add_features(_Xte, _yte, _fq, False, False)
        if flags["scaling"]:
            _sc = StandardScaler()
            _Xtr = _sc.fit_transform(_Xtr)
            _Xte = _sc.transform(_Xte)
        if flags["sg"] and _Xtr.shape[1] >= flags["sg_w"]:
            # Guard: K=1 yields 2 feats < window 3 -> savgol would raise;
            # passthrough raw (smoothing undefined on <window points).
            _Xtr = savgol_filter(_Xtr, window_length=flags["sg_w"], polyorder=flags["sg_p"])
            _Xte = savgol_filter(_Xte, window_length=flags["sg_w"], polyorder=flags["sg_p"])
        if flags["pca"]:
            from sklearn.decomposition import PCA as _PCA
            _pc = _PCA(n_components=0.95, random_state=seed)
            _Xtr = _pc.fit_transform(_Xtr)
            _Xte = _pc.transform(_Xte)
        if flags["lda"]:
            _ld = LinearDiscriminantAnalysis()
            _Xtr = _ld.fit_transform(_Xtr, _ytr)
            _Xte = _ld.transform(_Xte)
        if flags["qda"]:
            # reg_param>0 floors SVD eigenvalues: alpha dead bands are exact-constant
            # cols -> exact-zero S2 -> S^-0.5=inf, 0*inf=NaN posteriors (crashed NB
            # at K<=10). Floor is exactly reg_param at any feature scale.
            _qd = QuadraticDiscriminantAnalysis(reg_param=0.01)
            _qd.fit(_Xtr, _ytr)
            _Xtr = np.hstack((_Xtr, _qd.predict_proba(_Xtr)))
            _Xte = np.hstack((_Xte, _qd.predict_proba(_Xte)))
        if flags["ica"]:
            # Alpha dead bands are exact-constant cols -> zero singular values ->
            # inf/NaN in FastICA's default svd whitening (crashed the alpha arm).
            # Drop them (mask from TRAIN only) and scale components with K:
            # min(live feats, train classes - 1), so the K sweep stays meaningful
            # instead of collapsing every K to 2 dims.
            _Atr = np.asarray(_Xtr, dtype=float)
            _Ate = np.asarray(_Xte, dtype=float)
            _live = _Atr.std(axis=0) > 0
            _Atr, _Ate = _Atr[:, _live], _Ate[:, _live]
            _n_ic = min(_Atr.shape[1], len(np.unique(_ytr)) - 1)
            if _n_ic >= 1:
                _ic = FastICA(n_components=_n_ic, random_state=seed)
                _Xtr = _ic.fit_transform(_Atr)
                _Xte = _ic.transform(_Ate)
            else:
                _Xtr, _Xte = _Atr, _Ate
        _n_feat = _Xtr.shape[1]
        if int(_K) == 50 and _raw_path and _sel_models is not None:
            _fitted = list(_sel_models)
            _times = [0.0] * 5
        else:
            _res = train_models(_Xtr, _ytr, seed)
            _fitted, _times = list(_res[:5]), [float(t) for t in _res[5]]
        for _mi, _mn in enumerate(MODELS_ORDER):
            _yp = _fitted[_mi].predict(_Xte)
            _yv = _yte.values
            _m_eg = np.isin(_yv, ["E", "G"])
            _m_hj = np.isin(_yv, ["H", "J"])
            records.append({
                "fold": fold, "held_out_day": held_day, "norm_mode": norm, "option": option,
                "K": int(_K), "model": _mn,
                "acc": float(accuracy_score(_yte, _yp)),
                "prec": float(precision_score(_yte, _yp, average="weighted", zero_division=0)),
                "rec": float(recall_score(_yte, _yp, average="weighted", zero_division=0)),
                "f1": float(f1_score(_yte, _yp, average="weighted", zero_division=0)),
                "acc_EG": float((_yp[_m_eg] == _yv[_m_eg]).mean()) if _m_eg.sum() else float("nan"),
                "acc_HJ": float((_yp[_m_hj] == _yv[_m_hj]).mean()) if _m_hj.sum() else float("nan"),
                "n_feat": int(_n_feat),
                "train_time_s": float(_times[_mi]),
                "selected_freqs": ",".join(str(f) for f in _fq),
            })
    return {"records": records, "topK": _topK, "ranking": _ranking,
            "ref": _ref, "ref_lg": _ref_lg, "sel_time_s": float(_sel_time)}

def apply_pre_sg(df_long, window_length, polyorder):
    """Temporal Savitzky-Golay denoise of raw LG/HG before windowing.
    Applied per (Sample, Frequency, SourceFile) group along acquisition order,
    so no window ever spans a file/day boundary.
    """
    if int(window_length) % 2 != 1:
        raise ValueError(f"SG window_length must be odd, got {window_length}")
    if not (int(polyorder) < int(window_length)):
        raise ValueError(f"SG polyorder ({polyorder}) must be < window_length ({window_length})")
    from scipy.signal import savgol_filter as _sg
    parts = []
    for _, group in df_long.groupby(['Sample', 'Frequency (GHz)', 'SourceFile'], sort=False):
        vals = group[['LG (mV)', 'HG (mV)']].to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            raise ValueError("Non-finite LG/HG values in pre-SG input")
        if len(group) < int(window_length):
            parts.append(group)
            continue
        group = group.copy()
        group[['LG (mV)', 'HG (mV)']] = _sg(vals, window_length=int(window_length),
                                            polyorder=int(polyorder), axis=0)
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def build_pivot(notebook_nb_dir, window_s, outdir, pre_sg=False, pre_sg_w=5, pre_sg_p=2):
    """Windowing BEFORE the split, with the shared parquet/pickle cache."""
    _dp = (100 / 12) * window_s
    _pivot_cache = os.path.normpath(os.path.join(outdir, "lodo_pivot_cache"))
    _pivot_meta = _pivot_cache + ".json"

    def _pivot_cache_write(df):
        try:
            import pyarrow as _pa
            import pyarrow.parquet as _pq
            _pq.write_table(_pa.Table.from_pandas(df, preserve_index=False),
                            _pivot_cache + ".parquet")
            return "parquet"
        except Exception as _e:
            print(f"parquet cache failed ({type(_e).__name__}: {_e}); falling back to pickle",
                  flush=True)
            df.to_pickle(_pivot_cache + ".pkl")
            return "pickle"

    def _pivot_cache_read(fmt):
        if fmt == "parquet":
            import pyarrow.parquet as _pq
            return _pq.read_table(_pivot_cache + ".parquet").to_pandas()
        return pd.read_pickle(_pivot_cache + ".pkl")

    def _pivot_input_sig():
        h = hashlib.sha256()
        _d = os.path.normpath(os.path.join(notebook_nb_dir, "..", "..",
                                           "data/experiment_5_plastics/processed"))
        for _f in sorted(os.listdir(_d)):
            if _f.endswith(".csv"):
                _st = os.stat(os.path.join(_d, _f))
                h.update(f"{_f}:{_st.st_size}:{int(_st.st_mtime)}".encode())
        try:
            import pyarrow as _pa_sig
            _arrow_v = _pa_sig.__version__
        except Exception:
            _arrow_v = None
        return {"window_s": window_s, "dp": _dp, "inputs": h.hexdigest(),
                "pandas": pd.__version__, "pyarrow": _arrow_v,
                "pre_sg": bool(pre_sg), "pre_sg_w": int(pre_sg_w), "pre_sg_p": int(pre_sg_p)}

    _sig = _pivot_input_sig()
    df_pivot_full = None
    if os.path.exists(_pivot_meta):
        try:
            _meta = json.load(open(_pivot_meta, encoding="utf-8"))
            if _meta.get("sig") == _sig:
                df_pivot_full = _pivot_cache_read(_meta.get("fmt", "parquet"))
                print(f"pivot: loaded cache [{_meta.get('fmt')}] {df_pivot_full.shape}",
                      flush=True)
        except Exception as _e:
            print(f"pivot cache unreadable ({type(_e).__name__}: {_e}); rebuilding", flush=True)
            df_pivot_full = None
    if df_pivot_full is None:
        _df_long = load_grouped_data(notebook_nb_dir)
        if pre_sg:
            _t0 = time.time()

            def _temporal_noise(_df):
                # Median |first-difference| within acquisition groups: pure
                # temporal jitter, blind to between-band/polymer level shifts.
                _d = _df.groupby(['Sample', 'Frequency (GHz)', 'SourceFile'],
                                 sort=False)[['LG (mV)', 'HG (mV)']].diff().abs().stack()
                return float(_d.median())

            _noise_before = _temporal_noise(_df_long)
            _df_long = apply_pre_sg(_df_long, pre_sg_w, pre_sg_p)
            _noise_after = _temporal_noise(_df_long)
            print(f"pre-SG filter (w={pre_sg_w}, p={pre_sg_p}): "
                  f"temporal noise {_noise_before:.4f} -> {_noise_after:.4f} "
                  f"({time.time() - _t0:.1f}s)", flush=True)
        df_pivot_full = grouped_pivot(_df_long, _dp).dropna().reset_index(drop=True)
        _fmt = _pivot_cache_write(df_pivot_full)
        json.dump({"sig": _sig, "fmt": _fmt},
                  open(_pivot_meta, "w", encoding="utf-8"), indent=2, sort_keys=True)
        print(f"pivot: built + cached [{_fmt}] {df_pivot_full.shape}", flush=True)
    return df_pivot_full

def aggregate_stability(cv_results, topk_by_fold, rankings_by_fold, freqs_all, paper_norms, outdir, tag):
    """Stability table + universal sets (mirrors the notebook stability cell)."""
    _n_folds_eff = cv_results["fold"].nunique()
    _req = 4 if _n_folds_eff == 5 else _n_folds_eff
    _stab_rows = []
    for _n in paper_norms:
        for _K in sorted(cv_results["K"].unique()):
            _cnt, _ranks = Counter(), {}
            for _f in sorted(cv_results["fold"].unique()):
                for _freq in topk_by_fold[(_f, _n)][int(_K)]:
                    _cnt[_freq] += 1
                _rk = rankings_by_fold[(_f, _n)]
                if _rk is None:
                    continue  # fixed-set arm: no ranking (zero selection variance)
                for _freq, _r in _rk.set_index("Frequency")["rank"].items():
                    _ranks.setdefault(_freq, []).append(int(_r))
            for _freq in freqs_all:
                _sel = int(_cnt.get(_freq, 0))
                _rr = _ranks.get(_freq, [])
                _stab_rows.append({"norm_mode": _n, "K": int(_K), "freq": int(_freq),
                                   "folds_selected": f"{_sel}/{_n_folds_eff}",
                                   "n_selected": _sel,
                                   "mean_rank": round(float(np.mean(_rr)), 2) if _rr else None,
                                   "in_universal": bool(_sel >= _req)})
    stability_df = pd.DataFrame(_stab_rows)
    _sp = os.path.join(outdir, f"{OUT_PREFIX}{tag}stability.csv")
    stability_df.to_csv(_sp, index=False, sep=";")
    universal = {}
    for _n in paper_norms:
        for _K in sorted(cv_results["K"].unique()):
            _sub = stability_df[(stability_df["norm_mode"] == _n) & (stability_df["K"] == int(_K))
                             & stability_df["in_universal"]]
            universal[(_n, int(_K))] = sorted(_sub["freq"].tolist())
    return stability_df, universal

def save_result_plots(cv_results, stability_df, freqs_all, outdir, tag, prefix, norms):
    """Grouped accuracy bars per K + selection-stability bars.
    norms sets the compared arms, baseline first (bar offsets generalize to N arms).
    """
    _nrms = list(norms)
    _x = np.arange(len(MODELS_ORDER))
    _w = 0.35

    for _K in sorted(cv_results['K'].unique()):
        _fig, _ax = plt.subplots(figsize=(9, 4.5))

        for _j, _n in enumerate(_nrms):
            _mu, _sd = [], []

            for _m in MODELS_ORDER:
                _r = cv_results[
                    (cv_results['norm_mode'] == _n) &
                    (cv_results['K'] == int(_K)) &
                    (cv_results['model'] == _m)
                ]['acc']

                _mu.append(float(_r.mean()))
                _sd.append(float(_r.std(ddof=1)) if len(_r) > 1 else 0.0)

            _bars = _ax.bar(
                _x + (_j - (len(_nrms) - 1) / 2) * _w,
                _mu,
                _w,
                yerr=_sd,
                capsize=3,
                label=NORM_LABELS[_n]
            )

            for _bar, _value, _err in zip(_bars, _mu, _sd):
                _ax.text(
                    _bar.get_x() + _bar.get_width() / 2,
                    _bar.get_height() + _err + 0.02,
                    f'{_value:.1%}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )

        _ax.set_xticks(_x, MODELS_ORDER)
        _ax.set_ylim(0, 1.12)
        _ax.set_ylabel('Accuracy (mean±std over LODO folds)')
        _ax.set_title(f'Held-out-day accuracy | K={_K}')
        _ax.legend()
        _ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(outdir, f'{prefix}{tag}compare_{_K}freqs.pdf'),
            bbox_inches='tight',
            dpi=300
        )
        plt.close()

    for _K in [k for k in (20, 10, 5, 3)
               if k in set(int(k) for k in cv_results['K'].unique())]:

        _fig, _ax = plt.subplots(figsize=(10, 4))
        _fr = list(FREQS_ALL)
        _xi = np.arange(len(_fr))
        _w2 = 0.35
        _n_folds = cv_results['fold'].nunique()

        for _j, _n in enumerate(_nrms):
            _s = stability_df[
                    (stability_df['norm_mode'] == _n) &
                    (stability_df['K'] == int(_K))
                ].set_index('freq')

            _vals = [
                int(_s.loc[f, 'n_selected']) if f in _s.index else 0
                for f in _fr
            ]

            _bars = _ax.bar(
                _xi + (_j - (len(_nrms) - 1) / 2) * _w2,
                _vals,
                _w2,
                label=_n
            )

            for _bar, _value in zip(_bars, _vals):
                _percentage = 100 * _value / _n_folds
                _ax.text(
                    _bar.get_x() + _bar.get_width() / 2,
                    _value + 0.05,
                    f'{_percentage:.0f}%',
                    ha='center',
                    va='bottom',
                    fontsize=6,
                    rotation=90
                )

        _thr = 4 if _n_folds == 5 else _n_folds
        _ax.axhline(_thr, color='k', ls='--', lw=1,
                    label='universal threshold')
        _ax.set_xticks(_xi, _fr, rotation=90, fontsize=7)
        _ax.set_ylabel('Folds selected (%)')
        _ax.set_ylim(0, _n_folds + 1.2)
        _ax.set_title(
            f'Selection stability K={_K} '
            f'(universal ≥{_thr}/{_n_folds} folds)'
        )
        _ax.legend()
        _ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(outdir, f'{prefix}{tag}stability_K{_K}.pdf'),
            bbox_inches='tight',
            dpi=300
        )
        plt.close()

def plot_confusion_matrix_pdf(y_true, y_pred, labels, save_path, model_name):
    """Confusion matrix as headless PDF (Agg: saved, never shown)."""
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap=plt.cm.Blues
    cax = ax.matshow(conf_matrix, cmap=cmap)
    fig.colorbar(cax)

    # Determine text color based on cell value for better visibility
    for i in range(len(labels)):
        for j in range(len(labels)):
            # Calculate percentage
            percentage = conf_matrix[i, j] / np.sum(conf_matrix, axis=1)[i] * 100 if np.sum(conf_matrix, axis=1)[i] > 0 else 0

            # Determine text color based on cell darkness
            cell_value = conf_matrix[i, j]
            if cell_value > conf_matrix.max() / 3:
                text_color = 'white'

                plt.text(j, i, f'{conf_matrix[i, j]}\n{percentage:.1f}%',
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize=8,
                     ha='center', va='center',
                     color=text_color)
            else:
                text_color = cmap(1.0)

                if conf_matrix[i, j] != 0:

                    plt.text(j, i, f'{conf_matrix[i, j]}\n{percentage:.1f}%',
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=8,
                        ha='center', va='center',
                        color=text_color)

    plt.xlabel('Predicted', fontweight='bold', fontsize=12)
    plt.ylabel('True', fontweight='bold', fontsize=12)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, fontweight='bold')
    plt.yticks(np.arange(len(labels)), labels, fontweight='bold')
    plt.title('Confusion Matrix', fontweight='bold', fontsize=14)

    # Adjust layout to make room for rotated x labels
    plt.tight_layout()

    # Save the plot if a path is provided
    if save_path:
        # Create directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Create filename
        model_suffix = f"_{model_name}" if model_name else ""
        filename = f"confusion_matrix{model_suffix}.pdf"
        filepath = os.path.join(save_path, filename)

        # Save as PDF
        plt.savefig(filepath, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Confusion matrix saved to: {filepath}")
    else:
        print("Confusion matrix plot not saved.")

    plt.close(fig)

def save_confusion_pdfs(cv_results, band_refs, band_refs_lg, df_pivot_full, labels,
                        freqs_all, outdir, tag, prefix, seed, norms):
    """Pooled confusion per (norm, K) for the best model by 5-fold mean acc.

    K fixed by the loop; model = argmax of raw (unrounded) fold-mean acc,
    tiebreak by MODELS_ORDER. Predictions are concatenated over all folds
    (each sample is in exactly one test fold), one plot call per (norm, K).
    Refit recipe matches run_unit; alpha uses each fold's own train-fold refs."""
    _fitters = {
        'RF': lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                             n_jobs=-1, random_state=seed),
        'NB': lambda: GaussianNB(),
        'LR': lambda: make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, max_iter=5000)),
        'GB': lambda: GradientBoostingClassifier(random_state=seed),
        'SVM': lambda: SVC(random_state=seed),
    }
    _folds = sorted(cv_results['fold'].unique())
    for _norm in norms:
        for _K in sorted(cv_results['K'].unique()):
            _means = (cv_results[(cv_results['norm_mode'] == _norm) & (cv_results['K'] == int(_K))]
                      .groupby('model')['acc'].mean())
            if _means.empty:
                continue
            _best_acc = float(_means.max())
            _model = min([m for m in _means.index if float(_means[m]) == _best_acc],
                         key=lambda m: MODELS_ORDER.index(m))
            _fq = [int(f) for f in str(cv_results[(cv_results['norm_mode'] == _norm)
                                                  & (cv_results['K'] == int(_K))
                                                  & (cv_results['model'] == _model)]
                                          .iloc[0]['selected_freqs']).split(',')]
            _yt_all, _yp_all = [], []
            for _f in _folds:
                _held = int(cv_results[cv_results['fold'] == _f]['held_out_day'].iloc[0])
                _dtr = df_pivot_full[df_pivot_full['Day'] != _held].reset_index(drop=True)
                _dte = df_pivot_full[df_pivot_full['Day'] == _held].reset_index(drop=True)
                if _norm == 'baseline':
                    _tr_n, _te_n = _dtr.copy(), _dte.copy()
                elif _norm == 'alpha':
                    _tr_n = apply_alpha_pivoted(_dtr, band_refs[(_f, _norm)], band_refs_lg[(_f, _norm)], freqs_all)
                    _te_n = apply_alpha_pivoted(_dte, band_refs[(_f, _norm)], band_refs_lg[(_f, _norm)], freqs_all)
                else:
                    raise ValueError(f"unknown norm: {_norm}")
                _Xtr, _ytr = preprocess_data(_tr_n, labels, _fq, eliminate_std_dev=True)
                _Xtr = add_features(_Xtr, _ytr, _fq, False, False)
                _Xte, _yte = preprocess_data(_te_n, labels, _fq, eliminate_std_dev=True)
                _Xte = add_features(_Xte, _yte, _fq, False, False)
                _cm_model = _fitters[_model]()
                _cm_model.fit(_Xtr, _ytr)
                _yp = _cm_model.predict(_Xte)
                _yt_all.append(_yte.values)
                _yp_all.append(_yp)
            _yt = pd.Series(np.concatenate(_yt_all))
            _yp = np.concatenate(_yp_all)
            print(f'{_norm}: K={int(_K)} model={_model} mean_acc={_best_acc:.4f} pooled_acc={float((_yp == _yt.values).mean()):.4f}')
            plot_confusion_matrix_pdf(_yt, _yp, labels, outdir, f'{prefix}{tag}pooled_{_norm}_K{int(_K)}_{_model}')

def main():
    outdir, workers, smoke, seed = OUTDIR, WORKERS, SMOKE, SEED
    os.makedirs(outdir, exist_ok=True)
    tag = "smoke_" if smoke else ""
    K_list = list(K_LIST) if not smoke else [10, 3]
    paper_norms = ['baseline'] + [n for n in TEST_NORMS if n != 'baseline']
    assert set(paper_norms) <= {'baseline', 'alpha'}, f"unknown norms: {paper_norms}"
    all_norms = list(paper_norms)
    flags = {"scaling": bool(APPLY_SCALING), "sg": bool(APPLY_SG),
             "sg_w": int(SG_W), "sg_p": int(SG_P),
             "pre_sg": bool(APPLY_PRE_SG), "pre_sg_w": int(PRE_SG_W), "pre_sg_p": int(PRE_SG_P),
             "pca": bool(APPLY_PCA), "lda": bool(APPLY_LDA),
             "qda": bool(APPLY_QDA), "ica": bool(APPLY_ICA)}
    print(f"seed={seed} workers={workers} smoke={smoke} outdir={outdir}", flush=True)
    set_seed(seed)
    t_all = time.time()
    nb_dir = os.path.join(REPO, "src", "nb")
    df_pivot_full = build_pivot(nb_dir, float(WINDOW_S), outdir,
                                bool(APPLY_PRE_SG), int(PRE_SG_W), int(PRE_SG_P))
    print(f"pivot: {df_pivot_full.shape}, "
          f"days={sorted(df_pivot_full['Day'].unique())}", flush=True)
    assert set(df_pivot_full["Day"].unique()) == {1, 2, 3, 4, 5}
    splits = list(GroupKFold(n_splits=5).split(
        df_pivot_full, groups=df_pivot_full["Day"].values))
    folds_wanted = [0] if smoke else [0, 1, 2, 3, 4]
    held_by_fold = {}
    for fold in folds_wanted:
        _tri, _tei = splits[fold]
        _held = sorted(df_pivot_full.iloc[_tei]["Day"].unique())
        assert len(_held) == 1, f"fold {fold} mixes days: {_held}"
        held_by_fold[fold] = int(_held[0])
    # ---- outer evaluation (per-fold train-only selection x baseline + alpha, 5 models via run_unit)
    tasks = []
    for fold in folds_wanted:
        tri, tei = splits[fold]
        df_tr = df_pivot_full.iloc[tri].reset_index(drop=True)
        df_te = df_pivot_full.iloc[tei].reset_index(drop=True)
        tasks.append((fold, held_by_fold[fold], "baseline", df_tr, df_te,
                      None, list(K_list), "perfold"))
        tasks.append((fold, held_by_fold[fold], "alpha", df_tr, df_te,
                      None, list(K_list), "perfold"))
    workers = max(1, min(workers, len(tasks)))
    print(f"tasks={len(tasks)} workers={workers}", flush=True)
    outs = Parallel(n_jobs=workers, backend="loky")(
        delayed(run_unit)(fold, held, norm, tr, te, seed, kl,
                          list(LABELS), list(FREQS_ALL), flags,
                          fixed_topK=fx, option=opt)
        for fold, held, norm, tr, te, fx, kl, opt in tasks)
    records, band_refs, band_refs_lg = [], {}, {}
    topK_by_fold, rankings_by_fold = {}, {}
    for (_fold, _held, _norm, _tr, _te, _fx, _kl, _opt), o in zip(tasks, outs):
        records.extend(o["records"])
        topK_by_fold[(_fold, _norm)] = o["topK"]
        rankings_by_fold[(_fold, _norm)] = o["ranking"]
        if _norm == "alpha":
            band_refs[(_fold, "alpha")] = o["ref"]
            band_refs_lg[(_fold, "alpha")] = o["ref_lg"]
    records.sort(key=lambda r: (r["norm_mode"], r["K"], MODELS_ORDER.index(r["model"]), r["fold"]))
    cv_results = pd.DataFrame(records)
    pf = os.path.join(outdir, f"{OUT_PREFIX}{tag}per_fold.csv")
    cv_results.to_csv(pf, index=False, sep=";")
    # One result row per frequency group x algorithm x norm.
    summary = (cv_results.groupby(["norm_mode", "K", "model"])
               .agg(mean_acc=("acc", "mean"), std_acc=("acc", "std"),
                    mean_f1=("f1", "mean"), std_f1=("f1", "std"),
                    mean_EG=("acc_EG", "mean"), mean_HJ=("acc_HJ", "mean"),
                    n_folds=("fold", "nunique"))
               .reset_index().round(4)
               .sort_values(["norm_mode", "K", "model"]).reset_index(drop=True))
    sp = os.path.join(outdir, f"{OUT_PREFIX}{tag}summary.csv")
    summary.to_csv(sp, index=False, sep=";")
    print(summary.to_string(index=False), flush=True)
    _tk = topK_by_fold
    _rk = rankings_by_fold
    _stab, _univ = aggregate_stability(cv_results, _tk, _rk, list(FREQS_ALL),
                                       ["baseline", "alpha"], outdir, tag)
    save_result_plots(cv_results, _stab, list(FREQS_ALL), outdir, tag, OUT_PREFIX,
                      ["baseline", "alpha"])
    save_confusion_pdfs(cv_results, band_refs, band_refs_lg, df_pivot_full, list(LABELS),
                        list(FREQS_ALL), outdir, tag, OUT_PREFIX, seed, ["baseline", "alpha"])
    t_all = time.time() - t_all
    tt = cv_results.groupby("model")["train_time_s"].mean().round(3).to_dict()
    meta = {"seed": seed, "workers": workers, "smoke": smoke,
            "K": [int(k) for k in K_list], "norms": [str(n) for n in all_norms],
            "labels": [str(x) for x in LABELS], "window_s": float(WINDOW_S),
            "flags": flags, "rows": int(cv_results.shape[0]),
            "mean_train_time_s_per_model": {str(k): float(v) for k, v in tt.items()},
            "selection_time_s_total": float(sum(o["sel_time_s"] for o in outs)),
            "wall_time_s_total": float(t_all),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__,
                         "sklearn": __import__("sklearn").__version__,
                         "scipy": __import__("scipy").__version__}}
    meta["options"] = ["perfold"]
    meta["selection"] = ("unified systematic FI V20: bootstrap-10 RF-sqrt consensus, "
                         "HG-only, train-only")
    mp = os.path.join(outdir, f"{OUT_PREFIX}{tag}run_meta.json")
    json.dump(meta, open(mp, "w", encoding="utf-8"), indent=2, sort_keys=True)

SHARED_PREFIX = "nested_shared_"


def _derive_baseline_emergent(fold, df_outer_tr, seed, K_list):
    """One outer fold's shared bands: nested inner-LOO on BASELINE outer-train only."""
    _out, _audit = nested_emergent_sets(df_outer_tr, seed, list(K_list),
                                        list(LABELS), list(FREQS_ALL))
    return (int(fold),
            {int(k): [int(f) for f in v] for k, v in _out.items()},
            {int(k): v for k, v in _audit.items()})


def main_nested_shared():
    """Shared-set mode: baseline-derived emergent bands evaluated identically
    on both arms (same selected_freqs per fold/K; alpha uses outer-train refs)."""
    outdir, workers, smoke, seed = OUTDIR, WORKERS, SMOKE, SEED
    os.makedirs(outdir, exist_ok=True)
    K_list = list(K_LIST) if not smoke else [10, 3]
    flags = {"scaling": bool(APPLY_SCALING), "sg": bool(APPLY_SG),
             "sg_w": int(SG_W), "sg_p": int(SG_P),
             "pre_sg": bool(APPLY_PRE_SG), "pre_sg_w": int(PRE_SG_W), "pre_sg_p": int(PRE_SG_P),
             "pca": bool(APPLY_PCA), "lda": bool(APPLY_LDA),
             "qda": bool(APPLY_QDA), "ica": bool(APPLY_ICA)}
    print(f"nested-shared seed={seed} workers={workers} smoke={smoke} outdir={outdir}",
          flush=True)
    set_seed(seed)
    t_all = time.time()
    nb_dir = os.path.join(REPO, "src", "nb")
    df_pivot_full = build_pivot(nb_dir, float(WINDOW_S), outdir,
                                bool(APPLY_PRE_SG), int(PRE_SG_W), int(PRE_SG_P))
    print(f"pivot: {df_pivot_full.shape}, "
          f"days={sorted(df_pivot_full['Day'].unique())}", flush=True)
    assert set(df_pivot_full["Day"].unique()) == {1, 2, 3, 4, 5}
    splits = list(GroupKFold(n_splits=5).split(
        df_pivot_full, groups=df_pivot_full["Day"].values))
    folds_wanted = [0] if smoke else [0, 1, 2, 3, 4]
    outer = {}
    for fold in folds_wanted:
        tri, tei = splits[fold]
        dtr = df_pivot_full.iloc[tri].reset_index(drop=True)
        dte = df_pivot_full.iloc[tei].reset_index(drop=True)
        h = sorted(dte["Day"].unique().tolist())
        assert len(h) == 1, f"fold {fold} mixes days: {h}"
        assert int(h[0]) not in sorted(dtr["Day"].unique().tolist())
        outer[fold] = (int(h[0]), dtr, dte)
    # Phase 1: baseline-only nested derive (outer-train days only).
    dw = max(1, min(workers, len(folds_wanted)))
    t_derive = time.time()
    derived = Parallel(n_jobs=dw, backend="loky")(
        delayed(_derive_baseline_emergent)(f, outer[f][1], seed, list(K_list))
        for f in folds_wanted)
    derive_s = float(time.time() - t_derive)
    emerg = {f: e for f, e, _ in derived}
    audits = {f: a for f, _, a in derived}
    for f in folds_wanted:
        for K in K_list:
            assert len(emerg[f][int(K)]) == int(K)
    # Phase 2: outer evaluation with the SAME sets on both arms.
    tasks = []
    for f in folds_wanted:
        held, dtr, dte = outer[f]
        tasks.append((f, held, "baseline", dtr, dte, emerg[f]))
        tasks.append((f, held, "alpha", dtr, dte, emerg[f]))
    ew = max(1, min(workers, len(tasks)))
    print(f"eval_tasks={len(tasks)} workers={ew}", flush=True)
    outs = Parallel(n_jobs=ew, backend="loky")(
        delayed(run_unit)(f, h, n, tr, te, seed, list(K_list),
                          list(LABELS), list(FREQS_ALL), flags,
                          fixed_topK=fx, option="shared-base")
        for f, h, n, tr, te, fx in tasks)
    records = []
    for (_f, _h, _n, _tr, _te, _fx), o in zip(tasks, outs):
        assert o["sel_time_s"] == 0.0
        records.extend(o["records"])
    records.sort(key=lambda r: (r["norm_mode"], r["K"], MODELS_ORDER.index(r["model"]), r["fold"]))
    cv_results = pd.DataFrame(records)
    # Sharedness: both arms must carry identical band sets per (fold, K).
    _chk = cv_results.groupby(["fold", "K"])["selected_freqs"].nunique()
    assert int((_chk == 1).sum()) == len(_chk), "arms diverged in bands!"
    cv_results.to_csv(os.path.join(outdir, f"{SHARED_PREFIX}per_fold.csv"),
                      index=False, sep=";")
    summary = (cv_results.groupby(["norm_mode", "K", "model"])
               .agg(mean_acc=("acc", "mean"), std_acc=("acc", "std"),
                    mean_f1=("f1", "mean"), std_f1=("f1", "std"),
                    mean_EG=("acc_EG", "mean"), mean_HJ=("acc_HJ", "mean"),
                    n_folds=("fold", "nunique"))
               .reset_index().round(4)
               .sort_values(["norm_mode", "K", "model"]).reset_index(drop=True))
    summary.to_csv(os.path.join(outdir, f"{SHARED_PREFIX}summary.csv"), index=False, sep=";")
    print(summary.to_string(index=False), flush=True)
    sets_doc = {"mode": "nested-shared",
                "rule": ("per outer fold: nested inner-LOO over the 4 outer-train days on "
                         "BASELINE frames only (V20 voices); the identical emergent sets are "
                         "evaluated on both arms; alpha refs from outer-train rows only"),
                "folds": {str(f): {"held_out_day": outer[f][0],
                                   "emergent": {str(k): v for k, v in emerg[f].items()},
                                   "audit": {str(k): v for k, v in audits[f].items()}}
                          for f in folds_wanted}}
    json.dump(sets_doc, open(os.path.join(outdir, f"{SHARED_PREFIX}sets.json"),
                             "w", encoding="utf-8"), indent=2, sort_keys=True)
    t_all = time.time() - t_all
    tt = cv_results.groupby("model")["train_time_s"].mean().round(3).to_dict()
    meta = {"mode": "nested-shared", "seed": seed, "workers": workers, "smoke": smoke,
            "K": [int(k) for k in K_list], "norms": ["baseline", "alpha"],
            "labels": [str(x) for x in LABELS], "window_s": float(WINDOW_S),
            "flags": flags, "rows": int(cv_results.shape[0]),
            "option": "shared-base",
            "selection": ("shared baseline-derived emergent sets (V20 voices, nested inner-LOO "
                          "on baseline outer-train only), evaluated identically on both arms"),
            "mean_train_time_s_per_model": {str(k): float(v) for k, v in tt.items()},
            "derive_time_s_total": derive_s,
            "wall_time_s_total": float(t_all),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__,
                         "sklearn": __import__("sklearn").__version__,
                         "scipy": __import__("scipy").__version__}}
    json.dump(meta, open(os.path.join(outdir, f"{SHARED_PREFIX}run_meta.json"),
                         "w", encoding="utf-8"), indent=2, sort_keys=True)
    print(f"rows={len(cv_results)} NaNs={int(cv_results.isna().sum().sum())} "
          f"wall_s={t_all:.1f} derive_s={derive_s:.1f}", flush=True)


if __name__ == "__main__":
    if MODE == "nested-shared":
        main_nested_shared()
    elif MODE == "perfold":
        main()
    else:
        raise ValueError(f"unknown MODE: {MODE}")
