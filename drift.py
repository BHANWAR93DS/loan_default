"""
drift.py — Data Drift Detection
==================================
Calculates 3 types of drift between
training (reference) and test (current) data:

  PSI  — Population Stability Index
  CSI  — Characteristic Stability Index (per feature)
  KS   — Kolmogorov-Smirnov test

Logs all drift metrics + plots to MLflow.

PSI thresholds:
  < 0.10  → No significant drift
  0.10–0.25 → Moderate drift (monitor)
  > 0.25  → Significant drift (retrain)
"""

import os
import warnings
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
warnings.filterwarnings('ignore')

from scipy import stats
import mlflow

# ── Config ────────────────────────────────────────────────────────────────────
MLFLOW_URI = 'sqlite:///mlflow.db'
EXPERIMENT = 'Loan_Default_Prediction'
N_BINS     = 10

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)


# ── PSI ───────────────────────────────────────────────────────────────────────
def calculate_psi(expected, actual, n_bins=N_BINS):
    """
    Population Stability Index.
    Compares overall distribution shift between
    reference (train) and current (test).
    """
    # Create bins from expected distribution
    breakpoints = np.linspace(0, 100, n_bins + 1)
    expected_perc = np.percentile(expected, breakpoints)
    expected_perc = np.unique(expected_perc)

    expected_counts = np.histogram(expected, bins=expected_perc)[0]
    actual_counts   = np.histogram(actual,   bins=expected_perc)[0]

    # Normalize to percentages
    expected_pct = expected_counts / len(expected)
    actual_pct   = actual_counts   / len(actual)

    # Replace zeros to avoid log(0)
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct   = np.where(actual_pct   == 0, 1e-6, actual_pct)

    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return float(np.sum(psi_values))


# ── CSI ───────────────────────────────────────────────────────────────────────
def calculate_csi(expected, actual, n_bins=N_BINS):
    """
    Characteristic Stability Index.
    PSI calculated per individual feature.
    """
    return calculate_psi(expected, actual, n_bins)


# ── KS Test ───────────────────────────────────────────────────────────────────
def calculate_ks(expected, actual):
    """
    Kolmogorov-Smirnov test.
    Returns KS statistic and p-value.
    """
    ks_stat, p_value = stats.ks_2samp(expected, actual)
    return float(ks_stat), float(p_value)


# ── Drift label ───────────────────────────────────────────────────────────────
def psi_label(psi):
    if psi < 0.10:
        return '✅ No drift'
    elif psi < 0.25:
        return '⚠️  Moderate drift'
    else:
        return '🚨 Significant drift'


def run_drift():
    print("\n📊 Running Data Drift Detection...")

    # Load data
    print("   Loading reference (train) and current (test) data...")
    feature_columns = joblib.load('data/feature_columns.pkl')
    X_train         = np.load('data/X_train_sample.npy', allow_pickle=True)
    X_test          = np.load('data/X_test.npy',         allow_pickle=True)

    print(f"   Reference samples : {X_train.shape[0]}")
    print(f"   Current samples   : {X_test.shape[0]}")
    print(f"   Features          : {len(feature_columns)}")

    # ── Overall PSI (on first principal component as proxy) ───────────────────
    overall_psi = calculate_psi(X_train[:, 0], X_test[:, 0])

    with mlflow.start_run(run_name='Data_Drift_Analysis'):

        mlflow.log_param('reference_size',  X_train.shape[0])
        mlflow.log_param('current_size',    X_test.shape[0])
        mlflow.log_param('n_features',      len(feature_columns))
        mlflow.log_param('n_bins',          N_BINS)
        mlflow.log_param('drift_method',    'PSI + CSI + KS')

        mlflow.log_metric('overall_psi', round(overall_psi, 6))
        print(f"\n   Overall PSI : {overall_psi:.4f}  {psi_label(overall_psi)}")

        # ── Per-feature drift ─────────────────────────────────────────────────
        csi_values = {}
        ks_stats   = {}
        ks_pvalues = {}

        print("\n   Per-feature drift:")
        print(f"   {'Feature':<35} {'CSI':>8}  {'KS stat':>8}  {'p-value':>10}  Status")
        print("   " + "-" * 80)

        for i, feat in enumerate(feature_columns):
            ref = X_train[:, i]
            cur = X_test[:,  i]

            csi          = calculate_csi(ref, cur)
            ks_stat, p_v = calculate_ks(ref, cur)

            csi_values[feat] = csi
            ks_stats[feat]   = ks_stat
            ks_pvalues[feat] = p_v

            # Log to MLflow
            mlflow.log_metric(f'csi_{feat}',     round(csi,     6))
            mlflow.log_metric(f'ks_stat_{feat}',  round(ks_stat, 6))
            mlflow.log_metric(f'ks_pvalue_{feat}', round(p_v,    6))

            status = psi_label(csi)
            print(f"   {feat:<35} {csi:>8.4f}  {ks_stat:>8.4f}  {p_v:>10.4f}  {status}")

        # ── Summary stats ─────────────────────────────────────────────────────
        avg_csi    = np.mean(list(csi_values.values()))
        max_csi    = np.max(list(csi_values.values()))
        drifted_ft = sum(1 for v in csi_values.values() if v > 0.10)

        mlflow.log_metric('avg_csi',           round(avg_csi, 6))
        mlflow.log_metric('max_csi',           round(max_csi, 6))
        mlflow.log_metric('drifted_features',  drifted_ft)

        print(f"\n   Avg CSI            : {avg_csi:.4f}")
        print(f"   Max CSI            : {max_csi:.4f}")
        print(f"   Drifted features   : {drifted_ft} / {len(feature_columns)}")

        # ── Plot 1: CSI bar chart ─────────────────────────────────────────────
        feats  = list(csi_values.keys())
        values = list(csi_values.values())
        colors = ['#d32f2f' if v > 0.25
                  else '#f57c00' if v > 0.10
                  else '#388e3c'
                  for v in values]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(feats, values, color=colors, edgecolor='none', height=0.6)
        ax.axvline(x=0.10, color='orange', linestyle='--',
                   linewidth=1.5, label='Moderate drift (0.10)')
        ax.axvline(x=0.25, color='red',    linestyle='--',
                   linewidth=1.5, label='Significant drift (0.25)')
        ax.set_xlabel('CSI Value', fontsize=12)
        ax.set_title('Characteristic Stability Index (CSI) per Feature', fontsize=13)
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig('drift_csi.png', dpi=120, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact('drift_csi.png')
        os.remove('drift_csi.png')
        print("   ✅ drift_csi.png → MLflow")

        # ── Plot 2: KS statistics bar chart ───────────────────────────────────
        ks_vals = [ks_stats[f] for f in feats]
        ks_cols = ['#d32f2f' if v > 0.15 else '#388e3c' for v in ks_vals]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(feats, ks_vals, color=ks_cols, edgecolor='none', height=0.6)
        ax.axvline(x=0.10, color='orange', linestyle='--',
                   linewidth=1.5, label='KS = 0.10')
        ax.axvline(x=0.15, color='red',    linestyle='--',
                   linewidth=1.5, label='KS = 0.15')
        ax.set_xlabel('KS Statistic', fontsize=12)
        ax.set_title('Kolmogorov-Smirnov Statistic per Feature', fontsize=13)
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig('drift_ks.png', dpi=120, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact('drift_ks.png')
        os.remove('drift_ks.png')
        print("   ✅ drift_ks.png → MLflow")

        # ── Plot 3: Distribution comparison for top drifted feature ───────────
        top_drift_feat = max(csi_values, key=csi_values.get)
        feat_idx       = list(feature_columns).index(top_drift_feat)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(X_train[:, feat_idx], bins=30, alpha=0.6,
                label='Reference (train)', color='steelblue', density=True)
        ax.hist(X_test[:,  feat_idx], bins=30, alpha=0.6,
                label='Current (test)',    color='tomato',    density=True)
        ax.set_title(
            f'Distribution Shift: {top_drift_feat}\n'
            f'CSI={csi_values[top_drift_feat]:.4f}  '
            f'KS={ks_stats[top_drift_feat]:.4f}  '
            f'p={ks_pvalues[top_drift_feat]:.4f}'
        )
        ax.set_xlabel(top_drift_feat); ax.set_ylabel('Density')
        ax.legend()
        plt.tight_layout()
        plt.savefig('drift_distribution.png', dpi=120, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact('drift_distribution.png')
        os.remove('drift_distribution.png')
        print(f"   ✅ drift_distribution.png → MLflow  ({top_drift_feat})")

    print("\n✅ drift.py complete!")
    print("   Open MLflow UI → run 'Data_Drift_Analysis' to see results.")
    print("   Run: mlflow ui --port 5001")


if __name__ == '__main__':
    run_drift()