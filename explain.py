"""
explain.py — XAI using SHAP
==============================
1. Loads trained model + test data
2. Computes SHAP values (TreeExplainer)
3. Generates 3 plots:
   - Summary plot  (beeswarm)
   - Bar plot      (mean |SHAP|)
   - Waterfall     (highest risk sample)
4. Logs all plots + metrics to MLflow
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

import shap
import mlflow

# ── Config ────────────────────────────────────────────────────────────────────
MLFLOW_URI = 'sqlite:///mlflow.db'
EXPERIMENT = 'Loan_Default_Prediction'

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)


def get_shap_values(explainer, X_sample, model):
    """
    Safely extract SHAP values and expected value
    for both RandomForest and GradientBoosting.
    """
    shap_values  = explainer.shap_values(X_sample)
    model_name   = type(model).__name__

    # RandomForest → shap_values is list [class0, class1]
    if isinstance(shap_values, list) and len(shap_values) == 2:
        sv           = shap_values[1]
        raw_ev       = explainer.expected_value
        expected_val = float(raw_ev[1]) if hasattr(raw_ev, '__len__') else float(raw_ev)

    # GradientBoosting → shap_values is single array
    else:
        sv           = shap_values if not isinstance(shap_values, list) else shap_values[0]
        raw_ev       = explainer.expected_value
        expected_val = float(raw_ev[0]) if hasattr(raw_ev, '__len__') else float(raw_ev)

    print(f"   Model type   : {model_name}")
    print(f"   SHAP shape   : {sv.shape}")
    print(f"   Expected val : {expected_val:.4f}")
    return sv, expected_val


def run_shap():
    print("\n🔍 Running SHAP Explainability...")

    # Load artifacts
    print("   Loading model + test data...")
    model           = joblib.load('data/model.pkl')
    feature_columns = joblib.load('data/feature_columns.pkl')
    X_test          = np.load('data/X_test.npy', allow_pickle=True)

    # Use a sample for speed
    sample_size = min(300, len(X_test))
    X_sample    = X_test[:sample_size]
    print(f"   Using {sample_size} samples for SHAP")

    # Clean feature names for display
    feature_names = [f.replace('_', ' ').title() for f in feature_columns]

    # SHAP TreeExplainer
    print("   Computing SHAP values (this may take a minute)...")
    explainer        = shap.TreeExplainer(model)
    sv, expected_val = get_shap_values(explainer, X_sample, model)

    print("   ✅ SHAP values computed")

    with mlflow.start_run(run_name='SHAP_Explainability'):

        mlflow.log_param('model_type',   type(model).__name__)
        mlflow.log_param('shap_samples', sample_size)
        mlflow.log_param('explainer',    'TreeExplainer')

        # ── Plot 1: Summary beeswarm ──────────────────────────────────────────
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_sample,
                          feature_names=feature_names,
                          show=False, max_display=15)
        plt.title('SHAP Summary — Feature Impact on Loan Default', pad=12)
        plt.tight_layout()
        plt.savefig('shap_summary.png', dpi=120, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact('shap_summary.png')
        print("   ✅ shap_summary.png → MLflow")

        # ── Plot 2: Bar chart (mean |SHAP|) ───────────────────────────────────
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_sample,
                          feature_names=feature_names,
                          plot_type='bar', show=False, max_display=15)
        plt.title('SHAP Feature Importance (Mean |SHAP value|)', pad=12)
        plt.tight_layout()
        plt.savefig('shap_importance.png', dpi=120, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact('shap_importance.png')
        print("   ✅ shap_importance.png → MLflow")

        # ── Plot 3: Waterfall for highest-risk sample ─────────────────────────
        probs         = model.predict_proba(X_sample)[:, 1]
        high_risk_idx = int(np.argmax(probs))

        plt.figure(figsize=(10, 7))
        shap.waterfall_plot(
            shap.Explanation(
                values       = sv[high_risk_idx],
                base_values  = expected_val,
                data         = X_sample[high_risk_idx],
                feature_names= feature_names
            ),
            show=False, max_display=15
        )
        plt.title(
            f'SHAP Waterfall — Highest Risk Sample '
            f'(default prob = {probs[high_risk_idx]:.3f})',
            pad=12
        )
        plt.tight_layout()
        plt.savefig('shap_waterfall.png', dpi=120, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact('shap_waterfall.png')
        print("   ✅ shap_waterfall.png → MLflow")

        # ── Log top SHAP values as metrics ────────────────────────────────────
        mean_abs_shap = np.abs(sv).mean(axis=0)
        top_features  = sorted(
            zip(feature_columns, mean_abs_shap),
            key=lambda x: x[1], reverse=True
        )

        print("\n   Top 10 features by SHAP importance:")
        for i, (feat, val) in enumerate(top_features[:10]):
            mlflow.log_metric(f'shap_{feat}', round(float(val), 6))
            print(f"   {i+1:2}. {feat:35s}: {val:.4f}")

        mlflow.log_metric('shap_highest_risk_prob',
                          round(float(probs[high_risk_idx]), 4))

    print("\n✅ explain.py complete!")
    print("   Run drift.py next:")
    print("   python drift.py")


if __name__ == '__main__':
    run_shap()