"""
model.py — Loan Default Prediction
=====================================
Dataset: 121,856 rows × 40 columns
Target : default (0/1) — 8% default rate

Steps:
1. Load & clean column names
2. Feature engineering (6 new features)
3. Handle missing values  (SimpleImputer)
4. Handle outliers        (IQR clipping)
5. Handle class imbalance (SMOTE)
6. Experiment 1 : Random Forest      → MLflow
7. Experiment 2 : Gradient Boosting  → MLflow
8. Save ALL artifacts to data/
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
warnings.filterwarnings('ignore')

from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics         import (accuracy_score, roc_auc_score, f1_score,
                                      precision_score, recall_score,
                                      confusion_matrix, classification_report,
                                      roc_curve)
from sklearn.impute          import SimpleImputer
from imblearn.over_sampling  import SMOTE
import mlflow
import mlflow.sklearn

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join('data', 'Dataset.csv')
MLFLOW_URI = 'sqlite:///mlflow.db'
EXPERIMENT = 'Loan_Default_Prediction'

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)


# ── 1. Load & Clean ───────────────────────────────────────────────────────────
def load_data(path):
    print("\n📂 Loading dataset...")
    df = pd.read_csv(path, low_memory=False)

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # Drop ID column — not a feature
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Convert object columns that should be numeric
    num_cols_to_fix = [
        'client_income', 'credit_amount', 'loan_annuity',
        'population_region_relative', 'age_days', 'employed_days',
        'registration_days', 'id_days', 'score_source_3'
    ]
    for col in num_cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"   Shape        : {df.shape}")
    print(f"   Default rate : {df['default'].mean():.2%}")
    print(f"   Missing vals : {df.isnull().sum().sum()}")
    return df


# ── 2. Feature Engineering ────────────────────────────────────────────────────
def feature_engineering(df):
    print("\n🔧 Feature engineering...")
    df = df.copy()

    # Age in years (age_days is negative in some datasets)
    df['age_years'] = df['age_days'].abs() / 365

    # Employment length in years
    df['employed_years'] = df['employed_days'].abs() / 365

    # Loan to income ratio
    df['loan_to_income'] = df['credit_amount'] / (df['client_income'].abs() + 1)

    # Annuity to income ratio (monthly burden)
    df['annuity_to_income'] = df['loan_annuity'] / (df['client_income'].abs() + 1)

    # Credit bureau inquiries per year
    df['bureau_per_year'] = df['credit_bureau'] / (df['age_years'] + 1)

    # Social circle default risk (normalized)
    df['social_risk'] = df['social_circle_default'] / (df['client_family_members'] + 1)

    print("   ✅ 6 new features: age_years, employed_years, loan_to_income,")
    print("                      annuity_to_income, bureau_per_year, social_risk")
    return df


# ── 3. Preprocess ─────────────────────────────────────────────────────────────
def preprocess(df):
    print("\n⚙️  Preprocessing...")

    X = df.drop('default', axis=1)
    y = df['default']

    # ── Encode categoricals
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    le_dict  = {}
    for col in cat_cols:
        le           = LabelEncoder()
        X[col]       = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    print(f"   Encoded {len(cat_cols)} categorical columns")

    feature_columns = X.columns.tolist()

    # ── Impute missing values with median
    imputer = SimpleImputer(strategy='median')
    X_imp   = imputer.fit_transform(X)
    X       = pd.DataFrame(X_imp, columns=feature_columns)
    remaining_missing = X.isnull().sum().sum()
    print(f"   Imputed missing values → remaining: {remaining_missing}")

    # ── Clip outliers (1st–99th percentile)
    for col in X.select_dtypes(include=np.number).columns:
        q1, q3 = X[col].quantile(0.01), X[col].quantile(0.99)
        X[col]  = X[col].clip(q1, q3)
    print("   Clipped outliers (1st–99th percentile)")

    # ── Scale
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── SMOTE — handle 8% imbalance
    print(f"   Class balance before SMOTE — 0: {(y==0).sum()}  1: {(y==1).sum()}")
    sm           = SMOTE(random_state=42, sampling_strategy=0.3)
    X_res, y_res = sm.fit_resample(X_scaled, y)
    print(f"   Class balance after  SMOTE — 0: {(y_res==0).sum()}  1: {(y_res==1).sum()}")

    return X_res, y_res, scaler, le_dict, feature_columns, imputer


# ── 4. MLflow Experiment ──────────────────────────────────────────────────────
def run_experiment(X_train, X_test, y_train, y_test,
                   model, params, run_name):
    print(f"\n🔬 Running: {run_name}")

    with mlflow.start_run(run_name=run_name):

        # Log hyperparameters
        mlflow.log_params(params)

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        acc  = accuracy_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_prob)
        f1   = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring='roc_auc'
        )

        # Log all metrics
        mlflow.log_metrics({
            'accuracy'        : round(acc,  4),
            'roc_auc'         : round(auc,  4),
            'f1_score'        : round(f1,   4),
            'precision'       : round(prec, 4),
            'recall'          : round(rec,  4),
            'cv_auc_mean'     : round(float(cv_scores.mean()), 4),
            'cv_auc_std'      : round(float(cv_scores.std()),  4),
            'true_positives'  : int(tp),
            'false_positives' : int(fp),
            'true_negatives'  : int(tn),
            'false_negatives' : int(fn),
        })

        # ── ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'AUC = {auc:.4f}')
        plt.plot([0,1],[0,1], 'k--', lw=1)
        plt.fill_between(fpr, tpr, alpha=0.08, color='darkorange')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve — {run_name}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        roc_path = f'roc_{run_name}.png'
        plt.savefig(roc_path, dpi=120, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(roc_path)
        os.remove(roc_path)

        # ── Confusion Matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        cm_arr  = np.array([[tn, fp], [fn, tp]])
        im      = ax.imshow(cm_arr, cmap='Blues')
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['No Default','Default'])
        ax.set_yticklabels(['No Default','Default'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix — {run_name}')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm_arr[i,j], ha='center', va='center',
                        fontsize=13, color='red', fontweight='bold')
        plt.tight_layout()
        cm_path = f'cm_{run_name}.png'
        plt.savefig(cm_path, dpi=120, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        # Log model
        mlflow.sklearn.log_model(model, name="model")

        # Print summary
        print(f"   Accuracy  : {acc:.4f}")
        print(f"   ROC-AUC   : {auc:.4f}")
        print(f"   F1 Score  : {f1:.4f}")
        print(f"   Precision : {prec:.4f}")
        print(f"   Recall    : {rec:.4f}")
        print(f"   CV AUC    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(classification_report(y_test, y_pred,
              target_names=['No Default','Default'], zero_division=0))

    return model, auc


# ── 5. Main ───────────────────────────────────────────────────────────────────
def main():

    # Load
    df = load_data(DATA_PATH)

    # Feature engineering
    df = feature_engineering(df)

    # Preprocess
    X, y, scaler, le_dict, feature_columns, imputer = preprocess(df)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n   Train: {X_train.shape}  Test: {X_test.shape}")

    # ── Experiment 1: Random Forest ──────────────────────────────────────────
    params1 = {
        'model_type'        : 'RandomForest',
        'n_estimators'      : 100,
        'max_depth'         : 10,
        'min_samples_split' : 10,
        'min_samples_leaf'  : 4,
        'class_weight'      : 'balanced',
        'random_state'      : 42,
    }
    model1 = RandomForestClassifier(
        n_estimators      = params1['n_estimators'],
        max_depth         = params1['max_depth'],
        min_samples_split = params1['min_samples_split'],
        min_samples_leaf  = params1['min_samples_leaf'],
        class_weight      = params1['class_weight'],
        random_state      = params1['random_state'],
        n_jobs            = -1
    )
    model1, auc1 = run_experiment(
        X_train, X_test, y_train, y_test,
        model1, params1, 'Exp1_RandomForest'
    )

    # ── Experiment 2: Gradient Boosting ──────────────────────────────────────
    params2 = {
        'model_type'       : 'GradientBoosting',
        'n_estimators'     : 200,
        'max_depth'        : 5,
        'learning_rate'    : 0.05,
        'subsample'        : 0.8,
        'min_samples_split': 20,
        'random_state'     : 42,
    }
    model2 = GradientBoostingClassifier(
        n_estimators      = params2['n_estimators'],
        max_depth         = params2['max_depth'],
        learning_rate     = params2['learning_rate'],
        subsample         = params2['subsample'],
        min_samples_split = params2['min_samples_split'],
        random_state      = params2['random_state'],
    )
    model2, auc2 = run_experiment(
        X_train, X_test, y_train, y_test,
        model2, params2, 'Exp2_GradientBoosting'
    )

    # ── Best model ────────────────────────────────────────────────────────────
    if auc1 >= auc2:
        best_model, best_name = model1, 'RandomForest'
    else:
        best_model, best_name = model2, 'GradientBoosting'
    print(f"\n🏆 Best model : {best_name}  (AUC = {max(auc1,auc2):.4f})")

    # ── Save ALL artifacts ────────────────────────────────────────────────────
    print("\n💾 Saving artifacts to data/...")
    os.makedirs('data', exist_ok=True)

    joblib.dump(best_model,      'data/model.pkl')
    joblib.dump(scaler,          'data/scaler.pkl')
    joblib.dump(feature_columns, 'data/feature_columns.pkl')
    joblib.dump(le_dict,         'data/label_encoders.pkl')
    joblib.dump(imputer,         'data/imputer.pkl')

    # Save arrays for explain.py and drift.py
    np.save('data/X_train_sample.npy', X_train[:500])
    np.save('data/X_test.npy',         X_test)
    np.save('data/y_test.npy',         y_test)

    print("   ✅ data/model.pkl")
    print("   ✅ data/scaler.pkl")
    print("   ✅ data/feature_columns.pkl")
    print("   ✅ data/label_encoders.pkl")
    print("   ✅ data/imputer.pkl")
    print("   ✅ data/X_train_sample.npy")
    print("   ✅ data/X_test.npy")
    print("   ✅ data/y_test.npy")
    print("\n✅ model.py complete! Run explain.py next.")


if __name__ == '__main__':
    main()