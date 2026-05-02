"""
app.py — Loan Default Prediction API
=======================================
Flask REST API matching the real dataset columns.

Endpoints:
  GET  /          → API info
  GET  /health    → Health check
  GET  /features  → Expected input features
  GET  /sample    → Sample request body
  POST /predict   → Predict default probability
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

warnings.filterwarnings('ignore')

app      = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ── Load artifacts ────────────────────────────────────────────────────────────
print("=" * 50)
print("  Loan Default Prediction API")
print("=" * 50)
print("Loading model artifacts...")

try:
    model           = joblib.load(os.path.join(DATA_DIR, 'model.pkl'))
    scaler          = joblib.load(os.path.join(DATA_DIR, 'scaler.pkl'))
    feature_columns = joblib.load(os.path.join(DATA_DIR, 'feature_columns.pkl'))
    le_dict         = joblib.load(os.path.join(DATA_DIR, 'label_encoders.pkl'))
    imputer         = joblib.load(os.path.join(DATA_DIR, 'imputer.pkl'))
    print(f"✅ Model    : {type(model).__name__}")
    print(f"✅ Features : {len(feature_columns)}")
    print("=" * 50)
except FileNotFoundError as e:
    print(f"❌ Missing artifact: {e}")
    print("   Run 'python model.py' first!")
    raise


# ── Feature engineering (must match model.py) ────────────────────────────────
def add_engineered_features(df):
    df['age_years']        = df.get('age_days',        pd.Series([0])).abs() / 365
    df['employed_years']   = df.get('employed_days',   pd.Series([0])).abs() / 365
    df['loan_to_income']   = (df.get('credit_amount',  pd.Series([0]))
                              / (df.get('client_income', pd.Series([1])).abs() + 1))
    df['annuity_to_income']= (df.get('loan_annuity',   pd.Series([0]))
                              / (df.get('client_income', pd.Series([1])).abs() + 1))
    df['bureau_per_year']  = (df.get('credit_bureau',  pd.Series([0]))
                              / (df['age_years'] + 1))
    df['social_risk']      = (df.get('social_circle_default', pd.Series([0]))
                              / (df.get('client_family_members', pd.Series([1])) + 1))
    return df


def preprocess_input(data: dict) -> np.ndarray:
    df = pd.DataFrame([data])

    # Convert numeric fields that might come as strings
    num_fields = ['client_income','credit_amount','loan_annuity',
                  'population_region_relative','age_days','employed_days',
                  'registration_days','id_days','score_source_3']
    for col in num_fields:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Add engineered features
    df = add_engineered_features(df)

    # Encode categoricals
    for col, le in le_dict.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            known   = list(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else known[0])
            df[col] = le.transform(df[col])

    # Add missing columns with 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns].fillna(0)

    # Impute + scale
    X_imp    = imputer.transform(df)
    X_scaled = scaler.transform(X_imp)
    return X_scaled


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service'    : 'Loan Default Prediction API',
        'version'    : '1.0',
        'model'      : type(model).__name__,
        'description': 'Predicts probability of a client defaulting on a loan.',
        'endpoints'  : {
            'POST /predict' : 'Predict loan default probability',
            'GET  /health'  : 'Health check',
            'GET  /features': 'List expected input features',
            'GET  /sample'  : 'Sample request body',
        }
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status'  : 'ok',
        'model'   : type(model).__name__,
        'features': len(feature_columns),
    })


@app.route('/features', methods=['GET'])
def features():
    return jsonify({
        'total'   : len(feature_columns),
        'features': list(feature_columns)
    })


@app.route('/sample', methods=['GET'])
def sample():
    return jsonify({
        'description': 'POST this JSON body to /predict',
        'sample_request': {
            'client_income'             : 180000,
            'car_owned'                 : 1,
            'bike_owned'                : 0,
            'active_loan'               : 0,
            'house_own'                 : 1,
            'child_count'               : 1,
            'credit_amount'             : 450000,
            'loan_annuity'              : 20250,
            'accompany_client'          : 'Unaccompanied',
            'client_income_type'        : 'Service',
            'client_education'          : 'Secondary',
            'client_marital_status'     : 'Married',
            'client_gender'             : 'Male',
            'loan_contract_type'        : 'CL',
            'client_housing_type'       : 'House',
            'population_region_relative': 0.035,
            'age_days'                  : -12005,
            'employed_days'             : -2500,
            'registration_days'         : -4000,
            'id_days'                   : -1500,
            'own_house_age'             : 15,
            'mobile_tag'                : 1,
            'homephone_tag'             : 0,
            'workphone_working'         : 1,
            'client_occupation'         : 'Laborers',
            'client_family_members'     : 2,
            'cleint_city_rating'        : 2,
            'application_process_day'   : 3,
            'application_process_hour'  : 10,
            'client_permanent_match_tag': 'Yes',
            'client_contact_work_tag'   : 'Yes',
            'type_organization'         : 'Business Entity Type 3',
            'score_source_1'            : 0.5,
            'score_source_2'            : 0.6,
            'score_source_3'            : 0.5,
            'social_circle_default'     : 0,
            'phone_change'              : 1000,
            'credit_bureau'             : 1,
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No JSON body provided'}), 400

        X    = preprocess_input(data)
        prob = float(model.predict_proba(X)[0][1])
        pred = int(prob >= 0.5)

        if prob < 0.30:
            risk  = 'Low'
            color = 'green'
        elif prob < 0.60:
            risk  = 'Medium'
            color = 'orange'
        else:
            risk  = 'High'
            color = 'red'

        return jsonify({
            'default_probability': round(prob, 4),
            'prediction'         : pred,
            'risk_level'         : risk,
            'risk_color'         : color,
            'message'            : ('Default likely — high risk client'
                                    if pred == 1
                                    else 'No default expected — low risk client'),
        })

    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting Flask server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)