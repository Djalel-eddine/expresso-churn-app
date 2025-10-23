# === app.py ===
"""
Streamlit app to predict churn using saved model and preprocessor.

Usage:
    streamlit run app.py
The app auto-generates inputs from models/feature_metadata.json
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ---------- paths ----------
BASE_DIR = Path(r"C:\Users\pc\Desktop\GMC DS\Cours\Streamlit")
MODELS_DIR = BASE_DIR.joinpath('models')
from pathlib import Path

MODEL_PATH = MODELS_DIR / "lgbm_churn_model.joblib"
PREPROC_PATH = MODELS_DIR / "preprocessor.joblib"
META_PATH = MODELS_DIR / "feature_metadata.json"


# ---------- helpers ----------
def load_artifacts():
    if not MODEL_PATH.exists() or not PREPROC_PATH.exists() or not META_PATH.exists():
        st.error("Model, preprocessor, or metadata not found. Run train_model.py first.")
        st.stop()
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROC_PATH)
    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return model, preprocessor, meta

def build_input_row(meta):
    """
    Auto-build a single-row DataFrame from user inputs using metadata.
    Returns a DataFrame with the same columns expected by the preprocessor.
    """
    numeric_features = meta.get('numeric_features', [])
    categorical_features = list(meta.get('categorical_features', {}).keys())

    user_vals = {}
    st.sidebar.title("Customer features (input)")
    st.sidebar.write("Modify these values and click Predict")

    # Numeric inputs
    st.sidebar.subheader("Numeric features")
    for col in numeric_features:
        # choose reasonable defaults; you may adjust min/max/step as you prefer
        val = st.sidebar.number_input(label=col, value=0.0, step=1.0, format="%.3f")
        user_vals[col] = val

    # Categorical inputs
    st.sidebar.subheader("Categorical features")
    for col in categorical_features:
        options = meta['categorical_features'].get(col, [])
        # Provide an 'OTHER' option to handle unseen categories gracefully
        options_with_other = options + ["OTHER"]
        choice = st.sidebar.selectbox(label=col, options=options_with_other, index=0)
        # if user chooses OTHER, we keep the literal string 'OTHER' (preprocessor handles unknowns)
        user_vals[col] = choice

    # Build dataframe row
    row = pd.DataFrame([user_vals], columns=(numeric_features + categorical_features))
    return row

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Expresso Churn Predictor", layout="wide")
st.title("Expresso — Churn prediction")

model, preprocessor, meta = load_artifacts()
st.success("Model and preprocessor loaded.")

# Build inputs
input_df = build_input_row(meta)

st.write("### Input preview")
st.dataframe(input_df)

# Predict button
if st.button("Predict churn probability"):
    try:
        X_proc = preprocessor.transform(input_df)  # numeric & categorical pipeline will run
        proba = model.predict_proba(X_proc)[:, 1][0]
        pred = int(proba >= 0.5)
        st.metric(label="Churn probability", value=f"{proba:.3f}")
        st.write("Predicted class:", "CHURN" if pred == 1 else "NO CHURN")
        # optional: show model confidence details or raw probability distribution
        st.write("Raw probability (NOT thresholded):", proba)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.write("---")
st.write("Note: this demo applies the same preprocessing used at training. If you update the training data/encoders, re-train and update files in models/.")
