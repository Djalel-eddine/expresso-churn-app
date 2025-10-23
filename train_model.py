# === train_model.py ===
"""
Train a churn model from the cleaned CSV produced by the data prep step.

Saves:
 - models/lgbm_churn_model.joblib         (LightGBM model)
 - models/preprocessor.joblib             (sklearn ColumnTransformer)
 - models/feature_metadata.json           (json with numeric/categorical info)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier
from pathlib import Path

# ---------- CONFIG ----------
BASE_DIR = Path(r"C:\Users\pc\Desktop\GMC DS\Cours\Streamlit")
CLEAN_PATH = BASE_DIR / "data" / "clean_expresso.csv"
MODELS_DIR = BASE_DIR / "models"
os.makedirs(MODELS_DIR, exist_ok=True)
CLEAN_CSV = BASE_DIR / 'expresso_churn_clean.csv'   # ✅ Make sure the file exists here

MODEL_PATH = MODELS_DIR.joinpath('lgbm_churn_model.joblib')
PREPROC_PATH = MODELS_DIR.joinpath('preprocessor.joblib')
META_PATH = MODELS_DIR.joinpath('feature_metadata.json')

RANDOM_STATE = 42
TEST_SIZE = 0.20
SAMPLE_FRAC = None   # set to 0.1 to train on a sample for speed (or None for full)
# ----------------------------

def load_data(path, sample_frac=None):
    df = pd.read_csv(path)
    if sample_frac is not None and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=RANDOM_STATE)
    return df

def build_preprocessor(numeric_features, categorical_features):
    """
    Create ColumnTransformer that imputes & scales numeric, imputes & one-hot encodes categorical.
    handle_unknown='ignore' ensures Streamlit users with unseen categories won't break prediction.
    """
    # Numeric pipeline: median imputation + standard scaling
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: most frequent imputation + one-hot encoding (ignore unknown categories)
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ], remainder='drop', sparse_threshold=0)

    return preprocessor

def main():
    print("Loading cleaned data from:", CLEAN_CSV)
    df = load_data(CLEAN_CSV, sample_frac=SAMPLE_FRAC)

    # ----- features / target -----
    assert 'CHURN' in df.columns, "CHURN column must exist in cleaned CSV"
    X = df.drop(columns=['CHURN'])
    y = df['CHURN'].astype(int)

    # Keep track of numeric and categorical columns we used in data_prep (object -> categorical)
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Numeric features ({len(numeric_features)}): {numeric_features[:8]}{'...' if len(numeric_features)>8 else ''}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Preprocessor
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    print("Fitting preprocessor (this will transform train set)...")
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    print("Preprocessed shapes:", X_train_trans.shape, X_test_trans.shape)

    # LightGBM model with reasonable defaults
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Use early stopping by passing eval_set to model.fit
    print("Training LightGBM (with early stopping)...")
    model.fit(
        X_train_trans, y_train,
        eval_set=[(X_test_trans, y_test)],
        eval_metric='auc',

    )

    # Predictions & evaluation
    proba = model.predict_proba(X_test_trans)[:, 1]
    preds = (proba >= 0.5).astype(int)
    print("ROC AUC:", roc_auc_score(y_test, proba))
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Classification report:\n", classification_report(y_test, preds))
    # confusion matrix printed for reference
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    # Save model + preprocessor + metadata
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROC_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved preprocessor to {PREPROC_PATH}")

    # Save metadata for Streamlit to auto-generate form
    # For categorical fields, extract categories seen during training from the preprocessor pipeline
    meta = {
        'numeric_features': numeric_features,
        'categorical_features': {}
    }

    # Extract categories from OneHotEncoder (if any categorical features exist)
    if categorical_features:
        # navigator to the OneHotEncoder inside ColumnTransformer pipeline
        # ColumnTransformer named transformers: [('num', Pipeline...), ('cat', Pipeline...), ...]
        # We expect pipeline inside 'cat' with step named 'ohe'
        cat_step = None
        # find the transformer with name 'cat'
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'cat':
                # transformer is a Pipeline: ('imputer', SimpleImputer), ('ohe', OneHotEncoder)
                ohe = transformer.named_steps.get('ohe', None)
                if ohe is not None:
                    categories = ohe.categories_
                    # categories is list of arrays aligned with categorical_features order
                    for col, cats in zip(categorical_features, categories):
                        # convert numpy types to python types for JSON
                        meta['categorical_features'][col] = [str(c) for c in list(cats)]
                break

    # Write metadata JSON
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved metadata to {META_PATH}")
    print("Training complete.")

if __name__ == "__main__":
    main()
