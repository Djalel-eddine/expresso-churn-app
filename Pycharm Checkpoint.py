import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from ydata_profiling import ProfileReport

# -----------------------------
# 1️⃣ Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = r"C:\Users\pc\Desktop\GMC DS\Cours\Streamlit\Expresso_churn_dataset.csv"
REPORT_DIR = r"C:\Users\pc\Desktop\GMC DS\Cours\Streamlit\report"
CLEAN_PATH = r"C:\Users\pc\Desktop\GMC DS\Cours\Streamlit\expresso_churn_clean.csv"

# Ensure report directory exists
os.makedirs(REPORT_DIR, exist_ok=True)


# -----------------------------
# 2️⃣ Load data
# -----------------------------
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}\n")
    print("Info:")
    print(df.info())
    print("\nDescribe (numeric):")
    print(df.describe())
    print("\nNunique per column:")
    print(df.nunique())
    return df


# -----------------------------
# 3️⃣ Clean & Encode
# -----------------------------
def clean_and_encode(df):
    # Drop duplicate IDs if any
    if 'user_id' in df.columns:
        before = df.shape[0]
        df = df.drop_duplicates(subset=['user_id'])
        print(f"Dropped {before - df.shape[0]} duplicate user_id rows.")

    # Handle numeric columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    num_cols_valid = [col for col in num_cols if df[col].notna().any()]

    num_imp = SimpleImputer(strategy='median')
    df[num_cols_valid] = num_imp.fit_transform(df[num_cols_valid])

    # Fill numeric columns that were all NaN with 0
    for col in num_cols:
        if col not in num_cols_valid:
            df[col] = 0

    # Handle categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Drop columns that are identifiers or have too many unique values
    high_card_cols = [col for col in cat_cols if df[col].nunique() > 50]
    if high_card_cols:
        print(f"⚠️ Dropping high-cardinality columns: {high_card_cols}")
        df = df.drop(columns=high_card_cols)
        cat_cols = [col for col in cat_cols if col not in high_card_cols]

    # Impute categorical columns
    cat_imp = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imp.fit_transform(df[cat_cols])

    # One-hot encode the remaining categorical columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df



# -----------------------------
# 4️⃣ Generate profiling report
# -----------------------------
def generate_profile_report(df, report_path):
    print("Generating ydata-profiling report...")
    try:
        profile = ProfileReport(df, title="Expresso Churn Data Profile", minimal=True)
        profile.to_file(report_path)
        print(f"✅ Profiling report saved to: {report_path}")
    except Exception as e:
        print(f"⚠️ Profiling report generation failed: {e}")


# -----------------------------
# 5️⃣ Main function
# -----------------------------
def main():
    df = load_data()
    df_clean = clean_and_encode(df)
    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"✅ Clean data saved to: {CLEAN_PATH}")

    report_path = os.path.join(REPORT_DIR, 'expresso_profile.html')
    generate_profile_report(df, report_path)


# -----------------------------
# 6️⃣ Run script
# -----------------------------
if __name__ == "__main__":
    main()
