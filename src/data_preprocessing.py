"""
data_preprocessing.py
----------------------
Loads the heart failure clinical dataset, performs basic cleaning,
splits into train/test sets, and scales numeric features.

Usage:
    from data_preprocessing import load_and_prepare_data
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data("data/heart_failure_clinical_records_dataset.csv")
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Columns that are already binary (0/1) - we won't scale these
BINARY_COLS = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking", "DEATH_EVENT"]

# Continuous numeric columns that benefit from scaling for Logistic Regression
NUMERIC_COLS = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "time",
]

TARGET_COL = "DEATH_EVENT"


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame."""
    df = pd.read_csv(csv_path)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic sanity cleaning:
    - Drop exact duplicate rows
    - Confirm no missing values (dataset is known to be complete, but we check anyway)
    """
    df = df.drop_duplicates().reset_index(drop=True)

    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"Warning: found {missing} missing values. Filling with column median.")
        df = df.fillna(df.median(numeric_only=True))

    return df


def load_and_prepare_data(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Full prep pipeline:
    1. Load CSV
    2. Clean
    3. Split features/target
    4. Train/test split (stratified, since DEATH_EVENT is imbalanced)
    5. Scale numeric columns only (fit on train, transform both)

    Returns: X_train, X_test, y_train, y_test, scaler
    """
    df = load_data(csv_path)
    df = basic_clean(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    # Quick manual test when running this file directly
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data(
        "data/heart_failure_clinical_records_dataset.csv"
    )
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Class balance (train):")
    print(y_train.value_counts(normalize=True))