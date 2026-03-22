import pandas as pd
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

from xgboost import XGBClassifier


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

IN_FILE = DATA_DIR / "01_model_dataset.csv"

RF_MODEL_PATH = MODEL_DIR / "rf_outcome_model.joblib"
XGB_MODEL_PATH = MODEL_DIR / "xgb_outcome_model.joblib"


# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
SPLIT_DATE = "2021-06-01"

FEATURE_COLS = [
    "neutral",
    "home_elo_before",
    "away_elo_before",
    "home_rank_before",
    "away_rank_before",
    "elo_diff_before",
    "rank_diff_before",
    "tournament_weight",
    "home_matches_played_pre",
    "away_matches_played_pre",
    "home_form_points_avg_5",
    "away_form_points_avg_5",
    "home_form_points_sum_5",
    "away_form_points_sum_5",
    "home_gd_avg_5",
    "away_gd_avg_5",
    "home_gd_sum_5",
    "away_gd_sum_5",
    "home_gf_avg_5",
    "away_gf_avg_5",
    "home_ga_avg_5",
    "away_ga_avg_5",
    "home_win_rate_5",
    "away_win_rate_5",
    "home_last_match_gd",
    "away_last_match_gd",
    "home_last_match_points",
    "away_last_match_points",
    "home_days_since_last",
    "away_days_since_last",
    "form_points_avg_diff_5",
    "form_points_sum_diff_5",
    "gd_avg_diff_5",
    "gd_sum_diff_5",
    "gf_avg_diff_5",
    "ga_avg_diff_5",
    "win_rate_diff_5",
    "days_rest_diff",
]

TARGET_COL = "match_outcome"
LABEL_MAP = {"H": 0, "D": 1, "A": 2}
INV_LABEL_MAP = {0: "H", 1: "D", 2: "A"}


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(IN_FILE)

    if "date" not in df.columns:
        raise ValueError(f"'date' column not found in {IN_FILE}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    if TARGET_COL not in df.columns:
        raise ValueError(f"'{TARGET_COL}' column not found in {IN_FILE}")

    missing_features = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    return df


# -------------------------------------------------------------------
# Split
# -------------------------------------------------------------------
def time_split(df: pd.DataFrame, split_date: str):
    split_date = pd.Timestamp(split_date)

    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()

    if train_df.empty:
        raise ValueError("Training set is empty. Choose an earlier split date.")
    if test_df.empty:
        raise ValueError("Test set is empty. Choose a later split date.")

    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df[TARGET_COL].copy()

    X_test = test_df[FEATURE_COLS].copy()
    y_test = test_df[TARGET_COL].copy()

    # Fill missing values for model input
    X_train = X_train.fillna(-99.0)
    X_test = X_test.fillna(-99.0)

    return train_df, test_df, X_train, y_train, X_test, y_test


# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------
def train_rf_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgb_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    y_train_num = y_train.map(LABEL_MAP)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train_num)
    return model


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
def evaluate_rf_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob, labels=model.classes_)

    return acc, ll


def evaluate_xgb_model(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    y_test_num = y_test.map(LABEL_MAP)

    y_pred_num = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test_num, y_pred_num)
    ll = log_loss(y_test_num, y_prob, labels=[0, 1, 2])

    return acc, ll


# -------------------------------------------------------------------
# Save models
# -------------------------------------------------------------------
def save_models(rf_model: RandomForestClassifier, xgb_model: XGBClassifier):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(rf_model, RF_MODEL_PATH)
    joblib.dump(
        {
            "model": xgb_model,
            "label_map": LABEL_MAP,
            "inverse_label_map": INV_LABEL_MAP,
            "feature_cols": FEATURE_COLS,
        },
        XGB_MODEL_PATH,
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    df = load_dataset()

    train_df, test_df, X_train, y_train, X_test, y_test = time_split(df, SPLIT_DATE)

    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))
    print("Train date range:", train_df["date"].min().date(), "to", train_df["date"].max().date())
    print("Test date range:", test_df["date"].min().date(), "to", test_df["date"].max().date())
    print()

    rf_model = train_rf_model(X_train, y_train)
    xgb_model = train_xgb_model(X_train, y_train)

    rf_acc, rf_ll = evaluate_rf_model(rf_model, X_test, y_test)
    xgb_acc, xgb_ll = evaluate_xgb_model(xgb_model, X_test, y_test)

    save_models(rf_model, xgb_model)

    print("Random Forest Results")
    print(f"Accuracy: {rf_acc:.5f}")
    print(f"Log Loss: {rf_ll:.5f}")
    print()

    print("XGBoost Results")
    print(f"Accuracy: {xgb_acc:.5f}")
    print(f"Log Loss: {xgb_ll:.5f}")
    print()

    print(f"Saved Random Forest model to: {RF_MODEL_PATH}")
    print(f"Saved XGBoost model to: {XGB_MODEL_PATH}")


if __name__ == "__main__":
    main()