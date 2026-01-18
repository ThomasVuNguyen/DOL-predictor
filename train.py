"""
Train a small baseline model on the WHD predictive dataset with basic cleaning,
evaluation, and loss plotting.
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMNS = [
    "case_violtn_cnt",
    "cmp_assd",
    "ee_violtd_cnt",
    "bw_atp_amt",
    "ee_atp_cnt",
]


@dataclass
class CleanedData:
    features: pd.DataFrame
    targets: pd.DataFrame
    start_dates: pd.Series


def load_dataset(path: Path) -> pd.DataFrame:
    """Load JSON list of records into a flat DataFrame."""
    with path.open() as f:
        raw = json.load(f)
    df = pd.json_normalize(raw)
    # Flatten columns: input.xxx -> xxx, output.xxx -> target column
    df.columns = [col.split(".")[-1] for col in df.columns]
    return df


def parse_date_safe(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="%Y-%m-%d")


def clean_dataframe(df: pd.DataFrame, max_duration_days: int = 5 * 365) -> CleanedData:
    """Basic cleaning and feature engineering."""
    df = df.copy()
    df["findings_start_date"] = parse_date_safe(df["findings_start_date"])
    df["findings_end_date"] = parse_date_safe(df["findings_end_date"])

    # Drop rows with invalid dates or wildly implausible years.
    mask_valid_years = (
        df["findings_start_date"].dt.year.between(1980, 2025)
        & df["findings_end_date"].dt.year.between(1980, 2025)
    )
    df = df[mask_valid_years]

    df["duration_days"] = (
        df["findings_end_date"] - df["findings_start_date"]
    ).dt.days

    # Remove negative or extreme durations; most rows are ~2 years.
    df = df[(df["duration_days"] > 0) & (df["duration_days"] <= max_duration_days)]

    # Fill missing categories with explicit unknown tokens.
    df["flsa_repeat_violator"] = (
        df["flsa_repeat_violator"].replace("", "UNK").fillna("UNK")
    )
    df["naic_cd"] = df["naic_cd"].fillna("UNK").astype(str)
    df["naics_code_description"] = (
        df["naics_code_description"].replace("", "UNK").fillna("UNK")
    )
    df["st_cd"] = df["st_cd"].fillna("UNK")

    # Add simple temporal features.
    df["start_year"] = df["findings_start_date"].dt.year
    df["start_month"] = df["findings_start_date"].dt.month
    df["end_year"] = df["findings_end_date"].dt.year
    df["end_month"] = df["findings_end_date"].dt.month

    feature_cols = [
        "st_cd",
        "naic_cd",
        "naics_code_description",
        "flsa_repeat_violator",
        "duration_days",
        "start_year",
        "start_month",
        "end_year",
        "end_month",
    ]
    features = df[feature_cols]
    targets = df[TARGET_COLUMNS]
    return CleanedData(
        features=features,
        targets=targets,
        start_dates=df["findings_start_date"],
    )


def build_pipeline(
    cat_cols: List[str], num_cols: List[str], n_jobs: int, use_hist: bool
) -> Pipeline:
    # HistGradientBoostingRegressor requires dense input; switch encoder output accordingly.
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    # Cap cardinality to keep dense matrices manageable.
                    handle_unknown="infrequent_if_exist",
                    min_frequency=200,
                    max_categories=200,
                    sparse_output=False,
                ),
            ),
        ]
    )
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical, cat_cols),
            ("num", numeric, num_cols),
        ]
    )

    if use_hist:
        base_regressor = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_bins=255,
            l2_regularization=0.0,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=0,
        )
    else:
        base_regressor = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=0,
        )
    model = MultiOutputRegressor(base_regressor, n_jobs=n_jobs)

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def train_model(
    X_train: pd.DataFrame, y_train: pd.DataFrame, n_jobs: int, use_hist: bool
) -> MultiOutputRegressor:
    cat_cols = [
        "st_cd",
        "naic_cd",
        "naics_code_description",
        "flsa_repeat_violator",
    ]
    num_cols = [
        "duration_days",
        "start_year",
        "start_month",
        "end_year",
        "end_month",
    ]
    pipeline = build_pipeline(cat_cols, num_cols, n_jobs=n_jobs, use_hist=use_hist)
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate(
    model: MultiOutputRegressor, X: pd.DataFrame, y: pd.DataFrame
) -> Tuple[dict, dict]:
    preds = model.predict(X)
    mae = {}
    rmse = {}
    for i, target in enumerate(TARGET_COLUMNS):
        y_true = y.iloc[:, i]
        y_pred = preds[:, i]
        mae[target] = mean_absolute_error(y_true, y_pred)
        rmse[target] = mean_squared_error(y_true, y_pred) ** 0.5
    return mae, rmse


def collect_loss_curves(model: MultiOutputRegressor) -> List[np.ndarray]:
    """Extract train scores from each estimator and average per boosting iteration."""
    curves: List[np.ndarray] = []
    for estimator in model.estimators_:
        # GradientBoostingRegressor stores negative MSE per stage in train_score_
        if hasattr(estimator, "train_score_"):
            curves.append(-np.array(estimator.train_score_))
    if not curves:
        return []
    # Align lengths and average.
    min_len = min(len(c) for c in curves)
    trimmed = [c[:min_len] for c in curves]
    return trimmed


def plot_losses(loss_curves: List[np.ndarray], output_dir: Path) -> None:
    if not loss_curves:
        print("No loss curves available to plot.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    avg_curve = np.vstack(loss_curves).mean(axis=0)
    plt.figure(figsize=(8, 5))
    plt.plot(avg_curve, label="Average train loss (MSE)")
    plt.xlabel("Boosting iteration")
    plt.ylabel("Loss")
    plt.title("Training loss across targets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()


def save_metrics(mae: dict, rmse: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump({"mae": mae, "rmse": rmse}, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train baseline model on WHD predictive dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("whd_predictive_dataset.json"),
        help="Path to JSON dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Where to store plots and metrics.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for model training (MultiOutputRegressor). Default 1 avoids large memory duplication; increase cautiously.",
    )
    parser.add_argument(
        "--use-hist",
        dest="use_hist",
        action="store_true",
        default=True,
        help="Use HistGradientBoostingRegressor (faster, multicore). Enabled by default.",
    )
    parser.add_argument(
        "--no-hist",
        dest="use_hist",
        action="store_false",
        help="Disable hist gradient boosting and use classic GradientBoosting.",
    )
    args = parser.parse_args()

    print("Loading dataset...")
    df = load_dataset(args.dataset)
    print(f"Loaded {len(df)} raw rows.")

    cleaned = clean_dataframe(df)
    print(f"Cleaned rows: {len(cleaned.features)}")

    # Time-aware split: hold out most recent 20% by start date.
    sorted_idx = cleaned.start_dates.sort_values().index
    holdout_size = int(0.2 * len(sorted_idx))
    train_idx = sorted_idx[:-holdout_size]
    val_idx = sorted_idx[-holdout_size:]

    X_train = cleaned.features.loc[train_idx]
    y_train = cleaned.targets.loc[train_idx]
    X_val = cleaned.features.loc[val_idx]
    y_val = cleaned.targets.loc[val_idx]

    print("Training model...")
    model = train_model(X_train, y_train, n_jobs=args.n_jobs, use_hist=args.use_hist)

    print("Evaluating...")
    mae, rmse = evaluate(model, X_val, y_val)
    for target in TARGET_COLUMNS:
        print(
            f"{target}: MAE={mae[target]:.2f} RMSE={rmse[target]:.2f}"
        )

    loss_curves = collect_loss_curves(model.named_steps["model"])
    plot_losses(loss_curves, args.output_dir)
    save_metrics(mae, rmse, args.output_dir)

    print(f"Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
