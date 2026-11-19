"""Reproducible training entry point for the encroachment severity classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from shapely import wkt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
NUMERIC_FEATURES = [
    "distance_to_road", "encroachment_depth", "encroachment_area",
    "building_area", "encroachment_percentage", "latitude", "longitude",
]
CATEGORICAL_FEATURES = ["building_type"]
TARGET = "severity"


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}. Run the notebook export cell and place "
            "encroachments_analysis.csv in data/processed/."
        )
    frame = pd.read_csv(path)
    frame = frame.copy()
    # The notebook's export cell stores geometry as WKT and does not export
    # the later engineered latitude/longitude columns. Derive them when absent.
    if ("latitude" not in frame.columns or "longitude" not in frame.columns) and "geometry" in frame.columns:
        def centroid_coordinate(value):
            if pd.isna(value):
                return (float("nan"), float("nan"))
            try:
                centroid = wkt.loads(value).centroid
                return (centroid.y, centroid.x)
            except Exception:
                return (float("nan"), float("nan"))
        coords = frame["geometry"].apply(centroid_coordinate)
        if "latitude" not in frame.columns:
            frame["latitude"] = coords.apply(lambda pair: pair[0])
        if "longitude" not in frame.columns:
            frame["longitude"] = coords.apply(lambda pair: pair[1])
    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing) +
            ". Check the notebook export and feature engineering."
        )
    for col in NUMERIC_FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["building_type"] = frame["building_type"].fillna("unknown").astype(str)
    frame[TARGET] = frame[TARGET].astype(str).str.strip()
    frame = frame.dropna(subset=[TARGET])
    frame = frame[frame[TARGET].isin(["Low", "Medium", "High", "Critical"])]
    if frame.empty:
        raise ValueError("No valid severity-labelled rows remain after validation.")
    return frame


def build_pipeline() -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            decision_function_shape="ovr",
            random_state=42,
        )),
    ])


def train(input_path: Path, model_path: Path, output_dir: Path) -> dict:
    data = load_dataset(input_path)
    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data[TARGET]

    counts = y.value_counts()
    if counts.min() < 5:
        raise ValueError(
            "At least five examples per severity class are required for 5-fold "
            f"stratified cross-validation. Current class counts: {counts.to_dict()}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "classifier__C": [0.1, 1, 10, 100],
            "classifier__gamma": ["scale", "auto", 0.01, 0.001],
        },
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    predictions = search.predict(X_test)

    labels = ["Low", "Medium", "High", "Critical"]
    report = classification_report(
        y_test, predictions, labels=labels, output_dict=True, zero_division=0
    )
    metrics = {
        "dataset_rows": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_distribution": {str(k): int(v) for k, v in counts.items()},
        "accuracy": float(accuracy_score(y_test, predictions)),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "best_parameters": search.best_params_,
        "best_cv_weighted_f1": float(search.best_score_),
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "target": TARGET,
        "interpretation": (
            "Severity labels are deterministically derived from encroachment depth. "
            "This classifier reproduces the measured-severity classification and "
            "must not be described as forecasting future encroachment."
        ),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": search.best_estimator_,
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "target_classes": labels,
    }, model_path)
    (output_dir / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "data" / "processed" / "encroachments_analysis.csv"
    )
    parser.add_argument(
        "--model", type=Path, default=ROOT / "models" / "encroachment_svm.joblib"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    metrics = train(args.input, args.model, args.output_dir)
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model: {args.model}")


if __name__ == "__main__":
    main()
