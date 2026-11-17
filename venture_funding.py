"""Leakage-resistant, reproducible classifiers for historical applicant data.

The original course notebook is preserved separately. This is a modern, tested
reference implementation with a dummy baseline, logistic model, and small MLP.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "Starter_Code" / "Resources" / "applicants_data.csv"
TARGET = "IS_SUCCESSFUL"
IDENTIFIERS = ("EIN", "NAME")


def load_applicants(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    if TARGET not in data:
        raise ValueError(f"CSV is missing the {TARGET} target column")
    if data.empty:
        raise ValueError("No applicant records found")
    if data.columns.duplicated().any():
        raise ValueError("Duplicate CSV column names")
    target = pd.to_numeric(data[TARGET], errors="raise")
    if target.isna().any() or set(target.unique()) != {0, 1}:
        raise ValueError("IS_SUCCESSFUL must contain nonmissing binary labels (0 and 1)")
    data[TARGET] = target.astype(int)
    return data


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    category_columns = features.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    if not (category_columns or numeric_columns):
        raise ValueError("Applicant data has no model features")

    steps = []
    if category_columns:
        steps.append((
            "categorical",
            Pipeline([
                ("fill", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            category_columns,
        ))
    if numeric_columns:
        steps.append((
            "numeric",
            Pipeline([
                ("fill", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric_columns,
        ))
    return ColumnTransformer(steps, remainder="drop")


def evaluate_models(
    data: pd.DataFrame,
    random_state: int = 1,
    test_size: float = 0.25,
    models: tuple[str, ...] = ("baseline", "logistic", "mlp"),
) -> dict:
    if not 0 < test_size < 1:
        raise ValueError("test_size must lie between 0 and 1")
    target = data[TARGET]
    features = data.drop(columns=[TARGET, *IDENTIFIERS], errors="ignore")
    if TARGET in features.columns:
        raise AssertionError("Target leakage: target remains among features")
    if not models or any(name not in {"baseline", "logistic", "mlp"} for name in models):
        raise ValueError("Invalid model name")
    if target.nunique() != 2:
        raise ValueError("Both classes required")
    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=test_size, random_state=random_state,
        stratify=target,
    )
    estimators = {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "logistic": LogisticRegression(max_iter=1000, random_state=random_state),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=40,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=random_state,
        ),
    }
    report = {
        "records": int(len(data)),
        "train_count": int(len(train_x)),
        "test_count": int(len(test_x)),
        "input_features": features.columns.tolist(),
        "target": TARGET,
        "random_seed": random_state,
        "test_size": test_size,
        "test_positive_count": int(test_y.sum()),
        "method": "Stratified holdout; preprocessing fit only on training split",
        "models": {},
        "limitation": "Historical educational data; results are not a real-world credit/funding decision system.",
    }
    for name in models:
        estimator = Pipeline([
            ("preprocess", make_preprocessor(train_x)),
            ("model", estimators[name]),
        ])
        estimator.fit(train_x, train_y)
        predicted = estimator.predict(test_x)
        report["models"][name] = {
            "accuracy": float(accuracy_score(test_y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(test_y, predicted)),
            "precision": float(precision_score(test_y, predicted, zero_division=0)),
            "recall": float(recall_score(test_y, predicted, zero_division=0)),
            "f1": float(f1_score(test_y, predicted, zero_division=0)),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path("reports/latest.json"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model", choices=["all", "baseline", "logistic", "mlp"], default="all")
    parser.add_argument("--sample", type=int, default=None, help="Optional deterministic sample for a quick run")
    args = parser.parse_args()
    data = load_applicants(args.input)
    if args.sample is not None:
        if not 20 <= args.sample <= len(data):
            parser.error("--sample must be between 20 and the full record count")
        data = data.sample(n=args.sample, random_state=args.seed)
    selected = ("baseline", "logistic", "mlp") if args.model == "all" else (args.model,)
    report = evaluate_models(data, random_state=args.seed, models=selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluated {', '.join(selected)} on {report['test_count']} holdout applicants")


if __name__ == "__main__":
    main()
