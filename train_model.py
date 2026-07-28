import json
import os
from collections import Counter
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from eda_analysis import generate_eda_graphs, generate_model_evaluation_graphs
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score, recall_score,
                             roc_auc_score, roc_curve, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "accepted_datasets_cleaned.csv"
MODEL_PATH = "model.pkl"
PREPROCESSOR_PATH = "preprocessor.pkl"
INFO_PATH = "model_info.json"
GRAPH_DIR = "graphs"


def load_dataset(path):
    df = pd.read_csv(path)
    return df


def clean_dataset(df):
    df = df.copy()
    bool_map = {"True": 1, "False": 0, "true": 1, "false": 0, "YES": 1, "NO": 0, "Yes": 1, "No": 0}
    for col in ["AI_Supervision", "Failure_Within_7_Days"]:
        if col in df.columns:
            df[col] = df[col].map(bool_map).fillna(df[col])
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Machine_ID" in df.columns:
        df = df.drop(columns=["Machine_ID"])
    df = df.dropna(subset=["Failure_Within_7_Days"])
    return df


def build_model(df):
    target_column = "Failure_Within_7_Days"
    feature_columns = [col for col in df.columns if col != target_column]

    categorical_features = ["Machine_Type"]
    numeric_features = [
        col for col in feature_columns if col not in categorical_features
    ]

    # Preprocessing
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    scaler = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", encoder, categorical_features),
            ("num", scaler, numeric_features),
        ]
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        max_depth=16,
    )

    # Features & Target
    X = df[feature_columns]
    y = df[target_column].astype(int)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Fit preprocessor
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Train model
    model.fit(X_train_transformed, y_train)

    # Predictions
    y_pred = model.predict(X_test_transformed)
    y_proba = model.predict_proba(X_test_transformed)[:, 1]

    # Metrics
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred)

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    # Get feature names from fitted preprocessor
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        fitted_encoder = preprocessor.named_transformers_["cat"]

        feature_names = []

        if hasattr(fitted_encoder, "get_feature_names_out"):
            feature_names.extend(
                fitted_encoder.get_feature_names_out(
                    categorical_features
                ).tolist()
            )

        feature_names.extend(numeric_features)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "confusion_matrix": {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            },
            "classification_report": report,
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            },
        },
        "machine_types": sorted(
            df["Machine_Type"].dropna().unique().tolist()
        ),
    }

def save_artifacts(model, preprocessor, info):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    with open(INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    print("Loading dataset...")
    df = load_dataset(DATA_PATH)
    print(f"Dataset loaded with {len(df)} records.")

    print("Running exploratory data analysis and generating graphs...")
    df = generate_eda_graphs(
        df,
        graph_dir=GRAPH_DIR,
        clean_existing=True,
        remove_outliers=True,
    )

    df = clean_dataset(df)
    print(f"Cleaned dataset has {len(df)} records after EDA preprocessing.")

    artifacts = build_model(df)
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    feature_names = artifacts["feature_names"]
    metrics = artifacts["metrics"]
    machine_types = sorted(artifacts["machine_types"])

    info = {
        "dataset_name": os.path.basename(DATA_PATH),
        "num_records": int(len(df)),
        "num_features": int(len(df.columns) - 1),
        "target_variable": "Failure_Within_7_Days",
        "algorithm": "Random Forest Classifier",
        "model_timestamp": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
        "feature_importances": [
            {"name": name, "value": float(value)}
            for name, value in sorted(
                zip(feature_names, model.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )[:20]
        ],
        "roc_curve": metrics["roc_curve"],
        "machine_types": machine_types,
    }

    save_artifacts(model, preprocessor, info)
    generate_model_evaluation_graphs(
        info["feature_importances"],
        metrics["confusion_matrix"],
        metrics["roc_curve"],
        graph_dir=GRAPH_DIR,
    )
    print(f"Model saved to {MODEL_PATH}")
    print(f"Preprocessor saved to {PREPROCESSOR_PATH}")
    print(f"Model info saved to {INFO_PATH}")
    print(f"EDA and model graphs saved to {GRAPH_DIR}")


if __name__ == "__main__":
    main()
