import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "accepted_datasets_cleaned.csv"
TARGET_COLUMN = "Failure_Within_7_Days"
GRAPH_DIR = Path("graphs")
PLOT_SAMPLE_SIZE = 5000
PAIRPLOT_SAMPLE_SIZE = 1200

IMPORTANT_NUMERICAL_FEATURES = [
    "Operational_Hours",
    "Temperature_C",
    "Vibration_mms",
    "Sound_dB",
    "Oil_Level_pct",
    "Coolant_Level_pct",
    "Power_Consumption_kW",
    "Last_Maintenance_Days_Ago",
    "Failure_History_Count",
    "Remaining_Useful_Life_days",
    "Hydraulic_Pressure_bar",
    "Coolant_Flow_L_min",
    "Heat_Index",
]


def prepare_graph_directory(graph_dir=GRAPH_DIR):
    graph_dir = Path(graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    for graph_file in graph_dir.glob("*.png"):
        graph_file.unlink()
    return graph_dir


def load_dataset(path=DATA_PATH):
    return pd.read_csv(path)


def clean_dataset_for_analysis(df):
    df = df.copy()
    bool_map = {
        "True": 1,
        "False": 0,
        "true": 1,
        "false": 0,
        "YES": 1,
        "NO": 0,
        "Yes": 1,
        "No": 0,
        True: 1,
        False: 0,
    }

    for col in ["AI_Supervision", TARGET_COLUMN]:
        if col in df.columns:
            df[col] = df[col].map(bool_map).fillna(df[col])
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if TARGET_COLUMN in df.columns:
        df = df.dropna(subset=[TARGET_COLUMN])

    return df


def get_numerical_columns(df, include_target=False):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded = {"Machine_ID"}
    if not include_target:
        excluded.add(TARGET_COLUMN)
    return [col for col in numeric_columns if col not in excluded]


def get_categorical_columns(df):
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def sample_for_plotting(df, sample_size=PLOT_SAMPLE_SIZE):
    if len(df) <= sample_size:
        return df.copy()
    return df.sample(sample_size, random_state=42)


def safe_filename(name):
    return "".join(char if char.isalnum() else "_" for char in str(name)).strip("_").lower()


def format_label(column_name):
    return (
        str(column_name)
        .replace("_", " ")
        .replace("pct", "%")
        .replace("mms", "mm/s")
        .replace(" kW", " kW")
        .replace(" bar", " bar")
    )


def save_current_figure(graph_dir, filename):
    output_path = Path(graph_dir) / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def display_dataset_information(df):
    print("\nDATASET INFORMATION")
    print("=" * 70)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False))
    print(f"\nDuplicate records: {df.duplicated().sum()}")
    print("\nStatistical summary:")
    print(df.describe(include="all").transpose())


def remove_outliers_iqr(df, numerical_columns=None):
    df_clean = df.copy()
    numerical_columns = numerical_columns or get_numerical_columns(df_clean)
    original_rows = len(df_clean)

    for col in numerical_columns:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df_clean = df_clean[
            (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        ]

    removed_rows = original_rows - len(df_clean)
    print(f"\nOutlier removal: removed {removed_rows} records using the IQR method.")
    print(f"Dataset after outlier removal: {df_clean.shape}")
    return df_clean.reset_index(drop=True)


def plot_distribution_plots(df, numerical_columns, graph_dir):
    plot_df = sample_for_plotting(df)
    for col in numerical_columns:
        plt.figure(figsize=(9, 5.5))
        sns.histplot(plot_df[col].dropna(), kde=True, color="#1f77b4", bins=35)
        plt.title(f"Distribution of {format_label(col)}", fontsize=14, weight="bold")
        plt.xlabel(format_label(col))
        plt.ylabel("Frequency")
        save_current_figure(graph_dir, f"distribution_{safe_filename(col)}.png")


def plot_box_plots(df, numerical_columns, graph_dir):
    plot_df = sample_for_plotting(df)
    for col in numerical_columns:
        plt.figure(figsize=(9, 4.8))
        sns.boxplot(x=plot_df[col], color="#ff9f1c")
        plt.title(f"Box Plot of {format_label(col)}", fontsize=14, weight="bold")
        plt.xlabel(format_label(col))
        save_current_figure(graph_dir, f"boxplot_{safe_filename(col)}.png")


def plot_histograms(df, numerical_columns, graph_dir):
    plot_df = sample_for_plotting(df)
    for col in numerical_columns:
        plt.figure(figsize=(9, 5.5))
        plt.hist(plot_df[col].dropna(), bins=35, color="#2ca02c", edgecolor="white")
        plt.title(f"Histogram of {format_label(col)}", fontsize=14, weight="bold")
        plt.xlabel(format_label(col))
        plt.ylabel("Number of Records")
        save_current_figure(graph_dir, f"histogram_{safe_filename(col)}.png")


def plot_multiple_subplots(df, numerical_columns, graph_dir):
    selected_columns = [col for col in IMPORTANT_NUMERICAL_FEATURES if col in numerical_columns][:9]
    if not selected_columns:
        return

    plot_df = sample_for_plotting(df)
    rows = math.ceil(len(selected_columns) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(16, rows * 4.2))
    axes = np.array(axes).reshape(-1)

    for idx, col in enumerate(selected_columns):
        sns.histplot(plot_df[col].dropna(), kde=True, ax=axes[idx], color="#1f77b4")
        axes[idx].set_title(format_label(col), fontsize=11, weight="bold")
        axes[idx].set_xlabel(format_label(col))
        axes[idx].set_ylabel("Frequency")

    for ax in axes[len(selected_columns):]:
        ax.axis("off")

    fig.suptitle("Key Numerical Feature Distributions", fontsize=16, weight="bold")
    save_current_figure(graph_dir, "important_numerical_subplots.png")


def plot_univariate_analysis(df, graph_dir):
    plot_df = sample_for_plotting(df)
    numerical_columns = get_numerical_columns(plot_df, include_target=True)
    categorical_columns = get_categorical_columns(plot_df)

    for col in numerical_columns:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        sns.histplot(plot_df[col].dropna(), kde=True, ax=axes[0], color="#1f77b4")
        sns.boxplot(x=plot_df[col], ax=axes[1], color="#ff9f1c")
        axes[0].set_title(f"Frequency of {format_label(col)}", weight="bold")
        axes[1].set_title(f"Spread of {format_label(col)}", weight="bold")
        axes[0].set_xlabel(format_label(col))
        axes[1].set_xlabel(format_label(col))
        fig.suptitle(f"Univariate Analysis: {format_label(col)}", fontsize=15, weight="bold")
        save_current_figure(graph_dir, f"univariate_{safe_filename(col)}.png")

    for col in categorical_columns:
        plt.figure(figsize=(10, 5.5))
        order = plot_df[col].value_counts().head(20).index
        sns.countplot(data=plot_df, y=col, order=order, color="#1f77b4")
        plt.title(f"Univariate Analysis: {format_label(col)}", fontsize=14, weight="bold")
        plt.xlabel("Count")
        plt.ylabel(format_label(col))
        save_current_figure(graph_dir, f"univariate_{safe_filename(col)}.png")


def plot_categorical_count_plots(df, graph_dir):
    plot_df = sample_for_plotting(df)
    for col in get_categorical_columns(plot_df):
        plt.figure(figsize=(10, 5.5))
        order = plot_df[col].value_counts().head(20).index
        sns.countplot(data=plot_df, y=col, order=order, color="#17a2b8")
        plt.title(f"Distribution of {format_label(col)}", fontsize=14, weight="bold")
        plt.xlabel("Count")
        plt.ylabel(format_label(col))
        save_current_figure(graph_dir, f"categorical_count_{safe_filename(col)}.png")


def plot_scatter_relationships(df, graph_dir):
    if TARGET_COLUMN not in df.columns:
        return

    plot_df = sample_for_plotting(df)
    candidate_pairs = [
        ("Operational_Hours", "Temperature_C"),
        ("Operational_Hours", "Vibration_mms"),
        ("Temperature_C", "Heat_Index"),
        ("Vibration_mms", "Sound_dB"),
        ("Power_Consumption_kW", "Temperature_C"),
        ("Hydraulic_Pressure_bar", "Coolant_Flow_L_min"),
        ("Last_Maintenance_Days_Ago", "Failure_History_Count"),
        ("Remaining_Useful_Life_days", "Operational_Hours"),
    ]

    for x_col, y_col in candidate_pairs:
        if x_col not in plot_df.columns or y_col not in plot_df.columns:
            continue

        plt.figure(figsize=(9, 6))
        sns.scatterplot(
            data=plot_df,
            x=x_col,
            y=y_col,
            hue=TARGET_COLUMN,
            palette={0: "#2ca02c", 1: "#d62728"},
            alpha=0.72,
            s=34,
        )
        plt.title(
            f"{format_label(y_col)} vs {format_label(x_col)} by Failure Risk",
            fontsize=14,
            weight="bold",
        )
        plt.xlabel(format_label(x_col))
        plt.ylabel(format_label(y_col))
        plt.legend(title="Failure Within 7 Days")
        save_current_figure(graph_dir, f"scatter_{safe_filename(x_col)}_vs_{safe_filename(y_col)}.png")


def plot_target_box_plots(df, numerical_columns, graph_dir):
    if TARGET_COLUMN not in df.columns:
        return

    plot_df = sample_for_plotting(df)
    for col in numerical_columns:
        plt.figure(figsize=(8.5, 5.5))
        sns.boxplot(
            data=plot_df,
            x=TARGET_COLUMN,
            y=col,
            hue=TARGET_COLUMN,
            palette={0: "#2ca02c", 1: "#d62728"},
            legend=False,
        )
        plt.title(
            f"{format_label(col)} Compared by Failure Outcome",
            fontsize=14,
            weight="bold",
        )
        plt.xlabel("Failure Within 7 Days")
        plt.ylabel(format_label(col))
        save_current_figure(graph_dir, f"target_boxplot_{safe_filename(col)}.png")


def plot_pair_matrix(df, graph_dir):
    if TARGET_COLUMN not in df.columns:
        return

    selected_columns = [
        col for col in [
            "Operational_Hours",
            "Temperature_C",
            "Vibration_mms",
            "Power_Consumption_kW",
            "Remaining_Useful_Life_days",
            TARGET_COLUMN,
        ]
        if col in df.columns
    ]
    if len(selected_columns) < 3:
        return

    plot_df = sample_for_plotting(df[selected_columns].dropna(), PAIRPLOT_SAMPLE_SIZE)
    pair_grid = sns.pairplot(
        plot_df,
        hue=TARGET_COLUMN,
        diag_kind="hist",
        palette={0: "#2ca02c", 1: "#d62728"},
        plot_kws={"alpha": 0.62, "s": 24},
    )
    pair_grid.fig.suptitle("Scatter Matrix for Selected Maintenance Features", y=1.02, fontsize=16, weight="bold")
    pair_grid.savefig(Path(graph_dir) / "scatter_matrix_selected_features.png", dpi=300, bbox_inches="tight")
    plt.close(pair_grid.fig)


def plot_line_trends(df, graph_dir):
    selected_columns = [
        col for col in [
            "Operational_Hours",
            "Temperature_C",
            "Vibration_mms",
            "Power_Consumption_kW",
            "Remaining_Useful_Life_days",
        ]
        if col in df.columns
    ]
    if not selected_columns:
        return

    plot_df = sample_for_plotting(df, 1200).reset_index(drop=True)
    x_values = plot_df.index

    plt.figure(figsize=(13, 6))
    for col in selected_columns:
        rolling_values = plot_df[col].rolling(window=25, min_periods=1).mean()
        plt.plot(x_values, rolling_values, label=format_label(col), linewidth=1.8)

    plt.title("Trend Lines for Selected Numerical Variables", fontsize=15, weight="bold")
    plt.xlabel("Sample Order")
    plt.ylabel("Rolling Average")
    plt.legend(loc="best")
    save_current_figure(graph_dir, "line_trends_selected_variables.png")


def plot_correlation_heatmap(df, graph_dir):
    numerical_columns = get_numerical_columns(df, include_target=True)
    if len(numerical_columns) < 2:
        return

    corr = df[numerical_columns].corr(numeric_only=True)
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.35, annot=False, cbar_kws={"label": "Correlation"})
    plt.title("Correlation Heatmap for Numerical Features", fontsize=16, weight="bold")
    save_current_figure(graph_dir, "correlation_heatmap_numerical_features.png")


def train_model_for_visual_evaluation(df):
    if "Machine_ID" in df.columns:
        df = df.drop(columns=["Machine_ID"])

    target_column = TARGET_COLUMN
    feature_columns = [col for col in df.columns if col != target_column]
    categorical_features = [col for col in ["Machine_Type"] if col in feature_columns]
    numeric_features = [col for col in feature_columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )
    model = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        max_depth=16,
    )

    X = df[feature_columns]
    y = df[target_column].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    model.fit(X_train_transformed, y_train)

    y_pred = model.predict(X_test_transformed)
    y_proba = model.predict_proba(X_test_transformed)[:, 1]

    feature_names = preprocessor.get_feature_names_out().tolist()
    return {
        "model": model,
        "feature_names": feature_names,
        "feature_importances": [
            {"name": name, "value": float(value)}
            for name, value in sorted(
                zip(feature_names, model.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )[:20]
        ],
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_curve": roc_curve(y_test, y_proba),
    }


def plot_feature_importance(feature_importances, graph_dir):
    if not feature_importances:
        return

    plot_data = pd.DataFrame(feature_importances).sort_values("value", ascending=True)
    plt.figure(figsize=(11, 7))
    sns.barplot(data=plot_data, x="value", y="name", color="#1f77b4")
    plt.title("Top Feature Importances from Random Forest Model", fontsize=15, weight="bold")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    save_current_figure(graph_dir, "model_feature_importance.png")


def plot_confusion_matrix(confusion, graph_dir):
    if isinstance(confusion, dict):
        matrix = np.array([[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]])
    else:
        matrix = np.asarray(confusion)

    plt.figure(figsize=(7, 5.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
    )
    plt.title("Confusion Matrix", fontsize=15, weight="bold")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    save_current_figure(graph_dir, "model_confusion_matrix.png")


def plot_roc_curve(roc_data, graph_dir):
    if isinstance(roc_data, dict):
        fpr = roc_data.get("fpr", [])
        tpr = roc_data.get("tpr", [])
    else:
        fpr, tpr, _ = roc_data

    if len(fpr) == 0 or len(tpr) == 0:
        return

    plt.figure(figsize=(7.5, 6))
    plt.plot(fpr, tpr, color="#1f77b4", linewidth=2.2, label="Random Forest ROC")
    plt.plot([0, 1], [0, 1], color="#d62728", linestyle="--", label="Random Classifier")
    plt.title("Receiver Operating Characteristic Curve", fontsize=15, weight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    save_current_figure(graph_dir, "model_roc_curve.png")


def generate_model_evaluation_graphs(feature_importances, confusion, roc_data, graph_dir=GRAPH_DIR):
    graph_dir = Path(graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    plot_feature_importance(feature_importances, graph_dir)
    plot_confusion_matrix(confusion, graph_dir)
    plot_roc_curve(roc_data, graph_dir)


def generate_eda_graphs(df, graph_dir=GRAPH_DIR, clean_existing=True, remove_outliers=True):
    graph_dir = prepare_graph_directory(graph_dir) if clean_existing else Path(graph_dir)
    df = clean_dataset_for_analysis(df)
    display_dataset_information(df)

    numerical_columns = get_numerical_columns(df)
    if remove_outliers:
        df = remove_outliers_iqr(df, numerical_columns)
        numerical_columns = get_numerical_columns(df)

    sns.set_theme(style="whitegrid", context="notebook")
    plot_distribution_plots(df, numerical_columns, graph_dir)
    plot_box_plots(df, numerical_columns, graph_dir)
    plot_histograms(df, numerical_columns, graph_dir)
    plot_multiple_subplots(df, numerical_columns, graph_dir)
    plot_univariate_analysis(df, graph_dir)
    plot_categorical_count_plots(df, graph_dir)
    plot_scatter_relationships(df, graph_dir)
    plot_target_box_plots(df, numerical_columns, graph_dir)
    plot_pair_matrix(df, graph_dir)
    plot_line_trends(df, graph_dir)
    plot_correlation_heatmap(df, graph_dir)

    print(f"\nEDA graphs saved to: {Path(graph_dir).resolve()}")
    return df


def run_full_analysis(path=DATA_PATH, graph_dir=GRAPH_DIR):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = load_dataset(path)
    df = generate_eda_graphs(df, graph_dir=graph_dir, clean_existing=True, remove_outliers=True)
    evaluation = train_model_for_visual_evaluation(df)
    generate_model_evaluation_graphs(
        evaluation["feature_importances"],
        evaluation["confusion_matrix"],
        evaluation["roc_curve"],
        graph_dir=graph_dir,
    )
    print("Model evaluation graphs saved successfully.")
    return df


if __name__ == "__main__":
    run_full_analysis()
