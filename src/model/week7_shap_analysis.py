
"""
Week 7: SHAP analysis for 20D, 126D and 252D volatility models.

Run from repository root:
    python src/model/week7_shap_analysis.py

Uses the already-trained Random Forest pipelines from Week 6.
Does not retrain or modify any existing model.
"""

from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = ROOT / "data" / "processed" / "volatility_experiment"
MODEL_ROOT = DATA_ROOT / "model_results"
OUT_DIR = DATA_ROOT / "week7_shap"

HORIZONS = (20, 126, 252)
MAX_EXPLANATIONS = 200

EXCLUDE_COLS = {"Date", "Target_Date"}


def get_features(train):
    """Use the same numeric-feature selection as Week 6."""
    features = []

    for col in train.columns:
        if col in EXCLUDE_COLS or col.startswith("Target_"):
            continue

        if pd.api.types.is_numeric_dtype(train[col]):
            features.append(col)

    if not features:
        raise ValueError("No numeric predictor features found.")

    return features


def load_inputs(horizon):
    folder = DATA_ROOT / f"{horizon}d"

    train_path = folder / "train.csv"
    test_path = folder / "test.csv"

    model_path = (
        MODEL_ROOT / f"{horizon}d" /
        f"random_forest_{horizon}d.joblib"
    )

    for path in (train_path, test_path, model_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    pipeline = joblib.load(model_path)

    return train, test, pipeline


def normalize_shap_values(values):
    """Normalize SHAP output to a 2D samples-by-features array."""
    if isinstance(values, list):
        values = values[0]

    values = np.asarray(values)

    if values.ndim == 3:
        values = values[:, :, 0]

    if values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP array dimensions: {values.shape}"
        )

    return values


def save_importance(shap_values, features, horizon):
    importance = pd.DataFrame({
        "Feature": features,
        "Mean_Absolute_SHAP": np.abs(shap_values).mean(axis=0),
    })

    importance = importance.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    ).reset_index(drop=True)

    importance.insert(0, "Horizon_Days", horizon)

    importance.to_csv(
        OUT_DIR / f"shap_importance_{horizon}d.csv",
        index=False
    )

    return importance


def save_summary_plots(shap_values, x_df, horizon):
    # Global feature importance bar plot
    plt.figure(figsize=(10, 7))

    shap.summary_plot(
        shap_values,
        x_df,
        plot_type="bar",
        show=False,
        max_display=20
    )

    plt.title(f"{horizon}D - SHAP Feature Importance")
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / f"shap_bar_{horizon}d.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close("all")

    # SHAP beeswarm summary plot
    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_values,
        x_df,
        show=False,
        max_display=20
    )

    plt.title(f"{horizon}D - SHAP Summary")
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / f"shap_summary_{horizon}d.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close("all")


def save_dependence_plot(shap_values, x_df, feature, horizon):
    if feature not in x_df.columns:
        print(f"  Skipping missing feature: {feature}")
        return

    plt.figure(figsize=(9, 6))

    shap.dependence_plot(
        feature,
        shap_values,
        x_df,
        interaction_index="auto",
        show=False
    )

    plt.title(f"{horizon}D - SHAP Dependence: {feature}")
    plt.tight_layout()

    filename = f"shap_dependence_{feature}_{horizon}d.png"

    plt.savefig(
        OUT_DIR / filename,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close("all")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_importance = []

    print("=" * 70)
    print("WEEK 7: SHAP VOLATILITY MODEL ANALYSIS")
    print("=" * 70)

    for horizon in HORIZONS:
        print(f"\n{'=' * 60}")
        print(f"Analyzing {horizon}D volatility model")
        print("=" * 60)

        train, test, pipeline = load_inputs(horizon)

        target = f"Target_Annualized_RV_{horizon}D"

        if target not in train.columns:
            raise ValueError(f"Missing target column: {target}")

        features = get_features(train)

        missing = [f for f in features if f not in test.columns]

        if missing:
            raise ValueError(
                f"Test data is missing features: {missing}"
            )

        if "VIXCLS" not in features:
            raise ValueError("VIXCLS is not included in model features.")

        if "Sentiment_Score" not in features:
            raise ValueError(
                "Sentiment_Score is not included in model features."
            )

        # Use a deterministic sample from the test set.
        x_raw = (
            test[features]
            .replace([np.inf, -np.inf], np.nan)
            .head(MAX_EXPLANATIONS)
            .copy()
        )

        # Apply the exact imputer fitted during model training.
        imputer = pipeline.named_steps["imputer"]

        x_values = imputer.transform(x_raw)

        x_df = pd.DataFrame(
            x_values,
            columns=features,
            index=x_raw.index
        )

        # Explain the trained Random Forest model.
        rf_model = pipeline.named_steps["model"]

        explainer = shap.TreeExplainer(rf_model)

        raw_shap_values = explainer.shap_values(
            x_df,
            check_additivity=False
        )

        shap_values = normalize_shap_values(raw_shap_values)

        if shap_values.shape[1] != len(features):
            raise ValueError(
                "SHAP feature count does not match model feature count."
            )

        # Save global importance.
        importance = save_importance(
            shap_values,
            features,
            horizon
        )

        all_importance.append(importance)

        # Save summary visualizations.
        save_summary_plots(
            shap_values,
            x_df,
            horizon
        )

        # Focus on the two important new features.
        save_dependence_plot(
            shap_values,
            x_df,
            "VIXCLS",
            horizon
        )

        save_dependence_plot(
            shap_values,
            x_df,
            "Sentiment_Score",
            horizon
        )

        print(f"\nTest observations explained: {len(x_df)}")
        print(f"Number of features: {len(features)}")

        print("\nTop 10 SHAP features:")
        print(
            importance[
                ["Feature", "Mean_Absolute_SHAP"]
            ].head(10).to_string(index=False)
        )

        print("\nVIX and Sentiment SHAP importance:")

        focus = importance[
            importance["Feature"].isin(
                ["VIXCLS", "Sentiment_Score"]
            )
        ]

        print(focus.to_string(index=False))

        print(f"\nCompleted {horizon}D analysis.")

    # Combine all three horizons.
    combined = pd.concat(
        all_importance,
        ignore_index=True
    )

    combined.to_csv(
        OUT_DIR / "week7_shap_importance_all_horizons.csv",
        index=False
    )

    # Create a cross-horizon comparison table.
    comparison = combined.pivot(
        index="Feature",
        columns="Horizon_Days",
        values="Mean_Absolute_SHAP"
    )

    comparison.columns = [
        f"SHAP_{col}D" for col in comparison.columns
    ]

    comparison.to_csv(
        OUT_DIR / "week7_shap_horizon_comparison.csv"
    )

    print("\n" + "=" * 70)
    print("WEEK 7 SHAP ANALYSIS COMPLETED")
    print("=" * 70)
    print(f"Output directory: {OUT_DIR}")
    print("\nAll three volatility horizons have been analyzed.")
    print(
        "Interpretation note: SHAP values explain model predictions, "
        "not causal effects or actual option prices."
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
