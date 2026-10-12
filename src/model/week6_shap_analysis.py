"""
Week 6 SHAP explanations for the tuned Random Forest models.
Run from the repository root:
    python src/model/week6_shap_analysis.py
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
DATA_DIR = ROOT / "data" / "processed" / "week5"
OUT_DIR = ROOT / "data" / "processed" / "week6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = {
    "approach_a": {
        "model": "week6_random_forest_volatility.joblib",
        "target": "Target_Next_Day_Abs_Return",
        "exclude_prices": False,
        "label": "Approach A - Next-day Absolute Return",
    },
    "approach_b": {
        "model": "week6_end_to_end_random_forest.joblib",
        "target": "Target_Next_Day_Return",
        "exclude_prices": True,
        "label": "Approach B - Next-day Return",
    },
}
NON_FEATURE_COLS = {
    "Date", "Target_Date", "Target_Next_Day_Return",
    "Target_Next_Day_Abs_Return", "Target_Next_Day_Price",
}

def numeric_features(df, exclude_prices):
    cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    if exclude_prices:
        cols = [c for c in cols if c not in {"Close", "Adj Close"}]
    return cols

def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    if "Date" in train.columns:
        train["Date"] = pd.to_datetime(train["Date"], errors="coerce")
    if "Date" in test.columns:
        test["Date"] = pd.to_datetime(test["Date"], errors="coerce")

    all_importance = []
    for key, cfg in TARGETS.items():
        model_path = OUT_DIR / cfg["model"]
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}. Run week6_model_tuning.py first.")
        pipeline = joblib.load(model_path)
        features = numeric_features(train, cfg["exclude_prices"])
        missing = [f for f in features if f not in test.columns]
        if missing:
            raise ValueError(f"Test data is missing expected features: {missing}")

        # Explain a manageable, deterministic sample of test observations.
        x_raw = test[features].replace([np.inf, -np.inf], np.nan)
        x_raw = x_raw.head(min(200, len(x_raw))).copy()
        imputer = pipeline.named_steps["imputer"]
        x_values = imputer.transform(x_raw)
        x_df = pd.DataFrame(x_values, columns=features, index=x_raw.index)

        tree_model = pipeline.named_steps["model"]
        explainer = shap.TreeExplainer(tree_model)
        shap_values = explainer.shap_values(x_df, check_additivity=False)
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 0]

        # Save exact per-feature mean absolute SHAP values.
        importance = pd.DataFrame({
            "Feature": features,
            "Mean_Absolute_SHAP": np.abs(shap_values).mean(axis=0),
        }).sort_values("Mean_Absolute_SHAP", ascending=False)
        importance.to_csv(OUT_DIR / f"{key}_shap_importance.csv", index=False)
        importance.insert(0, "Approach", cfg["label"])
        all_importance.append(importance)

        # SHAP summary bar plot.
        plt.figure()
        shap.summary_plot(shap_values, x_df, plot_type="bar", show=False, max_display=15)
        plt.title(f"{cfg['label']}: SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{key}_shap_bar.png", dpi=180, bbox_inches="tight")
        plt.close("all")

        # SHAP beeswarm plot.
        plt.figure()
        shap.summary_plot(shap_values, x_df, show=False, max_display=15)
        plt.title(f"{cfg['label']}: SHAP Summary")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{key}_shap_summary.png", dpi=180, bbox_inches="tight")
        plt.close("all")

        print(f"\n{cfg['label']}")
        print(importance[["Feature", "Mean_Absolute_SHAP"]].head(10).to_string(index=False))
        print(f"Saved: {key}_shap_importance.csv, {key}_shap_bar.png, {key}_shap_summary.png")

    pd.concat(all_importance, ignore_index=True).to_csv(
        OUT_DIR / "week6_shap_importance_all.csv", index=False
    )
    print(f"\nSHAP analysis completed. Outputs saved to: {OUT_DIR}")
    print("Interpretation note: SHAP values explain model predictions, not causal effects or actual Chooser option prices.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
