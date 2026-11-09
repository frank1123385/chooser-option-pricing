"""
Compare expanding-window and rolling-window Random Forest volatility forecasts.

Run from the project root:
    python src/model/volatility_window_comparison.py

The existing train/validation/test CSVs are read but never modified.
The test set is used only for final evaluation.
"""

from pathlib import Path
import ast
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "processed" / "volatility_experiment"
RESULT_ROOT = DATA_ROOT / "window_comparison"
ROLLING_WINDOW = 252  # approximately one trading year

BEST_PARAMS = {
    20: {
        "max_depth": 3,
        "min_samples_leaf": 10,
        "n_estimators": 100,
    },
    126: {
        "max_depth": 5,
        "min_samples_leaf": 10,
        "n_estimators": 200,
    },
    252: {
        "max_depth": None,
        "min_samples_leaf": 5,
        "n_estimators": 100,
    },
}


def make_model(params, seed=42):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    **params,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def load_split(horizon, split):
    path = DATA_ROOT / f"{horizon}d" / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)
    target_col = f"Target_Annualized_RV_{horizon}D"

    required = {"Date", "Target_Date", target_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"])
    df["Target_Date"] = pd.to_datetime(df["Target_Date"])
    return df, target_col


def feature_columns(df, target_col):
    excluded = {"Date", "Target_Date", target_col}
    cols = [c for c in df.columns if c not in excluded]
    numeric_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        raise ValueError("No numeric feature columns were found.")
    return numeric_cols


def score_predictions(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
        "Bias": float(np.mean(np.asarray(predicted) - np.asarray(actual))),
    }


def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    all_predictions = []

    for horizon in [20, 126, 252]:
        print(f"\n========== {horizon}-DAY WINDOW COMPARISON ==========")

        train, target_col = load_split(horizon, "train")
        validation, _ = load_split(horizon, "validation")
        test, _ = load_split(horizon, "test")

        # Validation is used only to form the development pool.
        # The split construction already purged target windows at boundaries.
        development = pd.concat([train, validation], ignore_index=True)
        development = development.sort_values("Date").reset_index(drop=True)
        test = test.sort_values("Date").reset_index(drop=True)

        features = feature_columns(development, target_col)
        missing_test_features = set(features).difference(test.columns)
        if missing_test_features:
            raise ValueError(
                f"Test data missing features: {sorted(missing_test_features)}"
            )

        X_test = test[features]
        y_test = test[target_col].astype(float)

        params = BEST_PARAMS[horizon]

        strategies = {
            "Expanding_Window": development,
            "Rolling_Window_252D": development.tail(ROLLING_WINDOW),
        }

        for strategy_name, fit_data in strategies.items():
            X_fit = fit_data[features]
            y_fit = fit_data[target_col].astype(float)

            model = make_model(params)
            model.fit(X_fit, y_fit)
            predictions = model.predict(X_test)

            metrics = score_predictions(y_test, predictions)
            metrics.update(
                {
                    "Horizon_Days": horizon,
                    "Strategy": strategy_name,
                    "Train_Rows": len(fit_data),
                    "Test_Rows": len(test),
                    "Train_Start": fit_data["Date"].min().date().isoformat(),
                    "Train_End": fit_data["Date"].max().date().isoformat(),
                    "Test_Start": test["Date"].min().date().isoformat(),
                    "Test_End": test["Date"].max().date().isoformat(),
                }
            )
            all_metrics.append(metrics)

            pred_df = pd.DataFrame(
                {
                    "Date": test["Date"],
                    "Target_Date": test["Target_Date"],
                    "Actual": y_test,
                    "Prediction": predictions,
                    "Horizon_Days": horizon,
                    "Strategy": strategy_name,
                }
            )
            all_predictions.append(pred_df)

            print(
                f"{strategy_name}: "
                f"MAE={metrics['MAE']:.4f}, "
                f"RMSE={metrics['RMSE']:.4f}, "
                f"R2={metrics['R2']:.4f}, "
                f"Bias={metrics['Bias']:.4f}, "
                f"train_rows={len(fit_data)}"
            )

        # Include the existing historical-volatility baseline for a direct
        # comparison, if its predictions file is present.
        baseline_path = (
            DATA_ROOT
            / "model_results"
            / f"{horizon}d"
            / "test_predictions.csv"
        )
        if baseline_path.exists():
            baseline_df = pd.read_csv(baseline_path)
            baseline_col = "Historical_60D_volatility"
            if baseline_col in baseline_df.columns:
                baseline_pred = baseline_df[baseline_col].astype(float)
                # The saved baseline file is generated from the same ordered test split.
                if len(baseline_pred) == len(y_test):
                    metrics = score_predictions(y_test, baseline_pred)
                    metrics.update(
                        {
                            "Horizon_Days": horizon,
                            "Strategy": "Historical_60D_Baseline",
                            "Train_Rows": np.nan,
                            "Test_Rows": len(test),
                            "Train_Start": "",
                            "Train_End": "",
                            "Test_Start": test["Date"].min().date().isoformat(),
                            "Test_End": test["Date"].max().date().isoformat(),
                        }
                    )
                    all_metrics.append(metrics)

                    all_predictions.append(
                        pd.DataFrame(
                            {
                                "Date": test["Date"],
                                "Target_Date": test["Target_Date"],
                                "Actual": y_test,
                                "Prediction": baseline_pred,
                                "Horizon_Days": horizon,
                                "Strategy": "Historical_60D_Baseline",
                            }
                        )
                    )

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    metrics_path = RESULT_ROOT / "window_comparison_metrics.csv"
    predictions_path = RESULT_ROOT / "window_comparison_predictions.csv"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print("\n========== FINAL TEST COMPARISON ==========")
    print(
        metrics_df[
            ["Horizon_Days", "Strategy", "MAE", "RMSE", "R2", "Bias", "Train_Rows"]
        ]
        .sort_values(["Horizon_Days", "MAE"])
        .round(4)
        .to_string(index=False)
    )
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved predictions: {predictions_path}")
    print("\nOriginal Week 5/6 files and existing volatility-model outputs were not changed.")


if __name__ == "__main__":
    main()
