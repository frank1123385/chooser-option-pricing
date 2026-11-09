
"""
Volatility model training-window sensitivity experiment.

Compares:
- Expanding Window
- Rolling Window 252D
- Rolling Window 504D
- Rolling Window 756D
- Historical 60D baseline

The original window_comparison outputs are NOT overwritten.

Run from project root:
    python src/model/volatility_window_sensitivity.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = ROOT / "data" / "processed" / "volatility_experiment"

# Separate output folder: preserve all original results.
RESULT_ROOT = DATA_ROOT / "window_sensitivity"

ROLLING_WINDOWS = [252, 504, 756]

HORIZONS = [20, 126, 252]

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


def make_model(params):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    **params,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def load_split(horizon, split):
    path = DATA_ROOT / f"{horizon}d" / f"{split}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    target_col = f"Target_Annualized_RV_{horizon}D"

    required = {"Date", "Target_Date", target_col}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    df["Date"] = pd.to_datetime(df["Date"])
    df["Target_Date"] = pd.to_datetime(df["Target_Date"])

    return df, target_col


def get_features(df, target_col):
    excluded = {
        "Date",
        "Target_Date",
        target_col,
    }

    candidates = [
        c for c in df.columns
        if c not in excluded
    ]

    features = (
        df[candidates]
        .select_dtypes(include=[np.number])
        .columns.tolist()
    )

    if not features:
        raise ValueError("No numeric features found.")

    return features


def evaluate(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted),
        "Bias": float(
            np.mean(
                np.asarray(predicted) -
                np.asarray(actual)
            )
        ),
    }


def main():

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_metrics = []
    all_predictions = []

    for horizon in HORIZONS:

        print("\n" + "=" * 65)
        print(f"{horizon}D WINDOW SENSITIVITY EXPERIMENT")
        print("=" * 65)

        train, target_col = load_split(
            horizon, "train"
        )

        validation, _ = load_split(
            horizon, "validation"
        )

        test, _ = load_split(
            horizon, "test"
        )

        # Development pool uses Train + Validation.
        development = pd.concat(
            [train, validation],
            ignore_index=True,
        )

        development = (
            development
            .sort_values("Date")
            .reset_index(drop=True)
        )

        test = (
            test
            .sort_values("Date")
            .reset_index(drop=True)
        )

        features = get_features(
            development,
            target_col,
        )

        X_test = test[features]
        y_test = test[target_col].astype(float)

        params = BEST_PARAMS[horizon]

        strategies = {
            "Expanding_Window": development,
        }

        # Only include windows that fit within the
        # available development sample.
        for window in ROLLING_WINDOWS:

            if window <= len(development):

                strategies[
                    f"Rolling_Window_{window}D"
                ] = development.tail(window)

            else:

                print(
                    f"Skipping Rolling {window}D: "
                    f"only {len(development)} "
                    f"development observations available."
                )

        # Train and evaluate each strategy.
        for strategy_name, fit_data in strategies.items():

            X_fit = fit_data[features]
            y_fit = fit_data[target_col].astype(float)

            model = make_model(params)

            model.fit(X_fit, y_fit)

            predictions = model.predict(X_test)

            metrics = evaluate(
                y_test,
                predictions,
            )

            metrics.update({
                "Horizon_Days": horizon,
                "Strategy": strategy_name,
                "Train_Rows": len(fit_data),
                "Test_Rows": len(test),
                "Train_Start": (
                    fit_data["Date"]
                    .min()
                    .date()
                    .isoformat()
                ),
                "Train_End": (
                    fit_data["Date"]
                    .max()
                    .date()
                    .isoformat()
                ),
                "Test_Start": (
                    test["Date"]
                    .min()
                    .date()
                    .isoformat()
                ),
                "Test_End": (
                    test["Date"]
                    .max()
                    .date()
                    .isoformat()
                ),
            })

            all_metrics.append(metrics)

            pred_df = pd.DataFrame({
                "Date": test["Date"],
                "Target_Date": test["Target_Date"],
                "Actual": y_test,
                "Prediction": predictions,
                "Horizon_Days": horizon,
                "Strategy": strategy_name,
            })

            all_predictions.append(pred_df)

            print(
                f"{strategy_name}: "
                f"MAE={metrics['MAE']:.4f}, "
                f"RMSE={metrics['RMSE']:.4f}, "
                f"R2={metrics['R2']:.4f}, "
                f"Bias={metrics['Bias']:.4f}, "
                f"Train={len(fit_data)}"
            )

        # Add the same historical baseline used in
        # the original window comparison.
        baseline_path = (
            DATA_ROOT
            / "model_results"
            / f"{horizon}d"
            / "test_predictions.csv"
        )

        if baseline_path.exists():

            baseline_df = pd.read_csv(
                baseline_path
            )

            baseline_col = (
                "Historical_60D_volatility"
            )

            if baseline_col in baseline_df.columns:

                baseline_pred = (
                    baseline_df[baseline_col]
                    .astype(float)
                )

                if len(baseline_pred) == len(y_test):

                    metrics = evaluate(
                        y_test,
                        baseline_pred,
                    )

                    metrics.update({
                        "Horizon_Days": horizon,
                        "Strategy": (
                            "Historical_60D_Baseline"
                        ),
                        "Train_Rows": np.nan,
                        "Test_Rows": len(test),
                        "Train_Start": "",
                        "Train_End": "",
                        "Test_Start": (
                            test["Date"]
                            .min()
                            .date()
                            .isoformat()
                        ),
                        "Test_End": (
                            test["Date"]
                            .max()
                            .date()
                            .isoformat()
                        ),
                    })

                    all_metrics.append(metrics)

                    all_predictions.append(
                        pd.DataFrame({
                            "Date": test["Date"],
                            "Target_Date": test["Target_Date"],
                            "Actual": y_test,
                            "Prediction": baseline_pred,
                            "Horizon_Days": horizon,
                            "Strategy": (
                                "Historical_60D_Baseline"
                            ),
                        })
                    )

                else:
                    print(
                        "Warning: baseline length "
                        "does not match test length."
                    )

            else:
                print(
                    f"Warning: baseline column "
                    f"{baseline_col} not found."
                )

        else:
            print(
                f"Warning: baseline file not found: "
                f"{baseline_path}"
            )

    # Save results separately.
    metrics_df = pd.DataFrame(all_metrics)

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    metrics_path = (
        RESULT_ROOT /
        "window_sensitivity_metrics.csv"
    )

    predictions_path = (
        RESULT_ROOT /
        "window_sensitivity_predictions.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    print("\n" + "=" * 65)
    print("FINAL WINDOW SENSITIVITY RESULTS")
    print("=" * 65)

    print(
        metrics_df[
            [
                "Horizon_Days",
                "Strategy",
                "MAE",
                "RMSE",
                "R2",
                "Bias",
                "Train_Rows",
            ]
        ]
        .sort_values(
            ["Horizon_Days", "MAE"]
        )
        .round(4)
        .to_string(index=False)
    )

    print("\nSaved metrics:")
    print(metrics_path)

    print("\nSaved predictions:")
    print(predictions_path)

    print(
        "\nOriginal window_comparison files "
        "were not modified."
    )


if __name__ == "__main__":
    main()
