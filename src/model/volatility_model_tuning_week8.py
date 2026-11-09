"""Week 8: Train and evaluate future volatility forecasting models."""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)


# ==================================
# Paths and configuration
# ==================================

ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    ROOT / "data" / "processed" / "volatility_experiment_week8"
)

FEATURE_FILE = (
    ROOT / "data" / "processed" / "jpm_features_week8.csv"
)

OUT_ROOT = DATA_ROOT / "model_results"

HORIZONS = (20, 126, 252)

RANDOM_STATE = 42


# ==================================
# Exact Week 8 predictors
# ==================================

FEATURES = [
    "Close",
    "Volume",
    "DGS10",
    "VIXCLS",
    "Close_outlier",
    "Volume_outlier",
    "DGS10_outlier",
    "VIXCLS_outlier",
    "Dividend_Growth",
    "Daily_Return",
    "Rolling_Volatility_5D",
    "Rolling_Volatility_20D",
    "Rolling_Volatility_60D",
    "Price_Momentum_20D",
    "Volume_Change",
    "VIX_JPM_Correlation_20D",
    "Interest_Rate_Momentum_20D",
    "Sentiment_Score",
    "VIX_Change_20D",
    "Treasury_Yield_Change_20D",
]


# ==================================
# Load data
# ==================================

def load_data(horizon, split):

    path = DATA_ROOT / f"{horizon}d" / f"{split}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing input: {path}. "
            "Run volatility_feature_engineering_week8.py first."
        )

    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(
        df["Date"], errors="coerce"
    )

    df["Target_Date"] = pd.to_datetime(
        df["Target_Date"], errors="coerce"
    )

    return (
        df.dropna(subset=["Date", "Target_Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )


# ==================================
# Prepare X and y
# ==================================

def xy(df, target):

    missing = [
        col for col in FEATURES
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing predictors: {missing}"
        )

    x = df[FEATURES].copy()

    x = x.replace(
        [np.inf, -np.inf], np.nan
    )

    for col in FEATURES:
        x[col] = pd.to_numeric(
            x[col], errors="coerce"
        )

    y = pd.to_numeric(
        df[target], errors="coerce"
    )

    good = y.notna() & np.isfinite(y)

    return (
        x.loc[good].copy(),
        y.loc[good].astype(float).copy(),
    )


# ==================================
# Evaluation metrics
# ==================================

def score(y, pred):

    return {
        "MAE": float(
            mean_absolute_error(y, pred)
        ),
        "RMSE": float(
            np.sqrt(mean_squared_error(y, pred))
        ),
        "R2": float(
            r2_score(y, pred)
        ) if len(y) > 1 else np.nan,
    }


def add_metrics(
    rows, horizon, split, model_name, actual, prediction
):

    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    valid = (
        np.isfinite(actual)
        & np.isfinite(prediction)
    )

    if valid.sum() < 2:
        print(
            f"Warning: insufficient valid predictions "
            f"for {model_name}, {split}."
        )
        return

    rows.append({
        "Horizon_Days": horizon,
        "Split": split,
        "Model": model_name,
        **score(actual[valid], prediction[valid]),
        "Valid_Rows": int(valid.sum()),
    })


# ==================================
# Historical volatility benchmark
# ==================================

def historical_volatility_by_date():

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Missing Week 8 features: {FEATURE_FILE}"
        )

    source = pd.read_csv(FEATURE_FILE)

    source["Date"] = pd.to_datetime(
        source["Date"], errors="coerce"
    )

    source = (
        source.dropna(subset=["Date"])
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
    )

    returns = pd.to_numeric(
        source["Daily_Return"], errors="coerce"
    )

    # Backward-looking 60-day historical volatility.
    source["Historical_60D_Annualized_Vol"] = (
        returns.rolling(
            60, min_periods=20
        ).std(ddof=1) * np.sqrt(252)
    )

    return (
        source.set_index("Date")[
            "Historical_60D_Annualized_Vol"
        ]
    )


# ==================================
# Purged time-series cross-validation
# ==================================

def make_cv(n_rows, horizon):

    test_size = min(
        120,
        max(30, n_rows // 6)
    )

    while (
        n_rows - 2 * test_size - horizon < 30
        and test_size > 30
    ):
        test_size -= 5

    if n_rows - 2 * test_size - horizon < 30:

        raise ValueError(
            f"Insufficient training rows ({n_rows}) "
            f"for {horizon}-day purged CV."
        )

    return TimeSeriesSplit(
        n_splits=2,
        test_size=test_size,
        gap=horizon,
    )


# ==================================
# Main experiment
# ==================================

def main():

    OUT_ROOT.mkdir(
        parents=True, exist_ok=True
    )

    all_metrics = []
    all_params = []

    historical_vol = historical_volatility_by_date()

    print("========== WEEK 8 MODEL TRAINING ==========")
    print("Data directory:", DATA_ROOT)
    print("Output directory:", OUT_ROOT)
    print("Predictor count:", len(FEATURES))

    for horizon in HORIZONS:

        target = f"Target_Annualized_RV_{horizon}D"

        print(
            f"\n{'=' * 68}\n"
            f"HORIZON: {horizon} trading days"
        )

        # Load chronological splits
        train = load_data(horizon, "train")
        val = load_data(horizon, "validation")
        test = load_data(horizon, "test")

        x_train, y_train = xy(train, target)
        x_val, y_val = xy(val, target)
        x_test, y_test = xy(test, target)

        print(
            f"Rows: train={len(x_train)}, "
            f"validation={len(x_val)}, "
            f"test={len(x_test)}"
        )

        if len(x_train) != len(y_train):
            raise ValueError("Training X/y mismatch.")

        # Purged chronological CV
        cv = make_cv(len(x_train), horizon)

        print(
            f"Purged CV: n_splits=2, "
            f"test_size={cv.test_size}, "
            f"gap={horizon}"
        )

        # Pipeline
        pipe = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                RandomForestRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ])

        # Hyperparameter grid
        grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5, None],
            "model__min_samples_leaf": [5, 10],
        }

        search = GridSearchCV(
            pipe,
            grid,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )

        # Fit using training data only
        search.fit(x_train, y_train)

        best = search.best_estimator_

        print("\nBest parameters:")
        print(search.best_params_)

        print(
            f"Best CV MAE: {-search.best_score_:.6f}"
        )

        # ==================================
        # Validation predictions
        # ==================================

        val_rf = best.predict(x_val)

        val_mean = np.full(
            len(y_val), y_train.mean()
        )

        val_hist = (
            val["Date"]
            .map(historical_vol)
            .to_numpy(dtype=float)
        )

        split_rows = []

        add_metrics(
            split_rows,
            horizon,
            "Validation",
            "Training-mean baseline",
            y_val,
            val_mean,
        )

        add_metrics(
            split_rows,
            horizon,
            "Validation",
            "Historical 60D volatility",
            y_val,
            val_hist,
        )

        add_metrics(
            split_rows,
            horizon,
            "Validation",
            "Tuned Random Forest",
            y_val,
            val_rf,
        )

        val_frame = pd.DataFrame({
            "Date": val["Date"].to_numpy(),
            "Target_Date": val["Target_Date"].to_numpy(),
            "Actual": y_val.to_numpy(),
            "Training-mean_baseline": val_mean,
            "Historical_60D_volatility": val_hist,
            "Tuned_Random_Forest": val_rf,
        })

        # ==================================
        # Test predictions
        # ==================================

        test_rf = best.predict(x_test)

        test_mean = np.full(
            len(y_test), y_train.mean()
        )

        test_hist = (
            test["Date"]
            .map(historical_vol)
            .to_numpy(dtype=float)
        )

        add_metrics(
            split_rows,
            horizon,
            "Test",
            "Training-mean baseline",
            y_test,
            test_mean,
        )

        add_metrics(
            split_rows,
            horizon,
            "Test",
            "Historical 60D volatility",
            y_test,
            test_hist,
        )

        add_metrics(
            split_rows,
            horizon,
            "Test",
            "Tuned Random Forest",
            y_test,
            test_rf,
        )

        test_frame = pd.DataFrame({
            "Date": test["Date"].to_numpy(),
            "Target_Date": test["Target_Date"].to_numpy(),
            "Actual": y_test.to_numpy(),
            "Training-mean_baseline": test_mean,
            "Historical_60D_volatility": test_hist,
            "Tuned_Random_Forest": test_rf,
        })

        # ==================================
        # Training diagnostic
        # ==================================

        train_rf = best.predict(x_train)

        add_metrics(
            split_rows,
            horizon,
            "Train (in-sample diagnostic)",
            "Tuned Random Forest",
            y_train,
            train_rf,
        )

        # ==================================
        # Save outputs
        # ==================================

        horizon_out = OUT_ROOT / f"{horizon}d"

        horizon_out.mkdir(
            parents=True, exist_ok=True
        )

        joblib.dump(
            best,
            horizon_out /
            f"random_forest_{horizon}d_week8.joblib",
        )

        pd.DataFrame(
            search.cv_results_
        ).sort_values(
            "rank_test_score"
        ).to_csv(
            horizon_out / "cv_results.csv",
            index=False,
        )

        metrics_df = pd.DataFrame(split_rows)

        metrics_df.to_csv(
            horizon_out / "metrics.csv",
            index=False,
        )

        val_frame.to_csv(
            horizon_out / "validation_predictions.csv",
            index=False,
        )

        test_frame.to_csv(
            horizon_out / "test_predictions.csv",
            index=False,
        )

        # Feature importance
        rf = best.named_steps["model"]

        importance = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": rf.feature_importances_,
        }).sort_values(
            "Importance", ascending=False
        )

        importance.to_csv(
            horizon_out / "feature_importance.csv",
            index=False,
        )

        all_metrics.extend(split_rows)

        all_params.append({
            "Horizon_Days": horizon,
            "Best_CV_MAE": -search.best_score_,
            "Best_Parameters": str(search.best_params_),
            "Train_Rows": len(x_train),
            "Validation_Rows": len(x_val),
            "Test_Rows": len(x_test),
            "CV_Gap": horizon,
            "CV_Test_Size": cv.test_size,
        })

        print("\nModel evaluation:")
        print(metrics_df.to_string(index=False))

        print("\nTop 10 feature importance:")
        print(importance.head(10).to_string(index=False))

        print("\nSaved:", horizon_out)

    # ==================================
    # Combined summaries
    # ==================================

    pd.DataFrame(all_metrics).to_csv(
        OUT_ROOT / "all_horizons_metrics.csv",
        index=False,
    )

    pd.DataFrame(all_params).to_csv(
        OUT_ROOT / "tuning_summary.csv",
        index=False,
    )

    print("\n========== ALL EXPERIMENTS COMPLETED ==========")
    print("Output:", OUT_ROOT)
    print("Horizons:", HORIZONS)
    print("Predictors:", len(FEATURES))
    print("Week 7 models and results were preserved.")

    print(
        "\nCaution: these are volatility forecasts, "
        "not yet market-validated Chooser option prices."
    )


if __name__ == "__main__":
    main()
