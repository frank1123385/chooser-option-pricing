"""
Week 6 revised experiment: forecast future annualized realized volatility.
Run from the repository root:
    python src/model/volatility_model_tuning.py

Reads:
    data/processed/volatility_experiment/{20d,126d,252d}/{train,validation,test}.csv

Writes to:
    data/processed/volatility_experiment/model_results/

Uses chronological TimeSeriesSplit with a horizon-sized gap to reduce
overlap leakage between forward-looking volatility labels.
"""
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "processed" / "volatility_experiment"
OUT_ROOT = DATA_ROOT / "model_results"
HORIZONS = (20, 126, 252)
RANDOM_STATE = 42

EXCLUDE_EXACT = {"Date", "Target_Date"}
PRICE_FEATURES = {"Close", "Adj Close"}

def load_data(horizon, split):
    path = DATA_ROOT / f"{horizon}d" / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}. Run volatility_feature_engineering.py first.")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Target_Date"] = pd.to_datetime(df["Target_Date"], errors="coerce")
    return df.sort_values("Date").reset_index(drop=True)

def get_features(train):
    cols = []
    for col in train.columns:
        if col in EXCLUDE_EXACT or col.startswith("Target_"):
            continue
        if pd.api.types.is_numeric_dtype(train[col]):
            cols.append(col)
    if not cols:
        raise ValueError("No numeric predictor columns found.")
    return cols

def xy(df, features, target):
    x = df[features].replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(df[target], errors="coerce")
    good = y.notna() & np.isfinite(y)
    return x.loc[good].copy(), y.loc[good].astype(float).copy()

def score(y, pred):
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
    }

def historical_volatility_by_date():
    # Compute from the full historical series, then align by date. The rolling
    # window is backward-looking, so validation/test rows can use prior history
    # without being penalized for having a fresh split boundary.
    source = pd.read_csv(ROOT / "data" / "processed" / "jpm_features.csv")
    source["Date"] = pd.to_datetime(source["Date"], errors="coerce")
    source = source.dropna(subset=["Date"]).sort_values("Date")
    returns = pd.to_numeric(source["Daily_Return"], errors="coerce")
    source["Historical_60D_Annualized_Vol"] = (
        returns.rolling(60, min_periods=20).std(ddof=1) * np.sqrt(252)
    )
    return source.set_index("Date")["Historical_60D_Annualized_Vol"]

def make_cv(n_rows, horizon):
    # Two folds with a horizon-sized embargo. A 60-row validation block keeps
    # the 252-day case feasible with the available training history.
    test_size = min(120, max(30, n_rows // 6))
    while n_rows - 2 * test_size - horizon < 30 and test_size > 30:
        test_size -= 5
    if n_rows - 2 * test_size - horizon < 30:
        raise ValueError(
            f"Insufficient training rows ({n_rows}) for {horizon}-day purged CV. "
            "Do not remove the gap; obtain more historical data or use a documented alternative."
        )
    return TimeSeriesSplit(n_splits=2, test_size=test_size, gap=horizon)

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    all_params = []
    historical_vol = historical_volatility_by_date()

    for horizon in HORIZONS:
        target = f"Target_Annualized_RV_{horizon}D"
        print(f"\n{'=' * 68}\nHORIZON: {horizon} trading days | TARGET: {target}")
        train = load_data(horizon, "train")
        val = load_data(horizon, "validation")
        test = load_data(horizon, "test")

        features = get_features(train)
        x_train, y_train = xy(train, features, target)
        x_val, y_val = xy(val, features, target)
        x_test, y_test = xy(test, features, target)
        print(f"Rows: train={len(x_train)}, validation={len(x_val)}, test={len(x_test)}")

        cv = make_cv(len(x_train), horizon)
        print(f"Purged time-series CV: n_splits=2, test_size={cv.test_size}, gap={horizon}")

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
        ])
        grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5, None],
            "model__min_samples_leaf": [5, 10],
        }
        search = GridSearchCV(
            pipe, grid, scoring="neg_mean_absolute_error",
            cv=cv, n_jobs=-1, refit=True, error_score="raise",
        )
        search.fit(x_train, y_train)
        best = search.best_estimator_
        print("Best CV parameters:", search.best_params_)
        print(f"Best CV MAE: {-search.best_score_:.6f}")

        predictions = {
            "Training-mean baseline": np.full(len(y_val), y_train.mean()),
        }
        # Historical volatility uses only data available at each feature date.
        predictions["Historical 60D volatility"] = val["Date"].map(historical_vol).to_numpy()
        predictions["Tuned Random Forest"] = best.predict(x_val)
        split_rows = []
        for model_name, pred in predictions.items():
            valid = np.isfinite(pred)
            if valid.sum() < 2:
                continue
            split_rows.append({
                "Horizon_Days": horizon, "Split": "Validation",
                "Model": model_name, **score(y_val.iloc[np.where(valid)[0]], pred[valid]),
            })

        test_predictions = {
            "Training-mean baseline": np.full(len(y_test), y_train.mean()),
            "Historical 60D volatility": test["Date"].map(historical_vol).to_numpy(),
            "Tuned Random Forest": best.predict(x_test),
        }
        for model_name, pred in test_predictions.items():
            valid = np.isfinite(pred)
            if valid.sum() < 2:
                continue
            split_rows.append({
                "Horizon_Days": horizon, "Split": "Test",
                "Model": model_name, **score(y_test.iloc[np.where(valid)[0]], pred[valid]),
            })

        # In-sample training score is diagnostic only, not a fair generalization estimate.
        train_pred = best.predict(x_train)
        split_rows.append({
            "Horizon_Days": horizon, "Split": "Train (in-sample diagnostic)",
            "Model": "Tuned Random Forest", **score(y_train, train_pred),
        })

        horizon_out = OUT_ROOT / f"{horizon}d"
        horizon_out.mkdir(parents=True, exist_ok=True)
        joblib.dump(best, horizon_out / f"random_forest_{horizon}d.joblib")
        pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(
            horizon_out / "cv_results.csv", index=False
        )
        pd.DataFrame(split_rows).to_csv(horizon_out / "metrics.csv", index=False)

        pred_frame = pd.DataFrame({
            "Date": test.loc[y_test.index, "Date"].values,
            "Target_Date": test.loc[y_test.index, "Target_Date"].values,
            "Actual": y_test.to_numpy(),
        })
        for name, pred in test_predictions.items():
            pred_frame[name.replace(" ", "_")] = pred
        pred_frame.to_csv(horizon_out / "test_predictions.csv", index=False)

        rf = best.named_steps["model"]
        importance = pd.DataFrame({
            "Feature": features, "Importance": rf.feature_importances_,
        }).sort_values("Importance", ascending=False)
        importance.to_csv(horizon_out / "feature_importance.csv", index=False)

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
        print(pd.DataFrame(split_rows).to_string(index=False))
        print(f"Saved model and results: {horizon_out}")

    pd.DataFrame(all_metrics).to_csv(OUT_ROOT / "all_horizons_metrics.csv", index=False)
    pd.DataFrame(all_params).to_csv(OUT_ROOT / "tuning_summary.csv", index=False)
    print(f"\nAll volatility-model experiments completed: {OUT_ROOT}")
    print("Caution: these are volatility forecasts, not yet Chooser option price evaluations.")

if __name__ == "__main__":
    main()
