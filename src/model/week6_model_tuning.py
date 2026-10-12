"""
Week 6: time-series tuning and model comparison.
Run from the repository root:
    python src/model/week6_model_tuning.py

Uses the Week 5 chronological train/validation/test CSV files.
The test set is used only for final evaluation, not hyperparameter selection.
"""
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "week5"
OUT_DIR = ROOT / "data" / "processed" / "week6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_RETURN = "Target_Next_Day_Return"
TARGET_ABS_RETURN = "Target_Next_Day_Abs_Return"
PRICE_COL = "Adj Close"
NON_FEATURE_COLS = {
    "Date", "Target_Date", TARGET_RETURN, TARGET_ABS_RETURN,
    "Target_Next_Day_Price", "Target_Next_Day_Abs_Return",
}

def load_split(name):
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required dataset not found: {path}")
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)
    return df

def get_features(df, exclude_prices=False):
    cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    if exclude_prices:
        cols = [c for c in cols if c not in {"Close", "Adj Close"}]
    if not cols:
        raise ValueError("No numeric feature columns were found.")
    return cols

def clean_xy(df, features, target):
    if target not in df.columns:
        raise KeyError(
            f"Target column '{target}' is missing. Available columns: {list(df.columns)}"
        )
    x = df[features].replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.notna()
    x = x.loc[valid].copy()
    y = y.loc[valid].astype(float)
    # Imputation is fitted inside the model pipeline on training folds only.
    return x, y

def metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }

def save_predictions(test, y_true, predictions, filename, target_name):
    out = pd.DataFrame()
    if "Date" in test.columns:
        out["Date"] = test.loc[y_true.index, "Date"].values
    if "Target_Date" in test.columns:
        out["Target_Date"] = test.loc[y_true.index, "Target_Date"].values
    out["Actual"] = y_true.to_numpy()
    for model_name, pred in predictions.items():
        out[model_name.replace(" ", "_")] = np.asarray(pred)
    out.to_csv(OUT_DIR / filename, index=False)

def run_approach_a(train, val, test):
    target = TARGET_ABS_RETURN
    features = get_features(train)
    x_train, y_train = clean_xy(train, features, target)
    x_val, y_val = clean_xy(val, features, target)
    x_test, y_test = clean_xy(test, features, target)

    # Hyperparameter selection uses training-only expanding-window CV.
    cv = TimeSeriesSplit(n_splits=3)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])
    grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [5, None],
        "model__min_samples_leaf": [5, 10],
    }
    search = GridSearchCV(
        pipe, grid, scoring="neg_mean_absolute_error", cv=cv,
        n_jobs=-1, refit=True, error_score="raise"
    )
    search.fit(x_train, y_train)
    best_model = search.best_estimator_

    # Validation is reported for transparency; test is final, untouched evaluation.
    pred_val = np.maximum(best_model.predict(x_val), 0)
    pred_test = np.maximum(best_model.predict(x_test), 0)
    baseline_val = np.full(len(y_val), y_train.mean())
    baseline_test = np.full(len(y_test), y_train.mean())

    rows = []
    for split_name, actual, pred_base, pred_rf in [
        ("Validation", y_val, baseline_val, pred_val),
        ("Test", y_test, baseline_test, pred_test),
    ]:
        for name, pred in [
            ("Training-mean baseline", pred_base),
            ("Tuned Random Forest", pred_rf),
        ]:
            row = {"Approach": "A - Absolute-return proxy", "Split": split_name,
                   "Model": name, **metrics(actual, pred)}
            rows.append(row)

    joblib.dump(best_model, OUT_DIR / "week6_random_forest_volatility.joblib")
    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(
        OUT_DIR / "approach_a_cv_results.csv", index=False
    )
    pd.DataFrame(rows).to_csv(OUT_DIR / "approach_a_metrics.csv", index=False)
    save_predictions(test, y_test, {"Training-mean baseline": baseline_test,
                                    "Tuned Random Forest": pred_test},
                     "approach_a_test_predictions.csv", target)

    # Tree-based impurity importance, a complementary summary to SHAP.
    rf = best_model.named_steps["model"]
    imp = pd.DataFrame({"Feature": features, "Importance": rf.feature_importances_})
    imp.sort_values("Importance", ascending=False).to_csv(
        OUT_DIR / "approach_a_feature_importance.csv", index=False
    )
    return best_model, features, rows, search.best_params_

def run_approach_b(train, val, test):
    target = TARGET_RETURN
    features = get_features(train, exclude_prices=True)
    x_train, y_train = clean_xy(train, features, target)
    x_val, y_val = clean_xy(val, features, target)
    x_test, y_test = clean_xy(test, features, target)

    cv = TimeSeriesSplit(n_splits=3)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])
    grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5, None],
        "model__min_samples_leaf": [5, 10],
    }
    search = GridSearchCV(
        pipe, grid, scoring="neg_mean_absolute_error", cv=cv,
        n_jobs=-1, refit=True, error_score="raise"
    )
    search.fit(x_train, y_train)
    tuned_rf = search.best_estimator_

    # Linear regression uses its own preprocessing pipeline.
    linear = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])
    linear.fit(x_train, y_train)

    # A zero-return baseline predicts no change.
    predictions = {
        "Zero-return baseline": np.zeros(len(y_test)),
        "Linear Regression": linear.predict(x_test),
        "Tuned Random Forest": tuned_rf.predict(x_test),
    }
    val_predictions = {
        "Zero-return baseline": np.zeros(len(y_val)),
        "Linear Regression": linear.predict(x_val),
        "Tuned Random Forest": tuned_rf.predict(x_val),
    }

    def price_metrics(frame, actual_returns, pred_returns, indices):
        if PRICE_COL not in frame.columns:
            return {"Price_MAE": np.nan, "Price_RMSE": np.nan}
        current_price = pd.to_numeric(frame.loc[indices, PRICE_COL], errors="coerce").to_numpy()
        actual_price = current_price * (1 + actual_returns.to_numpy())
        predicted_price = current_price * (1 + np.asarray(pred_returns))
        valid = np.isfinite(actual_price) & np.isfinite(predicted_price)
        if not valid.any():
            return {"Price_MAE": np.nan, "Price_RMSE": np.nan}
        return {
            "Price_MAE": float(mean_absolute_error(actual_price[valid], predicted_price[valid])),
            "Price_RMSE": float(np.sqrt(mean_squared_error(actual_price[valid], predicted_price[valid]))),
        }

    rows = []
    for split_name, frame, actual, preds in [
        ("Validation", val, y_val, val_predictions),
        ("Test", test, y_test, predictions),
    ]:
        for name, pred in preds.items():
            # Price conversion is based on adjusted close, aligned to the feature row.
            base = metrics(actual, pred)
            base.update(price_metrics(frame, actual, pred, actual.index))
            rows.append({"Approach": "B - Next-day return", "Split": split_name,
                         "Model": name, **base})

    joblib.dump(tuned_rf, OUT_DIR / "week6_end_to_end_random_forest.joblib")
    joblib.dump(linear, OUT_DIR / "week6_end_to_end_linear_regression.joblib")
    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(
        OUT_DIR / "approach_b_cv_results.csv", index=False
    )
    pd.DataFrame(rows).to_csv(OUT_DIR / "approach_b_metrics.csv", index=False)
    save_predictions(test, y_test, predictions, "approach_b_test_predictions.csv", target)

    rf = tuned_rf.named_steps["model"]
    imp = pd.DataFrame({"Feature": features, "Importance": rf.feature_importances_})
    imp.sort_values("Importance", ascending=False).to_csv(
        OUT_DIR / "approach_b_feature_importance.csv", index=False
    )
    return tuned_rf, features, rows, search.best_params_

def main():
    print("Loading Week 5 chronological splits...")
    train, val, test = load_split("train"), load_split("validation"), load_split("test")
    print(f"Rows: train={len(train)}, validation={len(val)}, test={len(test)}")

    print("\nApproach A: tuning Random Forest for next-day absolute return...")
    model_a, features_a, rows_a, params_a = run_approach_a(train, val, test)
    print("Best CV parameters:", params_a)
    print(pd.DataFrame(rows_a).to_string(index=False))

    print("\nApproach B: tuning Random Forest for next-day return...")
    model_b, features_b, rows_b, params_b = run_approach_b(train, val, test)
    print("Best CV parameters:", params_b)
    print(pd.DataFrame(rows_b).to_string(index=False))

    all_rows = rows_a + rows_b
    pd.DataFrame(all_rows).to_csv(OUT_DIR / "week6_model_comparison.csv", index=False)
    print(f"\nWeek 6 outputs saved to: {OUT_DIR}")
    print("Note: These are stock-return/price proxy experiments, not actual Chooser option market-price evaluations.")

if __name__ == "__main__":
    main()
