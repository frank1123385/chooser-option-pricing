from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.dummy import DummyRegressor
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "week5"
OUTPUT_DIR = DATA_DIR
TARGET = "Target_Next_Day_Abs_Return"

DROP_COLUMNS = [
    "Date",
    TARGET,
]


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    for df in (train, validation, test):
        df["Date"] = pd.to_datetime(df["Date"])

    feature_columns = [
        col for col in train.columns
        if col not in DROP_COLUMNS
    ]

    return train, validation, test, feature_columns


def evaluate(model, X, y):
    predictions = model.predict(X)
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))

    return predictions, mae, rmse


def main():
    print("Loading Week 5 datasets...")
    train, validation, test, feature_columns = load_data()

    X_train = train[feature_columns]
    y_train = train[TARGET]

    X_validation = validation[feature_columns]
    y_validation = validation[TARGET]

    X_test = test[feature_columns]
    y_test = test[TARGET]

    # Baseline: predict the training-set mean.
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    _, baseline_mae, baseline_rmse = evaluate(
        baseline, X_test, y_test
    )

    candidates = [
        {"n_estimators": 100, "max_depth": 5, "min_samples_leaf": 5},
        {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 5},
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 10},
    ]

    best_model = None
    best_config = None
    best_validation_mae = float("inf")

    print("\nSelecting model using validation data...")
    for config in candidates:
        model = RandomForestRegressor(
            **config,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        _, val_mae, val_rmse = evaluate(
            model, X_validation, y_validation
        )

        print(
            f"Config={config} | "
            f"Validation MAE={val_mae:.6f} | "
            f"Validation RMSE={val_rmse:.6f}"
        )

        if val_mae < best_validation_mae:
            best_validation_mae = val_mae
            best_model = model
            best_config = config

    print("\nEvaluating selected model on test data...")
    test_predictions, test_mae, test_rmse = evaluate(
        best_model, X_test, y_test
    )

    results = pd.DataFrame([
        {
            "Model": "Training-mean baseline",
            "Test_MAE": baseline_mae,
            "Test_RMSE": baseline_rmse,
        },
        {
            "Model": "Random Forest",
            "Test_MAE": test_mae,
            "Test_RMSE": test_rmse,
        },
    ])

    results.to_csv(
        OUTPUT_DIR / "random_forest_results.csv",
        index=False,
    )

    predictions_df = test[["Date"]].copy()
    predictions_df["Actual_Next_Day_Abs_Return"] = y_test.to_numpy()
    predictions_df["Predicted_Next_Day_Abs_Return"] = test_predictions
    predictions_df.to_csv(
        OUTPUT_DIR / "random_forest_test_predictions.csv",
        index=False,
    )

    joblib.dump(
        best_model,
        OUTPUT_DIR / "random_forest_volatility_model.joblib",
    )

    importance = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": best_model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    importance.to_csv(
        OUTPUT_DIR / "random_forest_feature_importance.csv",
        index=False,
    )

    print("\nSelected configuration:")
    print(best_config)

    print("\nTest results:")
    print(results.to_string(index=False))

    print("\nTop 10 features:")
    print(importance.head(10).to_string(index=False))

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()