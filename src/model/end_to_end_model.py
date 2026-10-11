from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "week5"


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    for df in (train, validation, test):
        df["Date"] = pd.to_datetime(df["Date"])
        df["Target_Next_Day_Return"] = df["Daily_Return"].shift(-1)

    train = train.dropna(subset=["Target_Next_Day_Return"]).copy()
    validation = validation.dropna(subset=["Target_Next_Day_Return"]).copy()
    test = test.dropna(subset=["Target_Next_Day_Return"]).copy()

    excluded = {
        "Date",
        "Target_Next_Day_Abs_Return",
        "Target_Next_Day_Return",
        "Daily_Return",
        "Close",
        "Adj Close",
    }
    features = [c for c in train.columns if c not in excluded]

    return train, validation, test, features


def price_metrics(predicted_returns, actual_returns, current_close):
    predicted_prices = current_close * (1 + predicted_returns)
    actual_prices = current_close * (1 + actual_returns)

    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))

    return predicted_prices, mae, rmse


def main():
    train, validation, test, features = load_data()

    X_train = train[features]
    y_train = train["Target_Next_Day_Return"]

    X_validation = validation[features]
    y_validation = validation["Target_Next_Day_Return"]

    X_test = test[features]
    y_test = test["Target_Next_Day_Return"]
    current_close = test["Close"].to_numpy()

    models = {
        "Zero-return baseline": DummyRegressor(
            strategy="constant", constant=0
        ),
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }

    validation_scores = {}

    print("Validation results:")
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_predictions = model.predict(X_validation)
        _, val_mae, val_rmse = price_metrics(
            val_predictions,
            y_validation.to_numpy(),
            validation["Close"].to_numpy(),
        )
        validation_scores[name] = val_mae

        print(
            f"{name}: MAE={val_mae:.4f}, RMSE={val_rmse:.4f}"
        )

    best_name = min(validation_scores, key=validation_scores.get)
    print(f"\nBest model selected on validation set: {best_name}")

    # Report all models on the test set, not only the selected model.
    test_results = []
    prediction_columns = {"Date": test["Date"].to_numpy()}

    for name, model in models.items():
        predicted_returns = model.predict(X_test)
        predicted_prices, mae, rmse = price_metrics(
            predicted_returns,
            y_test.to_numpy(),
            current_close,
        )

        test_results.append({
            "Model": name,
            "Test_MAE": mae,
            "Test_RMSE": rmse,
        })

        prediction_columns[f"{name}_Predicted_Return"] = predicted_returns
        prediction_columns[f"{name}_Predicted_Close"] = predicted_prices

    results = pd.DataFrame(test_results)
    results.to_csv(DATA_DIR / "end_to_end_results.csv", index=False)

    predictions = pd.DataFrame(prediction_columns)
    predictions["Current_Close"] = current_close
    predictions["Actual_Next_Day_Return"] = y_test.to_numpy()
    predictions["Actual_Next_Day_Close"] = (
        current_close * (1 + y_test.to_numpy())
    )
    predictions.to_csv(
        DATA_DIR / "end_to_end_test_predictions.csv",
        index=False,
    )

    print("\nTest results:")
    print(results.to_string(index=False))
    print(f"\nResults saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()