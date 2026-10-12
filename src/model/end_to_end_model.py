
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
TARGET = "Target_Next_Day_Return"
PRICE_COL = "Adj Close"

FEATURES = [
    "Volume", "DGS10", "VIXCLS",
    "Close_outlier", "Adj Close_outlier",
    "Volume_outlier", "DGS10_outlier", "VIXCLS_outlier",
    "Dividend_Growth",
    "Rolling_Volatility_5D", "Rolling_Volatility_20D",
    "Rolling_Volatility_60D", "Price_Momentum_20D",
    "Volume_Change", "VIX_JPM_Correlation_20D",
    "Interest_Rate_Momentum_20D", "Sentiment_Score",
    "VIX_Change_20D", "Treasury_Yield_Change_20D",
]


def load_data():
    datasets = {}
    for name in ("train", "validation", "test"):
        path = DATA_DIR / f"{name}.csv"
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])

        required = FEATURES + [TARGET, PRICE_COL]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{name}.csv is missing columns: {missing}")

        datasets[name] = df.sort_values("Date").reset_index(drop=True)

    return datasets["train"], datasets["validation"], datasets["test"]


def return_metrics(actual, predicted):
    return {
        "Return_MAE": mean_absolute_error(actual, predicted),
        "Return_RMSE": np.sqrt(mean_squared_error(actual, predicted)),
    }


def price_metrics(actual_returns, predicted_returns, current_prices):
    actual_prices = current_prices * (1 + actual_returns)
    predicted_prices = current_prices * (1 + predicted_returns)

    return {
        "Price_MAE": mean_absolute_error(actual_prices, predicted_prices),
        "Price_RMSE": np.sqrt(
            mean_squared_error(actual_prices, predicted_prices)
        ),
    }


def main():
    train, validation, test = load_data()

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_val = validation[FEATURES]
    y_val = validation[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

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

    validation_rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        current_prices = validation[PRICE_COL].to_numpy()

        metrics = return_metrics(y_val, pred)
        metrics.update(price_metrics(
            y_val.to_numpy(), pred, current_prices
        ))
        validation_rows.append({"Model": name, **metrics})

    validation_results = pd.DataFrame(validation_rows)
    print("\nValidation results (adjusted-close price basis):")
    print(validation_results.to_string(index=False))

    best_name = validation_results.sort_values(
        "Price_MAE"
    ).iloc[0]["Model"]
    print(f"\nBest model selected on validation set: {best_name}")

    test_rows = []
    prediction_output = pd.DataFrame({
        "Date": test["Date"].dt.strftime("%Y-%m-%d"),
        "Current_Adj_Close": test[PRICE_COL].to_numpy(),
        "Actual_Next_Day_Return": y_test.to_numpy(),
        "Actual_Next_Day_Adj_Close": (
            test[PRICE_COL].to_numpy() * (1 + y_test.to_numpy())
        ),
    })

    current_prices = test[PRICE_COL].to_numpy()

    for name, model in models.items():
        pred = model.predict(X_test)
        metrics = return_metrics(y_test, pred)
        metrics.update(price_metrics(
            y_test.to_numpy(), pred, current_prices
        ))
        test_rows.append({"Model": name, **metrics})

        prediction_output[f"{name}_Predicted_Return"] = pred
        prediction_output[f"{name}_Predicted_Adj_Close"] = (
            current_prices * (1 + pred)
        )

    test_results = pd.DataFrame(test_rows)
    print("\nTest results (adjusted-close price basis):")
    print(test_results.to_string(index=False))

    validation_results.to_csv(
        DATA_DIR / "end_to_end_validation_results.csv", index=False
    )
    test_results.to_csv(
        DATA_DIR / "end_to_end_results.csv", index=False
    )
    prediction_output.to_csv(
        DATA_DIR / "end_to_end_test_predictions.csv", index=False
    )

    print(f"\nResults saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()
