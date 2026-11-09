
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

base = ROOT / "data" / "processed" / "volatility_experiment"

train_file = base / "126d" / "train.csv"
validation_file = base / "126d" / "validation.csv"
test_file = base / "126d" / "test.csv"

files = {
    "Train": train_file,
    "Validation": validation_file,
    "Test": test_file,
}

target = "Target_Annualized_RV_126D"

print("=" * 70)
print("126D TARGET DISTRIBUTION DIAGNOSTICS")
print("=" * 70)

for name, file in files.items():
    df = pd.read_csv(file)

    y = pd.to_numeric(df[target], errors="coerce").dropna()

    print(f"\n{name}")
    print("-" * 40)
    print("Rows:", len(y))
    print("Mean:", y.mean())
    print("Std:", y.std())
    print("Min:", y.min())
    print("25%:", y.quantile(0.25))
    print("Median:", y.median())
    print("75%:", y.quantile(0.75))
    print("Max:", y.max())

print("\n" + "=" * 70)
print("126D PREDICTION DISTRIBUTION")
print("=" * 70)

prediction_file = (
    base / "window_comparison" /
    "window_comparison_predictions.csv"
)

pred = pd.read_csv(prediction_file)

pred = pred[
    (pred["Horizon_Days"] == 126) &
    (pred["Strategy"] == "Expanding_Window")
].copy()

for col in ["Actual", "Prediction"]:
    pred[col] = pd.to_numeric(pred[col], errors="coerce")

print("\nActual:")
print(pred["Actual"].describe())

print("\nPrediction:")
print(pred["Prediction"].describe())

print("\nPrediction minus Actual:")
print((pred["Prediction"] - pred["Actual"]).describe())

print("\nNegative predictions:",
      (pred["Prediction"] < 0).sum())

print("\nPredictions above 100%:",
      (pred["Prediction"] > 1).sum())

print("\nDiagnostics completed.")
