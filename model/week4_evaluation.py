import pandas as pd
import numpy as np


INPUT_FILE = "data/processed/week4_bsm_predictions.csv"
OUTPUT_FILE = "data/processed/week4_evaluation.csv"


# =========================
# Load BSM predictions
# =========================

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

print("=" * 60)
print("Week 4 BSM Performance Evaluation")
print("=" * 60)

print(f"Number of BSM predictions: {len(df)}")


# =========================
# Check actual CME price
# =========================

if "Actual_CME_Price" not in df.columns:

    print("\nActual CME transaction price is not available.")

    print(
        "MAE and RMSE cannot be calculated "
        "until actual CME prices are provided."
    )

    print("\nCurrent columns:")

    print(df.columns.tolist())

    print("\nEvaluation framework is ready.")

else:

    # =========================
    # Calculate prediction errors
    # =========================

    df["Error"] = (
        df["BSM_Predicted_Price"]
        - df["Actual_CME_Price"]
    )

    df["Absolute_Error"] = df["Error"].abs()

    df["Squared_Error"] = df["Error"] ** 2


    # =========================
    # Calculate MAE / RMSE
    # =========================

    mae = df["Absolute_Error"].mean()

    rmse = np.sqrt(
        df["Squared_Error"].mean()
    )


    # =========================
    # Save evaluation data
    # =========================

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # =========================
    # Results
    # =========================

    print(
        f"\nMAE: {mae:.6f}"
    )

    print(
        f"RMSE: {rmse:.6f}"
    )

    print(
        f"\nEvaluation file saved to:"
    )

    print(OUTPUT_FILE)


print("=" * 60)