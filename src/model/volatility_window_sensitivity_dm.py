
"""
DM tests for volatility window sensitivity experiments.

Uses paired forecast errors and Newey-West HAC variance.
Applies HLN finite-sample correction and Holm adjustment.

Run from project root:
    python src/model/volatility_window_sensitivity_dm.py
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "volatility_experiment"
)

INPUT_FILE = (
    DATA_ROOT
    / "window_sensitivity"
    / "window_sensitivity_predictions.csv"
)

OUTPUT_DIR = DATA_ROOT / "window_sensitivity"

HORIZONS = [20, 126, 252]


def holm_adjust(p_values):
    """Holm step-down adjustment."""
    p_values = np.asarray(p_values, dtype=float)

    n = len(p_values)
    order = np.argsort(p_values)

    adjusted = np.empty(n)
    running_max = 0.0

    for rank, idx in enumerate(order):
        value = (n - rank) * p_values[idx]
        running_max = max(running_max, value)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted


def hac_long_run_variance(values, max_lag):
    """Bartlett/Newey-West long-run variance."""
    values = np.asarray(values, dtype=float)
    values = values - np.mean(values)

    n = len(values)

    gamma0 = np.dot(values, values) / n
    variance = gamma0

    for lag in range(1, max_lag + 1):
        gamma = (
            np.dot(values[lag:], values[:-lag]) / n
        )

        weight = 1.0 - lag / (max_lag + 1.0)

        variance += 2.0 * weight * gamma

    return float(variance)


def dm_test(loss_a, loss_b, horizon):
    """
    H0: equal expected loss.

    Positive mean difference means A has higher loss.
    """

    difference = (
        np.asarray(loss_a, dtype=float)
        - np.asarray(loss_b, dtype=float)
    )

    difference = difference[
        np.isfinite(difference)
    ]

    n = len(difference)

    if n < 10:
        return None

    max_lag = min(horizon - 1, n - 1)

    long_run_var = hac_long_run_variance(
        difference,
        max_lag,
    )

    if long_run_var <= 0:
        return None

    mean_difference = float(
        np.mean(difference)
    )

    dm_stat = (
        mean_difference
        / np.sqrt(long_run_var / n)
    )

    # Harvey-Leybourne-Newbold correction.
    hln_term = (
        n + 1
        - 2 * horizon
        + horizon * (horizon - 1) / n
    ) / n

    if hln_term <= 0:
        return {
            "N": n,
            "Mean_Loss_Difference": mean_difference,
            "DM_Statistic": dm_stat,
            "HLN_Statistic": np.nan,
            "P_Value": np.nan,
            "Max_HAC_Lag": max_lag,
            "HLN_Available": False,
        }

    hln_factor = np.sqrt(hln_term)

    hln_stat = dm_stat * hln_factor

    p_value = 2.0 * t.sf(
        abs(hln_stat),
        df=n - 1,
    )

    return {
        "N": n,
        "Mean_Loss_Difference": mean_difference,
        "DM_Statistic": dm_stat,
        "HLN_Statistic": hln_stat,
        "P_Value": p_value,
        "Max_HAC_Lag": max_lag,
        "HLN_Available": True,
    }


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "Date",
        "Target_Date",
        "Actual",
        "Prediction",
        "Horizon_Days",
        "Strategy",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df["Date"] = pd.to_datetime(df["Date"])
    df["Target_Date"] = pd.to_datetime(
        df["Target_Date"]
    )

    df["Actual"] = pd.to_numeric(
        df["Actual"],
        errors="coerce",
    )

    df["Prediction"] = pd.to_numeric(
        df["Prediction"],
        errors="coerce",
    )

    df["Horizon_Days"] = pd.to_numeric(
        df["Horizon_Days"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Actual",
            "Prediction",
            "Horizon_Days",
        ]
    )

    df["Absolute_Loss"] = abs(
        df["Actual"] - df["Prediction"]
    )

    df["Squared_Loss"] = (
        df["Actual"] - df["Prediction"]
    ) ** 2

    results = []

    for horizon in HORIZONS:

        subset = df[
            df["Horizon_Days"] == horizon
        ]

        strategies = sorted(
            subset["Strategy"].unique()
        )

        print(
            f"\n========== {horizon}D DM TESTS =========="
        )

        for strategy_a, strategy_b in combinations(
            strategies,
            2,
        ):

            a = subset[
                subset["Strategy"] == strategy_a
            ][
                [
                    "Date",
                    "Target_Date",
                    "Actual",
                    "Absolute_Loss",
                    "Squared_Loss",
                ]
            ]

            b = subset[
                subset["Strategy"] == strategy_b
            ][
                [
                    "Date",
                    "Target_Date",
                    "Actual",
                    "Absolute_Loss",
                    "Squared_Loss",
                ]
            ]

            # Match identical forecast dates and targets.
            paired = a.merge(
                b,
                on=["Date", "Target_Date"],
                suffixes=("_A", "_B"),
                validate="one_to_one",
            )

            if len(paired) == 0:
                continue

            # Confirm actual targets match.
            if not np.allclose(
                paired["Actual_A"],
                paired["Actual_B"],
                equal_nan=True,
            ):
                raise ValueError(
                    "Actual targets do not match "
                    f"for {strategy_a} vs {strategy_b}."
                )

            for loss_name, col_a, col_b in [
                (
                    "Absolute",
                    "Absolute_Loss_A",
                    "Absolute_Loss_B",
                ),
                (
                    "Squared",
                    "Squared_Loss_A",
                    "Squared_Loss_B",
                ),
            ]:

                test_result = dm_test(
                    paired[col_a],
                    paired[col_b],
                    horizon,
                )

                if test_result is None:
                    continue

                record = {
                    "Horizon_Days": horizon,
                    "Loss": loss_name,
                    "Model_A": strategy_a,
                    "Model_B": strategy_b,
                    **test_result,
                }

                results.append(record)

                print(
                    f"{loss_name}: "
                    f"A={strategy_a} vs "
                    f"B={strategy_b}, "
                    f"N={test_result['N']}, "
                    f"MeanDiff="
                    f"{test_result['Mean_Loss_Difference']:.6f}, "
                    f"HLN="
                    f"{test_result['HLN_Statistic']}, "
                    f"p="
                    f"{test_result['P_Value']}"
                )

    results_df = pd.DataFrame(results)

    if results_df.empty:
        raise RuntimeError(
            "No valid DM comparisons were produced."
        )

    results_df["Holm_P_Value"] = np.nan

    # Holm adjustment across all comparisons
    # within each loss type.
    for loss_name in results_df["Loss"].unique():

        mask = (
            (results_df["Loss"] == loss_name)
            & results_df["P_Value"].notna()
        )

        results_df.loc[
            mask,
            "Holm_P_Value",
        ] = holm_adjust(
            results_df.loc[
                mask,
                "P_Value",
            ].values
        )

    results_df["Significant_5pct"] = (
        results_df["Holm_P_Value"] < 0.05
    )

    results_df.to_csv(
        OUTPUT_DIR / "window_sensitivity_dm_results.csv",
        index=False,
    )

    summary_path = (
        OUTPUT_DIR / "window_sensitivity_dm_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "WINDOW SENSITIVITY DIEBOLD-MARIANO TESTS\n"
        )

        f.write(
            "H0: Two forecasting strategies have equal "
            "expected loss.\n"
        )

        f.write(
            "Positive A-minus-B loss difference means "
            "Model A has higher loss.\n"
        )

        f.write(
            "HAC: Bartlett/Newey-West, "
            "max lag = min(horizon - 1, N - 1).\n"
        )

        f.write(
            "HLN finite-sample correction applied "
            "when mathematically available.\n"
        )

        f.write(
            "Holm adjustment applied across all "
            "comparisons within each loss type.\n\n"
        )

        for _, row in results_df.iterrows():

            f.write(
                f"{int(row['Horizon_Days'])}D | "
                f"{row['Loss']} | "
                f"A={row['Model_A']} vs "
                f"B={row['Model_B']} | "
                f"N={int(row['N'])} | "
                f"Mean diff="
                f"{row['Mean_Loss_Difference']:.6f} | "
                f"HLN={row['HLN_Statistic']} | "
                f"p={row['P_Value']} | "
                f"Holm p={row['Holm_P_Value']} | "
                f"Significant="
                f"{row['Significant_5pct']}\n"
            )

        f.write(
            "\nCAUTION:\n"
            "Forward realized-volatility targets overlap. "
            "Loss differentials can be serially correlated. "
            "Long-horizon results, especially 252D with "
            "only 126 test observations, require caution.\n"
        )

    print("\n========== FINAL DM SUMMARY ==========")

    print(
        results_df[
            [
                "Horizon_Days",
                "Loss",
                "Model_A",
                "Model_B",
                "N",
                "Mean_Loss_Difference",
                "P_Value",
                "Holm_P_Value",
                "Significant_5pct",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    print("\nSaved:")
    print(
        OUTPUT_DIR
        / "window_sensitivity_dm_results.csv"
    )

    print(summary_path)


if __name__ == "__main__":
    main()
