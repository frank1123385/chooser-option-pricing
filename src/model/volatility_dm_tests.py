"""
Diebold-Mariano tests for volatility forecast comparisons.

Run from the project root:
    python src/model/volatility_dm_tests.py

Input:
    data/processed/volatility_experiment/window_comparison/
        window_comparison_predictions.csv

Outputs:
    data/processed/volatility_experiment/window_comparison/
        dm_test_results.csv
        dm_test_summary.txt

The script compares paired out-of-sample forecasts using absolute-error
and squared-error loss differentials. A Bartlett/Newey-West HAC variance
estimator uses max lag = min(horizon - 1, n - 1), reflecting overlapping
future-volatility targets. Holm adjustment is applied across all tests
within each loss definition.
"""

from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = (
    ROOT / "data" / "processed" / "volatility_experiment" / "window_comparison"
)
INPUT_FILE = RESULT_DIR / "window_comparison_predictions.csv"
OUTPUT_CSV = RESULT_DIR / "dm_test_results.csv"
OUTPUT_TXT = RESULT_DIR / "dm_test_summary.txt"

MODEL_ORDER = [
    "Historical_60D_Baseline",
    "Expanding_Window",
    "Rolling_Window_252D",
]


def hac_long_run_variance(x, max_lag):
    """Bartlett/Newey-West estimate of the long-run variance of x."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    if n < 3:
        raise ValueError("At least 3 paired forecast errors are required.")

    centered = x - x.mean()
    gamma0 = np.dot(centered, centered) / n
    long_run_var = gamma0

    for lag in range(1, min(max_lag, n - 1) + 1):
        gamma = np.dot(centered[lag:], centered[:-lag]) / n
        weight = 1.0 - lag / (max_lag + 1.0)
        long_run_var += 2.0 * weight * gamma

    return max(float(long_run_var), 0.0)


def dm_test(actual, forecast_a, forecast_b, horizon, loss_name):
    """
    H0: equal expected loss.
    Positive mean differential means forecast A has higher loss (is worse).
    """
    actual = np.asarray(actual, dtype=float)
    forecast_a = np.asarray(forecast_a, dtype=float)
    forecast_b = np.asarray(forecast_b, dtype=float)

    if not (len(actual) == len(forecast_a) == len(forecast_b)):
        raise ValueError("Actuals and forecasts must have equal lengths.")

    error_a = forecast_a - actual
    error_b = forecast_b - actual

    if loss_name == "Absolute":
        loss_a = np.abs(error_a)
        loss_b = np.abs(error_b)
    elif loss_name == "Squared":
        loss_a = error_a ** 2
        loss_b = error_b ** 2
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

    # Positive differential: A has larger loss than B.
    d = loss_a - loss_b
    n = len(d)
    max_lag = min(int(horizon) - 1, n - 1)

    lr_var = hac_long_run_variance(d, max_lag)
    if lr_var <= 1e-15:
        dm_stat = 0.0 if abs(d.mean()) <= 1e-15 else np.sign(d.mean()) * np.inf
    else:
        dm_stat = d.mean() / math.sqrt(lr_var / n)

    # Harvey-Leybourne-Newbold finite-sample correction.
    correction_term = (n + 1 - 2 * horizon + (horizon * (horizon - 1)) / n) / n
    hln_factor = math.sqrt(max(correction_term, 0.0))
    dm_hln = dm_stat * hln_factor

    # Use t(n-1) reference distribution as a finite-sample approximation.
    if np.isfinite(dm_hln):
        p_value = 2.0 * student_t.sf(abs(dm_hln), df=n - 1)
    else:
        p_value = 0.0

    return {
        "N": n,
        "Mean_Loss_Diff_A_minus_B": float(d.mean()),
        "DM_Statistic_HAC": float(dm_stat),
        "HLN_Corrected_Statistic": float(dm_hln),
        "P_Value": float(p_value),
        "HAC_Max_Lag": max_lag,
        "HAC_Long_Run_Variance": float(lr_var),
    }


def holm_adjust(p_values):
    """Holm-Bonferroni adjusted p-values, preserving original order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0

    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max

    return adjusted


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE}. Run volatility_window_comparison.py first."
        )

    raw = pd.read_csv(INPUT_FILE)
    required = {
        "Date", "Actual", "Prediction", "Horizon_Days", "Strategy"
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")

    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["Strategy"] = raw["Strategy"].astype(str)

    rows = []
    warnings_list = []

    for horizon in [20, 126, 252]:
        part = raw[raw["Horizon_Days"] == horizon].copy()
        wide_pred = part.pivot_table(
            index=["Date"], columns="Strategy", values="Prediction", aggfunc="first"
        )
        wide_actual = part.groupby("Date")["Actual"].first().rename("Actual")
        wide = wide_actual.to_frame().join(wide_pred, how="inner").sort_index()

        available = [m for m in MODEL_ORDER if m in wide.columns]
        if len(available) < 2:
            warnings_list.append(
                f"{horizon}D: fewer than two forecast strategies available."
            )
            continue

        # Check actual values are consistent across strategy rows.
        actual_counts = part.groupby("Date")["Actual"].nunique()
        if (actual_counts > 1).any():
            raise ValueError(
                f"{horizon}D: Actual values differ across strategies for a date."
            )

        comparisons = [
            ("Historical_60D_Baseline", "Expanding_Window"),
            ("Historical_60D_Baseline", "Rolling_Window_252D"),
            ("Expanding_Window", "Rolling_Window_252D"),
        ]

        for model_a, model_b in comparisons:
            if model_a not in wide.columns or model_b not in wide.columns:
                continue

            pair = wide[["Actual", model_a, model_b]].dropna()
            for loss_name in ["Absolute", "Squared"]:
                result = dm_test(
                    pair["Actual"],
                    pair[model_a],
                    pair[model_b],
                    horizon,
                    loss_name,
                )
                result.update(
                    {
                        "Horizon_Days": horizon,
                        "Loss": loss_name,
                        "Model_A": model_a,
                        "Model_B": model_b,
                        "Interpretation": (
                            "A worse than B"
                            if result["Mean_Loss_Diff_A_minus_B"] > 0
                            else "A better than B"
                            if result["Mean_Loss_Diff_A_minus_B"] < 0
                            else "Equal mean loss"
                        ),
                    }
                )
                rows.append(result)

        if horizon == 252 and len(wide) <= 130:
            warnings_list.append(
                "252D: the test sample is short relative to the 251-lag "
                "overlap structure; interpret DM p-values cautiously."
            )

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No pairwise DM tests could be calculated.")

    # Holm correction across all pairwise comparisons within each loss type.
    results["Holm_Adjusted_P"] = np.nan
    for loss_name, idx in results.groupby("Loss").groups.items():
        results.loc[idx, "Holm_Adjusted_P"] = holm_adjust(
            results.loc[idx, "P_Value"].values
        )

    results["Significant_5pct_Holm"] = results["Holm_Adjusted_P"] < 0.05
    results.to_csv(OUTPUT_CSV, index=False)

    lines = [
        "DIEBOLD-MARIANO FORECAST COMPARISON",
        "Null hypothesis: the two forecasts have equal expected loss.",
        "Positive A-minus-B loss difference means Model A has higher loss.",
        "HAC: Bartlett/Newey-West; max lag = min(horizon - 1, N - 1).",
        "Finite-sample adjustment: Harvey-Leybourne-Newbold (HLN).",
        "Multiple testing: Holm adjustment across all comparisons within each loss.",
        "",
    ]

    for _, row in results.iterrows():
        lines.append(
            f"{int(row['Horizon_Days'])}D | {row['Loss']} loss | "
            f"A={row['Model_A']} vs B={row['Model_B']} | "
            f"N={int(row['N'])} | mean diff={row['Mean_Loss_Diff_A_minus_B']:.6f} | "
            f"DM-HLN={row['HLN_Corrected_Statistic']:.4f} | "
            f"p={row['P_Value']:.6f} | Holm p={row['Holm_Adjusted_P']:.6f} | "
            f"significant={bool(row['Significant_5pct_Holm'])}"
        )

    if warnings_list:
        lines.extend(["", "CAUTIONS:"])
        lines.extend(f"- {item}" for item in warnings_list)

    lines.extend(
        [
            "",
            "Interpretation note:",
            "These are approximate finite-sample DM tests. The 126D and especially",
            "252D targets are overlapping forward-window volatility measures, so",
            "loss differentials can be strongly serially correlated. The 252D test",
            "has very few out-of-sample observations relative to its overlap horizon;",
            "do not treat a non-significant or significant p-value as definitive.",
            "Statistical significance does not establish economic significance.",
        ]
    )

    OUTPUT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("\n========== DIEBOLD-MARIANO TEST RESULTS ==========")
    display_cols = [
        "Horizon_Days", "Loss", "Model_A", "Model_B", "N",
        "Mean_Loss_Diff_A_minus_B", "HLN_Corrected_Statistic",
        "P_Value", "Holm_Adjusted_P", "Significant_5pct_Holm",
    ]
    print(results[display_cols].round(6).to_string(index=False))

    if warnings_list:
        print("\nCAUTIONS:")
        for item in warnings_list:
            print("-", item)

    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
