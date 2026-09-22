import numpy as np


def simulate_chooser_option(
    S0,
    risk_free_rate,
    sigma,
    strike,
    T1,
    T2,
    n_simulations=10000,
    seed=42,
):
    """
    Two-stage BSM Monte Carlo simulation for a European Chooser Option.

    Model structure follows:
    Huang, Wang & Wan (2021),
    "Exploration of JPMorgan Chooser Option Pricing".

    Parameters
    ----------
    S0 : float
        Initial stock price.
    risk_free_rate : float
        Risk-free interest rate, expressed as a decimal.
    sigma : float
        Annualized volatility, expressed as a decimal.
    strike : float
        Strike price.
    T1 : float
        Decision date in years.
    T2 : float
        Final maturity in years.
    n_simulations : int
        Number of Monte Carlo simulations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Simulation results and estimated chooser option value.
    """

    if T1 <= 0:
        raise ValueError("T1 must be greater than 0.")

    if T2 <= T1:
        raise ValueError("T2 must be greater than T1.")

    if S0 <= 0:
        raise ValueError("S0 must be greater than 0.")

    if sigma < 0:
        raise ValueError("sigma cannot be negative.")

    if strike <= 0:
        raise ValueError("strike must be greater than 0.")

    rng = np.random.default_rng(seed)

    # ==============================
    # First period: 0 -> T1
    # ==============================

    dt1 = T1

    z1 = rng.standard_normal(n_simulations)

    S1 = S0 * np.exp(
        (
            risk_free_rate
            - 0.5 * sigma ** 2
        ) * dt1
        + sigma * np.sqrt(dt1) * z1
    )

    # Decision at T1:
    # S1 > K -> Call
    # S1 <= K -> Put

    choice = np.where(
        S1 > strike,
        "CALL",
        "PUT"
    )

    # ==============================
    # Second period: T1 -> T2
    # ==============================

    dt2 = T2 - T1

    z2 = rng.standard_normal(n_simulations)

    S2 = S1 * np.exp(
        (
            risk_free_rate
            - 0.5 * sigma ** 2
        ) * dt2
        + sigma * np.sqrt(dt2) * z2
    )

    # ==============================
    # Final payoff
    # ==============================

    call_payoff = np.maximum(S2 - strike, 0)

    put_payoff = np.maximum(strike - S2, 0)

    payoff = np.where(
        choice == "CALL",
        call_payoff,
        put_payoff
    )

    # ==============================
    # Discount payoff to t = 0
    # ==============================

    discounted_payoff = (
        payoff
        * np.exp(-risk_free_rate * T2)
    )

    chooser_price = np.mean(discounted_payoff)

    # ==============================
    # Store simulation results
    # ==============================

    results = {
        "S1": S1,
        "Choice": choice,
        "S2": S2,
        "Payoff": payoff,
        "Discounted_Payoff": discounted_payoff,
        "Chooser_Price": chooser_price,
    }

    return results


# ==========================================
# Paper baseline parameters
# ==========================================

if __name__ == "__main__":

    S0 = 156.7
    risk_free_rate = 0.0015
    sigma = 0.282
    strike = 150
    T1 = 0.5
    T2 = 1.0

    results = simulate_chooser_option(
        S0=S0,
        risk_free_rate=risk_free_rate,
        sigma=sigma,
        strike=strike,
        T1=T1,
        T2=T2,
        n_simulations=10000,
        seed=42,
    )

    print("=" * 50)
    print("BSM Chooser Option Pricing Model")
    print("=" * 50)

    print(f"Initial Stock Price (S0): ${S0:.2f}")
    print(f"Risk-free Rate: {risk_free_rate:.2%}")
    print(f"Volatility (sigma): {sigma:.2%}")
    print(f"Strike Price (K): ${strike:.2f}")
    print(f"Decision Date (T1): {T1:.2f} years")
    print(f"Maturity (T2): {T2:.2f} years")

    print("-" * 50)

    call_count = np.sum(results["Choice"] == "CALL")
    put_count = np.sum(results["Choice"] == "PUT")

    print(f"Call Choices: {call_count}")
    print(f"Put Choices: {put_count}")

    print(
        f"Estimated Chooser Option Price: "
        f"${results['Chooser_Price']:.4f}"
    )

    print("=" * 50)