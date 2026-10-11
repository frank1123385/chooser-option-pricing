import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

S0, R, SIGMA, K, T1, T2, SEED = 231.45, 0.0458, 0.3075, 150.0, 0.5, 1.0, 42
N_MAX = 20000
rng = np.random.default_rng(SEED)
z1 = rng.standard_normal(N_MAX)
s1 = S0 * np.exp((R - 0.5*SIGMA**2)*T1 + SIGMA*np.sqrt(T1)*z1)
choose_call = s1 > K
z2 = rng.standard_normal(N_MAX)
s2 = s1 * np.exp((R - 0.5*SIGMA**2)*(T2-T1) + SIGMA*np.sqrt(T2-T1)*z2)
payoff = np.where(choose_call, np.maximum(s2-K, 0), np.maximum(K-s2, 0))
discounted = payoff * np.exp(-R*T2)

rows = []
for n in [100,250,500,1000,2000,3000,5000,7500,10000,15000,20000]:
    x = discounted[:n]
    mean = x.mean()
    se = x.std(ddof=1)/np.sqrt(n)
    rows.append({"Simulations":n,"Chooser_Price":mean,"Standard_Error":se,
                 "CI_95_Lower":mean-1.96*se,"CI_95_Upper":mean+1.96*se,
                 "Call_Ratio":choose_call[:n].mean(),"Put_Ratio":1-choose_call[:n].mean()})
results = pd.DataFrame(rows)
Path("data/processed").mkdir(parents=True, exist_ok=True)
results.to_csv("data/processed/week4_mc_convergence.csv",index=False)
results[results.Simulations.isin([5000,10000,20000])].to_csv("data/processed/week4_mc_sample_size_comparison.csv",index=False)
plt.figure(figsize=(8,4.8)); plt.plot(results.Simulations,results.Chooser_Price,marker="o",label="Estimate")
plt.fill_between(results.Simulations,results.CI_95_Lower,results.CI_95_Upper,alpha=.2,label="95% CI")
plt.axhline(results.loc[results.Simulations==20000,"Chooser_Price"].iloc[0],linestyle="--",label="20,000-path estimate")
plt.xlabel("Number of Monte Carlo simulations"); plt.ylabel("Estimated Chooser Option Price (USD)")
plt.title("Monte Carlo Convergence with 95% Confidence Intervals"); plt.legend(); plt.tight_layout()
plt.savefig("data/processed/week4_mc_convergence_curve.png",dpi=180); plt.close()
plt.figure(figsize=(8,4.8)); plt.plot(results.Simulations,results.Standard_Error,marker="o")
plt.xlabel("Number of Monte Carlo simulations"); plt.ylabel("Estimated Standard Error (USD)")
plt.title("Monte Carlo Standard Error vs. Simulation Count"); plt.tight_layout()
plt.savefig("data/processed/week4_mc_standard_error.png",dpi=180); plt.close()
print(results[results.Simulations.isin([5000,10000,20000])].to_string(index=False))
