import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def power_law(x, a, b):
    return a * np.power(x, b)

if __name__ == "__main__":
    file_path = "../data/isoflops_curves.json"
    with open(file_path, "r") as f:
        runs = json.load(f)
    
    os.makedirs("../results", exist_ok=True)

    df = pd.DataFrame(runs)
    run_opt = df.loc[df.groupby("compute_budget").final_loss.idxmin()]
    run_opt.sort_values("compute_budget", inplace=True)
    C_opt = run_opt["compute_budget"].values
    N_opt = run_opt["parameters"].values
    D_opt = C_opt / (6 * N_opt)
    
    popt_N, _ = curve_fit(power_law, C_opt, N_opt)
    a_N, b_N = popt_N
    
    popt_D, _ = curve_fit(power_law, C_opt, D_opt)
    a_D, b_D = popt_D
    
    C_pred = np.array([1e23, 1e24])
    N_pred = power_law(C_pred, *popt_N)
    D_pred = power_law(C_pred, *popt_D)
    
    print(f"Fitted N_opt = {a_N:.3e} * C^{b_N:.3f}")
    print(f"Predicted N_opt(1e23) = {N_pred[0]:.3e}")
    print(f"Predicted N_opt(1e24) = {N_pred[1]:.3e}\n")

    print(f"Fitted D_opt = {a_D:.3e} * C^{b_D:.3f}")
    print(f"Predicted D_opt(1e23) = {D_pred[0]:.3e}")
    print(f"Predicted D_opt(1e24) = {D_pred[1]:.3e}")
    
    plt.figure(figsize=(12, 5))
    plt.scatter(C_opt, N_opt, label = "data of N", color = "blue")
    plt.plot(C_opt, power_law(C_opt, *popt_N), color="blue", linestyle="--", label="fit N_opt")

    plt.scatter(C_opt, D_opt, label = "data of D", color = "orange")
    plt.plot(C_opt, power_law(C_opt, *popt_D), color="orange", linestyle="--", label="fit D_opt")
    
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Compute budget (FLOPs)")
    plt.ylabel("Model / Dataset size")
    plt.title("IsoFLOPs Scaling Laws (Chinchilla-style)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../results/isoflops_scaling.png", dpi=300)