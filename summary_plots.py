import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

J, T = 1.0, 1.5
# Exact Ising |m| for J = 1, T = 1.5 (about 0.9865). It only applies at alpha = 0.
exact_alpha0 = (1 - np.sinh(2 * J / T) ** -4) ** 0.125

s = pd.read_csv("scan_summary.csv")


g = s.groupby("alpha").agg(rate=("event_rate", "mean"), rate_se=("event_rate", "sem"),
                           abs_m=("mean_abs_m", "mean"), absm_se=("mean_abs_m", "sem"),
                           std_m=("std_m", "mean"), stdm_se=("std_m", "sem")).reset_index()

fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))

# Panel 1: event rate
ax[0].errorbar(g["alpha"], g["rate"], g["rate_se"], fmt="o-", label="mean over seeds")
ax[0].scatter(s["alpha"], s["event_rate"], s=10, color="grey", alpha=0.4, label="individual seeds")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("Event rate")
ax[0].legend(fontsize=8)

# Panel 2: mean |m|
ax[1].errorbar(g["alpha"], g["abs_m"], g["absm_se"], fmt="o-", label="mean over seeds")
ax[1].scatter(s["alpha"], s["mean_abs_m"], s=10, color="grey", alpha=0.4, label="individual seeds")
ax[1].axhline(exact_alpha0, color="orange", linestyle="--",
              label=f"exact Ising value at alpha = 0 ({exact_alpha0:.4f})")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("mean |m|")
ax[1].legend(fontsize=8)

# Panel 3: standard deviation of m within a run
ax[2].errorbar(g["alpha"], g["std_m"], g["stdm_se"], fmt="o-", label="mean over seeds")
ax[2].scatter(s["alpha"], s["std_m"], s=10, color="grey", alpha=0.4, label="individual seeds")
ax[2].set_xlabel("alpha")
ax[2].set_ylabel("std of m within a run")
ax[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("fig_alpha_scan.png", dpi=150)