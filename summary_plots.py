import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Variable that represents critical value for J (at T = 1)
Jc = 0.5 * np.log(1 + np.sqrt(2))

s = pd.read_csv("scan_summary.csv")
g = s.groupby('J').agg(rate= ("event_rate", "mean"), rate_se= ("event_rate", "sem")
                       , abs_m= ("mean_abs_m", "mean"), absm_se= ("mean_abs_m", "sem")).reset_index()


fig, ax = plt.subplots(1, 2 ,figsize=(10,3.5))

# Plotting for event rate graph
ax[0].errorbar(g['J'], g["rate"], g["rate_se"], fmt= "o-", label= "Simulation: Event rate")
ax[0].set_xlabel("J (T= 1)")
ax[0].set_ylabel("Event rate")

ax[0].axvline(
    Jc, color="grey", linestyle=":",
    label=fr"$J_c={Jc:.4f}$"
)

ax[0].legend(fontsize= 8)




# Plotting for mean |m| graph
ax[1].errorbar(g['J'], g["abs_m"], g["absm_se"], fmt= "o-", label= "Simulation: mean |m|")
ax[1].set_xlabel("J (T= 1)")
ax[1].set_ylabel("mean |m|")

# (1 - sinh(2x) **-4)**0.0125 represents Yang's derivation for exact spontaneous magnetisation (being non-zero above the criticacl coupling)
x = np.linspace(Jc + 1e-6, g['J'].max(), 200)
ax[1].plot(x, (1 - np.sinh(2 * x) ** -4) ** 0.125, "--", color= "orange", label= "Yang's Exact Solution ((1 - sinh(2x)^(-4))**0.0125)")

# Line that indicates the position of the critical coupling point
ax[1].axvline(
    Jc, color="grey", linestyle=":",
    label=fr"$J_c={Jc:.4f}$"
)

ax[1].legend(fontsize=8)

ax[0].axvline(
    Jc, color="grey", linestyle=":",
    label=fr"$J_c={Jc:.4f}$"
)


plt.tight_layout()
plt.savefig("fig_scan.png", dpi= 150)
