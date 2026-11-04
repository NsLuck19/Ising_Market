import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

s = pd.read_csv("scan_summary.csv")
g = s.groupby('J').agg(rate= ("event_rate", "mean"), rate_se= ("event_rate", "sem")
                       , abs_m= ("mean_abs_m", "mean"), absm_se= ("mean_abs_m", "sem")).reset_index()


fig, ax = plt.subplots(1, 2 ,figsize=(10,3.5))

# Plotting for event rate graph
ax[0].errorbar(g['J'], g["rate"], g["rate_se"], fmt= "o-")
ax[0].set_xlabel("J (T= 1)")
ax[0].set_ylabel("Event rate")

# Plotting for mean |m| graph
ax[1].errorbar(g['J'], g["abs_m"], g["absm_se"], fmt= "o-")
ax[1].set_xlabel("J (T= 1)")
ax[1].set_ylabel("mean |m|")

# Variable that represents critical value for J (at T = 1)
Jc = 0.5 * np.log(1 + np.sqrt(2))

# (1 - sinh(2x) **-4)**0.0125 represents Yang's derivation for exact spontaneous magnetisation (being non-zero above the criticacl coupling)
x = np.linspace(Jc + 1e-6, g['J'].max(), 200)
ax[1].plot(x, (1 - np.sinh(2 * x) ** -4) ** 0.125, "--")


plt.tight_layout()
plt.savefig("fig_scan.png", dpi= 150)
