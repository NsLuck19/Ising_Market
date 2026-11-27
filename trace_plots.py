import os
import matplotlib.pyplot as plt
import pandas as pd

# (alpha as written in the file name, seed). If other plots are needed, i can just change/add any (alpha,seed) combos when needed
runs = [('0.000000', 11037), ('2.400000', 11037), ('2.800000', 11037),
        ('2.800000', 11041), ('5.000000', 11037)]

fig, axes = plt.subplots(len(runs), 1, sharex=True, figsize=(9, 2 * len(runs)))
for ax, (a, seed) in zip(axes, runs):
    path = f'series/alpha{a}_seed{seed}.csv'
    if not os.path.exists(path):
        print('missing file:', path)
        continue
    df = pd.read_csv(path)
    ax.plot(df['t'], df['m'], lw=0.5)
    ax.set_ylabel(f'alpha={float(a):g}, seed {seed}')
    ax.set_ylim(-1, 1)
axes[-1].set_xlabel('sweep')
plt.tight_layout()
plt.savefig('fig_m_traces.png', dpi=150)