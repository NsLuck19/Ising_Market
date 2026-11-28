"""
Ising extension: does herding in an Ising-type market produce Hawkes self-excitation?

Simulates a 2D Ising / Bornholdt (2001) market of L x L traders, s_i = +1 (buy) / -1 (sell),
with heat-bath dynamics and local field
    h_i = J * (sum of the 4 neighbours) - alpha * s_i * |m|,     m = mean spin,
so alpha = 0 is the plain Ising model and alpha > 0 adds Bornholdt's global coupling.
Each sweep updates the two checkerboard sublattices in turn (a standard, vectorised
heat-bath scheme), so the script needs only numpy, scipy and matplotlib.

Events: sweeps where |m(t) - m(t-1)| exceeds 1.96 standard deviations of that run's own
changes. Event times are whole sweeps, so each is spread uniformly within its sweep, then a
Hawkes process is fitted with fit_hawkes() from hawkes.py.

Put this file and hawkes.py in the same folder (e.g. the Ising_Market repository), then:

    python ising_hawkes_extension.py                  # alpha scan, Bornholdt values J=1, T=1.5
    python ising_hawkes_extension.py --quick          # fast check (~1 min)
    python ising_hawkes_extension.py --mode ising     # plain Ising: scan J at T=1, alpha=0
    python ising_hawkes_extension.py --mode repo      # fit the repo's existing events/*.txt files
                                                      # (looks in ./events, then ../events;
                                                      #  or give --events-dir PATH)

Outputs: extension_results.csv and fig_extension.png (fitted n against the scanned coupling).
"""
import argparse
import glob
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hawkes import fit_hawkes, ks_test


def simulate(L, J, alpha, T, n_sweeps, n_burn, seed, ordered_start=True):
    """Return the magnetisation after each recorded sweep."""
    rng = np.random.default_rng(seed)
    s = np.ones((L, L), dtype=np.int8) if ordered_start else rng.choice(np.array([-1, 1], dtype=np.int8), (L, L))
    ii, jj = np.indices((L, L))
    masks = [(ii + jj) % 2 == 0, (ii + jj) % 2 == 1]          # checkerboard sublattices
    m_series = np.empty(n_sweeps)
    beta = 1.0 / T
    for sweep in range(n_burn + n_sweeps):
        for mask in masks:
            nb = (np.roll(s, 1, 0) + np.roll(s, -1, 0) + np.roll(s, 1, 1) + np.roll(s, -1, 1))
            m_abs = abs(s.mean())
            h = J * nb - alpha * s * m_abs
            p_up = 1.0 / (1.0 + np.exp(-2.0 * beta * h))
            new = np.where(rng.random((L, L)) < p_up, 1, -1).astype(np.int8)
            s = np.where(mask, new, s)
        if sweep >= n_burn:
            m_series[sweep - n_burn] = s.mean()
    return m_series


def events_from_m(m, k=1.96):
    d = np.diff(m)
    return np.where(np.abs(d) > k * d.std())[0] + 1.0       # sweep index after the change


def fit_events(ev, window, rng):
    if len(ev) < 15:
        return None
    t = np.sort(ev - rng.uniform(0, 1, len(ev)))            # spread each event within its sweep
    f = fit_hawkes(t, window)
    f["ks_p"] = ks_test(f, t)
    return f


def run_scan(values, param, args):
    rng = np.random.default_rng(0)
    rows = []
    for v in values:
        for seed in range(args.seeds):
            J, alpha, T = (args.J, v, args.T) if param == "alpha" else (v, 0.0, 1.0)
            m = simulate(args.L, J, alpha, T, args.sweeps, args.burn, 1000 + seed)
            ev = events_from_m(m)
            f = fit_events(ev, float(len(m)), rng)
            row = {param: v, "seed": seed, "events": len(ev), "mean_abs_m": np.abs(m).mean(),
                   "n": f["n"] if f else np.nan, "decay_time": 1 / f["beta"] if f else np.nan,
                   "ks_p": f["ks_p"] if f else np.nan}
            rows.append(row)
            print(f"  {param} = {v:<6} seed {seed}: events = {len(ev):5d}, n = {row['n']:.2f}, "
                  f"decay time = {row['decay_time']:.1f} sweeps, KS p = {row['ks_p']:.2f}, "
                  f"mean |m| = {row['mean_abs_m']:.2f}")
    return pd.DataFrame(rows)


def run_repo(args):
    """Fit the repository's existing event files (J scan or alpha scan, read from the file names)."""
    import os
    folder = args.events_dir or next((d for d in ("events", os.path.join("..", "events"))
                                      if os.path.isdir(d)), None)
    if folder is None:
        raise SystemExit("No events folder found: run from the repository (or a subfolder of it), "
                         "or pass --events-dir PATH")
    print(f"Reading event files from {os.path.abspath(folder)}")
    rng = np.random.default_rng(0)
    rows, param = [], None
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        name = os.path.basename(path)
        match = re.match(r"(alpha|J)([\d.]+)_seed(\d+)", name)
        if not match:
            continue
        param = match.group(1)
        with open(path) as fh:
            window = float(re.search(r"Window=(\d+)", fh.read()).group(1))
        ev = np.atleast_1d(np.loadtxt(path, comments="#", ndmin=1))
        f = fit_events(ev, window, rng)
        row = {param: float(match.group(2)), "seed": int(match.group(3)), "events": len(ev),
               "n": f["n"] if f else np.nan, "decay_time": 1 / f["beta"] if f else np.nan,
               "ks_p": f["ks_p"] if f else np.nan}
        rows.append(row)
        print(f"  {param} = {row[param]:<6} seed {row['seed']}: events = {len(ev):5d}, n = {row['n']:.2f}, "
              f"decay time = {row['decay_time']:.1f} sweeps, KS p = {row['ks_p']:.2f}")
    df = pd.DataFrame(rows)
    print(df.groupby(param).agg(events=("events", "mean"), n_mean=("n", "mean"), n_sd=("n", "std"),
                                decay=("decay_time", "median"),
                                ks_pass=("ks_p", lambda p: (p > 0.05).mean())).round(3))
    return df, param


def plot(df, param, args):
    """Top: every run's fitted n with the median. Bottom: fraction of runs in the bursty
    state (n > 0.5), since near the transition runs split into n ~ 0 or n ~ 1, which a
    mean with error bars would hide."""
    df = df.dropna(subset=["n"])
    vals = np.sort(df[param].unique())
    jit = np.random.default_rng(1).uniform(-0.04, 0.04, len(df)) * (vals.max() - vals.min()) / 8
    fig, ax = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax[0].scatter(df[param] + jit, df["n"], s=16, alpha=0.6, color="C0", label="individual runs")
    ax[0].plot(vals, df.groupby(param)["n"].median().loc[vals], "-", color="C3", lw=1.2, label="median")
    ax[0].axhline(1, color="k", ls="--", lw=0.8)
    ax[0].set(ylabel="fitted branching ratio n", ylim=(-0.05, 1.05),
              title="Hawkes branching ratio of simulated market events")
    ax[0].legend(fontsize=8, loc="center right")
    frac = df.assign(bursty=df["n"] > 0.5).groupby(param)["bursty"].mean().loc[vals]
    ax[1].bar(vals, frac, width=0.6 * np.min(np.diff(vals)) if len(vals) > 1 else 0.5, color="C1")
    ax[1].set(ylabel="fraction of runs\nwith n > 0.5", ylim=(0, 1.05))
    if param == "J":
        Jc = 0.5 * np.log(1 + np.sqrt(2))
        for a_ in ax:
            a_.axvline(Jc, color="grey", ls=":")
        ax[1].set(xlabel=f"coupling J (T = 1, plain Ising; dotted line: $J_c$ = {Jc:.4f})")
    else:
        ax[1].set(xlabel=rf"global coupling $\alpha$ (J = {args.J}, T = {args.T})")
    plt.tight_layout()
    plt.savefig("fig_extension.png", dpi=200)
    print("Saved fig_extension.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bornholdt", "ising", "repo"], default="bornholdt")
    ap.add_argument("--quick", action="store_true", help="fewer values, seeds and sweeps")
    ap.add_argument("--L", type=int, default=100)
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--T", type=float, default=1.5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sweeps", type=int, default=20000)
    ap.add_argument("--burn", type=int, default=5000)
    ap.add_argument("--events-dir", default=None, help="folder of event files for --mode repo")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.sweeps, args.burn = 2, 10000, 2000

    if args.mode == "repo":
        df, param = run_repo(args)
    elif args.mode == "ising":
        vals = [0.0, 0.2, 0.3, 0.4, 0.43, 0.44, 0.45] if not args.quick else [0.0, 0.3, 0.44]
        df, param = run_scan(vals, "J", args), "J"
    else:
        vals = [0, 1, 2, 3, 4, 5, 6, 8] if not args.quick else [0, 2, 4, 6]
        df, param = run_scan(vals, "alpha", args), "alpha"
    df.to_csv("extension_results.csv", index=False)
    print("Saved extension_results.csv")
    plot(df, param, args)