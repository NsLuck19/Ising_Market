"""
Python version of main.cpp: 2D Ising market scan that writes event files and a summary.

Same model and outputs as main.cpp:
  - L x L traders (spins +1 buy / -1 sell), periodic boundaries, heat-bath dynamics,
    local field h_i = J * (sum of 4 neighbours) + h, T = 1 (k_B = 1);
  - threshold theta = 1.96 * sd of the per-sweep change in m from a J = 0 run;
  - for each coupling J and seed: random start, burn-in, record m after each sweep,
    events = sweeps where |m(t) - m(t-1)| > theta;
  - writes events/J<J>_seed<seed>.txt (same header and format) and a summary CSV.

Differences from main.cpp:
  - Each sweep updates the two checkerboard sublattices in turn (vectorised with numpy)
    instead of N random single-spin updates. Equilibrium behaviour is the same; the
    dynamics differ slightly, and random numbers differ, so results match statistically,
    not run for run.
  - std_m is the standard deviation of m about its own mean. (main.cpp passes mean |m|
    to standard_deviation(), which measures spread about mean |m| instead.)
  - Optional extras, both off by default so the output matches main.cpp:
      --alpha A        adds the Bornholdt (2001) global term: h_i -= A * s_i * |m|
      --own-threshold  sets theta from each run's own changes in m (1.96 sd)
  - Writes to events_py/ and scan_summary_py.csv by default, so the C++ results in
    events/ and scan_summary.csv are never overwritten.

Usage (needs numpy only):
    python main.py                 # full scan, about 10 minutes
    python main.py --quick         # 3 couplings, 2 seeds, short runs (~20 s)
    python main.py --alpha 4 --own-threshold --out events_bornholdt
"""
import argparse
import os

import numpy as np


def neighbour_sum(s):
    return np.roll(s, 1, 0) + np.roll(s, -1, 0) + np.roll(s, 1, 1) + np.roll(s, -1, 1)


def sweep(s, J, h, beta, alpha, masks, rng):
    """One heat-bath sweep: update each checkerboard sublattice in turn."""
    for mask in masks:
        local = J * neighbour_sum(s) + h - alpha * s * abs(s.mean())
        p_up = 1.0 / (1.0 + np.exp(-2.0 * beta * local))
        new = np.where(rng.random(s.shape) < p_up, 1, -1).astype(np.int8)
        s = np.where(mask, new, s)
    return s


def run(L, J, h, beta, alpha, n_burn, n_sweeps, rng, start="random"):
    """Burn in, then return the magnetisation after each of n_sweeps sweeps."""
    s = (np.ones((L, L), dtype=np.int8) if start == "up"
         else rng.choice(np.array([-1, 1], dtype=np.int8), (L, L)))
    ii, jj = np.indices((L, L))
    masks = [(ii + jj) % 2 == 0, (ii + jj) % 2 == 1]
    for _ in range(n_burn):
        s = sweep(s, J, h, beta, alpha, masks, rng)
    m = np.empty(n_sweeps)
    for t in range(n_sweeps):
        s = sweep(s, J, h, beta, alpha, masks, rng)
        m[t] = s.mean()
    return m


def extract_events(m, theta):
    """Index in m of the value after each change larger than theta (as in main.cpp)."""
    return np.where(np.abs(np.diff(m)) > theta)[0] + 1


def write_events(path, events, info):
    with open(path, "w") as f:
        f.write(f"# L= {info['L']} |J= {info['J']:g} |h= {info['h']:g} |T= {info['T']:g} "
                f"|Beta= {info['beta']:g} |Theta= {info['theta']:.8g} |Seed= {info['seed']} "
                f"|Number of burns= {info['n_burn']} |Number of sweeps= {info['n_sweeps']}"
                + (f" |Alpha= {info['alpha']:g}" if info["alpha"] else "") + "\n")
        f.write(f"# Window={info['n_sweeps']}\n")
        for e in events:
            f.write(f"{e}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.0, help="Bornholdt global coupling (0 = plain Ising)")
    ap.add_argument("--own-threshold", action="store_true", help="threshold from each run's own changes")
    ap.add_argument("--out", default="events_py", help="folder for event files")
    ap.add_argument("--summary", default="scan_summary_py.csv")
    args = ap.parse_args()

    L, h, T = 100, 0.0, 1.0
    beta = 1.0 / T
    base_seed, n_seeds = 11037, 5
    n_burn, n_sweeps = 10000, 10000
    couplings = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.41, 0.42, 0.43, 0.435, 0.4375,
                 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5]
    if args.quick:
        n_seeds, n_burn, n_sweeps = 2, 1000, 2000
        couplings = [0.0, 0.3, 0.44]

    # Threshold from a J = 0 run started from all spins up, as in main.cpp
    rng = np.random.default_rng(base_seed)
    m0 = run(L, 0.0, h, beta, 0.0, n_burn, n_sweeps, rng, start="up")
    theta0 = 1.96 * np.diff(m0).std(ddof=1)
    print(f"Threshold from J = 0 run: theta = {theta0:.6f}")

    os.makedirs(args.out, exist_ok=True)
    with open(args.summary, "w") as f:
        f.write("J,seed,n_events,event_rate,mean_abs_m,std_m,n_burn,n_sweeps,alpha,theta\n")

    for J in couplings:
        for k in range(n_seeds):
            seed = base_seed + k
            rng = np.random.default_rng(seed)
            m = run(L, J, h, beta, args.alpha, n_burn, n_sweeps, rng)
            theta = 1.96 * np.diff(m).std(ddof=1) if args.own_threshold else theta0
            events = extract_events(m, theta)
            info = dict(L=L, J=J, h=h, T=T, beta=beta, theta=theta, seed=seed,
                        n_burn=n_burn, n_sweeps=n_sweeps, alpha=args.alpha)
            write_events(os.path.join(args.out, f"J{J:.6f}_seed{seed}.txt"), events, info)
            mean_abs_m = np.abs(m).mean()
            with open(args.summary, "a") as f:
                f.write(f"{J:g},{seed},{len(events)},{len(events) / (n_sweeps - 1):.8g},"
                        f"{mean_abs_m:.8g},{m.std(ddof=1):.8g},{n_burn},{n_sweeps},"
                        f"{args.alpha:g},{theta:.8g}\n")
            print(f"J = {J:<6} seed {seed}: events = {len(events):4d}, mean |m| = {mean_abs_m:.3f}")

    print(f"Saved event files in {args.out}/ and summary in {args.summary}")
