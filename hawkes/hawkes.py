"""
Univariate Hawkes process with an exponential kernel: simulation, fitting and checks.

Model (conditional intensity):
    lambda(t) = mu + alpha * sum_{t_i < t} exp(-beta * (t - t_i))

    mu    > 0   background (exogenous) rate
    alpha >= 0  jump in intensity after each event
    beta  > 0   decay rate of the excitation
    n = alpha / beta  branching ratio (mean number of direct offspring per event);
                      the process is stationary only for n < 1.

Contents
    simulate_hawkes   Ogata thinning (Ogata 1981)
    loglik            exact log-likelihood, O(N) recursion (Ozaki 1979)
    fit_hawkes        maximum-likelihood fit, parametrised directly by (mu, n, beta)
    rescaled_times    time-rescaling theorem: residual inter-event times ~ Exp(1)

Run this file directly to produce the validation checks and figures:
    python hawkes.py
Requires numpy, scipy and matplotlib only.
"""
import numpy as np
from scipy import optimize, stats


# ---------------------------------------------------------------------------
# Simulation: Ogata thinning
# ---------------------------------------------------------------------------
def simulate_hawkes(mu, alpha, beta, T, rng=None):
    """Simulate event times on [0, T] by Ogata thinning.

    Between events the exponential-kernel intensity only decays, so its value
    just after the current time is an upper bound until the next event. We
    propose candidate times from a Poisson process at that bound and accept
    each with probability lambda(candidate) / bound.
    """
    rng = np.random.default_rng() if rng is None else rng
    times = []
    t = 0.0
    excite = 0.0          # alpha * sum exp(-beta (t - t_i)) evaluated at time t
    while True:
        lam_bar = mu + excite                  # upper bound on the intensity
        w = rng.exponential(1.0 / lam_bar)     # candidate waiting time
        excite *= np.exp(-beta * w)            # decay the excitation to t + w
        t += w
        if t > T:
            break
        if rng.uniform() * lam_bar <= mu + excite:   # accept with prob lambda/lam_bar
            times.append(t)
            excite += alpha                    # each event adds a jump of alpha
    return np.array(times)


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------
def loglik(mu, alpha, beta, times, T):
    """Exact log-likelihood of event times on [0, T].

    log L = sum_i log lambda(t_i) - integral_0^T lambda(t) dt

    The sum uses the recursion A_1 = 0, A_i = exp(-beta (t_i - t_{i-1})) (1 + A_{i-1}),
    so lambda(t_i) = mu + alpha A_i and the cost is O(N) rather than O(N^2).
    The integral (the compensator) is
        mu T + (alpha / beta) sum_i (1 - exp(-beta (T - t_i))).
    """
    if len(times) == 0:
        return -mu * T
    dt = np.diff(times)
    decay = np.exp(-beta * dt)
    A = np.zeros(len(times))
    for i in range(1, len(times)):             # the O(N) recursion
        A[i] = decay[i - 1] * (1.0 + A[i - 1])
    lam = mu + alpha * A
    if np.any(lam <= 0):
        return -np.inf
    compensator = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - times)))
    return np.sum(np.log(lam)) - compensator


def fit_hawkes(times, T, x0=None):
    """Maximum-likelihood fit. Returns a dict with mu, alpha, beta, n and log-likelihood.

    times : sorted 1-D array of event times in [0, T] (any time unit, used consistently)
    T     : length of the observation window, in the same unit

    Parametrised by (mu, n, beta) with alpha = n * beta, so the stationarity
    condition becomes the simple bound 0 <= n < 1.
    """
    times = np.asarray(times, dtype=float)
    N = len(times)
    if x0 is None:
        rate = N / T
        x0 = (0.5 * rate, 0.5, 1.0)            # crude start: half the events exogenous

    def neg(x):
        mu, n, beta = x
        v = loglik(mu, n * beta, beta, times, T)
        return 1e300 if not np.isfinite(v) else -v

    bounds = [(1e-8, None), (0.0, 0.999), (1e-6, None)]
    res = optimize.minimize(neg, x0, method="L-BFGS-B", bounds=bounds)
    mu, n, beta = res.x
    return {"mu": mu, "alpha": n * beta, "beta": beta, "n": n,
            "loglik": -res.fun, "success": res.success, "N": N}


# ---------------------------------------------------------------------------
# Goodness of fit: time-rescaling theorem
# ---------------------------------------------------------------------------
def rescaled_times(mu, alpha, beta, times):
    """Transformed inter-event times Lambda(t_i) - Lambda(t_{i-1}).

    If the model is correct these are i.i.d. Exp(1) (Ogata 1988), which we test
    with a Kolmogorov-Smirnov test.
    """
    Lam = np.empty(len(times))
    A = 0.0           # sum over events up to and including the previous one of exp(-beta (prev - t_j))
    S = 0.0           # running sum of (1 - exp(-beta (t - t_j))) over earlier events
    prev = 0.0
    for i, t in enumerate(times):
        # moving from prev to t, every earlier event's term grows by exp(-beta (prev - t_j)) (1 - exp(-beta (t - prev)))
        S += A * (1.0 - np.exp(-beta * (t - prev)))
        Lam[i] = mu * t + (alpha / beta) * S
        A = A * np.exp(-beta * (t - prev)) + 1.0
        prev = t
    return np.diff(np.concatenate(([0.0], Lam)))


def ks_test(fit, times):
    taus = rescaled_times(fit["mu"], fit["alpha"], fit["beta"], times)
    return stats.kstest(taus, "expon").pvalue


# ---------------------------------------------------------------------------
# Event-file format agreed with strand B
# ---------------------------------------------------------------------------
def save_events(path, times, T):
    """One event time per line, ascending, in seconds; first line '# T=<window length>'."""
    np.savetxt(path, np.sort(times), header=f"T={T}", fmt="%.9f")


def load_events(path):
    with open(path) as f:
        T = float(f.readline().lstrip("#").strip().split("=")[1])
    return np.loadtxt(path), T


# ---------------------------------------------------------------------------
# Validation and parameter recovery (tasks 3 and 5)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(2026)
    MU, BETA, T = 0.5, 1.0, 2000.0

    # --- Task 3a: mean rate should be mu / (1 - n) -------------------------
    print("Check 1: stationary mean rate = mu / (1 - n)")
    for n in (0.0, 0.3, 0.6, 0.9):
        counts = [len(simulate_hawkes(MU, n * BETA, BETA, T, rng)) / T for _ in range(20)]
        print(f"  n = {n:.1f}: simulated {np.mean(counts):.3f} +/- {np.std(counts):.3f}, "
              f"theory {MU / (1 - n):.3f}")

    # --- Task 3b: alpha = 0 should be a Poisson process --------------------
    pois = simulate_hawkes(MU, 0.0, BETA, 20000.0, rng)
    p = stats.kstest(np.diff(pois) * MU, "expon").pvalue
    print(f"\nCheck 2: alpha = 0 waiting times vs Exp(mu): KS p-value = {p:.2f} "
          f"(p > 0.05 means consistent with Poisson)")

    # --- Figure 1: clustering compared with Poisson ------------------------
    hawk = simulate_hawkes(MU, 0.8 * BETA, BETA, 200.0, rng)
    pois_eq = simulate_hawkes(len(hawk) / 200.0, 0.0, BETA, 200.0, rng)  # same mean rate
    grid = np.linspace(0, 200, 4000)
    lam = MU + 0.8 * BETA * np.array([np.sum(np.exp(-BETA * (g - hawk[hawk < g]))) for g in grid])
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax[0].plot(grid, lam, lw=0.8, color="C1")
    ax[0].eventplot(hawk, lineoffsets=-0.4, linelengths=0.5, color="k", lw=0.5)
    ax[0].set(ylabel="intensity λ(t)", title=f"Hawkes, n = 0.8 ({len(hawk)} events)")
    ax[1].eventplot(pois_eq, lineoffsets=0.5, linelengths=0.8, color="k", lw=0.5)
    ax[1].set(yticks=[], xlabel="time", title=f"Poisson at the same mean rate ({len(pois_eq)} events)")
    plt.tight_layout()
    plt.savefig("fig1_hawkes_vs_poisson.png", dpi=150)

    # --- Task 5: parameter recovery -----------------------------------------
    print("\nParameter recovery: 20 realisations per n, T = 2000")
    n_true = [0.3, 0.5, 0.7, 0.9]
    results = {n: [] for n in n_true}
    t0 = time.time()
    for n in n_true:
        for _ in range(20):
            ev = simulate_hawkes(MU, n * BETA, BETA, T, rng)
            f = fit_hawkes(ev, T)
            f["ks_p"] = ks_test(f, ev)
            results[n].append(f)
        r = results[n]
        nh = np.array([f["n"] for f in r])
        print(f"  n = {n}: n_hat = {nh.mean():.3f} +/- {nh.std():.3f}, "
              f"mu_hat = {np.mean([f['mu'] for f in r]):.3f} (true {MU}), "
              f"beta_hat = {np.mean([f['beta'] for f in r]):.3f} (true {BETA}), "
              f"KS p > 0.05 in {np.mean([f['ks_p'] > 0.05 for f in r]) * 100:.0f}% of fits")
    print(f"  ({time.time() - t0:.0f} s)")

    # --- Figure 2: recovery plot -------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for n in n_true:
        nh = np.array([f["n"] for f in results[n]])
        ax[0].scatter(np.full(len(nh), n) + rng.uniform(-0.012, 0.012, len(nh)), nh,
                      s=10, alpha=0.4, color="C0")
        ax[0].errorbar(n, nh.mean(), yerr=nh.std(), fmt="o", color="C3", capsize=4, zorder=3)
    ax[0].plot([0.2, 1.0], [0.2, 1.0], "k--", lw=0.8, label="perfect recovery")
    ax[0].set(xlabel="true branching ratio n", ylabel="estimated n",
              title="Branching ratio recovered by MLE", xlim=(0.2, 1.0), ylim=(0.2, 1.0))
    ax[0].legend(loc="upper left")
    rel_mu = [np.array([f["mu"] for f in results[n]]) / MU for n in n_true]
    rel_beta = [np.array([f["beta"] for f in results[n]]) / BETA for n in n_true]
    pos = np.arange(len(n_true))
    ax[1].boxplot(rel_mu, positions=pos - 0.18, widths=0.3, patch_artist=True,
                  boxprops=dict(facecolor="C0", alpha=0.5))
    ax[1].boxplot(rel_beta, positions=pos + 0.18, widths=0.3, patch_artist=True,
                  boxprops=dict(facecolor="C1", alpha=0.5))
    ax[1].axhline(1, color="k", ls="--", lw=0.8)
    ax[1].set_xticks(pos, [str(n) for n in n_true])
    ax[1].set(xlabel="true branching ratio n", ylabel="estimate / true value",
              title="Background rate μ (blue) and decay β (orange)")
    plt.tight_layout()
    plt.savefig("fig2_parameter_recovery.png", dpi=150)
    print("\nSaved fig1_hawkes_vs_poisson.png and fig2_parameter_recovery.png")
