#!/usr/bin/env python3
"""Reproducible Monte Carlo supplement for Poisson EM on the 3x3 CT voxel model.

Corrected re-implementation of the MATLAB experiment in main.m / EM_algorithm.m /
CRLB.m (those files are left unchanged). Uses only the Python standard library and
NumPy. Writes, relative to this file:

    ../data/results.json   configuration, model, self-tests, per-gain metrics, trace checkpoints
    ../data/summary.csv    one row per (gain, checkpoint iteration)
    ../data/trace.csv      one row per EM iteration for a single reproducible replicate

Deliberate differences from the legacy MATLAB code:
  * the PREDEFINED 16x9 binary A from main.m is used (not random_model), with a fixed
    base vector b instead of unifrnd draws;
  * initialization is the least-squares solution clipped elementwise to >= 1e-8
    (legacy used inv(A'A)A'y without clipping, which can give negative EM iterates);
  * the log likelihood is exact (lgamma(y+1)), not Stirling's approximation
    (which is undefined at y = 0);
  * the CRLB is evaluated at the TRUE x, not at the current estimate;
  * normalization is by gain**2 (legacy divided by the gain *index* squared).

Usage:  python3 reproduce.py [--trials 10000] [--iterations 1000] [--seed 20260929]
"""

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

# Predefined model matrix from main.m (rows = 16 ray paths, columns = 9 voxels p1..p9).
A = np.array([
    [0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
], dtype=float)
BASE = np.array([120, 240, 360, 180, 720, 300, 90, 420, 540], dtype=float)
GAINS = (0.1, 1.0, 5.0, 10.0)
M_OBS, N_PARAMS = A.shape
SENS = A.sum(axis=0)  # column sensitivities s_j = sum_i A_ij

DEFAULTS = {"trials": 10000, "iterations": 1000, "seed": 20260929}
MC_CHECKPOINTS = (0, 20, 200, 1000)
TRACE_CHECKPOINTS = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
TRACE_GAIN = 1.0
INIT_FLOOR = 1e-8          # elementwise positive clip of the LS initialization
LOGLIK_RTOL = 1e-9         # allowed decrease: LOGLIK_RTOL * (1 + |previous loglik|)
CONVERGED_TOL = 1e-6       # threshold on relative iterate change
TINY = np.finfo(float).tiny

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def require(condition, message):
    """Assertion that survives `python -O`."""
    if not condition:
        raise AssertionError(message)


def em_step(X, Y, mu=None):
    """One batched multiplicative EM update: X_j *= sum_i A_ij y_i / mu_i / s_j.

    X and Y are (trials, n) and (trials, m). A ratio y_i / mu_i is set to zero only in
    the 0/0 case; a zero mean with a positive count is rejected by check_support.
    """
    if mu is None:
        mu = X @ A.T
    check_support(Y, mu)
    ratio = np.zeros_like(mu)
    np.divide(Y, mu, out=ratio, where=mu > 0)
    return X * (ratio @ A) / SENS


def em_step_scalar(y, x):
    """Plain-Python transcription of EM_algorithm.m, used only as a self-test."""
    den1 = [sum(A[i, j] * x[j] for j in range(N_PARAMS)) for i in range(M_OBS)]
    out = []
    for j in range(N_PARAMS):
        expectation = sum(y[i] * A[i, j] / den1[i] for i in range(M_OBS))
        den2 = sum(A[i, j] for i in range(M_OBS))
        out.append(x[j] / den2 * expectation)
    return out


def check_support(Y, mu):
    require(np.all(np.isfinite(mu)), "non-finite Poisson means")
    require(not np.any((Y > 0) & ~(mu > 0)), "zero Poisson mean where a count is positive")


def log_factorial(Y):
    """lgamma(y + 1) for an integer-valued array, via a table over its unique values."""
    uniq, inverse = np.unique(Y, return_inverse=True)
    table = np.array([math.lgamma(float(v) + 1.0) for v in uniq])
    return table[inverse].reshape(Y.shape)


def exact_loglik(Y, mu, log_fact):
    """Exact Poisson log likelihood per trial: sum_i y_i log mu_i - mu_i - log(y_i!).

    Terms with y_i = 0 contribute 0 to y log mu (also when mu_i = 0). Summing the
    elementwise terms keeps large y log mu and log(y!) values cancelling locally.
    """
    pos = Y > 0
    ylogmu = np.where(pos, Y * np.log(np.where(pos, mu, 1.0)), 0.0)
    return (ylogmu - mu - log_fact).sum(axis=-1)


def relative_change(new, old):
    return np.linalg.norm(new - old, axis=-1) / np.maximum(np.linalg.norm(old, axis=-1), TINY)


def change_stats(rc):
    return {
        "median": float(np.median(rc)),
        "p95": float(np.percentile(rc, 95)),
        "max": float(rc.max()),
        "fraction_below_1e-6": float(np.mean(rc < CONVERGED_TOL)),
    }


def crlb_covariance(x_true):
    """Inverse Fisher information at the TRUE parameter: F = A^T diag(1 / (A x)) A."""
    mu = A @ x_true
    F = A.T @ (A / mu[:, None])
    return np.linalg.solve(F, np.eye(N_PARAMS))


def mc_metrics(X, X_prev, gain, loglik, crlb_trace):
    """Monte Carlo summary of the (trials, n) estimates at one iteration."""
    trials = X.shape[0]
    Z = X / gain                                   # estimates in base units
    loss = np.mean((Z - BASE) ** 2, axis=1)        # per-trial MSE in base units
    mse = float(loss.mean())
    bias = Z.mean(axis=0) - BASE
    bias_sq = float(np.sum(bias ** 2) / N_PARAMS)
    variance = float(Z.var(axis=0, ddof=0).mean())
    require(math.isclose(bias_sq + variance, mse, rel_tol=1e-9, abs_tol=1e-12),
            "bias^2 + variance does not reproduce the MSE")
    crlb_base = crlb_trace / (N_PARAMS * gain * gain)
    return {
        "mse_base_units": mse,
        "mse_base_units_mcse": float(loss.std(ddof=1) / math.sqrt(trials)),
        "mse_param_units": mse * gain * gain,
        "crlb_base_units": crlb_base,
        "crlb_param_units": crlb_trace / N_PARAMS,
        "ratio_mse_to_crlb": mse / crlb_base,
        "bias_sq_base_units": bias_sq,
        "variance_base_units": variance,
        "bias_base_units": bias.tolist(),
        "mean_exact_loglik": float(loglik.mean()),
        "relative_change": None if X_prev is None else change_stats(relative_change(X, X_prev)),
    }


def trace_row(k, X, X_prev, gain, loglik):
    theta = X[0]
    return {
        "iteration": k,
        "exact_loglik": float(loglik[0]),
        "mse_base_units": float(np.mean((theta / gain - BASE) ** 2)),
        "theta": theta.tolist(),
        "relative_change": None if X_prev is None else float(relative_change(theta, X_prev[0])),
    }


def run_gain(gain, Y, iterations, checkpoints, want_trace):
    x_true = gain * BASE
    C = crlb_covariance(x_true)
    crlb_trace = float(np.trace(C))

    ls = np.linalg.lstsq(A, Y.T, rcond=None)[0].T
    init = {
        "method": "least squares, then elementwise max(., 1e-8)",
        "fraction_trials_any_negative_ls_coordinate": float(np.mean(np.any(ls < 0, axis=1))),
        "fraction_trials_any_clipped_coordinate": float(np.mean(np.any(ls < INIT_FLOOR, axis=1))),
        "fraction_trials_all_zero_counts": float(np.mean(np.all(Y == 0, axis=1))),
    }
    X = np.maximum(ls, INIT_FLOOR)
    log_fact = log_factorial(Y)
    mu = X @ A.T
    check_support(Y, mu)
    ll = exact_loglik(Y, mu, log_fact)

    records, trace = [], []
    max_decrease, min_increment = 0.0, math.inf
    X_prev = None
    for k in range(iterations + 1):
        if k in checkpoints:
            records.append({"iteration": k, **mc_metrics(X, X_prev, gain, ll, crlb_trace)})
        if want_trace:
            trace.append(trace_row(k, X, X_prev, gain, ll))
        if k == iterations:
            break
        X_new = em_step(X, Y, mu)
        require(np.all(np.isfinite(X_new)) and X_new.min() >= 0.0,
                f"non-finite or negative estimate at gain {gain}, iteration {k + 1}")
        mu_new = X_new @ A.T
        check_support(Y, mu_new)
        ll_new = exact_loglik(Y, mu_new, log_fact)
        diff = ll_new - ll
        require(np.all(diff >= -LOGLIK_RTOL * (1.0 + np.abs(ll))),
                f"log likelihood decreased at gain {gain}, iteration {k + 1}")
        max_decrease = max(max_decrease, float(np.max(-diff)))
        min_increment = min(min_increment, float(diff.min()))
        X_prev, X, mu, ll = X, X_new, mu_new, ll_new

    result = {
        "gain": gain,
        "x_true": x_true.tolist(),
        "crlb": {
            "evaluated_at": "true x",
            "trace_param_units": crlb_trace,
            "trace_over_9_param_units": crlb_trace / N_PARAMS,
            "trace_over_9_gain2_base_units": crlb_trace / (N_PARAMS * gain * gain),
            "diag_param_units": np.diag(C).tolist(),
        },
        "initialization": init,
        "monotonicity": {
            "quantity": "exact observed log likelihood, per trial, consecutive iterations",
            "tolerance": "decrease allowed up to 1e-9 * (1 + |previous loglik|)",
            "max_loglik_decrease": max_decrease,
            "min_loglik_increment": min_increment,
            "passed": True,
        },
        "final_relative_change": change_stats(relative_change(X, X_prev)),
        "checkpoints": records,
    }
    return result, trace


def self_tests():
    # Noise-free algebraic test: with y = A b (real valued), b is an EM fixed point.
    y = (A @ BASE)[None, :]
    fixed = em_step(BASE[None, :], y)[0]
    fixed_dev = float(np.max(np.abs(fixed - BASE) / BASE))

    # Vectorized update vs. the scalar formula of EM_algorithm.m on one positive case.
    y_test = [float((7 * i + 3) % 29 + 1) for i in range(M_OBS)]
    x_test = [float(v) / 100.0 + 1.0 for v in BASE]
    vec = em_step(np.array([x_test]), np.array([y_test]))[0]
    scal = np.array(em_step_scalar(y_test, x_test))
    equiv_dev = float(np.max(np.abs(vec - scal) / np.abs(scal)))

    tests = {
        "noise_free_fixed_point": {
            "description": "y = A b (real valued); one EM step from X = b",
            "max_relative_deviation": fixed_dev,
            "passed": fixed_dev < 1e-12,
        },
        "vectorized_vs_scalar_update": {
            "description": "batched update vs. loop transcription of EM_algorithm.m",
            "y_test": y_test,
            "x_test": x_test,
            "max_relative_difference": equiv_dev,
            "passed": equiv_dev < 1e-12,
        },
    }
    for name, t in tests.items():
        require(t["passed"], f"self-test failed: {name}")
    return tests


def fmt(v):
    return "" if v is None else repr(v) if isinstance(v, float) else str(v)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for row in rows:
            w.writerow([fmt(v) for v in row])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--trials", type=int, default=DEFAULTS["trials"])
    p.add_argument("--iterations", type=int, default=DEFAULTS["iterations"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    args = p.parse_args()
    if args.trials < 2:
        p.error("--trials must be >= 2 (standard error uses ddof=1)")
    if args.iterations < 1:
        p.error("--iterations must be >= 1")
    return args


def main():
    args = parse_args()
    start = time.perf_counter()

    rank = int(np.linalg.matrix_rank(A))
    require(A.shape == (16, 9), "A must be 16x9")
    require(rank == N_PARAMS, f"rank(A) = {rank}, expected {N_PARAMS}")
    require(np.all(SENS > 0), "every voxel needs positive sensitivity")
    require(np.all(A.sum(axis=1) > 0), "every ray must cross at least one voxel")

    checkpoints = sorted({k for k in MC_CHECKPOINTS if k <= args.iterations} | {args.iterations})
    trace_checkpoints = sorted({k for k in TRACE_CHECKPOINTS if k <= args.iterations} | {args.iterations})
    nondefault = {k: getattr(args, k) for k in DEFAULTS if getattr(args, k) != DEFAULTS[k]}

    tests = self_tests()

    rng = np.random.default_rng(args.seed)
    gain_results, trace = [], None
    for gain in GAINS:
        # The only RNG use: one (trials, 16) batch of independent Poisson counts per gain.
        Y = rng.poisson(A @ (gain * BASE), size=(args.trials, M_OBS)).astype(float)
        result, tr = run_gain(gain, Y, args.iterations, set(checkpoints), gain == TRACE_GAIN)
        gain_results.append(result)
        if gain == TRACE_GAIN:
            trace = tr
    elapsed = time.perf_counter() - start

    results = {
        "schema": "ct-poisson-em-supplement/v1",
        "config": {
            "trials": args.trials,
            "iterations": args.iterations,
            "seed": args.seed,
            "defaults": DEFAULTS,
            "nondefault_settings": nondefault,
            "gains": list(GAINS),
            "rng": "numpy.random.default_rng(seed); one Poisson batch per gain, in gain order",
            "bit_generator": type(rng.bit_generator).__name__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "mc_checkpoints": checkpoints,
            "init_floor": INIT_FLOOR,
            "loglik_rtol": LOGLIK_RTOL,
            "converged_threshold": CONVERGED_TOL,
            "percentile_method": "numpy default (linear)",
            "note": "elapsed time is printed to the console only, so this file is deterministic",
        },
        "model": {
            "A": A.astype(int).tolist(),
            "base_b": BASE.tolist(),
            "m_observations": M_OBS,
            "n_parameters": N_PARAMS,
            "rank": rank,
            "condition_number_2norm": float(np.linalg.cond(A)),
            "column_sensitivities": SENS.tolist(),
            "x_true": "gain * base_b",
        },
        "definitions": {
            "mse_base_units": "mean over trials of mean_j (X_j/gain - b_j)^2  (= MSE / gain^2)",
            "mse_base_units_mcse": "std(per-trial loss, ddof=1) / sqrt(trials)",
            "crlb_base_units": "trace(F^-1) / (9 gain^2), F = A^T diag(1/(A x_true)) A",
            "crlb_param_units": "trace(F^-1) / 9",
            "bias_base_units": "mean over trials of X/gain, minus b",
            "bias_sq_base_units": "||bias||^2 / 9",
            "variance_base_units": "mean_j var(X_j/gain, ddof=0); bias_sq + variance = mse",
            "relative_change": "||X_k - X_{k-1}|| / max(||X_{k-1}||, tiny), per trial",
            "exact_loglik": "sum_i y_i log mu_i - mu_i - lgamma(y_i + 1), 0 log mu := 0",
            "iteration_0": "clipped least-squares initialization, before any EM step",
        },
        "method_notes": [
            "Predefined A from main.m is used; random_model is not called.",
            "Initialization changed from legacy inv(A'A)A'y to np.linalg.lstsq clipped to >= 1e-8.",
            "No floor is added to means or parameters during EM iterations.",
            "Log likelihood is exact (lgamma), replacing the legacy Stirling approximation.",
            "CRLB uses the true x; legacy code evaluated it at the current estimate.",
            "Normalization is by gain^2; legacy code divided by the gain index squared.",
        ],
        "self_tests": tests,
        "gains": gain_results,
        "trace": {
            "source": f"first Monte Carlo replicate at gain = {TRACE_GAIN}",
            "units": "theta in parameter units (equal to base units at gain 1)",
            "full_trace_file": "trace.csv",
            "checkpoints": [r for r in trace if r["iteration"] in set(trace_checkpoints)],
        },
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, allow_nan=False)
        f.write("\n")

    bias_cols = [f"bias_{j + 1}_base_units" for j in range(N_PARAMS)]
    rc_keys = ["median", "p95", "max", "fraction_below_1e-6"]
    summary_rows = []
    for g in gain_results:
        for r in g["checkpoints"]:
            rc = r["relative_change"] or {}
            summary_rows.append(
                [g["gain"], r["iteration"], args.trials, args.iterations, args.seed,
                 r["mse_base_units"], r["mse_base_units_mcse"], r["mse_param_units"],
                 r["crlb_base_units"], r["crlb_param_units"], r["ratio_mse_to_crlb"],
                 r["bias_sq_base_units"], r["variance_base_units"], *r["bias_base_units"],
                 r["mean_exact_loglik"]] + [rc.get(k) for k in rc_keys])
    write_csv(DATA_DIR / "summary.csv",
              ["gain", "iteration", "trials", "iterations_total", "seed",
               "mse_base_units", "mse_base_units_mcse", "mse_param_units",
               "crlb_base_units", "crlb_param_units", "ratio_mse_to_crlb",
               "bias_sq_base_units", "variance_base_units", *bias_cols,
               "mean_exact_loglik"] + [f"relative_change_{k}" for k in rc_keys],
              summary_rows)
    write_csv(DATA_DIR / "trace.csv",
              ["iteration", "exact_loglik", "mse_base_units",
               *[f"theta_{j + 1}" for j in range(N_PARAMS)], "relative_change"],
              [[r["iteration"], r["exact_loglik"], r["mse_base_units"], *r["theta"],
                r["relative_change"]] for r in trace])

    # Console summary
    print(f"Poisson EM supplement: trials={args.trials} iterations={args.iterations} "
          f"seed={args.seed} ({results['config']['bit_generator']}, numpy {np.__version__})")
    if nondefault:
        print(f"  NOTE: non-default settings {nondefault}")
    print(f"  A: 16x9 predefined, rank {rank}, cond {results['model']['condition_number_2norm']:.4g}")
    for name, t in tests.items():
        dev = t.get("max_relative_deviation", t.get("max_relative_difference"))
        print(f"  self-test {name}: {'PASS' if t['passed'] else 'FAIL'} (max rel {dev:.2e})")
    print(f"\n{'gain':>6} {'iter':>5} {'MSE/g^2':>12} {'MCSE':>10} {'CRLB/g^2':>11} "
          f"{'MSE/CRLB':>9} {'bias^2':>10} {'conv<1e-6':>10}")
    for g in gain_results:
        for r in g["checkpoints"]:
            rc = r["relative_change"]
            conv = "-" if rc is None else f"{rc['fraction_below_1e-6']:.4f}"
            print(f"{g['gain']:>6g} {r['iteration']:>5d} {r['mse_base_units']:>12.5g} "
                  f"{r['mse_base_units_mcse']:>10.3g} {r['crlb_base_units']:>11.5g} "
                  f"{r['ratio_mse_to_crlb']:>9.4f} {r['bias_sq_base_units']:>10.3g} {conv:>10}")
        init, mono = g["initialization"], g["monotonicity"]
        print(f"{'':>6} init: any-negative LS {init['fraction_trials_any_negative_ls_coordinate']:.4f}, "
              f"clipped {init['fraction_trials_any_clipped_coordinate']:.4f}; "
              f"loglik max decrease {mono['max_loglik_decrease']:.3g}, "
              f"min increment {mono['min_loglik_increment']:.3g}")
    print(f"\nWrote {DATA_DIR / 'results.json'}, summary.csv, trace.csv "
          f"({len(trace)} trace rows) in {elapsed:.1f} s")


if __name__ == "__main__":
    main()
