"""Anytime performance assessment: attainment curves, ERT and data profiles (ECDF).

Fixed-budget indicators (IGD, hypervolume, ...) score only the *final* population.
Anytime assessment instead measures *how fast* a run reaches a quality target — the
fixed-target methodology used by the COCO/BBOB platform, which rewards convergence
speed and restart-driven robustness rather than a single end-of-run snapshot.

The building blocks compose as: run with ``save_history=True`` →
:func:`attainment_curve` (best score vs. evaluations) → :func:`first_hitting_time`
(evaluations to reach a target) → :func:`ert` (expected running time) and
:func:`data_profile` (fraction of run/target pairs solved vs. budget).
"""

import numpy as np


def attainment_curve(history, indicator, mode="min"):
    """Best indicator value reached so far as a function of function evaluations.

    Args:
        history: A run's ``res.history`` (needs ``save_history=True``) — a list of
            algorithm snapshots, each exposing ``evaluator.n_eval`` and ``opt``.
        indicator: Callable mapping an objective matrix ``F`` to a scalar score.
        mode: ``"min"`` if a lower score is better (IGD, gap-to-optimum) or ``"max"``
            if higher is better (hypervolume).

    Returns:
        Tuple ``(n_evals, values)`` of 1-D arrays: cumulative function evaluations and
        the best-so-far indicator value at each recorded generation.
    """
    if mode not in ("min", "max"):
        raise ValueError("mode must be 'min' or 'max'")
    better = min if mode == "min" else max

    n_evals, values, best = [], [], None
    for entry in history:
        score = indicator(entry.opt.get("F"))
        best = score if best is None else better(best, score)
        n_evals.append(entry.evaluator.n_eval)
        values.append(best)
    return np.array(n_evals, dtype=float), np.array(values, dtype=float)


def first_hitting_time(n_evals, values, target, mode="min"):
    """Function evaluations at which ``values`` first reaches ``target``.

    Returns ``np.inf`` if the target is never reached.
    """
    values = np.asarray(values)
    hit = values <= target if mode == "min" else values >= target
    return float(np.asarray(n_evals)[np.argmax(hit)]) if hit.any() else np.inf


def ert(runtimes, budget):
    """Expected running time — COCO's average runtime aRT.

    ``aRT = (sum of successful runtimes + sum of evaluations spent in unsuccessful
    trials) / number of successful trials`` (Hansen et al., COCO performance
    assessment). Note the denominator is the number of *successes*, not of trials;
    each failed trial contributes the evaluations it actually spent. Equivalent to
    ``E(RT^s) + (1 - p_s) / p_s * E(RT^us)``.

    Args:
        runtimes: Per-run first-hitting-times for one target (``np.inf`` = failed).
        budget: Evaluations spent by a failed run — a scalar (same budget for every
            run, the fixed-budget case) or a per-run array.

    Returns:
        The expected running time, or ``np.inf`` if no run reached the target.
    """
    runtimes = np.asarray(runtimes, dtype=float)
    success = np.isfinite(runtimes)
    n_success = int(success.sum())
    if n_success == 0:
        return np.inf
    budget = np.broadcast_to(np.asarray(budget, dtype=float), runtimes.shape)
    total = runtimes[success].sum() + budget[~success].sum()
    return float(total / n_success)


def ecdf(runtimes, budgets):
    """Empirical CDF / data profile: fraction of runtimes at or below each budget.

    Args:
        runtimes: First-hitting-times of any shape (flattened over runs and targets);
            ``np.inf`` entries count as unsolved.
        budgets: 1-D array of evaluation budgets at which to evaluate the CDF.

    Returns:
        1-D array (same length as ``budgets``) with the solved fraction at each budget.
    """
    rt = np.asarray(runtimes, dtype=float).ravel()
    if rt.size == 0:
        return np.zeros(len(budgets), dtype=float)
    return np.array([float(np.mean(rt <= b)) for b in np.asarray(budgets, dtype=float)])


def data_profile(curves, targets, budgets, mode="min"):
    """Aggregate ECDF over several runs and targets from their attainment curves.

    Args:
        curves: List of ``(n_evals, values)`` attainment curves, one per run.
        targets: Iterable of target indicator values.
        budgets: 1-D array of evaluation budgets at which to evaluate the profile.
        mode: Indicator direction (``"min"`` or ``"max"``).

    Returns:
        1-D array: fraction of (run, target) pairs solved within each budget.
    """
    runtimes = [
        first_hitting_time(n_evals, values, target, mode=mode)
        for (n_evals, values) in curves
        for target in targets
    ]
    return ecdf(runtimes, budgets)
