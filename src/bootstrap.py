import numpy as np


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute mean and confidence interval via bootstrap resampling.

    Args:
        values: Per-example metric values (bools are treated as 0/1).
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: mean, ci_low, ci_high, std.
    """
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.percentile(means, 100 * alpha)),
        "ci_high": float(np.percentile(means, 100 * (1 - alpha))),
        "std": float(means.std()),
    }
