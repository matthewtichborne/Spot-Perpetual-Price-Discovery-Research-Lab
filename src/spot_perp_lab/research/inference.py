"""HAC inference and paired day-block bootstrap utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    probability_positive: float
    days: int


def paired_day_bootstrap(
    daily_improvements: np.ndarray, replicates: int, random_seed: int
) -> BootstrapInterval:
    """Bootstrap the mean of paired day-level metric improvements."""

    if daily_improvements.ndim != 1 or daily_improvements.size < 2:
        raise ValueError("paired day bootstrap requires at least two daily values")
    generator = np.random.default_rng(random_seed)
    indices = generator.integers(
        0, daily_improvements.size, size=(replicates, daily_improvements.size)
    )
    bootstrap_means = daily_improvements[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return BootstrapInterval(
        estimate=float(np.mean(daily_improvements)),
        lower=float(lower),
        upper=float(upper),
        probability_positive=float(np.mean(bootstrap_means > 0)),
        days=int(daily_improvements.size),
    )


def hac_regression(
    features: np.ndarray,
    target: np.ndarray,
    feature_names: tuple[str, ...],
    max_lags: int,
    scope: str,
) -> list[dict[str, float | int | str]]:
    """Fit standardized OLS with Newey-West/HAC covariance."""

    imputed = SimpleImputer(strategy="median").fit_transform(features)
    standardized = StandardScaler().fit_transform(imputed)
    design = sm.add_constant(standardized, prepend=True)
    fitted = sm.OLS(target, design).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
    terms = ("intercept", *feature_names)
    return [
        {
            "scope": scope,
            "term": term,
            "coefficient": float(fitted.params[index]),
            "standard_error": float(fitted.bse[index]),
            "t_value": float(fitted.tvalues[index]),
            "p_value": float(fitted.pvalues[index]),
            "observations": int(target.size),
            "hac_max_lags": max_lags,
        }
        for index, term in enumerate(terms)
    ]
