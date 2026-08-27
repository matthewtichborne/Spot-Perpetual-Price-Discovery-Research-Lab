"""Realised-volatility feature expressions."""

import polars as pl


def realised_volatility(return_column: str, steps: int) -> pl.Expr:
    """Return square-root realised variance over a trailing fixed-bar window."""

    # Parallel rolling summation can produce a tiny negative round-off residue for
    # near-zero variance windows. Clipping at the mathematical lower bound avoids
    # turning that numerical noise into NaN during the square root.
    return (
        pl.col(return_column).pow(2).rolling_sum(steps, min_samples=1).clip(lower_bound=0.0).sqrt()
    )
