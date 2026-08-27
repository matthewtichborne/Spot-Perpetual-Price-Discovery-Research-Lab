"""Fixed Phase 4 statistical model pipelines."""

from __future__ import annotations

from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import (  # type: ignore[import-untyped]
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
from xgboost import XGBRegressor


def regression_pipeline(model: str, ridge_alpha: float) -> Pipeline:
    """Create an inside-fold imputation/scaling/regression pipeline."""

    if model == "linear":
        estimator = LinearRegression()
    elif model == "ridge":
        estimator = Ridge(alpha=ridge_alpha)
    else:
        raise ValueError(f"unknown regression model: {model}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def classification_pipeline(logistic_c: float, random_seed: int) -> Pipeline:
    """Create the fixed inside-fold logistic-regression pipeline."""

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=logistic_c,
                    max_iter=200,
                    solver="lbfgs",
                    random_state=random_seed,
                ),
            ),
        ]
    )


def xgboost_regressor(parameters: dict[str, int | float], random_seed: int) -> XGBRegressor:
    """Create a deterministic Phase 5 histogram-tree regressor."""

    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        reg_lambda=1.0,
        n_jobs=4,
        random_state=random_seed,
        **parameters,
    )
