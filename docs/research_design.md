# Frozen research design

Status: **frozen on 2026-08-25 before downloading the remaining development dates;
final holdout sealed**.

## Primary hypothesis and target

After controlling for lagged BTC spot returns, spot signed aggregate-trade flow,
activity and realised volatility, BTC USD-M perpetual signed aggregate-trade flow
adds information about the next five-second BTC spot log return.

The primary continuous target is `target_spot_log_return_5000ms`. The secondary
direction target is `target_spot_direction_5000ms`. “Flow” always means signed
aggregate-trade flow, not limit-order-book order-flow imbalance.

## Samples and sealed holdout

- Phase 4 development: 2025-01-02 through 2025-01-31.
- Phase 5 confirmation/robustness: 2025-02-01 through 2025-02-20.
- Final sealed holdout: 2025-02-21 through 2025-03-02.

BTC is the primary asset. ETH is retained for the pre-specified Phase 5 replication.
No command may access a configuration whose `holdout_status` is `sealed`.

## Predictor sets

The baseline set is fixed to these spot predictors:

```text
spot_log_return_1000ms
spot_log_return_5000ms
spot_quantity_imbalance_1000ms
spot_quantity_imbalance_5000ms
spot_signed_notional_1000ms
spot_signed_notional_5000ms
spot_trade_count_1000ms
spot_trade_count_5000ms
spot_notional_1000ms
spot_notional_5000ms
spot_realised_volatility_1000ms
spot_realised_volatility_5000ms
```

The expanded set is the baseline plus:

```text
perpetual_log_return_1000ms
perpetual_log_return_5000ms
perpetual_quantity_imbalance_1000ms
perpetual_quantity_imbalance_5000ms
perpetual_signed_notional_1000ms
perpetual_signed_notional_5000ms
perpetual_trade_count_1000ms
perpetual_trade_count_5000ms
perpetual_realised_volatility_1000ms
perpetual_realised_volatility_5000ms
spot_perp_log_basis
spot_perp_basis_change_1000ms
spot_perp_basis_zscore_10000ms
perpetual_spot_relative_quantity_1000ms
perpetual_spot_relative_intensity_1000ms
```

All predictors retain the Phase 3 one-base-bar lag. Median imputation and
standardisation are fitted inside each training fold only.

## Walk-forward folds and leakage controls

The 30 development days use expanding whole-day folds:

| Fold | Training dates | Evaluation dates |
|---:|---|---|
| 1 | Jan 2–11 | Jan 12–16 |
| 2 | Jan 2–16 | Jan 17–21 |
| 3 | Jan 2–21 | Jan 22–26 |
| 4 | Jan 2–26 | Jan 27–31 |

A ten-second purge is removed from the end of training and a ten-second embargo from
the start of evaluation. Dates never overlap, observations are never shuffled, and
daily-tail labels that cross a partition boundary remain null.

## Models and fixed parameters

Models are fitted in this order:

1. Training-target mean and zero-return reference predictions.
2. Unregularised linear regression, baseline and expanded.
3. Ridge regression with fixed `alpha=1.0`, baseline and expanded.
4. Logistic regression with fixed `C=1.0`, baseline and expanded.

No hyperparameter is chosen from these evaluation folds. XGBoost is deferred to
Phase 5. A model failure is retained in `phase4_failures.csv`, not silently omitted.

## Metrics and incremental comparison

Regression metrics are training-mean-referenced out-of-sample R-squared, MSE, MAE,
Pearson information coefficient and rank information coefficient. Classification
metrics are ROC AUC, precision-recall AUC, Brier score, accuracy and class balance.

The primary incremental comparison is expanded minus baseline for the same model.
For loss metrics, improvement is baseline loss minus expanded loss. Results are
reported by fold and UTC day. A 2,000-replicate paired day-block bootstrap with seed
`20260825` provides a 95% interval for mean daily MSE improvement.

## HAC inference

Interpretive OLS/HAC inference is separate from out-of-sample model evaluation. It
uses every fifth decision row from the development sample and a parsimonious model:

- Baseline: intercept, spot one/five-second return, spot one/five-second quantity
  imbalance, spot one-second activity and spot five-second realised volatility.
- Expanded additions: perpetual one/five-second quantity imbalance and log basis.

Predictors are median-imputed and standardised on this inference sample. Newey-West
standard errors use `maxlags=12`. HAC coefficients are not proof of causality or
tradability.

## Model-selection rule

The Phase 4 preferred linear specification is the model/scope with the highest mean
fold out-of-sample R-squared, provided its paired daily MSE improvement confidence
interval is reported. Ties or negligible differences favour the simpler baseline.
This rule may be applied to Phase 5 confirmation data but may not be changed after
seeing the final holdout.
