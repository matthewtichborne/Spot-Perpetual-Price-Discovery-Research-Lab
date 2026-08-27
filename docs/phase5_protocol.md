# Frozen Phase 5 protocol

Status: **frozen on 2026-08-26 before accessing any 2025-02 confirmation archive**.
The final holdout remains sealed.

## Samples and targets

- Model development and nonlinear tuning use only 2025-01-02 through 2025-01-31.
- Confirmation and robustness use 2025-02-01 through 2025-02-20 exactly once.
- The final 2025-02-21 through 2025-03-02 holdout is not accessed in Phase 5.
- BTC is primary and ETH is the prespecified cross-asset replication.
- The primary target is the five-second spot log return. One- and ten-second spot
  log returns are secondary signal-decay targets.
- The frozen 12-feature baseline and 27-feature expanded sets from
  `docs/research_design.md` are unchanged.

## Confirmation fit and comparison

All confirmation models are trained on the complete January development sample and
evaluated without refitting on the complete February confirmation sample. Median
imputation and standardisation for Ridge are fit on development only. The registered
Ridge model uses `alpha=1.0` and is evaluated for baseline and expanded features.
Training-mean and zero-return references are retained.

## Limited XGBoost tuning

XGBoost uses expanded features only. To bound computation, training rows whose
`decision_time_ns` is divisible by five seconds are used; all validation and
confirmation rows are scored. The tuning split is fixed before confirmation access:

- tuning train: 2025-01-02 through 2025-01-26, less the final ten-second purge;
- tuning validation: 2025-01-27 through 2025-01-31, after a ten-second embargo;
- objective: `reg:squarederror`;
- tree method: `hist`;
- fixed settings: `learning_rate=0.05`, `subsample=0.8`,
  `colsample_bytree=0.8`, `min_child_weight=20`, `reg_lambda=1`, `n_jobs=4`;
- deterministic seed: `20260825`;
- search grid: `max_depth` in `{2, 3}` crossed with `n_estimators` in `{100, 200}`.

The candidate with lowest validation MSE wins; ties choose fewer estimators and then
shallower depth. The selected candidate is refit on every fifth row of the complete
development period before confirmation scoring. No confirmation metric changes the
hyperparameters.

## Prespecified robustness checks

1. **Placebo alignment:** within each symbol-day, the 15 expanded-only columns are
   circularly shifted by exactly 900 one-second rows. Ridge/baseline is unchanged;
   Ridge/placebo-expanded is trained on development and scored on confirmation. This
   is a negative control and never an executable specification.
2. **Volatility regimes:** the low/high threshold is the median development value of
   `spot_realised_volatility_5000ms`. Confirmation Ridge metrics are reported below
   and at/above that fixed threshold.
3. **Horizon decay:** Ridge baseline and expanded specifications are independently
   fit and scored for the one-, five- and ten-second return targets. No horizon is
   selected from these secondary results.
4. **ETH replication:** the identical Ridge baseline/expanded five-second procedure
   is trained on ETH development data and evaluated on ETH confirmation data.

For primary BTC five-second Ridge and XGBoost comparisons, metrics include
training-mean-referenced OOS R-squared, MSE, MAE, Pearson IC and rank IC. Daily MSE
differences use the registered paired 2,000-replicate day-block bootstrap. Failures
and null results remain in the report.

## Final model-selection rule

The default final model is the registered Ridge/expanded five-second BTC model.
XGBoost/expanded replaces it only if both conditions hold on confirmation:

1. XGBoost OOS R-squared exceeds Ridge/expanded OOS R-squared by at least 0.001; and
2. the lower endpoint of the paired daily Ridge-MSE-minus-XGBoost-MSE 95% bootstrap
   interval is strictly positive.

Otherwise Ridge/expanded remains selected. Placebo, regime, horizon and ETH results
are diagnostic and cannot alter this rule. After Phase 5, the selected model class,
features, target, preprocessing, hyperparameters and training sample are written to
`data/manifests/final-model-specification.json`. The final configuration remains
`sealed`; Phase 8 requires an explicit opening record and one evaluation only. For
that single evaluation, Ridge is refit on every eligible BTC row from 2025-01-02
through 2025-02-20; XGBoost, if selected, is refit on the registered every-fifth-row
subsample over the same dates. No fitting or calibration uses the final period.
