# Phase 5 confirmation and robustness summary

These are prespecified confirmation results, not final-holdout or trading results.

- Selected final model class by the frozen rule: `xgboost` / `expanded`
- BTC Ridge OOS R-squared, baseline / expanded: 0.0103119 / 0.0242779
- BTC XGBoost expanded OOS R-squared: 0.0303015
- Ridge-minus-XGBoost daily MSE bootstrap 95% interval: [1.604e-10, 2.446e-10]
- Selected XGBoost candidate: {"max_depth": 2, "n_estimators": 200}
- BTC placebo-expanded OOS R-squared: 0.0102306
- ETH Ridge OOS R-squared, baseline / expanded: 0.00456633 / 0.0130729
- Recorded failures: 0

The final holdout remains sealed. Statistical predictability is not evidence of economic tradability; costs, latency and execution remain deferred to Phase 6.
