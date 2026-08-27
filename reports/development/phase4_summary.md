# Phase 4 development summary

These are walk-forward development results, not final-holdout or trading results.

- Development rows: 2,591,862
- Walk-forward folds: 4
- Preferred fixed linear specification by the frozen rule: `ridge` / `expanded`
- Mean fold OOS R-squared: 0.0169641
- Same-model baseline mean fold OOS R-squared: 0.00730988
- Expanded-minus-baseline mean OOS R-squared: 0.00965427
- Paired daily MSE-improvement 95% bootstrap interval: [1.697e-10, 5.864e-10]
- Evaluation days with lower expanded-model MSE: 19/20
- Mean logistic ROC AUC, baseline / expanded: 0.585761 / 0.626996
- Mean logistic Brier score, baseline / expanded: 0.237593 / 0.232177
- Recorded model failures: 0

Statistical predictability, if any, is not evidence of economic tradability. Costs, latency and execution are deferred to Phase 6.
