# Research design

Status: **preliminary — not frozen; final holdout sealed**.

The primary hypothesis is that perpetual-futures signed aggregate-trade flow adds
information about the next five-second BTC spot log return after controlling for
lagged spot returns, spot signed trade flow, spot volume and realised volatility.

Only chronological, whole-day splits will be used. The maximum target horizon will
be purged around fold boundaries, preprocessing will be fitted inside training folds,
and the final 15–20% of dates will remain sealed until the feature definitions,
model-selection rule and evaluation metrics are frozen here.

The primary comparison is a simple same-market baseline against an expanded model
that adds perpetual flow, intensity, basis and relative-activity features. “Flow”
always means signed aggregate-trade flow rather than order-book imbalance.
