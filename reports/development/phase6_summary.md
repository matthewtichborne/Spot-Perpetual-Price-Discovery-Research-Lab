# Phase 6 execution and portfolio summary

These are confirmation-sample cost/latency sensitivities, not exact fill or final-holdout results.

- Primary latency: 500 ms
- Primary all-in round-trip cost: 7 bps
- BTC / ETH portfolio risk weights: 0.6821 / 0.3179
- Combined trades: 1,271
- Combined gross P&L: 0.020998
- Combined net P&L: -0.407156
- Combined annualised daily net Sharpe: -1847.65
- Combined maximum drawdown: -0.407156
- Combined break-even round-trip cost: 0.343302 bps
- Primary conclusion: the statistical signal does not survive the registered cost assumption.

Gross and net P&L are reported separately. Aggregate trades do not expose the book, queue position or exact executable spread, so this is a sensitivity analysis rather than a fill simulator. The final holdout remains sealed.
