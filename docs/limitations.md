# Limitations

- Aggregate trades do not expose the full order book, quoted spread or queue position.
- Signed aggregate-trade flow is not limit-order-book order-flow imbalance.
- A later execution model will be a latency/cost sensitivity analysis, not an exact
  fill simulator.
- Binance-specific findings may not generalise across venues or market regimes.
- No predictive or trading result exists at the scaffold milestone.
