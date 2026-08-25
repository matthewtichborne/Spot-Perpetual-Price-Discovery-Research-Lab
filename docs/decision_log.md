# Decision log

## 2026-08-25 — Proceed after Phase 0 feasibility spike

- **Decision:** Proceed with Binance public daily aggregate-trade archives and begin
  the repository scaffold.
- **Evidence:** Credential-free downloads succeeded, official SHA-256 checksums
  matched, and BTC/ETH spot/perpetual archives overlapped on two consecutive dates.
- **Consequence:** Parsing must be market-specific. Timestamp units are validated
  before conversion to nanoseconds because sampled 2025 spot data is in microseconds
  and sampled USD-M futures data is in milliseconds.

## 2026-08-25 — Keep the final holdout sealed

- **Decision:** `configs/final.yaml` is a schema-valid placeholder marked `sealed`;
  it does not authorise holdout download or evaluation.
- **Reason:** The hypothesis, feature set, splits and model-selection rule have not
  yet been frozen.

## 2026-08-25 — Use a small typed Phase 1 core

- **Decision:** Start with strict Pydantic/YAML configuration, Typer CLI, pure archive
  URL construction and timestamp-scale validation.
- **Reason:** These components make the feasibility evidence executable and testable
  without prematurely adding the modelling stack or downloading data in CI.
