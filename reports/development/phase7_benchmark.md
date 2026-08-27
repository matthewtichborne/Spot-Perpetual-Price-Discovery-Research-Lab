# Phase 7 C++ replay benchmark

- Events: 2,000,000 total (1,000,000 per market)
- Grid rows: 36,000
- Python median: 3.568878 s (560,400 events/s)
- C++ median: 0.043651 s (45,818,096 events/s)
- Bounded-kernel speed-up: 81.76x
- Parity: passed at rtol/atol 1e-12
- Compiler: `Apple LLVM 17.0.0 (clang-1700.3.19.1)`

This compares identical in-memory merge and fixed-grid aggregation work. It is not a comparison with Polars and does not represent end-to-end research pipeline speed-up.
