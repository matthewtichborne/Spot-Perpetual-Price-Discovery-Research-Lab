# Final holdout opening record

Status: **opened and evaluated exactly once on 2026-08-27**.

## Frozen inputs

- Final period: 2025-02-21 through 2025-03-02.
- Final configuration SHA-256:
  `a4c8dfaf5ca2af0087966844b753b2a65573dd096d5c4c6e836b589cceb1f504`.
- Phase 5 protocol SHA-256:
  `53a36afa5ab52c9d23194b69d405c8e3410ba1efca830b26dcdb5c0e49883013`.
- Confirmation trade-manifest hash:
  `def139205d8dbdc3d44b168e026e4548b2676d25ea4a0b950ac2a68aa403aa1b`.
- Confirmation feature-manifest hash:
  `6700a50fa80ea52c01448696d547bc9149eed0d1b5ea997cfda80524a714d523`.
- Phase 5 result-manifest hash:
  `cb6ba4b23fd5e647a2e29ab5eee1169c5c53049b6b80b40911f2ac435390b97f`.
- Final model-specification hash:
  `fe69b128d018481b8db3e663668e012f3d5bd964f68316d9d0d1fde36deac570`.
- Phase 8 protocol SHA-256:
  `66663593bbcf03a7cda8ae7cb05e699473e21d15e2ec9fa7d1e3d991cb244530`.
- Phase 8 evaluator SHA-256:
  `43342ca12410d1a3932c440f6bb5ba57df25843f3950e84ddf292587ae2342fd`.
- One-time open configuration SHA-256:
  `68ed690e419722d2c83a3559a583942b8467c786e48237ffef00600ea4a27ece`.

## Locked evaluation

The final model is XGBoost/expanded for the five-second BTC spot-return target, with
depth 2, 200 trees and the remaining parameters in
`data/manifests/final-model-specification.json`. It will be refit once on the
registered every-fifth-row sample from 2025-01-02 through 2025-02-20, then score all
eligible final-period rows. No tuning, feature changes, calibration or alternative
selection is permitted after opening.

## Opening state

- `configs/final.yaml` remains `holdout_status: sealed`.
- No final-period raw, processed, feature or report artifact was present when this
  record was prepared.
- Phase 5 reports record zero model failures.
- Phase 6 execution analysis and Phase 7 replay work are complete. The user's Phase 8
  request authorises the sole registered opening. The evaluator passed its synthetic
  one-time and future-fill test before access; no final artifact existed at freeze time.

## Completed opening

- Opening timestamp: `2026-08-27T09:45:38.933464+00:00`.
- Final trade-manifest hash:
  `708de4ad209ce9f2fd96450c45a9b93ee97532e96111f12311dbcce885198d0d`.
- Final feature-manifest hash:
  `f644321249f0fb5611a5f9e5284b4be5c7428699ee55e16d6b926f82c0662b4d`.
- Final evaluation-manifest hash:
  `3c2edcd0787106eb990af0d44ca7ffd8922cdead33c87d3cf2f669931a12d0f7`.
- The one-time guard is active: a second invocation is refused while the final
  evaluation manifest exists.
