# psysem Docs

This is the initial documentation placeholder for `psysem`.

## Current Flow Diagrams

### Standard ESEM step workflow

![Standard ESEM step workflow](assets/images/flow-esem-standard.zh-CN.png)

### `psysem` step workflow

![psysem step workflow](assets/images/flow-psysem-step.zh-CN.png)

### Overall module flow

![Overall flow](assets/images/flow-overview-current.zh-CN.png)

### `data` module flow

![Data validation flow](assets/images/flow-data-validation.zh-CN.png)

### `efa` module flow

![EFA flow](assets/images/flow-efa-current.zh-CN.png)

## EFA Implementation Docs

- [EFA Phase 1 Implementation (ZH)](efa-phase1-implementation.zh-CN.md)
- [EFA Testing and Quality Gate (ZH)](efa-testing.zh-CN.md)

## SEM Planning Docs

- [SEM Phase Implementation (ZH)](sem-phase-implementation.zh-CN.md)
- [SEM Next Steps (ZH)](sem-next-steps.zh-CN.md)

## ESEM Baseline Research

- [ESEM Ecosystem and Practice Baseline (ZH)](esem-baseline-landscape.zh-CN.md)

## Current scope

- Package skeleton
- Public API stubs
- Smoke tests and CI setup
- `psysem.data` module for ESEM spec/data validation
- `psysem.efa` Phase 1: diagnostics (KMO/Bartlett) and factor-count suggestion (PA/MAP/Scree/Kaiser)
- `psysem.efa` Phase 2 (basic): candidate fitting, model scoring, and best factor-count selection
- `psysem.efa` Phase 3 (in progress): modular interpretation output (`item_table`, `factor_table`, `residual_top_pairs`)
- Expanded EFA test suite: validation, edge cases, reproducibility, and workflow integration
- `psysem` SEM Phase 3 (batch 7): measurement/structural parameter-index mapping (`Lambda/Theta`, `Beta/Gamma/Psi`), fit-result integration, ML implied-covariance optimization prototype, numerical inference prototype (`SE/z/p/CI`), basic fit-index computation (`AIC/BIC/SRMR/CFI/TLI/RMSEA`), and optimization robustness prototype (`SEMFitConfig`/restarts/failure diagnostics)

## Next milestones

- ESEM measurement-layer assembly from block-level models
- Structural SEM layer over measurement model
- Expanded rotations and extraction methods for EFA
- Psychology-focused reporting helpers
