# psysem Docs

This is the initial documentation placeholder for `psysem`.

## Current Flow Diagrams

### Overall module flow

![Overall flow](assets/images/flow-overview-current.zh-CN.png)

### `data` module flow

![Data validation flow](assets/images/flow-data-validation.zh-CN.png)

### `efa` module flow

![EFA flow](assets/images/flow-efa-current.zh-CN.png)

## EFA Implementation Docs

- [EFA Phase 1 Implementation (ZH)](efa-phase1-implementation.zh-CN.md)

## Current scope

- Package skeleton
- Public API stubs
- Smoke tests and CI setup
- `psysem.data` module for ESEM spec/data validation
- `psysem.efa` Phase 1: diagnostics (KMO/Bartlett) and factor-count suggestion (PA/MAP/Scree/Kaiser)

## Next milestones

- ESEM measurement-layer assembly from block-level models
- Structural SEM layer over measurement model
- Expanded rotations and extraction methods for EFA
- Psychology-focused reporting helpers
