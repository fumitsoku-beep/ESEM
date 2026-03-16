# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog, and this project currently uses pre-release versioning during the alpha stage.

## [Unreleased]

### Added

- SEM benchmark raw CSV assets and provenance metadata under `tests/data/`.
- Initial automated SEM benchmark tests for `HolzingerSwineford1939` and `PoliticalDemocracy`.
- Initial boundary benchmark tests covering invalid degrees-of-freedom and small-sample warning semantics.

### Changed

- Benchmark planning and testing workflow docs updated to reflect current SEM benchmark coverage.
- 中文 README 已补充当前 SEM benchmark 落地状态说明。

## [0.1.0a1] - 2026-03-16

### Changed

- 明确 `psysem.sem.*` 为 SEM 子系统的唯一直接模块入口。
- 顶层 `psysem` 继续保留稳定 public API re-export。
- ESEM 相关实现已改为依赖新的 `psysem.sem.*` 路径。
- 多份文档已更新为“迁移完成后的当前状态”表述。

### Removed

- 删除旧的根层 SEM shim 文件：`core.py`、`model.py`、`fit_indices.py`、`result.py`、`reporting.py`、`parameter_index.py`。
- 删除旧的包级 SEM shim 目录内容：`estimation/`、`inference/`、`measurement/`、`structural/` 下的兼容实现。

### Fixed

- 修复 `esem` 子系统中对旧 SEM 路径的残留依赖。
- 修复测试与说明文档中对旧导入路径的引用。

### Validation

- 全量测试通过：`193 passed`。

### Added

- Initial runnable ESEM MVP workflow entry: `run_esem_workflow(...)`.
- New `psysem.esem` module contracts for workflow/candidate/judge results.
- Example script: `examples/basic_esem.py`.
- Integration tests for ESEM MVP workflow path.
- Documentation page: `docs/esem-mvp-run.zh-CN.md`.

## [0.1.0a0] - 2026-03-12

### Added

- Initial alpha package structure for `psysem`.
- `psysem.data` validation flow for ESEM specs and input data.
- EFA workflow modules for diagnostics, factor-count suggestion, candidate comparison, and interpretation.
- SEM prototype pipeline covering syntax/spec normalization, measurement/structural design scaffolding, ML prototype optimization, inference prototype, and basic fit indices.
- Test suite, CI workflow, and project documentation skeleton.

### Notes

- EFA is currently the most usable analysis path.
- SEM remains prototype-level.
- Full ESEM workflow is under active design and implementation.
