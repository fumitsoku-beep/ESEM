# 总管线架构与当前 SEM 子系统布局（ZH）

当前日期：2026-03-16  
适用分支：`main`

---

## 1. 本文档目的

本文档说明两件已经明确下来的事情：

1. 当前仓库的总体架构如何理解；
2. SEM 子系统在本次整理后处于什么位置。

这里不再讨论“是否还要继续保留旧 shim”一类历史迁移策略；当前仓库已经进入**新结构生效后的状态**。

---

## 2. 当前结论

一句话结论：

- 项目继续采用“共享层 + 方法子系统 + 顶层公共 API”的结构；
- `SEM` 的直接实现层已经统一到 `psysem.sem.*`；
- 顶层 `psysem` 继续作为稳定 public API；
- 预留的 `pipeline/` 目录当前仍是架构骨架，不承载运行时主逻辑。

---

## 3. 当前推荐的架构分层

### 3.1 共享层

共享层继续放在 `src/psysem/` 根目录，主要包括：

1. `data/`
2. `preprocessing/`
3. 顶层 `__init__.py`
4. `_version.py`

这些模块服务于多个方法子系统，不属于某一个单独方法。

### 3.2 方法子系统层

当前方法子系统包括：

1. `efa/`
2. `sem/`
3. `esem/`

其中：

- `efa/` 负责 EFA 工作流与方法实现；
- `sem/` 负责 SEM 主实现；
- `esem/` 负责 ESEM workflow、候选、judge/selector 方向的演进。

### 3.3 顶层公共 API 层

顶层 `psysem` 负责：

1. 暴露稳定对外接口；
2. 统一 public API 风格；
3. 对用户隐藏内部目录收口细节。

---

## 4. 当前 SEM 布局

### 4.1 当前直接模块入口

当前 SEM 的直接模块入口统一为：

1. `psysem.sem.core`
2. `psysem.sem.model`
3. `psysem.sem.fit_indices`
4. `psysem.sem.result`
5. `psysem.sem.reporting`
6. `psysem.sem.parameter_index`
7. `psysem.sem.estimation`
8. `psysem.sem.inference`
9. `psysem.sem.measurement`
10. `psysem.sem.structural`

### 4.2 顶层稳定 API

虽然内部实现已经收口到 `sem/`，但顶层 `psysem` 仍保留以下稳定导出：

1. `SEMModel`
2. `sem`
3. `SEMResult`
4. `parse_model`
5. `to_markdown`
6. `SEMFitConfig`
7. `estimate_parameter_inference`
8. `compute_basic_fit_indices`
9. `build_measurement_design`
10. `build_structural_design`
11. `build_parameter_index_map`

因此：

- 用户侧建议优先使用 `from psysem import ...`；
- 需要直接进入实现层时，使用 `psysem.sem.*`。

### 4.3 已不再保留的历史路径

以下旧路径已不再作为当前版本支持面：

1. `psysem.core`
2. `psysem.model`
3. `psysem.fit_indices`
4. `psysem.result`
5. `psysem.reporting`
6. `psysem.parameter_index`
7. `psysem.estimation`
8. `psysem.inference`
9. `psysem.measurement`
10. `psysem.structural`

文档、测试、内部依赖都应以新的 `psysem.sem.*` 路径为准。

---

## 5. `pipeline/` 目录的当前定位

`src/psysem/pipeline/` 当前仍然是轻量骨架，主要用于：

1. 为未来统一编排层预留命名空间；
2. 避免后续继续把跨方法编排逻辑散落到各处；
3. 提前明确“编排层”和“数学实现层”分离的方向。

当前它**不是**实际运行主路径的一部分，也不替代 `efa/`、`sem/`、`esem/` 各自的方法实现。

---

## 6. 当前代码入口

### 6.1 共享/顶层

1. `src/psysem/__init__.py`
2. `src/psysem/data/`
3. `src/psysem/preprocessing/`

### 6.2 EFA

1. `src/psysem/efa/workflow.py`
2. `src/psysem/efa/fit.py`
3. `src/psysem/efa/evaluation.py`
4. `src/psysem/efa/interpretation.py`

### 6.3 SEM

1. `src/psysem/sem/core.py`
2. `src/psysem/sem/model.py`
3. `src/psysem/sem/estimation/`
4. `src/psysem/sem/inference/`
5. `src/psysem/sem/measurement/`
6. `src/psysem/sem/structural/`

### 6.4 ESEM

1. `src/psysem/esem/workflow.py`
2. `src/psysem/esem/contracts.py`

---

## 7. 后续建议

当前架构整理已经完成，后续更值得投入的方向是：

1. 建立正式 benchmark；
2. 继续增强 SEM 数值稳健性；
3. 扩展 ESEM generator / judge / selector；
4. 在 `pipeline/` 中逐步落地统一编排层，而不是继续做目录迁移。

---

## 8. 一句话结论

当前仓库已经不处于“SEM 迁移中”状态，而是处于：

- **SEM 已收口到 `psysem.sem.*`**；
- **顶层 `psysem` 继续提供稳定 public API**；
- **后续重点应从迁移转向 benchmark、数值基线和工作流扩展**。
