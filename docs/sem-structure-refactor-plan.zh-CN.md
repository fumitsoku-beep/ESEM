# SEM 结构收口完成说明（ZH）

当前日期：2026-03-16  
适用分支：`main`

---

## 1. 本文档目的

本文档用于记录本次 SEM 结构整理已经完成后的状态：

1. 当前哪些路径是正式有效的；
2. 哪些旧路径已经被移除；
3. 本次收口实际改动了什么；
4. 仓库当前验证状态如何。

---

## 2. 当前有效路径

### 2.1 顶层公共 API

以下接口仍建议从顶层 `psysem` 导入：

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

### 2.2 SEM 直接实现层

如果需要直接使用内部实现模块，当前应使用：

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

---

## 3. 已移除的旧路径

本次整理后，以下旧路径不再作为当前版本支持面：

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

对应地，以下旧文件/旧包 shim 已被删除：

1. `src/psysem/core.py`
2. `src/psysem/model.py`
3. `src/psysem/fit_indices.py`
4. `src/psysem/result.py`
5. `src/psysem/reporting.py`
6. `src/psysem/parameter_index.py`
7. `src/psysem/estimation/`
8. `src/psysem/inference/`
9. `src/psysem/measurement/`
10. `src/psysem/structural/`

---

## 4. 本次收口的实际内容

### 4.1 代码结构

SEM 真实实现已统一位于：

1. `src/psysem/sem/core.py`
2. `src/psysem/sem/model.py`
3. `src/psysem/sem/fit_indices.py`
4. `src/psysem/sem/result.py`
5. `src/psysem/sem/reporting.py`
6. `src/psysem/sem/parameter_index.py`
7. `src/psysem/sem/estimation/`
8. `src/psysem/sem/inference/`
9. `src/psysem/sem/measurement/`
10. `src/psysem/sem/structural/`

### 4.2 顶层导出

`src/psysem/__init__.py` 已切换为从 `psysem.sem.*` 重导出 SEM 相关对象。

### 4.3 下游依赖修正

`esem` 子系统中对旧 SEM 路径的残留依赖已修复，重点包括：

1. `src/psysem/esem/contracts.py`
2. `src/psysem/esem/workflow.py`

### 4.4 测试与文档同步

测试与文档已统一切换到新的路径语义，重点包括：

1. `tests/test_sem_import_compatibility.py`
2. `tests/test_sem_fit_indices.py`
3. `tests/test_sem_estimation_config.py`
4. `docs/pipeline-architecture-plan.zh-CN.md`
5. `docs/current-testing-workflows.zh-CN.md`

---

## 5. 当前 API 约束

### 5.1 用户侧建议

建议优先使用：

```python
from psysem import SEMModel, SEMFitConfig, parse_model
```

当需要直接引用实现层时，再使用：

```python
from psysem.sem.model import ModelSpec
from psysem.sem.estimation import optimize_ml_parameters
```

### 5.2 文档约束

从当前版本开始：

1. 文档中不再把旧 shim 路径当作有效支持面；
2. 文档中涉及 SEM 模块路径时，应优先写 `psysem.sem.*`；
3. 面向用户的示例优先使用顶层 `psysem` public API。

---

## 6. 验证结果

本次整理后的验证结果如下：

1. 已完成全量测试；
2. 测试结果：`193 passed`；
3. 说明当前新结构已经通过仓库级回归验证。

---

## 7. 后续建议

结构收口已经完成，后续重点不再是“继续搬目录”，而是：

1. 建立 benchmark；
2. 强化数值稳定性；
3. 扩展 ESEM 工作流；
4. 整理研究者可直接使用的结果与报告输出。

---

## 8. 一句话结论

当前仓库已经完成 SEM 结构收口：

- **顶层 `psysem` 保留稳定 public API；**
- **`psysem.sem.*` 成为唯一直接实现路径；**
- **旧 shim 路径已不再保留。**
