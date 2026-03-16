# psysem

面向心理测量场景的 Python EFA/ESEM/SEM 包（当前为 `alpha` 阶段）。

> `psysem` is now publicly available as an alpha-stage project.
> EFA is currently the most usable workflow, while SEM remains prototype-level and full ESEM workflow is still under development.

`psysem` 的目标是把心理学里常见的分析链路拆成可复现、可测试、可扩展的模块：

`data -> efa -> (esem measurement) -> sem structural`

---

## 当前适合什么场景

- 你需要在 Python 中完成 EFA 诊断、因子数建议、候选模型比较和解释输出。
- 你希望在 EFA 中直接使用 `minres`、`promax`、`oblimin`、`geomin`、`target rotation` 等常见方法。
- 你希望先把 ESEM/SEM 输入规范（`spec`）严格校验，减少后续拟合报错。
- 你希望用一个统一入口先跑通 SEM 原型（语法解析、参数索引、ML 优化重启、基础拟合指标）。

## 当前边界（请先了解）

- 完整心理学 ESEM 主流程仍在建设中，`esem measurement` 还未全部落地。
- `MLR/WLSMV` 在 SEM 估计层面仍是后续里程碑（当前主原型路径是 ML）。
- 多组不变性 API 仍是占位接口，尚未提供完整可用实现。

## SEM 模块路径说明

- 顶层公共 API 继续使用 `from psysem import ...`。
- 如果需要直接导入 SEM 子系统实现，请使用 `psysem.sem.*`。
- 旧的 `psysem.core`、`psysem.model`、`psysem.fit_indices`、`psysem.estimation` 等兼容路径已移除，不再作为文档承诺面。

---

## 快速安装

```bash
pip install -e ".[dev]"
```

基础质量检查：

```bash
python -m ruff check .
python -m mypy src
python -m pytest -q
```

---

## 5 分钟上手

### 1) 一键跑 EFA 工作流（诊断 -> 选因子 -> 候选比较）

```python
from pathlib import Path

import pandas as pd

from psysem import (
    EFADiagnosticsConfig,
    EFAEvaluationConfig,
    EFAWorkflowConfig,
    FactorSelectionConfig,
    run_efa_workflow,
)

data = pd.read_csv(Path("examples/data/efa_demo_input.csv"))
items = tuple(data.columns)

workflow = run_efa_workflow(
    data,
    EFAWorkflowConfig(
        items=items,
        diagnostics=EFADiagnosticsConfig(items=items),
        selection=FactorSelectionConfig(items=items, n_min=1, n_max=4, pa_iter=200, random_state=42),
        evaluation=EFAEvaluationConfig(),
        extraction="paf",
        rotation="varimax",
    ),
)

print("best_n_factors:", workflow.best_n_factors)
print(workflow.comparison_table[["n_factors", "score"]])
print(workflow.best_model.loadings.round(3))
```

### 2) 校验 ESEM 输入规格（`spec` + `data`）

```python
import pandas as pd

from psysem.data import esem_spec_from_dict, validate_esem_spec

payload = {
    "blocks": [
        {"name": "internalizing", "items": ["i1", "i2", "i3", "i4"], "n_factors": 2}
    ],
    "estimator": "WLSMV",
    "rotation": {"method": "geomin", "oblique": True},
    "variable_types": {
        "i1": "ordinal",
        "i2": "ordinal",
        "i3": "ordinal",
        "i4": "ordinal",
        "wellbeing": "continuous",
    },
    "structural": ["wellbeing ~ internalizing_f1 + internalizing_f2"],
}

data = pd.DataFrame(
    {
        "i1": [1, 2, 3],
        "i2": [2, 2, 3],
        "i3": [1, 3, 4],
        "i4": [2, 2, 3],
        "wellbeing": [45.0, 50.0, 62.0],
    }
)

spec = esem_spec_from_dict(payload)
validate_esem_spec(spec, data)
print("spec validation passed")
```

### 3) 跑通一次 ESEM 最小工作流（新增）

```python
import pandas as pd

from psysem import ESEMWorkflowConfig, SEMFitConfig, run_esem_workflow

data = pd.read_csv("examples/data/efa_demo_input.csv")
items = list(data.columns)

spec_payload = {
    "blocks": [{"name": "demo", "items": items, "n_factors": 2}],
    "estimator": "ML",
    "variable_types": {item: "continuous" for item in items},
}

workflow = run_esem_workflow(
    data,
    spec_payload,
    ESEMWorkflowConfig(
        fit_config=SEMFitConfig(max_iter=200, restarts=1, random_seed=42),
    ),
)

print(workflow.best_candidate_id)
print(workflow.comparison_table)
print(workflow.best_candidate.sem_result.summary())
```

### 4) 跑一个 SEM 原型模型

```python
import pandas as pd

from psysem import SEMFitConfig, SEMModel

data = pd.DataFrame(
    {
        "x1": [1.0, 2.0, 3.0, 4.0],
        "x2": [1.2, 1.9, 3.2, 3.9],
        "y": [0.9, 2.1, 2.8, 4.2],
    }
)

model = SEMModel("y ~ x1 + x2")
result = model.fit(
    data,
    fit_config=SEMFitConfig(max_iter=300, tol=1e-7, restarts=2, random_seed=42),
)

print(result.summary())
print(result.fit_indices)
print(result.optimization_info)
```

更多示例：

- `python examples/basic_efa.py`
- `python examples/basic_esem.py`
- `python examples/basic_sem.py`

### 5) 直接运行带目标矩阵的 EFA 旋转

```python
import numpy as np
import pandas as pd

from psysem import EFAConfig, fit_efa

data = pd.read_csv("examples/data/efa_demo_input.csv")
items = tuple(data.columns[:6])

target = pd.DataFrame(
    [
        [np.nan, 0.0],
        [np.nan, 0.0],
        [np.nan, 0.0],
        [0.0, np.nan],
        [0.0, np.nan],
        [0.0, np.nan],
    ],
    index=items,
    columns=["F1", "F2"],
)

weights = pd.DataFrame(
    [
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [2.0, 1.0],
        [2.0, 1.0],
    ],
    index=items,
    columns=["F1", "F2"],
)

result = fit_efa(
    data,
    EFAConfig(
        items=items,
        n_factors=2,
        extraction="minres",
        rotation="target",
        rotation_target=target,
        rotation_target_weights=weights,
        rotation_restarts=5,
        random_state=42,
    ),
)

print(result.loadings.round(3))
print(result.factor_correlation.round(3))
print(result.warnings)
```

### 6) `basic_esem.py` 预期输出（示例）

实际数值会随数据和随机种子有轻微变化，但结构应类似：

```text
Best candidate: block_full
Comparison table:
candidate_id   strategy  converged  total_score  cfi  tli  rmsea  srmr  aic  bic  n_warnings
  block_full block_full       True       ...      ...  ...  ...    ...   ...  ...      ...

SEM summary:
SEM Fit Summary
Converged: True
Estimator: ml
Fit indices:
  cfi: ...
  tli: ...
  rmsea: ...
  srmr: ...
```

---

## 推荐分析流程（当前版本）

1. 准备数据（宽表，一行一个被试）。
2. 用 `psysem.data` 做 `spec` + 数据列一致性校验。
3. 用 `psysem.efa` 做 EFA 诊断和因子数建议（PA/MAP/Scree/Kaiser）。
4. 用 EFA 候选比较与解释输出，确定候选结构。
5. 进入 `SEMModel.fit(...)` 跑结构路径原型并查看拟合指标。

---

## 结果对象速览

### `fit_efa` / `run_efa_workflow`

- `loadings`, `communalities`, `uniquenesses`, `complexity`
- `residual_matrix`, `residual_summary`, `cross_loaded_items`, `warnings`
- 当前 EFA 已支持提取法：`paf`、`pca`、`minres`
- 当前 EFA 已支持旋转法：`none`、`varimax`、`promax`、`oblimin`、`geomin`、`target`
- `target rotation` 支持 `rotation_target`、`rotation_target_weights`、`rotation_restarts`、`random_state`
- 工作流结果中包含 `comparison_table`, `best_n_factors`, `best_interpretation`

### `SEMModel.fit`

- `parameter_table`, `parameter_index_map`
- `fit_indices`（基础指标）
- `parameter_inference`（数值推断原型）
- `optimization_info`（重启次数、收敛状态、失败分类等）

---

## 流程图（当前实现）

### 标准 ESEM 逐步流程（心理学基线）

![Standard ESEM step workflow](docs/assets/images/flow-esem-standard.zh-CN.png)

### psysem 逐步流程（当前实现）

![psysem step workflow](docs/assets/images/flow-psysem-step.zh-CN.png)

### 总体流程（模块视角）

![Overall flow](docs/assets/images/flow-overview-current.zh-CN.png)

### `data` 模块流程

![Data validation flow](docs/assets/images/flow-data-validation.zh-CN.png)

### `efa` 模块流程

![EFA flow](docs/assets/images/flow-efa-current.zh-CN.png)

---

## 文档索引

- [ESEM baseline 对比文档（ZH）](docs/esem-baseline-landscape.zh-CN.md)
- [ESEM 最小可跑路径（ZH）](docs/esem-mvp-run.zh-CN.md)
- [参数总览（ZH）](docs/parameters.zh-CN.md)
- [共享预处理模块抽取与落地（ZH）](docs/preprocessing-module-extraction.zh-CN.md)
- [EFA Phase 1 实施与测试（ZH）](docs/efa-phase1-implementation.zh-CN.md)
- [EFA 方法扩展路线图（ZH）](docs/efa-method-expansion-roadmap.zh-CN.md)
- [SEM 分阶段实施与下一阶段路线（ZH）](docs/sem-phase-implementation.zh-CN.md)

---

## 路线图（简版）

1. 基于已落地的共享 preprocessing，启动 `network` MVP。
2. 继续推进 ESEM 模块化路线：generator / judge / selector，以及 `efa_seeded`。
3. 继续补强 ordinal/稳健路径：`polychoric` 数值稳健性、mixed-type 扩展、`MLR/WLSMV` 与不变性。
4. 完善报告导出、复现元数据和文档一致性。

---

## 许可证

MIT
