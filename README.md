# psysem

面向心理测量场景的 Python EFA/ESEM/SEM 包（当前为 `alpha` 阶段）。

> `psysem` is now publicly available as an alpha-stage project.
> EFA is currently the most usable workflow, while SEM remains prototype-level and full ESEM workflow is still under development.

`psysem` 的目标是把心理学里常见的分析链路拆成可复现、可测试、可扩展的模块：

`data -> efa -> (esem measurement) -> sem structural`

---

## 当前适合什么场景

- 你需要在 Python 中完成 EFA 诊断、因子数建议、候选模型比较和解释输出。
- 你希望先把 ESEM/SEM 输入规范（`spec`）严格校验，减少后续拟合报错。
- 你希望用一个统一入口先跑通 SEM 原型（语法解析、参数索引、ML 优化重启、基础拟合指标）。

## 当前边界（请先了解）

- 完整心理学 ESEM 主流程仍在建设中，`esem measurement` 还未全部落地。
- `MLR/WLSMV` 在 SEM 估计层面仍是后续里程碑（当前主原型路径是 ML）。
- 多组不变性 API 仍是占位接口，尚未提供完整可用实现。

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

### 3) 跑一个 SEM 原型模型

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
- `python examples/basic_sem.py`

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
- [参数总览（ZH）](docs/parameters.zh-CN.md)
- [EFA Phase 1 实施文档（ZH）](docs/efa-phase1-implementation.zh-CN.md)
- [EFA 测试与质量门禁（ZH）](docs/efa-testing.zh-CN.md)
- [SEM 分阶段实施（ZH）](docs/sem-phase-implementation.zh-CN.md)
- [SEM 下一步计划（ZH）](docs/sem-next-steps.zh-CN.md)

---

## 路线图（简版）

1. 完成 `esem measurement` 组装与估计主链路。
2. 打通 `EFA -> ESEM measurement -> SEM structural` 统一 pipeline。
3. 增加 target rotation、ordinal 场景和不变性流程。
4. 完善报告导出与复现元数据。

---

## 许可证

MIT
