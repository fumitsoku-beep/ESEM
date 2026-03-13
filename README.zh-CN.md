# psysem（中文说明）

`psysem` 是一个面向心理学 SEM/ESEM 分析的 Python 包。

对应英文文档：[`README.md`](README.md)

## 目标流程（ESEM）

1. 提供宽表 `pandas.DataFrame`（一行一个被试）
2. 提供模型规格 `spec`（字典或 dataclass 结构）
3. 运行拟合（估计器 + 旋转设置）
4. 输出载荷、因子相关、拟合指标和参数表

## 可配置参数（重点）

### 顶层 `spec` 参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `blocks` | `list[dict]` | 是 | - | ESEM 分块列表。 |
| `estimator` | `str` | 是 | - | 估计方法。当前支持：`ML`、`MLR`、`WLSMV`。 |
| `variable_types` | `dict[str, str]` | 是 | - | 变量类型映射。取值：`continuous`、`ordinal`。 |
| `rotation` | `dict` | 否 | `None` | 全局旋转设置（可被 block 内的 rotation 覆盖）。 |
| `structural` | `list[str]` | 否 | `[]` | 结构路径表达式（目标行为）。 |
| `group` | `str` | 否 | `None` | 多组分析分组列。 |
| `weight` | `str` | 否 | `None` | 抽样权重列。 |
| `cluster` | `str` | 否 | `None` | 聚类样本列。 |
| `id` | `str` | 否 | `None` | 被试 ID 列。 |
| `allow_item_overlap` | `bool` | 否 | `False` | 是否允许同一题项出现在多个 block。 |

### `blocks[]` 参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `name` | `str` | 是 | - | block 名称（必须唯一）。 |
| `items` | `list[str]` | 是 | - | 本 block 使用的题项列名。 |
| `n_factors` | `int` | 是 | - | 因子个数，必须 `> 0` 且 `< len(items)`。 |
| `rotation` | `dict` | 否 | `None` | block 级旋转设置（覆盖全局设置）。 |

### `rotation` 参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `method` | `str` | 是 | - | 旋转方法名，例如 `geomin`、`target`。 |
| `oblique` | `bool` | 否 | `True` | 是否允许因子相关。 |

## `spec` 示例

```python
spec = {
    "blocks": [
        {
            "name": "internalizing",
            "items": ["i1", "i2", "i3", "i4", "i5", "i6"],
            "n_factors": 2,
        }
    ],
    "estimator": "WLSMV",
    "rotation": {"method": "geomin", "oblique": True},
    "variable_types": {
        "i1": "ordinal",
        "i2": "ordinal",
        "i3": "ordinal",
        "i4": "ordinal",
        "i5": "ordinal",
        "i6": "ordinal",
    },
    "group": "gender",
}
```

## 当前实现状态

- 已实现：`spec` 数据结构与基础校验
- 已实现：EFA 提取法 `paf`、`pca`、`minres`
- 已实现：EFA 旋转法 `none`、`varimax`、`promax`、`oblimin`、`geomin`、`target`
- 已实现：`target rotation` 的目标矩阵、目标权重、多起点重启与随机种子控制
- 已实现：`SEMModel.fit(data, spec=...)` 主入口（ML 原型路径）
- 新增：`run_esem_workflow(...)` 最小可跑路径（`block_full` 候选）
- 后续：多候选 generator/judge/selector、更完整 ESEM 估计器

## 当前 EFA `target rotation` 说明

当使用 `rotation="target"` 时，当前版本支持：

- `rotation_target`：目标矩阵，有限值表示目标值，`NaN` 表示自由位置；
- `rotation_target_weights`：目标权重矩阵，可区分强约束、弱引导与自由位置；
- `rotation_restarts`：多起点重启次数；
- `random_state`：用于保证重启可复现。

这使得当前 EFA 已可以表达较基础的 target-pattern 旋转需求，并为后续 ESEM block workflow 做准备。

## 最小可跑 ESEM 入口（MVP）

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
```

预期输出结构（示例）：

```text
Best candidate: block_full
Comparison table: (含 candidate_id / total_score / cfi / rmsea / srmr ...)
SEM summary: (Converged, Estimator, Fit indices, Warnings)
```
