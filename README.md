# psysem

`psysem` 是一个面向心理测量场景的 Python SEM/ESEM 包（当前为早期 alpha 阶段）。

目标是把 ESEM 拆成清晰模块并逐步落地：

1. `data`：输入契约与数据校验
2. `efa`：可扩展的探索性因子提取/旋转
3. `esem_measurement`（规划中）：多 block 测量模型组装
4. `sem_structural`（规划中）：结构路径层与完整估计

---

## 当前状态（2026-03）

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| `psysem.data` | 可用 | `spec` 解析 + 规则校验 + 与 `DataFrame` 对齐校验 |
| `psysem.efa` | 可用（Phase 1） | `PAF/PCA` 提取、`varimax/none`、KMO/Bartlett、自动因子数建议（PA/MAP/Scree/Kaiser） |
| `SEMModel` | 占位接口 | 已有 `fit` 入口，完整 SEM/ESEM 估计器尚未接入 |
| `psysem.esem_spec` | 兼容层 | 旧导入路径，内部已转发到 `psysem.data` |

---

## 流程图（当前实现）

### 1) 总流程（模块视角）

![psysem current overview](docs/assets/images/flow-overview-current.zh-CN.png)

### 2) `data` 模块流程（spec 解析与校验）

![psysem data validation flow](docs/assets/images/flow-data-validation.zh-CN.png)

### 3) `efa` 模块流程（fit_efa）

![psysem efa flow](docs/assets/images/flow-efa-current.zh-CN.png)

---

## 安装

```bash
pip install -e .[dev]
```

开发环境常用命令：

```bash
ruff check .
mypy src
pytest
```

---

## 快速开始

### 1) SEM 占位拟合（当前 smoke API）

```python
import pandas as pd
from psysem import SEMModel

data = pd.DataFrame(
    {
        "x1": [1.0, 2.0, 3.0],
        "x2": [1.1, 2.1, 3.1],
        "y": [0.9, 2.0, 3.2],
    }
)

model = SEMModel("y ~ x1 + x2")
result = model.fit(data)
print(result.summary())
```

### 2) ESEM `spec` 解析与校验（`data` 模块）

```python
import pandas as pd
from psysem.data import esem_spec_from_dict, validate_esem_spec

payload = {
    "blocks": [
        {
            "name": "internalizing",
            "items": ["i1", "i2", "i3", "i4"],
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
```

### 3) EFA 拟合（`efa` 模块）

```python
import pandas as pd
from psysem import EFAConfig, fit_efa

data = pd.DataFrame(
    {
        "i1": [1.2, 1.5, 0.8, 1.7],
        "i2": [1.1, 1.3, 0.9, 1.8],
        "i3": [1.0, 1.4, 0.7, 1.6],
        "i4": [0.5, 0.9, 1.8, 1.7],
        "i5": [0.6, 1.0, 1.7, 1.8],
        "i6": [0.4, 0.8, 1.9, 1.6],
    }
)

config = EFAConfig(
    items=("i1", "i2", "i3", "i4", "i5", "i6"),
    n_factors=2,
    extraction="paf",    # paf | pca
    rotation="varimax",  # varimax | none
)

result = fit_efa(data, config)
print(result.loadings)
print(result.explained_variance)
```

### 4) EFA 诊断与自动建议因子数（Phase 1）

```python
import pandas as pd
from psysem import (
    EFADiagnosticsConfig,
    FactorSelectionConfig,
    run_efa_diagnostics,
    suggest_n_factors,
)

data = pd.read_csv("examples/data/efa_demo_input.csv")
items = tuple(data.columns)

diag = run_efa_diagnostics(data, EFADiagnosticsConfig(items=items))
selection = suggest_n_factors(
    data,
    FactorSelectionConfig(
        items=items,
        n_min=1,
        n_max=4,
        pa_iter=200,
        random_state=42,
    ),
)

print(diag.kmo_total, diag.kmo_label)
print(diag.bartlett_chi2, diag.bartlett_df, diag.bartlett_p)
print(selection.suggestions_by_method)
print(selection.consensus_n_factors)
```

---

## ESEM 输入契约（当前实现）

### 顶层字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `blocks` | `list[dict]` | 是 | ESEM block 列表 |
| `estimator` | `str` | 是 | 支持：`ML`、`MLR`、`WLSMV` |
| `variable_types` | `dict[str, str]` | 是 | 变量类型映射：`continuous`、`ordinal` |
| `rotation` | `dict` | 否 | 全局旋转设置 |
| `structural` | `list[str]` | 否 | 结构路径字符串列表 |
| `group` | `str` | 否 | 分组列名 |
| `weight` | `str` | 否 | 权重列名 |
| `cluster` | `str` | 否 | 聚类列名 |
| `id` | `str` | 否 | 个体 ID 列名 |
| `allow_item_overlap` | `bool` | 否 | 是否允许题项跨 block 重复 |

### `blocks[]` 字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `name` | `str` | 是 | block 名称（必须唯一） |
| `items` | `list[str]` | 是 | 题项列名列表 |
| `n_factors` | `int` | 是 | 因子个数，需满足 `0 < n_factors < len(items)` |
| `rotation` | `dict` | 否 | block 级旋转设置（覆盖全局） |

### `rotation` 字段

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `method` | `str` | 是 | - | 旋转方法名 |
| `oblique` | `bool` | 否 | `True` | 是否允许因子相关 |

---

## `data` 模块当前做了哪些校验

- `blocks` 非空，`name/items/n_factors` 合法
- `estimator` 在支持集合内
- `variable_types` 的 key/value 格式合法且类型值受限
- 未开启 `allow_item_overlap` 时，禁止题项跨 block 重复
- `structural` 中的观测变量必须在 `variable_types` 中声明
- 传入 `data` 时，校验：
  - block 题项列必须存在
  - `group/weight/cluster/id` 若声明则必须存在
  - `structural` 中观测变量列必须存在
  - 标记为 `ordinal` 的列必须是“有序整数型”数值

---

## EFA 参数与可扩展机制

`EFAConfig` 主要参数：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `items` | `tuple[str, ...]` | - | 用于 EFA 的题项列 |
| `n_factors` | `int` | - | 提取因子数 |
| `extraction` | `str` | `paf` | 提取方法名（注册表 key） |
| `rotation` | `str` | `varimax` | 旋转方法名（注册表 key） |
| `max_iter` | `int` | `200` | 迭代上限（如 PAF） |
| `tol` | `float` | `1e-6` | 收敛阈值 |
| `min_uniqueness` | `float` | `0.005` | 唯一性下界 |

`EFADiagnosticsConfig` 主要参数：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `items` | `tuple[str, ...]` | - | 用于诊断的题项列 |
| `dropna` | `bool` | `True` | 是否在诊断前删除缺失行 |
| `min_sample_ratio` | `float` | `5.0` | 最低样本比阈值（`n_obs / n_items`） |

`FactorSelectionConfig` 主要参数：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `items` | `tuple[str, ...]` | - | 用于因子数建议的题项列 |
| `n_min` | `int` | `1` | 搜索最小因子数 |
| `n_max` | `int \| None` | `None` | 搜索最大因子数（默认 `n_items - 1`） |
| `pa_iter` | `int` | `500` | Parallel Analysis 重采样次数 |
| `pa_percentile` | `float` | `0.95` | PA 分位阈值 |
| `random_state` | `int \| None` | `None` | 随机种子 |
| `enable_pa` | `bool` | `True` | 启用 PA |
| `enable_map` | `bool` | `True` | 启用 MAP |
| `enable_kaiser` | `bool` | `True` | 启用 Kaiser |
| `enable_scree` | `bool` | `True` | 启用 Scree 拐点建议 |

可注册自定义方法：

```python
import numpy as np
from psysem import (
    EFAConfig,
    register_extraction_method,
    register_rotation_method,
    list_extraction_methods,
    list_rotation_methods,
)

def my_extraction(corr: np.ndarray, config: EFAConfig):
    p = corr.shape[0]
    loadings = np.zeros((p, config.n_factors), dtype=float)
    communalities = np.zeros(p, dtype=float)
    return loadings, communalities, 1, True

def my_rotation(loadings: np.ndarray, _: EFAConfig):
    return loadings

register_extraction_method("my_extraction", my_extraction, overwrite=True)
register_rotation_method("my_rotation", my_rotation, overwrite=True)

print(list_extraction_methods())
print(list_rotation_methods())
```

---

## 项目结构（核心部分）

```text
src/psysem/
  data/
    contracts.py
    parser.py
    validator.py
    validators/
  efa/
    fit.py
  core.py
  model.py
  result.py
  reporting.py
```

---

## 兼容导入路径

旧路径仍可用：

```python
from psysem.esem_spec import esem_spec_from_dict, validate_esem_spec
```

推荐新路径：

```python
from psysem.data import esem_spec_from_dict, validate_esem_spec
```

---

## Roadmap

1. `esem_measurement`：基于 block 的测量层组装与估计接口
2. `sem_structural`：更严格结构语法与参数估计
3. 扩展 EFA/ESEM 的提取和旋转方法（插件化管理）
4. 报告层增强（fit 指标、参数表、导出）

EFA 分阶段实施文档（详细步骤）：

- [docs/efa-phase1-implementation.zh-CN.md](docs/efa-phase1-implementation.zh-CN.md)

---

## SEM 模块拆分思路（TODO）

下面是面向 `SEMModel.fit(...)` 的推荐分层，目标是把“语法、矩阵、优化、统计推断、报告”解耦，便于逐模块迭代和测试。

### 1) 模块边界（建议）

| 模块 | 主要职责 | 输入 | 输出 |
| --- | --- | --- | --- |
| `psysem.syntax` | 解析模型语法（测量/结构/约束） | 语法字符串或 `spec` | 标准化 AST/ModelSpec |
| `psysem.data` | 数据契约与预校验 | `DataFrame` + spec | 校验后的分析视图与元信息 |
| `psysem.measurement` | 测量层组装（CFA/ESEM block） | AST + 数据视图 | 测量模型矩阵定义 |
| `psysem.structural` | 结构路径层组装 | AST + latent/observed 索引 | 结构模型矩阵定义 |
| `psysem.estimation` | 目标函数、优化器、收敛控制 | 模型矩阵 + 数据矩阵 + estimator | 参数估计值 + 收敛状态 |
| `psysem.inference` | 标准误、检验、置信区间 | 参数估计 + 信息矩阵 | SE/z/p/CI |
| `psysem.fit` | 拟合指标计算 | 样本统计量 + 模型 implied 统计量 | CFI/TLI/RMSEA/SRMR/AIC/BIC |
| `psysem.reporting` | 结果对象和导出格式 | 参数表 + 拟合指标 + 警告 | `SEMResult`/Markdown/表格 |

### 2) 推荐目录（目标形态）

```text
src/psysem/
  syntax/
  data/
  measurement/
  structural/
  estimation/
  inference/
  fit/
  reporting/
  core.py
  result.py
```

### 3) TODO（按阶段）

#### Phase 1: 入口与契约统一

- [ ] 让 `SEMModel.fit(data, spec=...)` 支持显式 `ESEMSpec` 输入
- [ ] 统一 `syntax` 与 `spec` 两条入口到同一内部 `ModelSpec`
- [ ] 在 `SEMResult` 中增加标准字段占位：`warnings_`、`parameter_table_`、`fit_indices_`

#### Phase 2: 测量层（Measurement）

- [ ] 新建 `psysem.measurement`，支持单 block CFA/ESEM 的矩阵构造
- [ ] 支持多 block 组装与 block 级旋转覆盖
- [ ] 增加识别性检查（因子尺度设定、自由参数计数）

#### Phase 3: 结构层（Structural）

- [ ] 新建 `psysem.structural`，实现 latent/observed 回归路径映射
- [ ] 加入结构路径合法性检查（循环依赖、未知变量、重复路径）
- [ ] 输出统一参数索引，供估计层直接使用

#### Phase 4: 估计与推断

- [ ] 新建 `psysem.estimation`：先落地 ML/MLR，再扩展到 WLSMV
- [ ] 新建 `psysem.inference`：信息矩阵、SE、z/p、区间估计
- [ ] 收敛与数值稳定策略（初值、边界约束、容错与告警）

#### Phase 5: 指标与报告

- [ ] 新建 `psysem.fit`：CFI/TLI/RMSEA/SRMR/AIC/BIC 实现
- [ ] `reporting` 输出统一参数表和模型摘要
- [ ] 增加导出接口（Markdown/CSV）与可复现元数据

#### Phase 6: 测试与质量门禁

- [ ] 按模块补齐单元测试：`syntax/data/measurement/structural/estimation`
- [ ] 增加端到端测试：单组、多组、含序数变量
- [ ] 建立数值回归测试（固定种子 + 近似阈值）

### 4) 优先级建议

1. 先完成 `Phase 1-2`（能跑通 measurement-only 流程）
2. 再做 `Phase 3-4`（结构层与估计器）
3. 最后补齐 `Phase 5-6`（指标、报告、质量门禁）

---

## License

MIT
