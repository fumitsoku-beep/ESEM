# EFA Phase 实施文档（ZH）

本文档用于指导 `psysem` 的 EFA 能力从“当前基础实现”升级到“可诊断、可自动选因子数、可调参”的工程版本。

当前日期：2026-03-10  
适用分支：`main`

实现状态：Phase 1 已完成（诊断 + 因子数建议 API 与测试已落地）。

---

## 1. Phase 总览（1-4）

| Phase | 目标 | 产出 | 状态 |
| --- | --- | --- | --- |
| Phase 1 | 诊断与因子数建议 | `KMO`、`Bartlett`、`PA`、`MAP`、`Scree`、`Kaiser` | 已完成 |
| Phase 2 | 自动化候选拟合与最优因子数选择 | 候选比较表、`best_n_factors`、综合评分 | 已完成（基础版） |
| Phase 3 | R 风格解读输出增强 | `h2/u2/com`、残差摘要、因子结构告警 | 规划中 |
| Phase 4（可选） | 高级方法与性能优化 | ML 提取、斜交旋转扩展、并行与 bootstrap | 可选 |

---

## 2. Phase 1 目标与边界

### 2.1 目标

1. 在 EFA 拟合前给出“是否适合做 EFA”的诊断结果。  
2. 给出“建议因子数”的多方法结果，不绑定单一规则。  
3. 提供可调参数，便于自动化搜索和后续优化。  
4. 与现有 `fit_efa(...)` 保持兼容，不破坏当前 API。

### 2.2 非目标（Phase 1 不做）

1. 不实现完整 ML-EFA 拟合统计（留给 Phase 4）。  
2. 不实现 oblimin/promax（留给 Phase 4）。  
3. 不做复杂报告渲染（留给 Phase 3/5）。

---

## 3. Phase 1 目标模块结构

建议新增：

```text
src/psysem/efa/
  diagnostics.py     # KMO / Bartlett / 输入质量诊断
  n_factors.py       # PA / MAP / Scree / Kaiser / 聚合建议
  contracts.py       # Phase 1 结果 dataclass
  fit.py             # 保持现有拟合能力（少量对接）
```

建议测试：

```text
tests/
  test_efa_diagnostics.py
  test_efa_n_factors.py
```

---

## 4. Phase 1 逐步实施（详细步骤）

以下每一步都包含：实现内容、落地文件、完成标准。

### Step 1: 定义结果与参数契约

实现内容：

1. 新建 `efa/contracts.py`。  
2. 增加 `EFADiagnosticsConfig`、`EFADiagnosticsResult`。  
3. 增加 `FactorSelectionConfig`、`FactorSelectionResult`。  
4. 统一字段命名，避免后续重复改名。

落地文件：

1. `src/psysem/efa/contracts.py`
2. `src/psysem/efa/__init__.py`（导出新增类型）

完成标准：

1. 所有 dataclass 有类型注解。  
2. `mypy src` 不报错。  
3. 字段命名满足后续 Phase 2 复用。

---

### Step 2: 抽取相关矩阵与基础工具

实现内容：

1. 统一“选定题项 -> 相关矩阵”的逻辑，供 diagnostics/n_factors 共用。  
2. 增加数值稳定处理：对称化、对角线修正、非数处理。  
3. 增加最小样本检查（例如 `n_obs >= n_items * 5` 仅给 warning）。

落地文件：

1. `src/psysem/efa/diagnostics.py`（或 `_utils` 私有函数）

完成标准：

1. 输入包含 NaN/inf 时有明确策略与错误信息。  
2. 输出相关矩阵可用于特征分解。  
3. 单元测试覆盖边界输入。

---

### Step 3: 实现 KMO（总体 + 逐题）

实现内容：

1. 根据相关矩阵和偏相关矩阵计算总体 KMO。  
2. 输出逐题 MSA（每个变量一个值）。  
3. 增加常用解释区间（meritorious, middling, miserable 等）。

落地文件：

1. `src/psysem/efa/diagnostics.py`
2. `tests/test_efa_diagnostics.py`

完成标准：

1. 结果包含 `kmo_total` 和 `kmo_per_item`。  
2. `kmo_total` 在 `[0, 1]`。  
3. 小矩阵、病态矩阵场景有可读报错。

---

### Step 4: 实现 Bartlett 球形检验

实现内容：

1. 基于相关矩阵行列式计算卡方统计量。  
2. 输出 `chi2`, `df`, `p_value`。  
3. 若矩阵不可逆/行列式异常，给出明确 warning。

落地文件：

1. `src/psysem/efa/diagnostics.py`
2. `tests/test_efa_diagnostics.py`

完成标准：

1. 正常输入返回完整检验结果。  
2. 极端输入不会 silent failure。  
3. 结果可用于“是否建议继续 EFA”的判断。

---

### Step 5: 实现 Parallel Analysis（PA）

实现内容：

1. 在 `n_iter` 次随机数据中计算随机相关矩阵特征值分布。  
2. 支持百分位阈值（如 `95th percentile`）。  
3. 将真实特征值与随机阈值比较，给出建议因子数。  
4. 支持 `random_state` 保证可复现。

落地文件：

1. `src/psysem/efa/n_factors.py`
2. `tests/test_efa_n_factors.py`

完成标准：

1. 输出包含真实特征值、随机阈值、建议因子数。  
2. 重复运行（同 seed）结果一致。  
3. 计算耗时在可接受范围（默认参数下）。

---

### Step 6: 实现 MAP Test（Velicer）

实现内容：

1. 对不同 `k` 计算残差相关矩阵均方（或部分相关均方）。  
2. 选择 MAP 最小处作为建议因子数。  
3. 输出每个 `k` 的 MAP 值序列。

落地文件：

1. `src/psysem/efa/n_factors.py`
2. `tests/test_efa_n_factors.py`

完成标准：

1. `k` 范围可配置（`n_min, n_max`）。  
2. MAP 曲线可直接用于后续可视化。  
3. 边界 `k` 场景有防护（避免越界）。

---

### Step 7: 实现 Scree 与 Kaiser 规则

实现内容：

1. 计算排序特征值。  
2. Kaiser：特征值 > 1 的个数。  
3. Scree：先给“原始特征值序列 + 建议拐点算法结果（可选）”。

落地文件：

1. `src/psysem/efa/n_factors.py`
2. `tests/test_efa_n_factors.py`

完成标准：

1. 输出可直接用于画 scree 图。  
2. Kaiser 与 PA/MAP 可以并列比较。  
3. 不把 Kaiser 作为唯一建议来源。

---

### Step 8: 聚合多方法建议

实现内容：

1. 新增统一函数 `suggest_n_factors(...)`。  
2. 返回 `method -> suggestion` 明细。  
3. 给出 `consensus_n_factors`（默认多数投票，平局取较小值）。  
4. 输出 warnings（例如方法间分歧较大）。

落地文件：

1. `src/psysem/efa/n_factors.py`
2. `src/psysem/efa/__init__.py`
3. `tests/test_efa_n_factors.py`

完成标准：

1. 一次调用可拿到完整建议对象。  
2. 每个方法都可独立开关。  
3. 聚合策略可配置（后续 Phase 2 复用）。

---

### Step 9: 对接公开 API 与示例

实现内容：

1. 在 `__init__.py` 导出新 API。  
2. 新增或更新示例脚本，展示诊断 + 因子数建议。  
3. README 增加 “Phase 1 API 用法” 小节。

落地文件：

1. `src/psysem/efa/__init__.py`
2. `examples/basic_efa.py`（或新增 `examples/efa_diagnostics.py`）
3. `README.md`

完成标准：

1. 用户可在 10 行内调用完整诊断流程。  
2. 示例输出可复现。  
3. 与旧 `fit_efa` 用法兼容。

---

### Step 10: 质量门禁

实现内容：

1. 补全单元测试覆盖关键数学路径与边界场景。  
2. 运行 `ruff`, `mypy`, `pytest`。  
3. 增加最小回归数据集，防止未来重构破坏建议因子数逻辑。

落地文件：

1. `tests/test_efa_diagnostics.py`
2. `tests/test_efa_n_factors.py`

完成标准：

1. `pytest` 全绿。  
2. 新增测试对失败信息有断言。  
3. CI 可稳定通过。

---

## 5. Phase 1 建议 API（草案）

```python
from psysem import (
    EFADiagnosticsConfig,
    FactorSelectionConfig,
    run_efa_diagnostics,
    suggest_n_factors,
)

diag = run_efa_diagnostics(data, EFADiagnosticsConfig(items=("i1", "i2", "i3", "i4")))
sel = suggest_n_factors(data, FactorSelectionConfig(items=("i1", "i2", "i3", "i4"), n_min=1, n_max=4))
```

建议返回字段（最小集合）：

1. diagnostics: `kmo_total`, `kmo_per_item`, `bartlett_chi2`, `bartlett_df`, `bartlett_p`
2. selection: `eigenvalues`, `parallel_thresholds`, `map_values`, `suggestions_by_method`, `consensus_n_factors`, `warnings`

---

## 6. 参数表（Phase 1）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `n_min` | `1` | 搜索最小因子数 |
| `n_max` | `min(8, n_items - 1)` | 搜索最大因子数 |
| `pa_iter` | `500` | Parallel Analysis 随机重采样次数 |
| `pa_percentile` | `0.95` | PA 比较分位数 |
| `random_state` | `None` | 随机种子（建议固定） |
| `enable_map` | `True` | 是否运行 MAP |
| `enable_pa` | `True` | 是否运行 PA |
| `enable_kaiser` | `True` | 是否运行 Kaiser |
| `enable_scree` | `True` | 是否输出 scree 数据 |

---

## 7. 验收标准（Definition of Done）

1. `run_efa_diagnostics` 可稳定输出 KMO + Bartlett。  
2. `suggest_n_factors` 可同时输出 PA/MAP/Kaiser/Scree 结果。  
3. 有聚合建议 `consensus_n_factors`，且策略可配置。  
4. 文档、示例、测试齐全并通过质量门禁。  
5. 与现有 `fit_efa` 保持向后兼容。

---

## 8. 风险与缓解

1. 风险：小样本导致相关矩阵不稳定。  
缓解：增加样本充足性 warning，不直接静默失败。  

2. 风险：不同方法建议差异大。  
缓解：输出方法分歧警告，不强行给单一结论。  

3. 风险：计算成本偏高（PA）。  
缓解：支持 `pa_iter` 调参与可复现 seed。

---

## 9. 下一步执行顺序

1. 先实现 `contracts.py` + `diagnostics.py`（Step 1-4）。  
2. 再实现 `n_factors.py`（Step 5-8）。  
3. 最后接入 API/示例/测试（Step 9-10）。
