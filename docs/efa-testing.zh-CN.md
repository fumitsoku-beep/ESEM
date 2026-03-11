# EFA 测试与质量门禁（ZH）

本文档说明 `psysem.efa` 当前测试覆盖范围、运行方式与质量门禁。

当前日期：2026-03-11

当前测试数量：`81`（其中 EFA 相关测试 `67`）。

---

## 1. 覆盖模块

| 模块 | 关键测试点 |
| --- | --- |
| `fit.py` | 输入校验、方法注册、`PAF/PCA` 拟合、残差矩阵性质、告警触发 |
| `diagnostics.py` | `KMO/Bartlett` 计算、缺失值策略、常量项告警、奇异相关矩阵容错 |
| `n_factors.py` | `PA/MAP/Scree/Kaiser`、共识策略、参数边界、可复现性 |
| `evaluation.py` | 评分公式稳定性、阈值校验、空显著因子告警、表格导出 |
| `interpretation.py` | 题项表/因子表、残差 Top-N、阈值合法性、解读告警 |
| `workflow.py` | 端到端流程、候选策略、手动候选、配置一致性、解释层开关 |

---

## 2. 覆盖维度

1. 正常路径：可在合成数据上稳定输出结果。
2. 边界路径：`n_min/n_max`、阈值范围、空输入、缺失输入。
3. 错误路径：非法参数触发明确异常信息。
4. 数值路径：奇异相关矩阵、常量项、低样本比等场景可诊断。
5. 可复现性：固定随机种子后，PA 阈值与建议结果稳定。
6. API 路径：顶层导出与 workflow 返回结构一致。

---

## 3. 测试文件

| 文件 | 关注点 |
| --- | --- |
| `tests/test_efa.py` | `fit_efa` + 方法注册 |
| `tests/test_efa_diagnostics.py` | 诊断入口与前置数据质量 |
| `tests/test_efa_n_factors.py` | 因子数建议与聚合策略 |
| `tests/test_efa_evaluation.py` | 候选评分逻辑 |
| `tests/test_efa_interpretation.py` | Phase 3 解释层输出 |
| `tests/test_efa_workflow.py` | 诊断 -> 建议 -> 拟合 -> 评分 -> 解释 端到端流程 |

---

## 4. 运行命令

全量质量门禁：

```bash
python -m ruff check .
python -m mypy src
python -m pytest -q
```

仅运行 EFA 相关测试：

```bash
python -m pytest tests/test_efa.py tests/test_efa_* -q
```

---

## 5. 质量门禁标准

1. `ruff` 无 lint 错误。
2. `mypy` 在 `src` 下无类型错误。
3. `pytest` 全部通过。
4. EFA 新增/变更功能必须包含至少 1 个正常路径 + 1 个边界/异常路径测试。
