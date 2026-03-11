# SEM Phase 实施文档（ZH）

本文档用于规划 `psysem` 从当前 `SEMModel` 占位接口，升级到可用于心理测量场景的可估计 SEM/ESEM 主流程。

当前日期：2026-03-11  
适用分支：`main`

实现状态：Phase 1 已完成基础版；Phase 2 已完成第二批；Phase 3 已落地第六批（`Beta/Gamma/Psi` 草图 + 循环依赖基础检查 + 统一参数索引映射 + ML implied covariance/优化原型 + 数值推断原型 + 基础拟合指标原型）；估计器仍为占位。

---

## 1. Phase 总览（1-4）

| Phase | 目标 | 产出 | 状态 |
| --- | --- | --- | --- |
| Phase 1 | 入口与契约统一 | `ModelSpec` 扩展、`fit(data, spec=...)`、严格语法校验、标准结果字段 | 已完成（基础版） |
| Phase 2 | 测量层矩阵构建 | measurement block 组装、识别性检查、参数索引 | 进行中（第二批已完成） |
| Phase 3 | 结构层 + ML 估计闭环 | structural 路径映射、目标函数、优化、SE 与基础 fit 指标 | 进行中（第五批已完成） |
| Phase 4（可选） | 高级能力与性能优化 | MLR/WLSMV、多组与不变性、bootstrap、报告增强 | 可选 |

---

## 2. 当前基线（2026-03-11）

已具备：

1. `parse_model(syntax)` 基础校验（非空字符串）。  
2. `SEMModel.fit(data)` 占位接口，可返回 `SEMResult`。  
3. `compute_basic_fit_indices()` 已支持基础计算路径（无输入时仍保留 `nan` 占位兼容）。  
4. `to_markdown(result)` 基础报告输出。  

当前缺口：

1. 无正式语法树（AST）和参数约束表达。  
2. measurement/structural 矩阵层已起步，但估计闭环尚未实现。  
3. ML、推断与基础拟合指标已起步原型，但 MLR/WLSMV 与稳健统计尚未完成。  
4. 无端到端 SEM 数值回归测试。  

---

## 2.1 最新进展（2026-03-11）

Phase 1 已落地（第二批）：

1. `parse_model` 支持结构化关系对象（含 RHS term 细分）。  
2. 支持 term modifier：参数标签（如 `b1*x1`）与固定系数（如 `0.5*x1`）。  
3. 支持约束表达占位：`==`、`>=`、`<=`。  
4. 语法错误信息支持定位到 `statement` 与 `term`。  
5. `SEMModel.fit` 已支持 `fit(data, spec=...)` 并统一到 `ModelSpec`。  
6. `SEMResult` 扩展字段已接入 `summary` 与 `to_markdown`。  
7. `SEMModel.fit` 已生成参数表草稿（free/fixed/label）并写入 `parameter_table`。  

Phase 2 已落地（第二批）：

1. 新增 `measurement` 模块，支持 `Lambda`/`Theta` 矩阵草图构建。  
2. 增加基础识别性检查（最少指标数、marker 缺失告警）。  
3. 支持多 block 组装映射（`block_latent_pairs`）。  
4. measurement 层加载参数与全局 `parameter_table` 索引已对齐。  

Phase 3 已落地（第六批）：

1. 新增 `structural` 模块，支持 structural path table 构建。  
2. 产出 `Beta/Gamma` 矩阵草图（用于后续估计层输入）。  
3. 增加循环依赖基础检查（latent cycle）。  
4. `SEMModel.fit` 已接入 structural 设计并回传到 `SEMResult`。  
5. 新增全局 `parameter_index_map`（`parameter_index -> vector_position`）。  
6. measurement/structural 均回传统一参数索引矩阵（供估计层直接取值）。  
7. structural 增加 `Psi`（内生潜变量扰动方差）矩阵与索引映射。  
8. 新建 `estimation` 模块，落地 `gaussian_ml_discrepancy`、`build_implied_covariance` 与 `optimize_ml_parameters` 原型。  
9. `SEMModel.fit` 在样本量满足阈值时可自动触发 ML 原型优化并回填参数值。  
10. measurement 增加 `Theta` 参数索引矩阵，观测残差方差可映射到全局参数向量。  
11. 新建 `inference` 模块，落地数值 Hessian 推断原型（SE/z/p/CI）并接入 `SEMModel.fit`。  
12. `fit_indices` 升级为基础可计算版本（AIC/BIC/SRMR/CFI/TLI/RMSEA）并接入 `SEMModel.fit`。  

---

## 3. 目标模块结构（建议）

建议在现有 `src/psysem` 下逐步落地以下层：

```text
src/psysem/
  syntax/         # 语法解析、约束解析、ModelSpec 规范化
  measurement/    # 测量层矩阵构建（CFA/ESEM block）
  structural/     # 结构路径矩阵构建
  estimation/     # 目标函数、优化器、收敛控制
  inference/      # 信息矩阵、SE/z/p/CI
  fit/            # CFI/TLI/RMSEA/SRMR/AIC/BIC
  reporting/      # 参数表、摘要、导出
  core.py         # SEMModel 入口编排
  result.py       # SEMResult 与结果契约
```

说明：`data` 模块继续承担输入契约和预校验职责，不与 SEM 数值层重复。

---

## 4. Phase 1（入口与契约统一）

### 4.1 目标

1. 统一 `syntax` 与 `spec` 两条入口的内部表示。  
2. 把当前 token 级解析升级为结构化解析结果。  
3. 升级 `SEMResult` 契约，预留后续估计与报告字段。  

### 4.2 非目标

1. 不实现完整估计器。  
2. 不实现复杂多组与不变性。  

### 4.3 详细步骤

#### Step 1: 扩展模型契约

实现内容：

1. 将 `ModelSpec` 扩展为可承载 measurement/structural/constraint 的结构。  
2. 新增参数项元数据（参数名、类型、自由/固定、起始值、边界）。  
3. 定义统一错误类型（语法错误、识别性错误、数据错误）。  

落地文件（建议）：

1. `src/psysem/model.py`
2. `src/psysem/result.py`
3. `src/psysem/core.py`

完成标准：

1. `ModelSpec` 不再仅包含 `syntax: str`。  
2. 错误信息可定位到表达式片段。  

#### Step 2: 解析器升级

实现内容：

1. 支持测量表达式与结构表达式的显式区分。  
2. 支持基本参数约束（固定值、标签、等值约束的占位表达）。  
3. 对重复路径、未知变量、空右侧表达式报错。  

落地文件（建议）：

1. `src/psysem/model.py`
2. `tests/test_fit_smoke.py`（拆分/扩展为 parser 专项测试）

完成标准：

1. 关键非法语法均有单元测试断言。  
2. `parse_model` 返回结构化对象。  

#### Step 3: 统一 `fit` 入口

实现内容：

1. 让 `SEMModel.fit(data, spec=...)` 支持显式 `ESEMSpec`。  
2. 统一 `SEMModel(...syntax...).fit(...)` 与 `sem(...)` 的内部编排。  
3. 输出统一 warnings 容器，便于后续质量诊断。  

落地文件（建议）：

1. `src/psysem/core.py`
2. `src/psysem/__init__.py`
3. `tests/test_fit_smoke.py`

完成标准：

1. 两条入口得到同构的内部 `ModelSpec`。  
2. 兼容现有 smoke API。  

#### Step 4: 升级 `SEMResult` 契约

实现内容：

1. 新增 `parameter_table`、`warnings`、`optimization_info` 字段。  
2. 保留旧字段兼容，避免破坏外部调用。  
3. `summary()` 增加结构化输出（估计器、收敛、主要指标）。  

落地文件（建议）：

1. `src/psysem/result.py`
2. `src/psysem/reporting.py`
3. `tests/test_fit_smoke.py`

完成标准：

1. 结果对象可直接承载 Phase 2/3 的产出。  
2. 向后兼容测试通过。  

### 4.4 Phase 1 验收标准

1. 可稳定解析并标准化语法/`spec` 输入。  
2. `fit` 入口统一且错误信息清晰。  
3. `SEMResult` 新契约完成并保持兼容。  

---

## 5. Phase 2（测量层矩阵构建）

### 5.1 目标

1. 将测量模型映射到 SEM 矩阵表示。  
2. 支持单 block 到多 block 组装。  
3. 在估计前完成识别性与参数计数检查。  

### 5.2 详细步骤

#### Step 5: measurement 矩阵定义

实现内容：

1. 构建 `Lambda`, `Theta` 等测量层矩阵定义。  
2. 定义潜变量与观测变量索引映射。  
3. 显式区分固定参数与自由参数。  

落地文件（建议）：

1. `src/psysem/measurement/`（新建）
2. `tests/test_sem_measurement.py`（新建）

完成标准：

1. 可从 `ModelSpec` 生成测量层矩阵草图。  
2. 参数索引稳定可复现。  

#### Step 6: 多 block 组装与旋转策略入口

实现内容：

1. 支持按 block 组合 measurement 部分。  
2. 允许 block 级配置覆盖全局配置。  
3. 与 `data` 模块中 block 合法性校验结果联动。  

落地文件（建议）：

1. `src/psysem/measurement/assembler.py`
2. `tests/test_sem_measurement.py`

完成标准：

1. 多 block 输入可产生统一矩阵定义。  
2. block 覆盖规则有测试断言。  

#### Step 7: 识别性检查

实现内容：

1. 因子尺度设定检查（marker/loading/variance 规则）。  
2. 参数自由度检查与过识别/欠识别预警。  
3. 不可识别模型在估计前阻断。  

落地文件（建议）：

1. `src/psysem/measurement/identification.py`
2. `tests/test_sem_identification.py`（新建）

完成标准：

1. 不可识别模型有明确错误类型与信息。  
2. 识别性检查可独立调用。  

### 5.3 Phase 2 验收标准

1. 测量层可稳定转换为矩阵定义。  
2. 参数索引与识别性检查可复用到估计层。  
3. 多 block 场景可运行。  

---

## 6. Phase 3（结构层 + ML 估计闭环）

### 6.1 目标

1. 完成 structural 路径矩阵映射。  
2. 跑通单组连续变量 `ML` 估计主流程。  
3. 给出可解释参数结果与基础拟合指标。  

### 6.2 详细步骤

#### Step 8: structural 矩阵构建

实现内容：

1. 构建 `Beta`, `Gamma`, `Psi`（结构层）映射。  
2. 检查循环依赖、重复路径、未知变量。  
3. 合并 measurement + structural 参数索引。  

落地文件（建议）：

1. `src/psysem/structural/`（新建）
2. `tests/test_sem_structural.py`（新建）

完成标准：

1. 结构层矩阵输出稳定。  
2. 非法路径场景有明确断言。  

#### Step 9: ML 目标函数与优化编排

实现内容：

1. 建立 implied covariance 与损失函数。  
2. 封装优化器（初值、边界、收敛判据、失败信息）。  
3. 将优化状态写入 `SEMResult.optimization_info`。  

落地文件（建议）：

1. `src/psysem/estimation/`（新建）
2. `tests/test_sem_estimation_ml.py`（新建）

完成标准：

1. 在基准数据上稳定收敛。  
2. 失败场景可复现并给出可读原因。  

#### Step 10: 推断与拟合指标

实现内容：

1. 计算信息矩阵与参数标准误（先支持常规近似）。  
2. 输出 `z/p/CI` 到参数表。  
3. 将 `cfi/tli/rmsea/srmr/aic/bic` 从占位改为真实计算。  

落地文件（建议）：

1. `src/psysem/inference/`（新建）
2. `src/psysem/fit_indices.py`（升级）
3. `tests/test_sem_inference.py`（新建）
4. `tests/test_api_surface.py`（从 shape 检查升级到数值有效性）

完成标准：

1. 参数表含估计值与推断字段。  
2. 基础 fit 指标在正常模型下非 `nan`。  

### 6.3 Phase 3 验收标准

1. 单组连续变量 ML 流程可端到端运行。  
2. 结果包含参数估计、SE、p 值、基础 fit 指标。  
3. 文档与示例可复现。  

---

## 7. Phase 4（可选：高级能力与性能优化）

### 7.1 目标

1. 扩展估计器与鲁棒推断。  
2. 支持多组与不变性评估。  
3. 提升大样本/高维场景性能。  

### 7.2 可选子项

1. 估计器扩展：`MLR`、`WLSMV`。  
2. 多组与约束：configural/metric/scalar invariance。  
3. 置信区间增强：bootstrap。  
4. 性能优化：并行梯度/缓存矩阵分解。  
5. 报告增强：可导出 Markdown/CSV/HTML。  

### 7.3 Phase 4 验收标准

1. 高级功能可开关且不破坏 Phase 3 主流程。  
2. 回归测试覆盖核心 estimator 路径。  

---

## 8. 公共 API 草案（建议）

```python
from psysem import SEMModel

model = SEMModel("y ~ x1 + x2")
result = model.fit(
    data,
    # 新增（规划中）：
    # spec=esem_spec,
    # estimator="ml",
    # options=SEMFitConfig(...)
)
print(result.summary())
```

建议后续新增配置对象：

1. `SEMFitConfig`：估计器与收敛参数。  
2. `SEMInferenceConfig`：SE 与区间配置。  
3. `SEMReportConfig`：报告输出粒度。  

---

## 9. 测试与质量门禁（全阶段）

建议门禁命令：

```bash
python -m ruff check .
python -m mypy src
python -m pytest -q
```

建议测试结构：

```text
tests/
  test_fit_smoke.py
  test_sem_parser.py
  test_sem_measurement.py
  test_sem_identification.py
  test_sem_structural.py
  test_sem_estimation_ml.py
  test_sem_inference.py
  test_sem_fit_indices.py
```

---

## 10. 风险与缓解

1. 风险：识别性问题导致估计不稳定。  
缓解：将识别性检查前置为硬门禁。  

2. 风险：语法与 `spec` 双入口造成行为不一致。  
缓解：统一转换到同一 `ModelSpec` 中间层。  

3. 风险：统计指标与优化器实现复杂度高。  
缓解：先做单估计器（ML）闭环，再扩展。  

---

## 11. 执行优先级建议

1. 先做 Phase 1（入口与契约），降低后续返工成本。  
2. 再做 Phase 2（矩阵与识别性），保证估计层输入稳定。  
3. 然后做 Phase 3（ML 闭环），拿到首个可用 SEM 版本。  
4. 最后按需求选择 Phase 4（可选增强）。  
