# SEM 后续实施路线图（ZH）

本文档基于当前 `psysem` 代码状态，给出“下一步应该怎么做”的可执行路线。  
目标是把现有的 **ML 原型** 推进到“可解释、可复现、可扩展”的 SEM 工程版本。

当前日期：2026-03-11  
适用分支：`main`

执行进度（更新）：Phase A 已落地；Phase B 已启动并落地首批（基础拟合指标已接入 `fit`）。

---

## 1. 当前基线

当前已具备：

1. 语法入口统一：`syntax` 与 `spec` 已统一到 `ModelSpec`。  
2. 矩阵草图：measurement(`Lambda/Theta`) + structural(`Beta/Gamma/Psi`) 已可构建。  
3. 参数映射：全局 `parameter_index -> vector_position` 已统一。  
4. ML 原型：`build_implied_covariance` + `optimize_ml_parameters` 已接入 `SEMModel.fit`。  
5. 质量状态：`ruff + mypy + pytest` 全绿（当前 `137` tests）。

当前关键缺口：

1. 推断层仍需稳健化：当前为数值 Hessian 原型，精度与异常场景覆盖待增强。  
2. 拟合指标仍需完善：当前已实现基础版，但边界场景与稳健版本（如 robust 统计量）待补齐。  
3. 优化鲁棒性不足：当前仍是原型级初值/边界/失败诊断。  
4. 高级路径未覆盖：`MLR/WLSMV`、多组不变性、完整端到端数值回归。

---

## 2. 目标状态（下一阶段）

完成后应达到：

1. `SEMModel.fit(...)` 可输出可解释参数表（估计值 + SE + 显著性）。  
2. 可输出基础拟合指标，且在正常模型下非 `nan`。  
3. 优化失败时给出结构化诊断信息（不是仅 `placeholder/failed` 字符串）。  
4. 有稳定的数值回归测试，重构不易破坏结果。

---

## 3. 分阶段执行计划

## Phase A：推断层最小可用（优先级 P0）

目标：把“仅有参数估计值”升级为“可解释统计量”。

### A1. 新建 `inference` 模块骨架

实现内容：

1. 新建 `src/psysem/inference/`。  
2. 增加 `compute_standard_errors(...)`（先用数值近似）。  
3. 增加 `build_parameter_inference_table(...)`（估计值、SE、z、p、CI）。

建议落地文件：

1. `src/psysem/inference/__init__.py`
2. `src/psysem/inference/basic.py`
3. `src/psysem/result.py`
4. `src/psysem/reporting.py`

验收标准：

1. `SEMResult` 能返回参数统计字段。  
2. 至少 1 个简单模型可产出非空 SE/z/p。  
3. 失败时明确标注不可计算原因（而非静默 `nan`）。

### A2. 接入 `SEMModel.fit`

实现内容：

1. 在 ML 优化成功时调用推断模块。  
2. 把推断摘要写入 `optimization_info` 和 `warnings`。  
3. 在 `summary()` / `to_markdown()` 展示关键统计。

验收标准：

1. `fit` 输出不破坏旧字段兼容。  
2. 新增推断字段有测试覆盖。

---

## Phase B：拟合指标真实化（优先级 P0）

目标：替换当前占位 `fit_indices`。

### B1. 实现核心指标

实现内容：

1. 先实现 `AIC/BIC`（依赖目标函数与自由参数数）。  
2. 再实现 `SRMR`（样本协方差与 implied 协方差差异）。  
3. 最后实现 `CFI/TLI/RMSEA`（需要 baseline 模型与卡方相关量）。

建议落地文件：

1. `src/psysem/fit_indices.py`（由占位升级）
2. `src/psysem/estimation/ml.py`（补充所需中间量输出）
3. `tests/test_sem_fit_indices.py`（新建）

验收标准：

1. 正常模型下主要指标非 `nan`。  
2. 边界模型（欠识别/近奇异）能返回可解释 warning。  
3. 指标数值范围有基本合理性检查。

当前进展（2026-03-11）：

1. `fit_indices.py` 已由占位升级为基础可计算版本。  
2. `SEMModel.fit` 已接入基础拟合指标计算与诊断回写。  
3. `tests/test_sem_fit_indices.py` 已新增并覆盖正常路径 + 边界路径。

---

## Phase C：优化鲁棒性（优先级 P1）

目标：让优化器从“原型可跑”升级为“工程可用”。

### C1. 引入拟合配置对象

实现内容：

1. 新增 `SEMFitConfig`（`max_iter/method/tol/restarts/bounds`）。  
2. `SEMModel.fit(..., fit_config=...)` 可选接入。  
3. 明确默认值，保持向后兼容。

建议落地文件：

1. `src/psysem/estimation/contracts.py`
2. `src/psysem/core.py`
3. `tests/test_sem_estimation_config.py`（新建）

### C2. 增加失败诊断与容错

实现内容：

1. 收敛失败原因分层（初值问题/边界问题/矩阵不可逆）。  
2. 增加可选多起点重启策略。  
3. 标准化 warning 文案，便于后续报告展示。

验收标准：

1. 常见失败场景可复现并有明确诊断。  
2. 重启策略可提高成功率且有测试断言。

---

## Phase D：测试与回归门禁（优先级 P0）

目标：建立数值稳定的回归防线。

### D1. 补齐 SEM 专项测试矩阵

实现内容：

1. 新增 `tests/test_sem_inference.py`。  
2. 新增 `tests/test_sem_fit_indices.py`。  
3. 新增端到端测试：measurement-only、measurement+structural、`spec` 入口。

### D2. 建立数值回归基准

实现内容：

1. 固定随机种子与合成数据。  
2. 对关键指标设置容差断言（避免过严）。  
3. 在 CI 保持同一测试路径执行。

验收标准：

1. 重跑结果稳定（在容差内）。  
2. 重构后能及时发现数值回归。

---

## 4. 可选增强（Phase E，优先级 P2）

1. 支持 structural observed predictors 的完整 implied covariance 路径。  
2. 支持 `Theta/Psi` 非对角协方差（带开关和识别约束）。  
3. 扩展估计器：`MLR`、`WLSMV`。  
4. 多组测量不变性主流程（configural/metric/scalar）。  
5. 输出导出：参数表/指标表 Markdown + CSV。

---

## 5. 建议执行顺序（短周期）

建议按以下顺序推进：

1. `Phase A`（推断层最小可用）  
2. `Phase B`（拟合指标真实化）  
3. `Phase D`（测试回归门禁同步补齐）  
4. `Phase C`（优化鲁棒性增强）  
5. `Phase E`（高级能力按需求开启）

---

## 6. 每次迭代的完成定义（DoD）

每个小阶段完成前，至少满足：

1. 代码实现 + 文档更新（README/`docs/index.md`）同步。  
2. `python -m ruff check .` 通过。  
3. `python -m mypy src` 通过。  
4. `python -m pytest -q` 全绿。  
5. 新增能力至少包含“正常路径 + 边界路径”测试各 1 个。

---

## 7. 关联文档

1. [SEM Phase 实施文档（ZH）](sem-phase-implementation.zh-CN.md)  
2. [EFA 测试与质量门禁（ZH）](efa-testing.zh-CN.md)
