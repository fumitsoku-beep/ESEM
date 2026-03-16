# SEM Phase 实施文档（ZH）

本文档用于规划 `psysem` 从当前 `SEMModel` 占位接口，升级到可用于心理测量场景的可估计 SEM/ESEM 主流程。

> 说明：本文档保留了阶段性实施记录。若文中出现早期根层路径，请以当前正式实现路径 `psysem.sem.*` 与 `src/psysem/sem/` 为准。

当前日期：2026-03-11  
适用分支：`main`

实现状态：Phase 1 已完成基础版；Phase 2 已完成第二批；Phase 3 已落地第七批（`Beta/Gamma/Psi` 草图 + 循环依赖基础检查 + 统一参数索引映射 + ML implied covariance/优化原型 + 数值推断原型 + 基础拟合指标原型 + 优化鲁棒性原型）；估计器仍为占位。

---

## 1. Phase 总览（1-4）

| Phase           | 目标                 | 产出                                                                | 状态                   |
| --------------- | -------------------- | ------------------------------------------------------------------- | ---------------------- |
| Phase 1         | 入口与契约统一       | `ModelSpec` 扩展、`fit(data, spec=...)`、严格语法校验、标准结果字段 | 已完成（基础版）       |
| Phase 2         | 测量层矩阵构建       | measurement block 组装、识别性检查、参数索引                        | 进行中（第二批已完成） |
| Phase 3         | 结构层 + ML 估计闭环 | structural 路径映射、目标函数、优化、SE 与基础 fit 指标             | 进行中（第七批已完成） |
| Phase 4（可选） | 高级能力与性能优化   | MLR/WLSMV、多组与不变性、bootstrap、报告增强                        | 可选                   |

---

## 2. 当前基线（2026-03-11）

已具备：

1. `parse_model(syntax)` 基础校验（非空字符串）。
2. `SEMModel.fit(data)` 占位接口，可返回 `SEMResult`。
3. `compute_basic_fit_indices()` 已支持基础计算路径（无输入时仍保留 `nan` 占位兼容）。
4. `to_markdown(result)` 基础报告输出。

当前缺口：

1. 无正式语法树（AST）和参数约束表达。
2. measurement/structural 矩阵层已起步，稳健估计闭环尚未完善。
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

Phase 3 已落地（第七批）：

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
13. 引入 `SEMFitConfig` / `ParameterBoundsConfig`，支持拟合配置、重启策略与失败分类诊断。
14. 推断层已补充 `ok/partial/failed` 状态、失败原因与 SE 可用数量，并接入结果摘要与 Markdown 报告。
15. 拟合指标层已补充 `ok/partial/failed` 状态、失败原因与可用指标数量，并接入 `optimization_info` 与结果展示。

---

## 2.2 为什么当前 SEM 要按这条路线推进

这一节用于回答三个问题：

1. 为什么当前阶段不应一开始就把重心放在更多 estimator 上；
2. 为什么要先把 `measurement -> structural -> estimation -> inference -> fit indices` 这条主链打稳；
3. 为什么这条路线在心理学与数据分析上都更合理。

### 从心理学与心理测量角度看

#### 1. 先稳住 measurement，是因为心理学研究首先依赖“测量是否成立”

在心理学应用里，SEM 很少只是一个纯路径模型；它通常承担的是：

1. 检验量表题项是否能有效测量潜变量；
2. 检验潜变量之间的关系是否符合理论预期；
3. 在控制测量误差后，再解释结构路径。

因此，当前文档把 measurement 层放在 structural 层之前，是因为：

1. 如果测量模型本身不稳，后续路径系数很难被解释为“理论关系”；
2. 心理量表常见题项数量有限、负荷不均衡、交叉负荷风险较高；
3. 研究者真正需要的是“这个潜变量是否被可靠测量”，而不是仅看到一个可收敛的结构模型。

#### 2. 先做识别性检查，是因为心理学模型常常“理论上合理、统计上不可识别”

心理学研究者常常根据理论直接写模型，但统计实现里还需要满足：

1. 每个潜变量至少有足够题项支撑；
2. 因子尺度必须固定（marker / variance / loading 规则）；
3. 结构路径和测量路径不能共同造成欠识别。

如果缺少识别性检查，使用者会把问题误判成：

- “优化器不好用”

而真实问题可能是：

- “模型在统计上根本不能估”。

因此，文档中把识别性检查视作硬门禁，而不是一个可选增强项。

#### 3. 先把 `ML` 主路径做稳，是因为心理学场景需要“可解释结果”先站住

当前很多心理学数据最终会走向 ordinal / robust / multi-group，但如果现在连基础连续变量 `ML` 路径都还停留在“原型可跑”，那后面引入：

1. `MLR`
2. `WLSMV`
3. 多组不变性
4. bootstrap

都会缺少一个稳定基线。

从心理测量开发顺序看，更合理的是：

1. 先有一条解释链完整的基线；
2. 再扩 estimator；
3. 再扩更复杂研究设计。

### 从数据分析与统计实现角度看

#### 1. 先稳定推断层，是因为“估计值”不等于“可解释结果”

仅仅输出参数估计值，并不能支持研究判断；还需要知道：

1. 参数标准误有多大；
2. 估计是否稳定；
3. 显著性是否只是采样波动；
4. 哪些结果其实不应被解释。

因此，推断层不是报告装饰，而是结果可信度的一部分。

#### 2. 先补 fit indices 的边界行为，是因为 SEM 是“整体模型拟合”，不是单参数回归

与一般回归不同，SEM 关心的是：

1. 整体协方差结构是否被模型合理解释；
2. 拟合是否足以支持理论模型；
3. 不同模型之间能否进行比较。

如果 `CFI/TLI/RMSEA/SRMR/AIC/BIC` 在边界场景下行为不清楚，用户会面临两个问题：

1. 正常模型结果难比较；
2. 异常模型结果难解释。

#### 3. 先补数值回归测试，是因为 SEM 极易受数值细节影响

SEM 的实现高度依赖：

1. 参数向量到矩阵的映射；
2. 协方差矩阵求逆与行列式；
3. 初值、边界和重启策略；
4. 病态矩阵与近奇异情况的处理。

这意味着：

1. 功能层面“没坏”不代表统计含义没漂移；
2. 小改动也可能使收敛性、SE 或 fit indices 改变；
3. 没有稳定回归测试，就很难持续迭代。

### 这一节的开发结论

因此，当前 SEM 主线最合理的顺序不是：

1. 先加更多 estimator
2. 先做更多高级功能

而是：

1. 先把 measurement / structural / estimation 的基础闭环打稳；
2. 先让推断层和 fit indices 可解释；
3. 先建立数值回归防线；
4. 最后再扩到 `MLR/WLSMV`、多组不变性与更复杂设计。

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

### 3.1 建议的步骤编写模板

为了让后续文档既能指导规划，也能直接指导编码，建议后续每个关键步骤都尽量按下面模板补充：

1. **目标**：这一步最终要交付什么；
2. **为什么做**：它在心理测量解释链和统计实现链里解决什么问题；
3. **输入**：依赖哪些现有对象、矩阵或配置；
4. **输出**：新增哪些结果对象、字段或 warning；
5. **涉及模块**：会修改哪些文件；
6. **核心算法/规则**：最小数学逻辑或规则集合；
7. **边界与失败场景**：哪些情况必须阻断，哪些情况应 warning；
8. **测试要求**：至少包含哪些正常路径与异常路径；
9. **完成定义**：怎样判断这一步已经可以合并。

后文“下一阶段执行路线”的细化部分将优先按这个模板展开。

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

实施提示：

1. 先冻结 `ModelSpec` 的最小核心字段，避免一边写 parser 一边反复改契约；
2. 参数项元数据建议最少包含：参数类别、lhs/rhs、free/fixed、label、start、bounds；
3. 错误类型要尽量早统一，否则后续 parser、measurement、structural 会各自产生不兼容异常。

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

实施提示：

1. 先保证 parser 输出稳定，再追求语法丰富度；
2. 对心理学高频写法优先支持：测量式、回归式、参数标签、固定系数；
3. 每增加一类语法，都同步补错误定位测试，避免 parser 以后难重构。

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

实施提示：

1. 这一步的重点不是增加功能，而是减少入口分叉；
2. 建议所有入口都尽早收口到同一个内部 `ModelSpec`，再进入数值层；
3. 先统一 warnings 和结果结构，再考虑对外增加更多参数选项。

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

实施提示：

1. `SEMResult` 应优先服务“解释”而不是“内部中间量堆积”；
2. 顶层字段建议按用户阅读顺序组织：收敛 -> 警告 -> 参数表 -> 拟合指标 -> 诊断；
3. 如果某些字段暂时只在原型阶段使用，建议放进 `optimization_info`，不要污染主结果面。

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

实施提示：

1. 这一层要优先解决“题项与潜变量怎样映射”，而不是先追求完整估计；
2. 参数索引一旦确定，应尽量少改，因为后面 estimation / inference 都依赖它；
3. 对心理测量场景，优先保证常见 CFA/block 结构能稳定落成矩阵。

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

实施提示：

1. 先把 block 组装规则写清楚，再做复杂覆盖逻辑；
2. 覆盖优先级建议固定为：block 配置 > 全局配置 > 默认值；
3. 这一步要特别注意与 ESEM block 语义保持一致，避免未来重复拆层。

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

实施提示：

1. 先做最基础、最确定的识别性规则，不必一开始覆盖所有理论变体；
2. 明确区分“硬错误”和“仅 warning 的弱风险”；
3. 识别性输出最好能直接写入结果或异常信息，便于研究者理解为何模型不可估。

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

实施提示：

1. 这一步要先保证 latent-to-latent 与 observed-to-latent 的路径语义清楚；
2. 循环依赖检查要尽早加入，否则优化失败会很难诊断；
3. measurement 和 structural 的参数索引必须在这一层真正汇总为统一视图。

输入 / 输出契约（建议）：

- 输入：
  1.  已标准化的 `ModelSpec`
  2.  measurement 层变量索引与参数索引
  3.  结构路径定义（latent -> latent / observed -> latent）
- 输出：
  1.  `Beta/Gamma/Psi` 数值草图
  2.  结构层参数索引矩阵
  3.  结构层 warnings / 错误类型

建议算法与规则：

1. 先把 latent endogenous / exogenous 集合划清；
2. 再按路径类型写入 `Beta` 或 `Gamma`；
3. 对内生潜变量生成 `Psi` 扰动方差位置；
4. 对重复路径、未知节点、非法自回归和循环依赖做前置检查；
5. 最后把 measurement 与 structural 参数索引合并成统一参数视图。

最低测试要求：

1. 一个合法 latent 回归结构；
2. 一个 observed predictor -> latent 结构；
3. 一个循环依赖失败案例；
4. 一个重复路径失败案例。

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

实施提示：

1. 先把 implied covariance 路径与参数向量映射打通，再优化性能；
2. 优化失败信息要尽量结构化，而不是只保留优化器原始字符串；
3. 初值、边界与重启策略建议一开始就留接口，避免后续大改函数签名。

输入 / 输出契约（建议）：

- 输入：
  1.  measurement + structural 的统一参数索引
  2.  样本协方差矩阵与样本量
  3.  初值、边界、优化配置（如 `SEMFitConfig`）
- 输出：
  1.  优化后的参数向量
  2.  `optimization_info`
  3.  收敛状态、失败分类、重启统计

建议算法与规则：

1. 定义参数向量到模型矩阵的映射函数；
2. 定义 implied covariance 构造函数；
3. 定义基于高斯 ML 的目标函数；
4. 为边界、奇异矩阵、非法 implied covariance 设置明确失败路径；
5. 只在基线功能稳定后再考虑性能优化、梯度优化或缓存。

建议 `optimization_info` 最少包含：

1. `success`
2. `n_iter`
3. `objective_value`
4. `failure_category`
5. `n_attempts`
6. `best_attempt_index`
7. `warnings`

最低测试要求：

1. 一个可正常收敛的简单模型；
2. 一个边界命中案例；
3. 一个 implied covariance 非法案例；
4. 一个多起点重启改善成功率的案例。

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

实施提示：

1. 先做可解释的常规近似推断，不要一开始追求最复杂稳健推断；
2. 先让正常模型可稳定输出，再单独处理异常模型 warning；
3. 推断和 fit indices 的 warning 需要共享同一套收敛上下文。

输入 / 输出契约（建议）：

- 输入：
  1.  最终参数向量
  2.  样本协方差、implied covariance、样本量
  3.  优化收敛状态与失败分类
- 输出：
  1.  带 `estimate / se / z / p / ci_low / ci_high` 的参数表
  2.  `AIC/BIC/SRMR/CFI/TLI/RMSEA` 等 fit indices
  3.  顶层 warning 与推断/拟合摘要

建议算法与规则：

1. 先根据 Hessian 或近似信息矩阵求标准误；
2. 对不可逆或近奇异情形给出推断 warning，而不是静默返回正常结果；
3. fit indices 优先保证：
   - 正常模型非 `nan`
   - 异常模型可解释失败
4. 对 `summary()` / `to_markdown()` 统一展示：
   - 是否收敛
   - 是否可稳定推断
   - 主要拟合指标

最低测试要求：

1. 一个正常模型：推断字段非空、fit indices 有限；
2. 一个 Hessian 异常模型：参数仍可返回估计值，但推断有 warning；
3. 一个近奇异或欠识别模型：fit indices 行为可解释；
4. 一个结果展示测试：`summary()` 至少能展示收敛、推断与 fit 摘要。

补充实施提示：

1. 先追求“可解释”和“可诊断”，再追求更复杂的稳健统计；
2. 推断与 fit indices 应共享同一套收敛/失败上下文，避免结果互相矛盾；
3. 对异常模型，优先输出明确 warning，而不是把所有字段静默填成 `nan`。

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

---

## 12. 下一阶段执行路线（已整合原“SEM 后续实施路线图”）

本节聚焦“在现有实现基础上，接下来应该优先做什么”。与前文 Phase 1-4 的长期规划不同，这里强调的是**短周期执行顺序**与**当前工程缺口**。

### 12.1 当前工程状态（整合视角）

当前已具备：

1. `syntax` 与 `spec` 已统一到 `ModelSpec`。
2. measurement（`Lambda/Theta`）与 structural（`Beta/Gamma/Psi`）矩阵草图已可构建。
3. 全局 `parameter_index -> vector_position` 已统一。
4. ML 原型（`build_implied_covariance` + `optimize_ml_parameters`）已接入 `SEMModel.fit`。
5. 基础推断、基础拟合指标与优化鲁棒性原型已落地。

当前关键缺口：

1. 推断层已完成第一批状态化增强，但数值稳定性与异常场景覆盖仍需继续增强。
2. 拟合指标已完成第一批状态化增强，但稳健版本与更细边界解释仍需继续补齐。
3. 优化器容错能力仍需继续工程化。
4. 高级路径尚未覆盖：`MLR/WLSMV`、多组不变性、完整端到端数值回归。

### 12.2 目标状态（下一阶段）

完成后应达到：

1. `SEMModel.fit(...)` 可输出可解释参数表（估计值 + SE + 显著性）。
2. 正常模型下主要拟合指标不再是 `nan`。
3. 优化失败时可返回结构化诊断，而不是仅依赖笼统失败状态。
4. 有稳定的数值回归测试，重构不易破坏结果。

### 12.3 短周期优先级

这一部分给出的是“当前最适合直接开工”的任务，而不是长期愿景清单。为了便于实际开发，下面每个优先项都补充了：

1. 要解决什么问题；
2. 为什么重要；
3. 涉及哪些模块；
4. 应怎样逐步实行。

#### P0：推断层最小可用

目标：把“能估计参数”升级为“能解释参数”。

实现重点：

1. 稳定 `inference` 模块，继续完善 `SE/z/p/CI`。
2. 在 `summary()` / `to_markdown()` 中更稳定地展示推断摘要。
3. 对不可计算场景输出明确原因，而不是静默 `nan`。

验收标准：

1. 至少 1 个简单模型可稳定产出非空推断字段。
2. 新增推断字段不破坏旧结果结构。

##### 详细实施说明

**要解决什么问题**

当前 `SEMModel.fit(...)` 已能输出参数估计，但研究者还缺少最关键的解释信息：

1. 参数不确定性；
2. 显著性；
3. 哪些参数结果不可解释。

**为什么这一步最先做**

1. 从心理学解释角度，路径系数只有在结合 SE / CI / 显著性后才有研究意义；
2. 从统计实现角度，推断层是把“数值优化结果”转成“分析结果”的关键桥梁；
3. 如果这一层不稳，后续报告、比较和方法扩展都会缺少可信输出。

**涉及模块**

1. `src/psysem/inference/`
2. `src/psysem/result.py`
3. `src/psysem/reporting.py`
4. `src/psysem/core.py`
5. `tests/test_sem_inference.py`

**建议实施步骤**

1. 先盘点当前 `SEMResult` 中已经暴露的推断字段与缺失字段；
2. 明确 `inference` 层最小输出契约：`estimate / se / z / p / ci_low / ci_high / warning`；
3. 将不可计算场景分类：
   - Hessian 不可逆；
   - 近奇异；
   - 参数在边界；
   - 优化未稳定收敛；
4. 对这些场景统一 warning 文案，而不是只返回 `nan`；
5. 让 `summary()` / `to_markdown()` 能优先展示：
   - 是否收敛；
   - 是否成功计算 SE；
   - 是否存在不可解释参数；
6. 增加最小合成模型测试：
   - 一个正常可计算模型；
   - 一个 Hessian 异常或边界场景模型。

**边界处理要求**

1. 对不能计算推断的参数，不应假装结果有效；
2. 对整个模型不能稳定推断时，应在结果对象顶层给出摘要 warning；
3. 不能让推断失败破坏已有估计值输出。

#### P0：拟合指标真实化与边界解释

目标：把已有基础拟合指标从“可计算”推进到“可解释”。

实现重点：

1. 稳定 `AIC/BIC/SRMR/CFI/TLI/RMSEA` 的边界行为。
2. 在欠识别、近奇异、失败优化场景补充 warning。
3. 补更多正常路径与异常路径测试。

验收标准：

1. 正常模型下主要指标非 `nan`。
2. 边界模型能返回可解释 warning。

##### 详细实施说明

**要解决什么问题**

当前基础 fit indices 已接通，但还需要把“能算”进一步提升为“在边界场景下也能解释”。

**为什么这一步重要**

1. 从心理学研究角度，SEM 的结论不能只看单个参数，还要看整体拟合；
2. 从数据分析角度，`CFI/TLI/RMSEA/SRMR` 的边界值行为决定了模型比较是否可信；
3. 如果异常模型只输出 `nan`，用户很难区分是模型差、数据差还是实现有问题。

**涉及模块**

1. `src/psysem/fit_indices.py`
2. `src/psysem/estimation/`
3. `src/psysem/result.py`
4. `tests/test_sem_fit_indices.py`

**建议实施步骤**

1. 先列出每个指标当前依赖的输入：样本协方差、implied covariance、自由参数数、baseline model 等；
2. 对正常路径补“结果合理性”断言，而不只检查字段存在；
3. 对异常路径至少覆盖：
   - 欠识别；
   - 近奇异矩阵；
   - 优化失败；
   - baseline model 无法稳定计算；
4. 对每一类异常，明确是：
   - 返回 warning 后给出部分指标；
   - 还是阻断并不输出指标；
5. 在 `summary()` 中补 fit 指标摘要和异常来源说明。

**边界处理要求**

1. 不能把所有失败都折叠成同一个 warning；
2. 正常模型要优先保证非 `nan`；
3. 异常模型要优先保证“可解释失败”。

#### P1：优化鲁棒性

目标：让优化器从“原型可跑”升级为“工程可用”。

实现重点：

1. 完善 `SEMFitConfig` 的重启、失败分类和默认策略。
2. 增强初值、边界、矩阵奇异等失败场景诊断。
3. 让 `optimization_info` 更适合直接写入结果报告。

验收标准：

1. 常见失败场景可复现且有明确分类。
2. 重启策略有测试覆盖，且可提升成功率。

##### 详细实施说明

**要解决什么问题**

优化器当前已经能跑，但还需要从“偶尔可用”提升到“更可重复、更可诊断”。

**为什么排在推断和 fit indices 之后**

1. 在当前阶段，解释层稳定性比继续堆优化技巧更关键；
2. 只有先明确哪些结果值得解释，优化器增强才知道要为哪些失败模式服务；
3. 鲁棒性增强应建立在已有测试基线上，而不是先盲目重构。

**涉及模块**

1. `src/psysem/estimation/contracts.py`
2. `src/psysem/estimation/ml.py`
3. `src/psysem/core.py`
4. `tests/test_sem_estimation_config.py`
5. `tests/test_sem_estimation_ml.py`

**建议实施步骤**

1. 先枚举当前失败分类是否足够覆盖常见问题：
   - convergence
   - bounds
   - matrix singularity
   - implied covariance invalid
   - specification
2. 统一 `optimization_info` 字段命名，使其可直接进入结果报告；
3. 对重启策略补最小评估：
   - 单起点 vs 多起点是否改善成功率；
   - 是否导致结果不稳定漂移；
4. 明确默认策略：什么时候自动重启，什么时候直接失败；
5. 让 warning 与失败分类能一一对应，而不是散落在不同层。

**边界处理要求**

1. 重启不能掩盖模型本身不可识别的问题；
2. 边界命中应明确记录，而不是只默默裁剪参数；
3. 优化失败时仍应尽量保留诊断上下文。

#### P0：测试与回归门禁

目标：建立数值稳定的 SEM 回归防线。

实现重点：

1. 固定随机种子与合成数据。
2. 对关键指标设置容差断言。
3. 覆盖 measurement-only、measurement+structural、`spec` 入口三类主路径。

建议重点测试文件：

1. `tests/test_sem_inference.py`
2. `tests/test_sem_fit_indices.py`
3. `tests/test_sem_estimation_config.py`
4. `tests/test_sem_estimation_ml.py`

##### 详细实施说明

**要解决什么问题**

让 SEM 从“当前这次能跑”变成“后续每次改动后仍可控”。

**为什么这一项是 P0**

1. SEM 是高数值敏感模块，没有回归防线就很难持续开发；
2. 后续任何 estimator、reporting 或 ordinal 扩展都会依赖这些基线；
3. 这也是把原型阶段推进到工程阶段的必要条件。

**建议实施步骤**

1. 固定至少一组 measurement-only 合成数据；
2. 固定至少一组 measurement + structural 合成数据；
3. 为关键输出定义容差断言：
   - 是否收敛；
   - 指标是否有限；
   - 推断字段是否存在；
   - warning 是否符合预期；
4. 将 `syntax` 入口与 `spec` 入口都纳入回归测试；
5. 对失败场景建立专门测试，而不是只测 happy path。

**边界处理要求**

1. 测试不应过度绑定精确小数，避免脆弱；
2. 但也不能只测 shape，必须开始测“数值是否有效”；
3. 回归样例应覆盖至少一个失败分类路径。

### 12.4 建议执行顺序

建议按以下顺序推进：

1. 先补推断层稳定性。
2. 再补拟合指标边界行为。
3. 同步补齐数值回归测试。
4. 再继续增强优化鲁棒性。
5. 最后扩展 `MLR/WLSMV`、多组与不变性。

#### 12.4.1 可直接执行的短周期清单

| 顺序 | 任务             | 主要目标             | 首要产出                                 | 最低测试要求                                         |
| ---- | ---------------- | -------------------- | ---------------------------------------- | ---------------------------------------------------- |
| 1    | 推断层稳定化     | 让参数结果可解释     | 更稳定的 `SE/z/p/CI` 与 warning          | 1 个正常模型 + 1 个 Hessian/边界异常模型             |
| 2    | 拟合指标边界解释 | 让整体拟合可判断     | 更清晰的 fit warning 与非 `nan` 正常结果 | 1 个正常模型 + 1 个欠识别/近奇异模型                 |
| 3    | 数值回归门禁     | 建立稳定开发基线     | 固定 seed 的 regression tests            | measurement-only + measurement/structural + 失败路径 |
| 4    | 优化鲁棒性增强   | 提高可重复性与诊断性 | 更清晰的 `optimization_info` 与重启策略  | 失败分类测试 + 重启改进测试                          |
| 5    | estimator 扩展   | 在稳定基线后扩能力   | `MLR/WLSMV` 或多组路线设计               | 不破坏前 4 项回归                                    |

#### 12.4.2 每一轮开发前的检查问题

开始编码前，建议先确认：

1. 这次改动主要落在 `measurement`、`structural`、`estimation`、`inference` 还是 `fit_indices`？
2. 这次改动解决的是“解释问题”还是“数值问题”？
3. 是否已经明确正常路径和边界路径各至少一个测试？
4. 是否需要同步更新 `summary()`、`to_markdown()` 或结果字段说明？

### 12.5 每次迭代的完成定义（DoD）

每个小阶段完成前，至少满足：

1. 代码实现与文档同步更新。
2. `python -m ruff check .` 通过。
3. `python -m mypy src` 通过。
4. `python -m pytest -q` 全绿。
5. 新增能力至少包含 1 个正常路径测试与 1 个边界路径测试。

### 12.6 关联文档

1. [ESEM 模块化判断工作流实施文档（ZH）](esem-modular-workflow.zh-CN.md)
2. [ESEM 最小可跑路径（MVP，ZH）](esem-mvp-run.zh-CN.md)

---

## 13. 如果把当前文档补到“可直接开工”，还应如何使用

本节给出一个推荐用法，避免把本文档只当作路线图阅读。

### 13.1 作为开发前检查清单使用

开始某一步前，先回答：

1. 这一步属于 measurement、structural、estimation、inference 还是 fit indices？
2. 这一步解决的是解释问题、估计问题，还是工程稳健性问题？
3. 这一步的输入和输出契约是否已经在文档中写清楚？
4. 是否已经定义正常路径与边界路径测试？

### 13.2 作为迭代记录模板使用

建议每次完成一个小批次时，至少同步更新：

1. 当前进展；
2. 涉及模块；
3. 新增 warning / result 字段；
4. 新增测试；
5. 下一批建议。

### 13.3 当前最推荐的直接开工点

如果只选一个起点，建议优先做：

1. 推断层稳定化；
2. 然后补 fit indices 边界行为；
3. 同步建立回归测试。

原因不是它们最“炫”，而是它们最直接决定：

1. `SEMResult` 是否可解释；
2. 模型拟合是否可判断；
3. 后续 estimator 扩展是否有稳定基线。

### 13.4 如果要继续把全文补成“逐步开发手册”，建议优先补哪些部分

若后续还要继续细化，建议按下面顺序补全文档：

1. 先把 Phase 3 的 Step 9 / Step 10 扩成完整执行模板；
2. 再把 Phase 2 的识别性检查与多 block 组装补成模板；
3. 最后再回头补 parser/契约层的实施细节。

原因是：

1. 当前最接近实际开发的工作集中在 Phase 3；
2. 识别性与多 block 规则是后续心理测量解释的重要门槛；
3. parser/契约层虽然重要，但当前短周期里不是最紧迫的瓶颈。

### 13.5 给后续 agent 的直接开发约束

如果后续由新的 agent 继续开发 SEM，建议把下面这些要求视为硬约束，而不是建议项。

#### 约束 A：不要跳过当前优先级顺序

后续 agent 默认应按以下顺序推进，除非文档已明确改优先级：

1. 推断层稳定化；
2. fit indices 边界解释；
3. 数值回归测试；
4. 优化鲁棒性；
5. 最后才扩 `MLR/WLSMV`、多组与不变性。

#### 约束 B：任何新增功能必须同时更新三处

1. 代码实现；
2. 对应测试；
3. 本文档中的“当前进展 / 短周期优先级 / 使用说明”中至少一处。

#### 约束 C：不要只补 happy path

每个新能力至少补：

1. 一个正常路径测试；
2. 一个边界或失败路径测试。

对 SEM 来说，缺少失败路径测试等于没有真正完成实现。

#### 约束 D：不要把异常静默吞掉

如果某一步出现以下情况：

1. Hessian 不可逆；
2. implied covariance 非法；
3. 矩阵近奇异；
4. 优化未收敛；
5. 模型欠识别；

则 agent 需要优先：

1. 明确分类；
2. 给出 warning 或错误；
3. 在结果对象或 `optimization_info` 中保留上下文。

不允许只返回大量 `nan` 而不解释原因。

#### 约束 E：不要随意重构参数索引契约

measurement / structural / estimation / inference 当前都依赖统一参数索引视图。后续 agent：

1. 不能在没有回归测试的情况下改参数索引语义；
2. 如果必须改，必须同步更新：
   - 参数表
   - implied covariance 路径
   - 推断层
   - 回归测试

#### 约束 F：结果解释优先于功能堆叠

如果在两件事之间选择：

1. 再加一个 estimator 或高级功能；
2. 让当前 `ML` 结果更可解释、更可诊断；

优先选择第 2 项。

#### 约束 G：每次提交前最少检查项

后续 agent 在声明“完成”前，至少应确认：

1. 新字段是否进入 `SEMResult` 或相关输出；
2. `summary()` / `to_markdown()` 是否需要同步；
3. 正常路径测试是否通过；
4. 边界路径测试是否通过；
5. 本文档是否需要同步更新当前状态。
