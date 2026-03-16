# SEM 下一阶段完成文档（含外部 Benchmark 与目标包，ZH）

当前日期：2026-03-16  
适用分支：`main`

---

## 1. 本文档目的

本文档回答四个问题：

1. 当前如果优先完成 `SEM`，下一步最应该做什么；
2. 为什么现在不应优先继续堆 estimator 或高级功能；
3. 如果要建立研究级可信度，应该参考哪些成熟包；
4. 当前项目应优先补哪些外部 benchmark，用哪些数据、模型与指标来对照。

---

## 2. 当前判断：SEM 现在最缺什么

当前 `psysem` 的 SEM 已经具备：

1. `syntax/spec` 统一入口；
2. measurement / structural 构建；
3. 参数索引主线；
4. 基础 ML 优化；
5. 基础推断；
6. 基础 fit indices；
7. 结果摘要与 Markdown 输出。

因此，当前最缺的不是“再多一个功能点”，而是：

1. **外部可比性**；
2. **数值稳定基线**；
3. **可长期回归的 benchmark**；
4. **与成熟软件/成熟包接近的结果语义**。

一句话说：

> 当前 SEM 更需要“做成可信基线”，而不是“继续功能堆叠”。

---

## 3. 当前最应该做什么

建议按下面顺序推进。

### P0：补 SEM benchmark / 数值回归

优先补这四类模型：

1. 单因子 CFA benchmark；
2. 多因子 CFA benchmark；
3. 测量 + 结构路径 benchmark；
4. 边界 warning / failure benchmark。

这一步的目标不是只测“能跑”，而是测：

1. 参数估计是否合理；
2. `SE` 是否合理；
3. `CFI/TLI/RMSEA/SRMR/AIC/BIC` 是否接近成熟实现；
4. warning / status 是否稳定。

### P0：继续增强优化鲁棒性

在 benchmark 基线之后，应继续补：

1. 初值稳定性；
2. 重启策略表现；
3. 边界命中行为；
4. 失败分类是否清楚。

### P1：继续增强结果契约

当前已经有：

1. `inference_status`；
2. `fit_status`；
3. `summary()`；
4. `to_markdown()`。

下一步应继续向“研究者真正可用的结果对象”推进，例如：

1. 更清晰的参数表导出；
2. 更统一的 warning 分层；
3. 更接近研究报告的输出布局。

### P2：最后再扩 estimator

如：

1. `MLR`
2. `WLSMV`
3. 更完整 categorical / ordinal SEM

原因很简单：

如果基础 ML 主线还没有足够 benchmark，过早扩 estimator 会把问题放大。

---

## 4. 为什么现在必须这样做

### 4.1 从统计开发顺序看

一个 SEM 包如果要稳定，必须先保证：

1. 基础模型能稳定复现；
2. 基础结果与成熟实现大体一致；
3. 边界 warning 有一致语义；
4. 后续每次改动都能被回归测试约束。

如果跳过这一步，直接去扩：

1. `WLSMV`
2. 多组不变性
3. 更复杂 ESEM
4. 更复杂报告系统

会出现一个问题：

- 功能越来越多；
- 但最基础的 SEM 数值基线还没有被证明可信。

### 4.2 从研究者信任角度看

研究者真正信任一个 SEM 工具，不是因为它“功能多”，而是因为：

1. 同一个模型重复运行稳定；
2. 与常见成熟实现结论接近；
3. warning 清楚；
4. 结果可解释、可复现。

所以现在最该做的是：

- **建立基准信任**。

---

## 5. 当前应参考的成熟包与目标定位

这一节不是简单列“能参考哪些包”，而是明确：

1. 主要对照目标是谁；
2. 次级参考是谁；
3. 每个包/软件在本项目中的作用是什么。

### 5.1 第一目标：`lavaan`（R）

建议作为 **首要开源对照标准**。

原因：

1. 社区广泛使用；
2. 文档清楚；
3. 例子标准；
4. 内置多个经典 SEM 数据集与模型；
5. 便于提取参数、SE、fit indices 作为 benchmark 基准。

建议在文档与测试中把 `lavaan` 视为：

- **首要开放 benchmark 目标**。

特别适合用于：

1. CFA benchmark；
2. 经典 SEM benchmark；
3. baseline fit indices benchmark。

### 5.2 第一目标：`Mplus`

建议作为 **研究级闭源参考标准**。

原因：

1. SEM / ESEM / categorical 支持成熟；
2. ESEM 方法学与实务地位都很高；
3. User's Guide 中有大量标准示例；
4. 特别适合作为后续 ESEM、ordinal、robust 路线的高阶对照。

建议在文档与测试中把 `Mplus` 视为：

- **高阶方法与研究级结果参考目标**。

特别适合用于：

1. 复杂 SEM；
2. ESEM baseline；
3. future ordinal / robust benchmark。

### 5.3 第二目标：`semopy`（Python）

建议作为 **Python 生态内的横向参考对象**。

原因：

1. 同样是 Python 语境；
2. 使用类 `lavaan` 语法；
3. 支持 SEM、polychoric/polyserial、EFA 等能力；
4. 有助于比较 Python 生态中的实现方式与输出结构。

但不建议把它作为唯一主对照。

更合理的定位是：

- **次级参考实现**。

特别适合用于：

1. Python 内部接口参考；
2. 辅助验证估计与相关矩阵路线；
3. 后续 ordinal / polychoric 路线的补充比较。

### 5.4 第二目标：`psych`（R，用于 EFA）

虽然本文聚焦 SEM，但若涉及 EFA bridge 或后续 ESEM 测量部分，也建议参考 `psych`。

定位：

- **EFA/ESEM 前置结构参考工具**。

特别适合用于：

1. EFA extraction / rotation benchmark；
2. communalities / complexity / residual 参考；
3. ordinal 场景下的 EFA 前置结构对照。

### 5.5 不建议作为首要标准的对象

#### `AMOS`

`AMOS` 很适合作为研究者常用软件背景说明，但不建议把它作为首要 benchmark 标准。

原因：

1. 图形化工作流较强，但自动化提取 benchmark 不如 `lavaan`；
2. 在代码型项目中，不如 `lavaan` / `Mplus` 便于系统回归；
3. 更适合作为“结果是否大体一致”的人工验证对象。

因此定位更适合是：

- **人工二次核对参考**，而不是首要自动 benchmark 来源。

---

## 6. 当前建议优先采用的外部 benchmark 数据与模型

### 6.1 `HolzingerSwineford1939`（来自 `lavaan`）

建议用途：

1. **单因子/三因子 CFA benchmark**；
2. measurement 主线 benchmark；
3. fit indices benchmark。

经典模型：

```text
visual  =~ x1 + x2 + x3
textual =~ x4 + x5 + x6
speed   =~ x7 + x8 + x9
```

优点：

1. 教材级经典数据；
2. 社区大量使用；
3. `lavaan` 官方教程直接给出；
4. 很适合做第一批标准化 benchmark。

建议对照输出：

1. 载荷；
2. 潜变量协方差；
3. 观测残差方差；
4. `CFI/TLI/RMSEA/SRMR/AIC/BIC`；
5. 迭代次数和收敛状态；
6. 参数 `SE/z/p`。

### 6.2 `PoliticalDemocracy`（来自 `lavaan`）

建议用途：

1. **完整 SEM benchmark**；
2. measurement + structural 主线 benchmark；
3. 结构路径与拟合指标联合验证。

经典模型：

```text
ind60 =~ x1 + x2 + x3
dem60 =~ y1 + y2 + y3 + y4
dem65 =~ y5 + y6 + y7 + y8

dem60 ~ ind60
dem65 ~ ind60 + dem60

y1 ~~ y5
y2 ~~ y4 + y6
y3 ~~ y7
y4 ~~ y8
y6 ~~ y8
```

优点：

1. `lavaan` 官方 SEM 教程直接提供；
2. 包含测量层、结构层、残差相关；
3. 非常适合作为当前项目进入“完整 SEM benchmark”的第一批模型。

建议对照输出：

1. latent loadings；
2. regressions；
3. residual covariances；
4. fit indices；
5. 标准误与显著性方向；
6. warning / convergence 信息。

### 6.3 `Mplus User's Guide` Chapter 5 示例

建议用途：

1. 作为 `lavaan` 之外的 **研究级参考模型集合**；
2. 后续扩到更复杂 CFA/SEM 时作为补充 benchmark。

当前建议：

1. 暂先作为文档级目标来源；
2. 在基础 `lavaan` benchmark 建好后，再选 1-2 个简洁模型落地为回归测试。

### 6.4 `Mplus User's Guide` Chapter 4 示例

建议用途：

1. 作为后续 EFA / ESEM / exploratory measurement 的高阶参考来源；
2. 当前可先登记，不急于立刻转成自动测试。

### 6.5 `semopy` 自带或教程示例模型

建议用途：

1. 作为 Python 生态内补充对照；
2. 对照语法、参数命名、结果字段与估计主线。

当前定位：

1. 不作为第一批主 benchmark；
2. 可作为后续“横向 Python 包比较”资料。

---

## 7. 当前最值得优先落地的 benchmark 计划

### Benchmark SEM-1：`HolzingerSwineford1939` 三因子 CFA

目标包：

1. `lavaan`（主）
2. `semopy`（辅，可选）

目标：

1. 建立第一个正式 CFA benchmark；
2. 锁定 loadings / residual variances / latent covariances / fit indices / SE。

建议放入：

1. 新测试文件：`tests/test_sem_benchmark_hs1939.py`

建议断言：

1. 参数估计在容差内；
2. `CFI/TLI/RMSEA/SRMR` 在容差内；
3. `SE` 在数量级上与参考结果一致；
4. 收敛与 warning 状态稳定。

### Benchmark SEM-2：`PoliticalDemocracy` 完整 SEM

目标包：

1. `lavaan`（主）
2. `Mplus`（高阶人工参考）

目标：

1. 建立 measurement + structural 的完整 benchmark；
2. 锁定结构路径与拟合指标；
3. 检查 residual covariance 主线。

建议放入：

1. 新测试文件：`tests/test_sem_benchmark_political_democracy.py`

建议断言：

1. 载荷容差；
2. 路径系数容差；
3. 关键协方差容差；
4. fit indices 容差；
5. 结果状态字段稳定。

### Benchmark SEM-3：边界 warning benchmark

目标包：

1. `lavaan`（主）
2. `AMOS` / `Mplus`（人工参考）

目标：

1. 构建接近欠识别或近奇异的模型；
2. 比较当前项目与成熟软件的 warning 语义是否大体一致。

建议放入：

1. 新测试文件：`tests/test_sem_benchmark_boundary_cases.py`

建议断言：

1. 不是逐字对齐 warning；
2. 而是要求失败分类、部分可用状态与 warning 层级合理。

---

## 8. 建议的容差策略

由于不同包之间会存在：

1. 优化器差异；
2. 初值差异；
3. 标识化差异；
4. 因子符号/顺序差异；
5. 数值近似差异；

所以 benchmark 不应要求“逐位小数完全一致”。

建议采用分层容差：

### 8.1 参数估计

1. 载荷与路径：相对容差 + 绝对容差；
2. 先看数量级与方向，再看小数精度；
3. 必要时先做参数对齐（如因子顺序或符号翻转）。

### 8.2 拟合指标

1. `CFI/TLI/SRMR/RMSEA` 采用固定容差；
2. 对 `AIC/BIC` 采用较宽容差；
3. 若某指标在某包里因实现细节略有差异，可优先对比方向与相对排序。

### 8.3 标准误

1. 不要求极端精确；
2. 先要求非空、数量级合理、显著性方向一致；
3. 在基础主线稳定后再收紧容差。

---

## 9. 与其他包相比，为什么现在必须先做 benchmark

### 与 `lavaan` 相比

`lavaan` 的强项不是只有功能多，而是：

1. 官方教程模型标准；
2. 社区验证充分；
3. 参数、SE、fit indices 语义成熟；
4. 数据与例子可公开复用。

因此当前项目要真正进入“可信 SEM”，第一步就应该与 `lavaan` 建立稳定 benchmark。

### 与 `Mplus` 相比

`Mplus` 的强项在于：

1. 方法学覆盖更完整；
2. ESEM / ordinal / robust 路线成熟；
3. 研究界认可度高。

因此当前项目若未来想把 ESEM、ordinal、robust 做扎实，必须把 `Mplus` 放进长期目标，而不是只盯着内部 smoke tests。

### 与 `semopy` 相比

`semopy` 表明 Python 生态里已经存在研究级 SEM 方向的实现尝试。当前项目若想走得更稳，需要做到：

1. 不只是“也能跑”；
2. 而是“在 benchmark 与结果语义上更清楚、更稳定”。

---

## 10. 当前项目的目标包定位（建议直接写入开发约束）

建议把下面目标写成默认开发参照：

### 第一层目标包

1. `lavaan`：首要开放 benchmark 标准；
2. `Mplus`：高阶研究级参考标准。

### 第二层目标包

1. `semopy`：Python 生态横向参考；
2. `psych`：EFA / ESEM 前置结构参考。

### 第三层参考对象

1. `AMOS`：人工工作流与结果核对参考；
2. `SPSS`：EFA 报表与常规研究流程背景参考。

---

## 11. 当前建议的执行顺序

### 第一批（立刻做）

1. 建立 `HolzingerSwineford1939` benchmark；
2. 建立 `PoliticalDemocracy` benchmark；
3. 建立 1 个边界 case benchmark。

### 第二批（随后做）

1. 用 `semopy` 做横向复核；
2. 补 standardized 结果与参数表对照；
3. 收紧容差。

### 第三批（后续扩展）

1. 引入 `Mplus` 示例作为更高阶对照；
2. 进入 ordinal / robust / ESEM benchmark。

---

## 12. 完成标准（DoD）

当且仅当满足下面条件，才可以说“当前 SEM 基线更接近完成”：

1. 至少有 2 个正式外部 benchmark 数据/模型；
2. benchmark 已进入自动测试；
3. 参数、SE、fit indices 至少部分进入容差断言；
4. warning / status 行为有稳定测试；
5. 后续改动会被 benchmark 回归拦住。

---

## 13. 关联资料

### 当前仓库文档

1. [SEM Phase 实施文档（ZH）](sem-phase-implementation.zh-CN.md)
2. [当前测试流程总览（ZH）](current-testing-workflows.zh-CN.md)
3. [ESEM 模块化判断工作流实施文档（ZH）](esem-modular-workflow.zh-CN.md)

### 外部 benchmark 目标来源

1. `lavaan` CFA tutorial：`HolzingerSwineford1939`
2. `lavaan` SEM tutorial：`PoliticalDemocracy`
3. `Mplus User's Guide` Chapter 5：CFA / SEM examples
4. `Mplus User's Guide` Chapter 4：EFA examples
5. `semopy` 官方站点与教程

---

## 14. 一句话结论

当前 SEM 的下一步，不是优先继续加方法，而是优先把：

1. **benchmark**
2. **数值回归**
3. **warning 语义**
4. **外部可比性**

做成真正稳定的基线。
