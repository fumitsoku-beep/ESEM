# ESEM 生态与实践 Baseline（ZH）

当前日期：2026-03-11  
适用项目：`psysem`（`main` 分支）

## 1. 文档目的

这份文档回答一个问题：**“目前大家在做什么”**。  
用于后续重写用户向 README、绘制流程图、以及规划 `psysem` 的功能优先级。

范围聚焦心理测量/心理学场景下的 ESEM（含 EFA->ESEM->结构路径->不变性测试流程）。

---

## 2. 主流工具现状（包/软件对比）

| 工具 | 生态位置 | ESEM能力 | 心理学场景常见用法 | 结论 |
| --- | --- | --- | --- | --- |
| **Mplus** | 行业基准（商业软件） | 原生 ESEM（EFA 因子可直接嵌入 SEM）、多种旋转（GEOMIN/TARGET/BI-GEOMIN 等）、大量官方示例（如 5.24/5.25/5.26/5.27） | 量表结构验证、纵向/多组不变性、bifactor-ESEM、复杂模型 | 仍是“功能最全 + 教程最成熟”的参考标准 |
| **lavaan (R)** | 开源主流 SEM 引擎 | `efa("block")*` 直接在语法里定义 ESEM block；`sem(..., rotation="geomin")`；支持 EFA/CFA 混合块 | 用开源工作流复现 Mplus 典型 ESEM；心理学论文中使用持续增长 | 已是开源 ESEM 的核心实现之一 |
| **psych + GPArotation (R)** | 心理测量 EFA 基础设施 | 强在 EFA与旋转、因子数判定（PA/MAP/VSS），`psych::esem` 提供基于 factor-extension 的探索式 ESEM | 前期量表探索、因子保留决策、旋转稳定性分析 | 常被用作 ESEM 前置探索层，而非完整 SEM 终端 |
| **EFAtools (R)** | EFA 工具增强层 | 提供多方法/多实现（EFAtools/psych/SPSS）比较与平均、丰富旋转与保留策略 | 在心理测量里做“稳健 EFA 决策”和模型敏感性分析 | 对“自动化选因子 + 稳健流程”非常实用 |
| **esemComp (R)** | ESEM 辅助工具 | 面向 `lavaan` 生成 ESEM-within-CFA 语法，支持目标旋转矩阵构建 | 以 `lavaan` 为底座快速搭 EWC（ESEM-within-CFA） | 适合工程化封装，但底层仍依赖 lavaan |
| **Python: factor_analyzer** | Python EFA/CFA 常用包 | EFA + CFA（旋转丰富），但不是完整 ESEM-SEM 框架 | Python 里常用于前置 EFA、KMO/Bartlett、旋转解释 | 可做 EFA层，不足以独立覆盖主流心理学 ESEM 全流程 |
| **Python: semopy** | Python SEM 包 | 有 `semopy.efa`，但官方文档明确其 EFA 路径是“unorthodox approach”（基于聚类/稀疏PCA启发） | 可用于 SEM 研究与原型验证 | 不是心理测量领域“标准 ESEM 实践路径”的主力实现 |

---

## 3. 心理学里最常见的 ESEM 流程（实务视角）

> 实务上，不是“直接跑一个 ESEM 就结束”，而是分阶段验证并反复比较。

### Step 1: 明确测量理论与条目层假设
1. 先定义构念和条目归属假设（理论先行）。
2. 明确哪些跨负荷是“可预期的小跨负荷”，哪些是不合理跨负荷。

### Step 2: 数据准备与变量类型决策
1. 连续变量常用 ML/MLR；有序分类（Likert/ordinal）常优先 WLSMV 系列。
2. 先做缺失、异常、分布、相关矩阵可因子化检查（KMO/Bartlett 等）。

### Step 3: 因子数候选确定（EFA阶段）
1. 并行分析（PA）+ MAP + Scree + 理论可解释性联合决策。
2. 不是只看单一准则；心理测量里常用“统计证据 + 理论解释”双重筛选。

### Step 4: 并行比较 CFA 与 ESEM 测量模型
1. 先拟合 CFA（ICM-CFA）基线，再拟合 ESEM。
2. 比较拟合指标、跨负荷、因子相关大小与可解释性。
3. 心理学研究里常见现象：若强行零跨负荷，CFA 因子相关可能被抬高，区分效度受损。

### Step 5: 旋转策略与局部最优控制
1. 默认常见是 oblique geomin（尤其心理测量多维构念场景）。
2. 有强理论时常用 target rotation（接近“可确认式 ESEM”）。
3. 多起点随机重启检查局部最优，是实务稳定性关键步骤。

### Step 6: 进入结构路径（SEM 部分）
1. 在确认测量模型后再接结构路径（潜变量回归、中介等）。
2. 优先检查路径解释性与测量层稳定性是否一致。

### Step 7: 多组/纵向不变性
1. 常见顺序：configural -> metric -> scalar。
2. ESEM 已被广泛用于不变性框架（尤其在传统 CFA 过于刚性时）。

### Step 8: 报告与复现
1. 至少报告：估计器、旋转、随机起点设置、因子保留依据、主要拟合指标、跨负荷解释规则。
2. 心理测量论文越来越重视可复现（代码、输入、版本信息）。

---

## 4. 当前“大家在做什么”的共识点（可作为本项目 baseline）

1. **R + Mplus 双轨**仍是心理测量 ESEM 主流生态。  
2. **ESEM 不再只被当作纯探索工具**，而是常用于确认性比较（与 CFA 对照）和不变性测试。  
3. **流程化决策**是趋势：因子数、旋转、估计器、重启策略都要显式化、可复现。  
4. **Python 生态在 EFA 很常见，但完整心理学 ESEM 主流程仍偏弱**（与 Mplus/lavaan 相比）。  

---

## 5. 对 `psysem` 的 baseline 启发（下一步设计依据）

按优先级建议：

1. P0：把 `EFA -> ESEM measurement -> SEM structural` 串成可复现 pipeline（含 estimator/rotation/restarts 显式配置）。
2. P0：在结果层固定输出“心理学可读信息”：
   `loadings/cross-loadings/factor correlations/fit indices/invariance step summary`。
3. P1：完善 ordinal 路径（WLSMV 类）与不变性流程（configural/metric/scalar）。
4. P1：支持 target rotation 与 ESEM-within-CFA 工作流入口。
5. P2：再扩展 bifactor-ESEM、Bayesian/robust 变体。

---

## 6. baseline 对比矩阵（主流实践 vs 当前 `psysem`）

| 维度 | 主流实践（Mplus/lavaan/R） | 当前 `psysem`（截至 2026-03-11） | 差距判断 |
| --- | --- | --- | --- |
| 数据层 | 缺失/异常/可因子化检查流程化 | 已有 `data` 模块做输入与规格校验 | 基本对齐，可继续扩展缺失机制与数据诊断深度 |
| 因子数判定 | PA + MAP + Scree + 理论解释联合决策 | 已有 Phase 1/2 候选评分与选择 | 基本对齐，需补更多稳健准则与可视化输出 |
| 旋转与重启 | geomin/target + 多起点重启是常规 | 已有重启与失败诊断基础能力 | 方向正确，需补 target rotation 与更完整旋转族 |
| 测量模型 | ESEM 与 CFA 并行比较是常规 | 以 EFA 管线为主，ESEM 测量层尚未完整落地 | 核心缺口（P0） |
| 结构模型 | 在稳定测量层上接 SEM 路径 | SEM 已有参数映射/优化原型 | 方向正确，需与 ESEM 测量层完整打通 |
| 不变性 | configural/metric/scalar 是心理学常规报告 | 尚未形成完整自动化流程 | 核心缺口（P1） |
| 报告复现 | 强调可解释输出 + 版本可复现 | 已有文档与测试基础 | 需统一“心理学可读”报告模板（P0/P1） |

---

## 7. 主要参考来源（用于本 baseline）

1. lavaan ESEM/EFA 教程（官方）：https://lavaan.ugent.be/tutorial/efa.html  
2. lavaan 分类数据与估计器（官方）：https://lavaan.ugent.be/tutorial/cat.html  
3. lavaan 估计器说明（官方）：https://lavaan.ugent.be/tutorial/est.html  
4. Mplus 用户手册（ESEM 示例 5.24+）：https://www.statmodel.com/download/usersguide/MplusUserGuideVer_8.pdf  
5. Mplus 示例 5.25（ESEM with EFA + CFA factors）：https://www.statmodel.com/usersguide/chap5/ex5.25.html  
6. Mplus ANALYSIS 命令（旋转/并行分析/设置）：https://www.statmodel.com/download/usersguide/Chapter16.pdf  
7. Mplus 8.9/8.10/8.11 addendum（含 ESEM 不变性自动化）：https://www.statmodel.com/download/Version%208.9%208.10%20and%208.11%20Addendum.pdf  
8. psych 手册（含 `esem`、`fa.parallel`、旋转与局部最优讨论）：https://personality-project.org/r/psych/vignettes/psych.manual.pdf  
9. psych `fa.parallel` 帮助页：https://personality-project.org/r/psych/help/fa.parallel.html  
10. GPArotation 手册（旋转目标函数与算法）：https://r-forge.r-universe.dev/GPArotation/doc/manual.html  
11. EFAtools 手册：https://stat.ethz.ch/CRAN/web/packages/EFAtools/EFAtools.pdf  
12. esemComp 文档（ESEM-within-CFA 辅助）：https://mateuspsi.github.io/esemComp/  
13. factor_analyzer 文档：https://factor-analyzer.readthedocs.io/en/latest/  
14. semopy 文档（含 `semopy.efa` 描述）：https://www.semopy.com/docs/efa.html  
15. Marsh et al., 2014（临床心理学综述，ESEM 心理学应用框架）：https://www.vanderbilt.edu/psychological_sciences/graduate/programs/quantitative-methods/quantitative-content/marsh_morin_parker_kaur_2014.pdf
