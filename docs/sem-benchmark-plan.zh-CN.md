# SEM Benchmark 测试计划（ZH）

当前日期：2026-03-16  
适用分支：`main`

---

## 0. 当前状态（2026-03-16 更新）

本计划文档对应的 benchmark 准备工作已经开始落地：

1. 已固化第一批 benchmark 原始 CSV：
   - `tests/data/benchmark_hs1939_raw.csv`
   - `tests/data/benchmark_political_democracy_raw.csv`
2. 已为两组数据补充 provenance / citation 元数据：
   - `tests/data/benchmark_hs1939_reference.json`
   - `tests/data/benchmark_political_democracy_reference.json`
3. 已新增首个自动化 SEM benchmark 测试：
   - `tests/test_sem_benchmark_hs1939.py`
4. 已新增第二个自动化 SEM benchmark 测试（首版）：
   - `tests/test_sem_benchmark_political_democracy.py`
5. 已新增第一组边界 warning / failure benchmark 测试（首版）：
   - `tests/test_sem_benchmark_boundary_cases.py`

同时需要特别说明：

1. 当前仓库把这些数据视为“**公开可获取的 benchmark / example data**”；
2. 当前上游文档**不能证明这些数据属于 public domain（公共领域）**；
3. 因此仓库中显式保留来源、文献引用与复用说明，避免把“公开可获取”误写成“公共领域”。

---

## 1. 本文档目的

本文档把前一份 [SEM 下一阶段完成文档（ZH）](sem-next-steps.zh-CN.md) 中提出的 benchmark 目标，进一步细化成**可执行测试计划**。

它主要回答：

1. 先落地哪几个 benchmark；
2. 每个 benchmark 用什么外部数据或参考模型；
3. 用什么语法；
4. 比较哪些指标；
5. 容差如何设计；
6. 应该落在哪个测试文件中。

---

## 2. 总体执行原则

在把 benchmark 变成自动测试前，先统一几条规则。

### 2.1 benchmark 的目标不是逐位复制

不同包之间常常会存在：

1. 初值差异；
2. 优化器差异；
3. 标识化差异；
4. 因子顺序/符号差异；
5. 数值近似差异。

因此 benchmark 的目标不是“逐位完全相同”，而是：

1. 参数结构一致；
2. 数值量级接近；
3. 拟合指标接近；
4. warning / status 语义合理。

### 2.2 benchmark 分三层断言

建议所有 benchmark 都按三层断言写：

#### Level A：结构断言

1. 是否收敛；
2. 参数是否完整；
3. 拟合指标是否可用；
4. warning / status 是否符合预期。

#### Level B：方向与数量级断言

1. 参数正负方向是否一致；
2. 主次关系是否一致；
3. SE 是否非空且数量级合理；
4. 拟合指标是否落在合理区间。

#### Level C：容差数值断言

1. 载荷容差；
2. 路径系数容差；
3. 协方差/方差容差；
4. `CFI/TLI/RMSEA/SRMR/AIC/BIC` 容差。

### 2.3 benchmark 的优先顺序

建议按以下顺序落地：

1. `HolzingerSwineford1939` 三因子 CFA
2. `PoliticalDemocracy` 完整 SEM
3. 边界 warning benchmark
4. 后续再扩到 ordinal / robust / ESEM

---

## 3. Benchmark 1：`HolzingerSwineford1939` 三因子 CFA

### 3.1 benchmark 目的

这是当前最适合先落地的 **第一号 SEM benchmark**。

它用于验证：

1. measurement 主线是否稳定；
2. 多因子 CFA 参数是否合理；
3. `SE` 与 fit indices 是否大体接近 `lavaan`；
4. 当前项目是否已具备第一个标准公开 benchmark。

### 3.2 外部来源与目标包

#### 目标包

1. `lavaan`（主对照）
2. `semopy`（可选补充）

#### 数据来源

- `lavaan` 内置数据集：`HolzingerSwineford1939`

#### 推荐原因

1. 经典；
2. 官方教程直接给出；
3. 社区广泛使用；
4. 变量数适中；
5. 模型结构清楚。

### 3.3 推荐模型

```text
visual  =~ x1 + x2 + x3
textual =~ x4 + x5 + x6
speed   =~ x7 + x8 + x9
```

### 3.4 推荐参考输出

根据 `lavaan` 官方 CFA 教程，建议至少记录以下参考项：

1. `N = 301`
2. `df = 24`
3. `chi-square ≈ 85.306`
4. `CFI ≈ 0.931`
5. `TLI ≈ 0.896`
6. `RMSEA ≈ 0.092`
7. `SRMR ≈ 0.065`

参数层面，建议至少记录：

1. `x2` on `visual`
2. `x3` on `visual`
3. `x5` on `textual`
4. `x6` on `textual`
5. `x8` on `speed`
6. `x9` on `speed`
7. 三个潜变量之间的协方差
8. 9 个观测残差方差

### 3.5 当前项目中的建议测试步骤

1. 准备 `HolzingerSwineford1939` 数据副本；
2. 在测试中构造与 `lavaan` 一致的模型语法；
3. 用 `SEMModel(...).fit(data)` 跑模型；
4. 提取：
   - 载荷
   - 潜变量协方差
   - 残差方差
   - `SE`
   - fit indices
5. 与参考结果逐项比较。

### 3.6 建议断言

#### Level A：结构断言

1. `result.converged is True`
2. `result.parameter_inference` 非空
3. `result.optimization_info["fit_status"]` 为 `ok` 或稳定的 `partial`
4. 主要 fit indices 非 `nan`

#### Level B：方向与数量级断言

1. 所有非 marker 载荷均为正；
2. 主载荷大小排序与参考一致；
3. 三个潜变量协方差均为正；
4. `RMSEA`、`SRMR` 数值落在合理区间。

#### Level C：容差断言

建议第一批采用较宽容差：

1. 载荷：`abs_tol = 0.10 ~ 0.15`
2. 潜变量协方差：`abs_tol = 0.10 ~ 0.15`
3. 残差方差：`abs_tol = 0.10 ~ 0.20`
4. `CFI/TLI/RMSEA/SRMR`：`abs_tol = 0.03 ~ 0.05`

### 3.7 建议测试文件

- `tests/test_sem_benchmark_hs1939.py`

### 3.8 建议配套数据文件

建议新增：

- `tests/data/benchmark_hs1939_raw.csv`
- `tests/data/benchmark_hs1939_reference.json`

其中 `reference.json` 建议记录：

1. 数据来源说明；
2. 模型语法；
3. 参考软件与版本；
4. 关键参数与 fit 指标；
5. 容差建议。

---

## 4. Benchmark 2：`PoliticalDemocracy` 完整 SEM

### 4.1 benchmark 目的

这是当前最适合落地的 **第一号完整 SEM benchmark**。

它用于验证：

1. measurement + structural 主线是否稳定；
2. 结构路径估计是否合理；
3. residual covariance 处理是否合理；
4. 当前项目是否能在标准 SEM 模型上接近 `lavaan`。

### 4.2 外部来源与目标包

#### 目标包

1. `lavaan`（主对照）
2. `Mplus`（后续高阶人工核对）

#### 数据来源

- `lavaan` 内置数据集：`PoliticalDemocracy`

### 4.3 推荐模型

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

### 4.4 推荐参考输出

根据 `lavaan` 官方 SEM 教程，建议至少记录：

1. `N = 75`
2. `df = 35`
3. `chi-square ≈ 38.125`
4. 关键路径：
   - `dem60 ~ ind60 ≈ 1.483`
   - `dem65 ~ ind60 ≈ 0.572`
   - `dem65 ~ dem60 ≈ 0.837`
5. 各 latent loading 的参考值
6. 各 residual covariance 的参考值

### 4.5 当前项目中的建议测试步骤

1. 准备 `PoliticalDemocracy` 数据副本；
2. 用与 `lavaan` 一致的语法跑当前项目模型；
3. 提取：
   - measurement loadings
   - regressions
   - residual covariances
   - variances
   - `SE`
   - fit indices
4. 与参考值做分层比较。

### 4.6 建议断言

#### Level A：结构断言

1. 模型收敛；
2. `parameter_inference` 存在；
3. `fit_status` 为稳定状态；
4. 关键路径全部出现在结果对象中。

#### Level B：方向与数量级断言

1. `dem60 ~ ind60` 为正；
2. `dem65 ~ ind60` 为正；
3. `dem65 ~ dem60` 为正且应大于 `dem65 ~ ind60`；
4. residual covariance 至少方向与量级接近参考结果。

#### Level C：容差断言

建议第一批采用：

1. latent loading：`abs_tol = 0.15`
2. structural regression：`abs_tol = 0.10 ~ 0.15`
3. residual covariance：`abs_tol = 0.15 ~ 0.25`
4. fit indices：`abs_tol = 0.03 ~ 0.05`

### 4.7 建议测试文件

- `tests/test_sem_benchmark_political_democracy.py`

### 4.8 建议配套数据文件

建议新增：

- `tests/data/benchmark_political_democracy_raw.csv`
- `tests/data/benchmark_political_democracy_reference.json`

---

## 5. Benchmark 3：边界 warning / failure benchmark

### 5.1 benchmark 目的

前两个 benchmark 检查“正常模型”。

这一组 benchmark 检查：

1. 当前项目在边界场景下是否仍然可解释；
2. warning / status 是否稳定；
3. 与成熟实现的失败语义是否大体一致。

### 5.2 推荐场景

建议先从下面 3 类场景中选 1-2 个落地：

#### Case A：自由度过低

目标：

1. 构造 `df = 0` 或接近 0 的模型；
2. 验证 fit indices 是否转为 `partial` 或 `failed`；
3. warning 是否清楚。

#### Case B：近奇异/非正定问题

目标：

1. 构造强共线或近奇异协方差；
2. 验证推断层是否出现 `partial` 或 `failed`；
3. fit indices 是否给出可解释 warning。

#### Case C：参数边界/识别性脆弱

目标：

1. 构造接近识别性问题的模型；
2. 验证 measurement / structural warning 与 inference warning 是否一致。

### 5.3 目标包与参考对象

建议顺序：

1. `lavaan`（主）
2. `AMOS` / `Mplus`（人工语义参考）

这里不要求逐字匹配 warning，而是要求：

1. 失败类别大体一致；
2. 不会把明显失败静默伪装成正常结果；
3. `ok / partial / failed` 语义稳定。

### 5.4 建议断言

1. 必须有 warning 或明确的失败状态；
2. 不允许在关键结果无意义时静默输出“看似正常”的完整结果；
3. `summary()` / `to_markdown()` 中必须展示相应状态。

### 5.5 建议测试文件

- `tests/test_sem_benchmark_boundary_cases.py`

---

## 6. benchmark 数据获取策略

建议采用下面顺序，而不是运行时在线下载。

### 6.1 第一原则：测试仓库内固化副本

对于正式 benchmark，建议把数据和参考结果固化到仓库中：

1. `tests/data/*.csv`
2. `tests/data/*_reference.json`

原因：

1. 保证离线可运行；
2. 保证版本固定；
3. 保证回归稳定。

### 6.2 第二原则：文档中记录上游来源

每个 benchmark 都应在 `reference.json` 或文档注释中记录：

1. 上游来源；
2. 参考包版本；
3. 提取日期；
4. 参考语法；
5. 是否做过参数对齐或标准化处理。
6. 是否能确认 public domain；如果不能，应明确写为“公开可获取，但未确认 public domain”。

### 6.3 第三原则：不要在测试中依赖 GUI 软件

`AMOS` / `SPSS` 适合作为：

1. 方法学背景参考；
2. 人工交叉核对参考。

但不建议把它们设为自动测试的主依赖。

自动 benchmark 的首选还是：

1. `lavaan`
2. 固化的参考数据/参考输出

---

## 7. 当前项目的目标包目标（建议直接写给后续 agent）

后续如果继续推进 SEM benchmark，建议默认目标如下：

### 第一层目标

1. `lavaan`：默认首要 benchmark 标准
2. `Mplus`：后续高阶 SEM / ESEM 研究级参考

### 第二层目标

1. `semopy`：Python 生态横向参考
2. `psych`：EFA / bridge 测量层参考

### 第三层参考

1. `AMOS`：人工工作流和结果核对参考
2. `SPSS`：传统研究流程与 EFA 报表风格参考

---

## 8. benchmark 落地前的最小准备清单

在真正开始写 benchmark 测试前，建议先完成下面动作：

1. 确定是否在仓库中直接保存 benchmark 数据副本；
2. 确定参考值是从 `lavaan` 输出手工整理，还是用脚本离线生成；
3. 确定参数对齐规则（因子顺序、符号翻转、marker 固定）；
4. 确定第一批容差；
5. 确定测试文件命名与数据目录命名。

---

## 9. 建议的第一批实际交付物

如果下一步直接开始编码，建议第一批只交付：

1. `tests/data/benchmark_hs1939.csv`
   （当前已落地为 `tests/data/benchmark_hs1939_raw.csv`）
2. `tests/data/benchmark_hs1939_reference.json`
3. `tests/test_sem_benchmark_hs1939.py`
4. `tests/data/benchmark_political_democracy_raw.csv`
5. `tests/data/benchmark_political_democracy_reference.json`
6. `tests/test_sem_benchmark_political_democracy.py`

其中当前实际完成状态是：

1. `HolzingerSwineford1939`：原始 CSV + provenance JSON + 首版自动 benchmark 测试已完成；
2. `PoliticalDemocracy`：原始 CSV + provenance JSON + 首版自动 benchmark 测试已完成；
3. `Boundary cases`：首版 warning / failure benchmark 已完成；
4. 三组 benchmark 目前都仍处于 **prototype benchmark** 阶段：
   - 已覆盖 Level A / Level B；
   - 只有部分场景进入 Level C 的直接数值对照；
   - 更严格的参数级对照与 residual covariance 对照仍待下一步补强。

边界 benchmark 可以放到第二批。

---

## 10. 一句话结论

如果当前要真正把 `SEM` 从“原型闭环”推进到“可信基线”，那么最值得优先落地的两组 benchmark 是：

1. `HolzingerSwineford1939`
2. `PoliticalDemocracy`

并且默认以：

1. `lavaan` 为首要开放 benchmark 标准；
2. `Mplus` 为后续高阶研究级参考。
