# 共享预处理模块抽取实施与落地文档（ZH）

本文档用于说明如何把原先挂在 `psysem.efa` 下的输入预处理能力，整理成一个**可被 EFA / ESEM / Network Analysis 共用**的独立模块；同时记录这项抽取在当前代码基线上的实际落地状态。

当前日期：2026-03-14  
适用分支：`main`
当前代码基线：`a17c1ac refactor(efa): unify shared preprocessing across workflows`

---

## 0. 当前实现状态（2026-03-14）

截至当前分支，这项抽取已经不是纯计划，而是已经完成了两阶段落地：

1. **第一阶段已完成**：
   - 新增 `src/psysem/preprocessing/`
   - `build_association_matrix(...)` 已可独立运行
   - `efa.input_matrix` 已降成 wrapper
   - `fit_efa(...)` 已通过共享层进入输入矩阵准备
2. **第二阶段已完成**：
   - `efa/diagnostics.py` 已改为走共享关联矩阵
   - `efa/n_factors.py` 已改为走共享关联矩阵
   - `efa/workflow.py` 已能把 preprocessing 配置统一透传到 diagnostics / selection / fit
   - `esem/workflow.py` 的 block-level EFA bridge 已能透传 `variable_types`，并根据 block 类型自动选择 `pearson / spearman / polychoric`
3. **当前仍未完成的边界**：
   - 还没有开始 `network/` 模块本体
   - 还没有把 ordinal 路线推进到完整 SEM `WLSMV`
   - `polychoric` 的数值稳健性、mixed-type 扩展与 network 结果对象仍是后续重点

因此，本文档后续内容既保留“为什么要这样抽”和“按什么结构抽”，也补充“现在已经做到哪里了”。

---

## 1. 本文档要解决什么问题

当前仓库已经有一套能工作的输入预处理逻辑：

1. 缺失处理：`pairwise`、`dropna`
2. 相关矩阵类型：`pearson`、`spearman`、`polychoric`
3. 变量类型：`continuous`、`ordinal`
4. 矩阵稳定化：对称化、对角线修复、近正定修正
5. warning / recommendation：ordinal-like 提示、missing 提示、polychoric 提示

这些能力现在主要集中在：

1. `src/psysem/efa/input_matrix.py`
2. `src/psysem/efa/fit.py`

问题在于，这些逻辑的**职责已经不是 EFA 专属**了。

网络分析第一阶段同样需要：

1. 对题项列做选择与校验
2. 处理缺失
3. 构造 Pearson / Spearman / Polychoric 关联矩阵
4. 获得 pairwise 样本量与 warning
5. 输出稳定、可逆、可追踪的输入矩阵

如果继续把这些逻辑留在 `psysem.efa` 下面，后面会出现三个直接问题：

1. `network` 需要重复实现一套相同逻辑；
2. `ESEM` 的 ordinal block / `efa_seeded` 也会再次复制相同逻辑；
3. 预处理层的 warning、metadata、变量类型推断会分叉，最终难以维护。

因此，这一轮工作的核心不是“再加一个功能”，而是**把已经成型的通用输入层抽出来，形成共享预处理模块**。

---

## 2. 当前基线

截至当前代码基线，仓库已经具备以下事实：

### 2.1 已经落地的能力

1. 共享包 `src/psysem/preprocessing/` 已存在，包含：
   - `contracts.py`
   - `association.py`
   - `variable_types.py`
   - `polychoric.py`
   - `stabilization.py`
2. `fit_efa()` 不再直接手写相关矩阵逻辑，而是通过 `build_efa_input_matrix(...)` 进入统一输入层。
3. `EFAConfig` 已经显式支持：
   - `missing_strategy`
   - `correlation_method`
   - `variable_types`
4. `EFADiagnosticsConfig`、`FactorSelectionConfig`、`EFAWorkflowConfig` 已支持共享 preprocessing 透传配置。
5. `ESEMWorkflowConfig` 已支持：
   - `efa_missing_strategy`
   - `efa_correlation_method`
6. `polychoric` 已经有第一版实现，并已接入：
   - `fit_efa(...)`
   - `run_efa_diagnostics(...)`
   - `suggest_n_factors(...)`
   - `run_efa_workflow(...)`
   - `run_esem_workflow(...)` 的 ordinal block bridge
7. `tests/test_efa_input_matrix.py` 已经覆盖：
   - 默认 Pearson 路径
   - `dropna`
   - `pairwise`
   - `spearman`
   - `polychoric`
   - ordinal recommendation
   - 稳定化行为

### 2.2 当前剩余结构问题

这轮抽取已经完成，但还剩下几个明确边界没有推进：

1. `build_efa_input_matrix(...)` 仍保留私有 `_EFAInputMatrix` 兼容对象
2. 共享层已经可复用，但 `network/` 还没有正式起包
3. preprocessing recommendation / warning 已统一，但最终 reporting 还没有单独整理为 network 友好输出
4. ordinal 相关矩阵已经进入 EFA / ESEM bridge，但完整 SEM ordinal 估计链路仍未接上

### 2.3 当前最关键的判断

当前问题已经从“要不要抽共享预处理层”，转成了：

> 共享预处理层已经落地，下一步应直接基于它推进 `network`，而不是再在 `efa` 或 `esem` 下复制一套输入逻辑。

---

## 3. 这次到底要做什么

本次文档对应的实现目标只有一个：

> 把现有 `psysem.efa.input_matrix` 中的通用输入预处理能力，抽成一个共享模块，供 `EFA / ESEM / Network Analysis` 共用。

更具体地说，要完成以下五件事：

1. 把“输入矩阵准备”从 `efa` 语义中剥离出来；
2. 定义共享契约，而不是继续使用 `_EFAInputMatrix` 私有结果；
3. 让 EFA 通过轻量 wrapper 复用共享模块，而不是直接持有实现；
4. 为后续 `network` 模块提供稳定入口；
5. 增加更适合 network 的 metadata 输出，而不仅是一个相关矩阵。

---

## 4. 为什么必须这样做

### 4.1 为了避免复制实现

如果网络分析直接从零开始写：

1. `pairwise` missing
2. `polychoric`
3. ordinal 推断
4. 稳定化
5. warning

那么仓库里会同时存在两套几乎相同的逻辑。  
这不仅浪费时间，还会导致：

1. 相同数据在 EFA 与 Network 下得到不同 warning；
2. 相同 `polychoric` 输入在两个模块里有不同边界行为；
3. 修 bug 时必须双改。

### 4.2 为了让职责边界清楚

更合理的职责划分应该是：

1. 共享预处理层负责：准备关联矩阵
2. EFA 层负责：提取、旋转、解释
3. ESEM 层负责：候选生成、judge、selector
4. Network 层负责：precision / partial correlation / sparsification / centrality

现在 `polychoric` 和 missing handling 还挂在 EFA 下面，会模糊这个边界。

### 4.3 为了给网络分析打可复用地基

第一版网络分析最需要的不是绘图，而是：

1. 稳定的关联矩阵
2. 每个变量对的有效样本量
3. 变量类型信息
4. 稳定化信息
5. 可解释 warning

这些内容本质上就是“共享预处理层”的产物。

### 4.4 为了让后续 ordinal 路线更一致

当前 `polychoric` 已经开始落地。  
如果不马上把它从 EFA 专属实现转成共享模块，后续会出现：

1. EFA 一套 ordinal 输入路径
2. ESEM 一套 ordinal block 输入路径
3. Network 一套 ordinal 项目网络输入路径

这条路是错误的，应该在现在就收口。

---

## 5. 这次不做什么

为了避免范围失控，这一轮**不做**以下内容：

1. 不直接实现完整网络分析 workflow
2. 不直接实现 `EBICglasso`
3. 不直接实现 network 可视化
4. 不直接实现 `WLSMV` 或完整 SEM ordinal 估计
5. 不在这一轮引入黑箱 `auto` 自动切换策略

也就是说，这一轮的目标是：

> 只把“共享预处理基础设施”整理出来，不把 network 本体一起做掉。

---

## 5.1 对目前流程的影响与修改范围

这一节专门回答两个问题：

1. 抽共享预处理层后，当前运行流程会受什么影响？
2. 哪些模块必须改，哪些模块建议第二阶段再改？

为了控制风险，这次实现实际按 **第一阶段必改** 和 **第二阶段一致性改造** 两层落地；两层现在都已经完成。

### 5.1.1 对当前运行流程的实际影响

#### 流程 A：直接调用 `fit_efa(...)`

当前流程：

```text
data -> psysem.efa.input_matrix -> extraction -> rotation -> interpretation
```

抽取后建议变为：

```text
data -> psysem.preprocessing.build_association_matrix
     -> psysem.efa.input_matrix(wrapper)
     -> extraction -> rotation -> interpretation
```

这条流程的设计目标是：

1. 对外 API 不变；
2. 对用户参数不强行改名；
3. 行为尽量保持兼容；
4. 内部真实实现迁到共享层。

因此，对 `fit_efa(...)` 用户的影响应当是：

1. **公开调用方式基本不变**
2. warning 文案可能会更统一
3. 后续 result 中若补充 metadata，属于增强而不是破坏

#### 流程 B：直接调用 `build_efa_input_matrix(...)`

当前流程：

```text
user/test -> psysem.efa.input_matrix.build_efa_input_matrix(...)
```

抽取后建议保留：

```text
user/test -> psysem.efa.input_matrix.build_efa_input_matrix(...)
```

但内部变成 wrapper：

```text
psysem.efa.input_matrix -> psysem.preprocessing.build_association_matrix(...)
```

因此，对现有测试与可能的内部调用来说：

1. import 路径尽量不变
2. 行为尽量兼容
3. 但其角色从“实现层”变成“兼容层”

这意味着：

1. `tests/test_efa_input_matrix.py` 还会存在
2. 但它的定位会从“核心实现测试”变成“EFA wrapper 兼容测试”

#### 流程 C：`run_efa_workflow(...)`

当前流程：

```text
run_efa_diagnostics(...)
suggest_n_factors(...)
fit_efa(...)
```

这里有一个关键事实：

1. `fit_efa(...)` 已经走共享层 wrapper
2. `run_efa_diagnostics(...)` 与 `suggest_n_factors(...)` 也已经统一复用共享关联矩阵

当前实现状态是：

1. `run_efa_workflow(...)` 已进入 **第二阶段一致性版**
2. diagnostics / factor selection / fit 现在共享同一套 preprocessing 语义
3. workflow 会先解析统一的 `missing_strategy / correlation_method / variable_types`
4. 如果 diagnostics 与 selection 子配置彼此冲突，workflow 会直接拒绝运行

#### 流程 D：`run_esem_workflow(...)` 的 block-level EFA bridge

当前 `ESEM` workflow 会在 block 级调用 `fit_efa(...)`，见：

1. `src/psysem/esem/workflow.py`

当前实现状态是：

1. `ESEM` block-level EFA bridge 已继续保留 `fit_efa(...)` 兼容接口
2. 同时已经把 `spec.variable_types` 派生成 block 级 `EFAConfig.variable_types`
3. 已支持 `ESEMWorkflowConfig.efa_missing_strategy`
4. 已支持 `ESEMWorkflowConfig.efa_correlation_method`
5. 如果 `efa_correlation_method` 未显式指定：
   - 全 ordinal block 默认走 `polychoric`
   - mixed block 默认走 `spearman`
   - 全 continuous block 默认走 `pearson`

这意味着 `run_esem_workflow(...)` 现在已经真正吃到了共享 preprocessing 的 ordinal 路径，而不只是“间接受影响”。

因此，ESEM block-level EFA bridge 的第二阶段目标也已经在当前分支落地。

#### 流程 E：未来 `network` 模块

这是这次抽取的主要受益者。

目标流程应当是：

```text
data -> psysem.preprocessing.build_association_matrix(...)
     -> network estimation
```

也就是说：

1. network 不应依赖 `psysem.efa`
2. network 应直接吃共享层输出

---

### 5.1.2 第一阶段必须修改的模块

下面这些属于**必须改**。

#### 1. `src/psysem/preprocessing/`

这是新增模块，不存在可选空间。

需要新增：

1. `src/psysem/preprocessing/__init__.py`
2. `src/psysem/preprocessing/contracts.py`
3. `src/psysem/preprocessing/association.py`
4. `src/psysem/preprocessing/variable_types.py`
5. `src/psysem/preprocessing/polychoric.py`
6. `src/psysem/preprocessing/stabilization.py`

修改说明：

1. 这是新的共享实现层
2. 未来 `EFA / ESEM / Network` 都应依赖这里

#### 2. `src/psysem/efa/input_matrix.py`

必须改，但改法是“降级成 wrapper”，不是删除。

修改说明：

1. 保留原入口，避免 EFA 内部和测试大范围破裂
2. 内部转调 `build_association_matrix(...)`
3. 负责把共享结果适配成 EFA 当前需要的最小返回形状

#### 3. `src/psysem/efa/fit.py`

必须改，但主要是轻量适配。

修改说明：

1. 继续保留 `EFAConfig`
2. 继续保留 `fit_efa(...)`
3. 调整 import 与输入校验职责边界
4. 避免把与共享层重复的校验继续留在 `fit.py`

这里最需要注意的是：

1. 哪些校验仍属于 EFA 自身
2. 哪些校验已经属于共享预处理层

如果不收边界，代码会重复。

#### 4. 测试层

至少要改：

1. `tests/test_efa_input_matrix.py`

至少要新增：

1. `tests/test_preprocessing_association_matrix.py`
2. `tests/test_preprocessing_polychoric.py`

修改说明：

1. 共享层要有自己的测试
2. `efa.input_matrix` 测试要从实现测试转成兼容测试

#### 5. 导出层

如果共享预处理层要作为正式公共 API 使用，则必须改：

1. `src/psysem/__init__.py`
2. `src/psysem/efa/__init__.py`（若保留兼容导出策略）

修改说明：

1. 导出 `AssociationMatrixConfig`
2. 导出 `AssociationMatrixResult`
3. 导出 `build_association_matrix`

如果第一阶段暂时不想把它暴露到顶层，也至少要在 `psysem.preprocessing` 子包内导出。

#### 6. 文档层

必须改：

1. `docs/index.md`
2. `mkdocs.yml`
3. 当前这篇文档

建议同步改：

1. `docs/efa-method-expansion-roadmap.zh-CN.md`

修改说明：

1. 把“输入预处理层”从 EFA 内部能力升级为共享基础设施
2. 补清楚 wrapper 与共享层的关系

---

### 5.1.3 第二阶段建议修改的模块

下面这些不是第一阶段必须动，但如果想让整个流程真正一致，建议第二阶段处理。

#### 1. `src/psysem/efa/diagnostics.py`

当前它还在独立构造相关矩阵。

修改说明：

1. 可逐步改为复用共享层
2. 这样 KMO / Bartlett 与 EFA 拟合可使用同一矩阵准备语义

风险说明：

1. 一旦接入共享层，`EFADiagnosticsConfig` 可能需要补更多配置字段
2. 这会扩大 API 变化范围

#### 2. `src/psysem/efa/n_factors.py`

当前它通过 `build_efa_correlation_matrix(...)` 自己拿相关矩阵。

修改说明：

1. 第二阶段建议改成复用共享层
2. 这样 PA/MAP/Scree/Kaiser 与 EFA 拟合可共享同一预处理策略

风险说明：

1. `FactorSelectionConfig` 当前没有 `missing_strategy/correlation_method/variable_types`
2. 一旦统一，会涉及配置结构升级

#### 3. `src/psysem/efa/contracts.py`

如果 diagnostics / n_factors / workflow 要统一到共享层，这个文件会受影响。

修改说明：

可能需要给下面这些 config 增加共享预处理相关配置：

1. `EFADiagnosticsConfig`
2. `FactorSelectionConfig`
3. `EFAWorkflowConfig`

这里建议优先考虑两种方案之一：

1. 直接在每个 config 上增加字段
2. 新增一个嵌套的 `preprocessing` 配置对象

第二种长期更干净。

#### 4. `src/psysem/efa/workflow.py`

如果 workflow 想真正统一矩阵语义，这里第二阶段必改。

修改说明：

1. `run_efa_diagnostics(...)`
2. `suggest_n_factors(...)`
3. `fit_efa(...)`

三段最好最终共享一套 preprocessing 配置来源。

#### 5. `src/psysem/esem/workflow.py`

当前它已经不只是 block 级调用 `fit_efa(...)`，而是会显式把 `ESEMSpec.variable_types` 和 block 级相关矩阵策略透传进去。

修改说明：

当前已经完成：

1. block 级从 `spec.variable_types` 派生 `EFAConfig.variable_types`
2. 根据变量类型自动选择默认 `correlation_method`
3. 在 ordinal block 场景下允许 `polychoric` 路径

因此，ESEM 现在已经真正复用了共享预处理层，而不是只靠 `fit_efa(...)` 的间接兼容。

---

### 5.1.4 当前基本不受影响的模块

第一阶段基本可以不动：

1. `src/psysem/core.py`
2. `src/psysem/measurement/`
3. `src/psysem/structural/`
4. `src/psysem/estimation/`
5. `src/psysem/fit_indices.py`
6. `src/psysem/inference/`

原因很简单：

1. 它们不直接消费 `efa.input_matrix`
2. 这轮工作只抽“关联矩阵准备层”，不碰 SEM 数值主路径

---

### 5.1.5 建议采用的改动策略

如果目标是“先安全落地，再逐步放大复用范围”，当前分支已经按下面的方式落地：

#### 第一阶段：最小影响抽取（已完成）

已完成改动：

1. `preprocessing/` 新包
2. `efa/input_matrix.py`
3. `efa/fit.py`
4. 新测试
5. 导出与文档

第一阶段当时先不改：

1. `efa/diagnostics.py`
2. `efa/n_factors.py`
3. `efa/workflow.py`
4. `esem/workflow.py`

第一阶段的优点是：

1. 风险最小
2. 最快能服务 network
3. 不会一下子把整个 EFA workflow 配置结构改复杂

#### 第二阶段：流程一致性升级（已完成）

已完成改动：

1. `efa/diagnostics.py`
2. `efa/n_factors.py`
3. `efa/contracts.py`
4. `efa/workflow.py`
5. `esem/workflow.py`

第二阶段已经达到的目标：

1. 让 diagnostics / factor selection / fit / ESEM bridge 全部共享同一套预处理语义

---

## 6. 推荐的目标结构

建议新增：

```text
src/psysem/
  preprocessing/
    __init__.py
    contracts.py
    association.py
    variable_types.py
    polychoric.py
    stabilization.py
```

并保留：

```text
src/psysem/efa/
  input_matrix.py   # 过渡期 wrapper
```

### 为什么是 `preprocessing/`

因为它表达的是“共享数据进入建模前的准备阶段”，而不是 EFA 专属逻辑。  
这比下面这些名字更合适：

1. 继续放在 `efa/` 下：边界不清
2. 放在 `data/` 下：会和 `spec/data validation` 混在一起
3. 叫 `network/`：会让 EFA 反过来依赖 network，方向错误

### 为什么核心入口建议叫 `build_association_matrix(...)`

因为当前这层真正产出的不是“EFA 输入”，而是：

1. 一个矩阵
2. 与该矩阵相关的 metadata
3. warning

它最准确的语义是“关联矩阵准备”，而不是某个具体方法专用输入。

---

## 7. 共享模块应该承担哪些职责

共享预处理模块应当只负责下面这些内容：

### 7.1 数据列选择与基础校验

1. 读取 `items`
2. 检查列是否存在
3. 处理非数值输入到可分析形式
4. 识别空列、常量列、无效列

### 7.2 变量类型解析

1. 读取显式 `variable_types`
2. 对未声明变量做轻量推断
3. 输出 `resolved_variable_types`
4. 对不合法声明报错

### 7.3 缺失处理

1. `dropna`
2. `pairwise`
3. 记录有效样本量
4. 输出 dropped rows / pairwise count 相关 warning

### 7.4 关联矩阵构造

第一阶段至少支持：

1. `pearson`
2. `spearman`
3. `polychoric`

### 7.5 数值稳定化

1. 对称化
2. 对角线修复
3. 近正定修正
4. 输出 stabilization metadata

### 7.6 recommendation 与 warning

1. ordinal-like recommendation
2. non-ordinal + polychoric 错误
3. 样本量/缺失策略 warning
4. 稳定化 warning

### 7.7 metadata 输出

这是本轮抽取后必须增强的部分。

对于 network 而言，结果不应只包含矩阵，还应至少包含：

1. `pairwise_n`
2. `n_complete_rows`
3. `dropped_rows`
4. `resolved_variable_types`
5. `stabilization_applied`
6. `correlation_method`
7. `missing_strategy`
8. `warnings`

---

## 8. 哪些职责不应该放进共享预处理层

共享预处理层**不应**负责：

1. EFA extraction
2. EFA rotation
3. EFA candidate scoring
4. SEM parameter estimation
5. network precision estimation
6. network centrality / community

否则它会再次变成一个“什么都做的总控层”。

---

## 9. 共享契约建议

建议新增以下 dataclass。

### 9.1 `AssociationMatrixConfig`

建议字段：

1. `items: tuple[str, ...]`
2. `missing_strategy: str = "pairwise"`
3. `correlation_method: str = "pearson"`
4. `variable_types: dict[str, str] | None = None`
5. `stabilize: bool = True`
6. `min_eigenvalue: float = 1e-8`
7. `include_pairwise_counts: bool = True`

### 9.2 `AssociationMatrixResult`

建议字段：

1. `matrix: pd.DataFrame`
2. `item_names: tuple[str, ...]`
3. `correlation_method: str`
4. `missing_strategy: str`
5. `resolved_variable_types: dict[str, str]`
6. `pairwise_n: pd.DataFrame | None`
7. `n_complete_rows: int | None`
8. `dropped_rows: int`
9. `stabilization_applied: bool`
10. `warnings: tuple[str, ...]`

### 为什么要加这些字段

因为 network 分析比 EFA 更依赖这些“矩阵外的信息”。  
如果只返回一个 `numpy.ndarray`，后续很难解释：

1. 这个矩阵是 Pearson 还是 Polychoric
2. 是 pairwise 还是 dropna
3. 每对变量有多少样本
4. 是否被强行修正成近正定

---

## 10. 具体怎么做

下面给出建议的最小风险实施顺序。

## Step 0：冻结当前行为

### 目标

在抽模块之前，先把当前 `efa.input_matrix` 的行为通过测试固定下来，避免重构时悄悄改语义。

### 具体动作

1. 复查并补齐 `tests/test_efa_input_matrix.py`
2. 对以下路径加基线断言：
   - 默认 Pearson
   - `dropna`
   - `pairwise`
   - `spearman`
   - `polychoric`
   - ordinal recommendation
   - stabilization

### 为什么要先做

因为这次是“模块抽取”，不是“重新设计全部算法”。  
先锁住行为，后续重构才有边界。

### 完成标准

1. 当前 EFA 输入层测试能独立通过
2. 已明确哪些 warning 是现有兼容行为

---

## Step 1：先定义共享契约

### 目标

先固定共享模块输入输出结构，再移动实现。

### 具体动作

新增：

1. `src/psysem/preprocessing/contracts.py`
2. `AssociationMatrixConfig`
3. `AssociationMatrixResult`

### 为什么要这样做

如果先搬代码，再补契约，很容易把当前 `_EFAInputMatrix` 的私有形状原样扩散到新模块里。  
正确顺序应该是：先定义共享语义，再让实现服从新语义。

### 完成标准

1. 新契约能完整表达当前能力
2. 已覆盖未来 network 所需 metadata

---

## Step 2：把变量类型解析抽出来

### 目标

让 `variable_types` 的解析、推断和 recommendation 不再绑在 EFA 文件里。

### 具体动作

新增：

1. `src/psysem/preprocessing/variable_types.py`

建议迁移内容：

1. `_resolve_variable_types(...)`
2. `_infer_variable_type(...)`
3. ordinal recommendation 逻辑

### 为什么要这样做

变量类型推断不是 EFA 特有逻辑。  
network、ESEM、未来 mixed-type 路径都会用到。

### 完成标准

1. 输入变量类型解析不再依赖 `psysem.efa`
2. warning 文案不再写成 EFA 特定表述

---

## Step 3：把矩阵构造总控抽出来

### 目标

提供真正的共享入口：

```python
build_association_matrix(data, config)
```

### 具体动作

新增：

1. `src/psysem/preprocessing/association.py`

迁移内容：

1. missing strategy 调度
2. Pearson / Spearman / Polychoric 调度
3. pairwise counts 记录
4. warning 聚合

### 为什么要这样做

这是未来 EFA / network 共同依赖的主入口。  
如果这一层继续留在 `efa.input_matrix`，那抽模块这件事就没有真正完成。

### 完成标准

1. 新入口可在不依赖 EFA 的情况下独立工作
2. 能返回 `AssociationMatrixResult`

---

## Step 4：把 `polychoric` 单独拆文件

### 目标

让 ordinal 专用算法与总控编排解耦。

### 具体动作

新增：

1. `src/psysem/preprocessing/polychoric.py`

建议迁移：

1. `_compute_polychoric_matrix(...)`
2. `_estimate_polychoric_correlation(...)`
3. `_ordinal_codes_and_thresholds(...)`
4. `_polychoric_cell_probabilities(...)`
5. 双变量正态矩形概率计算

### 为什么要这样做

`polychoric` 是数值复杂度最高、最容易继续增长的部分。  
如果它还混在总控文件里，后续 mixed-type、tetrachoric、metadata 增强会把文件迅速做大。

### 完成标准

1. `polychoric` 算法实现从总控层解耦
2. 后续可单独补数值稳健性与测试

---

## Step 5：把稳定化逻辑单独拆文件

### 目标

让“矩阵稳定化”成为一个共享能力，而不是 EFA 私有后处理。

### 具体动作

新增：

1. `src/psysem/preprocessing/stabilization.py`

建议迁移：

1. 相关矩阵对称化
2. 对角线修复
3. 特征值裁剪
4. stabilization metadata 输出

### 为什么要这样做

网络分析对矩阵是否可逆、是否被修正过非常敏感。  
稳定化逻辑需要可见、可追踪，而不是藏在 EFA 文件底部。

### 完成标准

1. 稳定化逻辑可单独测试
2. `AssociationMatrixResult` 能明确标记是否发生修正

---

## Step 6：把 `efa.input_matrix` 降成 wrapper

### 目标

不破坏当前 EFA API，同时完成共享模块迁移。

### 具体动作

保留：

1. `src/psysem/efa/input_matrix.py`

但把它改成轻量适配层：

1. 把 `EFAConfig` 映射成 `AssociationMatrixConfig`
2. 调用 `build_association_matrix(...)`
3. 把结果转成 EFA 当前所需的最小形状

### 为什么要这样做

这样可以：

1. 保持 `fit_efa()` 外部行为稳定
2. 降低大范围改动风险
3. 先完成共享层抽取，再决定是否进一步统一公开 API

### 完成标准

1. `fit_efa()` 外部调用方式不变
2. EFA 继续通过原入口运行
3. 但真实实现已迁到共享层

---

## Step 7：为 network 预留直接调用入口

### 目标

让网络分析后续不必再依赖 `psysem.efa`。

### 具体动作

建议新增可直接导出的 API：

```python
from psysem.preprocessing import (
    AssociationMatrixConfig,
    AssociationMatrixResult,
    build_association_matrix,
)
```

### 为什么要这样做

network 模块后续应该直接写成：

```python
prepared = build_association_matrix(data, config)
```

而不是：

```python
from psysem.efa.input_matrix import ...
```

后者会造成错误依赖方向。

### 完成标准

1. 共享预处理层可以脱离 EFA 单独调用
2. network 新模块设计不再需要 import `psysem.efa`

---

## Step 8：补文档、测试与迁移说明

### 目标

让这次抽取不是“只有代码完成”，而是工程上真正完成。

### 具体动作

文档：

1. 更新 `docs/index.md`
2. 在 `mkdocs.yml` 中加入导航
3. 在 EFA 路线图中补链接说明

测试：

1. 新增 `tests/test_preprocessing_association_matrix.py`
2. 保留 `tests/test_efa_input_matrix.py`，但把重点改为 wrapper 兼容测试

### 为什么要这样做

如果测试文件仍全绑在 `efa` 名下，就说明抽取仍未真正完成。  
共享层需要有自己的测试和自己的文档入口。

### 完成标准

1. 文档能解释“为什么现在有 `preprocessing` 包”
2. 测试结构能体现共享层已独立存在

---

## 11. 为什么不建议直接让 network 复用 `psysem.efa.input_matrix`

表面上看，network 直接 import 当前 `efa.input_matrix` 好像更快。  
但这会带来四个问题：

1. **依赖方向错误**：network 反向依赖 EFA
2. **命名错误**：共享逻辑继续挂在 EFA 名下
3. **契约错误**：当前返回的是 `_EFAInputMatrix` 私有对象
4. **扩展成本高**：只要加 network-specific metadata，就会污染 EFA 私有模块

因此，哪怕当前代码已经可运行，也不应直接“借用现有文件名”作为长期方案。

---

## 12. 推荐的 API 草案

共享层建议 API：

```python
from psysem.preprocessing import AssociationMatrixConfig, build_association_matrix

prepared = build_association_matrix(
    data,
    AssociationMatrixConfig(
        items=("i1", "i2", "i3", "i4"),
        missing_strategy="pairwise",
        correlation_method="polychoric",
        variable_types={
            "i1": "ordinal",
            "i2": "ordinal",
            "i3": "ordinal",
            "i4": "ordinal",
        },
    ),
)

print(prepared.matrix)
print(prepared.pairwise_n)
print(prepared.resolved_variable_types)
print(prepared.warnings)
```

EFA 侧建议保持：

```python
result = fit_efa(
    data,
    EFAConfig(
        items=("i1", "i2", "i3", "i4"),
        n_factors=2,
        missing_strategy="pairwise",
        correlation_method="polychoric",
        variable_types={
            "i1": "ordinal",
            "i2": "ordinal",
            "i3": "ordinal",
            "i4": "ordinal",
        },
    ),
)
```

也就是说：

1. EFA 用户接口先不变
2. 共享层给 EFA 和 network 共用

---

## 13. 测试矩阵建议

建议新增并重构为下面的测试分层：

```text
tests/
  test_preprocessing_association_matrix.py
  test_preprocessing_polychoric.py
  test_efa_input_matrix.py
```

### 13.1 共享层必须覆盖

1. 默认 Pearson 路径
2. `dropna`
3. `pairwise`
4. `spearman`
5. `polychoric`
6. non-ordinal + polychoric 报错
7. ordinal recommendation
8. stabilization metadata
9. pairwise counts 输出

### 13.2 EFA wrapper 必须覆盖

1. `build_efa_input_matrix(...)` 仍能工作
2. wrapper 输出矩阵与共享层一致
3. warning 不丢失

### 13.3 未来 network 集成测试应覆盖

1. 共享层返回的矩阵可被 network 估计器直接消费
2. pairwise counts / warnings 可进入 network 结果对象

---

## 14. 风险与缓解

### 风险 1：抽模块时行为悄悄变化

缓解：

1. 先冻结当前测试
2. 先做 wrapper，再做内部迁移

### 风险 2：共享层过度设计

缓解：

1. 第一版只抽当前已经存在的能力
2. 不提前为所有未来 mixed-type 场景做复杂抽象

### 风险 3：EFA 兼容性被破坏

缓解：

1. 保留 `efa.input_matrix.py`
2. 先让它变成适配层
3. 不在这一轮修改 `fit_efa()` 公开签名

### 风险 4：network 需求把共享层拉得过重

缓解：

1. 共享层只负责关联矩阵准备
2. precision / centrality 仍放在未来 `network` 子模块

---

## 15. 实际执行顺序

当前分支实际按下面顺序推进：

1. 冻结现有输入层行为测试
2. 新建 `preprocessing/contracts.py`
3. 新建 `preprocessing/variable_types.py`
4. 新建 `preprocessing/association.py`
5. 新建 `preprocessing/polychoric.py`
6. 新建 `preprocessing/stabilization.py`
7. 把 `efa/input_matrix.py` 改成 wrapper
8. 把共享层测试独立出来
9. 文档与导航同步更新
10. 再统一 `efa/diagnostics.py`、`efa/n_factors.py`、`efa/workflow.py`
11. 再统一 `esem/workflow.py` 的 block-level EFA bridge

这条顺序的优点是：

1. 风险最低
2. 兼容性最好
3. 抽出来后就能立刻服务 network 下一步开发

---

## 16. 完成定义（DoD）

按当前分支实现状态，这一组 DoD 已经满足。

当这项工作完成时，至少应满足：

1. 仓库里存在独立的 `src/psysem/preprocessing/` 包
2. `build_association_matrix(...)` 可独立运行
3. EFA 继续通过原入口工作
4. `polychoric` 不再只存在于 `efa` 私有文件中
5. 共享层测试独立存在
6. 文档已说明为什么要抽、怎么抽、哪些模块复用它

---

## 17. 一句话结论

这次不应再把 network 所需的关联矩阵预处理逻辑继续挂在 `psysem.efa` 下。  
正确做法是：

**把现有 `input_matrix + variable_types + polychoric + stabilization` 抽成共享预处理模块，保留 EFA wrapper 兼容层，再让 network 基于共享层起步。**
