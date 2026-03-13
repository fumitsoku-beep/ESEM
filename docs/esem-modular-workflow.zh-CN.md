# ESEM 模块化判断工作流实施文档（ZH）

本文档给出一个**具体可落地**的 ESEM 演进方案：

- 保留当前项目“按 `block` 自动展开测量模型”的易用入口；
- 借鉴现有 `EFA` 模块的“多方法 + 配置化 + 共识聚合”设计；
- 将 ESEM 拆成 **候选生成（generator）/ 判断（judge）/ 聚合选择（selector）** 三层；
- 先做一个工程上最容易落地、且最符合当前仓库代码结构的版本。

当前日期：2026-03-12  
适用分支：`main`

执行进度（更新）：

1. 已落地最小可跑入口：`run_esem_workflow(data, spec, config)`；
2. 已支持 `block_full` 单候选策略、基础 judge（`convergence/fit_indices/efa_bridge`）与 `best_score` 选择；
3. 其余多候选策略与可插拔注册体系仍按本文档后续 Step 推进。

---

## 1. 本文档要解决的问题

当前 `psysem` 的 ESEM 主思路是：

1. 用户提供 `blocks`；
2. 每个 `block` 自动展开成 `block_f1`, `block_f2`, ...；
3. 每个因子默认连接该 `block` 内全部题项；
4. 再叠加 `structural` 路径，统一进入 `ModelSpec`。[src/psysem/model.py](../src/psysem/model.py#L101-L138)

这种做法的优点是简单、稳定、适合自动化；缺点是判断路径单一，缺少“多种候选结构 + 多种判断规则 + 自动选择”的灵活性。

本文档的目标不是直接改成所有 ESEM 学派，而是先落地一个**模块化版本**，让后续可以逐步接入：

1. `block` 全展开；
2. `target rotation`/目标模板；
3. `EFA` 引导式候选；
4. 更严格或更宽松的筛选规则；
5. 多种聚合与选择策略。

---

## 2. 本次建议采用的“具体方案”

本文建议优先实现的不是纯理论版 ESEM，而是一个**混合式模块化 ESEM 工作流**：

### 2.1 核心思想

从一个 `ESEMSpec` 出发，不再只生成一个模型，而是生成一组候选模型：

1. **Baseline 候选**：当前已有的 `block` 全展开方式；
2. **Simple-structure 候选**：只允许主加载自由，交叉加载固定或默认关闭；
3. **EFA-seeded 候选**：先跑 `EFA` 得到候选主载荷模式，再转成 ESEM/SEM 草图；
4. **Target-pattern 候选**：由用户或规则生成目标矩阵，再形成近似结构。

然后对这些候选做统一评估：

1. 测量可识别性；
2. EFA 风格结构质量（主载荷、交叉载荷、共同度、残差）；
3. SEM 拟合质量（如 `CFI/TLI/RMSEA/SRMR`）；
4. 理论约束一致性；
5. 最终用统一选择器挑选最优候选。

### 2.2 为什么选这个方案

因为它最符合当前仓库已有实现：

1. 已有 `ESEMSpec -> ModelSpec` 的 block 展开入口；[src/psysem/model.py](../src/psysem/model.py#L101-L138)
2. 已有完整 `EFA` 诊断、因子数建议、候选比较工作流；[src/psysem/efa/workflow.py](../src/psysem/efa/workflow.py#L18-L69)
3. 已有 measurement/structural/ML/inference/fit-indices 基础闭环；[src/psysem/measurement/builder.py](../src/psysem/measurement/builder.py#L12-L197) [src/psysem/structural/builder.py](../src/psysem/structural/builder.py#L12-L114) [src/psysem/estimation/ml.py](../src/psysem/estimation/ml.py#L164-L255)
4. 已有 EFA 中的“多方法建议 + 共识聚合”实践经验；[src/psysem/efa/n_factors.py](../src/psysem/efa/n_factors.py#L26-L79)

也就是说，这个方案不是另起炉灶，而是把现有模块重新编排。

---

## 3. 目标状态

完成后，理想使用方式类似：

```python
from psysem import run_esem_workflow, ESEMWorkflowConfig

result = run_esem_workflow(
    data,
    spec,
    ESEMWorkflowConfig(
        generator_strategies=("block_full", "efa_seeded", "target_pattern"),
        enabled_judges=("identification", "cross_loading", "communality", "fit_indices"),
        selector_strategy="weighted_vote",
    ),
)
```

输出至少应包含：

1. 每个候选模型的来源与配置；
2. 每个候选模型的 `SEMResult`；
3. 每个 judge 的打分、告警、是否通过；
4. 候选比较表；
5. 最终最佳候选与选择理由。

---

## 4. 模块化架构

建议引入如下结构：

```text
src/psysem/
  esem/
    contracts.py      # workflow/generator/judge/selector/result 配置与结果契约
    workflow.py       # run_esem_workflow 主编排
    generators.py     # 候选结构生成器注册与默认策略
    judges.py         # 判断器注册与默认策略
    selectors.py      # 候选聚合/选择策略
    adapters.py       # ESEMSpec <-> ModelSpec / EFA 结果桥接
```

这层不替代现有：

- `data`
- `model`
- `measurement`
- `structural`
- `estimation`
- `efa`

而是把它们组织成一个更高层的工作流。

---

## 5. 与前面提到的“其他思路”怎么结合

本文建议的模块化方案，不是排斥其他方法，而是把它们变成**可选策略**。

### 5.1 方法映射表

| 方法思路                        | 在工作流中的位置   | 第一版是否落地   |
| ------------------------------- | ------------------ | ---------------- |
| 当前 block 全展开               | generator          | 是               |
| 两阶段 EFA -> SEM               | generator/adapters | 是               |
| target rotation / target matrix | generator          | 是（先做轻量版） |
| bifactor / hierarchical         | generator          | 否，后续扩展     |
| 稀疏/正则化                     | judge + estimator  | 否               |
| 贝叶斯 ESEM                     | estimator          | 否               |

### 5.2 第一版只做哪三个

建议第一版只做：

1. `block_full`
2. `efa_seeded`
3. `target_pattern`

原因：

1. 覆盖当前仓库已有路线；
2. 覆盖“纯自动 / 数据引导 / 理论引导”三种模式；
3. 开发复杂度可控。

---

## 6. Phase 设计总览

| Phase   | 目标              | 产出                      | 优先级 |
| ------- | ----------------- | ------------------------- | ------ |
| Phase 0 | 契约与目录准备    | `contracts.py` + API 草案 | P0     |
| Phase 1 | 候选生成器        | 3 个默认 generator        | P0     |
| Phase 2 | judge 体系        | 4-6 个默认 judge          | P0     |
| Phase 3 | selector 与比较表 | 多候选自动选择            | P0     |
| Phase 4 | workflow 编排     | `run_esem_workflow(...)`  | P0     |
| Phase 5 | 文档/示例/测试    | 端到端回归防线            | P0     |
| Phase 6 | 高级扩展          | bifactor / 更复杂规则     | P2     |

---

## 7. 详细实施步骤

以下步骤按“能直接开工”的粒度给出。

---

## Step 0：冻结第一版范围

### 目标

避免一开始把所有 ESEM 变体都做进去，先确定第一版边界。

### 实现内容

明确第一版只支持：

1. 单组连续变量主流程；
2. 候选生成器：`block_full`、`efa_seeded`、`target_pattern`；
3. selector：`best_score`、`majority_vote`、`weighted_vote`；
4. judge：`identification`、`cross_loading`、`communality`、`fit_indices`、`factor_balance`；
5. 输出统一比较表。

### 不在第一版做

1. `WLSMV` 专用估计闭环；
2. bifactor；
3. 稀疏惩罚；
4. 贝叶斯推断；
5. 多组不变性。

### 落地文件

1. 本文档
2. [docs/sem-phase-implementation.zh-CN.md](sem-phase-implementation.zh-CN.md)（其中已整合后续执行路线）

### 完成标准

1. 所有人对第一版范围有一致理解；
2. 后续 API 命名围绕该边界设计。

---

## Step 1：定义 `esem` 工作流契约

### 目标

先把 workflow 的输入输出结构固定下来，避免后续反复改名。

### 实现内容

新增以下 dataclass：

1. `ESEMWorkflowConfig`
2. `ESEMCandidateConfig`
3. `ESEMCandidateResult`
4. `ESEMJudgeConfig`
5. `ESEMJudgeResult`
6. `ESEMWorkflowResult`

建议字段：

#### `ESEMWorkflowConfig`

1. `generator_strategies: tuple[str, ...]`
2. `enabled_judges: tuple[str, ...]`
3. `selector_strategy: str`
4. `include_sem_fit: bool = True`
5. `include_efa_bridge: bool = True`
6. `keep_all_candidates: bool = True`
7. `judge_weights: dict[str, float] | None = None`
8. `selector_weights: dict[str, float] | None = None`
9. `fit_config: SEMFitConfig | None = None`

#### `ESEMCandidateResult`

1. `candidate_id`
2. `strategy`
3. `model_spec`
4. `sem_result`
5. `judge_results`
6. `total_score`
7. `warnings`

#### `ESEMWorkflowResult`

1. `candidates`
2. `comparison_table`
3. `best_candidate_id`
4. `best_candidate`
5. `warnings`

### 落地文件

1. `src/psysem/esem/contracts.py`
2. `src/psysem/esem/__init__.py`

### 完成标准

1. `mypy` 可通过；
2. 命名与现有 `efa/contracts.py` 风格一致；
3. 结果对象足以承载后续全部步骤。

---

## Step 2：建立 generator 注册机制

### 目标

让 ESEM 候选结构生成方式可注册、可扩展，复用 EFA 中的注册式风格。[src/psysem/efa/fit.py](../src/psysem/efa/fit.py#L117-L139)

### 实现内容

新增：

1. `register_esem_generator(name, fn)`
2. `get_esem_generator(name)`
3. `list_esem_generators()`

统一函数签名建议：

```python
def generator(data, spec, workflow_config) -> list[ModelSpec]:
    ...
```

每个 generator 至少返回一个 `ModelSpec`，也允许返回多个候选。

### 落地文件

1. `src/psysem/esem/generators.py`
2. `tests/test_esem_generators.py`

### 完成标准

1. generator 可被单独调用；
2. 未注册名称时有明确错误；
3. 默认 generator 自动加载。

---

## Step 3：实现 `block_full` generator

### 目标

把当前已有 block 自动展开逻辑迁移为 workflow 中的基线候选。

### 实现内容

逻辑来源直接复用当前 `model_spec_from_esem_spec(...)`：[src/psysem/model.py](../src/psysem/model.py#L101-L138)

执行步骤：

1. 读取 `ESEMSpec.blocks`；
2. 对每个 block 按 `n_factors` 生成 `block_f1...block_fk`；
3. 对每个潜因子生成 `latent =~ item1 + item2 + ...`；
4. 叠加 `spec.structural`；
5. 输出 1 个 baseline `ModelSpec`。

### 落地文件

1. `src/psysem/esem/generators.py`
2. `tests/test_esem_generators.py`

### 完成标准

1. 在无额外配置时，其行为与当前 `model_spec_from_esem_spec` 一致；
2. 旧路径可平滑迁移。

---

## Step 4：实现 `efa_seeded` generator

### 目标

把“两阶段 EFA -> SEM”的思路变成一个候选生成器。

### 实现内容

对每个 block：

1. 调用现有 `run_efa_workflow(...)` 或 `fit_efa(...)`；[src/psysem/efa/workflow.py](../src/psysem/efa/workflow.py#L18-L69)
2. 读取最佳因子数 `best_n_factors`；
3. 根据载荷矩阵识别每题主因子；
4. 对高于阈值的交叉载荷保留自由参数；
5. 生成一份更“数据驱动”的 `ModelSpec`。

建议阈值：

1. 主载荷阈值：`0.30` 或 `0.35`
2. 交叉载荷阈值：`0.20` 或 `0.25`

### 关键决策

第一版不要让 `efa_seeded` 直接输出复杂旋转对象；先只输出一个**简化的载荷模式**：

1. 主载荷保留；
2. 强交叉载荷保留；
3. 极小交叉载荷去掉。

### 落地文件

1. `src/psysem/esem/adapters.py`
2. `src/psysem/esem/generators.py`
3. `tests/test_esem_generators.py`

### 完成标准

1. 至少能从一个 block 生成 1 个 EFA-seeded 候选；
2. 同一随机种子下行为可复现；
3. 与现有 EFA API 无缝对接。

---

## Step 5：实现 `target_pattern` generator

### 目标

引入“理论引导”的第三种候选生成方式。

### 实现内容

允许用户通过配置提供目标模式，例如：

```python
{
    "internalizing": {
        "i1": [1, 0],
        "i2": [1, 0],
        "i3": [0, 1],
        "i4": [0, 1],
    }
}
```

语义解释：

1. `1`：主载荷候选，默认自由；
2. `0`：默认固定为 0 或默认不生成；
3. 后续可扩展 `"~0"` 表示“近似零”，但第一版先不做。

### 第一版简化策略

第一版先把 target pattern 做成**载荷开关模板**，而不是完整 target rotation 数值优化。

也就是：

1. target 为 `1` 的位置，生成载荷；
2. target 为 `0` 的位置，不生成载荷；
3. 后续再升级成近似 target rotation。

### 落地文件

1. `src/psysem/esem/generators.py`
2. `src/psysem/esem/contracts.py`
3. `tests/test_esem_generators.py`

### 完成标准

1. 用户可提供 block 级 target pattern；
2. 生成结果可转成 `ModelSpec`；
3. 不提供 target 时不影响其他 generator。

---

## Step 6：建立 judge 注册机制

### 目标

像 EFA 的因子数判断一样，把 ESEM 评估拆成可插拔 judge，而不是写死一套规则。

### 实现内容

新增：

1. `register_esem_judge(name, fn)`
2. `get_esem_judge(name)`
3. `list_esem_judges()`

建议函数签名：

```python
def judge(candidate_result, workflow_config) -> ESEMJudgeResult:
    ...
```

每个 judge 返回：

1. `name`
2. `passed`
3. `score`
4. `details`
5. `warnings`

### 落地文件

1. `src/psysem/esem/judges.py`
2. `tests/test_esem_judges.py`

### 完成标准

1. judge 可独立测试；
2. 单个 judge 失败不会导致整个 workflow 崩溃；
3. 所有 judge 输出统一结构。

---

## Step 7：实现 `identification` judge

### 目标

复用现有 measurement/structural 检查，把“能不能估”作为第一道门。

### 实现内容

执行顺序：

1. 从候选 `ModelSpec` 构建 measurement design；[src/psysem/measurement/builder.py](../src/psysem/measurement/builder.py#L12-L197)
2. 调用 `check_measurement_identification(...)`；[src/psysem/measurement/identification.py](../src/psysem/measurement/identification.py#L6-L29)
3. 如有 structural，则调用 `check_structural_validity(...)`；[src/psysem/structural/validation.py](../src/psysem/structural/validation.py#L6-L24)
4. 根据 warning 数量和严重度给出 `passed/score`。

### 评分建议

1. 无 warning：`1.0`
2. 仅轻微 warning：`0.7`
3. 明显欠识别/结构异常：`0.0`

### 落地文件

1. `src/psysem/esem/judges.py`
2. `tests/test_esem_judges.py`

### 完成标准

1. 不可识别候选能被过滤或显著降分；
2. warning 文案可直接进入比较表。

---

## Step 8：实现 `cross_loading` judge

### 目标

把“结构是否足够清晰”变成显式规则。

### 实现内容

对每个候选拟合后的载荷结果，检查：

1. 每题是否只有一个主载荷；
2. 是否存在多个超过阈值的交叉载荷；
3. 交叉载荷比例是否过高。

### 评分建议

设：

- 主载荷阈值：$0.30$
- 交叉载荷阈值：$0.30$

可用：

$$
score = 1 - \frac{n_{cross}}{n_{items}}
$$

并裁剪到 $[0, 1]$。

### 数据来源

优先使用：

1. 若候选来自 `efa_seeded`，可复用 EFA 载荷；
2. 若候选已进入 SEM 结果，则从 measurement 参数中恢复主结构；
3. 第一版如果 SEM 结果还不足以恢复，可先仅对 `efa_seeded`/`target_pattern` 生效。

### 落地文件

1. `src/psysem/esem/judges.py`
2. `tests/test_esem_judges.py`

### 完成标准

1. 交叉载荷过多的候选能被降分；
2. 规则阈值可配置。

---

## Step 9：实现 `communality` judge

### 目标

用共同度衡量题项是否被因子结构充分解释。

### 实现内容

规则：

1. 统计低共同度题项数；
2. 题项共同度低于阈值则计入告警；
3. 低共同度比例越高，分数越低。

阈值建议：

1. `min_h2 = 0.20`

这个思路与现有 EFA 评价模块一致。[src/psysem/efa/evaluation.py](../src/psysem/efa/evaluation.py#L8-L54)

### 落地文件

1. `src/psysem/esem/judges.py`
2. `tests/test_esem_judges.py`

### 完成标准

1. 共同度差的候选能被识别；
2. 分数和告警可解释。

---

## Step 10：实现 `fit_indices` judge

### 目标

把 SEM 拟合指标也纳入候选比较，而不是只比较载荷模式。

### 实现内容

复用现有：

1. `SEMModel.fit(...)`；[src/psysem/core.py](../src/psysem/core.py#L33-L226)
2. `compute_fit_indices(...)`；[src/psysem/fit_indices.py](../src/psysem/fit_indices.py#L48-L167)

判断规则示例：

1. `CFI >= 0.90`
2. `TLI >= 0.90`
3. `RMSEA <= 0.08`
4. `SRMR <= 0.08`

评分可用简单加权：

$$
score = 0.3 \cdot CFI + 0.3 \cdot TLI + 0.2 \cdot (1 - RMSEA^*) + 0.2 \cdot (1 - SRMR^*)
$$

其中 $RMSEA^*$、$SRMR^*$ 为裁剪后的归一化值。

### 第一版注意事项

当前仓库的 SEM 估计仍是 prototype，因此该 judge 应允许：

1. 拟合失败时返回 `score=0` 但不使整个流程崩溃；
2. 指标缺失时给出 warning；
3. 保留“结构质量 judge”与“拟合质量 judge”并存。

### 落地文件

1. `src/psysem/esem/judges.py`
2. `tests/test_esem_judges.py`
3. `tests/test_esem_workflow.py`

### 完成标准

1. 可从 `SEMResult.fit_indices` 计算评分；
2. 指标不可用时有明确降级策略。

---

## Step 11：实现 `factor_balance` judge

### 目标

避免出现“一个因子几乎没有题，另一个因子题太多”的不平衡结构。

### 实现内容

统计每个因子下的 salient item 数，计算变异系数：

$$
cv = \frac{sd(n_1, n_2, ..., n_k)}{mean(n_1, n_2, ..., n_k)}
$$

建议评分：

$$
score = 1 - min(cv, 1)
$$

### 落地文件

1. `src/psysem/esem/judges.py`
2. `tests/test_esem_judges.py`

### 完成标准

1. 可识别明显不平衡因子结构；
2. 与 `cross_loading` judge 相互独立。

---

## Step 12：实现 selector 体系

### 目标

把多个 judge 的结果聚合成一个总分与最终选择结果。

### 实现内容

新增 selector：

1. `best_score`
2. `majority_vote`
3. `weighted_vote`

推荐第一版默认：`weighted_vote`

默认权重建议：

1. `identification`: `0.30`
2. `fit_indices`: `0.30`
3. `cross_loading`: `0.20`
4. `communality`: `0.10`
5. `factor_balance`: `0.10`

### 落地文件

1. `src/psysem/esem/selectors.py`
2. `tests/test_esem_selectors.py`

### 完成标准

1. 相同输入下选择结果稳定；
2. 平分场景有明确定义（如优先更简单模型）。

---

## Step 13：实现 `run_esem_workflow(...)`

### 目标

把 generator -> fit -> judge -> selector 串成一条主流程。

### 执行顺序

1. 读取 `workflow_config`；
2. 调用所有 generator，得到候选列表；
3. 对每个候选运行 `SEMModel.fit(...)`；
4. 依次执行所有 judge；
5. 聚合成总分；
6. 生成比较表；
7. 选择最佳候选；
8. 返回 `ESEMWorkflowResult`。

### 比较表最小字段

1. `candidate_id`
2. `strategy`
3. `n_blocks`
4. `n_latent`
5. `n_free_parameters`
6. `judge_identification`
7. `judge_cross_loading`
8. `judge_communality`
9. `judge_fit_indices`
10. `total_score`
11. `selected`

### 落地文件

1. `src/psysem/esem/workflow.py`
2. `src/psysem/esem/__init__.py`
3. `tests/test_esem_workflow.py`

### 完成标准

1. 用户可一行调用完整工作流；
2. 输出结构与 EFA workflow 风格一致；
3. 候选比较表可直接用于报告。

---

## Step 14：补示例与说明文档

### 目标

确保该工作流不是“只有代码能看懂”。

### 实现内容

新增：

1. `examples/basic_esem_workflow.py`
2. README 或 docs 中的快速示例
3. 一段“如何选 generator/judge/selector”的说明

### 示例至少演示

1. block 全展开基线；
2. 加入 `efa_seeded`；
3. 最终比较表输出。

### 落地文件

1. `examples/basic_esem_workflow.py`
2. `README.md`
3. `docs/index.md`

### 完成标准

1. 示例可运行；
2. 输出能说明“为什么选了这个候选”。

---

## Step 15：建立测试与质量门禁

### 目标

让这套 workflow 可重构、可维护。

### 最小测试矩阵

1. `test_esem_generators.py`
2. `test_esem_judges.py`
3. `test_esem_selectors.py`
4. `test_esem_workflow.py`

### 每类测试至少覆盖

1. 正常路径；
2. 空候选/非法配置；
3. 拟合失败降级；
4. 多候选平分；
5. 固定随机种子可复现。

### CI 完成标准

1. `python -m pytest -q` 全绿；
2. `ruff` 全绿；
3. `mypy src` 全绿；
4. 新增 API 已被 `__init__.py` 正确导出。

---

## 8. 第一版 API 草案

```python
from psysem import (
    ESEMWorkflowConfig,
    run_esem_workflow,
)

workflow = run_esem_workflow(
    data,
    spec,
    ESEMWorkflowConfig(
        generator_strategies=("block_full", "efa_seeded"),
        enabled_judges=(
            "identification",
            "cross_loading",
            "communality",
            "fit_indices",
        ),
        selector_strategy="weighted_vote",
    ),
)

print(workflow.comparison_table)
print(workflow.best_candidate.strategy)
print(workflow.best_candidate.sem_result.summary())
```

---

## 9. 为什么这个设计比“把逻辑直接写进 `model_spec_from_esem_spec`”更好

因为 `model_spec_from_esem_spec(...)` 只适合做：

1. 单一路径展开；
2. 轻量规范化；
3. 与底层 `ModelSpec` 对接。

但如果想引入：

1. 多种候选结构；
2. 多种判断规则；
3. 候选比较和自动选择；
4. EFA 桥接；

那么继续把逻辑塞进 [src/psysem/model.py](../src/psysem/model.py#L101-L138) 会很快失控。

更合理的做法是：

1. `model.py` 负责基础规范化；
2. `esem/generators.py` 负责候选生成；
3. `esem/judges.py` 负责评估；
4. `esem/selectors.py` 负责选择；
5. `esem/workflow.py` 负责总编排。

---

## 10. 推荐的最小落地顺序

如果只按最小风险路线推进，建议顺序为：

1. `Step 1` 契约
2. `Step 2` generator registry
3. `Step 3` `block_full`
4. `Step 6` judge registry
5. `Step 7` `identification`
6. `Step 10` `fit_indices`
7. `Step 12` selector
8. `Step 13` workflow
9. `Step 4` `efa_seeded`
10. `Step 8/9/11` 结构质量 judges
11. `Step 5` `target_pattern`
12. `Step 14/15` 示例与测试补齐

这样做的好处是：

1. 先把骨架跑起来；
2. 再接复杂候选生成器；
3. 风险和调试成本最低。

---

## 11. 完成定义（DoD）

第一版模块化 ESEM workflow 完成时，至少满足：

1. `run_esem_workflow(...)` 可运行；
2. 至少支持 `block_full` 与 `efa_seeded` 两类 generator；
3. 至少支持 `identification`、`fit_indices`、`cross_loading` 三类 judge；
4. 至少支持 `best_score` 与 `weighted_vote` 两类 selector；
5. 有候选比较表；
6. 有 1 个完整示例；
7. 有完整测试文件；
8. 文档已同步到 `docs/` 与导航。

---

## 12. 关联文档

1. [EFA Phase 实施文档（ZH）](efa-phase1-implementation.zh-CN.md)
2. [SEM Phase 实施文档（ZH）](sem-phase-implementation.zh-CN.md)
3. [SEM Phase 实施文档（ZH）](sem-phase-implementation.zh-CN.md)
4. [ESEM Baseline Research（ZH）](esem-baseline-landscape.zh-CN.md)
