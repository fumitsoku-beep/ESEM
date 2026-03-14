# ESEM 最小可跑路径（MVP，ZH）

当前日期：2026-03-14
适用分支：`main`

---

## 1. 本文档目的

给出一个**可实际执行**的最小 ESEM 路径，确保项目当前可以“跑通一次”：

1. `spec` 校验；
2. block 级 EFA 桥接（可选）；
3. `SEMModel.fit(..., spec=...)` 拟合；
4. 输出候选比较表与最佳结果。

---

## 2. 已落地入口

当前新增入口：

1. `run_esem_workflow(data, spec, config)`
2. `ESEMWorkflowConfig`
3. `ESEMWorkflowResult`

当前只实现一个生成策略：

1. `block_full`（按 `ESEMSpec.blocks` 全展开）

当前默认评估（judge）：

1. `convergence`
2. `fit_indices`
3. `efa_bridge`（可关闭）

### 2.1 当前实际执行顺序

当前 `run_esem_workflow(...)` 真正跑的是下面这条最小闭环：

1. `spec` 校验
2. 按 generator 生成候选（当前只有 `block_full`）
3. 对每个 block 收集 `variable_types`
4. block-level EFA bridge 进入 shared preprocessing
5. `SEMModel.fit(..., spec=...)` 跑 SEM 原型路径
6. judge 汇总 `convergence` / `fit_indices` / `efa_bridge`
7. selector 以 `best_score` 选最佳候选

也就是说，当前 ESEM 已经不是单纯的 `spec -> SEM fit`，而是：

```text
spec validation
-> block-level EFA bridge
-> judge/scoring
-> SEM fit
-> comparison/selection
```

### 2.2 最近更新让哪里发生了变化

最近这轮更新真正改变的是 **block-level EFA bridge 的输入层**，不是 SEM 优化器本身。

当前变化包括：

1. bridge 已统一复用 shared preprocessing，而不是单独维护一套相关矩阵准备逻辑
2. `ESEMWorkflowConfig` 已支持：
   - `efa_missing_strategy`
   - `efa_correlation_method`
3. 如果没有显式指定 `efa_correlation_method`，block 会按变量类型自动选择：
   - 全 `ordinal`：`polychoric`
   - 混合 `ordinal + continuous`：`spearman`
   - 全 `continuous`：`pearson`
4. 这些变化会直接影响：
   - `block_efa_results`
   - bridge warnings
   - `efa_bridge` judge
   - `comparison_table`
   - `total_score`
5. 这些变化目前**不会直接改变** `SEMModel.fit(...)` 的 ML 优化主链

这一点很重要：当前 bridge 仍是“候选解释与评分辅助层”，还不是“SEM 参数初始化层”。

---

## 3. 最小运行示例

```python
import pandas as pd

from psysem import ESEMWorkflowConfig, SEMFitConfig, run_esem_workflow

data = pd.read_csv("examples/data/efa_demo_input.csv")
items = list(data.columns)

spec_payload = {
    "blocks": [{"name": "demo", "items": items, "n_factors": 2}],
    "estimator": "ML",
    "variable_types": {item: "continuous" for item in items},
}

workflow = run_esem_workflow(
    data,
    spec_payload,
    ESEMWorkflowConfig(
        fit_config=SEMFitConfig(max_iter=200, restarts=1, random_seed=42),
    ),
)

print(workflow.best_candidate_id)
print(workflow.comparison_table)
print(workflow.best_candidate.sem_result.summary())
```

命令行示例：

```bash
python examples/basic_esem.py
```

---

## 4. 当前边界

1. 当前仅实现 `block_full` 单候选策略；
2. 多候选生成（如 `efa_seeded`、`target_pattern`）尚未接入；
3. selector 目前为单一 `best_score`；
4. block-level EFA bridge 虽已复用共享 preprocessing，并能按 block 自动选择 `pearson / spearman / polychoric`，但它当前仍主要服务于评分与解释，不会把 EFA 结果直接传进 SEM 起始值或约束；
5. 如果关闭 `include_efa_bridge`，最近这轮 preprocessing 更新基本不会改变 `SEMModel.fit(...)` 的拟合结果；
6. 当前 SEM 拟合主路径仍要求 `spec.estimator in {ML, MLR}`，`WLSMV` 等非 ML 主路径仍未接入完整 SEM 优化流程；
7. 该入口是“可跑通 MVP”，不是最终 ESEM 全能力版。

---

## 5. 下一步（如果继续沿 ESEM workflow 推进）

结合 [esem-modular-workflow.zh-CN.md](esem-modular-workflow.zh-CN.md) 和当前代码状态，建议按下面顺序推进：

1. **P0**：实现 generator 注册机制（Step 2）
2. **P0**：新增第二个候选策略 `efa_seeded`（Step 4）
3. **P0**：让 block-level EFA bridge 从“judge 辅助层”推进到“候选生成层”，即把 bridge 结果真正用于生成 `efa_seeded` 候选
4. **P0**：在 `SEMModel.fit(...)` 前补接 candidate-derived start values / target pattern，而不只是单独做 EFA 解释
5. **P0**：实现 judge 注册机制（Step 6）并把当前 judge 改为可插拔
6. **P0**：实现 selector 扩展（Step 12），支持多候选稳定选择
7. **P1**：补充端到端回归测试与多策略比较测试（Step 15）
8. **P1**：在上述主线稳定后，再推进 `WLSMV` 等非 ML 闭环，而不是在 MVP 阶段提前扩张

---

## 6. 关联文档

1. [ESEM 模块化判断工作流实施文档（ZH）](esem-modular-workflow.zh-CN.md)
2. [SEM Phase 实施文档（ZH）](sem-phase-implementation.zh-CN.md)
3. [共享预处理模块抽取与落地文档（ZH）](preprocessing-module-extraction.zh-CN.md)
