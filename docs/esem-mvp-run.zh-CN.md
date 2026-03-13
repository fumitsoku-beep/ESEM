# ESEM 最小可跑路径（MVP，ZH）

当前日期：2026-03-12  
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
4. `WLSMV` 等非 ML 主路径仍未接入完整 SEM 优化流程；
5. 该入口是“可跑通 MVP”，不是最终 ESEM 全能力版。

---

## 5. 下一步（依据现有文档的明确优先级）

结合 [esem-modular-workflow.zh-CN.md](esem-modular-workflow.zh-CN.md) 和当前代码状态，建议按下面顺序推进：

1. **P0**：实现 generator 注册机制（Step 2）
2. **P0**：新增第二个候选策略 `efa_seeded`（Step 4）
3. **P0**：实现 judge 注册机制（Step 6）并把当前 judge 改为可插拔
4. **P0**：实现 selector 扩展（Step 12），支持多候选稳定选择
5. **P1**：补充端到端回归测试与多策略比较测试（Step 15）

---

## 6. 关联文档

1. [ESEM 模块化判断工作流实施文档（ZH）](esem-modular-workflow.zh-CN.md)
2. [SEM Phase 实施文档（ZH）](sem-phase-implementation.zh-CN.md)
3. [SEM Phase 实施文档（ZH）](sem-phase-implementation.zh-CN.md)
