# Network MVP（ZH）

本文档说明当前 `psysem` 中已落地的网络分析最小可用版本。

## 1. 当前提供什么

当前 `network` 模块提供的是一个**横断面、无向、项目级**网络 MVP：

1. 共享复用 `psysem.preprocessing.build_association_matrix(...)`
2. 支持 `pairwise` / `dropna`
3. 支持 `pearson` / `spearman` / `polychoric`
4. 通过 precision matrix 估计 partial-correlation network
5. 输出：
   - `association_matrix`
   - `precision_matrix`
   - `partial_correlation_matrix`
   - `adjacency_matrix`
   - `edge_table`
   - `node_table`
   - preprocessing metadata 与 warnings

当前公开入口是：

```python
from psysem import NetworkConfig, fit_network
```

---

## 2. 最小使用示例

```python
import pandas as pd
from psysem import NetworkConfig, fit_network

data = pd.read_csv("examples/data/efa_demo_input.csv")

result = fit_network(
    data,
    NetworkConfig(
        items=("i1", "i2", "i3", "i4"),
        correlation_method="pearson",
        missing_strategy="pairwise",
        min_abs_edge=0.05,
    ),
)

print(result.edge_table.head())
print(result.node_table)
```

如果是 ordinal / Likert 数据，可直接改成：

```python
result = fit_network(
    data,
    NetworkConfig(
        items=("i1", "i2", "i3", "i4"),
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

---

## 3. 结果对象说明

`NetworkResult` 当前至少包含：

1. `association_matrix`
2. `precision_matrix`
3. `partial_correlation_matrix`
4. `adjacency_matrix`
5. `edge_table`
6. `node_table`
7. `pairwise_n`
8. `resolved_variable_types`
9. `warnings`

其中：

1. `partial_correlation_matrix` 保留原始 partial-correlation 权重
2. `adjacency_matrix` 会应用 `min_abs_edge` 阈值
3. `edge_table` 以 `abs_weight` 降序输出边
4. `node_table` 当前提供：
   - `degree`
   - `strength`
   - `expected_influence`
   - `positive_strength`
   - `negative_strength`

---

## 4. 当前边界

当前 `network` 仍是 MVP，不包含：

1. `EBICglasso`
2. bootstrap / stability 分析
3. community detection
4. network visualization
5. longitudinal / multilevel network
6. mixed-type network 的完整估计器

当前唯一网络估计器是：

1. `estimator="ggm"`

实现方式是：

1. 先构造关联矩阵
2. 再做 precision matrix 反演
3. 再转为 partial-correlation network

---

## 5. 下一步

如果继续沿 network 路线推进，建议优先级是：

1. 增强 `polychoric` 稳健性与 mixed-type 路径
2. 引入正则化网络估计（如 `EBICglasso`）
3. 补 bootstrap / centrality stability
4. 再补 visualization 与 reporting

---

## 6. 关联文档

1. [共享预处理模块抽取与落地（ZH）](preprocessing-module-extraction.zh-CN.md)
2. [EFA 方法扩展路线图（ZH）](efa-method-expansion-roadmap.zh-CN.md)
3. [ESEM 模块化判断工作流实施文档（ZH）](esem-modular-workflow.zh-CN.md)
