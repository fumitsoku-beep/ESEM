# 可配置参数（ESEM）

本页给出 `psysem` 中 `spec` 的可配置参数清单。

## 顶层 `spec` 参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `blocks` | `list[dict]` | 是 | - | ESEM 分块列表。 |
| `estimator` | `str` | 是 | - | 估计方法。规格层接受：`ML`、`MLR`、`WLSMV`；当前完整优化主路径仍以 `ML/MLR` 为主。 |
| `variable_types` | `dict[str, str]` | 是 | - | 变量类型映射。取值：`continuous`、`ordinal`。 |
| `rotation` | `dict` | 否 | `None` | 全局旋转设置（可被 block 内设置覆盖）。 |
| `structural` | `list[str]` | 否 | `[]` | 结构路径表达式。 |
| `group` | `str` | 否 | `None` | 多组分析分组列。 |
| `weight` | `str` | 否 | `None` | 抽样权重列。 |
| `cluster` | `str` | 否 | `None` | 聚类样本列。 |
| `id` | `str` | 否 | `None` | 被试 ID 列。 |
| `allow_item_overlap` | `bool` | 否 | `False` | 是否允许同一题项跨 block 复用。 |

## `blocks[]` 参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `name` | `str` | 是 | - | block 名称（必须唯一）。 |
| `items` | `list[str]` | 是 | - | 本 block 的题项列名。 |
| `n_factors` | `int` | 是 | - | 因子数量，必须 `> 0` 且 `< len(items)`。 |
| `rotation` | `dict` | 否 | `None` | block 级旋转设置。 |

## `rotation` 参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `method` | `str` | 是 | - | 旋转方法，例如 `geomin`、`target`。 |
| `oblique` | `bool` | 否 | `True` | 是否允许因子相关。 |

## 示例

```python
spec = {
    "blocks": [
        {"name": "internalizing", "items": ["i1", "i2", "i3", "i4"], "n_factors": 2}
    ],
    "estimator": "WLSMV",
    "rotation": {"method": "geomin", "oblique": True},
    "variable_types": {
        "i1": "ordinal",
        "i2": "ordinal",
        "i3": "ordinal",
        "i4": "ordinal",
    },
}
```
