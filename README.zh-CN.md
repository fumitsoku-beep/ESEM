# psysem（中文说明）

`psysem` 是一个面向心理学 SEM/ESEM 分析的 Python 包。

对应英文文档：[`README.md`](README.md)

## 目标流程（ESEM）

1. 提供宽表 `pandas.DataFrame`（一行一个被试）
2. 提供模型规格 `spec`（字典或 dataclass 结构）
3. 运行拟合（估计器 + 旋转设置）
4. 输出载荷、因子相关、拟合指标和参数表

## 可配置参数（重点）

### 顶层 `spec` 参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `blocks` | `list[dict]` | 是 | - | ESEM 分块列表。 |
| `estimator` | `str` | 是 | - | 估计方法。当前支持：`ML`、`MLR`、`WLSMV`。 |
| `variable_types` | `dict[str, str]` | 是 | - | 变量类型映射。取值：`continuous`、`ordinal`。 |
| `rotation` | `dict` | 否 | `None` | 全局旋转设置（可被 block 内的 rotation 覆盖）。 |
| `structural` | `list[str]` | 否 | `[]` | 结构路径表达式（目标行为）。 |
| `group` | `str` | 否 | `None` | 多组分析分组列。 |
| `weight` | `str` | 否 | `None` | 抽样权重列。 |
| `cluster` | `str` | 否 | `None` | 聚类样本列。 |
| `id` | `str` | 否 | `None` | 被试 ID 列。 |
| `allow_item_overlap` | `bool` | 否 | `False` | 是否允许同一题项出现在多个 block。 |

### `blocks[]` 参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `name` | `str` | 是 | - | block 名称（必须唯一）。 |
| `items` | `list[str]` | 是 | - | 本 block 使用的题项列名。 |
| `n_factors` | `int` | 是 | - | 因子个数，必须 `> 0` 且 `< len(items)`。 |
| `rotation` | `dict` | 否 | `None` | block 级旋转设置（覆盖全局设置）。 |

### `rotation` 参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `method` | `str` | 是 | - | 旋转方法名，例如 `geomin`、`target`。 |
| `oblique` | `bool` | 否 | `True` | 是否允许因子相关。 |

## `spec` 示例

```python
spec = {
    "blocks": [
        {
            "name": "internalizing",
            "items": ["i1", "i2", "i3", "i4", "i5", "i6"],
            "n_factors": 2,
        }
    ],
    "estimator": "WLSMV",
    "rotation": {"method": "geomin", "oblique": True},
    "variable_types": {
        "i1": "ordinal",
        "i2": "ordinal",
        "i3": "ordinal",
        "i4": "ordinal",
        "i5": "ordinal",
        "i6": "ordinal",
    },
    "group": "gender",
}
```

## 当前实现状态

- 已实现：`spec` 数据结构与基础校验
- 进行中：`SEMModel.fit(data, spec=...)` 主入口接入
- 后续：更严格 structural 语法解析 + ESEM 估计器实现
