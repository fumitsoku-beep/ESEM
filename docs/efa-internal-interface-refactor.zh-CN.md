# EFA 内部接口重构思路文档（ZH）

本文档用于说明：在继续扩展 `promax`、`oblimin`、`geomin`、`target rotation` 等方法之前，为什么需要先整理 `psysem` 当前 EFA 的内部接口，以及这次整理预计会修改什么、影响什么、为后续带来什么。

当前日期：2026-03-12  
适用分支：`main`

---

## 1. 这次修改的目的是什么？

这次修改的核心目的不是“再加一个方法”，而是把当前 EFA 主流程从：

- **能运行多个方法**

升级成：

- **能稳定承载不同类型方法**

也就是说，重点不只是支持新的方法名，而是让内部结构能够正确表达：

1. 不同提取法的差异；
2. 不同旋转法的差异；
3. 正交旋转与斜交旋转的差异；
4. 方法级诊断、warning、收敛信息；
5. 后续更复杂的输入相关矩阵、缺失处理、ESEM 衔接需求。

### 为什么现在就要做

当前 EFA 已经不再只有：

- `paf`
- `pca`
- `varimax`
- `none`

现在已经加入：

- `minres`
- `promax`

这意味着内部已经开始出现**方法学层面的结构差异**：

- `varimax` 属于正交旋转，默认因子不相关；
- `promax` 属于斜交旋转，需要显式返回因子相关矩阵；
- `minres` 比 `pca` / `paf` 更依赖优化路径和边界处理。

如果继续沿用“所有方法都只返回一个简单元组”的方式，后面很容易出现：

1. 新方法接入越来越脆弱；
2. `fit_efa()` 主流程越来越难读；
3. 斜交旋转的数学信息表达不完整；
4. 调试和测试成本越来越高。

因此，这次重构的目的本质上是：

> **让内部接口先稳定下来，再继续扩方法。**

---

## 2. 预计会修改什么？

这次重构预计主要会修改 `src/psysem/efa/fit.py` 内部结构，以及少量测试文件；对公开 API 的目标是尽量保持不变。

### 2.1 提取结果接口（Extraction Result）

当前提取方法返回的是一个较简单的元组，内容大致包括：

- `loadings`
- `communalities`
- `n_iter`
- `converged`

后续建议升级成统一的内部结果结构，例如：

- `loadings`
- `communalities`
- `n_iter`
- `converged`
- `warnings`
- `diagnostics`
- `method_metadata`

### 2.2 旋转结果接口（Rotation Result）

当前已经临时兼容了两种返回方式：

1. `loadings`
2. `(loadings, factor_correlation)`

这只是过渡状态。后续建议升级成统一内部结构，例如：

- `pattern_loadings`
- `structure_loadings`（可选）
- `factor_correlation`
- `rotation_matrix`
- `rotation_type`
- `warnings`
- `diagnostics`

### 2.3 后处理统一化

当前 `fit_efa()` 中还散落着一些后处理逻辑，例如：

- 共同度计算
- 唯一性计算
- explained variance
- residual matrix
- factor correlation
- cross-loading 检查

后续建议把这些整理成统一辅助函数，使正交与斜交旋转都走同一条清晰路径。

### 2.4 测试结构增强

为保证这次重构不会悄悄破坏行为，测试层面会补：

1. 正交旋转基线测试；
2. 斜交旋转基线测试；
3. 因子相关矩阵检查；
4. 极端数据 smoke tests；
5. 方法级 warning / 诊断测试。

---

## 3. 会影响目前的哪些链路？

## 3.1 会直接影响的链路

### 链路 A：`fit_efa(...)`

见 [src/psysem/efa/fit.py](../src/psysem/efa/fit.py)。

这是本次影响最大的部分，因为：

- 提取方法从这里进入；
- 旋转方法从这里进入；
- 所有结果后处理从这里汇总；
- `EFAResult` 也是从这里最终构建。

### 链路 B：`run_efa_workflow(...)`

见 [src/psysem/efa/workflow.py](../src/psysem/efa/workflow.py)。

工作流本身不一定要改公开接口，但会依赖 `fit_efa()` 的稳定行为。因此这次重构必须保证：

- 工作流输出不被破坏；
- 候选比较逻辑保持兼容；
- interpretation / evaluation 模块仍能正常读取 `EFAResult`。

### 链路 C：测试链路

包括：

- [tests/test_efa.py](../tests/test_efa.py)
- [tests/test_efa_workflow.py](../tests/test_efa_workflow.py)

这些测试会被同步调整，以确认内部重构没有破坏外部行为。

---

## 3.2 尽量不影响的链路

### 公开 API

以下公开 API 的目标是尽量不改：

- `fit_efa(...)`
- `run_efa_workflow(...)`
- `EFAConfig`
- `EFAResult`
- `list_extraction_methods()`
- `list_rotation_methods()`
- `register_extraction_method()`
- `register_rotation_method()`

也就是说，这次更偏向**内部契约重构**，而不是用户层 API 改名。

### EFA 诊断和因子数建议模块

以下模块目标上不应受本次重构直接影响：

- `diagnostics.py`
- `n_factors.py`
- `evaluation.py`
- `interpretation.py`

当然，如果后续要加入更丰富的结果字段，它们可以受益，但这不是这次重构的首要目标。

---

## 4. 这次重构会带来哪些好处？

## 4.1 对当前代码的好处

### 好处 1：主流程更清晰

`fit_efa()` 不再需要手动猜测“某个旋转方法到底返回了什么”，而是统一处理结构化结果。

### 好处 2：斜交旋转支持更自然

未来继续加入：

- `oblimin`
- `quartimin`
- `geomin`

时，不需要反复修改主流程。

### 好处 3：诊断能力增强

后续更容易加入：

- 方法级 warning
- 收敛信息
- 边界命中信息
- 方法内部诊断摘要

### 好处 4：测试更容易写

方法结果结构更明确后，可以更容易对：

- `factor_correlation`
- `rotation_type`
- `diagnostics`
- `warnings`

做稳定断言。

---

## 5. 对未来计划有什么影响？

## 5.1 对 P0 后续方法的影响

### `oblimin`

会明显受益。因为它和 `promax` 一样，属于斜交旋转，也需要：

- `factor_correlation`
- 更清晰的旋转结果接口

### `geomin`

也会受益。尤其如果后续 `geomin` 同时服务：

- EFA 斜交旋转
- ESEM 近似简单结构路线

那么现在把内部接口整理好，会减少后续返工。

---

## 5.2 对 EFA 方法扩展路线的影响

见 [docs/efa-method-expansion-roadmap.zh-CN.md](efa-method-expansion-roadmap.zh-CN.md)。

这次内部重构会让文档中列出的这些方法更容易接入：

- `ml`
- `oblimin`
- `geomin`
- `target rotation`
- `vss`（间接受益，主要受结果稳定性影响）
- `polychoric`（间接受益，主要受输入相关矩阵处理能力影响）

也就是说，这次重构虽然不是“直接新增所有方法”，但会显著降低后续开发成本。

---

## 5.3 对 ESEM 计划的影响

见 [docs/esem-modular-workflow.zh-CN.md](esem-modular-workflow.zh-CN.md)。

长期来看，这次重构对 ESEM 的帮助很大，因为 ESEM 后续会越来越依赖：

1. 更灵活的旋转结果表达；
2. 斜交因子结构；
3. 更丰富的方法级诊断；
4. 更稳定的载荷后处理。

尤其是：

- `target rotation`
- `geomin`
- `EFA-seeded` 候选

这些都需要更强的 EFA 内部基础设施。

---

## 6. 这次重构的风险是什么？

### 风险 1：内部重构可能破坏现有行为

哪怕不改公开 API，只要改了内部结果传递方式，就有可能让：

- `communalities`
- `residual_matrix`
- `factor_correlation`
- `warnings`

出现轻微变化。

### 风险 2：自定义注册方法兼容性

当前用户理论上可以注册自定义提取法和旋转法。如果内部接口变化太激进，可能影响这些自定义方法的兼容性。

### 风险 3：测试不足导致“看似通过，实际破坏”

因此这次重构必须搭配更完整的测试基线，尤其是：

- orthogonal vs oblique
- residual consistency
- factor correlation correctness

---

## 7. 这次重构的设计原则

为了控制风险，这次重构建议遵循以下原则：

### 原则 1：公开 API 尽量不变

尽量不修改用户直接调用的函数名和配置项。

### 原则 2：先做内部标准化，再继续新增方法

优先整理内部契约，而不是继续叠加“特判式补丁”。

### 原则 3：兼容旧注册方式

如果可能，应允许旧式自定义方法继续工作；或至少提供平滑兼容层。

### 原则 4：先把正交 / 斜交语义分清楚

这是后续所有旋转方法稳定接入的前提。

### 原则 5：测试先行

每做一步内部改造，都要用现有测试和新增测试双重验证。

---

## 8. 建议的实施顺序

建议按下面顺序推进：

1. 引入内部 `RotationResult` 结构；
2. 引入内部 `ExtractionResult` 结构；
3. 将 `fit_efa()` 的后处理统一走结构化结果；
4. 补正交/斜交测试基线；
5. 再继续实现 `oblimin`；
6. 再继续实现 `geomin`。

这样做的好处是：

- 先把地基打稳；
- 再往上加方法；
- 每一步回归风险都更可控。

---

## 9. 一句话总结

这次内部接口重构的本质，不是为了“重构而重构”，而是为了把 `psysem` 的 EFA 层从：

- 当前能跑的多方法原型

提升到：

- 能持续扩展、能稳定承载正交/斜交方法差异的基础设施。

如果后续要继续按 P0 路线完成：

- `oblimin`
- `geomin`

那么先做这次内部接口整理，是合理且必要的。
