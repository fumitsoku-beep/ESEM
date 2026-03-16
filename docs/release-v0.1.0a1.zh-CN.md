# psysem v0.1.0a1 发布说明与发布清单（ZH）

发布日期：2026-03-16  
版本：`0.1.0a1`

---

## 1. 版本定位

`v0.1.0a1` 是一次以 **SEM 结构收口、文档整理、发布准备** 为重点的 alpha 版本更新。

这一版本的核心目标不是新增大量方法，而是：

1. 固化新的 SEM 代码结构；
2. 删除不再需要的兼容层；
3. 统一文档与测试语义；
4. 为后续 benchmark 和正式版本节奏打基础。

---

## 2. 本次版本的核心变化

### 2.1 SEM 结构已正式收口

当前直接实现层统一为：

1. `psysem.sem.core`
2. `psysem.sem.model`
3. `psysem.sem.fit_indices`
4. `psysem.sem.result`
5. `psysem.sem.reporting`
6. `psysem.sem.parameter_index`
7. `psysem.sem.estimation`
8. `psysem.sem.inference`
9. `psysem.sem.measurement`
10. `psysem.sem.structural`

### 2.2 顶层 API 继续稳定

顶层 `psysem` 继续保留对外公共接口，用户仍可优先使用：

1. `SEMModel`
2. `sem`
3. `SEMResult`
4. `parse_model`
5. `SEMFitConfig`
6. `to_markdown`

### 2.3 旧 shim 已删除

旧的根层/包级 SEM 兼容层已移除，不再作为当前版本支持面。

### 2.4 文档与测试已同步

文档、测试、`esem` 依赖路径都已同步到新的 `psysem.sem.*` 结构。

---

## 3. 升级说明

### 3.1 推荐导入方式

推荐优先使用：

```python
from psysem import SEMModel, SEMFitConfig, parse_model
```

若需要直接导入 SEM 实现层，请使用：

```python
from psysem.sem.model import ModelSpec
from psysem.sem.estimation import optimize_ml_parameters
```

### 3.2 不再支持的旧导入路径

以下旧路径不再支持：

1. `psysem.core`
2. `psysem.model`
3. `psysem.fit_indices`
4. `psysem.result`
5. `psysem.reporting`
6. `psysem.parameter_index`
7. `psysem.estimation`
8. `psysem.inference`
9. `psysem.measurement`
10. `psysem.structural`

---

## 4. 验证结果

本次版本整理后的仓库验证结果：

1. 全量测试完成；
2. 测试结果：`193 passed`；
3. 新结构已通过仓库级回归验证。

补充说明（发布准备后的当日进展）：

1. 已补入第一批 SEM benchmark 原始 CSV 与 provenance JSON；
2. 已新增 `HolzingerSwineford1939`、`PoliticalDemocracy` 与 boundary cases 三组 benchmark 测试；
3. benchmark 当前仍定位为 prototype-level 基线验证，重点覆盖 Level A / Level B 与部分 warning 语义。

---

## 5. 发布清单

### 5.1 已完成

- [x] 版本号已更新到 `0.1.0a1`
- [x] `CHANGELOG.md` 已补充版本记录
- [x] 文档首页、README、架构说明已同步
- [x] 旧 SEM shim 文件已移除
- [x] `esem` 下游依赖已切换到新路径
- [x] 全量测试已通过

### 5.2 发布前人工确认

- [ ] 确认本次 tag 名称（建议：`v0.1.0a1`）
- [ ] 确认 GitHub Release 标题与摘要
- [ ] 确认是否同时发布源码压缩包 / wheel
- [ ] 确认 README 与文档站点已推送到目标分支

### 5.3 发布后建议动作

- [x] 补第一批 benchmark 数据与测试
- [x] 增加第一组 boundary warning benchmark
- [ ] 开始整理更正式的结果表输出
- [ ] 继续扩展 ESEM generator / judge / selector

---

## 6. 建议的 Release 摘要

可用于发布页摘要的简版说明：

> `psysem v0.1.0a1` 完成了 SEM 子系统目录收口，统一以 `psysem.sem.*` 作为直接实现路径，保留顶层 `psysem` 公共 API，并删除旧兼容层。文档、测试与 ESEM 下游依赖已同步更新，当前全量测试通过。

---

## 7. 一句话结论

`v0.1.0a1` 是一个“**把新结构真正落地并整理干净**”的 alpha 版本。
