# Agent 架构落实方案

## 1. 目标与约束

本方案解决 3 个核心目标：

1. 让 `final_agent` 从“文档完整、实现断线”的状态，落到“主链路可运行、上下文可透传、子图可消费”的状态。
2. 把“工具学习”的重心从**工具纠错**改成**工具选择学习**，尤其支持“并行/互补工具组合”的主动选择。
3. 在不推翻现有代码骨架的前提下，给短期记忆、长期记忆、上下文压缩、DeepAgent、文件系统与并行子 Agent 建立清晰边界。

本方案明确采用以下约束：

- 短期记忆继续使用 LangGraph `checkpointer + thread_id`。
- 长期记忆继续使用 Mem0，职责聚焦在跨线程用户背景、历史诊疗摘要、语义关系。
- `focus` 负责上下文压缩，不承担长期记忆与学习存储职责。
- 诊断子图优先建设“强工具学习”，养生子图优先建设“策略学习”。
- 文件系统不是所有子图都要暴露的通用工具，只在长链路、重结果、可回读的场景中启用。

## 2. 证据驱动的问题清单

### 2.1 运行链路硬故障

以下问题会直接导致 `final_agent` 主链路不可运行或行为偏离设计：

1. `backend/app/src/final_agent/builder.py` 仍导入 `super_agent` handler，但仓库中不存在 `super_agent` 目录。
2. `backend/app/src/final_agent/middleware/__init__.py` 仍导入 `super_agent.middleware.*`，而 `final_agent/middleware/` 已有本地实现。
3. `backend/app/src/final_agent/components/diagnose/nodes/complex/__init__.py` 导入不存在的 `deep_search_agent.py`，实际文件为 `deep_search_agent_custom.py`。
4. 多处 `final_agent` 文件仍从 `app.src.agent.*` 导入 `states/prompts/config/tools`，导致运行时混入旧架构。
5. 多处 handler/node 仍按 `state.xxx` 访问状态，但 `final_agent` 主图实际采用 `TypedDict`/dict 风格。
6. `backend/app/src/final_agent/components/diagnose/router.py` 仍将 `complex` 路径降级到 `moderate`。

### 2.2 中间件契约断裂

1. `LearningMiddleware.before_model()` 以 `source=self.source` 调用 `SelfLearner.get_thread_learning_snapshot(...)`，但当前 learner 签名并不接受 `source`。
2. `MemoryMiddleware` / `LearningMiddleware` 在同步中间件钩子里直接调用 `asyncio.run(...)`，与 LangGraph 异步节点组合时存在事件循环冲突风险。
3. `get_learner()` 当前未注入持久化存储，会导致跨线程学习虽然有结构定义，但没有真正落库与回放能力。

### 2.3 子图上下文透传不完整

1. 诊断子图 `DiagnoseInputState` 未显式接收 `user_id`、`conversation_id`、`memory_context`、`cross_thread_learning`、`thread_learning_context`。
2. `handle_diagnose_query()` 未把这些上下文从主图传入子图。
3. 养生子图同样缺少上述上下文字段，导致养生侧无法利用历史画像与学习经验。
4. `complex_diagnosis` 创建 DeepAgent 时重新生成隔离的 `thread_id`，并只传入一条 prompt，主图短期记忆无法延续到复杂诊断执行链。

### 2.4 工具学习方向偏差

当前代码和文档已经有“工具选择纠错”的事件结构，但仍缺 4 类关键资产：

1. **正向选择模板**：什么查询要主动组合 `web_search + case_vector_search`。
2. **阶段化工具计划**：先检索什么、并行什么、何时停止。
3. **成功序列回放**：不是只记错了什么，而是记住“什么组合在什么场景下有效”。
4. **子图差异化语义**：诊断是工具选择学习，养生当前更应该是策略升级学习，而不是照搬诊断工具规则。

### 2.5 文件系统与并行 Agent 边界不清

1. 当前仓库有 legacy `filesystem.py`，DeepAgent 也接了 `FilesystemMiddleware`，但主图/子图/DeepAgent 三层没有统一边界。
2. 当前 `deep_search_agent_custom.py` 将 Filesystem/SubAgent 直接硬编码在复杂诊断里，但没有将“何时需要文件系统”纳入架构说明。
3. 诊断复杂链路适合文件系统，是因为工具结果大、需要回读；简单/中等/养生链路不应默认暴露文件系统工具。

## 3. 架构原则

### 3.1 主图只做请求级协调，不做域内工具选择

主图负责：

- 安全守卫
- 短期记忆恢复（checkpointer）
- 长期记忆检索（Mem0）
- 全局路由
- 上下文压缩入口
- 脱敏与最终输出

主图不负责：

- 诊断域内部工具时机判断
- 养生域内部增强策略判断
- DeepAgent 内部专家协作细节

### 3.2 子图学习“怎么做”，记忆回答“看见谁”

- **短期记忆**：当前会话已说过什么、已追问什么。
- **长期记忆**：这个用户过去有什么体质/病程/疗效背景。
- **工具选择学习**：遇到这类问题，该先查什么、并行什么、何时停止。

结论：

> 记忆负责提供条件，学习负责决定流程。

### 3.3 文件系统是长链路上下文工具，不是全局默认工具

建议边界如下：

- 主图：不暴露文件系统工具。
- 简单/中等诊断：默认不暴露文件系统工具，只做轻量 prompt 注入。
- 复杂诊断 DeepAgent：启用文件系统，用于大工具结果驱逐、回读、搜索。
- 养生子图：在引入外部检索前不暴露文件系统工具。

### 3.4 并行 Agent 的价值是上下文隔离，不是角色堆叠

复杂诊断中的并行专家应继续保留，但要遵循：

- 只有 `complex` 路径才启用并行专家。
- 中等辨证优先用 Map-Reduce + 工具并行，不要过早上升为多专家。
- 并行专家主要解决“多视角推理”和“上下文隔离”，不替代工具选择计划。

## 4. 目标架构

### 4.1 主图层

职责：

1. `GuardrailsMiddleware`
2. `MemoryMiddleware`
3. `FocusContextMiddleware`（主图裁剪）
4. `LearningMiddleware(source="main_graph")`
5. Router
6. 子图调用
7. after middleware

输出给子图的统一上下文包：

- `user_id`
- `conversation_id`
- `messages`（已裁剪）
- `memory_context`
- `thread_learning_context`
- `cross_thread_learning`
- `router/extracted_entities`
- `llm_config`

### 4.2 诊断子图层

职责：

1. 信息收集与复杂度评估
2. 在 simple/moderate/complex 三条路径中消费历史记忆与学习经验
3. 生成诊断域内的工具选择计划
4. 记录本次工具选择成功/失败事件

诊断子图应沉淀 3 类资产：

- 工具时机规则
- 工具组合模板
- 成功工具序列

### 4.3 养生子图层

当前阶段不强行工具化，优先学习：

- 何时继续留在 wellness
- 何时升级为 diagnose
- 是否需要外部增强
- 历史画像如何影响建议模板

未来如果引入季节知识检索、体质库、外部资讯检索，再升级为轻工具学习。

### 4.4 Complex DeepAgent 层

DeepAgent 的职责不是替代主图/子图路由，而是在复杂诊断里执行：

- 多工具并行查询
- 并行专家分析
- 文件系统承载大结果
- 依据工具选择计划执行互补工具组合

它应接收：

- 裁剪后的 `messages`
- 用户画像/长期记忆摘要
- 鉴别规则/误诊模式/有效策略
- 结构化工具选择计划

## 5. “工具选择学习”落地方式

### 5.1 从“纠错记录”升级到“选择计划”

新增目标不是简单记录：

- `wrong_tool`
- `correct_tool`

而是让系统在执行前就得到：

- `query_goal`
- `required_evidence`
- `recommended_tools`
- `parallel_groups`
- `stop_conditions`
- `fallback_path`

### 5.2 场景化工具组合模板

至少建设以下主动组合：

1. **医案/相似病例请求**
   - `case_vector_search` + `web_search`
   - 前者找相似案例，后者补“近期公开资料/新近讨论/外部补证”

2. **复杂兼证/病程长/既往治疗无效**
   - `kg_syndrome_search` + `case_vector_search` + `classics_search`
   - 必要时再加 `web_search`

3. **理论依据/古籍出处请求**
   - `classics_search` + `kg_syndrome_search`

4. **现代研究/中西医对照请求**
   - `web_search` + `medical_research_search`

### 5.3 学习资产的来源

工具选择计划需要融合 3 类输入：

1. `thread_learning_context.tool_learning`：当前线程即时纠偏
2. `cross_thread_learning.effective_strategies`：跨线程高效模板
3. `memory_context`：用户长期背景（如既往复杂兼证、重复治疗失败）

### 5.4 成功经验必须入库

后续学习不能只靠失败反思，还要从成功案例提取：

- 哪个阶段用了哪些工具
- 哪些工具是并行的
- 哪个组合减少了轮次
- 哪个组合提高了证型置信度

## 6. 外部资料对照结论

### 6.1 LangGraph Checkpointer

依据 LangGraph 文档：

- `thread_id` 是短期记忆恢复主键。
- `MemorySaver` 适合开发环境。
- 生产环境应使用 `PostgresSaver` 等 durable saver。

因此本项目应坚持：

- 主图与复杂诊断不要随意重建新的 thread 语义。
- Complex DeepAgent 至少要感知主会话 ID，而不是每次重新生成完全隔离的新线程语义。

### 6.2 Mem0

依据 Mem0 官方示例与最佳实践：

- 应存储“长期有价值”的事实与摘要，而不是所有 filler 消息。
- 图关系适合语义实体与关系回查。

因此本项目应坚持：

- Mem0 存用户背景、历史诊断摘要、关键疗效反馈。
- 不把工具日志和临时推理链塞进 Mem0。

### 6.3 DeepAgents 文件系统与子 Agent

依据 DeepAgents 文档：

- 文件系统用于大结果驱逐与按需回读。
- 子 Agent 的首要价值是上下文隔离与并行，而不是把一切都拆成“专家角色扮演”。

因此本项目应坚持：

- 文件系统只在复杂链路启用。
- 子 Agent 聚焦复杂辨证，不下沉到所有简单任务。

## 7. 分阶段实施

### Phase 0：运行链路修复

目标：让 `final_agent` 真正可 import、可 build、可走通主链路。

内容：

- 修复 `super_agent` / `agent` 错误导入
- 修复 `complex` 路由降级
- 修复 `deep_search_agent_custom` / `__init__` 导出
- 修复 dict 风格 state 访问
- 修复中间件 async 契约问题

### Phase 1：上下文透传闭环

目标：主图的记忆/学习上下文能进入 diagnose / wellness。

内容：

- 扩展 diagnose/wellness state
- handler 全量透传
- simple/moderate/complex 消费 `memory_context` 与学习上下文

### Phase 2：工具选择计划落地

目标：在诊断子图中引入执行前的工具选择计划，而不是只在失败后纠错。

内容：

- 新增工具选择计划生成器
- 在 moderate / complex prompt 中注入推荐工具组合
- 对医案查询默认支持 `vector + web` 并行策略

### Phase 3：复杂诊断上下文工具化

目标：让 DeepAgent 可以查询主图记忆和学习规则，而不是只看一段拼接 prompt。

内容：

- 新增 `context_tools.py`
- DeepAgent 接收 `memory_context` / `learning_context`
- 复杂诊断传入裁剪后的 `messages`

### Phase 4：学习闭环增强

目标：既能记录失败，也能提取成功工具组合。

内容：

- 记录成功工具序列
- 形成 `effective tool sequence` 资产
- 让 cross-thread learning 可回放

### Phase 5：养生子图策略学习

目标：不急着把养生变成重工具链，先把“升级/增强/复用画像”策略做对。

## 8. 本次代码实现范围

本轮实现优先覆盖以下内容：

1. `final_agent` 运行链路修复
2. diagnose / wellness 上下文透传
3. diagnose 侧的工具选择计划基础设施
4. complex diagnosis 的 context tools 与 messages 直传

暂不在本轮强推：

- 全量跨线程学习持久化重构
- 养生子图重工具化
- 全量评估框架与自动聚合任务

## 9. 验证标准

本方案落地后，应满足：

1. `final_agent` 主图可成功 import/build。
2. 诊断子图能收到 `memory_context` / `thread_learning_context` / `cross_thread_learning`。
3. complex diagnosis 不再丢失主图 `messages`。
4. tool selection plan 能对“医案查询”主动推荐 `vector + web`。
5. DeepAgent 可使用上下文工具读取长期记忆与学习规则。
6. 相关测试、类型检查与构建验证可通过或明确暴露剩余阻塞点。
