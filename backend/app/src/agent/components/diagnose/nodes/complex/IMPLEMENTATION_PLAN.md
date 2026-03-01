# DeepSearch Agent 实施方案

> **基于 LangChain DeepAgents 框架的复杂辨证系统**
> 更新时间：2026-02-05
> 技术栈：DeepAgents + Skills + Middleware + 并行子 Agent
> **状态：基于现有架构的渐进式优化**

---

## 零、现有架构评估与优化策略

### 0.1 现有架构优势 ✅

SmartTCM Agent 系统已经具备企业级架构的核心要素：

| 模块 | 实现状态 | 质量评估 | 保留/优化 |
|------|---------|---------|----------|
| **中间件系统** | ✅ 完整实现（5个中间件） | 优秀 | **保留** |
| **意图识别** | ✅ 三层架构（规则+语义+LLM） | 优秀 | **保留** |
| **上下文工程** | ✅ 完整实现（5个组件） | 优秀 | **保留** |
| **Neo4j 集成** | ✅ 32个预定义模板 | 优秀 | **保留** |
| **Guardrails** | ✅ 两层检测（规则+LLM） | 优秀 | **保留** |
| **状态管理** | ✅ Pydantic 模型 | 良好 | **升级持久化** |
| **错误处理** | ✅ 多层降级 | 良好 | **补充重试** |
| **养生子图** | ✅ LangGraph 子图 | 良好 | **保留** |

### 0.2 关键缺失与优化优先级

| 功能 | 现状 | 影响 | 优先级 | 预期收益 |
|------|------|------|--------|---------|
| **并行执行** | ❌ 串行执行 | 延迟高 7秒+ | ⭐⭐⭐⭐⭐ | 延迟降低 50-70% |
| **大结果驱逐** | ❌ 无自动驱逐 | 上下文溢出风险 | ⭐⭐⭐⭐⭐ | 支持超大查询 |
| **Skills 系统** | ❌ 硬编码提示词 | 难以维护 | ⭐⭐⭐⭐ | 知识与代码分离 |
| **生产持久化** | ⚠️ MemorySaver | 重启丢失会话 | ⭐⭐⭐⭐ | 会话持久化 |
| **成本控制** | ⚠️ 功能有限 | 成本失控风险 | ⭐⭐⭐ | 防止失控 |
| **模型降级** | ❌ 无 | 可靠性不足 | ⭐⭐⭐ | 提升可靠性 |

### 0.3 优化策略

**核心原则**：
1. **渐进式迁移**：保留优秀的现有实现，只替换有问题的部分
2. **向后兼容**：确保优化不破坏现有功能
3. **性能优先**：优先解决性能瓶颈（并行执行）
4. **稳定性优先**：优先解决稳定性问题（大结果驱逐、持久化）

**不迁移的部分**（保留现有实现）：
- Guardrails 中间件（现有实现更专业）
- 意图识别系统（三层架构已经很完善）
- 上下文工程（自定义实现更符合 TCM 需求）
- Neo4j 工具（32个预定义模板已经很完整）
- 日志中间件（现有实现满足需求）

**优化实施顺序**：
1. **Phase 1（Critical）**：并行执行 + 大结果驱逐
2. **Phase 2（Important）**：PostgresSaver + Skills 系统 + 成本控制
3. **Phase 3（Nice to Have）**：LangSmith + 模型降级

---

## 一、技术方案概览

### 1.1 核心决策

**放弃手动实现，全面采用 DeepAgents 框架**，充分利用 LangChain 2026 年最新特性：

| 原计划（手动实现） | 新方案（DeepAgents） | 优势 |
|------------------|-------------------|------|
| 手动实现 Planner | `TodoListMiddleware` | 自动任务分解、状态跟踪、动态调整 |
| 手动实现 Router | `SubAgentMiddleware` | 自动并行、上下文隔离、专业化 |
| 依赖模型上下文 | `FilesystemMiddleware` | 15k+ tokens 自动驱逐到文件 |
| 手动实现 Synthesizer | 子 Agent 综合分析 | LLM 自动综合，支持迭代 |
| 手动实现 Validator | 中间件钩子 + 子 Agent | 自动触发、可跳回重新分析 |
| ❌ 缺失 | `HumanInTheLoopMiddleware` | 医生审批关键诊断 |
| ❌ 缺失 | `SummarizationMiddleware` | 超长病例自动压缩 |
| ❌ 缺失 | Skills 系统 | 渐进式加载知识 |

### 1.2 架构设计

```
┌────────────────────────────────────────────────────────────────┐
│                 DeepSearch Agent (主 Agent)                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────── Skills (渐进式加载) ─────────────────┐     │
│  │  - complex_diagnosis_workflow.md  (辨证流程)         │     │
│  │  - syndrome_theory.md              (证型理论)         │     │
│  │  - validation_criteria.md          (验证标准)         │     │
│  └──────────────────────────────────────────────────────┘     │
│           │                                                    │
│           │ 根据任务自动加载                                    │
│           │                                                    │
│  ┌──────────────── 中间件栈 (Middleware Stack) ─────────┐     │
│  │                                                       │     │
│  │  1. TodoListMiddleware        (任务规划与跟踪)         │     │
│  │  2. FilesystemMiddleware      (上下文管理,驱逐15k+)   │     │
│  │  3. SubAgentMiddleware        (并行子Agent调度)       │     │
│  │  4. SummarizationMiddleware   (对话历史压缩)          │     │
│  │                                                       │     │
│  └──────────────────────────────────────────────────────┘     │
│           │                                                    │
│           │ TodoList 任务分解(模型调用 #1)                      │
│           │                                                    │
│  ┌──────────────── 数据查询层 (4个工具,无模型调用) ──────┐     │
│  │                                                       │     │
│  │  ┌───────────┬──────────┬──────────┬────────────┐   │     │
│  │  │kg_syndrome│case_vector│classics  │web_search  │   │     │
│  │  │  _search  │  _search │ _search  │  (Tavily)  │   │     │
│  │  ├───────────┼──────────┼──────────┼────────────┤   │     │
│  │  │ Neo4j     │ Milvus/  │ 全文检索 │ 网络搜索   │   │     │
│  │  │ 证型查询  │ 医案检索 │ 古籍检索 │ 最新研究   │   │     │
│  │  └───────────┴──────────┴──────────┴────────────┘   │     │
│  │           │           │          │            │      │     │
│  │           └───────────┴──────────┴────────────┘      │     │
│  │                       │                              │     │
│  │                       ▼ 形成知识库                    │     │
│  │               [knowledge_base]                       │     │
│  │                       │                              │     │
│  └───────────────────────┼──────────────────────────────┘     │
│                          │                                    │
│                          │ 提供给专家层                        │
│                          │                                    │
│  ┌──────────────── 专家推理层 (5个子Agent,并行) ─────────┐     │
│  │                                                       │     │
│  │  ┌──────────┬──────────┬──────────┬──────────┐      │     │
│  │  │鉴别诊断  │治则治法  │方药推荐  │预后评估  │      │     │
│  │  │专家      │专家      │专家      │专家      │      │     │
│  │  ├──────────┼──────────┼──────────┼──────────┤      │     │
│  │  │使用已有  │使用已有  │使用已有  │使用已有  │      │     │
│  │  │提示词    │提示词    │提示词    │提示词    │      │     │
│  │  │          │          │          │          │      │     │
│  │  │   +      │    +     │    +     │    +     │      │     │
│  │  │          │          │          │          │      │     │
│  │  │质疑验证专家 (验证以上4位专家的结论)        │      │     │
│  │  └──────────┴──────────┴──────────┴──────────┘      │     │
│  │           │           │          │            │      │     │
│  │           └───────────┴──────────┴────────────┘      │     │
│  │                       │                              │     │
│  │                       ▼ 自动 Reduce                   │     │
│  │                  [专家意见汇总]                        │     │
│  │                       │                              │     │
│  └───────────────────────┼──────────────────────────────┘     │
│                          │                                    │
│                          ▼ 主Agent综合决策(模型调用 #2)        │
│                  最终辨证结果 + 置信度                         │
│                                                                │
│  💡 混合架构优势：7次模型调用(1主+5专家+1综合)                 │
│     专业性高 | 可复用提示词 | 可追溯推理链                     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 二、并行执行方案（混合架构：子 Agent + 工具）

### 2.1 核心架构：专家子 Agent + 数据工具

**关键决策**：使用子 Agent 实现专家推理，使用工具实现数据查询，两者结合。

**架构优势**：

| 层次 | 实现方式 | 调用模型 | 作用 |
|------|---------|---------|------|
| **专家推理层** | 子 Agent | ✅ 是 | 鉴别诊断、治则治法、方药推荐等专业推理 |
| **数据查询层** | 工具调用 | ❌ 否 | Neo4j、向量检索、古籍检索、Web搜索 |
| **任务编排层** | TodoList | ✅ 是 | 任务分解与综合分析 |

**与现有提示词的关系**：

`deepsearch_prompts.py` 中已有5个专家提示词（共800行）：
1. `DIFFERENTIAL_DIAGNOSIS_EXPERT_PROMPT` → 鉴别诊断子 Agent
2. `TREATMENT_PRINCIPLE_EXPERT_PROMPT` → 治则治法子 Agent
3. `PRESCRIPTION_RECOMMENDATION_EXPERT_PROMPT` → 方药推荐子 Agent
4. `PROGNOSIS_EVALUATION_EXPERT_PROMPT` → 预后评估子 Agent
5. `VERIFICATION_EXPERT_PROMPT` → 质疑验证子 Agent

**成本分析**：

| 方案 | 模型调用 | 总成本 | 优势 |
|------|---------|-------|------|
| **纯工具方案** | 2次 | 低 | 成本最低，但无法实现专业推理 |
| **混合方案** ✅ | 7次（主1次 + 5专家 + 综合1次） | 中 | 平衡成本与专业性 |
| **全LLM方案** | 10+次 | 高 | 专业但成本过高 |

**架构设计**：

```
用户问题："头痛、胸闷、失眠，腰膝酸软"
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  主 Agent (模型调用 #1)                                      │
│  - TodoListMiddleware 分解任务                               │
│  - 输出：需要并行的专家任务 + 数据查询任务                    │
└─────────────────────────────────────────────────────────────┘
    │
    ├─────────── 数据查询层（并行，无模型调用）──────────┐
    │                                                      │
    ├─ [kg_syndrome_search]      Neo4j查询证型           │
    ├─ [case_vector_search]      向量检索相似医案         │
    ├─ [classics_search]         全文检索古籍            │
    └─ [web_search]              网络搜索最新研究         │
           │                                              │
           │ 数据结果汇总                                  │
           ▼                                              │
    [knowledge_base] ─────────────────────────────────────┘
           │
           │ 提供给专家层
           ▼
    ┌───────── 专家推理层（并行，5个子 Agent）──────────┐
    │                                                    │
    ├─ [鉴别诊断专家]  区分相似证型，排除误诊           │
    │     ↓ DIFFERENTIAL_DIAGNOSIS_EXPERT_PROMPT        │
    │                                                    │
    ├─ [治则治法专家]  制定治疗策略                     │
    │     ↓ TREATMENT_PRINCIPLE_EXPERT_PROMPT           │
    │                                                    │
    ├─ [方药推荐专家]  推荐方剂与食疗                   │
    │     ↓ PRESCRIPTION_RECOMMENDATION_EXPERT_PROMPT   │
    │                                                    │
    ├─ [预后评估专家]  判断病情趋势                     │
    │     ↓ PROGNOSIS_EVALUATION_EXPERT_PROMPT          │
    │                                                    │
    └─ [质疑验证专家]  验证诊断合理性                   │
          ↓ VERIFICATION_EXPERT_PROMPT                   │
                                                         │
    专家意见汇总 ──────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  主 Agent (模型调用 #2)                                      │
│  - 综合5位专家的意见                                         │
│  - 输出：最终辨证结果 + 治疗建议                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 实现方式：SubAgentMiddleware 自动并行

**完整代码示例**：

```python
from langgraph.graph import StateGraph, START, END
from typing import List, Dict, Any
from langchain.tools import tool
from deepagents import create_deep_agent
from deepagents.middleware import SubAgentMiddleware, TodoListMiddleware

# ========== 1. 定义数据查询工具（无模型调用） ==========

@tool
async def kg_syndrome_search(symptoms: List[str]) -> Dict:
    """查询知识图谱中的证型"""
    # Neo4j 查询逻辑
    return {"syndromes": ["肝郁脾虚", "心肾不交"]}

@tool
async def case_vector_search(query: str) -> Dict:
    """检索相似医案"""
    # 向量检索逻辑
    return {"similar_cases": [{"id": "case_123", "syndrome": "肝郁脾虚"}]}

@tool  
async def classics_search(keywords: List[str]) -> Dict:
    """检索古籍论述"""
    # 全文检索逻辑
    return {"citations": [{"book": "伤寒论", "text": "..."}]}

@tool
async def web_search(query: str) -> Dict:
    """网络搜索"""
    # Tavily 搜索逻辑
    return {"results": [{"title": "...", "content": "..."}]}


# ========== 2. 创建专家子 Agent（使用 deepsearch_prompts.py）==========

from ..prompts.deepsearch_prompts import (
    DIFFERENTIAL_DIAGNOSIS_EXPERT_PROMPT,
    TREATMENT_PRINCIPLE_EXPERT_PROMPT,
    PRESCRIPTION_RECOMMENDATION_EXPERT_PROMPT,
    PROGNOSIS_EVALUATION_EXPERT_PROMPT,
    VERIFICATION_EXPERT_PROMPT,
)

def create_differential_expert(llm):
    """鉴别诊断专家子 Agent"""
    return create_deep_agent(
        model=llm,
        system_prompt=DIFFERENTIAL_DIAGNOSIS_EXPERT_PROMPT,
        # 无需工具，直接基于知识库进行推理
    )

def create_treatment_expert(llm):
    """治则治法专家子 Agent"""
    return create_deep_agent(
        model=llm,
        system_prompt=TREATMENT_PRINCIPLE_EXPERT_PROMPT,
    )

def create_prescription_expert(llm):
    """方药推荐专家子 Agent"""
    return create_deep_agent(
        model=llm,
        system_prompt=PRESCRIPTION_RECOMMENDATION_EXPERT_PROMPT,
    )

def create_prognosis_expert(llm):
    """预后评估专家子 Agent"""
    return create_deep_agent(
        model=llm,
        system_prompt=PROGNOSIS_EVALUATION_EXPERT_PROMPT,
    )

def create_verification_expert(llm):
    """质疑验证专家子 Agent"""
    return create_deep_agent(
        model=llm,
        system_prompt=VERIFICATION_EXPERT_PROMPT,
    )


# ========== 3. 创建主 Agent（配置中间件）==========

def create_deep_search_agent(llm):
    """创建 DeepSearch Agent"""
    
    # 工具列表（供主 Agent 并行查询数据）
    data_tools = [
        kg_syndrome_search,
        case_vector_search,
        classics_search,
        web_search,
    ]
    
    # 中间件配置
    middleware = [
        # 1. 任务分解
        TodoListMiddleware(
            system_prompt="""
使用 write_todos 分解辨证任务：

第一步：并行查询数据（4个工具）
- kg_syndrome_search
- case_vector_search
- classics_search
- web_search

第二步：并行咨询专家（5个子 Agent）
- differential_diagnosis_expert
- treatment_principle_expert
- prescription_recommendation_expert
- prognosis_evaluation_expert
- verification_expert

第三步：综合所有信息给出最终结论
"""
        ),
        
        # 2. 专家子 Agent 并行
        SubAgentMiddleware(
            default_model=llm,
            subagents=[
                {
                    "name": "differential_diagnosis_expert",
                    "description": "鉴别诊断专家：区分相似证型，避免误诊",
                    "agent": create_differential_expert(llm)
                },
                {
                    "name": "treatment_principle_expert",
                    "description": "治则治法专家：制定治疗策略",
                    "agent": create_treatment_expert(llm)
                },
                {
                    "name": "prescription_recommendation_expert",
                    "description": "方药推荐专家：推荐方剂与食疗",
                    "agent": create_prescription_expert(llm)
                },
                {
                    "name": "prognosis_evaluation_expert",
                    "description": "预后评估专家：判断病情趋势",
                    "agent": create_prognosis_expert(llm)
                },
                {
                    "name": "verification_expert",
                    "description": "质疑验证专家：验证诊断合理性",
                    "agent": create_verification_expert(llm)
                },
            ]
        ),
    ]
    
    # 创建主 Agent
    agent = create_deep_agent(
        model=llm,
        tools=data_tools,  # 数据查询工具
        middleware=middleware,
        system_prompt="""
你是中医智能诊断助手（DeepSearch Agent）。

## 工作流程

1. **数据收集阶段**：
   - 并行调用4个数据工具获取知识库信息
   
2. **专家咨询阶段**：
   - 将数据提供给5位专家进行分析
   - 专家们会并行工作，各抒己见
   
3. **综合决策阶段**：
   - 综合5位专家的意见
   - 给出最终辨证结果和治疗建议

## 重要提示

- 先完成数据查询，再咨询专家（专家需要基于数据）
- 如果专家意见有冲突，说明需要重新评估
- 最终结论必须有充分的证据支持
"""
    )
    
    return agent


# ========== 4. 使用示例 ==========

async def main():
    agent = create_deep_search_agent(llm)
    
    result = await agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": "患者头痛、胸闷、失眠、腰膝酸软，请进行深度辨证分析"
        }]
    })
    
    print("最终辨证结果：")
    print(result["messages"][-1].content)
```

### 2.3 执行流程详解

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 任务分解（模型调用 #1）                              │
├─────────────────────────────────────────────────────────────┤
│ 输入: "患者头痛、胸闷、失眠、腰膝酸软"                      │
│ TodoListMiddleware 输出任务清单：                            │
│                                                             │
│ 【数据收集任务】（并行，无模型调用）                          │
│   - kg_syndrome_search(["头痛", "胸闷", "失眠", "腰膝酸软"]) │
│   - case_vector_search("头痛 胸闷 失眠")                     │
│   - classics_search(["头痛", "失眠"])                       │
│   - web_search("头痛 现代医学诊断")                         │
│                                                             │
│ 【专家咨询任务】（并行，5个子 Agent）                         │
│   - differential_diagnosis_expert (鉴别诊断)                │
│   - treatment_principle_expert (治则治法)                   │
│   - prescription_recommendation_expert (方药推荐)           │
│   - prognosis_evaluation_expert (预后评估)                  │
│   - verification_expert (质疑验证)                          │
└─────────────────────────────────────────────────────────────┘
         │
         │ SubAgentMiddleware 自动并行工具调用
         ├──────────────┬──────────────┬──────────────┐
         ▼              ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ kg_syndrome │ │ case_vector │ │  classics   │ │ web_search  │
│   _search   │ │   _search   │ │   _search   │ │   (Tavily)  │
├─────────────┤ ├─────────────┤ ├─────────────┤ ├─────────────┤
│  Neo4j 查询  │ │  向量检索    │ │  全文检索    │ │  HTTP请求   │
│             │ │             │ │             │ │             │
│ 返回：证型   │ │ 返回：医案   │ │ 返回：古籍   │ │ 返回：网页   │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
         │              │              │              │
         └──────────────┴──────────────┴──────────────┘
                        │
                        ▼ 数据汇总形成知识库
                 [knowledge_base]
                        │
                        │ 提供给专家层
                        ▼
         SubAgentMiddleware 自动并行专家
         ├──────────┬──────────┬──────────┬──────────┐
         ▼          ▼          ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│鉴别诊断  │ │治则治法  │ │方药推荐  │ │预后评估  │ │质疑验证  │
│专家      │ │专家      │ │专家      │ │专家      │ │专家      │
│(模型调用)│ │(模型调用)│ │(模型调用)│ │(模型调用)│ │(模型调用)│
├──────────┤ ├──────────┤ ├──────────┤ ├──────────┤ ├──────────┤
│使用已有  │ │使用已有  │ │使用已有  │ │使用已有  │ │使用已有  │
│提示词    │ │提示词    │ │提示词    │ │提示词    │ │提示词    │
│          │ │          │ │          │ │          │ │          │
│返回：    │ │返回：    │ │返回：    │ │返回：    │ │返回：    │
│鉴别分析  │ │治疗策略  │ │方药建议  │ │预后判断  │ │验证意见  │
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
         │          │          │          │          │
         └──────────┴──────────┴──────────┴──────────┘
                        │
                        ▼ 自动 Reduce（汇总专家意见）
┌─────────────────────────────────────────────────────────────┐
│ Step 2: 综合分析（模型调用 #2）                              │
├─────────────────────────────────────────────────────────────┤
│ 输入: 5位专家的分析报告                                      │
│                                                             │
│ 主 Agent 综合判断：                                          │
│ - 采纳鉴别诊断专家的证型判断                                  │
│ - 采纳治则治法专家的治疗策略                                  │
│ - 采纳方药推荐专家的方剂建议                                  │
│ - 考虑预后评估专家的风险提示                                  │
│ - 响应验证专家的质疑（如有）                                  │
│                                                             │
│ 输出: 证型: 肝郁脾虚 + 心肾不交（复合证）                     │
│       病因病机: ...                                          │
│       治法: ...                                              │
│       方药: ...                                              │
│       置信度: 0.87                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、Skills 系统设计

### 3.1 Skills 目录结构

```
backend/app/src/agent/skills/
├── complex_diagnosis_workflow.md      # 复杂辨证工作流程
├── syndrome_theory.md                 # 辨证理论要点
├── validation_criteria.md             # 验证标准
└── case_analysis_template.md          # 医案分析模板
```

### 3.2 Skills 使用原则

**Skills 适用场景**：
- ✅ **操作流程**（如"如何进行辨证"）
- ✅ **理论指导**（如"肝郁脾虚的特征"）
- ✅ **模板规范**（如"医案分析步骤"）

**Skills 不适用场景**：
- ❌ **大量结构化数据**（如 10000 个症状） → 用 Neo4j
- ❌ **动态查询需求**（如"查找相似案例"） → 用工具（Tools）

### 3.3 Skills 内容示例

#### `complex_diagnosis_workflow.md`

```markdown
# 复杂辨证工作流程

## 何时使用此技能
- 患者症状复杂，涉及多个脏腑
- 症状之间存在矛盾
- 需要深度分析病因病机

## 辨证步骤

### 步骤 1：症状分析
使用 `write_todos` 工具创建任务清单：
- 分析主症与兼症
- 识别症状矛盾点
- 判断涉及的脏腑系统

### 步骤 2：知识图谱查询（并行）
调用子 Agent `kg_searcher`：
- 查询症状-证型关联
- 查询脏腑归经
- 查询经络循行

**示例**：
```
请使用 kg_searcher 查询：
1. "头痛 + 胸闷" 的相关证型
2. 涉及的脏腑系统
```

### 步骤 3：医案检索（并行）
调用子 Agent `case_retriever`：
- 查找相似病例
- 提取辨证思路
- 对比治疗方案

### 步骤 4：古籍论证（并行）
调用子 Agent `classics_researcher`：
- 检索《伤寒论》相关条文
- 检索《金匮要略》相关论述
- 引用历代医家经验

### 步骤 5：综合分析
基于并行查询结果：
1. 确定主证与兼证
2. 分析病因病机
3. 确定治法治则
4. 评估置信度

### 步骤 6：验证迭代
使用 validation_criteria.md 中的标准：
- 与相似医案对比
- 检查理论一致性
- 评估置信度
- 如不合理，跳回步骤 5 重新分析

## 输出格式
```json
{
  "证型": "肝郁脾虚 + 心肾不交",
  "病因病机": "肝郁横逆犯脾，脾失健运；心火上炎，肾水不济...",
  "治法": "疏肝健脾，交通心肾",
  "方药建议": "逍遥散合交泰丸加减",
  "置信度": 0.87,
  "依据": {
    "知识图谱": "...",
    "相似医案": "...",
    "古籍依据": "..."
  }
}
```
```

#### `syndrome_theory.md`

```markdown
# 辨证理论要点

## 八纲辨证
- **表里**：病位深浅
- **寒热**：疾病性质
- **虚实**：正邪盛衰
- **阴阳**：总纲

## 脏腑辨证要点

### 肝
- **主疏泄**：情志、气血运行
- **藏血**：储藏和调节血量
- **常见证型**：肝郁气滞、肝阳上亢、肝血不足

### 脾
- **主运化**：水谷精微、水液
- **统血**：控制血液在脉中运行
- **常见证型**：脾气虚、脾阳虚、脾不统血

### 心
- **主血脉**：推动血液运行
- **藏神**：主管精神意识
- **常见证型**：心气虚、心血虚、心火亢盛

### 肾
- **藏精**：先天之本
- **主水**：调节水液代谢
- **常见证型**：肾阳虚、肾阴虚、肾精不足

## 复合证型识别

### 肝郁脾虚
- **症状组合**：胸胁胀满 + 腹胀便溏
- **病机**：肝气横逆犯脾
- **治法**：疏肝健脾

### 心肾不交
- **症状组合**：心烦失眠 + 腰膝酸软
- **病机**：心火上炎，肾水不济
- **治法**：交通心肾
```

#### `validation_criteria.md`

```markdown
# 验证标准

## 辨证结果验证清单

### 1. 症状匹配度检查
- [ ] 主症是否与证型对应？
- [ ] 兼症是否符合病机演变？
- [ ] 是否存在矛盾症状未解释？

### 2. 理论一致性检查
- [ ] 病因病机是否符合中医理论？
- [ ] 脏腑关系是否合理？
- [ ] 是否符合八纲辨证？

### 3. 医案对比验证
- [ ] 是否有相似案例支持？
- [ ] 治疗效果如何？
- [ ] 置信度评估：
  - 0.9+：有多个相似案例，理论完全符合
  - 0.7-0.9：有部分案例支持，理论基本符合
  - <0.7：案例支持不足，需人工复核

### 4. 古籍依据验证
- [ ] 是否有经典条文支持？
- [ ] 历代医家经验是否一致？

## 不合理情况处理

如果验证不通过：
1. 记录不合理点
2. 跳回综合分析步骤
3. 调整辨证思路
4. 重新验证（最多迭代 3 次）
```

---

## 四、中间件详细配置

### 4.1 TodoListMiddleware（任务规划）

**功能**：自动将复杂辨证任务分解为离散步骤，跟踪进度。

```python
TodoListMiddleware(
    system_prompt="""
使用 write_todos 工具分解复杂辨证任务：

必选任务：
1. 分析主症与兼症关系
2. 判断涉及的脏腑系统

可选任务（根据复杂度）：
3. 查询知识图谱证型（kg_searcher）
4. 检索相似医案（case_retriever）
5. 检索古籍论述（classics_researcher）

必选任务：
6. 综合分析判断证型
7. 验证辨证结果合理性

任务状态：
- pending: 待执行
- in_progress: 执行中
- completed: 已完成
"""
)
```

**优势**：
- ✅ 自动任务分解
- ✅ 动态调整计划（根据新信息）
- ✅ 状态跟踪（pending → in_progress → completed）

### 4.2 FilesystemMiddleware（上下文管理）

**功能**：自动管理大型查询结果，防止上下文溢出。

```python
FilesystemMiddleware(
    backend=lambda rt: CompositeBackend(
        default=StateBackend(rt),  # 临时文件（当前会话）
        routes={
            "/kg_results/": StateBackend(rt),      # 知识图谱结果（临时）
            "/case_library/": StoreBackend(rt),    # 医案库（持久化）
            "/classics/": StoreBackend(rt),        # 古籍库（持久化）
            "/patient_history/": StoreBackend(rt)  # 患者档案（持久化）
        }
    ),
    tool_token_limit_before_evict=15000,  # 超过 15k tokens 自动写入文件
    system_prompt="""
当工具返回大量结果时：
1. 使用 write_file 保存完整结果到对应目录
2. 仅在上下文中保留摘要（前 200 tokens）
3. 需要详细内容时使用 read_file 读取

文件路径规则：
- /kg_results/task_{id}.json: 知识图谱查询结果
- /case_library/case_{id}.json: 相似医案详情
- /classics/{book}_{chapter}.txt: 古籍原文
- /patient_history/{user_id}.json: 患者历史档案
"""
)
```

**工作流程**：
1. 监控工具调用结果大小
2. 超过 15k tokens 时触发驱逐
3. 将完整结果写入文件系统
4. 在上下文中保留摘要 + 文件引用
5. Agent 按需使用 `read_file` 工具读取详细内容

**示例**：
```
# 查询结果过大
kg_result = {
    "nodes": [100个节点],  # 50k tokens
    "relationships": [200个关系]
}

# 自动驱逐
write_file("/kg_results/task1.json", kg_result)

# 上下文中只保留摘要
context_summary = """
知识图谱查询结果已保存到 /kg_results/task1.json
摘要：
- 找到 15 个相关证型
- 涉及 5 个脏腑
- 详见文件获取完整内容
"""
```

### 4.3 SubAgentMiddleware（并行子 Agent）

**功能**：注册专业化子 Agent，自动并行调度。

```python
SubAgentMiddleware(
    default_model="gpt-4o",
    subagents=[
        {
            "name": "kg_searcher",
            "description": "查询 Neo4j 知识图谱（症状-证型关系、脏腑归经）",
            "system_prompt": """
你是知识图谱查询专家，擅长从 Neo4j 中提取症状-证型关联。

可用工具：
- kg_syndrome_search: 查询症状对应的证型
- kg_organ_query: 查询脏腑归经关系

输出格式：
{
  "syndromes": ["证型1", "证型2"],
  "organs": ["脏腑1", "脏腑2"],
  "confidence": 0.85
}
""",
            "tools": [kg_syndrome_search, kg_organ_query],
            "model": "gpt-4o-mini"  # 使用小模型降低成本
        },
        {
            "name": "case_retriever",
            "description": "检索相似医案并提取辨证思路",
            "system_prompt": """
你是医案检索专家，从历史案例中找到相似症状并总结辨证逻辑。

可用工具：
- case_vector_search: 向量检索相似医案

分析要点：
1. 症状相似度
2. 辨证思路
3. 治疗方案
4. 疗效反馈

输出格式：
{
  "similar_cases": [
    {
      "id": "case_123",
      "similarity": 0.92,
      "syndrome": "肝郁脾虚",
      "treatment": "逍遥散加减",
      "outcome": "显效"
    }
  ]
}
""",
            "tools": [case_vector_search]
        },
        {
            "name": "classics_researcher",
            "description": "查找中医古籍相关论述",
            "system_prompt": """
你是中医文献专家，从《伤寒论》《金匮要略》等经典中找到理论依据。

可用工具：
- classics_search: 检索古籍原文

引用格式：
《书名·章节》："原文内容"
解析：理论意义

输出格式：
{
  "citations": [
    {
      "book": "伤寒论",
      "chapter": "辨太阳病脉证并治",
      "text": "太阳之为病，脉浮，头项强痛而恶寒。",
      "interpretation": "..."
    }
  ]
}
""",
            "tools": [classics_search]
        }
    ]
)
```

**并行调用示例**：
```python
# 主 Agent 自动识别并并行调用
result = await agent.ainvoke({
    "messages": [{
        "role": "user",
        "content": """
        患者：头痛、胸闷、失眠、腰膝酸软
        
        请并行执行：
        1. kg_searcher 查询证型
        2. case_retriever 查找相似案例
        3. classics_researcher 检索理论依据
        
        然后综合分析给出辨证结果。
        """
    }]
})

# 内部执行流程（自动）：
# asyncio.gather(
#     kg_searcher.ainvoke(...),
#     case_retriever.ainvoke(...),
#     classics_researcher.ainvoke(...)
# )
```

### 4.4 SummarizationMiddleware（对话压缩）

**功能**：自动压缩对话历史，支持超长病例。

```python
SummarizationMiddleware(
    model="gpt-4o-mini",  # 使用小模型摘要（降低成本）
    trigger=("tokens", 4000),  # 超过 4000 tokens 触发摘要
    keep=("messages", 10),      # 保留最近 10 条消息
    summary_prompt="""
总结患者的诊疗历史，保留关键信息：

必须保留：
1. 主要症状及变化趋势
2. 历史诊断和证型判断
3. 治疗方案及效果
4. 重要的既往病史

可省略：
5. 重复的问诊内容
6. 中间的诊断过程
"""
)
```

**应用场景**：
- 患者多次就诊，历史对话很长
- 防止上下文窗口溢出
- 降低 token 成本

**工作原理**：
```
对话历史 50 条消息（6000 tokens）
    │
    │ 触发摘要（超过 4000 tokens）
    ▼
保留最近 10 条消息（1000 tokens）
+ 历史摘要（500 tokens）
    │
    ▼
压缩后 1500 tokens
```



## 五、工具（Tools）设计

### 5.1 工具分类

| 工具类别 | 工具名称 | 所属子 Agent | 数据源 |
|---------|---------|-------------|--------|
| **知识图谱工具** | `kg_syndrome_search` | kg_searcher | Neo4j |
|  | `kg_organ_query` | kg_searcher | Neo4j |
| **向量检索工具** | `case_vector_search` | case_retriever | Milvus/Chroma |
| **古籍检索工具** | `classics_search` | classics_researcher | 全文检索 |
|  | `classics_citation` | classics_researcher | 全文检索 |
| **网络搜索工具** | `web_search` | web_researcher | Tavily/SerpAPI |
|  | `medical_research_search` | web_researcher | 医学数据库 |

### 5.2 并行 Web Search 设计

**应用场景**：
- 查询最新医学研究（如"某某症状的最新治疗进展"）
- 中西医结合参考（如"头痛的现代医学诊断标准"）
- 罕见病例参考（本地医案库未覆盖）

**并行机制**：
```python
# 创建 Web 研究子 Agent
web_subagent = create_deep_agent(
    model="gpt-4o-mini",
    tools=[web_search, medical_research_search],
    system_prompt="""
你是医学文献研究专家，擅长从互联网检索最新医学研究。

可并行搜索：
1. 症状的现代医学解释
2. 中西医结合治疗案例
3. 最新临床研究进展

输出格式：
{
  "modern_medicine": "现代医学观点",
  "integrated_cases": ["案例1", "案例2"],
  "latest_research": "最新研究摘要"
}
"""
)

# 注册到主 Agent
SubAgentMiddleware(
    subagents=[
        {"name": "kg_searcher", "agent": kg_subagent},
        {"name": "case_retriever", "agent": case_subagent},
        {"name": "classics_researcher", "agent": classics_subagent},
        {"name": "web_researcher", "agent": web_subagent},  # ← 新增
    ]
)

# ★★★ 自动并行搜索 ★★★
result = await agent.ainvoke({
    "messages": [{
        "role": "user",
        "content": """
        患者头痛反复发作2年，请：
        1. kg_searcher 查询中医证型
        2. case_retriever 检索相似医案
        3. web_researcher 查询现代医学头痛分类
        
        并行执行，然后综合中西医观点给出建议。
        """
    }]
})
```

**并行执行流程**：
```
用户问题
    │
    ▼
[TodoListMiddleware] 分解任务
    │
    ├─ Task 1: 中医证型（kg_searcher）
    ├─ Task 2: 相似医案（case_retriever）
    ├─ Task 3: 现代医学（web_researcher）
    │
    │ SubAgentMiddleware 自动并行
    ├──────────────┬──────────────┬────────────────┐
    ▼              ▼              ▼                │
[kg_searcher]  [case_retriever]  [web_researcher]  │
    │              │              │                │
 Neo4j查询       向量检索      并行Web搜索         │
    │              │              │                │
    │              │              ├─ Search 1: "头痛 中医证型"
    │              │              ├─ Search 2: "头痛 现代医学"
    │              │              └─ Search 3: "头痛 治疗指南"
    │              │              │                │
    └──────────────┴──────────────┴────────────────┘
                    │
                    ▼
          综合中西医观点，给出辨证建议
```

### 5.3 工具实现模板

#### `web_search`（网络搜索）

```python
from langchain.tools import tool
from tavily import TavilyClient
import os

tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

@tool
async def web_search(
    query: str,
    max_results: int = 5,
    include_domains: Optional[List[str]] = None
) -> Dict:
    """
    网络搜索工具（支持并行）
    
    Args:
        query: 搜索查询
        max_results: 返回结果数
        include_domains: 限定域名（如医学网站）
    
    Returns:
        {
            "results": [
                {
                    "title": "标题",
                    "url": "链接",
                    "content": "摘要",
                    "score": 0.95
                }
            ]
        }
    """
    # Tavily 搜索
    response = await tavily_client.search(
        query=query,
        max_results=max_results,
        include_domains=include_domains or [],
        include_raw_content=False
    )
    
    return {
        "results": [
            {
                "title": item["title"],
                "url": item["url"],
                "content": item["content"],
                "score": item.get("score", 0.0)
            }
            for item in response.get("results", [])
        ]
    }


@tool
async def medical_research_search(
    query: str,
    databases: List[str] = ["pubmed", "cnki"]
) -> Dict:
    """
    医学数据库搜索（中英文文献）
    
    Args:
        query: 搜索查询（支持中英文）
        databases: 数据库列表
    
    Returns:
        {
            "papers": [
                {
                    "title": "论文标题",
                    "authors": "作者",
                    "abstract": "摘要",
                    "year": 2025,
                    "database": "pubmed"
                }
            ]
        }
    """
    # 实现医学数据库检索（PubMed、CNKI等）
    # 这里使用 Tavily 作为替代
    response = await tavily_client.search(
        query=f"{query} site:pubmed.ncbi.nlm.nih.gov OR site:cnki.net",
        max_results=5
    )
    
    return {
        "papers": [
            {
                "title": item["title"],
                "abstract": item["content"],
                "url": item["url"],
                "database": "pubmed" if "pubmed" in item["url"] else "cnki"
            }
            for item in response.get("results", [])
        ]
    }
```

#### `kg_syndrome_search`（知识图谱证型查询）

```python
from langchain.tools import tool
from typing import List, Dict

@tool
async def kg_syndrome_search(
    symptoms: List[str],
    min_match_count: int = 2
) -> Dict:
    """
    从 Neo4j 知识图谱查询症状对应的证型
    
    Args:
        symptoms: 症状列表，如 ["头痛", "胸闷", "失眠"]
        min_match_count: 最少匹配症状数
    
    Returns:
        {
            "syndromes": [
                {
                    "name": "肝郁脾虚",
                    "matched_symptoms": ["头痛", "胸闷"],
                    "confidence": 0.85,
                    "description": "..."
                }
            ]
        }
    """
    # Cypher 查询
    query = """
    MATCH (s:Symptom)-[:INDICATES]->(syn:Syndrome)
    WHERE s.name IN $symptoms
    WITH syn, COUNT(s) as match_count
    WHERE match_count >= $min_match_count
    RETURN syn.name as syndrome,
           syn.description as description,
           match_count,
           toFloat(match_count) / size($symptoms) as confidence
    ORDER BY confidence DESC
    LIMIT 10
    """
    
    results = await neo4j_graph.aquery(
        query,
        params={"symptoms": symptoms, "min_match_count": min_match_count}
    )
    
    return {
        "syndromes": [
            {
                "name": r["syndrome"],
                "matched_symptoms": symptoms[:r["match_count"]],
                "confidence": r["confidence"],
                "description": r["description"]
            }
            for r in results
        ]
    }
```

#### `case_vector_search`（医案向量检索）

```python
@tool
async def case_vector_search(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.7
) -> Dict:
    """
    从向量数据库检索相似医案
    
    Args:
        query: 查询文本（症状描述）
        top_k: 返回最相似的 k 个案例
        similarity_threshold: 相似度阈值
    
    Returns:
        {
            "similar_cases": [
                {
                    "case_id": "case_123",
                    "similarity": 0.92,
                    "patient_info": "男，35岁",
                    "chief_complaint": "头痛3天",
                    "syndrome": "肝郁脾虚",
                    "treatment": "逍遥散加减",
                    "outcome": "显效"
                }
            ]
        }
    """
    # 向量检索
    query_embedding = await embedding_model.aembed_query(query)
    
    results = await vector_store.asimilarity_search_with_score(
        query_embedding,
        k=top_k,
        score_threshold=similarity_threshold
    )
    
    return {
        "similar_cases": [
            {
                "case_id": doc.metadata["case_id"],
                "similarity": score,
                "patient_info": doc.metadata["patient_info"],
                "chief_complaint": doc.metadata["chief_complaint"],
                "syndrome": doc.metadata["syndrome"],
                "treatment": doc.metadata["treatment"],
                "outcome": doc.metadata.get("outcome", "未知")
            }
            for doc, score in results
        ]
    }
```

#### `classics_search`（古籍检索）

```python
@tool
async def classics_search(
    keywords: List[str],
    books: List[str] = ["伤寒论", "金匮要略"],
    max_results: int = 5
) -> Dict:
    """
    从中医古籍中检索相关论述
    
    Args:
        keywords: 关键词列表，如 ["头痛", "肝郁"]
        books: 检索的书籍列表
        max_results: 最多返回结果数
    
    Returns:
        {
            "citations": [
                {
                    "book": "伤寒论",
                    "chapter": "辨太阳病脉证并治",
                    "section": "第5条",
                    "text": "太阳之为病，脉浮，头项强痛而恶寒。",
                    "keywords_matched": ["头痛"]
                }
            ]
        }
    """
    # 全文检索（Elasticsearch）
    results = await classics_index.asearch(
        query={
            "bool": {
                "should": [
                    {"match": {"text": keyword}} for keyword in keywords
                ],
                "filter": {"terms": {"book": books}}
            }
        },
        size=max_results
    )
    
    return {
        "citations": [
            {
                "book": hit["_source"]["book"],
                "chapter": hit["_source"]["chapter"],
                "section": hit["_source"]["section"],
                "text": hit["_source"]["text"],
                "keywords_matched": [
                    kw for kw in keywords
                    if kw in hit["_source"]["text"]
                ]
            }
            for hit in results["hits"]["hits"]
        ]
    }
```

---

## 六、文件结构与实施步骤

### 6.1 目录结构

```
backend/app/src/agent/components/diagnose/nodes/complex/
├── __init__.py                    # 模块导出
├── IMPLEMENTATION_PLAN.md         # 本文档
│
├── deep_search_agent.py           # 主 Agent 创建
├── deep_search_node.py            # 诊断节点（集成到图）
│
├── subagents/                     # 子 Agent 模块
│   ├── __init__.py
│   ├── kg_subagent.py             # 知识图谱子 Agent
│   ├── case_subagent.py           # 医案检索子 Agent
│   └── classics_subagent.py       # 古籍检索子 Agent
│
├── tools/                         # 工具模块
│   ├── __init__.py
│   ├── kg_tools.py                # Neo4j 工具
│   ├── vector_tools.py            # 向量检索工具
│   └── classics_tools.py          # 古籍检索工具
│
├── middleware/                    # 自定义中间件
│   ├── __init__.py
│   └── custom_middleware.py       # 自定义中间件（如需要）
│
└── skills/                        # Skills 知识库
    ├── complex_diagnosis_workflow.md
    ├── syndrome_theory.md
    ├── validation_criteria.md
    └── case_analysis_template.md
```

### 6.2 实施步骤（9个任务）

#### **Task 4.1：创建 Skills 知识库**

**文件**：
- `skills/complex_diagnosis_workflow.md`
- `skills/syndrome_theory.md`
- `skills/validation_criteria.md`

**内容**：参见第三节"Skills 系统设计"

**验收标准**：
- [ ] 3 个 Skill 文件创建完成
- [ ] 内容符合中医辨证理论
- [ ] Markdown 格式正确

---

#### **Task 4.2：创建子 Agent**

**文件**：
- `subagents/kg_subagent.py`
- `subagents/case_subagent.py`
- `subagents/classics_subagent.py`

**实现示例**（`kg_subagent.py`）：

```python
"""
知识图谱子 Agent

专门查询 Neo4j 知识图谱
"""

from typing import Any
from langchain_core.language_models import BaseChatModel
from deepagents import create_deep_agent

from ..tools.kg_tools import kg_syndrome_search, kg_organ_query


def create_kg_subagent(llm: BaseChatModel) -> Any:
    """
    创建知识图谱查询子 Agent
    
    功能：
    - 查询症状-证型关系
    - 查询脏腑归经
    - 查询经络循行
    
    Args:
        llm: 语言模型
        
    Returns:
        kg_subagent 实例
    """
    return create_deep_agent(
        model=llm,
        tools=[kg_syndrome_search, kg_organ_query],
        system_prompt="""
你是知识图谱查询专家，擅长从 Neo4j 数据库中提取中医症状-证型关联。

## 可用工具

### kg_syndrome_search
根据症状列表查询相关证型。
使用场景：患者主诉多个症状，需要匹配可能的证型。

### kg_organ_query
查询症状涉及的脏腑系统。
使用场景：分析病位，判断脏腑关系。

## 输出格式

```json
{
  "syndromes": [
    {
      "name": "证型名称",
      "confidence": 0.85,
      "matched_symptoms": ["症状1", "症状2"]
    }
  ],
  "organs": ["脏腑1", "脏腑2"],
  "explanation": "简要分析"
}
```

## 注意事项
- 证型名称必须标准化（如"肝郁脾虚"而非"肝郁加脾虚"）
- 置信度基于匹配症状比例
- 必须提供理论依据
"""
    )
```

**验收标准**：
- [ ] 3 个子 Agent 创建完成
- [ ] 每个子 Agent 有清晰的职责
- [ ] System Prompt 符合中医专业要求

---

#### **Task 4.3：配置中间件栈**

**文件**：`deep_search_agent.py`

**核心代码**：

```python
"""
DeepSearch Agent 主入口
"""

from typing import Dict, Any, Optional
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

from langchain.agents.middleware import (
    TodoListMiddleware,
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
)

from ...config import diagnose_config
from app.src.utils import get_logger, get_llm

from .subagents import (
    create_kg_subagent,
    create_case_subagent,
    create_classics_subagent,
)

logger = get_logger("deep_search_agent")


def create_deep_search_agent(
    llm: Optional[BaseChatModel] = None,
    enable_doctor_approval: bool = False,
    checkpointer: Optional[Any] = None,
    store: Optional[Any] = None,
) -> Any:
    """
    创建 DeepSearch Agent
    
    Args:
        llm: 语言模型实例
        enable_doctor_approval: 是否启用医生审批
        checkpointer: 状态持久化
        store: 长期存储
        
    Returns:
        DeepSearch Agent 实例
    """
    # 1. 初始化组件
    if llm is None:
        llm = get_llm(temperature=diagnose_config.DIAGNOSIS_TEMPERATURE)
    
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    if store is None:
        store = InMemoryStore()
    
    # 2. 创建子 Agent
    logger.info("创建子 Agent...")
    kg_subagent = create_kg_subagent(llm)
    case_subagent = create_case_subagent(llm)
    classics_subagent = create_classics_subagent(llm)
    
    # 3. Skills 路径
    import os
    skills_path = os.path.join(os.path.dirname(__file__), "skills")
    
    # 4. 构建中间件栈
    logger.info("构建中间件栈...")
    middleware_stack = [
        # (1) 任务规划与跟踪
        TodoListMiddleware(
            system_prompt="""
使用 write_todos 工具分解复杂辨证任务：

必选任务：
1. 分析主症与兼症关系
2. 判断涉及的脏腑系统

可选任务（根据复杂度，可并行执行）：
3. 查询知识图谱证型（kg_searcher）
4. 检索相似医案（case_retriever）
5. 检索古籍论述（classics_researcher）

必选任务：
6. 综合分析判断证型
7. 验证辨证结果合理性

任务状态管理：
- pending: 待执行
- in_progress: 执行中
- completed: 已完成
"""
        ),
        
        # (2) 上下文管理（大结果自动驱逐）
        FilesystemMiddleware(
            backend=lambda rt: CompositeBackend(
                default=StateBackend(rt),
                routes={
                    "/kg_results/": StateBackend(rt),
                    "/case_library/": StoreBackend(rt),
                    "/classics/": StoreBackend(rt),
                    "/patient_history/": StoreBackend(rt)
                }
            ),
            tool_token_limit_before_evict=15000,
            system_prompt="""
当工具返回大量结果（超过 15k tokens）时：
1. 使用 write_file 保存完整结果
2. 上下文中只保留摘要
3. 需要详细内容时使用 read_file 读取

文件路径规则：
- /kg_results/task_{id}.json: 知识图谱查询结果
- /case_library/case_{id}.json: 相似医案详情
- /classics/{book}_{chapter}.txt: 古籍原文
- /patient_history/{user_id}.json: 患者历史档案
"""
        ),
        
        # (3) 并行子 Agent 调度
        SubAgentMiddleware(
            default_model=llm,
            subagents=[
                {
                    "name": "kg_searcher",
                    "description": "查询 Neo4j 知识图谱（症状-证型关系、脏腑归经）",
                    "agent": kg_subagent
                },
                {
                    "name": "case_retriever",
                    "description": "检索相似医案并提取辨证思路",
                    "agent": case_subagent
                },
                {
                    "name": "classics_researcher",
                    "description": "查找中医古籍相关论述",
                    "agent": classics_subagent
                }
            ]
        ),
        
        # (4) 对话历史压缩
        SummarizationMiddleware(
            model=get_llm(model_name="gpt-4o-mini"),  # 使用小模型降低成本
            trigger=("tokens", 4000),
            keep=("messages", 10),
            summary_prompt="""
总结患者的诊疗历史，保留：
1. 主要症状及变化趋势
2. 历史诊断和证型判断
3. 治疗方案及效果
4. 重要的既往病史
"""
        ),
    ]
    
    # (5) 可选：医生审批
    if enable_doctor_approval:
        middleware_stack.append(
            HumanInTheLoopMiddleware(
                interrupt_on={"synthesizer": True},
                description_prefix="请医生审核以下辨证结果：",
            )
        )
    
    # 5. 创建主 Agent
    logger.info("创建 DeepSearch Agent...")
    agent = create_deep_agent(
        model=llm,
        checkpointer=checkpointer,
        store=store,
        skills=[skills_path],  # 加载 Skills
        middleware=middleware_stack,
        system_prompt="""
你是中医智能诊断助手，专注于复杂病例的辨证分析。

## 核心能力
1. 并行查询：同时调用 kg_searcher、case_retriever、classics_researcher
2. 深度分析：综合知识图谱、医案、古籍进行辨证
3. 迭代验证：自动验证辨证结果的合理性

## 工作流程
参考 Skills 中的 complex_diagnosis_workflow.md

## 输出标准
必须包含：
- 证型（主证 + 兼证）
- 病因病机
- 治法治则
- 方药建议
- 置信度（0-1）
- 依据（知识图谱/医案/古籍）
"""
    )
    
    logger.info("DeepSearch Agent 创建完成")
    return agent
```

**验收标准**：
- [ ] 5 个中间件正确配置
- [ ] 子 Agent 成功注册
- [ ] Skills 路径正确

---

#### **Task 4.4：实现并行查询工作流**

**说明**：并行由 `SubAgentMiddleware` 自动处理，无需额外实现。

**测试代码**（`tests/test_parallel.py`）：

```python
import pytest
import asyncio
from ..deep_search_agent import create_deep_search_agent

@pytest.mark.asyncio
async def test_parallel_execution():
    """测试并行查询"""
    agent = create_deep_search_agent()
    
    result = await agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": """
            患者：男，35岁
            主诉：头痛3天，伴胸闷、失眠
            
            请并行执行：
            1. kg_searcher 查询相关证型
            2. case_retriever 检索相似医案
            3. classics_researcher 检索古籍依据
            
            然后给出辨证结果。
            """
        }]
    })
    
    # 验证并行执行痕迹
    assert "kg_searcher" in str(result.get("steps", []))
    assert "case_retriever" in str(result.get("steps", []))
    assert "classics_researcher" in str(result.get("steps", []))
```

**验收标准**：
- [ ] 测试通过
- [ ] 确认 3 个子 Agent 并行执行
- [ ] 执行时间明显少于串行

---

#### **Task 4.5-4.7：对接工具**

**实现优先级**：
1. **4.5 Neo4j 工具**（最优先）
2. **4.6 向量检索工具**
3. **4.7 古籍检索工具**

**实现模板**参见第五节"工具（Tools）设计"。

**验收标准**：
- [ ] 每个工具单元测试通过
- [ ] 工具描述符合 LangChain 规范
- [ ] 错误处理完善

---

#### **Task 4.8：实现医生审批机制**

**配置**：在 `create_deep_search_agent()` 中设置 `enable_doctor_approval=True`

**集成到图**（`deep_search_node.py`）：

```python
async def deep_search_diagnosis(state: DiagnoseOverallState) -> Dict[str, Any]:
    """
    复杂辨证节点（集成医生审批）
    """
    # 创建 Agent（启用审批）
    agent = create_deep_search_agent(enable_doctor_approval=True)
    
    # 生成 thread_id
    thread_id = f"diagnosis_{state.get('user_id')}_{int(time.time())}"
    config = {"configurable": {"thread_id": thread_id}}
    
    # 执行辨证
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config
    )
    
    # 检查是否中断（等待审批）
    if "__interrupt__" in result:
        # 保存中断状态，等待医生操作
        return {
            "pending_approval": True,
            "diagnosis_draft": result["__interrupt__"],
            "thread_id": thread_id,
            "steps": ["复杂辨证: 等待医生审批"]
        }
    
    # 已批准，返回结果
    return {
        "answer": result["messages"][-1].content,
        "steps": ["复杂辨证: 完成"]
    }
```

**审批接口**（`app/src/controller/diagnosis_approval.py`）：

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class ApprovalRequest(BaseModel):
    thread_id: str
    decision: str  # "approve" | "edit" | "reject"
    edited_diagnosis: Optional[Dict] = None
    reject_message: Optional[str] = None

@router.post("/diagnosis/approve")
async def approve_diagnosis(request: ApprovalRequest):
    """医生审批接口"""
    from langgraph.types import Command
    
    agent = create_deep_search_agent()
    config = {"configurable": {"thread_id": request.thread_id}}
    
    if request.decision == "approve":
        result = await agent.ainvoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config
        )
    elif request.decision == "edit":
        result = await agent.ainvoke(
            Command(resume={"decisions": [{
                "type": "edit",
                "edited_diagnosis": request.edited_diagnosis
            }]}),
            config=config
        )
    elif request.decision == "reject":
        result = await agent.ainvoke(
            Command(resume={"decisions": [{
                "type": "reject",
                "message": request.reject_message
            }]}),
            config=config
        )
    
    return {"status": "success", "result": result}
```

**验收标准**：
- [ ] 中断机制正常工作
- [ ] 审批接口测试通过
- [ ] 支持 approve/edit/reject 三种决策

---

#### **Task 4.9：集成测试与性能优化**

**集成测试用例**：

```python
@pytest.mark.asyncio
async def test_deep_search_full_workflow():
    """完整工作流测试"""
    agent = create_deep_search_agent()
    
    # 1. 复杂病例
    result = await agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": """
            患者：女，42岁
            主诉：头痛反复发作2年，近1月加重
            症状：头痛如裹，胸胁胀满，食欲不振，腹胀便溏，
                  失眠多梦，腰膝酸软，畏寒肢冷
            舌象：舌质淡红，苔薄白
            脉象：弦细
            
            请进行深度辨证分析。
            """
        }]
    })
    
    # 2. 验证输出
    answer = result["messages"][-1].content
    assert "证型" in answer
    assert "病因病机" in answer
    assert "治法" in answer
    assert "置信度" in answer
    
    # 3. 验证并行执行
    steps = result.get("steps", [])
    assert any("kg_searcher" in step for step in steps)
    assert any("case_retriever" in step for step in steps)
```

**性能优化检查**：
- [ ] 并行查询时间 < 串行时间的 50%
- [ ] 大结果（>15k tokens）自动驱逐
- [ ] 对话历史（>4k tokens）自动压缩
- [ ] 平均响应时间 < 30 秒

---

## 七、技术规范

### 7.1 代码规范

1. **类型注解**：所有函数必须有完整的类型注解
2. **文档字符串**：所有公开类和函数必须有 docstring
3. **错误处理**：使用自定义异常类，不吞没异常
4. **日志**：使用 Python logging，不使用 print

### 7.2 日志规范

```python
logger.info(f"创建 DeepSearch Agent，enable_doctor_approval={enable_doctor_approval}")
logger.debug(f"子 Agent 注册：kg_searcher, case_retriever, classics_researcher")
logger.warning(f"查询结果超过阈值（{tokens} tokens），触发驱逐")
logger.error(f"辨证分析失败: {e}", exc_info=True)
```

### 7.3 测试规范

- 单元测试覆盖率 > 80%
- 集成测试覆盖主要流程
- 性能测试验证并行优势

---

## 八、依赖安装

```bash
# 安装 DeepAgents
pip install deepagents

# 安装 LangChain 最新版
pip install langchain>=0.3.0 langchain-core langgraph

# 安装数据库驱动
pip install neo4j pymilvus elasticsearch

# 安装其他依赖
pip install pydantic>=2.0 asyncio
```

---

## 九、常见问题 (FAQ)

### Q1: Skills 什么时候加载？
**A**: Agent 根据任务内容自动判断并加载相关 Skill，无需手动指定。

### Q2: 并行查询如何保证数据一致性？
**A**: 每个子 Agent 在独立上下文中执行，结果汇总时由主 Agent 统一处理。

### Q3: 如何调试并行执行？
**A**: 查看 `result["steps"]`，每个子 Agent 会记录执行日志。

### Q4: 如何动态增加子 Agent？
**A**: 在 `SubAgentMiddleware` 的 `subagents` 列表中添加新配置即可。

### Q5: 医生审批会阻塞其他用户吗？
**A**: 不会，每个诊断会话有独立的 `thread_id`，互不影响。

---

## 十、参考文档

- **LangChain 文档**: https://docs.langchain.com/oss/python/langchain/overview
- **DeepAgents 文档**: https://docs.langchain.com/oss/python/deepagents/overview
- **LangGraph 文档**: https://docs.langchain.com/oss/python/langgraph/overview
- **Teacher 架构文档**: `../../TEACHER_ARCHITECTURE.md`
- **LangChain 最新特性**: `../../../../docs/LangChain_DeepAgents_最新特性总结_2026.md`

---

## 附录：完整示例代码

### 示例：创建并调用 DeepSearch Agent

```python
import asyncio
from .deep_search_agent import create_deep_search_agent

async def main():
    # 创建 Agent
    agent = create_deep_search_agent(enable_doctor_approval=False)
    
    # 复杂病例
    result = await agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": """
            患者：女，42岁
            主诉：头痛反复发作2年，近1月加重
            症状：头痛如裹，胸胁胀满，食欲不振，腹胀便溏，
                  失眠多梦，腰膝酸软，畏寒肢冷
            舌象：舌质淡红，苔薄白
            脉象：弦细
            
            请进行深度辨证分析。
            """
        }]
    })
    
    print("辨证结果：")
    print(result["messages"][-1].content)
    
    print("\n执行步骤：")
    for step in result.get("steps", []):
        print(f"  - {step}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

**更新记录**：
- 2026-02-05: 初始版本，基于 DeepAgents 框架设计
