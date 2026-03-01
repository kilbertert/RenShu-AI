"""
DeepSearch Agent 主入口

基于 LangChain DeepAgents 框架的复杂辨证系统
使用中间件实现并行子 Agent 调度和上下文管理

中间件架构（2026-02-05 升级）：
- 稳定性: ModelRetry, ToolRetry, ModelFallback
- 成本控制: ModelCallLimit, ToolCallLimit
- 上下文: Filesystem, Summarization
- 子Agent: SubAgentMiddleware
- 任务管理: TodoListMiddleware
"""

from typing import Dict, Any, Optional, List
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
    # 稳定性中间件（2026-02-05 新增）
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ModelFallbackMiddleware,
    ToolRetryMiddleware,
    ModelRetryMiddleware,
)

from app.src.agent.tcm_builder import get_llm
from app.src.agent.components.diagnose.config import diagnose_config
from app.src.utils import get_logger

from .subagents import (
    create_differential_expert,
    create_treatment_expert,
    create_prescription_expert,
    create_prognosis_expert,
    create_verification_expert,
)

# 导入专家提示词
from app.src.agent.components.diagnose.prompts.deepsearch_prompts import (
    DIFFERENTIAL_DIAGNOSIS_EXPERT_PROMPT,
    TREATMENT_PRINCIPLE_EXPERT_PROMPT,
    PRESCRIPTION_RECOMMENDATION_EXPERT_PROMPT,
    PROGNOSIS_EVALUATION_EXPERT_PROMPT,
    VERIFICATION_EXPERT_PROMPT,
)

from .tools import (
    kg_syndrome_search,
    case_vector_search,
    classics_search,
    web_search,
)

logger = get_logger("deep_search_agent")


# ============================================================
# DeepSearch Agent 系统提示词（精简版）
# ============================================================

# 注意：详细的工作流程、辨证理论、验证标准等内容通过 Skills 渐进式加载
# Skills 目录：./skills/
#   - complex-diagnosis-workflow: 复杂辨证工作流程
#   - syndrome-theory: 辨证理论要点
#   - validation-criteria: 验证标准

DEEP_SEARCH_SYSTEM_PROMPT = """
你是中医智能诊断助手（DeepSearch Agent），专注于复杂病例的深度辨证分析。

## 核心能力

- 并行数据查询：知识图谱、医案、古籍、网络搜索
- 并行专家咨询：5 位专家子 Agent 协同分析
- 综合决策：汇总专家意见，给出最终诊断

## 重要提示

- 复杂辨证请参考 complex-diagnosis-workflow 技能
- 辨证理论请参考 syndrome-theory 技能
- 验证标准请参考 validation-criteria 技能
- 本系统不开具处方，仅提供参考建议
"""


# ============================================================
# 创建 DeepSearch Agent
# ============================================================

def create_deep_search_agent(
    llm: Optional[BaseChatModel] = None,
    enable_doctor_approval: bool = False,
    checkpointer: Optional[Any] = None,
    store: Optional[Any] = None,
) -> Any:
    """
    创建 DeepSearch Agent
    
    基于 LangChain DeepAgents 框架，配置：
    
    稳定性中间件（2026-02-05 新增）：
    - ModelRetryMiddleware: 模型调用重试
    - ToolRetryMiddleware: 工具调用重试
    - ModelFallbackMiddleware: 模型降级
    
    成本控制中间件（2026-02-05 新增）：
    - ModelCallLimitMiddleware: 模型调用次数限制
    - ToolCallLimitMiddleware: 工具调用次数限制
    
    原有中间件：
    - TodoListMiddleware: 任务规划与跟踪
    - FilesystemMiddleware: 上下文管理（大结果自动驱逐）
    - SubAgentMiddleware: 并行子 Agent 调度
    - SummarizationMiddleware: 对话历史压缩
    
    Args:
        llm: 语言模型实例，默认使用配置的模型
        enable_doctor_approval: 是否启用医生审批（当前服务端不支持）
        checkpointer: 状态持久化，默认 MemorySaver
        store: 长期存储，默认 InMemoryStore
        
    Returns:
        DeepSearch Agent 实例
    """
    logger.info(f"创建 DeepSearch Agent，enable_doctor_approval={enable_doctor_approval}")
    
    # 1. 初始化组件
    if llm is None:
        llm = get_llm(temperature=diagnose_config.DIAGNOSIS_TEMPERATURE)
    
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    if store is None:
        store = InMemoryStore()
    
    # 2. 创建子 Agent
    logger.info("创建专家子 Agent...")
    differential_expert = create_differential_expert(llm)
    treatment_expert = create_treatment_expert(llm)
    prescription_expert = create_prescription_expert(llm)
    prognosis_expert = create_prognosis_expert(llm)
    verification_expert = create_verification_expert(llm)
    logger.debug("子 Agent 注册：differential, treatment, prescription, prognosis, verification")
    
    # 3. 数据查询工具列表
    data_tools = [
        kg_syndrome_search,
        case_vector_search,
        classics_search,
        web_search,
    ]
    
    # 4. 构建中间件栈
    # 
    # 重要说明（来自官方文档）：
    # create_deep_agent 已经内置了以下功能，不需要手动添加对应中间件：
    # - Planning/Task decomposition: 内置 write_todos 工具（TodoListMiddleware）
    # - Context management: 内置文件系统工具 ls/read_file/write_file/edit_file（FilesystemMiddleware）
    # - Subagent spawning: 内置 task 工具（SubAgentMiddleware）
    # 
    # 因此只添加额外的控制中间件：
    logger.info("构建中间件栈...")
    middleware_stack = [
        # ============================================================
        # 稳定性中间件
        # ============================================================
        
        # (1) 模型调用重试
        ModelRetryMiddleware(
            max_retries=2,
            backoff_factor=1.5,
        ),
        
        # (2) 工具调用重试
        ToolRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
        ),
        
        # ============================================================
        # 成本控制中间件
        # ============================================================
        
        # (3) 模型调用次数限制
        ModelCallLimitMiddleware(
            thread_limit=30,   # 每会话最多 30 次模型调用
            run_limit=15,      # 单次运行最多 15 次
            exit_behavior="end"
        ),
        
        # (4) 工具调用次数限制
        ToolCallLimitMiddleware(
            thread_limit=15,   # 总共最多 15 次工具调用
            exit_behavior="end"
        ),
        
        # ============================================================
        # 可选：对话总结中间件（如果需要）
        # ============================================================
        
        # SummarizationMiddleware(
        #     model=get_llm(model="deepseek-chat"),
        #     trigger=[("tokens", 4000), ("messages", 15)],
        #     summary_prompt="总结患者诊疗历史，保留关键症状、诊断和治疗方案"
        # ),
    ]
    
    # (5) 医生审批（当前服务端不支持，保留代码但不启用）
    if enable_doctor_approval:
        logger.warning("医生审批功能当前服务端不支持，已跳过")
        # 未来支持时可取消注释：
        # from langchain.agents.middleware import HumanInTheLoopMiddleware
        # middleware_stack.append(
        #     HumanInTheLoopMiddleware(
        #         interrupt_on={"synthesizer": True},
        #         description_prefix="请医生审核以下辨证结果：",
        #     )
        # )
    
    # 5. Skills 配置
    # 根据 DeepAgents 文档，skills 路径使用正斜杠，相对于 backend 根目录
    import os
    skills_dir = os.path.join(os.path.dirname(__file__), "skills")
    
    # 检查 skills 目录是否存在
    if os.path.exists(skills_dir):
        # 转换为正斜杠路径（DeepAgents 要求）
        skills_path = skills_dir.replace("\\", "/")
        skills_config = [skills_path]
        logger.info(f"Skills 路径: {skills_path}")
    else:
        skills_config = None
        logger.warning("Skills 目录不存在，跳过 Skills 加载")
    
    # 6. 创建主 Agent
    logger.info("创建 DeepSearch Agent...")
    agent = create_deep_agent(
        model=llm,
        checkpointer=checkpointer,
        store=store,
        tools=data_tools,
        middleware=middleware_stack,
        skills=skills_config,
        system_prompt=DEEP_SEARCH_SYSTEM_PROMPT
    )
    
    logger.info("DeepSearch Agent 创建完成")
    return agent


# ============================================================
# 便捷函数
# ============================================================

async def run_deep_search_diagnosis(
    collected_info: Dict[str, Any],
    thread_id: Optional[str] = None,
    enable_doctor_approval: bool = False,
) -> Dict[str, Any]:
    """
    执行深度辨证分析
    
    这是一个便捷函数，封装了 Agent 的创建和调用。
    
    Args:
        collected_info: 收集的患者信息
        thread_id: 会话 ID，用于持久化
        enable_doctor_approval: 是否启用医生审批
        
    Returns:
        诊断结果字典
    """
    import time
    import json
    
    # 生成 thread_id
    if thread_id is None:
        thread_id = f"diagnosis_{int(time.time())}"
    
    # 创建 Agent
    agent = create_deep_search_agent(enable_doctor_approval=enable_doctor_approval)
    
    # 构建提示词
    prompt = f"""
请对以下患者信息进行深度辨证分析：

{json.dumps(collected_info, ensure_ascii=False, indent=2)}

请：
1. 先并行查询数据（知识图谱、医案、古籍）
2. 再并行咨询 5 位专家
3. 最后综合给出诊断结果

请严格按照输出标准提供完整的辨证分析。
"""
    
    # 执行
    config = {"configurable": {"thread_id": thread_id}}
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config
    )
    
    # 检查是否中断（等待审批）
    if "__interrupt__" in result:
        return {
            "pending_approval": True,
            "diagnosis_draft": result["__interrupt__"],
            "thread_id": thread_id,
            "steps": ["复杂辨证: 等待医生审批"]
        }
    
    # 返回结果
    return {
        "answer": result["messages"][-1].content if result.get("messages") else "",
        "steps": result.get("steps", ["复杂辨证: 完成"]),
        "thread_id": thread_id
    }
