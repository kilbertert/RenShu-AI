"""
使用 create_agent 完全自定义中间件的 DeepSearch Agent

对比：
- create_deep_agent: 内置 TodoList/Filesystem/SubAgent 功能，简单快速
- create_agent: 完全自定义，可以手动配置所有中间件

如果需要完全控制中间件配置，使用这个版本
"""

from typing import Dict, Any, Optional
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# 使用 create_agent 而不是 create_deep_agent
from langchain.agents import create_agent

from langchain.agents.middleware import (
    TodoListMiddleware,
    SummarizationMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
)
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

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

logger = get_logger("deep_search_agent_custom")


CUSTOM_SYSTEM_PROMPT = """
你是中医智能诊断助手（DeepSearch Agent - 自定义版本），专注于复杂病例的深度辨证分析。

## 核心能力

- 并行数据查询：知识图谱、医案、古籍、网络搜索
- 并行专家咨询：5 位专家子 Agent 协同分析
- 任务规划：使用 write_todos 工具分解任务
- 文件系统：使用 write_file/read_file 管理大量数据
- 综合决策：汇总专家意见，给出最终诊断
"""


def create_deep_search_agent_custom(
    llm: Optional[BaseChatModel] = None,
    enable_doctor_approval: bool = False,
    checkpointer: Optional[Any] = None,
    store: Optional[Any] = None,
) -> Any:
    """
    使用 create_agent 创建完全自定义的 DeepSearch Agent
    
    与 create_deep_search_agent 的区别：
    - 使用 create_agent API（LangChain 标准 API）
    - 可以完全自定义所有中间件配置
    - 需要手动添加 TodoList/Filesystem/SubAgent 中间件
    
    适用场景：
    - 需要精细控制中间件行为
    - 需要自定义任务规划逻辑
    - 需要特殊的文件系统配置
    - 需要复杂的子 Agent 调度策略
    
    Args:
        llm: 语言模型实例
        enable_doctor_approval: 是否启用医生审批
        checkpointer: 状态持久化
        store: 长期存储
        
    Returns:
        自定义配置的 DeepSearch Agent
    """
    logger.info(f"创建自定义 DeepSearch Agent，enable_doctor_approval={enable_doctor_approval}")
    
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
    
    # 3. 数据查询工具
    data_tools = [
        kg_syndrome_search,
        case_vector_search,
        classics_search,
        web_search,
    ]
    
    # 4. 构建完整的中间件栈（包括所有中间件）
    logger.info("构建完整中间件栈...")
    middleware_stack = [
        # ============================================================
        # P1: 稳定性中间件
        # ============================================================
        
        ModelRetryMiddleware(
            max_retries=2,
            backoff_factor=1.5,
        ),
        
        ToolRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
        ),
        
        # ============================================================
        # P2: 任务管理与上下文（完全自定义）
        # ============================================================
        
        # 任务规划与跟踪
        TodoListMiddleware(
            system_prompt="""
使用 write_todos 工具分解复杂辨证任务：

【第一步：数据收集】（并行，4 个工具）
1. kg_syndrome_search - 查询知识图谱证型
2. case_vector_search - 检索相似医案
3. classics_search - 检索古籍论述
4. web_search - 搜索最新研究（可选）

【第二步：专家咨询】（并行，5 个子 Agent）
5. 鉴别诊断专家：区分相似证型
6. 治则治法专家：制定治疗策略
7. 方药推荐专家：推荐方剂
8. 预后评估专家：判断病情趋势
9. 质疑验证专家：验证诊断合理性

【第三步：综合决策】
10. 综合所有专家意见
11. 解决分歧
12. 输出最终诊断
"""
        ),
        
        # 上下文管理（文件系统）
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
2. 上下文中只保留摘要（前 200 tokens）
3. 需要详细内容时使用 read_file 读取

文件路径规则：
- /kg_results/task_{id}.json: 知识图谱查询结果
- /case_library/case_{id}.json: 相似医案详情
- /classics/{book}_{chapter}.txt: 古籍原文
- /patient_history/{user_id}.json: 患者历史档案
"""
        ),
        
        # 并行子 Agent 调度
        SubAgentMiddleware(
            default_model=llm,
            subagents=[
                {
                    "name": "differential_diagnosis_expert",
                    "description": "鉴别诊断专家：区分相似证型，分析寒热虚实真假，避免误诊",
                    "agent": differential_expert,
                    "system_prompt": DIFFERENTIAL_DIAGNOSIS_EXPERT_PROMPT
                },
                {
                    "name": "treatment_principle_expert",
                    "description": "治则治法专家：根据辨证结果制定治疗策略（治本/治标/标本兼治）",
                    "agent": treatment_expert,
                    "system_prompt": TREATMENT_PRINCIPLE_EXPERT_PROMPT
                },
                {
                    "name": "prescription_recommendation_expert",
                    "description": "方药推荐专家：推荐方剂、食疗方案和穴位调理建议",
                    "agent": prescription_expert,
                    "system_prompt": PRESCRIPTION_RECOMMENDATION_EXPERT_PROMPT
                },
                {
                    "name": "prognosis_evaluation_expert",
                    "description": "预后评估专家：判断病情发展趋势，评估正邪进退",
                    "agent": prognosis_expert,
                    "system_prompt": PROGNOSIS_EVALUATION_EXPERT_PROMPT
                },
                {
                    "name": "verification_expert",
                    "description": "质疑验证专家：作为'魔鬼代言人'验证诊断的合理性和完整性",
                    "agent": verification_expert,
                    "system_prompt": VERIFICATION_EXPERT_PROMPT
                },
            ]
        ),
        
        # 对话历史压缩
        SummarizationMiddleware(
            model=get_llm(model="deepseek-chat"),
            trigger=[("tokens", 4000), ("messages", 15)],
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
        ),
        
        # ============================================================
        # P3: 成本控制中间件
        # ============================================================
        
        ModelCallLimitMiddleware(
            thread_limit=30,
            run_limit=15,
            exit_behavior="end"
        ),
        
        ToolCallLimitMiddleware(
            thread_limit=15,
            exit_behavior="end"
        ),
    ]
    
    # 5. 使用 create_agent 创建 Agent（支持所有中间件）
    logger.info("使用 create_agent 创建自定义 Agent...")
    agent = create_agent(
        model=llm,
        checkpointer=checkpointer,
        store=store,
        tools=data_tools,
        middleware=middleware_stack,
        system_prompt=CUSTOM_SYSTEM_PROMPT
    )
    
    logger.info("自定义 DeepSearch Agent 创建完成")
    return agent


# 便捷函数
async def run_custom_deep_search_diagnosis(
    collected_info: Dict[str, Any],
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    使用自定义版本执行深度辨证分析
    
    Args:
        collected_info: 收集的患者信息
        thread_id: 会话 ID
        
    Returns:
        诊断结果字典
    """
    import time
    import json
    
    if thread_id is None:
        thread_id = f"custom_diagnosis_{int(time.time())}"
    
    agent = create_deep_search_agent_custom()
    
    prompt = f"""
请对以下患者信息进行深度辨证分析：

{json.dumps(collected_info, ensure_ascii=False, indent=2)}

请：
1. 先使用 write_todos 规划任务
2. 并行查询数据（知识图谱、医案、古籍）
3. 并行咨询 5 位专家（使用 task 工具）
4. 最后综合给出诊断结果

请严格按照输出标准提供完整的辨证分析。
"""
    
    config = {"configurable": {"thread_id": thread_id}}
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config
    )
    
    return {
        "answer": result["messages"][-1].content if result.get("messages") else "",
        "steps": result.get("steps", ["自定义复杂辨证: 完成"]),
        "thread_id": thread_id
    }
