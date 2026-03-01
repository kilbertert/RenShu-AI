"""
TCM Multi-Agent State Definitions
中医多智能体状态定义

定义了中医问诊系统中各个智能体之间传递的状态结构
"""

from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from app.src.agent.intent_recognition.schemas import IntentClassification

# ============== 消息处理函数 ==============

def reduce_str(left: str | None, right: str | None) -> str:
    """字符串归约：右值覆盖左值"""
    if right is not None:
        return right
    return left or ""


def reduce_list(left: list | None, right: list | None) -> list:
    """列表归约：合并两个列表"""
    left = left or []
    right = right or []
    return left + right


# ============== 路由类型定义 ==============

TCMQueryType = Literal[
    "tcm-chat",      # 中医闲聊：普通对话、闲聊
    "tcm-wellness",  # 中医养生：日常养生、体质调理、季节养生
    "tcm-diagnose",  # 辨证问诊：症状分析、证型判断
    "tcm-prescription",  # 方剂推荐：根据证型推荐方剂
    "additional-info",   # 需要更多信息
]


class TCMRouter(BaseModel):
    """中医意图路由结果"""
    classification: Optional[IntentClassification] = Field(
        default=None,
        description="用户查询的类型分类（意图识别结果）"
    )
    query_type: TCMQueryType = Field(
        description="用户查询的类型分类"
    )
    reasoning: str = Field(
        default="",
        description="路由决策的推理过程"
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="路由决策的置信度"
    )
    extracted_entities: dict = Field(
        default_factory=dict,
        description="从用户输入中提取的实体（症状、药材名等）"
    )
    has_image: bool = Field(
        default=False,
        description="是否包含图片"
    )


# ============== 辨证问诊状态 ==============

class DiagnoseStage(BaseModel):
    """辨证问诊阶段状态"""
    current_stage: Literal[
        "initial",      # 初始阶段
        "cold_heat",    # 寒热问诊
        "sweat",        # 汗出问诊
        "stool_urine",  # 二便问诊
        "diet",         # 饮食问诊
        "sleep",        # 睡眠问诊
        "tongue",       # 舌诊
        "pulse",        # 脉诊
        "complete",     # 问诊完成
    ] = Field(default="initial", description="当前问诊阶段")

    collected_info: dict = Field(
        default_factory=dict,
        description="已收集的问诊信息"
    )

    missing_info: list[str] = Field(
        default_factory=list,
        description="尚未收集的信息项"
    )


class SyndromeResult(BaseModel):
    """辨证结果"""
    syndrome_name: str = Field(description="证型名称，如'风寒证'、'湿热证'")
    confidence: float = Field(ge=0.0, le=1.0, description="辨证置信度")
    symptoms_matched: list[str] = Field(default_factory=list, description="匹配的症状")
    treatment_principle: str = Field(default="", description="治则治法")
    recommended_prescriptions: list[str] = Field(default_factory=list, description="推荐方剂")


# ============== 药材相关状态 ==============

class HerbInfo(BaseModel):
    """药材信息"""
    name: str = Field(description="药材名称")
    pinyin: str = Field(default="", description="拼音")
    category: str = Field(default="", description="药材分类")
    nature: str = Field(default="", description="药性（寒/热/温/凉/平）")
    flavor: list[str] = Field(default_factory=list, description="药味（酸/苦/甘/辛/咸）")
    meridians: list[str] = Field(default_factory=list, description="归经")
    effects: list[str] = Field(default_factory=list, description="功效")
    indications: list[str] = Field(default_factory=list, description="主治")
    contraindications: list[str] = Field(default_factory=list, description="禁忌")
    dosage: str = Field(default="", description="用量")


class HerbCompatibilityResult(BaseModel):
    """药材配伍校验结果"""
    is_compatible: bool = Field(description="是否配伍安全")
    warnings: list[str] = Field(default_factory=list, description="配伍警告")
    incompatible_pairs: list[tuple[str, str]] = Field(
        default_factory=list,
        description="不兼容的药对（十八反、十九畏）"
    )
    suggestions: list[str] = Field(default_factory=list, description="配伍建议")


# ============== 方剂相关状态 ==============

class PrescriptionInfo(BaseModel):
    """方剂信息"""
    name: str = Field(description="方剂名称")
    source: str = Field(default="", description="出处")
    composition: list[dict] = Field(
        default_factory=list,
        description="组成药材及用量，如[{'herb': '桂枝', 'dosage': '9g'}]"
    )
    effects: str = Field(default="", description="功效")
    indications: str = Field(default="", description="主治")
    syndrome: str = Field(default="", description="适用证型")
    usage: str = Field(default="", description="用法")
    cautions: list[str] = Field(default_factory=list, description="注意事项")


# ============== 古籍/医案相关状态 ==============

class ClassicReference(BaseModel):
    """古籍条文引用"""
    book_name: str = Field(description="书名")
    chapter: str = Field(default="", description="章节")
    content: str = Field(description="条文内容")
    interpretation: str = Field(default="", description="释义")
    clinical_application: str = Field(default="", description="临床应用")


class CaseReference(BaseModel):
    """医案引用"""
    case_id: str = Field(description="医案ID")
    patient_info: str = Field(default="", description="患者信息（脱敏）")
    chief_complaint: str = Field(description="主诉")
    symptoms: list[str] = Field(default_factory=list, description="症状")
    syndrome: str = Field(description="辨证")
    treatment: str = Field(description="治法")
    prescription: str = Field(description="处方")
    outcome: str = Field(default="", description="疗效")
    source: str = Field(default="", description="来源")


# ============== 图像分析状态 ==============

class TongueAnalysisResult(BaseModel):
    """舌诊分析结果"""
    tongue_color: str = Field(default="", description="舌色（淡红/红/绛/紫/淡白）")
    tongue_shape: str = Field(default="", description="舌形（胖大/瘦薄/齿痕/裂纹）")
    coating_color: str = Field(default="", description="苔色（白/黄/灰/黑）")
    coating_texture: str = Field(default="", description="苔质（薄/厚/腻/燥/滑）")
    analysis: str = Field(default="", description="舌诊分析")
    syndrome_hints: list[str] = Field(default_factory=list, description="证型提示")


# ============== 模型配置 ==============

class LLMConfig(BaseModel):
    """LLM 模型配置"""
    provider_name: str = Field(default="", description="提供商名称 (openai/deepseek/ollama)")
    model_name: str = Field(default="", description="模型名称")
    api_key: str = Field(default="", description="API Key")
    base_url: Optional[str] = Field(default=None, description="API Base URL")
    temperature: float = Field(default=0.7, description="温度参数")
    top_p: float = Field(default=1.0, description="Top-P 采样参数")
    max_tokens: int = Field(default=2000, description="最大 token 数")
    enable_thinking: bool = Field(default=False, description="是否启用思考过程展示（类似 Gemini 的推理过程）")


# ============== 主状态定义 ==============

class TCMInputState(BaseModel):
    """中医智能体输入状态"""
    messages: Annotated[list[AnyMessage], add_messages] = Field(
        default_factory=list,
        description="对话消息历史"
    )
    user_id: str = Field(default="", description="用户ID")
    conversation_id: str = Field(default="", description="会话ID")

    # 用户画像
    user_profile: dict = Field(
        default_factory=dict,
        description="用户基础画像（体质、年龄、性别、既往史等）"
    )

    # 模型配置（从前端传入）
    llm_config: Optional[LLMConfig] = Field(
        default=None,
        description="LLM 模型配置"
    )


class TCMAgentState(TCMInputState):
    """中医智能体主状态"""
    
    # 允许额外字段（兼容中间件动态添加字段）
    model_config = {"extra": "allow"}

    # 路由信息
    router: Optional[TCMRouter] = Field(
        default=None,
        description="路由决策结果"
    )

    # 辨证问诊
    diagnose_stage: Optional[DiagnoseStage] = Field(
        default=None,
        description="辨证问诊阶段状态"
    )
    syndrome_result: Optional[SyndromeResult] = Field(
        default=None,
        description="辨证结果"
    )

    # 查询结果
    herbs: Annotated[list[HerbInfo], reduce_list] = Field(
        default_factory=list,
        description="查询到的药材信息"
    )
    prescriptions: Annotated[list[PrescriptionInfo], reduce_list] = Field(
        default_factory=list,
        description="查询到的方剂信息"
    )
    classics: Annotated[list[ClassicReference], reduce_list] = Field(
        default_factory=list,
        description="查询到的古籍条文"
    )
    cases: Annotated[list[CaseReference], reduce_list] = Field(
        default_factory=list,
        description="查询到的医案"
    )

    # 图像分析
    tongue_analysis: Optional[TongueAnalysisResult] = Field(
        default=None,
        description="舌诊分析结果"
    )

    # 校验结果
    compatibility_check: Optional[HerbCompatibilityResult] = Field(
        default=None,
        description="配伍校验结果"
    )

    # 执行追踪
    steps: Annotated[list[str], reduce_list] = Field(
        default_factory=list,
        description="执行步骤记录"
    )
    cypher_queries: Annotated[list[str], reduce_list] = Field(
        default_factory=list,
        description="执行的Cypher查询"
    )

    # 最终输出
    answer: Annotated[str, reduce_str] = Field(
        default="",
        description="最终回答"
    )

    # 错误处理
    error: Optional[str] = Field(
        default=None,
        description="错误信息"
    )

    # 中间件状态
    jump_to: Optional[str] = Field(
        default=None,
        description="中间件跳转目标（如 'end' 表示跳过后续处理）"
    )
    should_seek_doctor: bool = Field(
        default=False,
        description="是否建议就医（紧急情况/严重症状）"
    )

    # === 支持字典式访问（兼容 LangChain 内置中间件） ===
    def __getitem__(self, key: str):
        """支持 state["key"] 访问方式"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in {self.__class__.__name__}")

    def __setitem__(self, key: str, value):
        """支持 state["key"] = value 赋值方式（允许动态字段）"""
        setattr(self, key, value)

    def get(self, key: str, default=None):
        """支持 state.get("key", default) 访问方式"""
        return getattr(self, key, default)


class TCMOutputState(BaseModel):
    """中医智能体输出状态"""
    answer: str = Field(description="最终回答")
    query_type: TCMQueryType = Field(description="查询类型")

    # 可选的结构化数据
    syndrome_result: Optional[SyndromeResult] = Field(
        default=None,
        description="辨证结果（如果有）"
    )
    herbs: list[HerbInfo] = Field(
        default_factory=list,
        description="药材信息（如果有）"
    )
    prescriptions: list[PrescriptionInfo] = Field(
        default_factory=list,
        description="方剂信息（如果有）"
    )
    classics: list[ClassicReference] = Field(
        default_factory=list,
        description="古籍条文（如果有）"
    )
    cases: list[CaseReference] = Field(
        default_factory=list,
        description="医案（如果有）"
    )
    tongue_analysis: Optional[TongueAnalysisResult] = Field(
        default=None,
        description="舌诊分析（如果有）"
    )

    # 追踪信息
    steps: list[str] = Field(default_factory=list, description="执行步骤")
    cypher_queries: list[str] = Field(default_factory=list, description="执行的查询")

    # 后续建议
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="建议的后续问题"
    )
    should_seek_doctor: bool = Field(
        default=False,
        description="是否建议就医"
    )


# ============== 知识图谱子图状态 ==============

class KGInputState(BaseModel):
    """知识图谱子图输入状态"""
    question: str = Field(description="用户问题")
    query_type: TCMQueryType = Field(description="查询类型")
    entities: dict = Field(default_factory=dict, description="提取的实体")
    history: list[dict] = Field(default_factory=list, description="对话历史")


class KGTask(BaseModel):
    """知识图谱查询任务"""
    task_id: str = Field(description="任务ID")
    task_type: Literal[
        "cypher_query",       # 自动生成Cypher查询
        "predefined_cypher",  # 预定义Cypher模板
        "graphrag_search",    # GraphRAG语义检索
        "vector_search",      # 向量检索
    ] = Field(description="任务类型")
    description: str = Field(description="任务描述")
    parameters: dict = Field(default_factory=dict, description="任务参数")
    parent_task_id: Optional[str] = Field(default=None, description="父任务ID")


class KGOverallState(BaseModel):
    """知识图谱子图主状态"""
    question: str = Field(description="用户问题")
    query_type: TCMQueryType = Field(description="查询类型")
    entities: dict = Field(default_factory=dict, description="提取的实体")

    # 任务管理
    tasks: Annotated[list[KGTask], reduce_list] = Field(
        default_factory=list,
        description="待执行的任务列表"
    )
    next_action: Literal["planner", "execute", "summarize", "end"] = Field(
        default="planner",
        description="下一步动作"
    )

    # 查询结果
    cypher_queries: Annotated[list[str], reduce_list] = Field(
        default_factory=list,
        description="执行的Cypher查询"
    )
    query_results: Annotated[list[dict], reduce_list] = Field(
        default_factory=list,
        description="查询结果"
    )

    # 汇总
    summary: Annotated[str, reduce_str] = Field(
        default="",
        description="结果汇总"
    )

    # 执行追踪
    steps: Annotated[list[str], reduce_list] = Field(
        default_factory=list,
        description="执行步骤"
    )

    # 历史
    history: list[dict] = Field(default_factory=list, description="对话历史")


class KGOutputState(BaseModel):
    """知识图谱子图输出状态"""
    answer: str = Field(description="回答")
    query_results: list[dict] = Field(default_factory=list, description="查询结果")
    cypher_queries: list[str] = Field(default_factory=list, description="执行的查询")
    steps: list[str] = Field(default_factory=list, description="执行步骤")


# ============== Cypher执行状态 ==============

class CypherExecutionState(BaseModel):
    """Cypher查询执行状态"""
    statement: str = Field(description="Cypher语句")
    parameters: dict = Field(default_factory=dict, description="查询参数")

    # 执行结果
    records: list[dict] = Field(default_factory=list, description="查询结果记录")
    error: Optional[str] = Field(default=None, description="错误信息")

    # 重试控制
    attempts: int = Field(default=0, description="尝试次数")
    max_attempts: int = Field(default=3, description="最大尝试次数")

    # 校验
    is_validated: bool = Field(default=False, description="是否已校验")
    validation_errors: list[str] = Field(default_factory=list, description="校验错误")
