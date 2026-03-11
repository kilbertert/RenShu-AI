"""
主图状态定义。
使用 TypedDict 明确 LangGraph state 中各模块上下文的结构。
"""

from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional, Union, cast

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict

class TCMRouter(BaseModel):
    """路由决策结果"""

    query_type: str = Field(description="查询类型")
    reasoning: str = Field(default="", description="路由推理")
    route_source: str = Field(default="rule", description="路由来源")
    confidence: float = Field(default=1.0, description="置信度")
    has_image: bool = Field(default=False, description="是否有图片")
    primary_intent: Optional[str] = Field(default=None, description="主要意图")
    sub_type: Optional[str] = Field(default=None, description="子类型")
    requires_image: bool = Field(default=False, description="是否需要图片")


class LLMConfig(BaseModel):
    """LLM 配置"""

    provider_name: Optional[str] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    top_p: Optional[float] = None
    max_tokens: int = 4096
    enable_thinking: bool = False


class UserProfileData(TypedDict, total=False):
    """数据库/事实画像中的用户画像"""
    """
      session_metadata:   "age": "string",
                          "gender": "string",
                          "healthScore": 0,
                          "chiefComplaint": "string",
                          "suspectedDiagnosis": "string",
                          "recommendedTreatment": "string"
      base_profile:体质类型（constitution_type）
                   禁忌项（taboo_items）
                   既往病史(medical_history)、
                   家族病史( family_history) 、 
                   allergy_info（过敏信息）                    
                          
         
          
    
    """
    constitution_type:str #体质类型
    taboo_items:str #禁忌项
    medical_history:str #既往病史
    family_history:str#家族病史
    allergy_info:str #过敏信息




class FactualMemorySources(TypedDict):
    """事实记忆来源"""

    source: str
    locations: List[str]
    memory_type: str
    memory_class: str


class FactualMemoryContext(TypedDict):
    """factual 长期上下文"""

    sources: FactualMemorySources
    user_profile: UserProfileData


class LongTermMemoryMetadata(TypedDict, total=False):
    """Mem0 记忆元数据"""

    memory_type: str
    memory_class: str
    user_id: str
    session_id: str
    syndrome: str
    symptoms: Dict[str, Any]
    prescription: str
    summary: str
    key_points: List[str]
    task_type: str
    related_entities: List[str]
    created_at: str
    confidence: float
    source: str


class LongTermMemoryRecord(TypedDict, total=False):
    """归一化后的长期记忆记录"""

    id: str
    memory: str
    text: str
    score: float
    created_at: str
    updated_at: str
    metadata: LongTermMemoryMetadata
    memory_type: str
    memory_class: str




class EpisodicMemorySource(TypedDict,total=False):
    source: str
    locations: List[str]
    memory_type: str
    memory_class: str

class EpisodicMemoryContext(TypedDict):
    """episodic 长期上下文"""

    sources: EpisodicMemorySource
    summaries: List[LongTermMemoryRecord]


class SemanticRelationRecord(TypedDict, total=False):
    """语义关系记录"""

    relation: str
    subject: str
    predicate: str
    object: str
    source: str
    target: str
    type: str
    score: float
    metadata: Dict[str, Any]


class SemanticMemorySources(TypedDict):
    """语义记忆来源"""
    source: str
    locations: List[str]
    memory_type: str
    memory_class: str


class SemanticMemoryContext(TypedDict):
    """semantic 长期上下文"""

    sources: SemanticMemorySources
    relations: List[SemanticRelationRecord]
    entities: List[str]


class MemoryContext(TypedDict):
    """MemoryMiddleware 写入主图的长期上下文"""

    user_id: str
    user_profile: UserProfileData
    factual: FactualMemoryContext
    episodic: EpisodicMemoryContext
    semantic: SemanticMemoryContext



class ContextUsageStatus(TypedDict):
    """FocusContextMiddleware 的 token 使用状态"""

    status: Literal["normal", "approaching", "critical"]
    current_tokens: int
    max_tokens: int
    usage_ratio: float
    remaining_tokens: int


class FocusStats(TypedDict):
    """Focus 引擎统计信息"""

    total_phases: int
    total_compressions: int
    total_dropped_messages: int
    total_tokens_saved: int
    current_phase: Optional[str]


class CompressionAppliedDetails(TypedDict):
    """Focus 压缩明细"""

    level: str
    tokens_before: int
    tokens_after: int
    tokens_saved: int


class FocusEnrichedContext(TypedDict):
    """FocusContextMiddleware 写入主图的上下文工程信息"""

    focus_stats: FocusStats
    context_status: ContextUsageStatus
    compression_applied: bool
    compression_details: NotRequired[CompressionAppliedDetails]


class IntentRouteCorrectionRecord(TypedDict):
    """主图路由纠错记录"""

    user_input: str
    wrong_route: Optional[str]
    correct_route: Optional[str]
    reason: str
    hint: str
    thread_intent_summary: str
    source: str
    trigger: str
    created_at: str


class ToolSelectionCorrectionRecord(TypedDict):
    """子图工具选择纠错记录"""

    wrong_tool: Optional[str]
    correct_tool: Optional[str]
    reason: str
    missing_info: List[str]
    hint: str
    source: str
    subgraph: str
    trigger: str
    created_at: str


class UserOutputCorrectionRecord(TypedDict):
    """用户输出纠错记录"""

    wrong_understanding: str
    user_correction: str
    invalidated_assumption: str
    hint: str
    source: str
    subgraph: str
    trigger: str
    created_at: str


class LearningEventRecord(TypedDict):
    """学习事件记录"""

    event_type: str
    conversation_id: str
    source: str
    trigger: str
    payload: Dict[str, Any]
    created_at: str
    metadata: Dict[str, Any]


class IntentLearningSnapshot(TypedDict):
    """单线程意图学习摘要"""

    thread_intent_summary: str
    intent_route_corrections: List[IntentRouteCorrectionRecord]
    intent_disambiguation_hints: List[str]


class ThreadLearningContextDetails(TypedDict):
    """结构化单线程学习上下文"""

    conversation_id: str
    intent_learning: IntentLearningSnapshot
    tool_learning: List[ToolSelectionCorrectionRecord]
    correction_learning: List[UserOutputCorrectionRecord]
    last_updated_at: str
    learning_events: NotRequired[List[LearningEventRecord]]


class ThreadLearningContext(TypedDict):
    """LearningMiddleware 写入主图的单线程学习快照"""

    session_id: str
    recent_error_reflections: List[str]
    recent_corrections: List[str]
    recent_effective_strategies: List[str]
    current_thread_summary: str
    thread_learning_context: ThreadLearningContextDetails
    conversation_id: NotRequired[str]
    intent_learning: NotRequired[IntentLearningSnapshot]
    tool_learning: NotRequired[List[ToolSelectionCorrectionRecord]]
    correction_learning: NotRequired[List[UserOutputCorrectionRecord]]
    last_updated_at: NotRequired[str]
    learning_events: NotRequired[List[LearningEventRecord]]


class DiscriminatingRuleRecord(TypedDict):
    """跨线程辨证鉴别规则"""

    syndrome_a: str
    syndrome_b: str
    rule: str
    symptoms: List[str]
    frequency: int


class MisdiagnosisPatternRecord(TypedDict):
    """跨线程误诊模式"""

    pattern_name: str
    wrong_syndrome: str
    correct_syndrome: str
    common_causes: List[str]
    missed_symptoms: List[str]
    prevention_rule: str
    occurrence_count: int
    severity: str


class EffectiveStrategyRecord(TypedDict):
    """跨线程高效策略"""

    strategy_name: str
    strategy_type: str
    description: str
    applicable_symptoms: List[str]
    applicable_syndromes: List[str]
    optimal_questions: List[str]
    avg_rounds: float
    satisfaction: float
    usage_count: int


class CrossThreadLearningContext(TypedDict):
    """LearningMiddleware 写入主图的跨线程学习上下文"""

    discriminating_rules: List[DiscriminatingRuleRecord]
    misdiagnosis_patterns: List[MisdiagnosisPatternRecord]
    effective_strategies: List[EffectiveStrategyRecord]


class MainInput(TypedDict):
    """主图输入状态"""

    messages: Annotated[list[BaseMessage], "add_messages"]
    user_id: str
    conversation_id: str
    llm_config: Optional[LLMConfig]





class MainState(TypedDict):
    """主图状态"""

    messages: Annotated[list[BaseMessage], "add_messages"]
    user_id: str
    conversation_id: str
    llm_config: Optional[LLMConfig]

    memory_context: Optional[MemoryContext]
    enriched_context: Optional[FocusEnrichedContext]
    thread_learning_context: Optional[ThreadLearningContext]
    cross_thread_learning: Optional[CrossThreadLearningContext]
    focus_reminder: Optional[str]

    jump_to: Optional[str]
    router: Optional[TCMRouter]

    answer: str
    structured_data: Optional[Dict[str, Any]]
    error: Optional[str]

    learning_recorded: bool
    memory_saved: bool

    steps: Annotated[list[str], "extend"]


class MainOutput(TypedDict):
    """主图输出状态"""

    answer: str
    structured_data: Optional[Dict[str, Any]]
    error: Optional[str]
    steps: list[str]


MemorySection = Union[FactualMemoryContext, EpisodicMemoryContext, SemanticMemoryContext]


def get_memory_context(state: Mapping[str, Any]) -> MemoryContext:
    """获取主图长期记忆上下文。"""
    memory_context = state.get("memory_context")
    return cast(MemoryContext, memory_context) if isinstance(memory_context, dict) else cast(MemoryContext, {})


def get_memory_section(
    state: Mapping[str, Any],
    section: str,
) -> MemorySection:
    """按分区获取长期记忆。"""
    memory_context = get_memory_context(state)
    value = memory_context.get(section)
    return cast(MemorySection, value) if isinstance(value, dict) else cast(MemorySection, {})


def get_factual_memory(state: Mapping[str, Any]) -> FactualMemoryContext:
    """获取 factual 长期记忆分区。"""
    return cast(FactualMemoryContext, get_memory_section(state, "factual"))


def get_factual_user_profile(state: Mapping[str, Any]) -> UserProfileData:
    """获取 factual 用户画像。"""
    factual = get_factual_memory(state)
    profile = factual.get("user_profile")
    return cast(UserProfileData, profile) if isinstance(profile, dict) else cast(UserProfileData, {})


def get_episodic_memory(state: Mapping[str, Any]) -> EpisodicMemoryContext:
    """获取 episodic 长期记忆分区。"""
    return cast(EpisodicMemoryContext, get_memory_section(state, "episodic"))


def get_episodic_summaries(state: Mapping[str, Any]) -> List[LongTermMemoryRecord]:
    """获取跨线程情景摘要列表。"""
    episodic = get_episodic_memory(state)
    summaries = episodic.get("summaries")
    if not isinstance(summaries, list):
        return []
    return cast(List[LongTermMemoryRecord], [item for item in summaries if isinstance(item, dict)])


def get_semantic_memory(state: Mapping[str, Any]) -> SemanticMemoryContext:
    """获取 semantic 长期记忆分区。"""
    return cast(SemanticMemoryContext, get_memory_section(state, "semantic"))


def get_semantic_relations(state: Mapping[str, Any]) -> List[SemanticRelationRecord]:
    """获取跨线程语义关系。"""
    semantic = get_semantic_memory(state)
    relations = semantic.get("relations")
    if not isinstance(relations, list):
        return []
    return cast(List[SemanticRelationRecord], [item for item in relations if isinstance(item, dict)])


def get_semantic_entities(state: Mapping[str, Any]) -> List[str]:
    """获取跨线程稳定语义实体。"""
    semantic = get_semantic_memory(state)
    entities = semantic.get("entities")
    if not isinstance(entities, list):
        return []
    return [str(item).strip() for item in entities if str(item).strip()]
