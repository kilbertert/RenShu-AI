"""
Intent Recognition Schemas
意图识别数据模型定义
"""

from enum import Enum
from typing import Optional, Literal
from pydantic import BaseModel, Field


# ============== 意图类型枚举 ==============

class IntentType(str, Enum):
    """五大核心意图类型"""
    GENERAL = "general"             # 一般性知识、闲聊、科普
    WELLNESS = "wellness"           # 养生类
    PRESCRIPTION = "prescription"   # 方剂类
    HERB = "herb"                   # 药材类
    DIAGNOSIS = "diagnosis"         # 问诊类


class WellnessSubType(str, Enum):
    """养生类二级路由"""
    DAILY = "daily"           # 日常科普
    SEASONAL = "seasonal"     # 季节养生
    CONSTITUTION = "constitution"  # 体质调理
    COMPLEX = "complex"       # 复杂养生


class PrescriptionSubType(str, Enum):
    """方剂类二级路由"""
    QUERY = "query"           # 方剂查询
    COMPOSITION = "composition"  # 方剂组成
    RECOMMEND = "recommend"   # 方剂推荐
    COMPARE = "compare"       # 方剂对比


class HerbSubType(str, Enum):
    """药材类二级路由"""
    EFFECT = "effect"         # 药材功效
    COMPATIBILITY = "compatibility"  # 配伍禁忌
    USAGE = "usage"           # 用法用量
    IDENTIFY = "identify"     # 药材鉴别


class DiagnosisSubType(str, Enum):
    """问诊类二级路由"""
    SYMPTOM = "symptom"       # 症状咨询
    TONGUE = "tongue"         # 舌诊分析
    COMPREHENSIVE = "comprehensive"  # 综合问诊
    CASE = "case"             # 医案参考


class WellnessLevel(str, Enum):
    """养生路由级别"""
    L1 = "l1"  # 简单养生 - 直接LLM回答
    L2 = "l2"  # 复杂养生 - Web Search + LLM


class EmergencyType(str, Enum):
    """急症类型"""
    CARDIAC = "cardiac"             # 心脏急症
    CEREBRAL = "cerebral"           # 脑血管
    POISONING = "poisoning"         # 中毒
    TRAUMA = "trauma"               # 严重外伤
    CONSCIOUSNESS = "consciousness" # 意识障碍
    HIGH_FEVER = "high_fever"       # 高热危象
    ALLERGY = "allergy"             # 严重过敏
    NONE = "none"                   # 非急症


class OOSReason(str, Enum):
    """拒识原因"""
    LOW_CONFIDENCE = "low_confidence"   # 置信度过低
    WESTERN_MEDICINE = "western_medicine"  # 西医问题
    NON_MEDICAL = "non_medical"         # 非医疗问题
    INAPPROPRIATE = "inappropriate"      # 不当内容
    GIBBERISH = "gibberish"             # 无意义输入


# ============== 情感分析模型 ==============

class SentimentAnalysis(BaseModel):
    """情感分析结果"""
    polarity: Literal["positive", "neutral", "negative"] = Field(
        default="neutral",
        description="情感极性"
    )
    anxiety_score: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="焦虑程度 0-1"
    )
    urgency: Literal["low", "medium", "high"] = Field(
        default="low",
        description="紧迫程度"
    )


# ============== 实体提取模型 ==============

class ExtractedEntities(BaseModel):
    """提取的中医实体"""
    herbs: list[str] = Field(
        default_factory=list,
        description="药材名称列表"
    )
    prescriptions: list[str] = Field(
        default_factory=list,
        description="方剂名称列表"
    )
    symptoms: list[str] = Field(
        default_factory=list,
        description="症状列表"
    )
    body_parts: list[str] = Field(
        default_factory=list,
        description="身体部位"
    )
    syndromes: list[str] = Field(
        default_factory=list,
        description="证型"
    )
    books: list[str] = Field(
        default_factory=list,
        description="古籍名称"
    )
    symptom_nature: list[str] = Field(
        default_factory=list,
        description="症状性质（隐痛、刺痛等）"
    )
    triggers: list[str] = Field(
        default_factory=list,
        description="诱因（受寒后、生气后等）"
    )
    duration: Optional[str] = Field(
        default=None,
        description="病程"
    )


# ============== 上下文特征模型 ==============

class UserProfile(BaseModel):
    """用户画像"""
    user_id: str = Field(description="用户ID")
    gender: Optional[str] = Field(default=None, description="性别")
    age_group: Optional[str] = Field(default=None, description="年龄段")
    constitution: Optional[str] = Field(default=None, description="体质类型")
    chronic_conditions: list[str] = Field(
        default_factory=list,
        description="慢性病史"
    )
    allergies: list[str] = Field(
        default_factory=list,
        description="过敏史"
    )


class EnvironmentContext(BaseModel):
    """环境上下文"""
    solar_term: Optional[str] = Field(default=None, description="当前节气")
    season: Optional[str] = Field(default=None, description="季节")
    region: Optional[str] = Field(default=None, description="地域")
    weather: Optional[str] = Field(default=None, description="天气")


class ConversationContext(BaseModel):
    """对话上下文"""
    recent_symptoms: list[str] = Field(
        default_factory=list,
        description="最近提及的症状"
    )
    mentioned_herbs: list[str] = Field(
        default_factory=list,
        description="提及的药材"
    )
    mentioned_prescriptions: list[str] = Field(
        default_factory=list,
        description="提及的方剂"
    )
    diagnosis_stage: Optional[str] = Field(
        default=None,
        description="问诊阶段"
    )
    turn_count: int = Field(default=0, description="对话轮数")


class EnrichedContext(BaseModel):
    """增强后的上下文"""
    user_profile: UserProfile
    environment: EnvironmentContext
    conversation: ConversationContext


# ============== 意图分类结果模型 ==============

class IntentClassification(BaseModel):
    """意图分类完整结果"""

    # 主意图（四大类）
    primary_intent: IntentType = Field(description="主要意图类型")
    confidence: float = Field(
        ge=0,
        le=1,
        description="置信度"
    )

    # 二级路由
    sub_type: Optional[str] = Field(
        default=None,
        description="二级路由类型"
    )

    # 养生专用
    wellness_level: Optional[WellnessLevel] = Field(
        default=None,
        description="养生路由级别 L1/L2"
    )

    # 情感分析
    sentiment: SentimentAnalysis = Field(
        default_factory=SentimentAnalysis,
        description="情感分析结果"
    )

    # 实体提取
    entities: ExtractedEntities = Field(
        default_factory=ExtractedEntities,
        description="提取的实体"
    )

    # 路由决策
    reasoning: str = Field(
        default="",
        description="分类理由"
    )
    suggested_follow_up: Optional[str] = Field(
        default=None,
        description="建议的追问"
    )

    # 多模态标记
    requires_image: bool = Field(
        default=False,
        description="是否需要图片输入"
    )
    has_image: bool = Field(
        default=False,
        description="用户是否已提供图片"
    )

    # 路由来源
    route_source: Literal["rule", "semantic", "llm"] = Field(
        default="llm",
        description="路由决策来源"
    )


# ============== 路由结果模型 ==============

class EmergencyResult(BaseModel):
    """急救阻断结果"""
    is_emergency: bool = Field(description="是否为急症")
    emergency_type: EmergencyType = Field(
        default=EmergencyType.NONE,
        description="急症类型"
    )
    matched_keywords: list[str] = Field(
        default_factory=list,
        description="匹配的关键词"
    )
    response: Optional[str] = Field(
        default=None,
        description="急救提示响应"
    )


class OOSResult(BaseModel):
    """拒识检测结果"""
    is_oos: bool = Field(description="是否超出服务范围")
    reason: Optional[OOSReason] = Field(
        default=None,
        description="拒识原因"
    )
    action: Literal["allow", "clarify", "reject"] = Field(
        default="allow",
        description="处理动作"
    )
    response: Optional[str] = Field(
        default=None,
        description="拒识响应"
    )


class IntentRouteResult(BaseModel):
    """意图路由最终结果"""

    # 急救检查
    emergency: EmergencyResult = Field(
        default_factory=lambda: EmergencyResult(is_emergency=False)
    )

    # 拒识检查
    oos: OOSResult = Field(
        default_factory=lambda: OOSResult(is_oos=False)
    )

    # 意图分类
    classification: Optional[IntentClassification] = Field(
        default=None,
        description="意图分类结果"
    )

    # 上下文
    context: Optional[EnrichedContext] = Field(
        default=None,
        description="增强上下文"
    )

    # 路由路径追踪
    route_path: list[str] = Field(
        default_factory=list,
        description="路由路径记录"
    )

    # 处理时间
    latency_ms: float = Field(
        default=0,
        description="处理延迟（毫秒）"
    )

    def should_terminate(self) -> bool:
        """判断是否应该终止流程"""
        return self.emergency.is_emergency or self.oos.is_oos

    def get_final_response(self) -> Optional[str]:
        """获取终止响应"""
        if self.emergency.is_emergency:
            return self.emergency.response
        if self.oos.is_oos:
            return self.oos.response
        return None
