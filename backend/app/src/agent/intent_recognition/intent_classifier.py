"""
LLM Intent Classifier - LLM意图分类器
L3层：使用大模型进行深度意图分类和实体提取

借鉴大厂做法：
- 阿里PAI：通义千问 + 结构化输出
- 美团：RoBERTa多任务学习
- 百度：意图分类-需求检测-动态召回三级架构

使用方式：
    classifier = create_intent_classifier(
        provider_name="openai",
        model_name="gpt-4o-mini",
        api_key="sk-xxx"
    )
    result = await classifier.classify(query)
"""

import json
from typing import Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

from app.src.utils import get_logger

from .schemas import (
    IntentType,
    IntentClassification,
    ExtractedEntities,
    SentimentAnalysis,
    WellnessLevel,
    EnrichedContext,
)


logger = get_logger("intent_classifier")


def _content_to_text(content: Any) -> str:
    """兼容普通字符串与 Anthropic 内容块。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _parse_json_classification(content: Any) -> IntentClassification:
    """从纯 JSON 或 Markdown JSON 代码块中恢复意图分类。"""
    text = _content_to_text(content).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型响应中没有 JSON 对象")
    return IntentClassification.model_validate(json.loads(text[start : end + 1]))


# LLM意图分类系统提示词
INTENT_CLASSIFICATION_SYSTEM_PROMPT = """你是一个专业的中医智能助手意图分类器。你的任务是准确识别用户问题的意图类型，提取关键实体，并进行情感分析。

## 五大核心意图类型

### 1. general (一般性知识/闲聊)
- **定义**: 中医基础知识科普、理论介绍、闲聊问候等非业务类问题
- **典型问题**:
  - 中医基础概念："什么是中医"、"五行是什么"、"阴阳理论"
  - 中医历史文化："中医的历史"、"中医和西医的区别"
  - 问候闲聊："你好"、"今天天气怎么样"
- **注意**: 这类问题**不涉及**具体养生建议、药材查询、方剂推荐、症状诊断

### 2. wellness (养生类)
- **定义**: 日常养生保健、体质调理、季节养生等非诊疗性**实用建议**
- **二级分类 (sub_type)**:
  - daily: 日常科普养生
  - seasonal: 季节养生（春夏秋冬、节气）
  - constitution: 体质调理（气虚、阴虚、阳虚等）
  - complex: 复杂养生（多主题组合、专业术语）
- **L1/L2级别判定**:
  - L1 (简单): 通用科普、情感中性、无焦虑、单一主题
  - L2 (复杂): 体质调理、有焦虑担忧、专业问题
- **典型问题**: "春季如何养肝"、"气虚体质怎么调理"、"夏天吃什么好"

### 3. prescription (方剂类)
- **定义**: 询问中药方剂相关信息
- **二级分类 (sub_type)**:
  - query: 方剂基本信息查询（是什么、介绍）
  - composition: 方剂组成查询（成分、药材、配方）
  - recommend: 方剂推荐（根据证型/症状推荐）
  - compare: 方剂对比（区别、异同）

### 4. herb (药材类)
- **定义**: 询问中药材相关信息
- **二级分类 (sub_type)**:
  - effect: 药材功效（功效、作用、好处）
  - compatibility: 配伍禁忌（不能一起吃、相克）
  - usage: 用法用量（怎么吃、用量、煎法）
  - identify: 药材鉴别（真假、选购）

### 5. diagnosis (问诊类)
- **定义**: 涉及症状分析、辨证论治的诊疗性问题
- **二级分类 (sub_type)**:
  - symptom: 症状咨询（描述症状、求分析）
  - tongue: 舌诊分析（舌苔、舌头图片）
  - comprehensive: 综合问诊（多症状、体质信息）
  - case: 医案参考（类似病例、名医经验）

## 分类优先级规则

1. **区分 general 和 wellness（关键！）**:
   - **general**: 询问**概念、定义、原理、历史、理论**等知识
     - 问法特征："是什么"、"介绍"、"概念"、"原理"、"理论"、"历史"、"区别"
     - 示例："什么是中医"、"五行理论"、"中医和西医的区别"
   - **wellness**: 寻求**具体养生建议、调理方法、饮食推荐**
     - 问法特征："如何"、"怎么做"、"吃什么"、"注意什么"、"怎么调理"
     - 示例："春季如何养肝"、"气虚怎么调理"、"夏天吃什么好"
2. **优先问诊类**: 如果描述了具体身体症状，即使提到药材/方剂，也归为 diagnosis
3. **区分药材与方剂**: 单味药为 herb，复方为 prescription
4. **养生 vs 问诊**: 无具体症状的调理问题归 wellness，有症状归 diagnosis
5. **舌诊归问诊**: 涉及舌苔/舌头分析归为 diagnosis (sub_type=tongue)

## 情感分析指标

- **焦虑程度** (anxiety_score): 0-0.3低, 0.3-0.6中, 0.6-1.0高
- **紧迫程度** (urgency): low/medium/high

## 用户画像（如有）
{user_profile}

## 环境信息
{environment}

## 输出要求

请以JSON格式输出，包含以下字段：
- primary_intent: 主意图类型 (general/wellness/prescription/herb/diagnosis)
- sub_type: 二级分类
- confidence: 置信度(0-1)
- wellness_level: 养生级别(仅wellness时有效，l1或l2)
- sentiment: 情感分析对象
  - polarity: positive/neutral/negative
  - anxiety_score: 0-1
  - urgency: low/medium/high
- entities: 提取的实体对象
  - symptoms: 症状列表
  - herbs: 药材列表
  - prescriptions: 方剂列表
  - body_parts: 身体部位
  - syndromes: 证型
  - books: 古籍名
  - triggers: 诱因
- reasoning: 分类理由
- suggested_follow_up: 建议追问(信息不足时)
- requires_image: 是否需要图片(舌诊时为true)
"""


class IntentClassifier:
    """
    LLM意图分类器

    使用大模型进行深度意图分类，支持：
    1. 联合意图+实体提取（Joint Intent + Slot Filling）
    2. 情感分析
    3. 上下文感知分类

    使用 create_intent_classifier() 工厂函数创建实例
    """

    def __init__(self, llm: BaseChatModel):
        """
        初始化分类器

        Args:
            llm: LangChain BaseChatModel 实例
        """
        self.llm = llm

    async def classify(
        self,
        query: str,
        context: Optional[EnrichedContext] = None,
        has_image: bool = False,
    ) -> IntentClassification:
        """
        分类用户意图

        Args:
            query: 用户输入
            context: 增强上下文
            has_image: 是否包含图片

        Returns:
            IntentClassification: 分类结果
        """
        # 构建提示词
        user_profile_str = ""
        environment_str = ""

        if context:
            if context.user_profile:
                profile = context.user_profile
                user_profile_str = f"""
- 性别: {profile.gender or '未知'}
- 年龄段: {profile.age_group or '未知'}
- 体质: {profile.constitution or '未知'}
- 慢性病史: {', '.join(profile.chronic_conditions) if profile.chronic_conditions else '无'}
"""
            if context.environment:
                env = context.environment
                environment_str = f"""
- 节气: {env.solar_term or '未知'}
- 季节: {env.season or '未知'}
- 地域: {env.region or '未知'}
"""

        system_prompt = INTENT_CLASSIFICATION_SYSTEM_PROMPT.format(
            user_profile=user_profile_str or "未提供",
            environment=environment_str or "未提供",
        )

        user_prompt = f"""请分析以下用户输入：

用户输入: {query}
是否包含图片: {'是' if has_image else '否'}

请输出JSON格式的意图分类结果。"""

        # 使用结构化输出
        llm_with_structure = self.llm.with_structured_output(IntentClassification)

        try:
            result = await llm_with_structure.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            if result is None:
                raise ValueError("结构化意图分类返回空结果")
            if not isinstance(result, IntentClassification):
                result = IntentClassification.model_validate(result)
        except Exception as structured_error:
            logger.warning(
                "结构化意图分类失败，尝试解析普通 JSON 响应: %s",
                structured_error,
            )
            try:
                raw_result = await self.llm.ainvoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ])
                result = _parse_json_classification(raw_result.content)
            except Exception as fallback_error:
                logger.warning("普通 JSON 意图分类也失败: %s", fallback_error)
                # 最终降级：保持低置信度，由上层要求用户澄清。
                return IntentClassification(
                    primary_intent=IntentType.WELLNESS,
                    sub_type="complex",
                    confidence=0.3,
                    wellness_level=WellnessLevel.L2,
                    reasoning=(
                        "分类失败，降级处理: "
                        f"{type(structured_error).__name__} / "
                        f"{type(fallback_error).__name__}"
                    ),
                    route_source="llm",
                    sentiment=SentimentAnalysis(),
                    entities=ExtractedEntities(),
                )

        # 标记路由来源
        result.route_source = "llm"
        result.has_image = has_image

        # 如果有图片且意图不明确，倾向于舌诊分析（diagnosis.tongue）
        if has_image and result.confidence < 0.8:
            result.primary_intent = IntentType.DIAGNOSIS
            result.sub_type = "tongue"
            result.requires_image = True
            result.confidence = 0.85

        return result

    async def classify_with_fallback(
        self,
        query: str,
        rule_result: Optional[IntentClassification],
        context: Optional[EnrichedContext] = None,
        has_image: bool = False,
    ) -> IntentClassification:
        """
        带降级的分类

        如果规则层已有高置信度结果，直接使用；否则调用LLM

        Args:
            query: 用户输入
            rule_result: 规则层结果
            context: 上下文
            has_image: 是否有图片

        Returns:
            IntentClassification: 分类结果
        """
        # 规则层高置信度结果直接使用
        if rule_result and rule_result.confidence >= 0.85:
            return rule_result

        # LLM分类
        llm_result = await self.classify(query, context, has_image)

        # 如果规则层有结果但置信度不够高，比较两者
        if rule_result:
            # 模型鉴权、网关或结构化输出失败时，已有规则结果优先于低置信度
            # 的通用 wellness 占位，避免把药材/方剂/症状咨询误判为 OOS。
            if (
                llm_result.confidence < 0.5
                and str(llm_result.reasoning or "").startswith("分类失败")
            ):
                rule_result.confidence = max(rule_result.confidence, 0.85)
                rule_result.reasoning = (
                    f"{rule_result.reasoning}；LLM 分类不可用，采用确定性规则降级"
                )
                return rule_result

            # 规则层实体提取通常更准确，合并实体
            merged_entities = self._merge_entities(
                rule_result.entities,
                llm_result.entities
            )
            llm_result.entities = merged_entities

            # 如果规则层和LLM结果一致，提高置信度
            if rule_result.primary_intent == llm_result.primary_intent:
                llm_result.confidence = min(1.0, llm_result.confidence + 0.1)

        return llm_result

    def _merge_entities(
        self,
        entities1: ExtractedEntities,
        entities2: ExtractedEntities
    ) -> ExtractedEntities:
        """合并两个实体提取结果"""
        return ExtractedEntities(
            herbs=list(set(entities1.herbs + entities2.herbs)),
            prescriptions=list(set(entities1.prescriptions + entities2.prescriptions)),
            symptoms=list(set(entities1.symptoms + entities2.symptoms)),
            body_parts=list(set(entities1.body_parts + entities2.body_parts)),
            syndromes=list(set(entities1.syndromes + entities2.syndromes)),
            books=list(set(entities1.books + entities2.books)),
            symptom_nature=list(set(entities1.symptom_nature + entities2.symptom_nature)),
            triggers=list(set(entities1.triggers + entities2.triggers)),
            duration=entities1.duration or entities2.duration,
        )


# 工厂函数
def create_intent_classifier(
    provider_name: str = "openai",
    model_name: str = "gpt-4o-mini",
    api_key: str = "",
    base_url: Optional[str] = None,
    top_p: Optional[float] = None,
    temperature: Optional[float] = None,
) -> IntentClassifier:
    """
    创建意图分类器

    Args:
        provider_name: 提供商名称，默认 "openai"
        model_name: 模型名称，默认 "gpt-4o-mini"
        api_key: API Key
        base_url: Base URL (可选，不提供则使用默认值)

    Returns:
        IntentClassifier: 配置好的意图分类器实例

    Example:
        classifier = create_intent_classifier(
            provider_name="openai",
            model_name="gpt-4o-mini",
            api_key="sk-xxx"
        )
        result = await classifier.classify(query)
    """
    from app.src.core.language_model.llm_provider import get_intent_classifier_llm

    llm = get_intent_classifier_llm(
        provider_name=provider_name,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        top_p=top_p,
        temperature=temperature
    )

    return IntentClassifier(llm=llm)
