"""
Wellness Router - 养生L1/L2分流路由
根据情感分析判定路由级别

借鉴大厂做法：
- 阿里小蜜：识别隐性需求与情绪变化
- 美团：退单纠纷语义理解，情绪识别
"""

from typing import Optional
from app.src.agent.intent_recognition.schemas import (
    IntentRouteResult,
    IntentClassification,
    IntentType,
    OOSResult,
    OOSReason,
    EnrichedContext,
    UserProfile,
    EnvironmentContext,
    ConversationContext, WellnessLevel, SentimentAnalysis,
)

class WellnessRouter:
    """
    养生路由器

    核心功能：
    1. 根据情感分析判定L1/L2级别
    2. L1: 直接LLM回答（简单养生）
    3. L2: Web Search + LLM（复杂养生）
    """

    # L1判定条件阈值
    L1_ANXIETY_THRESHOLD = 0.3
    L1_URGENCY = "low"

    # 复杂养生关键词（触发L2）
    L2_KEYWORDS = [
        # 体质相关
        "气虚", "血虚", "阴虚", "阳虚", "痰湿", "湿热", "血瘀", "气郁", "特禀",
        "体质调理", "体质改善",
        # 慢性问题
        "长期", "一直", "总是", "经常", "反复",
        # 专业术语
        "辨证", "证型", "配伍", "君臣佐使",
        # 焦虑词
        "担心", "害怕", "焦虑", "会不会", "是不是",
        # 复杂问题
        "该怎么办", "怎么调理", "如何改善",
    ]

    # 简单养生关键词（强化L1）
    L1_KEYWORDS = [
        # 季节养生
        "春季", "夏季", "秋季", "冬季", "春天", "夏天", "秋天", "冬天",
        # 节气养生
        "立春", "立夏", "立秋", "立冬", "冬至", "夏至",
        # 通用养生
        "日常", "平时", "一般", "简单",
        # 询问类型
        "吃什么好", "注意什么", "怎么养生",
    ]

    def __init__(self):
        """初始化路由器"""
        pass

    def determine_level(
        self,
        classification: IntentClassification,
        query: str,
        context: Optional[EnrichedContext] = None,
    ) -> WellnessLevel:
        """
        判定养生路由级别

        Args:
            classification: 意图分类结果
            query: 用户查询
            context: 上下文

        Returns:
            WellnessLevel: L1或L2
        """
        # 如果已有wellness_level，直接使用
        if classification.wellness_level:
            return classification.wellness_level

        # 情感分析判定
        sentiment = classification.sentiment
        is_l1_by_sentiment = self._check_sentiment_for_l1(sentiment)

        # 关键词判定
        has_l2_keywords = self._has_l2_keywords(query)
        has_l1_keywords = self._has_l1_keywords(query)

        # 症状判定（有具体症状倾向L2）
        has_symptoms = len(classification.entities.symptoms) > 0

        # 综合判定
        if has_symptoms:
            # 有症状 -> L2（可能需要更详细的分析）
            return WellnessLevel.L2

        if has_l2_keywords:
            # 有L2关键词 -> L2
            return WellnessLevel.L2

        if is_l1_by_sentiment and has_l1_keywords:
            # 情感平和 + L1关键词 -> L1
            return WellnessLevel.L1

        if is_l1_by_sentiment and not has_l2_keywords:
            # 情感平和 + 无L2关键词 -> L1
            return WellnessLevel.L1

        # 默认L2（更谨慎）
        return WellnessLevel.L2

    def _check_sentiment_for_l1(self, sentiment: SentimentAnalysis) -> bool:
        """检查情感是否符合L1条件"""
        return (
            sentiment.anxiety_score < self.L1_ANXIETY_THRESHOLD
            and sentiment.urgency == self.L1_URGENCY
        )

    def _has_l2_keywords(self, query: str) -> bool:
        """检查是否包含L2关键词"""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.L2_KEYWORDS)

    def _has_l1_keywords(self, query: str) -> bool:
        """检查是否包含L1关键词"""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.L1_KEYWORDS)

    def get_routing_advice(self, level: WellnessLevel) -> dict:
        """
        获取路由建议

        Args:
            level: 路由级别

        Returns:
            dict: 路由建议配置
        """
        if level == WellnessLevel.L1:
            return {
                "level": "L1",
                "strategy": "direct_llm",
                "description": "直接使用LLM回答",
                "temperature": 0.7,
                "max_tokens": 1000,
                "use_web_search": False,
            }
        else:
            return {
                "level": "L2",
                "strategy": "search_enhanced",
                "description": "Web Search + RAG + LLM",
                "temperature": 0.5,
                "max_tokens": 2000,
                "use_web_search": True,
                "search_sites": [
                    "www.cntcm.com.cn",  # 中国中医药网
                    "www.satcm.gov.cn",  # 国家中医药管理局
                ],
            }


# 单例
_wellness_router: Optional[WellnessRouter] = None


def get_wellness_router() -> WellnessRouter:
    """获取养生路由器单例"""
    global _wellness_router
    if _wellness_router is None:
        _wellness_router = WellnessRouter()
    return _wellness_router
