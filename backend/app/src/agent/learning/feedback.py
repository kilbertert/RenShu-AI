"""
反馈收集模块 (Feedback Collection)

收集、分析和聚合用户反馈
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """反馈类型"""
    RATING = "rating"               # 评分 (1-5星)
    THUMBS_UP_DOWN = "thumbs"       # 点赞/点踩
    CORRECTION = "correction"       # 纠错
    SUGGESTION = "suggestion"       # 建议
    SYMPTOM_TRACKING = "symptom"    # 症状跟踪
    OUTCOME_REPORT = "outcome"      # 治疗结果报告


@dataclass
class UserFeedback:
    """用户反馈"""
    feedback_type: FeedbackType
    session_id: str
    user_id: str

    # 反馈内容
    rating: Optional[int] = None            # 评分 (1-5)
    thumbs_up: Optional[bool] = None        # 点赞/点踩
    correction: Optional[str] = None        # 纠错内容
    suggestion: Optional[str] = None        # 建议
    symptom_improvement: Optional[int] = None  # 症状改善 (1-5)
    treatment_outcome: Optional[str] = None   # 治疗结果

    # 上下文信息
    query: Optional[str] = None
    agent_response: Optional[str] = None
    agent_diagnosis: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # 验证评分范围
        if self.rating is not None:
            self.rating = max(1, min(5, self.rating))
        if self.symptom_improvement is not None:
            self.symptom_improvement = max(1, min(5, self.symptom_improvement))

    @property
    def sentiment(self) -> str:
        """情感倾向"""
        if self.rating:
            if self.rating >= 4:
                return "positive"
            elif self.rating <= 2:
                return "negative"
            else:
                return "neutral"
        if self.thumbs_up is not None:
            return "positive" if self.thumbs_up else "negative"
        if self.symptom_improvement:
            if self.symptom_improvement >= 4:
                return "positive"
            elif self.symptom_improvement <= 2:
                return "negative"
        return "neutral"

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "feedback_type": self.feedback_type.value,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "rating": self.rating,
            "thumbs_up": self.thumbs_up,
            "correction": self.correction,
            "suggestion": self.suggestion,
            "symptom_improvement": self.symptom_improvement,
            "treatment_outcome": self.treatment_outcome,
            "sentiment": self.sentiment,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class FeedbackCollector:
    """反馈收集器"""

    def __init__(self):
        self.feedbacks: List[UserFeedback] = []
        self._session_feedbacks: Dict[str, List[UserFeedback]] = {}

    def collect(
        self,
        feedback_type: FeedbackType,
        session_id: str,
        user_id: str,
        **kwargs
    ) -> UserFeedback:
        """
        收集反馈

        Args:
            feedback_type: 反馈类型
            session_id: 会话ID
            user_id: 用户ID
            **kwargs: 反馈内容

        Returns:
            UserFeedback 对象
        """
        feedback = UserFeedback(
            feedback_type=feedback_type,
            session_id=session_id,
            user_id=user_id,
            **kwargs
        )

        self.feedbacks.append(feedback)

        # 按会话分组
        if session_id not in self._session_feedbacks:
            self._session_feedbacks[session_id] = []
        self._session_feedbacks[session_id].append(feedback)

        logger.info(
            f"[Feedback] Collected {feedback_type.value} from {user_id}: "
            f"sentiment={feedback.sentiment}"
        )

        return feedback

    def get_session_feedbacks(self, session_id: str) -> List[UserFeedback]:
        """获取会话的所有反馈"""
        return self._session_feedbacks.get(session_id, [])

    def get_user_feedbacks(self, user_id: str, limit: int = 100) -> List[UserFeedback]:
        """获取用户的所有反馈"""
        user_feedbacks = [f for f in self.feedbacks if f.user_id == user_id]
        return sorted(user_feedbacks, key=lambda f: f.timestamp, reverse=True)[:limit]

    def get_feedback_stats(self, session_id: Optional[str] = None) -> Dict:
        """
        获取反馈统计

        Args:
            session_id: 会话ID (可选，不指定则统计全部)

        Returns:
            统计信息
        """
        feedbacks = self.get_session_feedbacks(session_id) if session_id else self.feedbacks

        if not feedbacks:
            return {
                "total": 0,
                "average_rating": None,
                "sentiment_distribution": {}
            }

        # 评分统计
        ratings = [f.rating for f in feedbacks if f.rating is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else None

        # 情感分布
        sentiment_dist: Dict[str, int] = {}
        for f in feedbacks:
            sentiment_dist[f.sentiment] = sentiment_dist.get(f.sentiment, 0) + 1

        # 症状改善统计
        improvements = [f.symptom_improvement for f in feedbacks if f.symptom_improvement is not None]
        avg_improvement = sum(improvements) / len(improvements) if improvements else None

        return {
            "total": len(feedbacks),
            "average_rating": avg_rating,
            "sentiment_distribution": sentiment_dist,
            "average_symptom_improvement": avg_improvement,
            "positive_rate": sentiment_dist.get("positive", 0) / len(feedbacks) if feedbacks else 0
        }


class FeedbackAggregator:
    """反馈聚合器"""

    def __init__(self, collector: FeedbackCollector):
        self.collector = collector

    def aggregate_by_period(
        self,
        hours: int = 24,
        user_id: Optional[str] = None
    ) -> Dict:
        """
        按时间段聚合反馈

        Args:
            hours: 时间窗口 (小时)
            user_id: 用户ID (可选)

        Returns:
            聚合结果
        """
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(hours=hours)

        feedbacks = [
            f for f in self.collector.feedbacks
            if f.timestamp > cutoff and (user_id is None or f.user_id == user_id)
        ]

        return {
            "period_hours": hours,
            "total_feedbacks": len(feedbacks),
            "by_type": self._count_by_field(feedbacks, "feedback_type"),
            "by_sentiment": self._count_by_field(feedbacks, "sentiment"),
            "average_rating": self._average_rating(feedbacks)
        }

    def identify_patterns(self) -> List[Dict]:
        """识别反馈模式"""
        patterns = []

        # 检查低评分模式
        low_ratings = [f for f in self.collector.feedbacks if f.rating and f.rating <= 2]
        if len(low_ratings) > 5:
            # 分析共同主题
            common_queries = {}
            for f in low_ratings:
                if f.query:
                    common_queries[f.query] = common_queries.get(f.query, 0) + 1

            if common_queries:
                most_common = max(common_queries, key=common_queries.get)
                patterns.append({
                    "type": "low_rating_pattern",
                    "description": f"频繁低评分: {most_common}",
                    "count": len(low_ratings),
                    "severity": "high" if len(low_ratings) > 10 else "medium"
                })

        # 检查纠错模式
        corrections = [f for f in self.collector.feedbacks if f.correction]
        if len(corrections) > 3:
            patterns.append({
                "type": "correction_pattern",
                "description": f"用户纠错频繁: {len(corrections)}次",
                "count": len(corrections),
                "severity": "medium"
            })

        # 检查症状改善模式
        improvements = [f for f in self.collector.feedbacks if f.symptom_improvement]
        if improvements:
            avg_improvement = sum(f.symptom_improvement for f in improvements) / len(improvements)
            if avg_improvement >= 4:
                patterns.append({
                    "type": "high_improvement",
                    "description": f"症状改善良好: 平均{avg_improvement:.1f}/5",
                    "count": len(improvements),
                    "severity": "positive"
                })

        return patterns

    def _count_by_field(self, feedbacks: List[UserFeedback], field: str) -> Dict:
        """按字段计数"""
        result: Dict = {}
        for f in feedbacks:
            value = getattr(f, field, None)
            if value:
                key = value.value if isinstance(value, Enum) else value
                result[key] = result.get(key, 0) + 1
        return result

    def _average_rating(self, feedbacks: List[UserFeedback]) -> Optional[float]:
        """计算平均评分"""
        ratings = [f.rating for f in feedbacks if f.rating is not None]
        return sum(ratings) / len(ratings) if ratings else None
