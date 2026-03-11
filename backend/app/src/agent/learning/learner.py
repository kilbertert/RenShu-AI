"""
自学习器 (Self-Learner)

整合反馈、反思、进化三个层次的学习机制
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .feedback import FeedbackCollector, FeedbackType, UserFeedback, FeedbackAggregator
from .reflection import SelfReflection, ReflectionType
from .evolution import EvolutionEngine, EvolutionStrategy, PerformanceMetrics


logger = logging.getLogger(__name__)


@dataclass
class LearningConfig:
    """学习配置"""
    # 反馈收集
    enable_feedback: bool = True
    feedback_threshold: int = 5  # 触发学习的反馈阈值

    # 自我反思
    enable_reflection: bool = True
    reflect_on_error: bool = True
    reflect_on_feedback: bool = True

    # 长期进化
    enable_evolution: bool = True
    evolution_threshold: float = 0.7  # 性能阈值
    evolution_check_interval: int = 100  # 检查间隔（交互次数）

    # 其他
    max_feedback_storage: int = 1000
    max_reflection_storage: int = 100
    learning_rate: float = 0.1  # 学习率


@dataclass
class LearningSession:
    """学习会话"""
    session_id: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    # 会话统计
    total_interactions: int = 0
    successful_interactions: int = 0
    total_feedback: int = 0

    # 学习活动
    reflections_count: int = 0
    evolutions_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_interactions": self.total_interactions,
            "successful_interactions": self.successful_interactions,
            "total_feedback": self.total_feedback,
            "reflections_count": self.reflections_count,
            "evolutions_count": self.evolutions_count,
            "success_rate": self.successful_interactions / self.total_interactions if self.total_interactions > 0 else 0
        }


@dataclass
class LearningReport:
    """学习报告"""
    timestamp: datetime = field(default_factory=datetime.now)

    # 反馈摘要
    feedback_summary: Dict = field(default_factory=dict)

    # 反思摘要
    reflection_summary: Dict = field(default_factory=dict)

    # 进化摘要
    evolution_summary: Dict = field(default_factory=dict)

    # 整体评估
    overall_assessment: str = ""
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "feedback_summary": self.feedback_summary,
            "reflection_summary": self.reflection_summary,
            "evolution_summary": self.evolution_summary,
            "overall_assessment": self.overall_assessment,
            "recommendations": self.recommendations
        }


class SelfLearner:
    """
    自学习器

    三层学习架构：
    1. 反馈层 (Feedback): 收集和分析用户反馈
    2. 反思层 (Reflection): 对失败和反馈进行自我反思
    3. 进化层 (Evolution): 长期性能优化和策略调整
    """

    def __init__(self, config: Optional[LearningConfig] = None, llm=None):
        self.config = config or LearningConfig()

        # 初始化三个层次的组件
        self.feedback_collector = FeedbackCollector()
        self.feedback_aggregator = FeedbackAggregator(self.feedback_collector)
        self.reflection = SelfReflection(llm=llm)
        self.evolution = EvolutionEngine()

        # 会话管理
        self.sessions: Dict[str, LearningSession] = {}
        self.current_session: Optional[LearningSession] = None

        # 交互计数
        self.interaction_count = 0

        logger.info("[SelfLearner] Initialized with 3-layer learning architecture")

    def start_session(self, session_id: str) -> LearningSession:
        """开始新的学习会话"""
        session = LearningSession(session_id=session_id)
        self.sessions[session_id] = session
        self.current_session = session
        logger.info(f"[SelfLearner] Started session: {session_id}")
        return session

    def end_session(self, session_id: Optional[str] = None) -> LearningSession:
        """结束学习会话"""
        sid = session_id or (self.current_session.session_id if self.current_session else None)
        if sid and sid in self.sessions:
            session = self.sessions[sid]
            session.end_time = datetime.now()
            if self.current_session and self.current_session.session_id == sid:
                self.current_session = None
            return session
        return None

    def record_interaction(
        self,
        session_id: str,
        success: bool,
        context: Optional[Dict] = None
    ) -> None:
        """记录交互"""
        self.interaction_count += 1

        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.total_interactions += 1
            if success:
                session.successful_interactions += 1

        # 更新进化指标
        metrics_update = {"success_rate": success}
        self.evolution.update_metrics(metrics_update)

        # 检查是否需要进化
        if self.config.enable_evolution and self.interaction_count % self.config.evolution_check_interval == 0:
            if self.evolution.should_evolve(self.config.evolution_threshold):
                logger.info("[SelfLearner] Evolution threshold reached")

    async def collect_feedback(
        self,
        session_id: str,
        user_id: str,
        feedback_type: FeedbackType,
        **kwargs
    ) -> UserFeedback:
        """
        收集用户反馈

        Args:
            session_id: 会话ID
            user_id: 用户ID
            feedback_type: 反馈类型
            **kwargs: 反馈内容

        Returns:
            反馈对象
        """
        feedback = self.feedback_collector.collect(
            feedback_type=feedback_type,
            session_id=session_id,
            user_id=user_id,
            **kwargs
        )

        # 更新会话统计
        if session_id in self.sessions:
            self.sessions[session_id].total_feedback += 1

        # 检查是否需要触发反思
        if self.config.enable_reflection:
            await self._check_reflection_trigger(feedback)

        return feedback

    async def _check_reflection_trigger(self, feedback: UserFeedback) -> None:
        """检查是否需要触发反思"""
        # 低评分触发反思
        if feedback.rating and feedback.rating <= 2:
            if feedback.agent_response:
                await self.reflection.reflect_on_feedback(
                    original_response=feedback.agent_response,
                    feedback=feedback.correction or feedback.suggestion or "低评分反馈",
                    feedback_type=feedback.feedback_type.value
                )
                if feedback.session_id in self.sessions:
                    self.sessions[feedback.session_id].reflections_count += 1

        # 纠错触发反思
        if feedback.correction:
            if feedback.agent_response:
                await self.reflection.reflect_on_feedback(
                    original_response=feedback.agent_response,
                    feedback=feedback.correction,
                    feedback_type="correction"
                )

    async def reflect_on_error(
        self,
        session_id: str,
        goal: str,
        result: str,
        error: str
    ) -> None:
        """对错误进行反思"""
        if not self.config.enable_reflection:
            return

        reflection = await self.reflection.reflect_on_error(
            goal=goal,
            result=result,
            error=error
        )

        if session_id in self.sessions:
            self.sessions[session_id].reflections_count += 1

        logger.info(f"[SelfLearner] Error reflection completed for session {session_id}")

    async def reflect_on_completion(
        self,
        session_id: str,
        goal: str,
        result: str,
        feedback: Optional[str] = None
    ) -> None:
        """对完成进行反思"""
        if not self.config.enable_reflection:
            return

        reflection = await self.reflection.reflect_on_completion(
            goal=goal,
            result=result,
            feedback=feedback
        )

        if session_id in self.sessions:
            self.sessions[session_id].reflections_count += 1

    async def trigger_evolution(
        self,
        context: Optional[Dict] = None
    ) -> Optional[EvolutionStrategy]:
        """触发进化"""
        if not self.config.enable_evolution:
            return None

        # 推荐策略
        strategy = self.evolution.recommend_strategy()
        if not strategy:
            logger.info("[SelfLearner] No evolution needed")
            return None

        # 执行进化
        record = await self.evolution.evolve(strategy, context)

        # 更新会话统计
        if self.current_session:
            self.current_session.evolutions_count += 1

        return strategy

    def generate_report(self, session_id: Optional[str] = None) -> LearningReport:
        """生成学习报告"""
        report = LearningReport()

        # 反馈摘要
        if session_id:
            report.feedback_summary = self.feedback_collector.get_feedback_stats(session_id)
        else:
            # 全局摘要
            report.feedback_summary = self.feedback_aggregator.aggregate_by_period(hours=24)

        # 反思摘要
        report.reflection_summary = self.reflection.get_reflection_summary()

        # 进化摘要
        report.evolution_summary = self.evolution.get_evolution_summary()

        # 识别模式
        patterns = self.feedback_aggregator.identify_patterns()
        if patterns:
            report.recommendations.extend([
                f"检测到模式: {p['description']}" for p in patterns
            ])

        # 整体评估
        overall_score = self.evolution.current_metrics.calculate_overall()
        if overall_score >= 0.8:
            report.overall_assessment = "excellent"
        elif overall_score >= 0.7:
            report.overall_assessment = "good"
        elif overall_score >= 0.6:
            report.overall_assessment = "fair"
        else:
            report.overall_assessment = "needs_improvement"

        return report

    def get_session(self, session_id: str) -> Optional[LearningSession]:
        """获取会话信息"""
        return self.sessions.get(session_id)

    def get_all_sessions(self) -> List[LearningSession]:
        """获取所有会话"""
        return list(self.sessions.values())
