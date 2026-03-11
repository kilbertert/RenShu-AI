"""
中医问诊自学习器 (TCM Self-Learner)

整合反馈、反思、进化三个层次的学习机制，持续提升诊断准确率

架构：
- 单线程学习：当前对话中的即时纠偏和优化
- 跨线程学习：从历史案例中提取可泛化的模式和规则
"""

import logging
from typing import Any, Dict, List, Optional, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from .feedback import TCMFeedbackCollector, TCMFeedbackType, TCMUserFeedback, TCMFeedbackAggregator
from .reflection import TCMReflectionEngine, TCMReflectionType, TCMReflectionResult
from .evolution import AccuracyEvolutionEngine, AccuracyMetrics, EvolutionStrategy, EvolutionRecord
from .events import (
    IntentRouteCorrection,
    LearningEvent,
    LearningEventInput,
    LearningEventType,
    ThreadLearningContext,
    ToolSelectionCorrection,
    UserOutputCorrection,
)


logger = logging.getLogger(__name__)


@dataclass
class LearningConfig:
    """学习配置"""
    # 反馈收集
    enable_feedback: bool = True
    feedback_threshold: int = 5

    # 反思
    enable_reflection: bool = True
    reflect_on_error: bool = True
    reflect_on_feedback: bool = True

    # 进化
    enable_evolution: bool = True
    evolution_threshold: float = 0.7
    evolution_check_interval: int = 100

    # 存储
    max_feedback_storage: int = 1000
    max_reflection_storage: int = 100

    # 线程学习
    enable_thread_learning: bool = True
    max_event_history: int = 50
    max_intent_corrections: int = 10
    max_tool_learning_items: int = 10
    max_correction_learning_items: int = 10
    max_snapshot_items: int = 3

    # 准确率阈值
    min_intent_accuracy: float = 0.90
    min_diagnosis_accuracy: float = 0.80
    min_tool_selection_accuracy: float = 0.85
    min_inquiry_efficiency: float = 0.70


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

    # 准确率指标
    intent_accuracy: Optional[float] = None
    diagnosis_accuracy: Optional[float] = None
    tool_selection_accuracy: Optional[float] = None
    inquiry_efficiency: Optional[float] = None

    # 诊断信息
    complexity_score: Optional[float] = None
    syndrome_type: Optional[str] = None
    diagnosis_success: bool = False
    user_satisfaction: Optional[float] = None

    # 纠正次数
    total_corrections: int = 0

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
            "success_rate": self.successful_interactions / self.total_interactions if self.total_interactions > 0 else 0,
            "intent_accuracy": self.intent_accuracy,
            "diagnosis_accuracy": self.diagnosis_accuracy,
            "tool_selection_accuracy": self.tool_selection_accuracy,
            "inquiry_efficiency": self.inquiry_efficiency,
            "complexity_score": self.complexity_score,
            "syndrome_type": self.syndrome_type,
            "diagnosis_success": self.diagnosis_success,
            "user_satisfaction": self.user_satisfaction,
            "total_corrections": self.total_corrections
        }


@dataclass
class LearningReport:
    """学习报告"""
    timestamp: datetime = field(default_factory=datetime.now)
    feedback_summary: Dict = field(default_factory=dict)
    reflection_summary: Dict = field(default_factory=dict)
    evolution_summary: Dict = field(default_factory=dict)
    thread_learning_summary: Dict = field(default_factory=dict)
    overall_assessment: str = ""
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "feedback_summary": self.feedback_summary,
            "reflection_summary": self.reflection_summary,
            "evolution_summary": self.evolution_summary,
            "thread_learning_summary": self.thread_learning_summary,
            "overall_assessment": self.overall_assessment,
            "recommendations": self.recommendations
        }


@dataclass
class ThreadLearningState:
    """线程内自学习状态"""
    session_id: str
    recent_error_reflections: List[str] = field(default_factory=list)
    recent_corrections: List[str] = field(default_factory=list)
    recent_effective_strategies: List[str] = field(default_factory=list)
    current_thread_summary: str = ""
    thread_learning_context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        data = {
            "session_id": self.session_id,
            "recent_error_reflections": self.recent_error_reflections.copy(),
            "recent_corrections": self.recent_corrections.copy(),
            "recent_effective_strategies": self.recent_effective_strategies.copy(),
            "current_thread_summary": self.current_thread_summary,
            "thread_learning_context": self.thread_learning_context.copy(),
        }
        if self.thread_learning_context:
            data.update(self.thread_learning_context)
        return data


class SelfLearner:
    """
    中医问诊自学习器

    三层学习架构：
    1. 反馈层 (Feedback): 收集准确率相关的用户反馈
    2. 反思层 (Reflection): 分析错误原因，生成改进规则
    3. 进化层 (Evolution): 从案例中提取知识，提升准确率
    """

    def __init__(self, config: Optional[LearningConfig] = None, llm=None, db_session=None, vector_store=None):
        self.config = config or LearningConfig()

        # 三大核心模块
        self.feedback_collector = TCMFeedbackCollector()
        self.feedback_aggregator = TCMFeedbackAggregator(self.feedback_collector)
        self.reflection = TCMReflectionEngine(llm=llm)
        self.evolution = AccuracyEvolutionEngine(llm=llm)

        # 存储层 (可选)
        self.db_session = db_session
        self.storage = None
        if db_session:
            from .storage import (
                ThreadLearningStorage,
                FeedbackStorage,
                ReflectionStorage,
                CrossThreadKnowledgeStorage,
                EvolutionStorage,
            )
            self.storage = {
                "thread": ThreadLearningStorage(db_session),
                "feedback": FeedbackStorage(db_session),
                "reflection": ReflectionStorage(db_session),
                "cross_thread": CrossThreadKnowledgeStorage(db_session, vector_store),
                "evolution": EvolutionStorage(db_session),
            }

        # 会话管理
        self.sessions: Dict[str, LearningSession] = {}
        self.current_session: Optional[LearningSession] = None

        # 线程学习
        self.thread_contexts: Dict[str, ThreadLearningContext] = {}
        self.thread_learning: Dict[str, ThreadLearningState] = {}
        self.learning_events: Dict[str, List[LearningEvent]] = {}

        # 交互计数
        self.interaction_count = 0

        logger.info("[SelfLearner] Initialized with TCM accuracy-focused learning")

    # ========== 会话管理 ==========

    def _ensure_session(self, session_id: str) -> LearningSession:
        """确保会话与线程学习状态存在"""
        session = self.sessions.get(session_id)
        if session is None:
            session = LearningSession(session_id=session_id)
            self.sessions[session_id] = session

        if session_id not in self.thread_learning:
            self.thread_learning[session_id] = ThreadLearningState(session_id=session_id)
        if session_id not in self.thread_contexts:
            self.thread_contexts[session_id] = ThreadLearningContext(conversation_id=session_id)

        self.current_session = session
        return session

    def start_session(self, session_id: str) -> LearningSession:
        """开始新的学习会话"""
        session = self._ensure_session(session_id)
        logger.info(f"[SelfLearner] Started session: {session_id}")
        return session

    def end_session(self, session_id: Optional[str] = None, user_id: Optional[str] = None) -> Optional[LearningSession]:
        """结束学习会话"""
        sid = session_id or (self.current_session.session_id if self.current_session else None)
        if sid and sid in self.sessions:
            session = self.sessions[sid]
            session.end_time = datetime.now()

            # 更新会话准确率指标
            accuracy_stats = self.feedback_collector.get_accuracy_stats(sid)
            session.intent_accuracy = accuracy_stats.get("intent_accuracy")
            session.diagnosis_accuracy = accuracy_stats.get("diagnosis_accuracy")
            session.inquiry_efficiency = accuracy_stats.get("inquiry_efficiency")

            # 持久化线程学习上下文
            if self.storage and sid in self.thread_contexts:
                from uuid import UUID
                thread_context = self.thread_contexts[sid]
                learning_context = {
                    "intent_learning": thread_context.intent_learning.to_dict(),
                    "subgraph_learning": {
                        "tool_learning": thread_context.tool_learning,
                        "correction_learning": thread_context.correction_learning,
                    },
                    "thread_summary": self.thread_learning.get(sid, ThreadLearningState(sid)).current_thread_summary,
                    "total_corrections": session.total_corrections,
                    "complexity_score": session.complexity_score,
                    "interaction_rounds": session.total_interactions,
                }

                try:
                    self.storage["thread"].save_thread_learning(
                        conversation_id=UUID(sid),
                        user_id=UUID(user_id) if user_id else UUID(sid),
                        learning_context=learning_context
                    )
                except Exception as e:
                    logger.warning(f"[SelfLearner] Failed to persist thread learning: {e}")

            if self.current_session and self.current_session.session_id == sid:
                self.current_session = None

            logger.info(f"[SelfLearner] Ended session: {sid}")
            return session
        return None

    # ========== 反馈收集 ==========

    def collect_intent_correction(
        self,
        session_id: str,
        user_id: str,
        user_query: str,
        wrong_intent: str,
        correct_intent: str,
        reason: Optional[str] = None
    ) -> TCMUserFeedback:
        """收集意图纠正反馈"""
        self._ensure_session(session_id)

        feedback = self.feedback_collector.collect_intent_correction(
            session_id=session_id,
            user_id=user_id,
            user_query=user_query,
            wrong_intent=wrong_intent,
            correct_intent=correct_intent,
            reason=reason
        )

        self.sessions[session_id].total_feedback += 1
        self.sessions[session_id].total_corrections += 1

        # 持久化反馈
        if self.storage:
            self.storage["feedback"].save_feedback(feedback)

        # 触发反思
        if self.config.enable_reflection:
            import asyncio
            asyncio.create_task(
                self._reflect_on_intent_error(session_id, user_query, wrong_intent, correct_intent)
            )

        return feedback

    def collect_diagnosis_error(
        self,
        session_id: str,
        user_id: str,
        subgraph: str,
        agent_syndrome: str,
        correct_syndrome: str,
        error_type: str,
        symptoms: Optional[Dict] = None,
        missed_symptoms: List[str] = None
    ) -> TCMUserFeedback:
        """收集诊断错误反馈"""
        self._ensure_session(session_id)

        feedback = self.feedback_collector.collect_diagnosis_error(
            session_id=session_id,
            user_id=user_id,
            subgraph=subgraph,
            agent_syndrome=agent_syndrome,
            correct_syndrome=correct_syndrome,
            error_type=error_type,
            missed_symptoms=missed_symptoms
        )

        self.sessions[session_id].total_feedback += 1
        self.sessions[session_id].total_corrections += 1

        # 持久化反馈
        if self.storage:
            self.storage["feedback"].save_feedback(feedback)

        # 触发反思
        if self.config.enable_reflection:
            import asyncio
            asyncio.create_task(
                self._reflect_on_diagnosis_error(
                    session_id, symptoms or {}, agent_syndrome, correct_syndrome, error_type
                )
            )

        return feedback

    def collect_tool_selection_error(
        self,
        session_id: str,
        user_id: str,
        subgraph: str,
        wrong_tool: str,
        correct_tool: str,
        reason: str,
        missing_info: List[str] = None
    ) -> TCMUserFeedback:
        """收集工具选择错误反馈"""
        self._ensure_session(session_id)

        feedback = self.feedback_collector.collect_tool_selection_error(
            session_id=session_id,
            user_id=user_id,
            subgraph=subgraph,
            wrong_tool=wrong_tool,
            correct_tool=correct_tool,
            reason=reason,
            missing_info=missing_info
        )

        self.sessions[session_id].total_feedback += 1
        self.sessions[session_id].total_corrections += 1

        # 持久化反馈
        if self.storage:
            self.storage["feedback"].save_feedback(feedback)

        return feedback

    def collect_inquiry_inefficiency(
        self,
        session_id: str,
        user_id: str,
        subgraph: str,
        redundant_questions: List[str],
        missed_info: List[str],
        actual_rounds: int,
        expected_rounds: int
    ) -> TCMUserFeedback:
        """收集追问低效反馈"""
        self._ensure_session(session_id)

        feedback = self.feedback_collector.collect_inquiry_inefficiency(
            session_id=session_id,
            user_id=user_id,
            subgraph=subgraph,
            redundant_questions=redundant_questions,
            missed_info=missed_info,
            actual_rounds=actual_rounds,
            expected_rounds=expected_rounds
        )

        self.sessions[session_id].total_feedback += 1

        # 持久化反馈
        if self.storage:
            self.storage["feedback"].save_feedback(feedback)

        return feedback

    # ========== 反思触发 ==========

    async def _reflect_on_intent_error(
        self,
        session_id: str,
        user_query: str,
        wrong_intent: str,
        correct_intent: str
    ):
        """反思意图识别错误"""
        try:
            reflection = await self.reflection.reflect_on_intent_error(
                session_id=session_id,
                user_query=user_query,
                predicted_intent=wrong_intent,
                correct_intent=correct_intent
            )

            self.sessions[session_id].reflections_count += 1
            self._record_thread_reflection(session_id, reflection)

            # 持久化反思结果
            if self.storage:
                self.storage["reflection"].save_reflection(reflection)

        except Exception as e:
            logger.error(f"[SelfLearner] Intent error reflection failed: {e}")

    async def _reflect_on_diagnosis_error(
        self,
        session_id: str,
        symptoms: Dict,
        agent_syndrome: str,
        correct_syndrome: str,
        error_type: str
    ):
        """反思诊断错误"""
        try:
            reflection = await self.reflection.reflect_on_diagnosis_error(
                session_id=session_id,
                symptoms=symptoms,
                agent_syndrome=agent_syndrome,
                correct_syndrome=correct_syndrome,
                error_type=error_type
            )

            self.sessions[session_id].reflections_count += 1
            self._record_thread_reflection(session_id, reflection)

            # 持久化反思结果
            if self.storage:
                self.storage["reflection"].save_reflection(reflection)

        except Exception as e:
            logger.error(f"[SelfLearner] Diagnosis error reflection failed: {e}")

    def _record_thread_reflection(self, session_id: str, reflection: TCMReflectionResult):
        """将反思结果沉淀到线程内经验"""
        thread_state = self.thread_learning.get(session_id)
        if thread_state is None:
            thread_state = ThreadLearningState(session_id=session_id)
            self.thread_learning[session_id] = thread_state

        reflection_type = reflection.reflection_type

        if reflection_type == TCMReflectionType.INTENT_RECOGNITION_ERROR:
            if reflection.intent_improvement_rule:
                self._append_limited(
                    thread_state.recent_error_reflections,
                    reflection.intent_improvement_rule
                )
                self._append_limited(
                    thread_state.recent_corrections,
                    reflection.intent_improvement_rule
                )

        elif reflection_type == TCMReflectionType.DIAGNOSIS_REASONING_ERROR:
            if reflection.discriminating_rule:
                self._append_limited(
                    thread_state.recent_error_reflections,
                    reflection.discriminating_rule
                )
                self._append_limited(
                    thread_state.recent_effective_strategies,
                    reflection.discriminating_rule
                )

        elif reflection_type == TCMReflectionType.INQUIRY_INEFFICIENCY:
            if reflection.lessons_learned:
                self._append_limited(
                    thread_state.recent_effective_strategies,
                    reflection.lessons_learned
                )

        self._refresh_thread_summary(session_id)

    @staticmethod
    def _normalize_text(value: Optional[Any]) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split())

    @classmethod
    def _append_limited(cls, items: List[str], value: Optional[str], limit: int = 3) -> None:
        """追加非空文本并限制长度"""
        normalized = cls._normalize_text(value)
        if not normalized:
            return

        if normalized in items:
            items.remove(normalized)

        items.append(normalized)
        if len(items) > limit:
            del items[:-limit]

    def _refresh_thread_summary(self, session_id: str) -> None:
        """刷新线程摘要"""
        thread_state = self.thread_learning.get(session_id)
        if thread_state is None:
            return

        summary_parts: List[str] = []

        if thread_state.recent_error_reflections:
            summary_parts.append(f"最近需避免：{thread_state.recent_error_reflections[-1]}")

        if thread_state.recent_corrections:
            summary_parts.append(f"用户刚纠正：{thread_state.recent_corrections[-1]}")

        if thread_state.recent_effective_strategies:
            summary_parts.append(f"本线程有效策略：{thread_state.recent_effective_strategies[-1]}")

        thread_state.current_thread_summary = "；".join(summary_parts)
        thread_state.thread_learning_context = self.get_thread_learning_context(session_id)

    # ========== 线程学习上下文 ==========

    def get_thread_learning_snapshot(self, session_id: str) -> Dict[str, Any]:
        """获取线程内学习快照，供上下文注入使用"""
        thread_state = self.thread_learning.get(session_id)
        if thread_state is None:
            thread_state = ThreadLearningState(session_id=session_id)

        data = thread_state.to_dict()
        if not data.get("thread_learning_context"):
            thread_context = self.get_thread_learning_context(session_id)
            data["thread_learning_context"] = thread_context
            data.update(thread_context)

        return data

    def get_thread_learning_context(
        self,
        conversation_id: str,
        source: Optional[str] = None,
        include_event_history: bool = False,
    ) -> Dict[str, Any]:
        """按需读取结构化线程学习上下文"""
        context = self.thread_contexts.get(conversation_id)
        if context is None:
            data = ThreadLearningContext(conversation_id=conversation_id).to_dict()
        else:
            data = context.to_dict()

        normalized_source = self._normalize_text(source)
        if normalized_source:
            if normalized_source == "main_graph":
                data["tool_learning"] = []
                data["correction_learning"] = []
            else:
                data["tool_learning"] = [
                    item for item in data["tool_learning"] if self._matches_source(item, normalized_source)
                ]
                data["correction_learning"] = [
                    item for item in data["correction_learning"] if self._matches_source(item, normalized_source)
                ]

        if include_event_history:
            data["learning_events"] = self.list_learning_events(
                conversation_id,
                source=normalized_source or None,
            )

        return data

    def _matches_source(self, record: Dict[str, Any], source: str) -> bool:
        normalized = self._normalize_text(source)
        return normalized in {
            self._normalize_text(record.get("source")),
            self._normalize_text(record.get("subgraph")),
        }

    def list_learning_events(
        self,
        conversation_id: str,
        event_type: Optional[str] = None,
        source: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        events = list(self.learning_events.get(conversation_id, []))
        normalized_type = self._normalize_text(event_type)
        normalized_source = self._normalize_text(source)

        if normalized_type:
            events = [event for event in events if event.event_type.value == normalized_type]

        if normalized_source:
            events = [
                event
                for event in events
                if event.source == normalized_source
                or self._normalize_text(event.payload.get("subgraph")) == normalized_source
            ]

        if limit is not None and limit >= 0:
            events = events[-limit:]

        return [event.to_dict() for event in events]

    # ========== 交互记录 ==========

    def record_interaction(
        self,
        session_id: str,
        success: bool,
        context: Optional[Dict] = None
    ) -> None:
        """记录交互"""
        self.interaction_count += 1
        session = self._ensure_session(session_id)
        session.total_interactions += 1
        if success:
            session.successful_interactions += 1

        # 更新进化指标
        accuracy_stats = self.feedback_collector.get_accuracy_stats(session_id)
        metrics_update = {
            "intent_recognition_accuracy": accuracy_stats.get("intent_accuracy") or 0.0,
            "diagnosis_accuracy": accuracy_stats.get("diagnosis_accuracy") or 0.0,
            "inquiry_efficiency": accuracy_stats.get("inquiry_efficiency") or 0.0
        }
        self.evolution.update_metrics(metrics_update)

        # 检查是否需要进化
        if self.config.enable_evolution and self.interaction_count % self.config.evolution_check_interval == 0:
            evolution_needed = self.evolution.should_evolve()
            if evolution_needed:
                logger.info(f"[SelfLearner] Evolution needed: {evolution_needed}")

    # ========== 进化触发 ==========

    async def trigger_evolution(
        self,
        strategy: Optional[EvolutionStrategy] = None
    ) -> Optional[EvolutionRecord]:
        """触发进化"""
        if not self.config.enable_evolution:
            return None

        # 如果没有指定策略，推荐一个
        if strategy is None:
            strategy = self.evolution.recommend_strategy()
            if not strategy:
                logger.info("[SelfLearner] No evolution needed")
                return None

        # 准备进化数据
        data = self._prepare_evolution_data(strategy)

        # 执行进化
        record = await self.evolution.evolve(strategy, data)

        # 更新会话统计
        if self.current_session:
            self.current_session.evolutions_count += 1

        # 持久化进化记录
        if self.storage:
            self.storage["evolution"].save_evolution_record(record)

        return record

    def _prepare_evolution_data(self, strategy: EvolutionStrategy) -> Dict:
        """准备进化所需数据"""
        data = {}

        if strategy == EvolutionStrategy.INTENT_RULES:
            # 收集意图识别错误
            intent_errors = [
                f.to_dict() for f in self.feedback_collector.feedbacks
                if f.feedback_type == TCMFeedbackType.INTENT_CORRECTION
            ]
            data["intent_errors"] = intent_errors

        elif strategy == EvolutionStrategy.TOOL_SELECTION_RULES:
            # 收集工具选择错误
            tool_errors = [
                f.to_dict() for f in self.feedback_collector.feedbacks
                if f.feedback_type == TCMFeedbackType.TOOL_SELECTION_ERROR
            ]
            data["tool_errors"] = tool_errors

        elif strategy == EvolutionStrategy.DISCRIMINATING_RULES:
            # 收集诊断错误
            diagnosis_errors = [
                f.to_dict() for f in self.feedback_collector.feedbacks
                if f.feedback_type == TCMFeedbackType.DIAGNOSIS_ERROR
            ]
            data["diagnosis_errors"] = diagnosis_errors

        elif strategy == EvolutionStrategy.INQUIRY_OPTIMIZATION:
            # 收集成功案例
            successful_cases = [
                session.to_dict() for session in self.sessions.values()
                if session.diagnosis_success and session.user_satisfaction and session.user_satisfaction >= 4.5
            ]
            data["successful_cases"] = successful_cases

        elif strategy == EvolutionStrategy.ERROR_PREVENTION:
            # 收集错误模式
            error_patterns = self.feedback_aggregator.get_error_patterns()
            data["error_patterns"] = error_patterns.get("diagnosis_confusion", {})

        return data

    # ========== 学习报告 ==========

    def generate_report(self, session_id: Optional[str] = None) -> LearningReport:
        """生成学习报告"""
        report = LearningReport()

        if session_id:
            report.feedback_summary = self.feedback_collector.get_accuracy_stats(session_id)
            report.thread_learning_summary = self._get_session_learning_summary(session_id)
        else:
            report.feedback_summary = self.feedback_collector.get_accuracy_stats()
            report.thread_learning_summary = self._get_all_sessions_learning_summary()

        report.reflection_summary = self.reflection.get_reflection_summary()
        report.evolution_summary = self.evolution.get_evolution_summary()

        # 识别准确率问题
        issues = self.feedback_aggregator.identify_accuracy_issues()
        if issues:
            report.recommendations.extend([
                f"检测到问题: {issue['description']}" for issue in issues
            ])

        # 整体评估
        overall_quality = self.evolution.current_metrics.calculate_overall_quality()
        if overall_quality >= 0.8:
            report.overall_assessment = "excellent"
        elif overall_quality >= 0.7:
            report.overall_assessment = "good"
        elif overall_quality >= 0.6:
            report.overall_assessment = "fair"
        else:
            report.overall_assessment = "needs_improvement"

        return report

    def _get_session_learning_summary(self, session_id: str) -> Dict:
        """获取会话学习摘要"""
        session = self.sessions.get(session_id)
        if not session:
            return {}

        return {
            "session_id": session_id,
            "total_interactions": session.total_interactions,
            "total_corrections": session.total_corrections,
            "reflections_count": session.reflections_count,
            "intent_accuracy": session.intent_accuracy,
            "diagnosis_accuracy": session.diagnosis_accuracy,
            "inquiry_efficiency": session.inquiry_efficiency
        }

    def _get_all_sessions_learning_summary(self) -> Dict:
        """获取所有会话学习摘要"""
        total_sessions = len(self.sessions)
        total_interactions = sum(s.total_interactions for s in self.sessions.values())
        total_corrections = sum(s.total_corrections for s in self.sessions.values())
        total_reflections = sum(s.reflections_count for s in self.sessions.values())

        return {
            "total_sessions": total_sessions,
            "total_interactions": total_interactions,
            "total_corrections": total_corrections,
            "total_reflections": total_reflections
        }

    def get_session(self, session_id: str) -> Optional[LearningSession]:
        """获取会话信息"""
        return self.sessions.get(session_id)

    def get_all_sessions(self) -> List[LearningSession]:
        """获取所有会话"""
        return list(self.sessions.values())
