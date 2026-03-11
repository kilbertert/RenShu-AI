"""
学习中间件 (Learning Middleware)

职责：
1. after_model: 记录交互结果，触发反馈/反思/进化
2. 不负责将学习成果注入上下文（那是 FocusContextMiddleware 的职责）

优先级: P30 (在业务处理之后)

上下文工程 vs 记忆工程的分工:
- 本中间件 = 记忆工程: "记什么、怎么记"
- FocusContextMiddleware = 上下文工程: "给模型看什么、怎么拼"
"""

import logging
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass

from .base import BaseMiddleware, MiddlewareConfig

logger = logging.getLogger(__name__)


@dataclass
class LearningMiddlewareConfig(MiddlewareConfig):
    """学习中间件配置"""
    # 交互记录
    enable_interaction_recording: bool = True

    # 反思触发
    enable_error_reflection: bool = True
    enable_feedback_reflection: bool = True

    # 进化触发
    enable_evolution_check: bool = True


class LearningMiddleware(BaseMiddleware):
    """
    学习中间件

    在 after_model 阶段：
    1. 记录交互结果（成功/失败）
    2. 对错误进行自我反思
    3. 收集反馈信号
    4. 检查是否需要触发长期进化

    学习成果（反思经验、进化建议）的"注入"由 FocusContextMiddleware 负责
    """

    def __init__(
        self,
        learner=None,
        source: Optional[str] = None,
        config: Optional[LearningMiddlewareConfig] = None,
    ):
        config = config or LearningMiddlewareConfig(
            enabled=True,
            priority=30,
            name="LearningMiddleware",
        )
        super().__init__(config)
        self.learner = learner
        self.source = source  # "main_graph" | "diagnose_subgraph" | "wellness_subgraph"

    def before_model(
        self, state: Dict[str, Any], runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """
        模型调用前：加载学习经验

        加载内容：
        1. 单线程学习：当前 conversation_id 的纠错、反思（从 learner 内存）
        2. 跨线程学习：历史案例的鉴别规则、误诊模式、高效策略（从 DB）

        source 过滤：
        - main_graph: 仅加载意图学习，去除 tool_learning 和 correction_learning
        - diagnose_subgraph: 加载完整学习经验 + successful_cases
        - wellness_subgraph: 加载养生专属策略
        """
        if not self.config.enabled or not self.learner:
            return None

        conversation_id = self._get_state_value(state, "conversation_id", "default")
        updates = {}
        steps = []

        # 1. 加载单线程学习上下文（按 source 过滤）
        try:
            thread_learning = self.learner.get_thread_learning_snapshot(
                conversation_id, source=self.source
            )
            if thread_learning:
                updates["thread_learning_context"] = thread_learning
                steps.append(f"[学习] 加载单线程学习: {len(thread_learning.get('recent_corrections', []))}条纠错")
        except Exception as e:
            logger.warning(f"[LearningMiddleware] 加载单线程学习失败: {e}")

        # 2. 加载跨线程学习经验（按 source 过滤）
        try:
            cross_thread = self._load_cross_thread_learning()
            if cross_thread:
                updates["cross_thread_learning"] = cross_thread
                total_rules = (
                    len(cross_thread.get("discriminating_rules", [])) +
                    len(cross_thread.get("misdiagnosis_patterns", [])) +
                    len(cross_thread.get("effective_strategies", []))
                )
                steps.append(f"[学习] 加载跨线程学习: {total_rules}条规则")
        except Exception as e:
            logger.warning(f"[LearningMiddleware] 加载跨线程学习失败: {e}")

        if steps:
            updates["steps"] = steps

        return updates if updates else None

    def _load_cross_thread_learning(self) -> Optional[Dict[str, Any]]:
        """
        从数据库加载跨线程学习经验

        按 source 过滤：
        - main_graph: 加载鉴别规则和误诊模式，不加载具体诊断策略
        - diagnose_subgraph: 加载全部内容
        - wellness_subgraph: 仅加载有效策略，不加载鉴别规则和误诊模式
        """
        if not self.learner or not self.learner.storage:
            return None

        try:
            storage = self.learner.storage["cross_thread"]

            cross_thread = {
                "discriminating_rules": [],
                "misdiagnosis_patterns": [],
                "effective_strategies": []
            }

            # 养生子图不需要鉴别规则和误诊模式
            if self.source != "wellness_subgraph":
                # 加载鉴别规则（高频错误的前10条）
                rules = storage.get_top_discriminating_rules(limit=10, min_frequency=3)
                cross_thread["discriminating_rules"] = [
                    {
                        "syndrome_a": r.syndrome_a,
                        "syndrome_b": r.syndrome_b,
                        "rule": r.discriminating_rule,
                        "symptoms": r.discriminating_symptoms,
                        "frequency": r.error_frequency
                    }
                    for r in rules
                ]

                # 加载误诊模式（高风险的前10条）
                patterns = storage.get_all_high_risk_patterns(min_occurrence=5, limit=10)
                cross_thread["misdiagnosis_patterns"] = [
                    {
                        "pattern_name": p.pattern_name,
                        "wrong_syndrome": p.wrong_syndrome,
                        "correct_syndrome": p.correct_syndrome,
                        "common_causes": p.common_causes,
                        "missed_symptoms": p.missed_symptoms,
                        "prevention_rule": p.prevention_rule,
                        "occurrence_count": p.occurrence_count,
                        "severity": p.severity
                    }
                    for p in patterns
                ]

            # 加载高效策略（高满意度的前10条）
            strategies = storage.get_top_strategies(min_satisfaction=4.0, min_usage=3, limit=10)
            cross_thread["effective_strategies"] = [
                {
                    "strategy_name": s.strategy_name,
                    "strategy_type": s.strategy_type,
                    "description": s.strategy_description,
                    "applicable_symptoms": s.applicable_symptoms,
                    "applicable_syndromes": s.applicable_syndromes,
                    "optimal_questions": s.optimal_question_sequence,
                    "avg_rounds": s.avg_rounds_to_diagnosis,
                    "satisfaction": s.avg_user_satisfaction,
                    "usage_count": s.usage_count
                }
                for s in strategies
            ]

            # 主图不加载具体策略（仅加载鉴别规则和误诊模式用于路由辅助）
            if self.source == "main_graph":
                cross_thread["effective_strategies"] = []

            logger.debug(
                f"[LearningMiddleware] 跨线程学习加载完成 (source={self.source}): "
                f"{len(cross_thread['discriminating_rules'])}条鉴别规则, "
                f"{len(cross_thread['misdiagnosis_patterns'])}条误诊模式, "
                f"{len(cross_thread['effective_strategies'])}条高效策略"
            )

            return cross_thread

        except Exception as e:
            logger.error(f"[LearningMiddleware] 加载跨线程学习异常: {e}")
            return None

    def after_model(
        self, state: Dict[str, Any], runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """
        模型调用后：记录交互并触发学习

        1. 记录交互（成功/失败）
        2. 错误时触发反思
        3. 检查进化条件
        """
        if not self.config.enabled or not self.learner:
            return None

        import asyncio

        config: LearningMiddlewareConfig = self.config
        session_id = self._get_state_value(state, "conversation_id", "default")
        user_id = self._get_state_value(state, "user_id", "")
        error = self._get_state_value(state, "error")
        answer = self._get_state_value(state, "answer", "")

        updates = {}
        steps = []

        # 1. 记录交互
        if config.enable_interaction_recording:
            success = not bool(error)
            self.learner.record_interaction(
                session_id=session_id,
                success=success,
                context={
                    "user_id": user_id,
                    "has_answer": bool(answer),
                    "has_error": bool(error),
                },
            )
            steps.append(f"[学习] 记录交互: {'成功' if success else '失败'}")

        # 2. 错误反思
        if config.enable_error_reflection and error:
            try:
                query = self._extract_last_user_query(state)
                asyncio.run(
                    self.learner.reflect_on_error(
                        session_id=session_id,
                        goal=query or "处理用户查询",
                        result=answer or "无结果",
                        error=str(error),
                    )
                )
                steps.append("[学习] 错误反思完成")
            except Exception as e:
                logger.warning(f"[LearningMiddleware] 错误反思失败: {e}")

        # 3. 成功完成时也反思（积累经验）
        if not error and answer and config.enable_feedback_reflection:
            try:
                query = self._extract_last_user_query(state)
                router = self._get_state_value(state, "router")
                query_type = getattr(router, "query_type", "general") if router else "general"

                # 只对诊断类交互进行完成反思（避免过度反思）
                if query_type in ("tcm-diagnose", "tcm-wellness"):
                    asyncio.run(
                        self.learner.reflect_on_completion(
                            session_id=session_id,
                            goal=query or "处理用户查询",
                            result=answer[:500],
                        )
                    )
                    steps.append("[学习] 完成反思")
            except Exception as e:
                logger.warning(f"[LearningMiddleware] 完成反思失败: {e}")

        # 4. 进化检查
        if config.enable_evolution_check:
            try:
                if self.learner.evolution.should_evolve():
                    strategy = self.learner.evolution.recommend_strategy()
                    if strategy:
                        steps.append(f"[学习] 建议进化策略: {strategy.value}")
                        updates["evolution_recommended"] = strategy.value
            except Exception as e:
                logger.warning(f"[LearningMiddleware] 进化检查失败: {e}")

        updates["learning_recorded"] = True
        if steps:
            updates["steps"] = steps

        return updates if updates else None

    def _extract_last_user_query(self, state: Dict[str, Any]) -> Optional[str]:
        """从消息中提取最后一条用户消息"""
        messages = self._get_state_value(state, "messages", [])
        for msg in reversed(messages):
            role = None
            if hasattr(msg, "type"):
                role = msg.type
            elif isinstance(msg, dict):
                role = msg.get("role")
            if role in ("human", "user"):
                if hasattr(msg, "content"):
                    return msg.content
                elif isinstance(msg, dict):
                    return msg.get("content", "")
        return None

    def wrap_tool_call(
        self, tool_call: Callable, tool_name: str, state: Dict[str, Any]
    ) -> Callable:
        """包装工具调用（不做额外处理）"""
        return tool_call


def create_learning_middleware(
    learner=None,
    source: Optional[str] = None,
) -> LearningMiddleware:
    """创建学习中间件的便捷函数

    Args:
        learner: SelfLearner 实例
        source: 来源标识 ("main_graph" | "diagnose_subgraph" | "wellness_subgraph")
    """
    return LearningMiddleware(learner=learner, source=source)
