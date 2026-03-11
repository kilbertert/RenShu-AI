"""
学习系统存储层

负责学习数据的持久化和检索
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from sqlmodel import Session, select

from ..model.learning_models import (
    ThreadLearningRecord,
    LearningEventRecord,
    FeedbackRecord,
    ReflectionRecord,
    ComplexCaseLibrary,
    EffectiveStrategyLibrary,
    DiscriminatingRule,
    MisdiagnosisPattern,
    EvolutionRecord,
)
from .feedback import TCMUserFeedback
from .reflection import TCMReflectionResult
from .evolution import EvolutionRecord as EvolutionRecordData


logger = logging.getLogger(__name__)


class ThreadLearningStorage:
    """单线程学习存储层"""

    def __init__(self, db_session: Session):
        self.db = db_session

    def save_thread_learning(
        self,
        conversation_id: UUID,
        user_id: UUID,
        learning_context: Dict[str, Any]
    ) -> ThreadLearningRecord:
        """保存单线程学习上下文"""

        # 查找或创建
        record = self.db.exec(
            select(ThreadLearningRecord)
            .where(ThreadLearningRecord.conversation_id == conversation_id)
        ).first()

        if not record:
            record = ThreadLearningRecord(
                conversation_id=conversation_id,
                user_id=user_id
            )

        # 更新学习内容
        record.intent_learning = learning_context.get("intent_learning", {})
        record.subgraph_learning = learning_context.get("subgraph_learning", {})
        record.thread_summary = learning_context.get("thread_summary", "")
        record.total_corrections = learning_context.get("total_corrections", 0)
        record.complexity_score = learning_context.get("complexity_score")
        record.interaction_rounds = learning_context.get("interaction_rounds", 0)
        record.updated_at = datetime.now()

        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)

        logger.info(f"[Storage] Saved thread learning for conversation {conversation_id}")

        return record

    def load_thread_learning(
        self,
        conversation_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """加载单线程学习上下文"""

        record = self.db.exec(
            select(ThreadLearningRecord)
            .where(ThreadLearningRecord.conversation_id == conversation_id)
        ).first()

        if not record:
            return None

        return {
            "conversation_id": str(record.conversation_id),
            "intent_learning": record.intent_learning,
            "subgraph_learning": record.subgraph_learning,
            "thread_summary": record.thread_summary,
            "total_corrections": record.total_corrections,
            "complexity_score": record.complexity_score,
            "interaction_rounds": record.interaction_rounds,
            "last_updated_at": record.updated_at.isoformat()
        }

    def save_learning_event(
        self,
        conversation_id: UUID,
        event_type: str,
        source: str,
        trigger: str,
        payload: Dict[str, Any]
    ) -> LearningEventRecord:
        """保存学习事件"""

        record = LearningEventRecord(
            conversation_id=conversation_id,
            event_type=event_type,
            source=source,
            trigger=trigger,
            payload=payload
        )

        self.db.add(record)
        self.db.commit()

        return record

    def get_recent_events(
        self,
        conversation_id: UUID,
        event_type: Optional[str] = None,
        limit: int = 10
    ) -> List[LearningEventRecord]:
        """获取最近的学习事件"""

        query = select(LearningEventRecord).where(
            LearningEventRecord.conversation_id == conversation_id
        )

        if event_type:
            query = query.where(LearningEventRecord.event_type == event_type)

        query = query.order_by(LearningEventRecord.created_at.desc()).limit(limit)

        return list(self.db.exec(query).all())


class FeedbackStorage:
    """反馈存储层"""

    def __init__(self, db_session: Session):
        self.db = db_session

    def save_feedback(
        self,
        feedback: TCMUserFeedback
    ) -> FeedbackRecord:
        """保存反馈"""

        record = FeedbackRecord(
            conversation_id=UUID(feedback.session_id),
            user_id=UUID(feedback.user_id),
            feedback_type=feedback.feedback_type.value,
            conversation_stage=feedback.conversation_stage,
            subgraph=feedback.subgraph,
            feedback_data=feedback.to_dict(),
            intent_accuracy=feedback.intent_accuracy,
            symptom_understanding_accuracy=feedback.symptom_understanding_accuracy,
            diagnosis_accuracy=feedback.diagnosis_accuracy,
            inquiry_efficiency=feedback.inquiry_efficiency,
            overall_satisfaction=feedback.overall_satisfaction,
            sentiment=feedback.sentiment
        )

        self.db.add(record)
        self.db.commit()

        logger.info(f"[Storage] Saved feedback: {feedback.feedback_type.value}")

        return record

    def get_feedbacks_by_session(
        self,
        conversation_id: UUID,
        feedback_type: Optional[str] = None
    ) -> List[FeedbackRecord]:
        """获取会话的所有反馈"""

        query = select(FeedbackRecord).where(
            FeedbackRecord.conversation_id == conversation_id
        )

        if feedback_type:
            query = query.where(FeedbackRecord.feedback_type == feedback_type)

        query = query.order_by(FeedbackRecord.created_at.desc())

        return list(self.db.exec(query).all())

    def get_accuracy_stats(
        self,
        conversation_id: Optional[UUID] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """获取准确率统计"""

        from datetime import timedelta

        query = select(FeedbackRecord)

        if conversation_id:
            query = query.where(FeedbackRecord.conversation_id == conversation_id)

        # 最近N天
        cutoff = datetime.now() - timedelta(days=days)
        query = query.where(FeedbackRecord.created_at >= cutoff)

        feedbacks = list(self.db.exec(query).all())

        if not feedbacks:
            return {}

        # 计算平均准确率
        intent_accuracies = [f.intent_accuracy for f in feedbacks if f.intent_accuracy is not None]
        diagnosis_accuracies = [f.diagnosis_accuracy for f in feedbacks if f.diagnosis_accuracy is not None]
        inquiry_efficiencies = [f.inquiry_efficiency for f in feedbacks if f.inquiry_efficiency is not None]

        return {
            "total_feedbacks": len(feedbacks),
            "intent_accuracy": sum(intent_accuracies) / len(intent_accuracies) if intent_accuracies else None,
            "diagnosis_accuracy": sum(diagnosis_accuracies) / len(diagnosis_accuracies) if diagnosis_accuracies else None,
            "inquiry_efficiency": sum(inquiry_efficiencies) / len(inquiry_efficiencies) if inquiry_efficiencies else None,
        }


class ReflectionStorage:
    """反思存储层"""

    def __init__(self, db_session: Session):
        self.db = db_session

    def save_reflection(
        self,
        reflection: TCMReflectionResult
    ) -> ReflectionRecord:
        """保存反思"""

        record = ReflectionRecord(
            conversation_id=UUID(reflection.session_id),
            reflection_type=reflection.reflection_type.value,
            reflection_data=reflection.to_dict(),
            error_root_cause=reflection.error_root_cause,
            lessons_learned=reflection.lessons_learned
        )

        self.db.add(record)
        self.db.commit()

        logger.info(f"[Storage] Saved reflection: {reflection.reflection_type.value}")

        return record

    def get_reflections_by_session(
        self,
        conversation_id: UUID,
        reflection_type: Optional[str] = None
    ) -> List[ReflectionRecord]:
        """获取会话的所有反思"""

        query = select(ReflectionRecord).where(
            ReflectionRecord.conversation_id == conversation_id
        )

        if reflection_type:
            query = query.where(ReflectionRecord.reflection_type == reflection_type)

        query = query.order_by(ReflectionRecord.created_at.desc())

        return list(self.db.exec(query).all())


class CrossThreadKnowledgeStorage:
    """跨线程知识存储层"""

    def __init__(self, db_session: Session, vector_store=None):
        self.db = db_session
        self.vector_store = vector_store

    # ========== 疑难案例库 ==========

    def save_complex_case(
        self,
        conversation_id: UUID,
        case_data: Dict[str, Any],
        embedding: Optional[Any] = None
    ) -> ComplexCaseLibrary:
        """保存疑难杂症案例"""

        case = ComplexCaseLibrary(
            conversation_id=conversation_id,
            symptom_pattern=case_data["symptom_pattern"],
            misleading_symptoms=case_data.get("misleading_symptoms", []),
            key_differential_points=case_data.get("key_differential_points", []),
            final_syndrome=case_data["final_syndrome"],
            complexity_score=case_data["complexity_score"],
            successful_inquiry_path=case_data.get("successful_inquiry_path", []),
            interaction_rounds=case_data["interaction_rounds"],
            diagnosis_success=case_data.get("diagnosis_success", False),
            user_satisfaction=case_data.get("user_satisfaction")
        )

        self.db.add(case)
        self.db.commit()
        self.db.refresh(case)

        # 保存向量嵌入
        if embedding is not None and self.vector_store:
            embedding_id = self.vector_store.add(
                vector=embedding,
                metadata={
                    "case_id": str(case.id),
                    "syndrome": case.final_syndrome,
                    "complexity": case.complexity_score
                }
            )
            case.embedding_id = embedding_id
            self.db.commit()

        logger.info(f"[Storage] Saved complex case: {case.final_syndrome}")

        return case

    def retrieve_similar_complex_cases(
        self,
        query_embedding: Optional[Any] = None,
        top_k: int = 5,
        min_complexity: float = 7.0
    ) -> List[ComplexCaseLibrary]:
        """检索相似的疑难案例"""

        # 向量检索
        if query_embedding is not None and self.vector_store:
            similar_ids = self.vector_store.search(
                query_vector=query_embedding,
                top_k=top_k,
                filter={"complexity": {"$gte": min_complexity}}
            )

            cases = self.db.exec(
                select(ComplexCaseLibrary)
                .where(ComplexCaseLibrary.id.in_([UUID(id) for id in similar_ids]))
            ).all()

            return list(cases)

        # 结构化检索
        cases = self.db.exec(
            select(ComplexCaseLibrary)
            .where(ComplexCaseLibrary.complexity_score >= min_complexity)
            .where(ComplexCaseLibrary.diagnosis_success == True)
            .order_by(ComplexCaseLibrary.user_satisfaction.desc())
            .limit(top_k)
        ).all()

        return list(cases)

    # ========== 高效策略库 ==========

    def save_effective_strategy(
        self,
        strategy_data: Dict[str, Any],
        embedding: Optional[Any] = None
    ) -> EffectiveStrategyLibrary:
        """保存高效策略"""

        # 检查是否已存在相似策略
        existing = self.db.exec(
            select(EffectiveStrategyLibrary)
            .where(EffectiveStrategyLibrary.strategy_name == strategy_data["strategy_name"])
        ).first()

        if existing:
            # 更新统计信息
            existing.usage_count += 1
            existing.avg_rounds_to_diagnosis = (
                (existing.avg_rounds_to_diagnosis * (existing.usage_count - 1) +
                 strategy_data["avg_rounds_to_diagnosis"]) / existing.usage_count
            )
            existing.updated_at = datetime.now()
            self.db.commit()
            return existing

        # 创建新策略
        strategy = EffectiveStrategyLibrary(
            strategy_type=strategy_data["strategy_type"],
            strategy_name=strategy_data["strategy_name"],
            strategy_description=strategy_data.get("strategy_description"),
            applicable_symptoms=strategy_data.get("applicable_symptoms", []),
            applicable_syndromes=strategy_data.get("applicable_syndromes", []),
            optimal_question_sequence=strategy_data.get("optimal_question_sequence", []),
            usage_count=1,
            avg_rounds_to_diagnosis=strategy_data["avg_rounds_to_diagnosis"],
            avg_user_satisfaction=strategy_data.get("avg_user_satisfaction", 0.0)
        )

        self.db.add(strategy)
        self.db.commit()
        self.db.refresh(strategy)

        # 保存向量
        if embedding is not None and self.vector_store:
            embedding_id = self.vector_store.add(
                vector=embedding,
                metadata={
                    "strategy_id": str(strategy.id),
                    "strategy_type": strategy.strategy_type
                }
            )
            strategy.embedding_id = embedding_id
            self.db.commit()

        logger.info(f"[Storage] Saved effective strategy: {strategy.strategy_name}")

        return strategy

    def get_top_strategies(
        self,
        min_satisfaction: float = 4.0,
        min_usage: int = 3,
        limit: int = 10
    ) -> List[EffectiveStrategyLibrary]:
        """获取高效策略（按满意度和使用次数排序）"""

        strategies = self.db.exec(
            select(EffectiveStrategyLibrary)
            .where(EffectiveStrategyLibrary.avg_user_satisfaction >= min_satisfaction)
            .where(EffectiveStrategyLibrary.usage_count >= min_usage)
            .order_by(EffectiveStrategyLibrary.avg_user_satisfaction.desc())
            .order_by(EffectiveStrategyLibrary.usage_count.desc())
            .limit(limit)
        ).all()

        return list(strategies)

    def get_strategies_by_type(
        self,
        strategy_type: str,
        limit: int = 5
    ) -> List[EffectiveStrategyLibrary]:
        """按策略类型获取高效策略"""

        strategies = self.db.exec(
            select(EffectiveStrategyLibrary)
            .where(EffectiveStrategyLibrary.strategy_type == strategy_type)
            .order_by(EffectiveStrategyLibrary.avg_user_satisfaction.desc())
            .limit(limit)
        ).all()

        return list(strategies)

    # ========== 辨证鉴别规则库 ==========

    def save_discriminating_rule(
        self,
        syndrome_a: str,
        syndrome_b: str,
        rule: str,
        discriminating_symptoms: List[str],
        source_conversation_ids: List[UUID]
    ) -> DiscriminatingRule:
        """保存辨证鉴别规则"""

        # 标准化证型对（按字母顺序）
        syndromes = sorted([syndrome_a, syndrome_b])

        # 检查是否已存在
        existing = self.db.exec(
            select(DiscriminatingRule)
            .where(DiscriminatingRule.syndrome_a == syndromes[0])
            .where(DiscriminatingRule.syndrome_b == syndromes[1])
        ).first()

        if existing:
            # 更新规则
            existing.discriminating_rule = rule
            existing.discriminating_symptoms = discriminating_symptoms
            existing.error_frequency += 1
            existing.source_conversation_ids = [str(id) for id in source_conversation_ids]
            existing.updated_at = datetime.now()
            self.db.commit()
            return existing

        # 创建新规则
        rule_record = DiscriminatingRule(
            syndrome_a=syndromes[0],
            syndrome_b=syndromes[1],
            discriminating_rule=rule,
            discriminating_symptoms=discriminating_symptoms,
            error_frequency=1,
            source_conversation_ids=[str(id) for id in source_conversation_ids]
        )

        self.db.add(rule_record)
        self.db.commit()
        self.db.refresh(rule_record)

        logger.info(f"[Storage] Saved discriminating rule: {syndromes[0]} vs {syndromes[1]}")

        return rule_record

    def get_discriminating_rule(
        self,
        syndrome_a: str,
        syndrome_b: str
    ) -> Optional[DiscriminatingRule]:
        """获取两个证型的鉴别规则"""

        syndromes = sorted([syndrome_a, syndrome_b])

        rule = self.db.exec(
            select(DiscriminatingRule)
            .where(DiscriminatingRule.syndrome_a == syndromes[0])
            .where(DiscriminatingRule.syndrome_b == syndromes[1])
        ).first()

        return rule

    def get_top_discriminating_rules(
        self,
        limit: int = 10,
        min_frequency: int = 3
    ) -> List[DiscriminatingRule]:
        """获取高频鉴别规则（按错误频率排序）"""

        rules = self.db.exec(
            select(DiscriminatingRule)
            .where(DiscriminatingRule.error_frequency >= min_frequency)
            .order_by(DiscriminatingRule.error_frequency.desc())
            .limit(limit)
        ).all()

        return list(rules)

    # ========== 误诊模式库 ==========

    def save_misdiagnosis_pattern(
        self,
        pattern_name: str,
        wrong_syndrome: str,
        correct_syndrome: str,
        pattern_data: Dict[str, Any]
    ) -> MisdiagnosisPattern:
        """保存误诊模式"""

        # 检查是否已存在
        existing = self.db.exec(
            select(MisdiagnosisPattern)
            .where(MisdiagnosisPattern.pattern_name == pattern_name)
        ).first()

        if existing:
            # 更新统计
            existing.occurrence_count += 1
            existing.updated_at = datetime.now()
            self.db.commit()
            return existing

        # 创建新模式
        pattern = MisdiagnosisPattern(
            pattern_name=pattern_name,
            wrong_syndrome=wrong_syndrome,
            correct_syndrome=correct_syndrome,
            common_causes=pattern_data.get("common_causes", []),
            missed_symptoms=pattern_data.get("missed_symptoms", []),
            misinterpreted_symptoms=pattern_data.get("misinterpreted_symptoms", []),
            prevention_checklist=pattern_data.get("prevention_checklist", []),
            prevention_rule=pattern_data.get("prevention_rule"),
            occurrence_count=1,
            severity=pattern_data.get("severity", "medium")
        )

        self.db.add(pattern)
        self.db.commit()
        self.db.refresh(pattern)

        logger.info(f"[Storage] Saved misdiagnosis pattern: {pattern_name}")

        return pattern

    def get_high_risk_patterns(
        self,
        suspected_syndrome: str = None,
        min_occurrence: int = 5,
        limit: int = 10
    ) -> List[MisdiagnosisPattern]:
        """获取高风险误诊模式"""

        query = select(MisdiagnosisPattern).where(
            MisdiagnosisPattern.occurrence_count >= min_occurrence
        ).where(
            MisdiagnosisPattern.severity == "high"
        )

        if suspected_syndrome:
            query = query.where(MisdiagnosisPattern.wrong_syndrome == suspected_syndrome)

        query = query.order_by(MisdiagnosisPattern.occurrence_count.desc()).limit(limit)

        patterns = self.db.exec(query).all()

        return list(patterns)

    def get_all_high_risk_patterns(
        self,
        min_occurrence: int = 5,
        limit: int = 10
    ) -> List[MisdiagnosisPattern]:
        """获取所有高风险误诊模式（不限定证型）"""

        patterns = self.db.exec(
            select(MisdiagnosisPattern)
            .where(MisdiagnosisPattern.occurrence_count >= min_occurrence)
            .where(MisdiagnosisPattern.severity.in_(["high", "medium"]))
            .order_by(MisdiagnosisPattern.occurrence_count.desc())
            .limit(limit)
        ).all()

        return list(patterns)


class EvolutionStorage:
    """进化记录存储层"""

    def __init__(self, db_session: Session):
        self.db = db_session

    def save_evolution_record(
        self,
        evolution_record: EvolutionRecordData
    ) -> EvolutionRecord:
        """保存进化记录"""

        record = EvolutionRecord(
            strategy=evolution_record.strategy.value,
            before_metrics=evolution_record.before_metrics.to_dict() if evolution_record.before_metrics else None,
            after_metrics=evolution_record.after_metrics.to_dict() if evolution_record.after_metrics else None,
            change_description=evolution_record.change_description,
            changed_rules=evolution_record.changed_rules,
            improvement=evolution_record.improvement,
            successful=evolution_record.successful
        )

        self.db.add(record)
        self.db.commit()

        logger.info(f"[Storage] Saved evolution record: {evolution_record.strategy.value}")

        return record

    def get_evolution_history(
        self,
        strategy: Optional[str] = None,
        limit: int = 10
    ) -> List[EvolutionRecord]:
        """获取进化历史"""

        query = select(EvolutionRecord)

        if strategy:
            query = query.where(EvolutionRecord.strategy == strategy)

        query = query.order_by(EvolutionRecord.created_at.desc()).limit(limit)

        return list(self.db.exec(query).all())
