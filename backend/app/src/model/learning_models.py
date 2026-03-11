"""
学习系统数据模型

定义学习相关的数据库表结构
"""

from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4
from sqlmodel import SQLModel, Field, Column, Index
from sqlalchemy import JSON, Text


class ThreadLearningRecord(SQLModel, table=True):
    """单线程学习记录（按 conversation_id 存储）"""
    __tablename__ = "thread_learning_records"

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    conversation_id: UUID = Field(foreign_key="conversations.id", description="对话ID")
    user_id: UUID = Field(foreign_key="accounts.id", description="用户ID")

    # 主图学习（意图识别）
    intent_learning: Dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False, default={}),
        description="意图识别学习上下文"
    )

    # 子图学习（按子图名称索引）
    subgraph_learning: Dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False, default={}),
        description="子图学习上下文"
    )

    # 线程摘要
    thread_summary: Optional[str] = Field(
        sa_column=Column(Text, nullable=True),
        description="线程学习摘要"
    )

    # 元数据
    total_corrections: int = Field(default=0, description="纠正次数")
    complexity_score: Optional[float] = Field(default=None, description="复杂度评分")
    interaction_rounds: int = Field(default=0, description="交互轮数")

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    __table_args__ = (
        Index("idx_thread_learning_conversation_id", "conversation_id"),
        Index("idx_thread_learning_user_id", "user_id"),
        Index("idx_thread_learning_updated_at", "updated_at"),
    )


class LearningEventRecord(SQLModel, table=True):
    """学习事件记录（详细事件日志）"""
    __tablename__ = "learning_event_records"

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    conversation_id: UUID = Field(foreign_key="conversations.id")
    event_type: str = Field(max_length=50, description="事件类型")
    source: str = Field(max_length=50, description="来源（main_graph/diagnose_subgraph等）")
    trigger: str = Field(max_length=50, description="触发器")

    payload: Dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False),
        description="事件载荷"
    )

    created_at: datetime = Field(default_factory=datetime.now)

    __table_args__ = (
        Index("idx_learning_event_conversation_id", "conversation_id"),
        Index("idx_learning_event_type", "event_type"),
        Index("idx_learning_event_source", "source"),
        Index("idx_learning_event_created_at", "created_at"),
    )


class FeedbackRecord(SQLModel, table=True):
    """用户反馈记录"""
    __tablename__ = "feedback_records"

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    conversation_id: UUID = Field(foreign_key="conversations.id")
    user_id: UUID = Field(foreign_key="accounts.id")

    feedback_type: str = Field(max_length=50, description="反馈类型")
    conversation_stage: str = Field(max_length=50, description="对话阶段")
    subgraph: Optional[str] = Field(max_length=50, default=None, description="子图名称")

    # 反馈内容
    feedback_data: Dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False),
        description="反馈详细数据"
    )

    # 准确率评分
    intent_accuracy: Optional[float] = Field(default=None)
    symptom_understanding_accuracy: Optional[float] = Field(default=None)
    diagnosis_accuracy: Optional[float] = Field(default=None)
    inquiry_efficiency: Optional[float] = Field(default=None)
    overall_satisfaction: Optional[float] = Field(default=None)

    sentiment: str = Field(max_length=20, default="neutral", description="情感倾向")

    created_at: datetime = Field(default_factory=datetime.now)

    __table_args__ = (
        Index("idx_feedback_conversation_id", "conversation_id"),
        Index("idx_feedback_user_id", "user_id"),
        Index("idx_feedback_type", "feedback_type"),
        Index("idx_feedback_created_at", "created_at"),
    )


class ReflectionRecord(SQLModel, table=True):
    """反思记录"""
    __tablename__ = "reflection_records"

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    conversation_id: UUID = Field(foreign_key="conversations.id")

    reflection_type: str = Field(max_length=50, description="反思类型")

    # 反思结果
    reflection_data: Dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False),
        description="反思详细数据"
    )

    # 关键字段（便于查询）
    error_root_cause: Optional[str] = Field(sa_column=Column(Text, nullable=True))
    lessons_learned: Optional[str] = Field(sa_column=Column(Text, nullable=True))

    created_at: datetime = Field(default_factory=datetime.now)

    __table_args__ = (
        Index("idx_reflection_conversation_id", "conversation_id"),
        Index("idx_reflection_type", "reflection_type"),
        Index("idx_reflection_created_at", "created_at"),
    )


class ComplexCaseLibrary(SQLModel, table=True):
    """疑难杂症案例库"""
    __tablename__ = "complex_case_library"

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    conversation_id: UUID = Field(foreign_key="conversations.id")

    # 案例特征
    symptom_pattern: Dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False),
        description="症状组合模式"
    )
    misleading_symptoms: Dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False, default=[]),
        description="误导性症状"
    )
    key_differential_points: Dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False, default=[]),
        description="关键鉴别点"
    )

    # 诊断信息
    final_syndrome: str = Field(max_length=100, description="最终证型")
    complexity_score: float = Field(description="复杂度评分")

    # 成功路径
    successful_inquiry_path: Dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False, default=[]),
        description="成功的追问路径"
    )
    interaction_rounds: int = Field(description="交互轮数")

    # 效果评估
    diagnosis_success: bool = Field(default=False, description="诊断是否成功")
    user_satisfaction: Optional[float] = Field(default=None, description="用户满意度")

    # 向量嵌入ID（指向向量数据库）
    embedding_id: Optional[str] = Field(default=None, max_length=255)

    created_at: datetime = Field(default_factory=datetime.now)

    __table_args__ = (
        Index("idx_complex_case_syndrome", "final_syndrome"),
        Index("idx_complex_case_complexity", "complexity_score"),
        Index("idx_complex_case_success", "diagnosis_success"),
        Index("idx_complex_case_created_at", "created_at"),
    )


class EffectiveStrategyLibrary(SQLModel, table=True):
    """高效策略库"""
    __tablename__ = "effective_strategy_library"

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)

    # 策略类型
    strategy_type: str = Field(max_length=50, description="策略类型")
    strategy_name: str = Field(max_length=200, description="策略名称")
    strategy_description: Optional[str] = Field(sa_column=Column(Text, nullable=True))

    # 适用场景
    applicable_symptoms: Dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False, default=[]),
        description="适用症状"
    )
    applicable_syndromes: Dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False, default=[]),
        description="适用证型"
    )

    # 策略内容
    optimal_question_sequence: Dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False, default=[]),
        description="最优追问序列"
    )

    # 效果统计
    usage_count: int = Field(default=0, description="使用次数")
    avg_rounds_to_diagnosis: float = Field(default=0.0, description="平均诊断轮数")
    avg_user_satisfaction: float = Field(default=0.0, description="平均用户满意度")

    # 向量嵌入ID
    embedding_id: Optional[str] = Field(default=None, max_length=255)

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    __table_args__ = (
        Index("idx_effective_strategy_type", "strategy_type"),
        Index("idx_effective_strategy_usage", "usage_count"),
        Index("idx_effective_strategy_satisfaction", "avg_user_satisfaction"),
    )


class DiscriminatingRule(SQLModel, table=True):
    """辨证鉴别规则库"""
    __tablename__ = "discriminating_rules"

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)

    # 鉴别对象
    syndrome_a: str = Field(max_length=100, description="证型A")
    syndrome_b: str = Field(max_length=100, description="证型B")

    # 鉴别规则
    discriminating_rule: str = Field(sa_column=Column(Text, nullable=False), description="鉴别规则描述")
    discriminating_symptoms: Dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False, default=[]),
        description="鉴别症状列表"
    )

    # 来源统计
    error_frequency: int = Field(default=1, description="错误出现次数")
    source_conversation_ids: Dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False, default=[]),
        description="来源会话ID列表"
    )

    # 效果评估
    rule_effectiveness: float = Field(default=0.0, description="规则有效性 (0-1)")
    applied_count: int = Field(default=0, description="应用次数")
    success_count: int = Field(default=0, description="成功次数")

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    __table_args__ = (
        Index("idx_discriminating_rules_syndrome_a", "syndrome_a"),
        Index("idx_discriminating_rules_syndrome_b", "syndrome_b"),
        Index("idx_discriminating_rules_frequency", "error_frequency"),
        Index("idx_discriminating_rules_effectiveness", "rule_effectiveness"),
    )


class MisdiagnosisPattern(SQLModel, table=True):
    """常见误诊模式库"""
    __tablename__ = "misdiagnosis_patterns"

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)

    # 错误模式
    pattern_name: str = Field(max_length=200, description="模式名称")
    wrong_syndrome: str = Field(max_length=100, description="错误的证型")
    correct_syndrome: str = Field(max_length=100, description="正确的证型")

    # 错误原因
    common_causes: Dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False, default=[]),
        description="常见原因"
    )
    missed_symptoms: Dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False, default=[]),
        description="常被忽略的症状"
    )
    misinterpreted_symptoms: Dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False, default=[]),
        description="常被误解的症状"
    )

    # 预防策略
    prevention_checklist: Dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False, default=[]),
        description="预防检查清单"
    )
    prevention_rule: Optional[str] = Field(sa_column=Column(Text, nullable=True))

    # 统计信息
    occurrence_count: int = Field(default=1, description="出现次数")
    severity: str = Field(max_length=20, default="medium", description="严重程度")

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    __table_args__ = (
        Index("idx_misdiagnosis_pattern_name", "pattern_name"),
        Index("idx_misdiagnosis_wrong_syndrome", "wrong_syndrome"),
        Index("idx_misdiagnosis_correct_syndrome", "correct_syndrome"),
        Index("idx_misdiagnosis_occurrence", "occurrence_count"),
        Index("idx_misdiagnosis_severity", "severity"),
    )


class EvolutionRecord(SQLModel, table=True):
    """进化记录"""
    __tablename__ = "evolution_records"

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)

    strategy: str = Field(max_length=50, description="进化策略")

    # 变更前后的指标
    before_metrics: Dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=True),
        description="变更前指标"
    )
    after_metrics: Dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=True),
        description="变更后指标"
    )

    # 变更内容
    change_description: str = Field(sa_column=Column(Text, nullable=False))
    changed_rules: Dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False, default=[]),
        description="变更的规则列表"
    )

    # 效果评估
    improvement: float = Field(default=0.0, description="准确率提升幅度")
    successful: bool = Field(default=False, description="是否成功")

    created_at: datetime = Field(default_factory=datetime.now)

    __table_args__ = (
        Index("idx_evolution_strategy", "strategy"),
        Index("idx_evolution_successful", "successful"),
        Index("idx_evolution_created_at", "created_at"),
    )
