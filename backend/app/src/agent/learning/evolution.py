"""
长期进化模块 (Long-term Evolution)

基于 Self-Evolving Agents (arXiv:2507.21046) 的长期进化机制
"""

import logging
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


logger = logging.getLogger(__name__)


class EvolutionStrategy(str, Enum):
    """进化策略"""
    PROMPT_OPTIMIZATION = "prompt_optimization"     # 提示词优化
    KNOWLEDGE_UPDATE = "knowledge_update"          # 知识更新
    BEHAVIOR_ADJUSTMENT = "behavior_adjustment"    # 行为调整
    MODEL_SWITCH = "model_switch"                  # 模型切换


@dataclass
class PerformanceMetrics:
    """性能指标"""
    success_rate: float = 0.0
    accuracy: float = 0.0
    user_satisfaction: float = 0.0
    avg_response_time: float = 0.0
    error_rate: float = 0.0

    # TCM 特定指标
    diagnosis_accuracy: float = 0.0
    prescription_acceptance: float = 0.0
    symptom_improvement_rate: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "success_rate": self.success_rate,
            "accuracy": self.accuracy,
            "user_satisfaction": self.user_satisfaction,
            "avg_response_time": self.avg_response_time,
            "error_rate": self.error_rate,
            "diagnosis_accuracy": self.diagnosis_accuracy,
            "prescription_acceptance": self.prescription_acceptance,
            "symptom_improvement_rate": self.symptom_improvement_rate
        }

    def calculate_overall(self) -> float:
        """计算综合评分"""
        weights = {
            "success_rate": 0.2,
            "accuracy": 0.15,
            "user_satisfaction": 0.25,
            "error_rate": -0.15,  # 负权重
            "diagnosis_accuracy": 0.15,
            "symptom_improvement_rate": 0.1
        }
        return (
            self.success_rate * weights["success_rate"] +
            self.accuracy * weights["accuracy"] +
            self.user_satisfaction * weights["user_satisfaction"] +
            self.error_rate * weights["error_rate"] +
            self.diagnosis_accuracy * weights["diagnosis_accuracy"] +
            self.symptom_improvement_rate * weights["symptom_improvement_rate"]
        )


@dataclass
class EvolutionRecord:
    """进化记录"""
    timestamp: datetime = field(default_factory=datetime.now)
    strategy: EvolutionStrategy = EvolutionStrategy.PROMPT_OPTIMIZATION

    # 变更前后的指标
    before_metrics: Optional[PerformanceMetrics] = None
    after_metrics: Optional[PerformanceMetrics] = None

    # 变更内容
    change_description: str = ""
    changed_parameters: Dict[str, Any] = field(default_factory=dict)

    # 效果评估
    improvement: float = 0.0  # 性能提升幅度
    successful: bool = False

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "strategy": self.strategy.value,
            "before_metrics": self.before_metrics.to_dict() if self.before_metrics else None,
            "after_metrics": self.after_metrics.to_dict() if self.after_metrics else None,
            "change_description": self.change_description,
            "changed_parameters": self.changed_parameters,
            "improvement": self.improvement,
            "successful": self.successful
        }


class EvolutionEngine:
    """进化引擎"""

    def __init__(self):
        self.records: List[EvolutionRecord] = []
        self.current_metrics = PerformanceMetrics()
        self.baseline_metrics = PerformanceMetrics()
        self.evolution_callbacks: Dict[EvolutionStrategy, List[Callable]] = {
            strategy: [] for strategy in EvolutionStrategy
        }

    def register_callback(
        self,
        strategy: EvolutionStrategy,
        callback: Callable[[Dict], Any]
    ) -> None:
        """注册进化回调"""
        self.evolution_callbacks[strategy].append(callback)

    def update_metrics(self, metrics: Partial[PerformanceMetrics]) -> None:
        """更新当前指标"""
        for key, value in metrics.items():
            if hasattr(self.current_metrics, key):
                setattr(self.current_metrics, key, value)

    def should_evolve(self, threshold: float = 0.7) -> bool:
        """
        判断是否应该进化

        Args:
            threshold: 性能阈值

        Returns:
            是否需要进化
        """
        current_score = self.current_metrics.calculate_overall()

        # 如果当前性能低于阈值，需要进化
        if current_score < threshold:
            return True

        # 检查是否有持续下降趋势
        recent_records = self.records[-5:] if len(self.records) >= 5 else self.records
        if len(recent_records) >= 3:
            improvements = [r.improvement for r in recent_records]
            if all(imp < 0 for imp in improvements[-3:]):
                return True

        return False

    async def evolve(
        self,
        strategy: EvolutionStrategy,
        context: Optional[Dict] = None
    ) -> EvolutionRecord:
        """
        执行进化

        Args:
            strategy: 进化策略
            context: 上下文信息

        Returns:
            进化记录
        """
        record = EvolutionRecord(
            strategy=strategy,
            before_metrics=self.current_metrics
        )

        # 执行进化回调
        callbacks = self.evolution_callbacks.get(strategy, [])
        changes = {}

        for callback in callbacks:
            try:
                result = await callback(context or {})
                if isinstance(result, dict):
                    changes.update(result)
            except Exception as e:
                logger.error(f"[Evolution] Callback error: {e}")

        # 更新记录
        record.changed_parameters = changes
        record.change_description = self._describe_change(strategy, changes)

        # 执行变更
        await self._apply_changes(strategy, changes, context)

        logger.info(f"[Evolution] Executed {strategy.value}: {record.change_description}")

        return record

    async def _apply_changes(
        self,
        strategy: EvolutionStrategy,
        changes: Dict,
        context: Optional[Dict]
    ) -> None:
        """应用变更"""
        if strategy == EvolutionStrategy.PROMPT_OPTIMIZATION:
            # 提示词优化逻辑
            pass
        elif strategy == EvolutionStrategy.KNOWLEDGE_UPDATE:
            # 知识更新逻辑
            pass
        elif strategy == EvolutionStrategy.BEHAVIOR_ADJUSTMENT:
            # 行为调整逻辑
            pass
        elif strategy == EvolutionStrategy.MODEL_SWITCH:
            # 模型切换逻辑
            pass

    def _describe_change(self, strategy: EvolutionStrategy, changes: Dict) -> str:
        """描述变更"""
        if strategy == EvolutionStrategy.PROMPT_OPTIMIZATION:
            return f"优化提示词: {list(changes.keys())}"
        elif strategy == EvolutionStrategy.KNOWLEDGE_UPDATE:
            return f"更新知识: {len(changes)}条"
        elif strategy == EvolutionStrategy.BEHAVIOR_ADJUSTMENT:
            return f"调整行为: {list(changes.keys())}"
        elif strategy == EvolutionStrategy.MODEL_SWITCH:
            return f"切换模型: {changes.get('model', 'unknown')}"
        return "未知变更"

    def record_evolution_result(self, record: EvolutionRecord) -> None:
        """记录进化结果"""
        # 计算改进幅度
        if record.before_metrics and record.after_metrics:
            before_score = record.before_metrics.calculate_overall()
            after_score = record.after_metrics.calculate_overall()
            record.improvement = after_score - before_score
            record.successful = record.improvement > 0

        self.records.append(record)

        # 如果改进成功，更新基线
        if record.successful and record.after_metrics:
            self.baseline_metrics = record.after_metrics

        logger.info(
            f"[Evolution] Record: {record.strategy.value}, "
            f"improvement: {record.improvement:+.2%}, "
            f"successful: {record.successful}"
        )

    def get_evolution_summary(self) -> Dict:
        """获取进化摘要"""
        if not self.records:
            return {
                "total_evolutions": 0,
                "by_strategy": {},
                "overall_improvement": 0.0
            }

        by_strategy: Dict = {}
        total_improvement = 0.0
        successful_count = 0

        for record in self.records:
            strategy = record.strategy.value
            by_strategy[strategy] = by_strategy.get(strategy, 0) + 1
            total_improvement += record.improvement
            if record.successful:
                successful_count += 1

        return {
            "total_evolutions": len(self.records),
            "by_strategy": by_strategy,
            "successful_count": successful_count,
            "success_rate": successful_count / len(self.records) if self.records else 0,
            "overall_improvement": total_improvement / len(self.records) if self.records else 0,
            "current_score": self.current_metrics.calculate_overall(),
            "baseline_score": self.baseline_metrics.calculate_overall()
        }

    def recommend_strategy(self) -> Optional[EvolutionStrategy]:
        """推荐进化策略"""
        current = self.current_metrics

        # 根据不同指标推荐策略
        if current.user_satisfaction < 0.6:
            return EvolutionStrategy.PROMPT_OPTIMIZATION
        elif current.accuracy < 0.7 or current.diagnosis_accuracy < 0.7:
            return EvolutionStrategy.KNOWLEDGE_UPDATE
        elif current.error_rate > 0.1:
            return EvolutionStrategy.BEHAVIOR_ADJUSTMENT
        elif current.success_rate < 0.8:
            return EvolutionStrategy.MODEL_SWITCH

        return None


# 用于类型提示的Partial类
class Partial(dict):
    """部分更新字典"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
