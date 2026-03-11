"""
灰度部署模块 (Canary Deployment)

逐步上线新功能
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


logger = logging.getLogger(__name__)


class DeploymentStage(str, Enum):
    """部署阶段"""
    PLANNING = "planning"           # 规划中
    INTERNAL_TEST = "internal"     # 内部测试
    CANARY_1 = "canary_1%"        # 1% 灰度
    CANARY_5 = "canary_5%"        # 5% 灰度
    CANARY_10 = "canary_10%"      # 10% 灰度
    CANARY_25 = "canary_25%"      # 25% 灰度
    CANARY_50 = "canary_50%"      # 50% 灰度
    FULL_ROLLOUT = "full"         # 全量
    ROLLED_BACK = "rolled_back"    # 回滚


@dataclass
class DeploymentMetrics:
    """部署指标"""
    stage: DeploymentStage
    user_percentage: float        # 用户百分比
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    # 性能指标
    error_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    user_satisfaction: float = 0.0

    # 业务指标
    success_rate: float = 0.0
    feedback_score: float = 0.0

    def is_healthy(self) -> bool:
        """检查部署是否健康"""
        return (
            self.error_rate < 0.01 and  # 错误率低于1%
            self.p95_latency_ms < 5000 and  # P95延迟低于5秒
            self.user_satisfaction > 0.7  # 用户满意度高于70%
        )


@dataclass
class DeploymentConfig:
    """部署配置"""
    # 灰度发布配置
    auto_promote: bool = True          # 自动晋级
    promote_interval_hours: int = 24    # 晋级检查间隔
    rollback_threshold: float = 0.05    # 回滚阈值 (5%错误率)

    # 监控配置
    enable_monitoring: bool = True
    alert_channels: List[str] = field(default_factory=list)

    # 用户分流
    user_segments: List[str] = field(default_factory=lambda: ["internal", "beta_testers"])


class CanaryDeployment:
    """
    灰度部署管理器

    实现渐进式功能发布
    """

    def __init__(self, config: Optional[DeploymentConfig] = None):
        self.config = config or DeploymentConfig()
        self.current_stage = DeploymentStage.PLANNING
        self.metrics_history: List[DeploymentMetrics] = []

    def start_deployment(self) -> DeploymentMetrics:
        """开始部署"""
        logger.info("[Deployment] Starting canary deployment...")
        self.current_stage = DeploymentStage.INTERNAL_TEST

        metrics = DeploymentMetrics(
            stage=DeploymentStage.INTERNAL_TEST,
            user_percentage=0.0
        )
        self.metrics_history.append(metrics)

        return metrics

    def promote(self) -> Optional[DeploymentMetrics]:
        """
        晋升到下一阶段

        Returns:
            新阶段指标，如果不能晋级则返回 None
        """
        current_metrics = self._get_current_metrics()

        # 检查当前阶段是否健康
        if not current_metrics.is_healthy():
            logger.warning(
                f"[Deployment] Current stage not healthy: "
                f"error_rate={current_metrics.error_rate:.2%}"
            )
            return None

        # 晋级到下一阶段
        stage_sequence = [
            DeploymentStage.INTERNAL_TEST,
            DeploymentStage.CANARY_1,
            DeploymentStage.CANARY_5,
            DeploymentStage.CANARY_10,
            DeploymentStage.CANARY_25,
            DeploymentStage.CANARY_50,
            DeploymentStage.FULL_ROLLOUT
        ]

        try:
            current_index = stage_sequence.index(self.current_stage)
            next_stage = stage_sequence[current_index + 1]

            self.current_stage = next_stage
            new_metrics = DeploymentMetrics(
                stage=next_stage,
                user_percentage=self._get_percentage_for_stage(next_stage)
            )
            self.metrics_history.append(new_metrics)

            logger.info(
                f"[Deployment] Promoted to {next_stage.value} "
                f"({new_metrics.user_percentage:.0%} users)"
            )

            return new_metrics

        except (ValueError, IndexError):
            logger.info("[Deployment] Already at final stage")
            return None

    def rollback(self, reason: str = "") -> None:
        """回滚部署"""
        logger.warning(f"[Deployment] Rolling back: {reason}")
        self.current_stage = DeploymentStage.ROLLED_BACK

    def update_metrics(self, metrics: Partial[DeploymentMetrics]) -> None:
        """更新当前阶段指标"""
        current = self._get_current_metrics()

        for key, value in metrics.items():
            if hasattr(current, key):
                setattr(current, key, value)

    def should_promote(self) -> bool:
        """判断是否应该晋级"""
        current = self._get_current_metrics()

        # 检查健康状态
        if not current.is_healthy():
            return False

        # 检查是否应该回滚
        if current.error_rate > self.config.rollback_threshold:
            logger.warning(
                f"[Deployment] Error rate {current.error_rate:.2%} "
                f"exceeds threshold {self.config.rollback_threshold:.2%}"
            )
            return False

        return True

    def get_deployment_summary(self) -> Dict:
        """获取部署摘要"""
        current = self._get_current_metrics()

        return {
            "current_stage": self.current_stage.value,
            "user_percentage": current.user_percentage,
            "start_time": current.start_time.isoformat(),
            "is_healthy": current.is_healthy(),
            "metrics": {
                "error_rate": current.error_rate,
                "avg_latency_ms": current.avg_latency_ms,
                "p95_latency_ms": current.p95_latency_ms,
                "user_satisfaction": current.user_satisfaction
            },
            "stages_completed": len(self.metrics_history),
            "can_promote": self.should_promote()
        }

    def _get_current_metrics(self) -> DeploymentMetrics:
        """获取当前阶段指标"""
        return self.metrics_history[-1] if self.metrics_history else DeploymentMetrics(
            stage=self.current_stage,
            user_percentage=0.0
        )

    def _get_percentage_for_stage(self, stage: DeploymentStage) -> float:
        """获取阶段对应的用户百分比"""
        mapping = {
            DeploymentStage.INTERNAL_TEST: 0.0,
            DeploymentStage.CANARY_1: 0.01,
            DeploymentStage.CANARY_5: 0.05,
            DeploymentStage.CANARY_10: 0.10,
            DeploymentStage.CANARY_25: 0.25,
            DeploymentStage.CANARY_50: 0.50,
            DeploymentStage.FULL_ROLLOUT: 1.00,
            DeploymentStage.ROLLED_BACK: 0.0
        }
        return mapping.get(stage, 0.0)


# 用于类型提示
class Partial(dict):
    """部分更新字典"""
    pass
