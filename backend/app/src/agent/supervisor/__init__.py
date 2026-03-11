"""
TCM Supervisor 多Agent协调模块

提供Supervisor模式的实现，包括：
- TCMSupervisor: 主协调器
- AgentRole: Agent角色定义
- ResponseType: 响应类型
- AgentTask/AgentResponse: 任务和响应数据结构
- Specialists: 各个专门Agent的实现
"""

from .tcm_supervisor import (
    TCMSupervisor,
    SupervisorConfig,
    AgentRole,
    ResponseType,
    AgentTask,
    AgentResponse,
    create_supervisor,
    direct_pass_response,
    handoff_to,
)
from .specialists import (
    BaseSpecialist,
    DiagnosisSpecialist,
    PrescriptionSpecialist,
    WellnessSpecialist,
    HerbSpecialist,
    GeneralSpecialist,
    get_all_specialists,
)

__all__ = [
    # Supervisor
    "TCMSupervisor",
    "SupervisorConfig",
    "AgentRole",
    "ResponseType",
    "AgentTask",
    "AgentResponse",
    "create_supervisor",
    "direct_pass_response",
    "handoff_to",
    # Specialists
    "BaseSpecialist",
    "DiagnosisSpecialist",
    "PrescriptionSpecialist",
    "WellnessSpecialist",
    "HerbSpecialist",
    "GeneralSpecialist",
    "get_all_specialists",
]
