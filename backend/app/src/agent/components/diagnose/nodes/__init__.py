"""
诊断子图节点模块
"""

from .collect_info import collect_info
from .analyze_follow_up import analyze_and_follow_up
from .assess_complexity import assess_complexity
from .simple.simple_diagnosis import simple_diagnosis
from .moderate_diagnosis.moderate_diagnosis import moderate_diagnosis
# from .generate_result import generate_diagnosis_result

__all__ = [
    "collect_info",
    "analyze_and_follow_up",
    "assess_complexity",
    "simple_diagnosis",
    "moderate_diagnosis",
    # "generate_diagnosis_result",
]
