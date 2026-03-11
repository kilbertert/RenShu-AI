"""
中医问诊反馈收集模块 (TCM Feedback Collection)

职责：收集与诊断准确率直接相关的用户反馈
- 主图：意图识别准确率反馈
- 子图：工具选择、症状理解、辨证推理、追问策略准确率反馈
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


logger = logging.getLogger(__name__)


class TCMFeedbackType(str, Enum):
    """中医问诊反馈类型"""

    # === 主图反馈（意图识别） ===
    INTENT_CORRECTION = "intent_correction"           # 意图纠正

    # === 子图反馈（诊断流程） ===
    TOOL_SELECTION_ERROR = "tool_selection_error"     # 工具选择错误
    SYMPTOM_MISUNDERSTANDING = "symptom_misunderstanding"  # 症状理解错误
    DIAGNOSIS_ERROR = "diagnosis_error"               # 辨证错误
    INQUIRY_INEFFICIENCY = "inquiry_inefficiency"     # 追问低效

    # === 结果反馈 ===
    DIAGNOSIS_ACCURACY = "diagnosis_accuracy"         # 诊断准确性评分
    PRESCRIPTION_CONCERN = "prescription_concern"     # 处方疑虑

    # === 长期反馈（治疗后） ===
    SYMPTOM_IMPROVEMENT = "symptom_improvement"       # 症状改善
    TREATMENT_EFFECTIVENESS = "treatment_effectiveness"  # 治疗有效性


@dataclass
class TCMUserFeedback:
    """中医问诊用户反馈"""
    feedback_type: TCMFeedbackType
    session_id: str
    user_id: str
    conversation_stage: str  # "intent_recognition", "info_collection", "diagnosis", "prescription"
    subgraph: Optional[str] = None

    # === 主图反馈：意图识别 ===
    wrong_intent: Optional[str] = None
    correct_intent: Optional[str] = None
    intent_confusion_reason: Optional[str] = None

    # === 子图反馈1：工具选择 ===
    wrong_tool: Optional[str] = None
    correct_tool: Optional[str] = None
    tool_error_reason: Optional[str] = None
    missing_info_for_tool: List[str] = field(default_factory=list)

    # === 子图反馈2：症状理解 ===
    misunderstood_symptom: Optional[str] = None
    correct_symptom_interpretation: Optional[str] = None
    symptom_disambiguation_needed: Optional[str] = None

    # === 子图反馈3：辨证诊断 ===
    agent_syndrome: Optional[str] = None
    correct_syndrome: Optional[str] = None
    diagnosis_error_type: Optional[str] = None  # "证型混淆", "信息不足", "辨证思路错误"
    missed_key_symptoms: List[str] = field(default_factory=list)

    # === 子图反馈4：追问策略 ===
    redundant_questions: List[str] = field(default_factory=list)
    missed_critical_info: List[str] = field(default_factory=list)
    inquiry_rounds: Optional[int] = None
    expected_rounds: Optional[int] = None

    # === 准确率评分 ===
    intent_accuracy: Optional[float] = None          # 意图识别准确度 (0-1)
    symptom_understanding_accuracy: Optional[float] = None  # 症状理解准确度 (0-1)
    diagnosis_accuracy: Optional[float] = None       # 诊断准确度 (0-1)
    inquiry_efficiency: Optional[float] = None       # 追问效率 (0-1)
    overall_satisfaction: Optional[float] = None     # 总体满意度 (1-5)

    # 元数据
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def sentiment(self) -> str:
        """情感倾向"""
        if self.overall_satisfaction:
            if self.overall_satisfaction >= 4:
                return "positive"
            elif self.overall_satisfaction <= 2:
                return "negative"
            else:
                return "neutral"

        # 根据准确率判断
        if self.diagnosis_accuracy is not None:
            if self.diagnosis_accuracy >= 0.8:
                return "positive"
            elif self.diagnosis_accuracy < 0.5:
                return "negative"

        return "neutral"

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "feedback_type": self.feedback_type.value,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "conversation_stage": self.conversation_stage,
            "subgraph": self.subgraph,

            # 主图
            "wrong_intent": self.wrong_intent,
            "correct_intent": self.correct_intent,
            "intent_confusion_reason": self.intent_confusion_reason,

            # 子图
            "wrong_tool": self.wrong_tool,
            "correct_tool": self.correct_tool,
            "tool_error_reason": self.tool_error_reason,
            "missing_info_for_tool": self.missing_info_for_tool,

            "misunderstood_symptom": self.misunderstood_symptom,
            "correct_symptom_interpretation": self.correct_symptom_interpretation,

            "agent_syndrome": self.agent_syndrome,
            "correct_syndrome": self.correct_syndrome,
            "diagnosis_error_type": self.diagnosis_error_type,
            "missed_key_symptoms": self.missed_key_symptoms,

            "redundant_questions": self.redundant_questions,
            "missed_critical_info": self.missed_critical_info,
            "inquiry_rounds": self.inquiry_rounds,
            "expected_rounds": self.expected_rounds,

            # 准确率
            "intent_accuracy": self.intent_accuracy,
            "symptom_understanding_accuracy": self.symptom_understanding_accuracy,
            "diagnosis_accuracy": self.diagnosis_accuracy,
            "inquiry_efficiency": self.inquiry_efficiency,
            "overall_satisfaction": self.overall_satisfaction,

            "sentiment": self.sentiment,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class TCMFeedbackCollector:
    """中医反馈收集器"""

    def __init__(self):
        self.feedbacks: List[TCMUserFeedback] = []
        self._session_feedbacks: Dict[str, List[TCMUserFeedback]] = {}

    def collect_intent_correction(
        self,
        session_id: str,
        user_id: str,
        user_query: str,
        wrong_intent: str,
        correct_intent: str,
        reason: Optional[str] = None
    ) -> TCMUserFeedback:
        """
        收集意图纠正反馈

        Args:
            session_id: 会话ID
            user_id: 用户ID
            user_query: 用户查询
            wrong_intent: 错误的意图
            correct_intent: 正确的意图
            reason: 纠正原因
        """
        feedback = TCMUserFeedback(
            feedback_type=TCMFeedbackType.INTENT_CORRECTION,
            session_id=session_id,
            user_id=user_id,
            conversation_stage="intent_recognition",
            wrong_intent=wrong_intent,
            correct_intent=correct_intent,
            intent_confusion_reason=reason,
            intent_accuracy=0.0,  # 识别错误，准确率为0
            metadata={"user_query": user_query}
        )

        self._add_feedback(feedback)

        logger.info(
            f"[Feedback] Intent correction: {wrong_intent} -> {correct_intent}"
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

        feedback = TCMUserFeedback(
            feedback_type=TCMFeedbackType.TOOL_SELECTION_ERROR,
            session_id=session_id,
            user_id=user_id,
            conversation_stage="info_collection",
            subgraph=subgraph,
            wrong_tool=wrong_tool,
            correct_tool=correct_tool,
            tool_error_reason=reason,
            missing_info_for_tool=missing_info or []
        )

        self._add_feedback(feedback)

        logger.info(
            f"[Feedback] Tool selection error: {wrong_tool} -> {correct_tool}"
        )

        return feedback

    def collect_symptom_misunderstanding(
        self,
        session_id: str,
        user_id: str,
        subgraph: str,
        misunderstood_symptom: str,
        correct_interpretation: str,
        disambiguation_needed: Optional[str] = None
    ) -> TCMUserFeedback:
        """收集症状理解错误反馈"""

        feedback = TCMUserFeedback(
            feedback_type=TCMFeedbackType.SYMPTOM_MISUNDERSTANDING,
            session_id=session_id,
            user_id=user_id,
            conversation_stage="info_collection",
            subgraph=subgraph,
            misunderstood_symptom=misunderstood_symptom,
            correct_symptom_interpretation=correct_interpretation,
            symptom_disambiguation_needed=disambiguation_needed,
            symptom_understanding_accuracy=0.0
        )

        self._add_feedback(feedback)

        logger.info(
            f"[Feedback] Symptom misunderstanding: {misunderstood_symptom}"
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
        missed_symptoms: List[str] = None
    ) -> TCMUserFeedback:
        """收集诊断错误反馈"""

        feedback = TCMUserFeedback(
            feedback_type=TCMFeedbackType.DIAGNOSIS_ERROR,
            session_id=session_id,
            user_id=user_id,
            conversation_stage="diagnosis",
            subgraph=subgraph,
            agent_syndrome=agent_syndrome,
            correct_syndrome=correct_syndrome,
            diagnosis_error_type=error_type,
            missed_key_symptoms=missed_symptoms or [],
            diagnosis_accuracy=0.0  # 诊断错误，准确率为0
        )

        self._add_feedback(feedback)

        logger.info(
            f"[Feedback] Diagnosis error: {agent_syndrome} -> {correct_syndrome} ({error_type})"
        )

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

        efficiency = expected_rounds / actual_rounds if actual_rounds > 0 else 0

        feedback = TCMUserFeedback(
            feedback_type=TCMFeedbackType.INQUIRY_INEFFICIENCY,
            session_id=session_id,
            user_id=user_id,
            conversation_stage="info_collection",
            subgraph=subgraph,
            redundant_questions=redundant_questions,
            missed_critical_info=missed_info,
            inquiry_rounds=actual_rounds,
            expected_rounds=expected_rounds,
            inquiry_efficiency=efficiency
        )

        self._add_feedback(feedback)

        logger.info(
            f"[Feedback] Inquiry inefficiency: {actual_rounds} rounds (expected {expected_rounds})"
        )

        return feedback

    def collect_diagnosis_accuracy_rating(
        self,
        session_id: str,
        user_id: str,
        diagnosis_accuracy: float,
        overall_satisfaction: float,
        comments: Optional[str] = None
    ) -> TCMUserFeedback:
        """收集诊断准确性评分"""

        feedback = TCMUserFeedback(
            feedback_type=TCMFeedbackType.DIAGNOSIS_ACCURACY,
            session_id=session_id,
            user_id=user_id,
            conversation_stage="diagnosis",
            diagnosis_accuracy=diagnosis_accuracy,
            overall_satisfaction=overall_satisfaction,
            metadata={"comments": comments} if comments else {}
        )

        self._add_feedback(feedback)

        logger.info(
            f"[Feedback] Diagnosis accuracy rating: {diagnosis_accuracy:.2f}"
        )

        return feedback

    def _add_feedback(self, feedback: TCMUserFeedback):
        """添加反馈"""
        self.feedbacks.append(feedback)

        # 按会话分组
        if feedback.session_id not in self._session_feedbacks:
            self._session_feedbacks[feedback.session_id] = []
        self._session_feedbacks[feedback.session_id].append(feedback)

    def get_session_feedbacks(self, session_id: str) -> List[TCMUserFeedback]:
        """获取会话的所有反馈"""
        return self._session_feedbacks.get(session_id, [])

    def get_accuracy_stats(self, session_id: Optional[str] = None) -> Dict:
        """
        获取准确率统计

        Args:
            session_id: 会话ID (可选，不指定则统计全部)

        Returns:
            准确率统计信息
        """
        feedbacks = self.get_session_feedbacks(session_id) if session_id else self.feedbacks

        if not feedbacks:
            return {
                "total": 0,
                "intent_accuracy": None,
                "symptom_understanding_accuracy": None,
                "diagnosis_accuracy": None,
                "inquiry_efficiency": None,
                "overall_satisfaction": None
            }

        # 计算各维度准确率
        intent_accuracies = [f.intent_accuracy for f in feedbacks if f.intent_accuracy is not None]
        symptom_accuracies = [f.symptom_understanding_accuracy for f in feedbacks if f.symptom_understanding_accuracy is not None]
        diagnosis_accuracies = [f.diagnosis_accuracy for f in feedbacks if f.diagnosis_accuracy is not None]
        inquiry_efficiencies = [f.inquiry_efficiency for f in feedbacks if f.inquiry_efficiency is not None]
        satisfactions = [f.overall_satisfaction for f in feedbacks if f.overall_satisfaction is not None]

        return {
            "total": len(feedbacks),
            "intent_accuracy": sum(intent_accuracies) / len(intent_accuracies) if intent_accuracies else None,
            "symptom_understanding_accuracy": sum(symptom_accuracies) / len(symptom_accuracies) if symptom_accuracies else None,
            "diagnosis_accuracy": sum(diagnosis_accuracies) / len(diagnosis_accuracies) if diagnosis_accuracies else None,
            "inquiry_efficiency": sum(inquiry_efficiencies) / len(inquiry_efficiencies) if inquiry_efficiencies else None,
            "overall_satisfaction": sum(satisfactions) / len(satisfactions) if satisfactions else None,
            "by_type": self._count_by_type(feedbacks)
        }

    def _count_by_type(self, feedbacks: List[TCMUserFeedback]) -> Dict:
        """按类型统计反馈"""
        result: Dict = {}
        for f in feedbacks:
            key = f.feedback_type.value
            result[key] = result.get(key, 0) + 1
        return result


class TCMFeedbackAggregator:
    """中医反馈聚合器"""

    def __init__(self, collector: TCMFeedbackCollector):
        self.collector = collector

    def identify_accuracy_issues(self) -> List[Dict]:
        """识别准确率问题"""
        issues = []

        # 检查意图识别准确率
        intent_errors = [f for f in self.collector.feedbacks
                        if f.feedback_type == TCMFeedbackType.INTENT_CORRECTION]
        if len(intent_errors) > 5:
            issues.append({
                "type": "intent_recognition_low_accuracy",
                "description": f"意图识别错误频繁: {len(intent_errors)}次",
                "count": len(intent_errors),
                "severity": "high" if len(intent_errors) > 10 else "medium"
            })

        # 检查诊断准确率
        diagnosis_errors = [f for f in self.collector.feedbacks
                           if f.feedback_type == TCMFeedbackType.DIAGNOSIS_ERROR]
        if len(diagnosis_errors) > 5:
            # 分析错误模式
            error_patterns = {}
            for f in diagnosis_errors:
                if f.agent_syndrome and f.correct_syndrome:
                    pattern = f"{f.agent_syndrome} vs {f.correct_syndrome}"
                    error_patterns[pattern] = error_patterns.get(pattern, 0) + 1

            most_common = max(error_patterns, key=error_patterns.get) if error_patterns else None

            issues.append({
                "type": "diagnosis_low_accuracy",
                "description": f"诊断错误频繁: {len(diagnosis_errors)}次",
                "count": len(diagnosis_errors),
                "most_common_pattern": most_common,
                "severity": "high"
            })

        # 检查追问效率
        inquiry_issues = [f for f in self.collector.feedbacks
                         if f.feedback_type == TCMFeedbackType.INQUIRY_INEFFICIENCY]
        if len(inquiry_issues) > 3:
            issues.append({
                "type": "inquiry_low_efficiency",
                "description": f"追问效率低: {len(inquiry_issues)}次",
                "count": len(inquiry_issues),
                "severity": "medium"
            })

        return issues

    def get_error_patterns(self) -> Dict[str, List[Dict]]:
        """获取错误模式"""
        patterns = {
            "intent_confusion": [],
            "diagnosis_confusion": [],
            "symptom_misunderstanding": []
        }

        # 意图混淆模式
        intent_errors = [f for f in self.collector.feedbacks
                        if f.feedback_type == TCMFeedbackType.INTENT_CORRECTION]
        intent_confusion = {}
        for f in intent_errors:
            if f.wrong_intent and f.correct_intent:
                key = f"{f.wrong_intent} -> {f.correct_intent}"
                if key not in intent_confusion:
                    intent_confusion[key] = {"count": 0, "examples": []}
                intent_confusion[key]["count"] += 1
                intent_confusion[key]["examples"].append(f.metadata.get("user_query", ""))

        patterns["intent_confusion"] = [
            {"pattern": k, **v} for k, v in intent_confusion.items()
        ]

        # 诊断混淆模式
        diagnosis_errors = [f for f in self.collector.feedbacks
                           if f.feedback_type == TCMFeedbackType.DIAGNOSIS_ERROR]
        diagnosis_confusion = {}
        for f in diagnosis_errors:
            if f.agent_syndrome and f.correct_syndrome:
                key = f"{f.agent_syndrome} -> {f.correct_syndrome}"
                if key not in diagnosis_confusion:
                    diagnosis_confusion[key] = {"count": 0, "error_types": []}
                diagnosis_confusion[key]["count"] += 1
                if f.diagnosis_error_type:
                    diagnosis_confusion[key]["error_types"].append(f.diagnosis_error_type)

        patterns["diagnosis_confusion"] = [
            {"pattern": k, **v} for k, v in diagnosis_confusion.items()
        ]

        return patterns
