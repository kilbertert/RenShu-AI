"""
中医问诊反思模块 (TCM Reflection)

职责：分析准确率相关的错误原因，生成可操作的改进规则
- 主图：意图识别错误反思
- 子图：工具选择、症状理解、辨证推理、追问策略错误反思
"""

import logging
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


logger = logging.getLogger(__name__)


class TCMReflectionType(str, Enum):
    """中医反思类型"""

    # === 主图反思 ===
    INTENT_RECOGNITION_ERROR = "intent_recognition_error"

    # === 子图反思 ===
    TOOL_SELECTION_ERROR = "tool_selection_error"
    SYMPTOM_INTERPRETATION_ERROR = "symptom_interpretation_error"
    DIAGNOSIS_REASONING_ERROR = "diagnosis_reasoning_error"
    INQUIRY_INEFFICIENCY = "inquiry_inefficiency"


@dataclass
class TCMReflectionResult:
    """中医反思结果"""
    reflection_type: TCMReflectionType
    session_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # === 错误分析 ===
    error_root_cause: Optional[str] = None
    error_pattern: Optional[str] = None

    # === 主图反思：意图识别 ===
    intent_confusion_cause: Optional[str] = None
    correct_intent_clues: List[str] = field(default_factory=list)
    intent_improvement_rule: Optional[str] = None
    disambiguation_question: Optional[str] = None

    # === 子图反思：工具选择 ===
    tool_selection_error_cause: Optional[str] = None
    correct_tool_timing: Optional[str] = None
    tool_selection_rule: Optional[str] = None
    missing_info_for_tool: List[str] = field(default_factory=list)

    # === 子图反思：症状理解 ===
    symptom_misinterpretation_cause: Optional[str] = None
    correct_symptom_meaning: Optional[str] = None
    symptom_disambiguation_rule: Optional[str] = None

    # === 子图反思：辨证推理 ===
    diagnosis_error_cause: Optional[str] = None
    missed_key_symptoms: List[str] = field(default_factory=list)
    misinterpreted_symptoms: List[str] = field(default_factory=list)
    correct_reasoning_path: Optional[str] = None
    discriminating_rule: Optional[str] = None
    prevention_checklist: List[str] = field(default_factory=list)

    # === 子图反思：追问策略 ===
    inquiry_inefficiency_cause: Optional[str] = None
    redundancy_analysis: Optional[str] = None
    missing_info_analysis: Optional[str] = None
    optimal_inquiry_sequence: List[str] = field(default_factory=list)

    # === 通用 ===
    lessons_learned: Optional[str] = None
    actionable_improvements: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "reflection_type": self.reflection_type.value,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "error_root_cause": self.error_root_cause,
            "error_pattern": self.error_pattern,

            # 主图
            "intent_confusion_cause": self.intent_confusion_cause,
            "correct_intent_clues": self.correct_intent_clues,
            "intent_improvement_rule": self.intent_improvement_rule,
            "disambiguation_question": self.disambiguation_question,

            # 子图
            "tool_selection_error_cause": self.tool_selection_error_cause,
            "correct_tool_timing": self.correct_tool_timing,
            "tool_selection_rule": self.tool_selection_rule,
            "missing_info_for_tool": self.missing_info_for_tool,

            "symptom_misinterpretation_cause": self.symptom_misinterpretation_cause,
            "correct_symptom_meaning": self.correct_symptom_meaning,
            "symptom_disambiguation_rule": self.symptom_disambiguation_rule,

            "diagnosis_error_cause": self.diagnosis_error_cause,
            "missed_key_symptoms": self.missed_key_symptoms,
            "misinterpreted_symptoms": self.misinterpreted_symptoms,
            "correct_reasoning_path": self.correct_reasoning_path,
            "discriminating_rule": self.discriminating_rule,
            "prevention_checklist": self.prevention_checklist,

            "inquiry_inefficiency_cause": self.inquiry_inefficiency_cause,
            "redundancy_analysis": self.redundancy_analysis,
            "missing_info_analysis": self.missing_info_analysis,
            "optimal_inquiry_sequence": self.optimal_inquiry_sequence,

            "lessons_learned": self.lessons_learned,
            "actionable_improvements": self.actionable_improvements
        }


# 中医专用反思提示词
TCM_REFLECTION_PROMPTS = {
    TCMReflectionType.INTENT_RECOGNITION_ERROR: """
你的意图识别模块出错了：

【用户输入】{user_query}
【你识别的意图】{predicted_intent}
【正确的意图】{correct_intent}
【上下文】{context}

请深入分析：
1. 为什么会识别错？（关键词误导？上下文理解错误？）
2. 用户输入中哪些线索指向正确意图？
3. 如何改进意图识别规则？（具体的判断规则）
4. 如果不确定，应该如何消歧？

返回JSON格式：
{{
  "error_root_cause": "用户说'开点药'，我误以为是处方咨询，实际是草药查询",
  "correct_intent_clues": ["'开点药'在口语中常指草药", "没有提到具体症状"],
  "intent_improvement_rule": "当用户说'开/配药'但未描述症状时，优先判断为草药咨询",
  "disambiguation_question": "您是想了解某种草药，还是需要根据症状开处方？"
}}
""",

    TCMReflectionType.TOOL_SELECTION_ERROR: """
你在{subgraph}中选错了工具：

【当前症状】{symptoms}
【已收集信息】{collected_info}
【你选择的工具】{wrong_tool}
【正确的工具】{correct_tool}

请分析：
1. 为什么选错了？（信息不足？判断逻辑错误？）
2. 什么时候应该调用 {correct_tool}？
3. 如何改进工具选择逻辑？

返回JSON格式：
{{
  "error_root_cause": "症状只有'怕冷'，信息不足，不应该调用辨证工具",
  "correct_tool_timing": "至少收集寒热、汗出、二便、舌脉后才能辨证",
  "tool_selection_rule": "症状数量 < 5 时，优先调用信息收集工具",
  "missing_info_for_tool": ["汗出情况", "二便情况", "舌脉"]
}}
""",

    TCMReflectionType.SYMPTOM_INTERPRETATION_ERROR: """
你误解了用户的症状描述：

【用户描述】{user_symptom}
【你的理解】{agent_interpretation}
【正确理解】{correct_interpretation}

请分析：
1. 为什么会误解？
2. 正确的理解应该是什么？
3. 如何避免类似误解？（消歧规则）

返回JSON格式：
{{
  "error_root_cause": "将'怕冷'理解为恶寒（外感），实际是畏寒（阳虚）",
  "correct_symptom_meaning": "用户说的是平时怕冷，不是发冷",
  "symptom_disambiguation_rule": "遇到'怕冷'，必须追问：是最近突然怕冷（外感），还是一直怕冷（阳虚）"
}}
""",

    TCMReflectionType.DIAGNOSIS_REASONING_ERROR: """
你的辨证出错了：

【症状信息】
主症：{main_symptoms}
次症：{secondary_symptoms}
舌象：{tongue}
脉象：{pulse}

【你的辨证】{agent_syndrome}
【正确辨证】{correct_syndrome}
【错误类型】{error_type}

请深入分析：
1. 为什么会误诊？（哪个症状误判了？辨证思路错了？）
2. 关键鉴别点是什么？
3. 如何避免这类错误？（具体的判断规则）

返回JSON格式：
{{
  "error_root_cause": "将'腰膝酸软'当作普通疲劳，忽略了肾的定位意义",
  "missed_key_symptoms": ["腰膝酸软（肾的定位症状）", "夜尿频多（肾阳虚特征）"],
  "misinterpreted_symptoms": ["畏寒 -> 误认为单纯阳虚，未区分脏腑"],
  "correct_reasoning_path": "应先看定位症状确定脏腑（腰膝->肾），再看性质症状确定寒热虚实（畏寒->阳虚），得出肾阳虚",
  "discriminating_rule": "肾阳虚必有腰膝症状或夜尿，脾阳虚必有消化症状，据此区分",
  "prevention_checklist": ["遇到阳虚证，必问腰膝和消化", "腰膝症状->肾", "消化症状->脾"]
}}
""",

    TCMReflectionType.INQUIRY_INEFFICIENCY: """
这次追问效率不高：

【追问历史】{inquiry_history}
【冗余问题】{redundant_questions}
【遗漏信息】{missed_info}
【总轮数】{rounds}

请分析：
1. 为什么会有冗余追问？
2. 为什么会遗漏关键信息？
3. 最优的追问顺序应该是什么？

返回JSON格式：
{{
  "redundancy_analysis": "第3轮和第5轮都问了寒热，因为没有记住之前的回答",
  "missing_info_analysis": "过于关注主症，忘记问舌脉",
  "optimal_inquiry_sequence": ["主诉", "寒热", "汗出", "二便", "舌脉", "辨证"],
  "lessons_learned": "按十问歌顺序，每个维度只问一次，问完立即记录"
}}
"""
}


class TCMReflectionEngine:
    """中医反思引擎"""

    def __init__(self, llm=None):
        self.llm = llm
        self.reflection_memory: List[TCMReflectionResult] = []

    # ========== 主图反思：意图识别错误 ==========

    async def reflect_on_intent_error(
        self,
        session_id: str,
        user_query: str,
        predicted_intent: str,
        correct_intent: str,
        context: Optional[Dict] = None
    ) -> TCMReflectionResult:
        """
        反思意图识别错误

        Args:
            session_id: 会话ID
            user_query: 用户查询
            predicted_intent: 预测的意图
            correct_intent: 正确的意图
            context: 上下文信息

        Returns:
            反思结果
        """
        prompt = TCM_REFLECTION_PROMPTS[TCMReflectionType.INTENT_RECOGNITION_ERROR].format(
            user_query=user_query,
            predicted_intent=predicted_intent,
            correct_intent=correct_intent,
            context=context or {}
        )

        reflection = await self._generate_reflection(
            TCMReflectionType.INTENT_RECOGNITION_ERROR,
            session_id,
            prompt
        )

        self.reflection_memory.append(reflection)
        logger.info(f"[Reflection] Intent error reflection completed for session {session_id}")

        return reflection

    # ========== 子图反思：工具选择错误 ==========

    async def reflect_on_tool_selection_error(
        self,
        session_id: str,
        subgraph: str,
        symptoms: List[str],
        collected_info: Dict,
        wrong_tool: str,
        correct_tool: str
    ) -> TCMReflectionResult:
        """反思工具选择错误"""

        prompt = TCM_REFLECTION_PROMPTS[TCMReflectionType.TOOL_SELECTION_ERROR].format(
            subgraph=subgraph,
            symptoms=symptoms,
            collected_info=collected_info,
            wrong_tool=wrong_tool,
            correct_tool=correct_tool
        )

        reflection = await self._generate_reflection(
            TCMReflectionType.TOOL_SELECTION_ERROR,
            session_id,
            prompt
        )

        self.reflection_memory.append(reflection)
        logger.info(f"[Reflection] Tool selection error reflection completed")

        return reflection

    # ========== 子图反思：症状理解错误 ==========

    async def reflect_on_symptom_interpretation_error(
        self,
        session_id: str,
        user_symptom: str,
        agent_interpretation: str,
        correct_interpretation: str
    ) -> TCMReflectionResult:
        """反思症状理解错误"""

        prompt = TCM_REFLECTION_PROMPTS[TCMReflectionType.SYMPTOM_INTERPRETATION_ERROR].format(
            user_symptom=user_symptom,
            agent_interpretation=agent_interpretation,
            correct_interpretation=correct_interpretation
        )

        reflection = await self._generate_reflection(
            TCMReflectionType.SYMPTOM_INTERPRETATION_ERROR,
            session_id,
            prompt
        )

        self.reflection_memory.append(reflection)
        logger.info(f"[Reflection] Symptom interpretation error reflection completed")

        return reflection

    # ========== 子图反思：辨证推理错误 ==========

    async def reflect_on_diagnosis_error(
        self,
        session_id: str,
        symptoms: Dict,
        agent_syndrome: str,
        correct_syndrome: str,
        error_type: str
    ) -> TCMReflectionResult:
        """
        反思辨证错误

        Args:
            session_id: 会话ID
            symptoms: 症状信息
            agent_syndrome: Agent的证型
            correct_syndrome: 正确的证型
            error_type: 错误类型

        Returns:
            反思结果
        """
        prompt = TCM_REFLECTION_PROMPTS[TCMReflectionType.DIAGNOSIS_REASONING_ERROR].format(
            main_symptoms=symptoms.get('main', []),
            secondary_symptoms=symptoms.get('secondary', []),
            tongue=symptoms.get('tongue', '未采集'),
            pulse=symptoms.get('pulse', '未采集'),
            agent_syndrome=agent_syndrome,
            correct_syndrome=correct_syndrome,
            error_type=error_type
        )

        reflection = await self._generate_reflection(
            TCMReflectionType.DIAGNOSIS_REASONING_ERROR,
            session_id,
            prompt
        )

        self.reflection_memory.append(reflection)
        logger.info(f"[Reflection] Diagnosis error reflection completed")

        return reflection

    # ========== 子图反思：追问策略低效 ==========

    async def reflect_on_inquiry_inefficiency(
        self,
        session_id: str,
        inquiry_history: List[str],
        redundant_questions: List[str],
        missed_info: List[str],
        rounds: int
    ) -> TCMReflectionResult:
        """反思追问策略低效"""

        prompt = TCM_REFLECTION_PROMPTS[TCMReflectionType.INQUIRY_INEFFICIENCY].format(
            inquiry_history=inquiry_history,
            redundant_questions=redundant_questions,
            missed_info=missed_info,
            rounds=rounds
        )

        reflection = await self._generate_reflection(
            TCMReflectionType.INQUIRY_INEFFICIENCY,
            session_id,
            prompt
        )

        self.reflection_memory.append(reflection)
        logger.info(f"[Reflection] Inquiry inefficiency reflection completed")

        return reflection

    # ========== 反思生成 ==========

    async def _generate_reflection(
        self,
        reflection_type: TCMReflectionType,
        session_id: str,
        prompt: str
    ) -> TCMReflectionResult:
        """生成反思内容"""

        if self.llm:
            try:
                response = await self.llm.ainvoke(prompt)
                content = response.content if hasattr(response, 'content') else str(response)

                # 尝试解析JSON
                try:
                    data = json.loads(content)
                    return self._build_reflection_result(reflection_type, session_id, data)
                except json.JSONDecodeError:
                    logger.warning(f"[Reflection] Failed to parse JSON response: {content[:100]}")
            except Exception as e:
                logger.error(f"[Reflection] LLM reflection failed: {e}")

        # 返回默认反思
        return TCMReflectionResult(
            reflection_type=reflection_type,
            session_id=session_id,
            error_root_cause="LLM反思未完成",
            lessons_learned="需要人工分析"
        )

    def _build_reflection_result(
        self,
        reflection_type: TCMReflectionType,
        session_id: str,
        data: Dict
    ) -> TCMReflectionResult:
        """构建反思结果"""

        result = TCMReflectionResult(
            reflection_type=reflection_type,
            session_id=session_id,
            error_root_cause=data.get("error_root_cause")
        )

        # 根据反思类型填充不同字段
        if reflection_type == TCMReflectionType.INTENT_RECOGNITION_ERROR:
            result.intent_confusion_cause = data.get("error_root_cause")
            result.correct_intent_clues = data.get("correct_intent_clues", [])
            result.intent_improvement_rule = data.get("intent_improvement_rule")
            result.disambiguation_question = data.get("disambiguation_question")

        elif reflection_type == TCMReflectionType.TOOL_SELECTION_ERROR:
            result.tool_selection_error_cause = data.get("error_root_cause")
            result.correct_tool_timing = data.get("correct_tool_timing")
            result.tool_selection_rule = data.get("tool_selection_rule")
            result.missing_info_for_tool = data.get("missing_info_for_tool", [])

        elif reflection_type == TCMReflectionType.SYMPTOM_INTERPRETATION_ERROR:
            result.symptom_misinterpretation_cause = data.get("error_root_cause")
            result.correct_symptom_meaning = data.get("correct_symptom_meaning")
            result.symptom_disambiguation_rule = data.get("symptom_disambiguation_rule")

        elif reflection_type == TCMReflectionType.DIAGNOSIS_REASONING_ERROR:
            result.diagnosis_error_cause = data.get("error_root_cause")
            result.missed_key_symptoms = data.get("missed_key_symptoms", [])
            result.misinterpreted_symptoms = data.get("misinterpreted_symptoms", [])
            result.correct_reasoning_path = data.get("correct_reasoning_path")
            result.discriminating_rule = data.get("discriminating_rule")
            result.prevention_checklist = data.get("prevention_checklist", [])

        elif reflection_type == TCMReflectionType.INQUIRY_INEFFICIENCY:
            result.inquiry_inefficiency_cause = data.get("error_root_cause")
            result.redundancy_analysis = data.get("redundancy_analysis")
            result.missing_info_analysis = data.get("missing_info_analysis")
            result.optimal_inquiry_sequence = data.get("optimal_inquiry_sequence", [])
            result.lessons_learned = data.get("lessons_learned")

        return result

    def get_reflection_summary(self) -> Dict:
        """获取反思摘要"""
        return {
            "total_reflections": len(self.reflection_memory),
            "by_type": self._count_by_type(),
            "recent_lessons": self._get_recent_lessons(5)
        }

    def _count_by_type(self) -> Dict:
        """按类型统计反思"""
        result: Dict = {}
        for r in self.reflection_memory:
            key = r.reflection_type.value
            result[key] = result.get(key, 0) + 1
        return result

    def _get_recent_lessons(self, n: int = 5) -> List[str]:
        """获取最近的经验教训"""
        lessons = []
        for r in reversed(self.reflection_memory[-n:]):
            if r.lessons_learned:
                lessons.append(r.lessons_learned)
            elif r.discriminating_rule:
                lessons.append(r.discriminating_rule)
            elif r.intent_improvement_rule:
                lessons.append(r.intent_improvement_rule)
        return lessons
