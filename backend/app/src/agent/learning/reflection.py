"""
自我反思模块 (Self-Reflection)

基于 Reflexion (NeurIPS 2023) 的自我反思机制
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


logger = logging.getLogger(__name__)


class ReflectionType(str, Enum):
    """反思类型"""
    ON_ERROR = "on_error"           # 错误反思
    ON_COMPLETION = "on_completion"  # 完成反思
    PERIODIC = "periodic"           # 定期反思
    ON_FEEDBACK = "on_feedback"     # 反馈反思


@dataclass
class ReflectionPrompt:
    """反思提示"""
    reflection_type: ReflectionType
    template: str
    context_fields: List[str] = field(default_factory=list)

    def build(self, context: Dict) -> str:
        """构建反思提示"""
        values = {k: context.get(k, "") for k in self.context_fields}
        return self.template.format(**values)


# 预定义反思提示
REFLECTION_PROMPTS = {
    ReflectionType.ON_ERROR: ReflectionPrompt(
        reflection_type=ReflectionType.ON_ERROR,
        template="""任务执行失败，请进行自我反思：

【任务目标】{goal}

【执行结果】{result}

【错误信息】{error}

请分析：
1. 为什么会失败？
2. 哪些假设是错误的？
3. 应该如何改进？

请以JSON格式返回：
{{
  "root_cause": "根本原因",
  "incorrect_assumptions": ["错误假设1", "错误假设2"],
  "improvement_plan": "改进计划",
  "new_strategy": "新策略"
}}""",
        context_fields=["goal", "result", "error"]
    ),

    ReflectionType.ON_COMPLETION: ReflectionPrompt(
        reflection_type=ReflectionType.ON_COMPLETION,
        template="""任务已完成，请进行自我反思：

【任务目标】{goal}

【执行结果】{result}

【用户反馈】{feedback}

请分析：
1. 结果是否达到预期？
2. 哪些做法是有效的？
3. 哪些地方可以改进？

请以JSON格式返回：
{{
  "success_assessment": "成功评估",
  "effective_practices": ["有效做法1", "有效做法2"],
  "improvement_areas": ["改进点1", "改进点2"],
  "lessons_learned": "学到的经验"
}}""",
        context_fields=["goal", "result", "feedback"]
    ),

    ReflectionType.ON_FEEDBACK: ReflectionPrompt(
        reflection_type=ReflectionType.ON_FEEDBACK,
        template="""收到用户反馈，请进行反思：

【原始响应】{original_response}

【用户反馈】{feedback}

【反馈类型】{feedback_type}

请分析：
1. 用户为什么不满意？
2. 响应中存在什么问题？
3. 如何改进类似的响应？

请以JSON格式返回：
{{
  "issue_identified": "识别出的问题",
  "root_cause": "根本原因",
  "correction_strategy": "纠正策略",
  "prevention_plan": "预防计划"
}}""",
        context_fields=["original_response", "feedback", "feedback_type"]
    ),
}


@dataclass
class ReflectionResult:
    """反思结果"""
    reflection_type: ReflectionType
    timestamp: datetime = field(default_factory=datetime.now)

    # 分析内容
    root_cause: Optional[str] = None
    incorrect_assumptions: List[str] = field(default_factory=list)
    improvement_plan: Optional[str] = None
    new_strategy: Optional[str] = None

    # 成功评估
    success_assessment: Optional[str] = None
    effective_practices: List[str] = field(default_factory=list)
    improvement_areas: List[str] = field(default_factory=list)
    lessons_learned: Optional[str] = None

    # 纠正内容
    issue_identified: Optional[str] = None
    correction_strategy: Optional[str] = None
    prevention_plan: Optional[str] = None

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "reflection_type": self.reflection_type.value,
            "timestamp": self.timestamp.isoformat(),
            "root_cause": self.root_cause,
            "incorrect_assumptions": self.incorrect_assumptions,
            "improvement_plan": self.improvement_plan,
            "new_strategy": self.new_strategy,
            "success_assessment": self.success_assessment,
            "effective_practices": self.effective_practices,
            "improvement_areas": self.improvement_areas,
            "lessons_learned": self.lessons_learned,
            "issue_identified": self.issue_identified,
            "correction_strategy": self.correction_strategy,
            "prevention_plan": self.prevention_plan
        }


class ReflectionMemory:
    """反思记忆"""

    def __init__(self, max_memories: int = 100):
        self.reflections: List[ReflectionResult] = []
        self.max_memories = max_memories

    def add(self, reflection: ReflectionResult) -> None:
        """添加反思记录"""
        self.reflections.append(reflection)
        if len(self.reflections) > self.max_memories:
            # 移除最旧的记录
            self.reflections.pop(0)

    def get_recent(self, n: int = 5) -> List[ReflectionResult]:
        """获取最近的反思"""
        return self.reflections[-n:]

    def get_by_type(self, reflection_type: ReflectionType) -> List[ReflectionResult]:
        """按类型获取反思"""
        return [r for r in self.reflections if r.reflection_type == reflection_type]

    def get_relevant_lessons(self, context: str) -> List[str]:
        """获取相关的经验教训"""
        lessons = []
        for r in self.reflections:
            if r.lessons_learned:
                lessons.append(r.lessons_learned)
            if r.improvement_plan:
                lessons.append(r.improvement_plan)
        return lessons[-10:]  # 最近10条

    def clear_old(self, days: int = 30) -> None:
        """清除旧记录"""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=days)
        self.reflections = [r for r in self.reflections if r.timestamp > cutoff]


class SelfReflection:
    """自我反思引擎"""

    def __init__(self, llm=None, memory_max: int = 100):
        self.llm = llm
        self.memory = ReflectionMemory(max_memories=memory_max)
        self.prompts = REFLECTION_PROMPTS

    async def reflect_on_error(
        self,
        goal: str,
        result: str,
        error: str,
        context: Optional[Dict] = None
    ) -> ReflectionResult:
        """
        对错误进行反思

        Args:
            goal: 任务目标
            result: 执行结果
            error: 错误信息
            context: 额外上下文

        Returns:
            反思结果
        """
        prompt = self.prompts[ReflectionType.ON_ERROR].build({
            "goal": goal,
            "result": result,
            "error": error
        })

        reflection = await self._generate_reflection(
            ReflectionType.ON_ERROR,
            prompt,
            context
        )

        self.memory.add(reflection)
        logger.info(f"[Reflection] Error reflection completed")

        return reflection

    async def reflect_on_completion(
        self,
        goal: str,
        result: str,
        feedback: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> ReflectionResult:
        """
        对完成进行反思

        Args:
            goal: 任务目标
            result: 执行结果
            feedback: 用户反馈
            context: 额外上下文

        Returns:
            反思结果
        """
        prompt = self.prompts[ReflectionType.ON_COMPLETION].build({
            "goal": goal,
            "result": result,
            "feedback": feedback or "无反馈"
        })

        reflection = await self._generate_reflection(
            ReflectionType.ON_COMPLETION,
            prompt,
            context
        )

        self.memory.add(reflection)
        logger.info(f"[Reflection] Completion reflection completed")

        return reflection

    async def reflect_on_feedback(
        self,
        original_response: str,
        feedback: str,
        feedback_type: str,
        context: Optional[Dict] = None
    ) -> ReflectionResult:
        """
        对反馈进行反思

        Args:
            original_response: 原始响应
            feedback: 用户反馈
            feedback_type: 反馈类型
            context: 额外上下文

        Returns:
            反思结果
        """
        prompt = self.prompts[ReflectionType.ON_FEEDBACK].build({
            "original_response": original_response,
            "feedback": feedback,
            "feedback_type": feedback_type
        })

        reflection = await self._generate_reflection(
            ReflectionType.ON_FEEDBACK,
            prompt,
            context
        )

        self.memory.add(reflection)
        logger.info(f"[Reflection] Feedback reflection completed")

        return reflection

    async def _generate_reflection(
        self,
        reflection_type: ReflectionType,
        prompt: str,
        context: Optional[Dict]
    ) -> ReflectionResult:
        """生成反思内容"""
        if self.llm:
            try:
                import json
                response = await self.llm.ainvoke(prompt)
                content = response.content if hasattr(response, 'content') else str(response)

                # 尝试解析JSON
                try:
                    data = json.loads(content)
                    return ReflectionResult(
                        reflection_type=reflection_type,
                        **{k: v for k, v in data.items() if hasattr(ReflectionResult, k)}
                    )
                except json.JSONDecodeError:
                    pass
            except Exception as e:
                logger.warning(f"[Reflection] LLM reflection failed: {e}")

        # 返回默认反思
        return ReflectionResult(
            reflection_type=reflection_type,
            root_cause="LLM反思未完成",
            improvement_plan="需要人工分析"
        )

    def get_reflection_summary(self) -> Dict:
        """获取反思摘要"""
        return {
            "total_reflections": len(self.memory.reflections),
            "by_type": {
                "on_error": len(self.memory.get_by_type(ReflectionType.ON_ERROR)),
                "on_completion": len(self.memory.get_by_type(ReflectionType.ON_COMPLETION)),
                "on_feedback": len(self.memory.get_by_type(ReflectionType.ON_FEEDBACK))
            },
            "recent_lessons": self.memory.get_relevant_lessons("")
        }
