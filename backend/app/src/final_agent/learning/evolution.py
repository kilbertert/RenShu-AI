"""
中医问诊准确率进化引擎 (TCM Accuracy Evolution Engine)

职责：从错误和成功案例中提取知识，持续提升诊断准确率
- 主图：意图识别规则进化
- 子图：工具选择规则、辨证鉴别规则、追问策略进化
"""

import logging
import json
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


logger = logging.getLogger(__name__)


class EvolutionStrategy(str, Enum):
    """进化策略（针对准确率提升）"""

    # 主图进化
    INTENT_RULES = "intent_rules"                    # 意图识别规则进化

    # 子图进化
    TOOL_SELECTION_RULES = "tool_selection_rules"    # 工具选择规则进化
    DISCRIMINATING_RULES = "discriminating_rules"    # 辨证鉴别规则进化
    INQUIRY_OPTIMIZATION = "inquiry_optimization"    # 追问策略优化
    ERROR_PREVENTION = "error_prevention"            # 错误预防规则进化


@dataclass
class AccuracyMetrics:
    """准确率指标"""

    # 主图指标
    intent_recognition_accuracy: float = 0.0         # 意图识别准确率

    # 子图指标
    tool_selection_accuracy: float = 0.0             # 工具选择准确率
    symptom_understanding_accuracy: float = 0.0      # 症状理解准确率
    diagnosis_accuracy: float = 0.0                  # 辨证准确率
    inquiry_efficiency: float = 0.0                  # 追问效率
    prescription_accuracy: float = 0.0               # 处方准确率

    # 错误率指标
    misdiagnosis_rate: float = 0.0                   # 误诊率
    premature_diagnosis_rate: float = 0.0            # 过早诊断率
    information_insufficiency_rate: float = 0.0      # 信息不足率

    # 效率指标
    avg_rounds_to_correct_diagnosis: float = 0.0     # 平均诊断轮数

    # 错误模式统计
    error_patterns: Dict[str, int] = field(default_factory=dict)  # {"肾阳虚vs脾阳虚混淆": 15}

    def calculate_overall_quality(self) -> float:
        """计算综合诊断质量"""
        return (
            self.diagnosis_accuracy * 0.4 +
            (1 - self.misdiagnosis_rate) * 0.3 +
            self.inquiry_efficiency * 0.2 +
            (1 - self.premature_diagnosis_rate) * 0.1
        )

    def to_dict(self) -> Dict:
        return {
            "intent_recognition_accuracy": self.intent_recognition_accuracy,
            "tool_selection_accuracy": self.tool_selection_accuracy,
            "symptom_understanding_accuracy": self.symptom_understanding_accuracy,
            "diagnosis_accuracy": self.diagnosis_accuracy,
            "inquiry_efficiency": self.inquiry_efficiency,
            "prescription_accuracy": self.prescription_accuracy,
            "misdiagnosis_rate": self.misdiagnosis_rate,
            "premature_diagnosis_rate": self.premature_diagnosis_rate,
            "information_insufficiency_rate": self.information_insufficiency_rate,
            "avg_rounds_to_correct_diagnosis": self.avg_rounds_to_correct_diagnosis,
            "error_patterns": self.error_patterns,
            "overall_quality": self.calculate_overall_quality()
        }


@dataclass
class EvolutionRecord:
    """进化记录"""
    timestamp: datetime = field(default_factory=datetime.now)
    strategy: EvolutionStrategy = EvolutionStrategy.DISCRIMINATING_RULES

    # 变更前后的指标
    before_metrics: Optional[AccuracyMetrics] = None
    after_metrics: Optional[AccuracyMetrics] = None

    # 变更内容
    change_description: str = ""
    changed_rules: List[str] = field(default_factory=list)

    # 效果评估
    improvement: float = 0.0  # 准确率提升幅度
    successful: bool = False

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "strategy": self.strategy.value,
            "before_metrics": self.before_metrics.to_dict() if self.before_metrics else None,
            "after_metrics": self.after_metrics.to_dict() if self.after_metrics else None,
            "change_description": self.change_description,
            "changed_rules": self.changed_rules,
            "improvement": self.improvement,
            "successful": self.successful
        }


class IntentRuleEvolution:
    """意图识别规则进化"""

    def __init__(self, llm=None):
        self.llm = llm
        self.intent_rules: List[str] = []

    async def evolve_from_errors(
        self,
        intent_errors: List[Dict]
    ) -> List[str]:
        """
        从意图识别错误中提取改进规则

        Args:
            intent_errors: 意图识别错误案例列表

        Returns:
            新的意图识别规则列表
        """
        if not intent_errors or len(intent_errors) < 3:
            return []

        # 聚类相似错误
        error_patterns = self._cluster_intent_errors(intent_errors)

        new_rules = []
        for pattern in error_patterns:
            if pattern["frequency"] >= 3:
                rule = await self._extract_intent_rule(pattern)
                if rule:
                    new_rules.append(rule)

        return new_rules

    def _cluster_intent_errors(self, errors: List[Dict]) -> List[Dict]:
        """聚类相似的意图识别错误"""
        patterns: Dict[str, Dict] = {}

        for error in errors:
            wrong = error.get("wrong_intent", "")
            correct = error.get("correct_intent", "")
            key = f"{wrong} -> {correct}"

            if key not in patterns:
                patterns[key] = {
                    "wrong_intent": wrong,
                    "correct_intent": correct,
                    "frequency": 0,
                    "examples": []
                }

            patterns[key]["frequency"] += 1
            patterns[key]["examples"].append(error.get("user_query", ""))

        return list(patterns.values())

    async def _extract_intent_rule(self, pattern: Dict) -> Optional[str]:
        """提取意图识别规则"""
        if not self.llm:
            return None

        prompt = f"""
以下是 {pattern['frequency']} 个相似的意图识别错误：

错误模式：{pattern['wrong_intent']} -> {pattern['correct_intent']}
示例查询：{pattern['examples'][:5]}

请总结一条**明确的意图识别规则**：

要求：
1. 规则必须具体、可操作
2. 基于用户输入特征，不要模糊描述
3. 一句话说清楚

示例格式：
"当用户说'开药'但未描述症状时，优先判断为草药咨询而非处方咨询"

请给出规则：
"""

        try:
            response = await self.llm.ainvoke(prompt)
            rule = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            return rule
        except Exception as e:
            logger.error(f"[Evolution] Failed to extract intent rule: {e}")
            return None


class ToolSelectionEvolution:
    """工具选择规则进化"""

    def __init__(self, llm=None):
        self.llm = llm
        self.tool_rules: List[Dict] = []

    async def evolve_from_errors(
        self,
        tool_errors: List[Dict]
    ) -> List[Dict]:
        """从工具选择错误中提取规则"""
        if not tool_errors or len(tool_errors) < 3:
            return []

        rules = []
        for error in tool_errors:
            rule = {
                "condition": self._extract_condition(error),
                "wrong_tool": error.get("wrong_tool"),
                "correct_tool": error.get("correct_tool"),
                "reason": error.get("reason")
            }
            rules.append(rule)

        return rules

    def _extract_condition(self, error: Dict) -> str:
        """提取工具选择条件"""
        # 简单规则提取
        if "信息不足" in error.get("reason", ""):
            return "症状数量 < 5"
        elif "过早" in error.get("reason", ""):
            return "缺少舌脉信息"
        else:
            return "未知条件"


class DiagnosisRuleEvolution:
    """辨证鉴别规则进化（最核心）"""

    def __init__(self, llm=None):
        self.llm = llm
        self.discriminating_rules: List[str] = []

    async def evolve_from_errors(
        self,
        diagnosis_errors: List[Dict]
    ) -> List[str]:
        """
        从辨证错误中提取鉴别规则

        Args:
            diagnosis_errors: 辨证错误案例列表

        Returns:
            新的鉴别规则列表
        """
        if not diagnosis_errors or len(diagnosis_errors) < 5:
            return []

        # 按错误模式聚类
        error_patterns = self._cluster_diagnosis_errors(diagnosis_errors)

        discriminating_rules = []
        for pattern, cases in error_patterns.items():
            if len(cases) >= 5:  # 高频错误
                rule = await self._extract_discriminating_rule(pattern, cases)
                if rule:
                    discriminating_rules.append(rule)

        return discriminating_rules

    def _cluster_diagnosis_errors(self, errors: List[Dict]) -> Dict[str, List[Dict]]:
        """聚类诊断错误"""
        patterns: Dict[str, List[Dict]] = {}

        for error in errors:
            wrong = error.get("agent_syndrome", "")
            correct = error.get("correct_syndrome", "")
            if wrong and correct:
                # 标准化证型对（按字母顺序）
                syndromes = tuple(sorted([wrong, correct]))
                key = f"{syndromes[0]} vs {syndromes[1]}"

                if key not in patterns:
                    patterns[key] = []
                patterns[key].append(error)

        return patterns

    async def _extract_discriminating_rule(
        self,
        pattern: str,
        cases: List[Dict]
    ) -> Optional[str]:
        """提取鉴别规则"""
        if not self.llm:
            return None

        # 提取症状信息
        all_symptoms = []
        for case in cases[:5]:  # 最多取5个案例
            symptoms = case.get("symptoms", {})
            all_symptoms.append(symptoms)

        prompt = f"""
以下是 {len(cases)} 个"{pattern}"的误诊案例：

案例症状：
{json.dumps(all_symptoms, ensure_ascii=False, indent=2)}

请总结一条**明确的鉴别规则**，帮助区分这两个证型：

要求：
1. 基于症状特征，具体可操作
2. 一句话说清楚
3. 指出关键鉴别点

示例：
"肾阳虚必有腰膝酸软或夜尿频多，脾阳虚必有食少便溏或腹胀，据此区分"

请给出规则：
"""

        try:
            response = await self.llm.ainvoke(prompt)
            rule = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            return rule
        except Exception as e:
            logger.error(f"[Evolution] Failed to extract discriminating rule: {e}")
            return None


class InquiryOptimization:
    """追问策略优化"""

    def __init__(self):
        self.optimized_paths: Dict[str, List[str]] = {}

    async def evolve_from_successful_cases(
        self,
        successful_cases: List[Dict]
    ) -> Dict[str, List[str]]:
        """从成功案例中提取高效追问路径"""
        if not successful_cases or len(successful_cases) < 5:
            return {}

        # 按证型分组
        cases_by_syndrome = self._group_by_syndrome(successful_cases)

        optimized_paths = {}
        for syndrome, cases in cases_by_syndrome.items():
            if len(cases) >= 5:
                # 找出最短且成功的追问路径
                optimal_path = self._find_optimal_path(cases)
                optimized_paths[syndrome] = optimal_path

        return optimized_paths

    def _group_by_syndrome(self, cases: List[Dict]) -> Dict[str, List[Dict]]:
        """按证型分组"""
        grouped: Dict[str, List[Dict]] = {}
        for case in cases:
            syndrome = case.get("syndrome", "unknown")
            if syndrome not in grouped:
                grouped[syndrome] = []
            grouped[syndrome].append(case)
        return grouped

    def _find_optimal_path(self, cases: List[Dict]) -> List[str]:
        """找出最优追问路径"""
        # 统计每个案例的追问序列
        all_sequences = [case.get("inquiry_sequence", []) for case in cases]

        if not all_sequences:
            return []

        # 找出最短且成功率最高的序列
        optimal = min(all_sequences, key=lambda seq: (
            len(seq),  # 优先选短的
            -cases[all_sequences.index(seq)].get("user_satisfaction", 0)  # 满意度高的
        ))

        return optimal


class ErrorPreventionEvolution:
    """错误预防规则进化"""

    def __init__(self, llm=None):
        self.llm = llm
        self.prevention_rules: List[Dict] = []

    async def evolve_from_error_patterns(
        self,
        error_patterns: Dict[str, int]
    ) -> List[Dict]:
        """从错误模式中生成预防规则"""
        prevention_rules = []

        for pattern, frequency in error_patterns.items():
            if frequency >= 5:  # 高频错误
                rule = await self._generate_prevention_rule(pattern, frequency)
                if rule:
                    prevention_rules.append(rule)

        return prevention_rules

    async def _generate_prevention_rule(
        self,
        error_pattern: str,
        frequency: int
    ) -> Optional[Dict]:
        """生成预防规则"""
        if not self.llm:
            return None

        prompt = f"""
错误模式：{error_pattern}
出现次数：{frequency}

请生成一条预防规则，在诊断时自动检查，避免这类错误。

要求：
1. 规则必须可自动执行（基于症状特征判断）
2. 明确触发条件和检查内容

示例格式：
{{
  "trigger": "当诊断为'肾阳虚'时",
  "check": "必须确认存在'腰膝酸软'或'夜尿频多'",
  "action": "如果不存在，提示'可能是脾阳虚，请重新辨证'"
}}

请给出规则（JSON格式）：
"""

        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            rule = json.loads(content)
            return rule
        except Exception as e:
            logger.error(f"[Evolution] Failed to generate prevention rule: {e}")
            return None


class AccuracyEvolutionEngine:
    """准确率进化引擎"""

    def __init__(self, llm=None):
        self.llm = llm
        self.records: List[EvolutionRecord] = []
        self.current_metrics = AccuracyMetrics()
        self.baseline_metrics = AccuracyMetrics()

        # 进化策略
        self.evolution_strategies = {
            EvolutionStrategy.INTENT_RULES: IntentRuleEvolution(llm),
            EvolutionStrategy.TOOL_SELECTION_RULES: ToolSelectionEvolution(llm),
            EvolutionStrategy.DISCRIMINATING_RULES: DiagnosisRuleEvolution(llm),
            EvolutionStrategy.INQUIRY_OPTIMIZATION: InquiryOptimization(),
            EvolutionStrategy.ERROR_PREVENTION: ErrorPreventionEvolution(llm)
        }

    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """更新当前指标"""
        for key, value in metrics.items():
            if hasattr(self.current_metrics, key):
                setattr(self.current_metrics, key, value)

    def should_evolve(self) -> Dict[str, bool]:
        """
        判断哪些维度需要进化

        Returns:
            各维度是否需要进化的字典
        """
        evolution_needed = {}

        # 主图：意图识别准确率 < 90%
        if self.current_metrics.intent_recognition_accuracy < 0.90:
            evolution_needed[EvolutionStrategy.INTENT_RULES.value] = True

        # 子图：工具选择准确率 < 85%
        if self.current_metrics.tool_selection_accuracy < 0.85:
            evolution_needed[EvolutionStrategy.TOOL_SELECTION_RULES.value] = True

        # 子图：辨证准确率 < 80%（最重要）
        if self.current_metrics.diagnosis_accuracy < 0.80:
            evolution_needed[EvolutionStrategy.DISCRIMINATING_RULES.value] = True

        # 子图：追问效率 < 0.7
        if self.current_metrics.inquiry_efficiency < 0.7:
            evolution_needed[EvolutionStrategy.INQUIRY_OPTIMIZATION.value] = True

        # 高频错误模式
        for pattern, freq in self.current_metrics.error_patterns.items():
            if freq >= 5:
                evolution_needed[EvolutionStrategy.ERROR_PREVENTION.value] = True
                break

        return evolution_needed

    async def evolve(
        self,
        strategy: EvolutionStrategy,
        data: Dict
    ) -> EvolutionRecord:
        """
        执行进化

        Args:
            strategy: 进化策略
            data: 进化所需数据

        Returns:
            进化记录
        """
        record = EvolutionRecord(
            strategy=strategy,
            before_metrics=self.current_metrics
        )

        evolution_strategy = self.evolution_strategies.get(strategy)
        if not evolution_strategy:
            logger.warning(f"[Evolution] Unknown strategy: {strategy}")
            return record

        # 执行进化
        try:
            if strategy == EvolutionStrategy.INTENT_RULES:
                new_rules = await evolution_strategy.evolve_from_errors(
                    data.get("intent_errors", [])
                )
                record.changed_rules = new_rules
                record.change_description = f"新增 {len(new_rules)} 条意图识别规则"

            elif strategy == EvolutionStrategy.TOOL_SELECTION_RULES:
                new_rules = await evolution_strategy.evolve_from_errors(
                    data.get("tool_errors", [])
                )
                record.changed_rules = [str(r) for r in new_rules]
                record.change_description = f"新增 {len(new_rules)} 条工具选择规则"

            elif strategy == EvolutionStrategy.DISCRIMINATING_RULES:
                new_rules = await evolution_strategy.evolve_from_errors(
                    data.get("diagnosis_errors", [])
                )
                record.changed_rules = new_rules
                record.change_description = f"新增 {len(new_rules)} 条辨证鉴别规则"

            elif strategy == EvolutionStrategy.INQUIRY_OPTIMIZATION:
                optimized_paths = await evolution_strategy.evolve_from_successful_cases(
                    data.get("successful_cases", [])
                )
                record.changed_rules = [f"{k}: {v}" for k, v in optimized_paths.items()]
                record.change_description = f"优化 {len(optimized_paths)} 个证型的追问路径"

            elif strategy == EvolutionStrategy.ERROR_PREVENTION:
                prevention_rules = await evolution_strategy.evolve_from_error_patterns(
                    data.get("error_patterns", {})
                )
                record.changed_rules = [str(r) for r in prevention_rules]
                record.change_description = f"新增 {len(prevention_rules)} 条错误预防规则"

            logger.info(f"[Evolution] {strategy.value}: {record.change_description}")

        except Exception as e:
            logger.error(f"[Evolution] Failed to evolve {strategy.value}: {e}")

        self.records.append(record)
        return record

    def record_evolution_result(
        self,
        record: EvolutionRecord,
        after_metrics: AccuracyMetrics
    ) -> None:
        """记录进化结果"""
        record.after_metrics = after_metrics

        # 计算改进幅度
        if record.before_metrics:
            before_quality = record.before_metrics.calculate_overall_quality()
            after_quality = after_metrics.calculate_overall_quality()
            record.improvement = after_quality - before_quality
            record.successful = record.improvement > 0

        # 如果改进成功，更新基线
        if record.successful:
            self.baseline_metrics = after_metrics

        logger.info(
            f"[Evolution] Result: {record.strategy.value}, "
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
            "current_quality": self.current_metrics.calculate_overall_quality(),
            "baseline_quality": self.baseline_metrics.calculate_overall_quality()
        }

    def recommend_strategy(self) -> Optional[EvolutionStrategy]:
        """推荐进化策略"""
        current = self.current_metrics

        # 根据不同指标推荐策略（按优先级）
        if current.diagnosis_accuracy < 0.75:
            return EvolutionStrategy.DISCRIMINATING_RULES  # 最重要

        if current.intent_recognition_accuracy < 0.85:
            return EvolutionStrategy.INTENT_RULES

        if current.tool_selection_accuracy < 0.80:
            return EvolutionStrategy.TOOL_SELECTION_RULES

        if current.inquiry_efficiency < 0.7:
            return EvolutionStrategy.INQUIRY_OPTIMIZATION

        # 检查高频错误
        for pattern, freq in current.error_patterns.items():
            if freq >= 5:
                return EvolutionStrategy.ERROR_PREVENTION

        return None
