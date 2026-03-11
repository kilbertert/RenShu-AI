"""
Diagnosis State Manager - 问诊状态管理
管理多轮追问逻辑和舌诊图片提示判定

核心功能：
1. 问诊信息完整度评估
2. 多轮追问触发判定
3. 舌诊图片提示判定
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DiagnosisStage(str, Enum):
    """问诊阶段"""
    INITIAL = "initial"           # 初始阶段 - 收集主诉
    SYMPTOM_DETAIL = "symptom_detail"  # 症状详情
    COLD_HEAT = "cold_heat"       # 寒热问诊
    BODY_FLUID = "body_fluid"     # 汗出/口渴
    SLEEP_EMOTION = "sleep_emotion"   # 睡眠情绪
    TONGUE_PULSE = "tongue_pulse"     # 舌脉问诊
    COMPLETE = "complete"         # 信息采集完成


class CollectedSymptoms(BaseModel):
    """已采集的症状信息"""

    # === 必要信息 (权重高) ===
    chief_complaint: Optional[str] = Field(
        default=None,
        description="主诉症状"
    )
    duration: Optional[str] = Field(
        default=None,
        description="症状持续时间"
    )
    cold_heat_preference: Optional[str] = Field(
        default=None,
        description="寒热倾向: 怕冷/怕热/无明显"
    )

    # === 重要信息 (权重中) ===
    symptom_nature: Optional[str] = Field(
        default=None,
        description="症状性质: 隐痛/刺痛/胀痛等"
    )
    symptom_timing: Optional[str] = Field(
        default=None,
        description="发作时间规律: 晨起/夜间/饭后等"
    )
    aggravating_factors: list[str] = Field(
        default_factory=list,
        description="加重因素: 受寒/劳累/情绪等"
    )
    relieving_factors: list[str] = Field(
        default_factory=list,
        description="缓解因素: 休息/热敷/按压等"
    )

    # === 辅助信息 (权重低) ===
    sweating: Optional[str] = Field(
        default=None,
        description="汗出情况: 多汗/少汗/盗汗/自汗"
    )
    thirst: Optional[str] = Field(
        default=None,
        description="口渴情况: 口渴喜冷/喜热/不渴"
    )
    appetite: Optional[str] = Field(
        default=None,
        description="饮食情况: 食欲好/差/腹胀"
    )
    stool: Optional[str] = Field(
        default=None,
        description="大便情况: 正常/便秘/腹泻/溏薄"
    )
    urine: Optional[str] = Field(
        default=None,
        description="小便情况: 正常/频数/短少/色黄"
    )
    sleep: Optional[str] = Field(
        default=None,
        description="睡眠情况: 正常/失眠/多梦/易醒"
    )
    emotion: Optional[str] = Field(
        default=None,
        description="情绪状态: 正常/焦虑/抑郁/易怒"
    )

    # === 舌象信息 ===
    tongue_color: Optional[str] = Field(
        default=None,
        description="舌质颜色: 淡红/红/绛红/淡白/青紫"
    )
    tongue_coating: Optional[str] = Field(
        default=None,
        description="舌苔: 薄白/白腻/黄腻/少苔/无苔"
    )
    tongue_shape: Optional[str] = Field(
        default=None,
        description="舌形: 正常/胖大/瘦薄/齿痕/裂纹"
    )
    has_tongue_image: bool = Field(
        default=False,
        description="是否已上传舌象图片"
    )

    # === 其他 ===
    other_symptoms: list[str] = Field(
        default_factory=list,
        description="其他症状列表"
    )
    medical_history: list[str] = Field(
        default_factory=list,
        description="既往病史"
    )


class DiagnosisState(BaseModel):
    """问诊状态"""

    current_stage: DiagnosisStage = Field(
        default=DiagnosisStage.INITIAL,
        description="当前问诊阶段"
    )
    collected: CollectedSymptoms = Field(
        default_factory=CollectedSymptoms,
        description="已采集的症状信息"
    )
    follow_up_count: int = Field(
        default=0,
        description="追问轮数"
    )
    max_follow_ups: int = Field(
        default=3,
        description="最大追问轮数"
    )
    candidate_syndromes: list[str] = Field(
        default_factory=list,
        description="候选证型列表"
    )

    # === 配置阈值 ===
    MIN_COMPLETENESS_THRESHOLD: float = 0.6  # 最低完整度阈值
    TONGUE_PROMPT_ROUND: int = 2  # 第几轮开始提示舌诊

    def calculate_completeness(self) -> float:
        """
        计算信息完整度

        权重分配:
        - 必要信息: 各15% (共45%)
        - 重要信息: 各8% (共32%)
        - 辅助信息: 各3% (共23%)

        Returns:
            float: 完整度 0-1
        """
        score = 0.0

        # 必要信息 (45%)
        if self.collected.chief_complaint:
            score += 0.20
        if self.collected.duration:
            score += 0.15
        if self.collected.cold_heat_preference:
            score += 0.10

        # 重要信息 (32%)
        if self.collected.symptom_nature:
            score += 0.08
        if self.collected.symptom_timing:
            score += 0.08
        if self.collected.aggravating_factors:
            score += 0.08
        if self.collected.relieving_factors:
            score += 0.08

        # 辅助信息 (23%)
        auxiliary_fields = [
            self.collected.sweating,
            self.collected.thirst,
            self.collected.appetite,
            self.collected.stool,
            self.collected.urine,
            self.collected.sleep,
            self.collected.emotion,
        ]
        filled_auxiliary = sum(1 for f in auxiliary_fields if f is not None)
        score += (filled_auxiliary / len(auxiliary_fields)) * 0.23

        return min(score, 1.0)

    def get_missing_required_info(self) -> list[str]:
        """获取缺失的必要信息"""
        missing = []
        if not self.collected.chief_complaint:
            missing.append("主诉症状")
        if not self.collected.duration:
            missing.append("症状持续时间")
        if not self.collected.cold_heat_preference:
            missing.append("寒热倾向")
        return missing

    def should_follow_up(self) -> bool:
        """
        判断是否需要追问

        触发条件:
        1. 追问轮数未超过上限
        2. 必要信息有缺失 或 完整度低于阈值

        Returns:
            bool: 是否需要追问
        """
        # 超过最大追问轮数，停止追问
        if self.follow_up_count >= self.max_follow_ups:
            return False

        # 必要信息缺失 -> 追问
        if self.get_missing_required_info():
            return True

        # 完整度低于阈值 -> 追问
        if self.calculate_completeness() < self.MIN_COMPLETENESS_THRESHOLD:
            return True

        return False

    def should_prompt_tongue_image(self) -> bool:
        """
        判断是否应该提示用户上传舌象图片

        触发条件:
        1. 尚未上传舌象图片
        2. 满足以下任一条件:
           a. 追问轮数 >= 2 且无舌象信息
           b. 候选证型涉及内热/湿热/阴虚等需要舌象验证的证型
           c. 候选证型模糊（多个证型置信度接近）

        Returns:
            bool: 是否应该提示上传舌象
        """
        # 已有舌象图片，不再提示
        if self.collected.has_tongue_image:
            return False

        # 已有舌象文字描述，不强制要求图片
        if self.collected.tongue_color or self.collected.tongue_coating:
            return False

        # 条件a: 追问轮数达到阈值且无舌象信息
        if self.follow_up_count >= self.TONGUE_PROMPT_ROUND:
            return True

        # 条件b: 候选证型需要舌象验证
        tongue_related_syndromes = [
            "阴虚", "内热", "湿热", "痰湿", "血瘀", "气滞",
            "肝火", "心火", "胃热", "肺热", "血热"
        ]
        for syndrome in self.candidate_syndromes:
            if any(s in syndrome for s in tongue_related_syndromes):
                return True

        # 条件c: 多个候选证型（辨证模糊）
        if len(self.candidate_syndromes) >= 3:
            return True

        return False

    def get_next_question(self) -> Optional[str]:
        """
        获取下一个追问问题

        基于当前缺失的信息生成追问

        Returns:
            Optional[str]: 追问问题，若不需要追问返回None
        """
        if not self.should_follow_up():
            return None

        missing = self.get_missing_required_info()

        # 优先追问必要信息
        if "主诉症状" in missing:
            return "请详细描述一下您的主要不适症状是什么？"

        if "症状持续时间" in missing:
            return "这个症状持续多长时间了？是最近才出现还是已经有一段时间了？"

        if "寒热倾向" in missing:
            return "您平时怕冷还是怕热？有没有发热或者怕风的情况？"

        # 其次追问重要信息
        if not self.collected.symptom_nature:
            return "您能描述一下症状的具体感觉吗？比如是隐隐作痛、刺痛还是胀痛？"

        if not self.collected.aggravating_factors:
            return "什么情况下症状会加重？比如受寒、劳累、情绪波动后？"

        # 辅助信息追问
        completeness = self.calculate_completeness()
        if completeness < self.MIN_COMPLETENESS_THRESHOLD:
            if not self.collected.sleep:
                return "您最近睡眠情况怎么样？有没有失眠、多梦或者容易醒？"
            if not self.collected.appetite:
                return "您的食欲如何？吃饭后有没有腹胀或者消化不良？"
            if not self.collected.stool:
                return "大便情况怎么样？是正常、偏干还是偏稀？"

        return None

    def get_tongue_prompt_message(self) -> str:
        """获取舌诊图片提示消息"""
        return (
            "为了更准确地进行辨证分析，建议您上传一张舌象照片。\n\n"
            "📸 **拍摄要点**：\n"
            "• 自然光线下拍摄\n"
            "• 舌头自然伸出，不要过度用力\n"
            "• 拍摄前不要进食有颜色的食物\n\n"
            "如果不方便上传，您也可以描述一下舌头的颜色和舌苔情况。"
        )

    def is_ready_for_diagnosis(self) -> bool:
        """判断是否可以进行辨证分析"""
        # 必要信息必须完整
        if self.get_missing_required_info():
            return False

        # 完整度达到阈值 或 追问轮数已满
        if self.calculate_completeness() >= self.MIN_COMPLETENESS_THRESHOLD:
            return True

        if self.follow_up_count >= self.max_follow_ups:
            return True

        return False

    def advance_stage(self):
        """推进问诊阶段"""
        stage_order = [
            DiagnosisStage.INITIAL,
            DiagnosisStage.SYMPTOM_DETAIL,
            DiagnosisStage.COLD_HEAT,
            DiagnosisStage.BODY_FLUID,
            DiagnosisStage.SLEEP_EMOTION,
            DiagnosisStage.TONGUE_PULSE,
            DiagnosisStage.COMPLETE,
        ]

        current_index = stage_order.index(self.current_stage)
        if current_index < len(stage_order) - 1:
            self.current_stage = stage_order[current_index + 1]

    def update_from_entities(self, entities: dict):
        """
        从实体提取结果更新症状信息

        Args:
            entities: 提取的实体字典
        """
        if entities.get("symptoms"):
            if not self.collected.chief_complaint:
                self.collected.chief_complaint = "、".join(entities["symptoms"][:3])
            self.collected.other_symptoms.extend(entities["symptoms"])

        if entities.get("duration"):
            self.collected.duration = entities["duration"]

        if entities.get("body_parts"):
            # 可以用于定位症状部位
            pass

        if entities.get("triggers"):
            self.collected.aggravating_factors.extend(entities["triggers"])


# 舌诊关键词检测
TONGUE_KEYWORDS = [
    "舌苔", "舌头", "舌质", "舌色", "看舌", "舌诊",
    "舌尖", "舌根", "舌边", "舌面", "苔色", "苔质",
    "白苔", "黄苔", "腻苔", "剥苔", "裂纹舌", "齿痕舌"
]


def should_trigger_tongue_analysis(query: str) -> bool:
    """
    检测用户输入是否触发舌诊分析

    Args:
        query: 用户输入

    Returns:
        bool: 是否触发舌诊
    """
    query_lower = query.lower()
    return any(kw in query_lower for kw in TONGUE_KEYWORDS)
