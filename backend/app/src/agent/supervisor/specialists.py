"""
专门Agent实现示例

展示如何实现各个专门Agent，这些Agent可以被Supervisor调用。
"""

import logging
from typing import Dict, Any

from .tcm_supervisor import AgentRole, AgentTask, AgentResponse, ResponseType
from ..memory.tcm_memory import TCMMemory


logger = logging.getLogger(__name__)


class BaseSpecialist:
    """专门Agent基类"""

    def __init__(self, role: AgentRole, memory: TCMMemory = None):
        self.role = role
        self.memory = memory

    async def execute(
        self,
        task: AgentTask,
        context: Dict[str, Any]
    ) -> AgentResponse:
        """
        执行任务（子类实现）

        Args:
            task: Agent任务
            context: 隔离的上下文

        Returns:
            Agent响应
        """
        raise NotImplementedError

    def _create_response(
        self,
        task: AgentTask,
        content: str,
        response_type: ResponseType = ResponseType.SYNTHESIZED,
        confidence: float = 0.8,
        **metadata
    ) -> AgentResponse:
        """创建响应的便捷方法"""
        return AgentResponse(
            agent_role=self.role,
            task_id=task.task_id,
            response_type=response_type,
            content=content,
            confidence=confidence,
            data=metadata
        )


class DiagnosisSpecialist(BaseSpecialist):
    """
    诊断专门Agent

    负责症状分析和辨证论治
    """

    def __init__(self, memory: TCMMemory = None):
        super().__init__(AgentRole.DIAGNOSIS, memory)

    async def execute(
        self,
        task: AgentTask,
        context: Dict[str, Any]
    ) -> AgentResponse:
        """执行诊断任务"""
        logger.info(f"[DiagnosisSpecialist] 执行诊断任务: {task.description}")

        # 从上下文获取症状
        symptoms = task.input_data.get("symptoms", {})
        query = task.input_data.get("query", "")

        # 这里可以调用实际的诊断逻辑
        # 示例：简单的规则匹配
        syndrome = self._analyze_syndrome(symptoms, query)

        content = f"根据您提供的症状分析，可能的证型为：{syndrome}。\n\n"
        content += self._generate_advice(syndrome)

        # 保存诊断记忆
        if self.memory:
            user_id = context.get("user_id", "unknown")
            await self.memory.add_diagnosis_memory(
                user_id=user_id,
                syndrome=syndrome,
                symptoms=symptoms,
                confidence=0.8
            )

        # 对于简单的诊断，可以直接返回给用户
        if self._is_simple_case(symptoms):
            return self._create_response(
                task,
                content,
                response_type=ResponseType.DIRECT_PASS,
                syndrome=syndrome,
                symptoms=symptoms
            )

        # 复杂案例需要Supervisor合成
        return self._create_response(
            task,
            content,
            response_type=ResponseType.SYNTHESIZED,
            syndrome=syndrome,
            symptoms=symptoms
        )

    def _analyze_syndrome(self, symptoms: Dict[str, Any], query: str) -> str:
        """简单的证型分析（示例）"""
        # 实际实现应该使用RAG + LLM
        if symptoms.get("怕冷"):
            if symptoms.get("乏力"):
                return "阳虚证"
            return "风寒证"
        elif symptoms.get("怕热"):
            return "热证"
        elif symptoms.get("乏力"):
            return "气虚证"
        return "待进一步辨证"

    def _generate_advice(self, syndrome: str) -> str:
        """生成调理建议"""
        advices = {
            "阳虚证": "建议注意保暖，避免生冷食物，可适当食用温补类食物。",
            "风寒证": "建议发汗解表，注意保暖，多饮热水。",
            "热证": "建议清热解毒，饮食清淡，避免辛辣。",
            "气虚证": "建议补气养血，适当运动，避免过度劳累。"
        }
        return advices.get(syndrome, "建议咨询专业医师进行详细诊断。")

    def _is_simple_case(self, symptoms: Dict[str, Any]) -> bool:
        """判断是否为简单案例"""
        return len(symptoms) <= 2


class PrescriptionSpecialist(BaseSpecialist):
    """
    方剂专门Agent

    负责方剂推荐和解释
    """

    def __init__(self, memory: TCMMemory = None):
        super().__init__(AgentRole.PRESCRIPTION, memory)

    async def execute(
        self,
        task: AgentTask,
        context: Dict[str, Any]
    ) -> AgentResponse:
        """执行方剂任务"""
        logger.info(f"[PrescriptionSpecialist] 执行方剂任务: {task.description}")

        query = task.input_data.get("query", "")

        # 分析查询意图
        prescription_name = self._extract_prescription(query)
        syndrome = context.get("syndrome", "")

        if prescription_name:
            # 解释方剂
            content = self._describe_prescription(prescription_name)
        else:
            # 推荐方剂
            prescription_name = self._recommend_prescription(syndrome or query)
            content = f"根据您的情况，推荐方剂：{prescription_name}\n\n"
            content += self._describe_prescription(prescription_name)

        return self._create_response(
            task,
            content,
            response_type=ResponseType.DIRECT_PASS,
            prescription=prescription_name
        )

    def _extract_prescription(self, query: str) -> str:
        """提取方剂名称"""
        # 简单实现：检查常见方剂名
        common_prescriptions = [
            "桂枝汤", "麻黄汤", "银翘散", "桑菊饮",
            "四君子汤", "补中益气汤", "归脾汤",
            "六味地黄丸", "肾气丸"
        ]
        for prescription in common_prescriptions:
            if prescription in query:
                return prescription
        return ""

    def _recommend_prescription(self, syndrome_or_query: str) -> str:
        """推荐方剂"""
        # 简单映射
        recommendations = {
            "阳虚证": "金匮肾气丸",
            "风寒证": "桂枝汤",
            "热证": "银翘散",
            "气虚证": "四君子汤"
        }
        return recommendations.get(syndrome_or_query, "请咨询医师")

    def _describe_prescription(self, name: str) -> str:
        """描述方剂"""
        descriptions = {
            "桂枝汤": "桂枝汤是解表剂，具有解肌发表、调和营卫之功效。主治外感风寒表虚证。",
            "麻黄汤": "麻黄汤是发汗解表剂，具有发汗解表、宣肺平喘之功效。主治外感风寒实证。",
            "银翘散": "银翘散是清热解毒剂，具有辛凉透表、清热解毒之功效。主治风热表证。",
            "四君子汤": "四君子汤是补气剂，具有益气健脾之功效。主治脾胃气虚证。",
            "补中益气汤": "补中益气汤是补气剂，具有补中益气、升阳举陷之功效。主治脾胃气虚、气虚下陷证。",
            "六味地黄丸": "六味地黄丸是滋补剂，具有滋补肝肾之功效。主治肝肾阴虚证。",
            "金匮肾气丸": "金匮肾气丸是温补剂，具有温补肾阳之功效。主治肾阳虚证。"
        }
        return descriptions.get(name, f"{name}：具体方义请咨询专业医师。")


class WellnessSpecialist(BaseSpecialist):
    """
    养生专门Agent

    负责养生建议和健康指导
    """

    def __init__(self, memory: TCMMemory = None):
        super().__init__(AgentRole.WELLNESS, memory)

    async def execute(
        self,
        task: AgentTask,
        context: Dict[str, Any]
    ) -> AgentResponse:
        """执行养生任务"""
        logger.info(f"[WellnessSpecialist] 执行养生任务: {task.description}")

        query = task.input_data.get("query", "").lower()
        user_profile = context.get("user_profile", {})

        # 根据用户体质生成建议
        constitution = user_profile.get("constitution", "平和质")

        content = self._generate_wellness_advice(query, constitution)

        # 保存养生记忆
        if self.memory:
            user_id = context.get("user_id", "unknown")
            await self.memory.add_wellness_memory(
                user_id=user_id,
                category="一般养生",
                advice=content,
                confidence=0.7
            )

        return self._create_response(
            task,
            content,
            response_type=ResponseType.DIRECT_PASS
        )

    def _generate_wellness_advice(self, query: str, constitution: str) -> str:
        """生成养生建议"""
        # 季节性建议
        seasonal_advices = {
            "春季": "春季养生重点在于养肝。建议保持心情舒畅，早睡早起，多食用绿色蔬菜。",
            "夏季": "夏季养生重点在于养心。建议保持心态平和，适当午休，饮食清淡。",
            "秋季": "秋季养生重点在于养肺。建议多饮水，食用润燥食物，适当运动。",
            "冬季": "冬季养生重点在于养肾。建议注意保暖，早睡晚起，适当进补。"
        }

        # 体质建议
        constitution_advices = {
            "气虚质": "建议补气养血，适当运动，避免过度劳累。",
            "阳虚质": "建议注意保暖，避免生冷食物，可食用温补类食物。",
            "阴虚质": "建议清热润燥，避免辛辣食物，保持充足睡眠。",
            "痰湿质": "建议健脾祛湿，饮食清淡，适当运动。",
            "湿热质": "建议清热利湿，饮食清淡，避免油腻食物。",
            "血瘀质": "建议活血化瘀，适当运动，保持心情舒畅。",
            "气郁质": "建议疏肝解郁，保持心情愉快，多参加户外活动。",
            "特禀质": "建议避免过敏原，根据具体情况调整饮食和生活习惯。"
        }

        advice = "根据您的体质，给出以下建议：\n\n"
        advice += constitution_advices.get(constitution, "建议保持良好的生活习惯。")

        return advice


class HerbSpecialist(BaseSpecialist):
    """
    药材专门Agent

    负责中药材相关咨询
    """

    def __init__(self, memory: TCMMemory = None):
        super().__init__(AgentRole.HERB, memory)

    async def execute(
        self,
        task: AgentTask,
        context: Dict[str, Any]
    ) -> AgentResponse:
        """执行药材任务"""
        logger.info(f"[HerbSpecialist] 执行药材任务: {task.description}")

        query = task.input_data.get("query", "")

        # 提取药材名
        herb_name = self._extract_herb_name(query)

        if herb_name:
            content = self._describe_herb(herb_name)
        else:
            content = "请问您想了解哪味中药？我可以为您介绍药材的功效、用法和禁忌。"

        return self._create_response(
            task,
            content,
            response_type=ResponseType.DIRECT_PASS
        )

    def _extract_herb_name(self, query: str) -> str:
        """提取药材名称"""
        # 常见药材
        common_herbs = [
            "人参", "黄芪", "白术", "茯苓", "甘草",
            "当归", "熟地", "白芍", "川芎",
            "桂枝", "麻黄", "柴胡", "黄芩",
            "半夏", "陈皮", "枳实", "厚朴"
        ]
        for herb in common_herbs:
            if herb in query:
                return herb
        return ""

    def _describe_herb(self, name: str) -> str:
        """描述药材"""
        descriptions = {
            "人参": "人参是补气药，具有大补元气、补脾益肺、生津止渴、安神益智之功效。主治气虚欲脱、脉微欲绝、脾气不足、肺气亏虚等证。",
            "黄芪": "黄芪是补气药，具有健脾补中、升阳举陷、益卫固表、利尿、托毒生肌之功效。主治脾气虚证、肺气虚证、气虚自汗证等。",
            "当归": "当归是补血药，具有补血调经、活血止痛、润肠通便之功效。主治血虚萎黄、月经不调、经闭痛经、虚寒腹痛等证。",
            "柴胡": "柴胡是解表药，具有解表退热、疏肝解郁、升举阳气之功效。主治表证发热、肝郁气滞、气虚下陷等证。"
        }
        return descriptions.get(name, f"{name}：具体药性请咨询专业医师或查阅中药学典籍。")


class GeneralSpecialist(BaseSpecialist):
    """
    通用咨询Agent

    处理不属于其他专门Agent范围的查询
    """

    def __init__(self, memory: TCMMemory = None):
        super().__init__(AgentRole.GENERAL, memory)

    async def execute(
        self,
        task: AgentTask,
        context: Dict[str, Any]
    ) -> AgentResponse:
        """执行通用咨询任务"""
        logger.info(f"[GeneralSpecialist] 执行通用任务: {task.description}")

        query = task.input_data.get("query", "")

        # 检查是否需要移交给其他Agent
        if "诊断" in query or "症状" in query:
            return AgentResponse(
                agent_role=self.role,
                task_id=task.task_id,
                response_type=ResponseType.HANDOFF,
                content="这个问题更适合由诊断专家来回答。",
                handoff_target=AgentRole.DIAGNOSIS,
                data={"handoff_reason": "涉及诊断，移交给诊断专家"}
            )

        # 通用回答
        content = f"您好！关于您的问题：{query}\n\n"
        content += "我是中医咨询助手，可以帮您解答关于中医诊断、方剂、养生、药材等方面的问题。"
        content += "请问您具体想了解什么？"

        return self._create_response(
            task,
            content,
            response_type=ResponseType.DIRECT_PASS
        )


def get_all_specialists(memory: TCMMemory = None) -> Dict[AgentRole, callable]:
    """
    获取所有专门Agent的执行器

    Returns:
        角色到执行函数的映射
    """
    diagnosis_specialist = DiagnosisSpecialist(memory)
    prescription_specialist = PrescriptionSpecialist(memory)
    wellness_specialist = WellnessSpecialist(memory)
    herb_specialist = HerbSpecialist(memory)
    general_specialist = GeneralSpecialist(memory)

    return {
        AgentRole.DIAGNOSIS: diagnosis_specialist.execute,
        AgentRole.PRESCRIPTION: prescription_specialist.execute,
        AgentRole.WELLNESS: wellness_specialist.execute,
        AgentRole.HERB: herb_specialist.execute,
        AgentRole.GENERAL: general_specialist.execute,
    }
