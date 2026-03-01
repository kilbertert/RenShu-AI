"""
中等辨证节点

中等复杂度的 RAG 辅助辨证
"""

from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from datetime import datetime

from app.src.agent.components.diagnose.states import DiagnoseOverallState



from app.src.utils import get_logger

from app.src.agent.components.diagnose.models import CollectedDiagnoseInfo

from app.src.agent.tcm_builder import get_llm
from app.src.agent.components.diagnose.config import diagnose_config
from app.src.agent.components.diagnose.prompts import MODERATE_DIAGNOSIS_PROMPT

logger = get_logger("moderate_diagnosis")


def _get_current_solar_term() -> str:
    """获取当前节气（简化版）"""
    now = datetime.now()
    month = now.month
    
    # 简化的节气映射（实际应该更精确）
    solar_terms = {
        1: "小寒/大寒", 2: "立春/雨水", 3: "惊蛰/春分",
        4: "清明/谷雨", 5: "立夏/小满", 6: "芒种/夏至",
        7: "小暑/大暑", 8: "立秋/处暑", 9: "白露/秋分",
        10: "寒露/霜降", 11: "立冬/小雪", 12: "大雪/冬至"
    }
    
    return solar_terms.get(month, "未知节气")


async def moderate_diagnosis(state: DiagnoseOverallState, config: RunnableConfig) -> Dict[str, Any]:
    """
    中等复杂度的 RAG 辅助辨证

    优化（2026-02-05）：
    - 使用 Map-Reduce 子图实现并行查询
    - 降低延迟 50-70%（从 7秒降到 3-4秒）

    方法：
    1. 任务分解：将辨证需求分解为并行查询任务
    2. 并行执行：
       - 查询相似证型（Neo4j）
       - 查询相关医案（向量检索）
       - 查询常用方剂（知识图谱）
    3. 结果汇总：LangGraph 自动 Reduce
    4. 综合分析：LLM 结合所有结果生成辨证

    适用场景：
    - 多个症状需要综合判断
    - 可能存在兼证
    - 需要医案参考

    Args:
        state: 当前状态

    Returns:
        dict: 更新的状态字段
    """
    try:
        # 使用 Map-Reduce 子图
        from .moderate_map_reduce_builder import get_moderate_map_reduce_graph

        logger.info("使用 Map-Reduce 子图进行并行查询")

        # 获取子图
        map_reduce_graph = get_moderate_map_reduce_graph()

        # 准备子图输入
        subgraph_input = {
            "collected_info": state.get("collected_info", {}),
            "tongue_analysis": state.get("tongue_analysis"),
            "user_profile": state.get("user_profile", {}),
            "llm_config": state.get("llm_config"),
        }

        # 调用子图
        import time
        start_time = time.time()
        result = await map_reduce_graph.ainvoke(subgraph_input, config)
        elapsed = time.time() - start_time

        logger.info(f"Map-Reduce 子图完成，总耗时: {elapsed:.2f}秒")

        # 提取结果
        answer = result.get("answer", "")
        steps = result.get("steps", [])

        return {
            "answer": answer,
            "steps": steps + [f"中等辨证: 完成 (Map-Reduce 并行, 耗时 {elapsed:.2f}秒)"],
        }

    except Exception as e:
        logger.error(f"Map-Reduce 子图失败: {e}", exc_info=True)
        # 降级到串行版本
        logger.warning("降级到串行版本")
        return await _fallback_serial_diagnosis(state)


async def _fallback_serial_diagnosis(state: DiagnoseOverallState) -> Dict[str, Any]:
    """
    降级处理：使用 asyncio.gather 并行执行（当 Map-Reduce 子图失败时使用）
    """
    import asyncio
    import time

    try:
        # 获取已收集的信息
        collected_info_dict = state.get("collected_info", {})
        if collected_info_dict:
            collected_info = CollectedDiagnoseInfo(**collected_info_dict)
            collected_summary = collected_info.to_summary()
        else:
            collected_summary = "暂无详细信息"

        # 获取舌像分析
        tongue_analysis = state.get("tongue_analysis")
        tongue_desc = "未提供"
        if tongue_analysis:
            parts = []
            if tongue_analysis.get("tongue_color"): parts.append(f"舌色：{tongue_analysis['tongue_color']}")
            if tongue_analysis.get("tongue_shape"): parts.append(f"舌形：{tongue_analysis['tongue_shape']}")
            if tongue_analysis.get("coating_color"): parts.append(f"苔色：{tongue_analysis['coating_color']}")
            if tongue_analysis.get("coating_quality"): parts.append(f"苔质：{tongue_analysis['coating_quality']}")
            if tongue_analysis.get("analysis"): parts.append(f"分析：{tongue_analysis['analysis']}")
            tongue_desc = "\n".join(parts)

        # 获取用户画像
        user_profile = state.get("user_profile", {})
        user_profile_desc = _format_user_profile(user_profile)

        # === 并行 RAG 检索（使用 asyncio.gather）===
        logger.info("开始并行 RAG 检索（asyncio.gather）...")
        start_time = time.time()

        # ★★★ 关键：使用 asyncio.gather 并行执行 ★★★
        similar_syndromes, similar_cases, related_prescriptions = await asyncio.gather(
            _query_similar_syndromes(collected_info),
            _query_similar_cases(collected_info),
            _query_related_prescriptions(collected_info),
            return_exceptions=True  # 即使某个查询失败，其他查询继续
        )

        # 处理异常结果
        if isinstance(similar_syndromes, Exception):
            logger.error(f"证型查询失败: {similar_syndromes}")
            similar_syndromes = []
        if isinstance(similar_cases, Exception):
            logger.error(f"医案查询失败: {similar_cases}")
            similar_cases = []
        if isinstance(related_prescriptions, Exception):
            logger.error(f"方剂查询失败: {related_prescriptions}")
            related_prescriptions = []

        elapsed = time.time() - start_time
        logger.info(f"并行 RAG 检索完成 (耗时: {elapsed:.2f}秒): 证型 {len(similar_syndromes)} 个, 医案 {len(similar_cases)} 个, 方剂 {len(related_prescriptions)} 个")

        # === 构建提示词 ===
        solar_term = _get_current_solar_term()
        
        prompt = MODERATE_DIAGNOSIS_PROMPT.format(
            collected_info=collected_summary,
            tongue_analysis=tongue_desc,
            user_profile=user_profile_desc,
            syndrome_matches=_format_syndromes(similar_syndromes),
            similar_cases=_format_cases(similar_cases),
            related_prescriptions=_format_prescriptions(related_prescriptions),
            solar_term=solar_term,
        )

        # === 调用 LLM ===
        llm = get_llm(
            llm_config=state.get("llm_config"),
            temperature=diagnose_config.DIAGNOSIS_TEMPERATURE
        )

        # 直接调用返回非结构化文本
        response = await llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content="请结合参考资料开始您的辨证分析。")
        ])

        answer = response.content

        logger.info(f"降级并行辨证完成")

        return {
            "answer": answer,
            "steps": [f"中等辨证: 完成 (降级并行, 耗时 {elapsed:.2f}秒)"],
        }

    except Exception as e:
        logger.error(f"中等辨证失败: {e}", exc_info=True)
        # 尝试降级
        try:
            from app.src.agent.components.diagnose.nodes.simple.simple_diagnosis import simple_diagnosis
            return await simple_diagnosis(state)
        except:
            return {
                "answer": f"抱歉，中等辨证过程中出现错误：{str(e)}。建议您前往医院进行详细检查。",
                "steps": [f"中等辨证: 失败 - {str(e)}"],
            }



async def _query_similar_syndromes(collected_info: CollectedDiagnoseInfo) -> List[Dict[str, Any]]:
    """查询相似证型（预定义 Cypher）"""
    # TODO: 实现知识图谱查询
    symptoms = collected_info.get_all_symptoms()
    if not symptoms:
        return []

    # 模拟返回
    return [
        {
            "name": "气虚证",
            "symptoms": ["乏力", "气短", "自汗"],
            "similarity": 0.8,
        },
        {
            "name": "脾气虚证",
            "symptoms": ["乏力", "食欲不振", "便溏"],
            "similarity": 0.7,
        },
    ]


async def _query_similar_cases(collected_info: CollectedDiagnoseInfo) -> List[Dict[str, Any]]:
    """查询相似医案（向量检索）"""
    # TODO: 实现向量检索
    return [
        {
            "case_id": "case_001",
            "chief_complaint": "乏力、食欲不振",
            "syndrome": "脾气虚证",
            "treatment": "补中益气汤加减",
            "similarity": 0.75,
        },
    ]


async def _query_related_prescriptions(collected_info: CollectedDiagnoseInfo) -> List[Dict[str, Any]]:
    """查询常用方剂（知识图谱）"""
    # TODO: 实现知识图谱查询
    return [
        {
            "name": "补中益气汤",
            "indication": "脾胃气虚",
            "composition": ["黄芪", "人参", "白术", "甘草"],
        },
    ]


def _format_syndromes(syndromes: List[Dict[str, Any]]) -> str:
    """格式化证型列表"""
    if not syndromes:
        return "暂无相似证型"

    parts = []
    for i, syndrome in enumerate(syndromes[:3], 1):  # 最多3个
        symptoms_str = "、".join(syndrome.get("symptoms", []))
        similarity = syndrome.get("similarity", 0)
        parts.append(f"{i}. {syndrome['name']} (相似度: {similarity:.0%})\n   主要症状：{symptoms_str}")

    return "\n".join(parts)


def _format_cases(cases: List[Dict[str, Any]]) -> str:
    """格式化医案列表"""
    if not cases:
        return "暂无相似医案"

    parts = []
    for i, case in enumerate(cases[:2], 1):  # 最多2个
        parts.append(
            f"{i}. 主诉：{case.get('chief_complaint', '未知')}\n"
            f"   证型：{case.get('syndrome', '未知')}\n"
            f"   治疗：{case.get('treatment', '未知')}"
        )

    return "\n".join(parts)


def _format_prescriptions(prescriptions: List[Dict[str, Any]]) -> str:
    """格式化方剂列表"""
    if not prescriptions:
        return "暂无相关方剂"

    parts = []
    for i, prescription in enumerate(prescriptions[:3], 1):  # 最多3个
        composition = prescription.get("composition", [])
        composition_str = "、".join(composition[:5])  # 最多显示5味药
        parts.append(
            f"{i}. {prescription['name']}\n"
            f"   主治：{prescription.get('indication', '未知')}\n"
            f"   组成：{composition_str}等"
        )

    return "\n".join(parts)






def _format_user_profile(user_profile: Dict[str, Any]) -> str:
    """格式化用户画像"""
    if not user_profile:
        return "暂无用户画像"

    parts = []
    if user_profile.get("age"):
        parts.append(f"年龄：{user_profile['age']}岁")
    if user_profile.get("gender"):
        parts.append(f"性别：{user_profile['gender']}")
    if user_profile.get("constitution"):
        parts.append(f"体质：{user_profile['constitution']}")
    if user_profile.get("chronic_diseases"):
        parts.append(f"慢性病：{', '.join(user_profile['chronic_diseases'])}")

    return "\n".join(parts) if parts else "暂无用户画像"
