# """
# [DEPRECATED] 复杂诊断工作流（旧版本）
#
# 警告：此文件已废弃！
#
# 新架构使用 DeepAgents 框架实现，请参考：
# - deep_search_agent.py: 主 Agent（使用 create_deep_agent）
# - subagents/: 专家子 Agent（使用 SubAgentMiddleware 自动并行调度）
# - tools/: 数据查询工具
#
# 此文件保留仅用于向后兼容和参考。
# 新功能请使用 complex_diagnosis.py 中的 complex_diagnosis() 函数。
# """
#
# from typing import Dict, Any, Optional, List
# from langchain_core.language_models import BaseChatModel
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.state import CompiledStateGraph
# import asyncio
# import json
#
# from app.src.utils import get_logger, get_llm
# from app.src.agent.components.diagnose.config import diagnose_config
#
# from .complex_state import ComplexDiagnosisState
# from .subagents import (
#     create_differential_diagnosis_subagent,
#     create_treatment_principle_subagent,
#     create_prescription_subagent,
#     create_prognosis_subagent,
#     create_verification_subagent,
# )
#
# logger = get_logger("complex_diagnosis_workflow")
#
#
# # ============================================================
# # 节点函数
# # ============================================================
#
# async def initial_analysis_node(state: ComplexDiagnosisState) -> Dict[str, Any]:
#     """
#     初步分析节点
#
#     进行症状分析、八纲辨证、脏腑辨证、病因病机分析
#     """
#     logger.info("开始初步分析")
#
#     collected_info = state["collected_info"]
#     llm = get_llm(temperature=diagnose_config.DIAGNOSIS_TEMPERATURE)
#
#     # 构建初步分析提示词
#     prompt = f"""
# 作为中医诊断专家，请对以下患者信息进行初步分析：
#
# {json.dumps(collected_info, ensure_ascii=False, indent=2)}
#
# 请提供：
# 1. 症状分析：主症、兼症、症状关系
# 2. 八纲辨证：表里、寒热、虚实、阴阳
# 3. 脏腑辨证：涉及的脏腑系统
# 4. 病因病机：病因、病机、病位、病性
#
# 输出JSON格式：
# {{
#     "symptom_analysis": "症状分析",
#     "ba_gang_analysis": "八纲辨证",
#     "organ_analysis": "脏腑辨证",
#     "etiology_analysis": "病因病机",
#     "preliminary_diagnosis": "初步诊断"
# }}
# """
#
#     try:
#         response = await llm.ainvoke(prompt)
#         result = json.loads(response.content)
#
#         return {
#             "symptom_analysis": result.get("symptom_analysis"),
#             "ba_gang_analysis": result.get("ba_gang_analysis"),
#             "organ_analysis": result.get("organ_analysis"),
#             "etiology_analysis": result.get("etiology_analysis"),
#             "preliminary_diagnosis": result.get("preliminary_diagnosis"),
#             "steps": ["初步分析完成"]
#         }
#     except Exception as e:
#         logger.error(f"初步分析失败: {e}", exc_info=True)
#         return {
#             "error": f"初步分析失败: {str(e)}",
#             "steps": ["初步分析失败"]
#         }
#
#
# async def parallel_expert_analysis_node(state: ComplexDiagnosisState) -> Dict[str, Any]:
#     """
#     并行专家分析节点
#
#     同时调用多个专家 SubAgent 进行分析
#     """
#     logger.info("开始并行专家分析")
#
#     llm = get_llm(temperature=diagnose_config.DIAGNOSIS_TEMPERATURE)
#
#     # 创建专家 SubAgents
#     differential_agent = create_differential_diagnosis_subagent(llm)
#     treatment_agent = create_treatment_principle_subagent(llm)
#     prescription_agent = create_prescription_subagent(llm)
#     prognosis_agent = create_prognosis_subagent(llm)
#
#     # 准备输入
#     base_input = {
#         "preliminary_diagnosis": state.get("preliminary_diagnosis"),
#         "symptom_analysis": state.get("symptom_analysis"),
#         "ba_gang_analysis": state.get("ba_gang_analysis"),
#         "organ_analysis": state.get("organ_analysis"),
#         "etiology_analysis": state.get("etiology_analysis"),
#         "collected_info": state.get("collected_info"),
#     }
#
#     # 并行调用专家
#     try:
#         results = await asyncio.gather(
#             _invoke_differential_expert(differential_agent, base_input),
#             _invoke_treatment_expert(treatment_agent, base_input),
#             _invoke_prescription_expert(prescription_agent, base_input),
#             _invoke_prognosis_expert(prognosis_agent, base_input),
#             return_exceptions=True
#         )
#
#         differential_result, treatment_result, prescription_result, prognosis_result = results
#
#         return {
#             "differential_diagnosis_result": differential_result if not isinstance(differential_result, Exception) else None,
#             "treatment_principle_result": treatment_result if not isinstance(treatment_result, Exception) else None,
#             "prescription_result": prescription_result if not isinstance(prescription_result, Exception) else None,
#             "prognosis_result": prognosis_result if not isinstance(prognosis_result, Exception) else None,
#             "steps": ["并行专家分析完成"]
#         }
#     except Exception as e:
#         logger.error(f"并行专家分析失败: {e}", exc_info=True)
#         return {
#             "error": f"并行专家分析失败: {str(e)}",
#             "steps": ["并行专家分析失败"]
#         }
#
#
# async def verification_node(state: ComplexDiagnosisState) -> Dict[str, Any]:
#     """
#     验证节点
#
#     质疑验证专家对所有分析结果进行验证
#     """
#     logger.info("开始验证分析")
#
#     llm = get_llm(temperature=diagnose_config.DIAGNOSIS_TEMPERATURE)
#     verification_agent = create_verification_subagent(llm)
#
#     # 准备验证输入
#     verification_input = {
#         "collected_info": state.get("collected_info"),
#         "preliminary_diagnosis": state.get("preliminary_diagnosis"),
#         "expert_analyses": {
#             "differential_diagnosis": state.get("differential_diagnosis_result"),
#             "treatment_principle": state.get("treatment_principle_result"),
#             "prescription": state.get("prescription_result"),
#             "prognosis": state.get("prognosis_result"),
#         }
#     }
#
#     try:
#         verification_result = await _invoke_verification_expert(verification_agent, verification_input)
#
#         # 判断是否需要迭代
#         should_continue = False
#         current_iteration = state.get("iteration_count", 0)
#         max_iterations = state.get("max_iterations", 3)
#
#         if verification_result and current_iteration < max_iterations:
#             validity = verification_result.get("verification_result", {}).get("overall_validity")
#             if validity in ["需要补充", "需要重新评估"]:
#                 should_continue = True
#
#         return {
#             "verification_result": verification_result,
#             "should_continue_iteration": should_continue,
#             "iteration_count": current_iteration + 1,
#             "steps": ["验证分析完成"]
#         }
#     except Exception as e:
#         logger.error(f"验证分析失败: {e}", exc_info=True)
#         return {
#             "error": f"验证分析失败: {str(e)}",
#             "should_continue_iteration": False,
#             "steps": ["验证分析失败"]
#         }
#
#
# async def synthesis_node(state: ComplexDiagnosisState) -> Dict[str, Any]:
#     """
#     综合节点
#
#     综合所有专家意见，生成最终诊断
#     """
#     logger.info("开始综合分析")
#
#     llm = get_llm(temperature=diagnose_config.DIAGNOSIS_TEMPERATURE)
#
#     # 构建综合提示词
#     prompt = f"""
# 作为中医诊断总结专家，请综合以下所有专家的分析结果，给出最终诊断：
#
# ## 初步分析
# - 症状分析：{state.get('symptom_analysis')}
# - 八纲辨证：{state.get('ba_gang_analysis')}
# - 脏腑辨证：{state.get('organ_analysis')}
# - 病因病机：{state.get('etiology_analysis')}
#
# ## 专家分析
# - 鉴别诊断：{json.dumps(state.get('differential_diagnosis_result'), ensure_ascii=False)}
# - 治则治法：{json.dumps(state.get('treatment_principle_result'), ensure_ascii=False)}
# - 方药推荐：{json.dumps(state.get('prescription_result'), ensure_ascii=False)}
# - 预后评估：{json.dumps(state.get('prognosis_result'), ensure_ascii=False)}
#
# ## 验证结果
# {json.dumps(state.get('verification_result'), ensure_ascii=False)}
#
# 请给出最终诊断结果，包括：
# 1. 证型（主证+兼证）
# 2. 病因病机
# 3. 治则治法
# 4. 方药建议
# 5. 预后评估
# 6. 置信度（0-1）
#
# 输出JSON格式。
# """
#
#     try:
#         response = await llm.ainvoke(prompt)
#         final_diagnosis = json.loads(response.content)
#
#         # 提取置信度
#         confidence = final_diagnosis.get("confidence", 0.8)
#         if state.get("verification_result"):
#             adjusted_confidence = state["verification_result"].get("verification_result", {}).get("confidence_adjustment", {}).get("adjusted_confidence")
#             if adjusted_confidence:
#                 confidence = adjusted_confidence
#
#         return {
#             "final_diagnosis": final_diagnosis,
#             "confidence": confidence,
#             "steps": ["综合分析完成"]
#         }
#     except Exception as e:
#         logger.error(f"综合分析失败: {e}", exc_info=True)
#         return {
#             "error": f"综合分析失败: {str(e)}",
#             "steps": ["综合分析失败"]
#         }
#
#
# # ============================================================
# # 辅助函数
# # ============================================================
#
# async def _invoke_differential_expert(agent: CompiledStateGraph, input_data: Dict) -> Dict:
#     """调用鉴别诊断专家"""
#     prompt = f"""
# 请进行鉴别诊断分析：
#
# 初步诊断：{input_data.get('preliminary_diagnosis')}
# 症状分析：{input_data.get('symptom_analysis')}
# 八纲辨证：{input_data.get('ba_gang_analysis')}
# 脏腑辨证：{input_data.get('organ_analysis')}
#
# 请输出鉴别诊断结果（JSON格式）。
# """
#     result = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
#     return _extract_json_from_response(result)
#
#
# async def _invoke_treatment_expert(agent: CompiledStateGraph, input_data: Dict) -> Dict:
#     """调用治则治法专家"""
#     prompt = f"""
# 请制定治则治法：
#
# 辨证结果：{input_data.get('preliminary_diagnosis')}
# 八纲辨证：{input_data.get('ba_gang_analysis')}
# 脏腑辨证：{input_data.get('organ_analysis')}
# 病因病机：{input_data.get('etiology_analysis')}
#
# 请输出治则治法结果（JSON格式）。
# """
#     result = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
#     return _extract_json_from_response(result)
#
#
# async def _invoke_prescription_expert(agent: CompiledStateGraph, input_data: Dict) -> Dict:
#     """调用方药推荐专家"""
#     prompt = f"""
# 请推荐方药：
#
# 辨证结果：{input_data.get('preliminary_diagnosis')}
# 患者信息：{json.dumps(input_data.get('collected_info'), ensure_ascii=False)}
#
# 请输出方药推荐结果（JSON格式）。
# """
#     result = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
#     return _extract_json_from_response(result)
#
#
# async def _invoke_prognosis_expert(agent: CompiledStateGraph, input_data: Dict) -> Dict:
#     """调用预后评估专家"""
#     prompt = f"""
# 请评估预后：
#
# 辨证结果：{input_data.get('preliminary_diagnosis')}
# 病因病机：{input_data.get('etiology_analysis')}
# 患者信息：{json.dumps(input_data.get('collected_info'), ensure_ascii=False)}
#
# 请输出预后评估结果（JSON格式）。
# """
#     result = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
#     return _extract_json_from_response(result)
#
#
# async def _invoke_verification_expert(agent: CompiledStateGraph, input_data: Dict) -> Dict:
#     """调用质疑验证专家"""
#     prompt = f"""
# 请验证诊断结果：
#
# 原始症状信息：{json.dumps(input_data.get('collected_info'), ensure_ascii=False)}
# 各专家分析：{json.dumps(input_data.get('expert_analyses'), ensure_ascii=False)}
#
# 请输出验证结果（JSON格式）。
# """
#     result = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
#     return _extract_json_from_response(result)
#
#
# def _extract_json_from_response(result: Dict) -> Dict:
#     """从 agent 响应中提取 JSON"""
#     try:
#         if "messages" in result and len(result["messages"]) > 0:
#             content = result["messages"][-1].content
#             # 尝试解析 JSON
#             if isinstance(content, str):
#                 # 移除可能的 markdown 代码块标记
#                 content = content.strip()
#                 if content.startswith("```json"):
#                     content = content[7:]
#                 if content.startswith("```"):
#                     content = content[3:]
#                 if content.endswith("```"):
#                     content = content[:-3]
#                 content = content.strip()
#                 return json.loads(content)
#             return content
#         return {}
#     except Exception as e:
#         logger.warning(f"解析 JSON 失败: {e}")
#         return {"raw_content": str(result)}
#
#
# # ============================================================
# # 路由函数
# # ============================================================
#
# def should_continue_iteration(state: ComplexDiagnosisState) -> str:
#     """判断是否继续迭代"""
#     if state.get("should_continue_iteration", False):
#         logger.info(f"继续迭代，当前第 {state.get('iteration_count')} 次")
#         return "iterate"
#     else:
#         logger.info("停止迭代，进入综合节点")
#         return "synthesize"
#
#
# # ============================================================
# # 创建工作流
# # ============================================================
#
# def create_complex_diagnosis_workflow() -> CompiledStateGraph:
#     """
#     创建复杂诊断工作流
#
#     工作流程：
#     1. 初步分析（症状、八纲、脏腑、病因病机）
#     2. 并行专家分析（鉴别诊断、治则治法、方药推荐、预后评估）
#     3. 验证分析（质疑验证专家）
#     4. 判断是否迭代
#     5. 综合分析（生成最终诊断）
#
#     Returns:
#         编译后的工作流图
#     """
#     logger.info("创建复杂诊断工作流")
#
#     # 创建状态图
#     builder = StateGraph(ComplexDiagnosisState)
#
#     # 添加节点
#     builder.add_node("initial_analysis", initial_analysis_node)
#     builder.add_node("parallel_expert_analysis", parallel_expert_analysis_node)
#     builder.add_node("verification", verification_node)
#     builder.add_node("synthesis", synthesis_node)
#
#     # 添加边
#     builder.add_edge(START, "initial_analysis")
#     builder.add_edge("initial_analysis", "parallel_expert_analysis")
#     builder.add_edge("parallel_expert_analysis", "verification")
#
#     # 条件边：根据验证结果决定是否迭代
#     builder.add_conditional_edges(
#         "verification",
#         should_continue_iteration,
#         {
#             "iterate": "parallel_expert_analysis",  # 重新进行专家分析
#             "synthesize": "synthesis"  # 进入综合节点
#         }
#     )
#
#     builder.add_edge("synthesis", END)
#
#     # 编译图
#     workflow = builder.compile()
#
#     logger.info("复杂诊断工作流创建完成")
#     return workflow
