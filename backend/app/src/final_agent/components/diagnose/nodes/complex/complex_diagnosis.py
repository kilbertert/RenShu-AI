"""
复杂诊断节点

基于 DeepAgents 框架的深度辨证分析
使用 SubAgentMiddleware 实现并行专家调度
"""

from typing import Dict, Any
import time
import json

from app.src.utils import get_logger
from app.src.agent.components.diagnose.states import DiagnoseOverallState

from .deep_search_agent_custom import create_deep_search_agent_custom

logger = get_logger("complex_diagnosis")


async def complex_diagnosis(state: DiagnoseOverallState) -> Dict[str, Any]:
    """
    复杂诊断节点
    
    使用 DeepSearch Agent（基于 DeepAgents 框架）进行深度辨证分析：
    
    1. 数据收集阶段（并行）：
       - kg_syndrome_search: 查询知识图谱证型
       - case_vector_search: 检索相似医案
       - classics_search: 检索古籍论述
       - web_search: 搜索最新研究
    
    2. 专家咨询阶段（并行）：
       - 鉴别诊断专家
       - 治则治法专家
       - 方药推荐专家
       - 预后评估专家
       - 质疑验证专家
    
    3. 综合决策阶段：
       - 综合所有专家意见
       - 输出最终诊断结果
    
    Args:
        state: 诊断总状态
        
    Returns:
        包含诊断结果的字典
    """
    logger.info("开始复杂诊断（DeepAgents 架构）")
    start_time = time.time()
    
    try:
        # 获取收集的信息
        collected_info = state.get("collected_info", {})
        
        if not collected_info:
            logger.warning("未找到收集的患者信息")
            return {
                "answer": "无法进行复杂诊断：缺少患者信息",
                "steps": ["复杂诊断: 缺少患者信息"],
                "error": "缺少患者信息"
            }
        
        # 创建 DeepSearch Agent
        logger.info("创建 DeepSearch Agent...")
        agent = create_deep_search_agent_custom(enable_doctor_approval=False)
        
        # 生成会话 ID
        user_id = state.get("user_id", "unknown")
        thread_id = f"diagnosis_{user_id}_{int(time.time())}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # 构建诊断请求
        prompt = _build_diagnosis_prompt(collected_info)
        
        # 执行诊断
        logger.info("执行深度辨证分析...")
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=config
        )
        
        # 检查是否中断（等待医生审批）
        if "__interrupt__" in result:
            logger.info("诊断等待医生审批")
            return {
                "pending_approval": True,
                "diagnosis_draft": result["__interrupt__"],
                "thread_id": thread_id,
                "steps": ["复杂诊断: 等待医生审批"]
            }
        
        # 提取最终结果
        answer = ""
        if result.get("messages"):
            answer = result["messages"][-1].content
        
        # 尝试解析结构化结果
        diagnosis_result = _parse_diagnosis_result(answer)
        confidence = diagnosis_result.get("confidence", 0.8)
        
        elapsed_time = time.time() - start_time
        logger.info(f"复杂诊断完成，耗时 {elapsed_time:.2f} 秒，置信度 {confidence:.2f}")
        
        return {
            "answer": answer,
            "steps": result.get("steps", []) + [f"复杂诊断: 完成（置信度 {confidence:.2f}）"],
            "confidence": confidence,
            "diagnosis_result": diagnosis_result,
            "thread_id": thread_id
        }
        
    except Exception as e:
        logger.error(f"复杂诊断异常: {e}", exc_info=True)
        return {
            "answer": f"复杂诊断异常: {str(e)}",
            "steps": ["复杂诊断: 异常"],
            "error": str(e)
        }


def _build_diagnosis_prompt(collected_info: Dict[str, Any]) -> str:
    """
    构建诊断请求提示词
    
    Args:
        collected_info: 收集的患者信息
        
    Returns:
        格式化的诊断请求
    """
    prompt = f"""
请对以下患者信息进行深度辨证分析：

## 患者信息
{json.dumps(collected_info, ensure_ascii=False, indent=2)}

## 分析要求

请按照以下步骤进行分析：

### 步骤 1: 数据收集（并行查询）
1. 使用 kg_syndrome_search 查询知识图谱中的相关证型
2. 使用 case_vector_search 检索相似的历史医案
3. 使用 classics_search 检索中医古籍的相关论述
4. 如需要，使用 web_search 搜索最新医学研究

### 步骤 2: 专家咨询（并行分析）
咨询 5 位专家：
1. differential_diagnosis_expert - 进行鉴别诊断分析
2. treatment_principle_expert - 制定治则治法
3. prescription_recommendation_expert - 推荐方药
4. prognosis_evaluation_expert - 评估预后
5. verification_expert - 验证诊断合理性

### 步骤 3: 综合决策
综合所有专家意见，给出最终诊断结果。

## 输出要求

请提供完整的辨证分析结果，包括：
1. **证型**：主证 + 兼证（如有）
2. **病因病机**：病因、病机、病位、病性
3. **治则治法**：治疗原则和方法
4. **方药建议**：推荐方剂（仅供参考）
5. **预后评估**：病情趋势和注意事项
6. **置信度**：0-1 的评分
7. **依据**：知识图谱/医案/古籍的支持证据

**重要提示**：本诊断结果仅供参考，具体诊疗请咨询专业中医师。
"""
    return prompt


def _parse_diagnosis_result(answer: str) -> Dict[str, Any]:
    """
    解析诊断结果
    
    尝试从回答中提取结构化数据
    
    Args:
        answer: Agent 的回答文本
        
    Returns:
        解析后的诊断结果字典
    """
    result = {
        "raw_answer": answer,
        "confidence": 0.8
    }
    
    try:
        # 尝试提取 JSON 块
        if "```json" in answer:
            json_start = answer.find("```json") + 7
            json_end = answer.find("```", json_start)
            if json_end > json_start:
                json_str = answer[json_start:json_end].strip()
                parsed = json.loads(json_str)
                result.update(parsed)
        
        # 尝试提取置信度
        if "置信度" in answer:
            import re
            match = re.search(r"置信度[：:]\s*(\d+\.?\d*)", answer)
            if match:
                result["confidence"] = float(match.group(1))
                if result["confidence"] > 1:
                    result["confidence"] /= 100  # 转换百分比
                    
    except Exception as e:
        logger.warning(f"解析诊断结果失败: {e}")
    
    return result


# 保持向后兼容的函数名
async def run_complex_diagnosis(state: DiagnoseOverallState) -> Dict[str, Any]:
    """
    运行复杂诊断（向后兼容的别名）
    
    Args:
        state: 诊断总状态
        
    Returns:
        诊断结果
    """
    return await complex_diagnosis(state)
