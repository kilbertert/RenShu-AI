from typing import Dict, List, Optional, Literal
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field


# ============== 流式消息类型定义 ==============

class StreamMessageType(str, Enum):
    """流式消息基础类型（非状态类消息）"""
    CONTENT = "content"    # LLM 内容流
    THINKING = "thinking"  # 思考过程（类似 Gemini 的推理展示）
    DONE = "done"          # 完成
    ERROR = "error"        # 错误


# ============== 节点显示注册表 ==============
# 统一管理所有 LangGraph 节点（含子图）的中文显示名称
# type: 前端显示的步骤类型名称
# content: 该步骤的描述性文案

NODE_DISPLAY_REGISTRY: Dict[str, Dict[str, str]] = {

    # ============== 主图节点 ==============
    "middleware_before":         {"type": "安全检查",   "content": "正在进行安全检查..."},
    "analyze_and_route_query":   {"type": "意图识别",   "content": "正在识别您的意图..."},
    "respond_to_general_query":  {"type": "生成回答",   "content": "正在生成回答..."},
    "wellness_subgraph_node":    {"type": "养生分析",   "content": "正在查询养生知识..."},
    "handle_diagnose_query":     {"type": "辨证推理",   "content": "正在进行辨证推理..."},
    "handle_herb_query":         {"type": "药材查询",   "content": "正在查询药材知识..."},
    "handle_prescription_query": {"type": "方剂查询",   "content": "正在查询方剂信息..."},
    "middleware_after":          {"type": "完成处理",   "content": "正在完成处理..."},

    # ============== 诊断子图节点 ==============
    "collect_info":              {"type": "信息收集",   "content": "正在收集问诊信息..."},
    "analyze_follow_up":         {"type": "追问分析",   "content": "正在分析是否需要追问..."},
    "assess_complexity":         {"type": "复杂度评估", "content": "正在评估病情复杂度..."},
    "simple_diagnosis":          {"type": "辨证分析",   "content": "正在进行辨证分析..."},
    "moderate_diagnosis":        {"type": "深度辨证",   "content": "正在进行深度辨证..."},
    "complex_diagnosis":         {"type": "综合辨证",   "content": "正在进行综合辨证分析..."},

    # ============== 养生子图节点 ==============
    "wellness_router_node":         {"type": "养生路由",  "content": "正在分析养生类型..."},
    "handle_wellness_seasonal":     {"type": "季节养生",  "content": "正在生成季节养生建议..."},
    "handle_wellness_daily":        {"type": "日常养生",  "content": "正在生成日常养生建议..."},
    "handle_wellness_constitution": {"type": "体质调理",  "content": "正在生成体质调理方案..."},
    "handle_wellness_general":      {"type": "养生建议",  "content": "正在生成养生建议..."},

    # ============== 中等辨证 Map-Reduce 子图 ==============
    "plan_queries":              {"type": "查询规划",   "content": "正在规划知识图谱查询..."},
    "execute_query":             {"type": "知识检索",   "content": "正在检索中医知识..."},
    "synthesize_diagnosis":      {"type": "诊断综合",   "content": "正在综合诊断结果..."},
}


# ============== 模型配置 ==============

class ModelConfiguration(BaseModel):
    provider_id: str
    model_id: str
    model_name: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 2000

class ChatRequest(BaseModel):
    """
    聊天请求参数
    """
    user_id: str=Field(..., description="用户ID")
    conversation_id: str=Field(..., description="会话ID")
    query: str=Field(..., description="用户询问的问题")
    model_configuration: ModelConfiguration=Field(..., description="模型配置")
    stream: bool=Field(False, description="是否流式返回")
    enable_thinking: bool=Field(False, description="是否启用思考过程展示（仅支持thinking的模型有效）")

class PersonaAnalysisRequest(BaseModel):
    """
    用户画像分析请求参数
    """
    user_id: str = Field(..., description="用户ID")
    text: str = Field(..., description="用户最新输入文本")
    current_persona: Optional[dict] = Field(None, description="当前用户画像数据")
    conversation_id: Optional[str] = Field(None, description="会话ID，用于更新会话画像")
    model_configuration: ModelConfiguration = Field(..., description="模型配置")

class ChatResumeRequest(BaseModel):
    """
    恢复被 interrupt 暂停的聊天请求
    """
    conversation_id: str = Field(..., description="会话ID")
    thread_id: str = Field(..., description="LangGraph thread_id")
    query: str = Field(..., description="用户追问回答")
    model_configuration: ModelConfiguration = Field(..., description="模型配置")

