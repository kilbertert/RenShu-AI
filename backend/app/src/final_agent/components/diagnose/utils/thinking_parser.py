"""
Thinking Content Parser - 思考内容解析器

用于从不同 LLM 提供商的响应中提取思考过程（reasoning/thinking content）

支持的格式：
1. 标准格式：DeepSeek, Qwen, 智谱 GLM - chunk.additional_kwargs['reasoning_content']
2. OpenAI 格式：OpenAI gpt-5 - chunk.content[i]['summary'] (type='reasoning')
3. Claude/Gemini 格式：Claude, Gemini - chunk.content[i]['thinking'] (type='thinking')
"""

from typing import Optional, Dict, Any
from langchain_core.messages import AIMessageChunk


def extract_thinking_from_chunk(chunk: AIMessageChunk) -> Optional[str]:
    """
    从 LangChain AIMessageChunk 中提取思考内容
    
    支持三种格式：
    1. 标准格式（DeepSeek, Qwen, 智谱 GLM）：chunk.additional_kwargs['reasoning_content']
    2. OpenAI 格式（gpt-5 系列）：chunk.content[i]['summary'] (type='reasoning')
    3. Claude/Gemini 格式：chunk.content[i]['thinking'] (type='thinking')
    
    Args:
        chunk: LangChain AIMessageChunk 对象
        
    Returns:
        思考内容字符串，如果没有则返回 None
    """
    if not isinstance(chunk, AIMessageChunk):
        return None
    
    # 格式 1：标准格式 - DeepSeek, Qwen, 智谱 GLM
    # 这是最常见的格式，优先检查
    if hasattr(chunk, 'additional_kwargs') and 'reasoning_content' in chunk.additional_kwargs:
        return chunk.additional_kwargs['reasoning_content']
    
    # 格式 2 & 3：OpenAI 和 Claude/Gemini 格式
    # 思考内容在 content 字段中，需要根据 type 区分
    if hasattr(chunk, 'content') and isinstance(chunk.content, list):
        thinking_text = ""
        
        for item in chunk.content:
            if not isinstance(item, dict):
                continue
            
            # OpenAI 格式：type='reasoning', 内容在 summary 字段
            if item.get('type') == 'reasoning':
                summary_list = item.get('summary', [])
                for summary_item in summary_list:
                    if isinstance(summary_item, dict):
                        thinking_text += summary_item.get('text', '')
            
            # Claude/Gemini 格式：type='thinking', 内容在 thinking 字段
            elif item.get('type') == 'thinking':
                thinking_text += item.get('thinking', '')
        
        if thinking_text:
            return thinking_text
    
    # 备用方式：检查其他可能的字段名（用于兼容性）
    if hasattr(chunk, 'additional_kwargs'):
        for key in ['thinking', 'reasoning', 'thought']:
            if key in chunk.additional_kwargs:
                return chunk.additional_kwargs[key]
    
    return None


def extract_answer_from_chunk(chunk: AIMessageChunk) -> Optional[str]:
    """
    从 LangChain AIMessageChunk 中提取答案内容
    
    处理不同格式：
    1. 标准格式：chunk.content 是字符串
    2. OpenAI/Claude 格式：chunk.content 是列表，需要提取 type='text' 的内容
    
    Args:
        chunk: LangChain AIMessageChunk 对象
        
    Returns:
        答案内容字符串，如果没有则返回 None
    """
    if not isinstance(chunk, AIMessageChunk):
        return None
    
    if not hasattr(chunk, 'content'):
        return None
    
    # 标准格式：content 是字符串
    if isinstance(chunk.content, str):
        return chunk.content if chunk.content else None
    
    # OpenAI/Claude/Gemini 格式：content 是列表
    if isinstance(chunk.content, list):
        answer_text = ""
        
        for item in chunk.content:
            if not isinstance(item, dict):
                continue
            
            # 提取 type='text' 的内容（通常 index=1）
            if item.get('type') == 'text':
                answer_text += item.get('text', '')
        
        return answer_text if answer_text else None
    
    return None


def parse_thinking_tags(content: str) -> Dict[str, str]:
    """
    从文本中解析 <think> 标签
    
    某些模型（如 DeepSeek）可能在 content 中包含 <think>...</think> 标签
    
    Args:
        content: 包含可能的 <think> 标签的文本
        
    Returns:
        字典，包含 'thinking' 和 'answer' 两个键
    """
    import re
    
    # 匹配 <think>...</think> 标签
    think_pattern = r'<think>(.*?)</think>'
    matches = re.findall(think_pattern, content, re.DOTALL)
    
    if matches:
        # 提取思考内容
        thinking = '\n'.join(matches)
        # 移除 <think> 标签，得到最终答案
        answer = re.sub(think_pattern, '', content, flags=re.DOTALL).strip()
        
        return {
            'thinking': thinking,
            'answer': answer
        }
    
    # 没有找到标签，全部作为答案
    return {
        'thinking': '',
        'answer': content
    }


def format_thinking_for_display(thinking: str, max_length: int = 500) -> str:
    """
    格式化思考内容用于显示
    
    Args:
        thinking: 思考内容
        max_length: 最大显示长度（用于日志）
        
    Returns:
        格式化后的思考内容
    """
    if not thinking:
        return ""
    
    # 截断过长的内容
    if len(thinking) > max_length:
        return thinking[:max_length] + "..."
    
    return thinking


def should_emit_thinking(provider_name: str) -> bool:
    """
    判断是否应该发送思考内容到前端
    
    Args:
        provider_name: 提供商名称
        
    Returns:
        是否应该发送思考内容
    """
    # 支持思考内容的提供商
    thinking_providers = [
        'qwen', 'dashscope', 'tongyi', 'alibaba', 'aliyun',  # Qwen 系列
        'deepseek',  # DeepSeek
        'openai',  # OpenAI (GPT-5+)
        'anthropic', 'claude',  # Claude
        'google', 'gemini',  # Gemini
        'zhipu', 'glm',  # 智谱 GLM
        'moonshot', 'kimi',  # Moonshot (Kimi)
        'doubao', 'bytedance', 'volcengine',  # 豆包
    ]
    
    return provider_name.lower() in thinking_providers
