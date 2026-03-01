"""
TCM Tongue Image Analyzer
中医舌诊图像分析器

集成多模态LLM进行舌诊分析
"""

import os
import base64
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .tcm_states import TongueAnalysisResult


# 舌诊分析系统提示词
TONGUE_ANALYSIS_SYSTEM_PROMPT = """你是一位经验丰富的中医舌诊专家。请根据用户提供的舌象图片进行专业分析。

分析要点：
1. 舌色：观察舌质颜色（淡红、红、绛红、紫暗、淡白等）
2. 舌形：观察舌体形态（胖大、瘦薄、齿痕、裂纹、芒刺等）
3. 苔色：观察舌苔颜色（白、黄、灰、黑等）
4. 苔质：观察舌苔质地（薄、厚、腻、燥、滑、剥落等）
5. 舌态：观察舌体动态（强硬、痿软、颤动、歪斜等）

请根据以上观察，给出：
- 详细的舌象描述
- 可能的证型提示
- 养生建议

注意：
- 分析应客观准确，基于图像实际情况
- 证型提示仅供参考，不作为诊断依据
- 如图片质量不佳，请说明并给出建议
"""


class TongueAnalyzer:
    """舌诊分析器 - 集成多模态LLM"""

    def __init__(self, model_name: str = None):
        """
        初始化舌诊分析器

        Args:
            model_name: 模型名称，支持 qwen-vl-plus / gpt-4o / moonshot-v1-128k
        """
        self.service = os.getenv("VISION_SERVICE", "TONGYI")
        self.model_name = model_name or self._get_default_model()
        self._llm = None

    def _get_default_model(self) -> str:
        """获取默认模型名称"""
        model_map = {
            "TONGYI": "qwen-vl-plus",
            "OPENAI": "gpt-4o",
            "MOONSHOT": "moonshot-v1-128k-vision-preview",
        }
        return model_map.get(self.service, "qwen-vl-plus")

    def _get_llm(self) -> ChatOpenAI:
        """获取多模态LLM实例"""
        if self._llm is not None:
            return self._llm

        if self.service == "TONGYI":
            self._llm = ChatOpenAI(
                model=self.model_name,
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url=os.getenv(
                    "DASHSCOPE_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1"
                ),
                temperature=0.3,
            )
        elif self.service == "OPENAI":
            self._llm = ChatOpenAI(
                model=self.model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
                temperature=0.3,
            )
        elif self.service == "MOONSHOT":
            self._llm = ChatOpenAI(
                model=self.model_name,
                api_key=os.getenv("MOONSHOT_API_KEY"),
                base_url=os.getenv(
                    "MOONSHOT_BASE_URL",
                    "https://api.moonshot.cn/v1"
                ),
                temperature=0.3,
            )
        else:
            # 默认使用通义千问
            self._llm = ChatOpenAI(
                model="qwen-vl-plus",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                temperature=0.3,
            )

        return self._llm

    async def analyze_tongue_image(
        self,
        image_url: str,
        additional_info: str = None
    ) -> TongueAnalysisResult:
        """
        分析舌诊图像

        Args:
            image_url: 图片URL或base64编码的图片数据
            additional_info: 用户提供的额外信息（如症状描述）

        Returns:
            TongueAnalysisResult: 结构化的舌诊分析结果
        """
        llm = self._get_llm()

        # 构建多模态消息
        user_content = []

        # 添加图片
        if image_url.startswith("data:"):
            # base64格式
            user_content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
        elif image_url.startswith("http"):
            # URL格式
            user_content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
        else:
            # 假设是纯base64字符串
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_url}"}
            })

        # 添加文本提示
        prompt_text = "请分析这张舌象图片。"
        if additional_info:
            prompt_text += f"\n\n用户补充信息：{additional_info}"

        user_content.append({
            "type": "text",
            "text": prompt_text
        })

        # 使用结构化输出
        llm_with_structure = llm.with_structured_output(TongueAnalysisResult)

        try:
            result = await llm_with_structure.ainvoke([
                SystemMessage(content=TONGUE_ANALYSIS_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ])
            return result
        except Exception as e:
            # 如果结构化输出失败，尝试普通调用后解析
            try:
                response = await llm.ainvoke([
                    SystemMessage(content=TONGUE_ANALYSIS_SYSTEM_PROMPT),
                    HumanMessage(content=user_content),
                ])
                return self._parse_response_to_result(response.content)
            except Exception as inner_e:
                # 返回错误结果
                return TongueAnalysisResult(
                    tongue_color="无法识别",
                    tongue_shape="无法识别",
                    coating_color="无法识别",
                    coating_texture="无法识别",
                    analysis=f"图像分析失败：{str(e)}。请确保图片清晰，光线充足。",
                    syndrome_hints=[]
                )

    def _parse_response_to_result(self, content: str) -> TongueAnalysisResult:
        """
        解析LLM响应为结构化结果

        Args:
            content: LLM响应内容

        Returns:
            TongueAnalysisResult: 解析后的结果
        """
        # 简单解析，提取关键信息
        result = TongueAnalysisResult(
            tongue_color="",
            tongue_shape="",
            coating_color="",
            coating_texture="",
            analysis=content,
            syndrome_hints=[]
        )

        # 尝试提取舌色
        color_keywords = ["淡红", "红", "绛红", "紫暗", "淡白", "青紫"]
        for keyword in color_keywords:
            if keyword in content:
                result.tongue_color = keyword
                break

        # 尝试提取苔色
        coating_colors = ["白苔", "黄苔", "灰苔", "黑苔", "白", "黄", "灰", "黑"]
        for keyword in coating_colors:
            if keyword in content:
                result.coating_color = keyword.replace("苔", "")
                break

        # 尝试提取苔质
        coating_textures = ["薄苔", "厚苔", "腻苔", "燥苔", "滑苔", "薄", "厚", "腻", "燥", "滑"]
        for keyword in coating_textures:
            if keyword in content:
                result.coating_texture = keyword.replace("苔", "")
                break

        # 尝试提取证型提示
        syndrome_keywords = [
            "气虚", "血虚", "阴虚", "阳虚", "痰湿", "湿热", "血瘀",
            "气滞", "寒湿", "风热", "风寒", "肝郁", "脾虚", "肾虚"
        ]
        for keyword in syndrome_keywords:
            if keyword in content:
                result.syndrome_hints.append(keyword)

        return result


def extract_image_from_message(message_content) -> Optional[str]:
    """
    从消息内容中提取图片URL

    Args:
        message_content: 消息内容，可能是字符串或列表

    Returns:
        Optional[str]: 图片URL或None
    """
    if isinstance(message_content, str):
        # 纯文本消息，没有图片
        return None

    if isinstance(message_content, list):
        # 多模态消息格式
        for item in message_content:
            if isinstance(item, dict):
                if item.get("type") == "image_url":
                    image_url = item.get("image_url", {})
                    if isinstance(image_url, dict):
                        return image_url.get("url")
                    return image_url
                elif item.get("type") == "image":
                    # 另一种格式
                    return item.get("source", {}).get("data") or item.get("url")

    return None


def extract_text_from_message(message_content) -> str:
    """
    从消息内容中提取文本

    Args:
        message_content: 消息内容，可能是字符串或列表

    Returns:
        str: 提取的文本
    """
    if isinstance(message_content, str):
        return message_content

    if isinstance(message_content, list):
        texts = []
        for item in message_content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
            elif isinstance(item, str):
                texts.append(item)
        return " ".join(texts)

    return str(message_content)
