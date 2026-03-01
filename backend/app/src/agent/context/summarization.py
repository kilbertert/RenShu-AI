"""
TCM 专用摘要生成器

针对中医对话特点的摘要生成：
- 保留关键医学信息（症状、诊断、方剂）
- 结构化摘要格式
- 支持增量摘要
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel


# TCM 摘要模板
TCM_SUMMARY_TEMPLATE = """请对以下中医对话进行摘要，保留关键医学信息。

## 摘要要求
1. 保留所有症状描述
2. 保留舌象、脉象等诊断信息
3. 保留辨证结论和证型
4. 保留方剂和用药建议
5. 忽略寒暄和重复内容

## 对话内容
{conversation}

## 输出格式
请按以下格式输出摘要：

【主诉】患者主要症状和就诊原因
【症状】详细症状列表
【四诊信息】舌象、脉象等
【辨证】证型判断
【建议】已给出的方剂或建议

如果某项信息未提及，标注"未提及"。
"""


@dataclass
class TCMSummaryResult:
    """TCM 摘要结果"""
    summary: str
    chief_complaint: Optional[str] = None
    symptoms: List[str] = None
    diagnosis_info: Optional[str] = None
    syndrome: Optional[str] = None
    suggestions: Optional[str] = None
    original_token_count: int = 0
    summary_token_count: int = 0

    def __post_init__(self):
        if self.symptoms is None:
            self.symptoms = []


class TCMSummarizer:
    """
    TCM 专用摘要生成器

    特点：
    - 保留中医关键信息
    - 结构化输出
    - 支持增量更新
    """

    # 关键信息标记
    SECTION_MARKERS = {
        "chief_complaint": ["主诉", "主要症状"],
        "symptoms": ["症状", "表现"],
        "diagnosis_info": ["四诊", "舌象", "脉象"],
        "syndrome": ["辨证", "证型"],
        "suggestions": ["建议", "方剂", "处方"],
    }

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        max_summary_tokens: int = 500,
    ):
        """
        初始化摘要生成器

        Args:
            llm: 语言模型（用于生成摘要）
            max_summary_tokens: 摘要最大 token 数
        """
        self.llm = llm
        self.max_summary_tokens = max_summary_tokens
        self._cached_summary: Optional[TCMSummaryResult] = None

    def summarize_messages(
        self,
        messages: List[Any],
        use_llm: bool = True,
    ) -> TCMSummaryResult:
        """
        对消息列表生成摘要

        Args:
            messages: 消息列表
            use_llm: 是否使用 LLM（False 则使用规则提取）

        Returns:
            摘要结果
        """
        # 提取对话内容
        conversation_text = self._extract_conversation(messages)
        original_tokens = self._estimate_tokens(conversation_text)

        if use_llm and self.llm:
            # 使用 LLM 生成摘要
            summary = self._llm_summarize(conversation_text)
        else:
            # 使用规则提取关键信息
            summary = self._rule_based_summarize(messages)

        summary_tokens = self._estimate_tokens(summary)

        # 解析结构化信息
        parsed = self._parse_summary(summary)

        return TCMSummaryResult(
            summary=summary,
            chief_complaint=parsed.get("chief_complaint"),
            symptoms=parsed.get("symptoms", []),
            diagnosis_info=parsed.get("diagnosis_info"),
            syndrome=parsed.get("syndrome"),
            suggestions=parsed.get("suggestions"),
            original_token_count=original_tokens,
            summary_token_count=summary_tokens,
        )

    def _extract_conversation(self, messages: List[Any]) -> str:
        """提取对话文本"""
        lines = []
        for msg in messages:
            role = self._get_role(msg)
            content = self._get_content(msg)

            if role == "system":
                continue  # 跳过系统消息

            role_name = "患者" if role in ["human", "user"] else "医生"
            if content:
                lines.append(f"{role_name}: {content}")

        return "\n".join(lines)

    def _llm_summarize(self, conversation: str) -> str:
        """使用 LLM 生成摘要"""
        if not self.llm:
            return self._rule_based_summarize_text(conversation)

        prompt = TCM_SUMMARY_TEMPLATE.format(conversation=conversation)

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            # 降级到规则提取
            return self._rule_based_summarize_text(conversation)

    def _rule_based_summarize(self, messages: List[Any]) -> str:
        """基于规则的摘要提取"""
        sections = {
            "chief_complaint": [],
            "symptoms": [],
            "diagnosis_info": [],
            "syndrome": [],
            "suggestions": [],
        }

        for msg in messages:
            content = self._get_content(msg)
            if not content:
                continue

            # 检查各类关键信息
            for section, markers in self.SECTION_MARKERS.items():
                for marker in markers:
                    if marker in content:
                        sections[section].append(content)
                        break

        # 构建摘要
        lines = []
        if sections["chief_complaint"]:
            lines.append(f"【主诉】{sections['chief_complaint'][0][:100]}")
        if sections["symptoms"]:
            symptoms_text = "; ".join(s[:50] for s in sections["symptoms"][:5])
            lines.append(f"【症状】{symptoms_text}")
        if sections["diagnosis_info"]:
            lines.append(f"【四诊信息】{sections['diagnosis_info'][0][:100]}")
        if sections["syndrome"]:
            lines.append(f"【辨证】{sections['syndrome'][0][:100]}")
        if sections["suggestions"]:
            lines.append(f"【建议】{sections['suggestions'][-1][:200]}")

        return "\n".join(lines) if lines else "对话内容较少，无需摘要"

    def _rule_based_summarize_text(self, text: str) -> str:
        """对文本进行规则摘要"""
        lines = text.split("\n")
        important_lines = []

        keywords = ["症状", "主诉", "舌", "脉", "方", "药", "证", "诊断", "建议"]

        for line in lines:
            if any(kw in line for kw in keywords):
                important_lines.append(line)

        if important_lines:
            return "\n".join(important_lines[:10])
        else:
            # 保留前后几行
            return "\n".join(lines[:3] + ["..."] + lines[-3:])

    def _parse_summary(self, summary: str) -> Dict[str, Any]:
        """解析摘要中的结构化信息"""
        result = {}

        # 解析各部分
        section_patterns = {
            "chief_complaint": "【主诉】",
            "diagnosis_info": "【四诊信息】",
            "syndrome": "【辨证】",
            "suggestions": "【建议】",
        }

        for key, pattern in section_patterns.items():
            if pattern in summary:
                start = summary.index(pattern) + len(pattern)
                # 找到下一个 【 或结束
                end = summary.find("【", start)
                if end == -1:
                    end = len(summary)
                result[key] = summary[start:end].strip()

        # 解析症状（可能有多个）
        if "【症状】" in summary:
            start = summary.index("【症状】") + len("【症状】")
            end = summary.find("【", start)
            if end == -1:
                end = len(summary)
            symptoms_text = summary[start:end].strip()
            result["symptoms"] = [s.strip() for s in symptoms_text.split(";") if s.strip()]

        return result

    def _get_role(self, message: Any) -> str:
        """获取消息角色"""
        if isinstance(message, dict):
            return message.get("role", "")
        elif isinstance(message, SystemMessage):
            return "system"
        elif isinstance(message, HumanMessage):
            return "human"
        elif isinstance(message, AIMessage):
            return "assistant"
        elif hasattr(message, "type"):
            return message.type
        return ""

    def _get_content(self, message: Any) -> str:
        """获取消息内容"""
        if isinstance(message, dict):
            return message.get("content", "")
        elif hasattr(message, "content"):
            return message.content or ""
        return ""

    def _estimate_tokens(self, text: str) -> int:
        """估算 token 数量"""
        if not text:
            return 0
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    def create_summary_message(self, result: TCMSummaryResult) -> Dict[str, Any]:
        """
        创建摘要消息（用于插入对话历史）

        Args:
            result: 摘要结果

        Returns:
            可插入对话的消息字典
        """
        return {
            "role": "system",
            "content": f"[对话摘要]\n{result.summary}",
            "metadata": {
                "type": "summary",
                "original_tokens": result.original_token_count,
                "summary_tokens": result.summary_token_count,
            }
        }

    def get_compression_ratio(self, result: TCMSummaryResult) -> float:
        """计算压缩比"""
        if result.original_token_count == 0:
            return 0.0
        return 1 - (result.summary_token_count / result.original_token_count)
