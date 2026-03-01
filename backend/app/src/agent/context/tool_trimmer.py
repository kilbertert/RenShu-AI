"""
工具消息智能裁剪器

针对工具调用结果的智能裁剪：
- 保留关键结果，裁剪冗余数据
- 合并连续的工具调用
- 支持按工具类型定制裁剪策略
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ToolType(Enum):
    """工具类型"""
    KNOWLEDGE_QUERY = "knowledge_query"    # 知识库查询
    DATABASE_QUERY = "database_query"      # 数据库查询
    WEB_SEARCH = "web_search"              # 网络搜索
    CALCULATION = "calculation"            # 计算
    OTHER = "other"                        # 其他


@dataclass
class ToolRound:
    """一轮工具调用"""
    tool_name: str
    tool_type: ToolType
    call_message: Any  # 工具调用消息
    result_message: Any  # 工具结果消息
    estimated_tokens: int
    is_essential: bool = False  # 是否必要保留


@dataclass
class TrimmedToolRound:
    """裁剪后的工具轮次"""
    original: ToolRound
    trimmed_result: str
    trimmed_tokens: int
    trim_ratio: float


class SmartToolTrimmer:
    """
    智能工具消息裁剪器

    策略：
    1. 识别工具调用轮次
    2. 根据工具类型应用裁剪策略
    3. 保留关键结果，移除冗余
    4. 合并连续查询
    """

    # 工具类型识别关键词
    TOOL_TYPE_KEYWORDS = {
        ToolType.KNOWLEDGE_QUERY: ["search", "query", "lookup", "知识", "查询"],
        ToolType.DATABASE_QUERY: ["sql", "database", "db", "数据库"],
        ToolType.WEB_SEARCH: ["web", "internet", "网络", "搜索"],
        ToolType.CALCULATION: ["calc", "compute", "math", "计算"],
    }

    # 各类型的最大保留 token 数
    MAX_TOKENS_BY_TYPE = {
        ToolType.KNOWLEDGE_QUERY: 500,
        ToolType.DATABASE_QUERY: 300,
        ToolType.WEB_SEARCH: 400,
        ToolType.CALCULATION: 100,
        ToolType.OTHER: 200,
    }

    # TCM 关键词（结果中包含这些词时优先保留）
    TCM_KEYWORDS = [
        "方剂", "处方", "药材", "用法", "用量", "功效", "主治",
        "症状", "证型", "辨证", "舌象", "脉象", "禁忌",
    ]

    def __init__(
        self,
        max_tokens_per_tool: int = 300,
        max_total_tool_tokens: int = 2000,
        keep_last_n_rounds: int = 2,
    ):
        """
        初始化裁剪器

        Args:
            max_tokens_per_tool: 单个工具结果最大 token
            max_total_tool_tokens: 所有工具结果总 token 上限
            keep_last_n_rounds: 保留最后 N 轮工具调用
        """
        self.max_tokens_per_tool = max_tokens_per_tool
        self.max_total_tool_tokens = max_total_tool_tokens
        self.keep_last_n_rounds = keep_last_n_rounds

    def identify_tool_rounds(
        self,
        messages: List[Any]
    ) -> List[ToolRound]:
        """
        识别消息中的工具调用轮次

        Args:
            messages: 消息列表

        Returns:
            工具调用轮次列表
        """
        rounds = []
        i = 0

        while i < len(messages):
            msg = messages[i]

            # 检查是否是工具调用
            if self._is_tool_call(msg):
                # 查找对应的结果
                result_msg = None
                for j in range(i + 1, min(i + 3, len(messages))):
                    if self._is_tool_result(messages[j]):
                        result_msg = messages[j]
                        break

                if result_msg:
                    tool_name = self._get_tool_name(msg)
                    tool_type = self._identify_tool_type(tool_name)
                    tokens = self._estimate_tokens(self._get_content(result_msg))

                    rounds.append(ToolRound(
                        tool_name=tool_name,
                        tool_type=tool_type,
                        call_message=msg,
                        result_message=result_msg,
                        estimated_tokens=tokens,
                        is_essential=self._is_essential_result(result_msg),
                    ))

            i += 1

        return rounds

    def trim_tool_rounds(
        self,
        rounds: List[ToolRound],
        target_tokens: Optional[int] = None,
    ) -> List[TrimmedToolRound]:
        """
        裁剪工具调用轮次

        Args:
            rounds: 工具轮次列表
            target_tokens: 目标 token 数（可选）

        Returns:
            裁剪后的轮次列表
        """
        if not rounds:
            return []

        target = target_tokens or self.max_total_tool_tokens
        trimmed = []

        # 计算每轮的分配额度
        total_rounds = len(rounds)
        essential_rounds = [r for r in rounds if r.is_essential]

        # 为必要轮次分配更多额度
        essential_budget = int(target * 0.6)
        other_budget = target - essential_budget

        for i, round_item in enumerate(rounds):
            # 最后 N 轮保留更多
            is_recent = i >= total_rounds - self.keep_last_n_rounds

            if round_item.is_essential:
                max_tokens = min(
                    essential_budget // max(1, len(essential_rounds)),
                    self.MAX_TOKENS_BY_TYPE.get(round_item.tool_type, 200)
                )
            elif is_recent:
                max_tokens = self.max_tokens_per_tool
            else:
                max_tokens = self.max_tokens_per_tool // 2

            # 裁剪结果
            original_content = self._get_content(round_item.result_message)
            trimmed_content = self._trim_content(
                original_content,
                max_tokens,
                round_item.tool_type,
            )

            trimmed_tokens = self._estimate_tokens(trimmed_content)
            trim_ratio = 1 - (trimmed_tokens / max(1, round_item.estimated_tokens))

            trimmed.append(TrimmedToolRound(
                original=round_item,
                trimmed_result=trimmed_content,
                trimmed_tokens=trimmed_tokens,
                trim_ratio=trim_ratio,
            ))

        return trimmed

    def _trim_content(
        self,
        content: str,
        max_tokens: int,
        tool_type: ToolType,
    ) -> str:
        """
        裁剪内容

        Args:
            content: 原始内容
            max_tokens: 最大 token 数
            tool_type: 工具类型

        Returns:
            裁剪后的内容
        """
        if not content:
            return ""

        current_tokens = self._estimate_tokens(content)
        if current_tokens <= max_tokens:
            return content

        # 根据工具类型应用不同策略
        if tool_type == ToolType.KNOWLEDGE_QUERY:
            return self._trim_knowledge_result(content, max_tokens)
        elif tool_type == ToolType.DATABASE_QUERY:
            return self._trim_database_result(content, max_tokens)
        else:
            return self._trim_generic(content, max_tokens)

    def _trim_knowledge_result(self, content: str, max_tokens: int) -> str:
        """裁剪知识库查询结果"""
        # 优先保留包含 TCM 关键词的段落
        paragraphs = content.split("\n\n")
        important = []
        other = []

        for p in paragraphs:
            if any(kw in p for kw in self.TCM_KEYWORDS):
                important.append(p)
            else:
                other.append(p)

        # 构建结果
        result_parts = []
        current_tokens = 0

        # 先添加重要段落
        for p in important:
            p_tokens = self._estimate_tokens(p)
            if current_tokens + p_tokens <= max_tokens:
                result_parts.append(p)
                current_tokens += p_tokens

        # 再添加其他段落
        for p in other:
            p_tokens = self._estimate_tokens(p)
            if current_tokens + p_tokens <= max_tokens:
                result_parts.append(p)
                current_tokens += p_tokens

        if result_parts:
            return "\n\n".join(result_parts)
        else:
            # 强制截断
            return self._trim_generic(content, max_tokens)

    def _trim_database_result(self, content: str, max_tokens: int) -> str:
        """裁剪数据库查询结果"""
        # 如果是表格数据，保留前几行和统计信息
        lines = content.split("\n")

        # 检测是否是表格格式
        if len(lines) > 5 and any("|" in line for line in lines[:3]):
            # 保留表头和前几行
            header_lines = lines[:3]
            data_lines = lines[3:]

            # 保留前 5 行数据
            kept_data = data_lines[:5]

            # 添加行数统计
            if len(data_lines) > 5:
                kept_data.append(f"... (共 {len(data_lines)} 行数据)")

            return "\n".join(header_lines + kept_data)

        return self._trim_generic(content, max_tokens)

    def _trim_generic(self, content: str, max_tokens: int) -> str:
        """通用裁剪：保留前后部分"""
        # 估算需要保留的字符数
        # 中文约 1.5 字符/token，取保守值
        max_chars = int(max_tokens * 1.5)

        if len(content) <= max_chars:
            return content

        # 保留前 60% 和后 30%
        front_chars = int(max_chars * 0.6)
        back_chars = int(max_chars * 0.3)

        front = content[:front_chars]
        back = content[-back_chars:] if back_chars > 0 else ""

        return f"{front}\n... [内容已裁剪] ...\n{back}"

    def _is_tool_call(self, message: Any) -> bool:
        """检查是否是工具调用"""
        if isinstance(message, dict):
            return "tool_calls" in message
        elif hasattr(message, "tool_calls"):
            return bool(message.tool_calls)
        return False

    def _is_tool_result(self, message: Any) -> bool:
        """检查是否是工具结果"""
        if isinstance(message, dict):
            return message.get("role") == "tool" or "tool_call_id" in message
        elif hasattr(message, "type"):
            return message.type == "tool"
        return False

    def _get_tool_name(self, message: Any) -> str:
        """获取工具名称"""
        if isinstance(message, dict):
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                return tool_calls[0].get("function", {}).get("name", "unknown")
        elif hasattr(message, "tool_calls") and message.tool_calls:
            return message.tool_calls[0].get("name", "unknown")
        return "unknown"

    def _identify_tool_type(self, tool_name: str) -> ToolType:
        """识别工具类型"""
        tool_name_lower = tool_name.lower()
        for tool_type, keywords in self.TOOL_TYPE_KEYWORDS.items():
            if any(kw in tool_name_lower for kw in keywords):
                return tool_type
        return ToolType.OTHER

    def _is_essential_result(self, message: Any) -> bool:
        """判断结果是否必要保留"""
        content = self._get_content(message)
        # 包含 TCM 关键词的结果更重要
        return any(kw in content for kw in self.TCM_KEYWORDS)

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

    def apply_trimming(
        self,
        messages: List[Any],
        target_tokens: Optional[int] = None,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        应用裁剪到消息列表

        Args:
            messages: 原始消息列表
            target_tokens: 目标 token 数

        Returns:
            (裁剪后的消息列表, 裁剪统计信息)
        """
        # 识别工具轮次
        rounds = self.identify_tool_rounds(messages)

        if not rounds:
            return messages, {"trimmed": False}

        # 裁剪
        trimmed_rounds = self.trim_tool_rounds(rounds, target_tokens)

        # 构建结果消息列表
        result = []
        trimmed_results_map = {
            id(tr.original.result_message): tr.trimmed_result
            for tr in trimmed_rounds
        }

        original_tokens = 0
        trimmed_tokens = 0

        for msg in messages:
            if id(msg) in trimmed_results_map:
                # 替换为裁剪后的内容
                trimmed_content = trimmed_results_map[id(msg)]
                original_tokens += self._estimate_tokens(self._get_content(msg))
                trimmed_tokens += self._estimate_tokens(trimmed_content)

                if isinstance(msg, dict):
                    result.append({**msg, "content": trimmed_content})
                else:
                    # 对于 BaseMessage，创建新实例
                    new_msg = type(msg)(content=trimmed_content)
                    result.append(new_msg)
            else:
                result.append(msg)

        stats = {
            "trimmed": True,
            "tool_rounds": len(rounds),
            "original_tokens": original_tokens,
            "trimmed_tokens": trimmed_tokens,
            "saved_tokens": original_tokens - trimmed_tokens,
        }

        return result, stats
