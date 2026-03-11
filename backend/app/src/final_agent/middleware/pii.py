"""
PII 检测与脱敏中间件

功能：
1. 检测用户输入中的个人敏感信息（PII）
2. 对敏感信息进行脱敏处理
3. 检测模型输出中的 PII 泄露

支持的 PII 类型：
- 手机号码
- 身份证号
- 银行卡号
- 邮箱地址
- 姓名（可选）
- 地址（可选）
"""

import re
from typing import Any, Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from langchain_core.messages import AIMessage, HumanMessage

from .base import BaseMiddleware, MiddlewareConfig


class PIIType(Enum):
    """PII 类型"""
    PHONE = "phone"           # 手机号
    ID_CARD = "id_card"       # 身份证
    BANK_CARD = "bank_card"   # 银行卡
    EMAIL = "email"           # 邮箱
    NAME = "name"             # 姓名
    ADDRESS = "address"       # 地址


@dataclass
class PIIMatch:
    """PII 匹配结果"""
    pii_type: PIIType
    original: str
    masked: str
    start: int
    end: int


@dataclass
class PIIConfig(MiddlewareConfig):
    """PII 中间件配置"""
    # 检测开关
    detect_phone: bool = True
    detect_id_card: bool = True
    detect_bank_card: bool = True
    detect_email: bool = True
    detect_name: bool = False  # 姓名检测误报率高，默认关闭
    detect_address: bool = False  # 地址检测误报率高，默认关闭

    # 脱敏开关
    mask_input: bool = True   # 是否脱敏输入
    mask_output: bool = True  # 是否检测输出

    # 脱敏方式
    mask_char: str = "*"      # 脱敏字符
    keep_prefix: int = 3      # 保留前缀长度
    keep_suffix: int = 4      # 保留后缀长度


class TCMPIIMiddleware(BaseMiddleware):
    """
    PII 检测与脱敏中间件

    使用正则表达式检测常见的 PII 类型，并进行脱敏处理。
    """

    # ==================== 正则模式 ====================

    # 手机号：1开头的11位数字
    PHONE_PATTERN = re.compile(r"1[3-9]\d{9}")

    # 身份证：18位，最后一位可能是X
    ID_CARD_PATTERN = re.compile(r"\d{17}[\dXx]")

    # 银行卡：16-19位数字
    BANK_CARD_PATTERN = re.compile(r"\d{16,19}")

    # 邮箱
    EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

    # 常见姓氏（用于姓名检测）
    COMMON_SURNAMES = [
        "王", "李", "张", "刘", "陈", "杨", "黄", "赵", "周", "吴",
        "徐", "孙", "马", "朱", "胡", "郭", "何", "高", "林", "罗",
    ]

    def __init__(self, config: Optional[PIIConfig] = None):
        """初始化 PII 中间件"""
        super().__init__(config or PIIConfig(name="TCMPIIMiddleware", priority=20))
        self.pii_config = config or PIIConfig()

    def before_model(
        self,
        state: Dict[str, Any],
        runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """
        模型调用前：检测并脱敏用户输入中的 PII
        """
        if not self.pii_config.mask_input:
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        last_message = messages[-1]
        if not isinstance(last_message, HumanMessage):
            return None

        content = last_message.content
        masked_content, pii_matches = self._detect_and_mask(content)

        if pii_matches:
            # 记录检测到的 PII
            pii_log = [f"{m.pii_type.value}: {m.original} -> {m.masked}" for m in pii_matches]

            # 更新消息内容
            new_message = HumanMessage(content=masked_content)
            new_messages = messages[:-1] + [new_message]

            return {
                "messages": new_messages,
                "steps": [f"PII检测: 发现 {len(pii_matches)} 处敏感信息已脱敏"],
                "_pii_detected": pii_log,  # 内部记录
            }

        return None

    def after_model(
        self,
        state: Dict[str, Any],
        runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """
        模型调用后：检测输出中是否有 PII 泄露
        """
        if not self.pii_config.mask_output:
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            return None

        content = last_message.content
        masked_content, pii_matches = self._detect_and_mask(content)

        if pii_matches:
            # 输出中发现 PII，进行脱敏
            new_message = AIMessage(content=masked_content)
            new_messages = messages[:-1] + [new_message]

            return {
                "messages": new_messages,
                "steps": [f"PII检测: 输出中发现 {len(pii_matches)} 处敏感信息已脱敏"],
            }

        return None

    def _detect_and_mask(self, text: str) -> Tuple[str, List[PIIMatch]]:
        """
        检测并脱敏文本中的 PII

        Args:
            text: 原始文本

        Returns:
            (脱敏后的文本, PII匹配列表)
        """
        matches: List[PIIMatch] = []
        masked_text = text

        # 检测手机号
        if self.pii_config.detect_phone:
            for match in self.PHONE_PATTERN.finditer(text):
                original = match.group()
                masked = self._mask_string(original, keep_prefix=3, keep_suffix=4)
                matches.append(PIIMatch(
                    pii_type=PIIType.PHONE,
                    original=original,
                    masked=masked,
                    start=match.start(),
                    end=match.end(),
                ))

        # 检测身份证
        if self.pii_config.detect_id_card:
            for match in self.ID_CARD_PATTERN.finditer(text):
                original = match.group()
                # 验证身份证格式（简单校验）
                if self._is_valid_id_card(original):
                    masked = self._mask_string(original, keep_prefix=6, keep_suffix=4)
                    matches.append(PIIMatch(
                        pii_type=PIIType.ID_CARD,
                        original=original,
                        masked=masked,
                        start=match.start(),
                        end=match.end(),
                    ))

        # 检测银行卡
        if self.pii_config.detect_bank_card:
            for match in self.BANK_CARD_PATTERN.finditer(text):
                original = match.group()
                # 排除已识别为身份证的
                if not any(m.original == original for m in matches):
                    masked = self._mask_string(original, keep_prefix=4, keep_suffix=4)
                    matches.append(PIIMatch(
                        pii_type=PIIType.BANK_CARD,
                        original=original,
                        masked=masked,
                        start=match.start(),
                        end=match.end(),
                    ))

        # 检测邮箱
        if self.pii_config.detect_email:
            for match in self.EMAIL_PATTERN.finditer(text):
                original = match.group()
                masked = self._mask_email(original)
                matches.append(PIIMatch(
                    pii_type=PIIType.EMAIL,
                    original=original,
                    masked=masked,
                    start=match.start(),
                    end=match.end(),
                ))

        # 按位置逆序替换（避免位置偏移）
        matches.sort(key=lambda m: m.start, reverse=True)
        for match in matches:
            masked_text = masked_text[:match.start] + match.masked + masked_text[match.end:]

        # 恢复正序
        matches.reverse()

        return masked_text, matches

    def _mask_string(
        self,
        text: str,
        keep_prefix: int = 3,
        keep_suffix: int = 4
    ) -> str:
        """
        脱敏字符串

        Args:
            text: 原始字符串
            keep_prefix: 保留前缀长度
            keep_suffix: 保留后缀长度

        Returns:
            脱敏后的字符串
        """
        if len(text) <= keep_prefix + keep_suffix:
            return self.pii_config.mask_char * len(text)

        prefix = text[:keep_prefix]
        suffix = text[-keep_suffix:]
        middle_len = len(text) - keep_prefix - keep_suffix
        middle = self.pii_config.mask_char * middle_len

        return prefix + middle + suffix

    def _mask_email(self, email: str) -> str:
        """脱敏邮箱"""
        if "@" not in email:
            return self._mask_string(email)

        local, domain = email.split("@", 1)
        if len(local) <= 2:
            masked_local = self.pii_config.mask_char * len(local)
        else:
            masked_local = local[0] + self.pii_config.mask_char * (len(local) - 2) + local[-1]

        return f"{masked_local}@{domain}"

    def _is_valid_id_card(self, id_card: str) -> bool:
        """简单校验身份证格式"""
        if len(id_card) != 18:
            return False

        # 检查前6位是否为有效地区码（简化：检查是否为数字）
        if not id_card[:6].isdigit():
            return False

        # 检查出生日期（简化：检查年份范围）
        year = id_card[6:10]
        if not year.isdigit():
            return False
        year_int = int(year)
        if year_int < 1900 or year_int > 2100:
            return False

        return True


# ==================== 工厂函数 ====================

def get_tcm_pii_middleware(**kwargs) -> TCMPIIMiddleware:
    """
    获取 PII 中间件实例

    Args:
        **kwargs: PIIConfig 参数

    Returns:
        TCMPIIMiddleware 实例
    """
    config = PIIConfig(**kwargs)
    return TCMPIIMiddleware(config)
