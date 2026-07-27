"""心理危机与自伤风险的确定性前置检测。"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


class CrisisDetection(BaseModel):
    """心理危机检测结果，供主图守卫和中断恢复路径共同使用。"""

    is_crisis: bool = False
    matched_text: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


_CRISIS_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"不想活(?:了|下去)?",
        r"活着(?:也)?没(?:有)?意思",
        r"想(?:去)?死",
        r"结束(?:掉)?(?:自己|生命|这一切)",
        r"轻生",
        r"自杀",
        r"自伤",
        r"伤害自己",
        r"准备告别",
        r"最后的告别",
    )
)

_DOUBLE_NEGATION_MARKERS = (
    "不是没有",
    "并非没有",
    "不能排除",
    "不排除",
    "不是不",
    "并非不",
)


def _is_negated_crisis(text: str, start: int) -> bool:
    """过滤明确否认，同时保留双重否定和不确定风险表达。"""

    prefix = re.sub(r"\s+", "", text[max(0, start - 20):start])
    if any(marker in prefix for marker in _DOUBLE_NEGATION_MARKERS):
        return False

    return bool(re.search(
        r"(?:没有|没|无|否认|未有|从未|从没|并无|不伴)"
        r"(?:任何)?(?:过)?(?:想过|想要|想|出现|存在|产生|有)?$",
        prefix,
    )) or bool(re.search(r"(?:并不|不)(?:想|会|要)?$", prefix))


def detect_psychological_crisis(text: str) -> CrisisDetection:
    """检测当前文本是否包含未被否定的自杀、自伤或轻生意图。"""

    original = str(text or "").strip()
    if not original:
        return CrisisDetection()

    for pattern in _CRISIS_PATTERNS:
        match = pattern.search(original)
        if match and not _is_negated_crisis(original, match.start()):
            return CrisisDetection(
                is_crisis=True,
                matched_text=match.group(0),
                confidence=1.0,
            )
    return CrisisDetection()


def psychological_crisis_response() -> str:
    """返回不继续辨证或给药的心理危机干预话术。"""

    return """⚠️ **请先确保人身安全**

听起来您现在非常痛苦。此刻最重要的不是辨证或用中药，而是让您马上获得真人帮助：

1. **不要独处**，立即联系一位可信任的家人、朋友、同事或老师，请对方现在陪着您。
2. 暂时远离药物、刀具、高处、车辆等可能伤害自己的物品或环境。
3. 如果您已经准备实施、手边有工具，或已经伤害了自己，请立即拨打 **120 或 110**，或让身边的人陪同前往最近的医院急诊。
4. 也可以拨打全国统一心理援助热线 **12356**，直接告诉接线人员：“我现在有伤害自己的想法，需要立即帮助。”

我不能为这种情况提供中药建议或继续中医辨证。请先回复身边的人并求助；如果可以，只需告诉我：**您现在是否独处、是否已经准备了具体方式或工具？**"""
