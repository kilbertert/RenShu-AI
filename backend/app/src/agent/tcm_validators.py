"""
TCM Validators
中医校验模块

包含配伍禁忌校验、辨证逻辑校验等
"""

from typing import Optional
from pydantic import BaseModel, Field

from .tcm_states import HerbCompatibilityResult


# ============== 辨证逻辑校验 ==============

# 证型与药性的对应关系
SYNDROME_HERB_NATURE_MAP = {
    # 寒证用温热药
    "风寒证": {"preferred": ["温", "热"], "avoid": ["寒", "凉"]},
    "寒湿证": {"preferred": ["温", "热"], "avoid": ["寒", "凉"]},
    "阳虚证": {"preferred": ["温", "热"], "avoid": ["寒", "凉"]},
    "脾胃虚寒": {"preferred": ["温"], "avoid": ["寒", "凉"]},

    # 热证用寒凉药
    "风热证": {"preferred": ["寒", "凉"], "avoid": ["温", "热"]},
    "湿热证": {"preferred": ["寒", "凉"], "avoid": ["温", "热"]},
    "阴虚证": {"preferred": ["凉", "平"], "avoid": ["热"]},
    "实热证": {"preferred": ["寒", "凉"], "avoid": ["温", "热"]},
    "血热证": {"preferred": ["寒", "凉"], "avoid": ["温", "热"]},

    # 虚证用补药
    "气虚证": {"preferred": ["平", "温"], "avoid": []},
    "血虚证": {"preferred": ["平", "温"], "avoid": []},
    "气血两虚": {"preferred": ["平", "温"], "avoid": []},

    # 实证用泻药
    "痰湿证": {"preferred": ["温"], "avoid": []},
    "血瘀证": {"preferred": ["温", "平"], "avoid": []},
    "气滞证": {"preferred": ["平", "温"], "avoid": []},
}

# 常见药材的药性
HERB_NATURE_DATA = {
    # 寒性药
    "黄连": "寒", "黄芩": "寒", "黄柏": "寒", "栀子": "寒",
    "金银花": "寒", "连翘": "寒", "板蓝根": "寒", "大青叶": "寒",
    "石膏": "寒", "知母": "寒", "天花粉": "寒", "芦根": "寒",
    "生地黄": "寒", "玄参": "寒", "牡丹皮": "寒", "赤芍": "寒",
    "大黄": "寒", "芒硝": "寒",

    # 凉性药
    "薄荷": "凉", "菊花": "凉", "桑叶": "凉", "蝉蜕": "凉",
    "决明子": "凉", "夏枯草": "凉", "青蒿": "凉",

    # 温性药
    "桂枝": "温", "生姜": "温", "紫苏": "温", "荆芥": "温",
    "防风": "温", "羌活": "温", "白芷": "温", "细辛": "温",
    "当归": "温", "川芎": "温", "白术": "温", "茯苓": "平",
    "陈皮": "温", "半夏": "温", "厚朴": "温", "苍术": "温",
    "黄芪": "温", "党参": "平", "白芍": "微寒",

    # 热性药
    "附子": "热", "干姜": "热", "肉桂": "热", "吴茱萸": "热",
    "花椒": "热", "胡椒": "热", "丁香": "温",

    # 平性药
    "甘草": "平", "大枣": "平", "山药": "平", "莲子": "平",
    "芡实": "平", "薏苡仁": "凉", "扁豆": "平",
}


class SyndromeHerbValidationResult(BaseModel):
    """辨证用药校验结果"""
    is_valid: bool = Field(description="是否符合辨证用药原则")
    syndrome: str = Field(description="证型")
    warnings: list[str] = Field(default_factory=list, description="警告信息")
    suggestions: list[str] = Field(default_factory=list, description="建议")
    herb_analysis: list[dict] = Field(default_factory=list, description="药材分析")


def validate_syndrome_herb_match(
    syndrome: str,
    herbs: list[str],
) -> SyndromeHerbValidationResult:
    """
    校验证型与药材是否匹配

    Args:
        syndrome: 证型名称
        herbs: 药材列表

    Returns:
        SyndromeHerbValidationResult: 校验结果
    """
    warnings = []
    suggestions = []
    herb_analysis = []

    # 获取证型对应的药性要求
    syndrome_rules = None
    for syn_name, rules in SYNDROME_HERB_NATURE_MAP.items():
        if syn_name in syndrome or syndrome in syn_name:
            syndrome_rules = rules
            break

    if not syndrome_rules:
        return SyndromeHerbValidationResult(
            is_valid=True,
            syndrome=syndrome,
            warnings=["未找到该证型的用药规则，请谨慎用药"],
            suggestions=[],
            herb_analysis=[],
        )

    preferred_natures = syndrome_rules["preferred"]
    avoid_natures = syndrome_rules["avoid"]

    # 分析每味药材
    for herb in herbs:
        nature = HERB_NATURE_DATA.get(herb)

        if nature is None:
            herb_analysis.append({
                "herb": herb,
                "nature": "未知",
                "status": "unknown",
                "message": f"未找到 {herb} 的药性信息",
            })
            continue

        if nature in avoid_natures:
            warnings.append(f"⚠️ {herb}（{nature}性）不宜用于{syndrome}，可能加重病情")
            herb_analysis.append({
                "herb": herb,
                "nature": nature,
                "status": "avoid",
                "message": f"{herb}药性为{nature}，不适合{syndrome}",
            })
        elif nature in preferred_natures:
            herb_analysis.append({
                "herb": herb,
                "nature": nature,
                "status": "preferred",
                "message": f"{herb}药性为{nature}，适合{syndrome}",
            })
        else:
            herb_analysis.append({
                "herb": herb,
                "nature": nature,
                "status": "neutral",
                "message": f"{herb}药性为{nature}",
            })

    # 生成建议
    if warnings:
        if "寒" in avoid_natures or "凉" in avoid_natures:
            suggestions.append("该证型宜用温热药，建议选用桂枝、干姜、附子等温阳散寒之品")
        if "温" in avoid_natures or "热" in avoid_natures:
            suggestions.append("该证型宜用寒凉药，建议选用黄连、黄芩、金银花等清热之品")

    return SyndromeHerbValidationResult(
        is_valid=len(warnings) == 0,
        syndrome=syndrome,
        warnings=warnings,
        suggestions=suggestions,
        herb_analysis=herb_analysis,
    )


# ============== 综合校验 ==============

class TCMValidationResult(BaseModel):
    """中医综合校验结果"""
    is_safe: bool = Field(description="是否安全")
    compatibility_result: Optional[HerbCompatibilityResult] = Field(
        None, description="配伍校验结果"
    )
    syndrome_herb_result: Optional[SyndromeHerbValidationResult] = Field(
        None, description="辨证用药校验结果"
    )
    all_warnings: list[str] = Field(default_factory=list, description="所有警告")
    all_suggestions: list[str] = Field(default_factory=list, description="所有建议")


def validate_tcm_prescription(
    herbs: list[str],
    syndrome: Optional[str] = None,
) -> TCMValidationResult:
    """
    综合校验中医处方

    Args:
        herbs: 药材列表
        syndrome: 证型（可选）

    Returns:
        TCMValidationResult: 综合校验结果
    """
    all_warnings = []
    all_suggestions = []

    # 1. 配伍禁忌校验
    # compat_result = check_herb_compatibility(herbs)
    compat_result={}
    compatibility_result = HerbCompatibilityResult(
        is_compatible=compat_result["is_compatible"],
        warnings=compat_result["warnings"],
        incompatible_pairs=compat_result["incompatible_pairs"],
        suggestions=[],
    )

    if not compat_result["is_compatible"]:
        all_warnings.extend(compat_result["warnings"])
        all_suggestions.append("请调整处方，避免配伍禁忌药物同用")

    # 2. 辨证用药校验（如果提供了证型）
    syndrome_herb_result = None
    if syndrome:
        syndrome_herb_result = validate_syndrome_herb_match(syndrome, herbs)
        all_warnings.extend(syndrome_herb_result.warnings)
        all_suggestions.extend(syndrome_herb_result.suggestions)

    # 综合判断是否安全
    is_safe = (
        compat_result["is_compatible"] and
        (syndrome_herb_result is None or syndrome_herb_result.is_valid)
    )

    return TCMValidationResult(
        is_safe=is_safe,
        compatibility_result=compatibility_result,
        syndrome_herb_result=syndrome_herb_result,
        all_warnings=all_warnings,
        all_suggestions=all_suggestions,
    )


# ============== 妊娠禁忌校验 ==============

PREGNANCY_CONTRAINDICATED_HERBS = {
    # 禁用药（绝对禁忌）
    "禁用": [
        "巴豆", "牵牛子", "大戟", "芫花", "甘遂", "商陆", "麝香",
        "三棱", "莪术", "水蛭", "虻虫", "斑蝥", "雄黄", "砒霜",
        "马钱子", "川乌", "草乌", "附子", "天南星", "半夏",
    ],
    # 慎用药（相对禁忌）
    "慎用": [
        "桃仁", "红花", "牛膝", "大黄", "芒硝", "番泻叶",
        "枳实", "枳壳", "厚朴", "肉桂", "干姜", "丁香",
        "通草", "瞿麦", "冬葵子", "薏苡仁",
    ],
}


class PregnancyValidationResult(BaseModel):
    """妊娠用药校验结果"""
    is_safe: bool = Field(description="是否安全")
    prohibited_herbs: list[str] = Field(default_factory=list, description="禁用药材")
    caution_herbs: list[str] = Field(default_factory=list, description="慎用药材")
    warnings: list[str] = Field(default_factory=list, description="警告信息")


def validate_pregnancy_safety(herbs: list[str]) -> PregnancyValidationResult:
    """
    校验妊娠期用药安全性

    Args:
        herbs: 药材列表

    Returns:
        PregnancyValidationResult: 校验结果
    """
    prohibited = []
    caution = []
    warnings = []

    for herb in herbs:
        if herb in PREGNANCY_CONTRAINDICATED_HERBS["禁用"]:
            prohibited.append(herb)
            warnings.append(f"🚫 {herb} 为妊娠禁用药，绝对禁止使用")
        elif herb in PREGNANCY_CONTRAINDICATED_HERBS["慎用"]:
            caution.append(herb)
            warnings.append(f"⚠️ {herb} 为妊娠慎用药，需权衡利弊")

    return PregnancyValidationResult(
        is_safe=len(prohibited) == 0,
        prohibited_herbs=prohibited,
        caution_herbs=caution,
        warnings=warnings,
    )
