"""ITCM 人工关系表字段映射测试。"""

from scripts.import_itcm_rels import extract_hi_relation_fields


def test_extract_hi_relation_fields_uses_english_ingredient_name():
    row = {
        "HERB+(CHN)": " 蝉蜕 ",
        "INGREDIENT(CHN)": "3,4-二羟基苯甲酸",
        "INGREDIENT(ENG)": " 3,4-Dihydroxybenzoic acid ",
        "related target (gene symbol)": " GAA ",
    }

    assert extract_hi_relation_fields(row) == (
        "蝉蜕",
        "3,4-Dihydroxybenzoic acid",
        "GAA",
    )


def test_extract_hi_relation_fields_normalizes_missing_values():
    assert extract_hi_relation_fields({}) == ("", "", "")
