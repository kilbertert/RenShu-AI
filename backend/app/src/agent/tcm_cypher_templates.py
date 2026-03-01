"""
TCM Diagnosis Cypher Templates
中医诊断 Cypher 查询模板

基于实际 Neo4j 图谱结构:
  节点: Symptom(symptom), Syndrome(syndrome), Formula(formula)
  关系: (Symptom)-[:INDICATES]->(Syndrome)-[:TREATS_WITH]->(Formula)

适用于 Light Search 模式（普通问诊场景）
"""

from typing import Literal, Optional, Any
from pydantic import BaseModel, Field


# ============== 核心诊断查询模板 ==============

DIAGNOSIS_CYPHER_TEMPLATES = {
    # ========== 症状 → 证候 查询 ==========
    "symptom_to_syndrome": {
        "single_symptom": {
            "description": "根据单个症状查询可能的证候",
            "cypher": """
                MATCH (s:Symptom)-[:INDICATES]->(syn:Syndrome)
                WHERE s.symptom = $symptom
                RETURN syn.syndrome AS 证候,
                       COUNT(*) AS 关联度
                ORDER BY 关联度 DESC
            """,
            "parameters": ["symptom"],
            "example": {"symptom": "便血"}
        },

        "multiple_symptoms": {
            "description": "根据多个症状查询证候（核心诊断查询）",
            "cypher": """
                MATCH (s:Symptom)-[:INDICATES]->(syn:Syndrome)
                WHERE s.symptom IN $symptoms
                WITH syn,
                     COUNT(DISTINCT s) AS match_count,
                     COLLECT(DISTINCT s.symptom) AS matched_symptoms
                ORDER BY match_count DESC
                LIMIT $limit
                RETURN syn.syndrome AS 证候,
                       matched_symptoms AS 匹配症状,
                       match_count AS 匹配数量,
                       round(match_count * 100.0 / $total_symptoms) AS 匹配度
            """,
            "parameters": ["symptoms", "limit", "total_symptoms"],
            "defaults": {"limit": 5},
            "example": {"symptoms": ["便血", "神疲乏力", "面色萎黄"], "limit": 5, "total_symptoms": 3}
        },

        "symptoms_with_formulas": {
            "description": "根据症状直接查询证候及推荐方剂（完整诊断路径）",
            "cypher": """
                MATCH (s:Symptom)-[:INDICATES]->(syn:Syndrome)
                WHERE s.symptom IN $symptoms
                WITH syn,
                     COUNT(DISTINCT s) AS match_count,
                     COLLECT(DISTINCT s.symptom) AS matched_symptoms
                ORDER BY match_count DESC
                LIMIT $syndrome_limit
                OPTIONAL MATCH (syn)-[:TREATS_WITH]->(f:Formula)
                RETURN syn.syndrome AS 证候,
                       matched_symptoms AS 匹配症状,
                       match_count AS 匹配数量,
                       COLLECT(DISTINCT f.formula)[0..$formula_limit] AS 推荐方剂
            """,
            "parameters": ["symptoms", "syndrome_limit", "formula_limit"],
            "defaults": {"syndrome_limit": 3, "formula_limit": 5},
            "example": {"symptoms": ["便血", "神疲乏力", "面色萎黄"], "syndrome_limit": 3, "formula_limit": 5}
        },
    },

    # ========== 证候 → 方剂 查询 ==========
    "syndrome_to_formula": {
        "by_syndrome_name": {
            "description": "根据证候名称查询推荐方剂",
            "cypher": """
                MATCH (syn:Syndrome)-[:TREATS_WITH]->(f:Formula)
                WHERE syn.syndrome = $syndrome
                RETURN f.formula AS 方剂,
                       syn.syndrome AS 证候
                LIMIT $limit
            """,
            "parameters": ["syndrome", "limit"],
            "defaults": {"limit": 10},
            "example": {"syndrome": "脾胃气虚证", "limit": 10}
        },

        "by_syndrome_keyword": {
            "description": "根据证候关键词模糊查询方剂",
            "cypher": """
                MATCH (syn:Syndrome)-[:TREATS_WITH]->(f:Formula)
                WHERE syn.syndrome CONTAINS $keyword
                RETURN syn.syndrome AS 证候,
                       COLLECT(DISTINCT f.formula)[0..$limit] AS 推荐方剂
            """,
            "parameters": ["keyword", "limit"],
            "defaults": {"limit": 5},
            "example": {"keyword": "气虚", "limit": 5}
        },

        "syndrome_formula_count": {
            "description": "查询证候对应的方剂数量",
            "cypher": """
                MATCH (syn:Syndrome)-[:TREATS_WITH]->(f:Formula)
                WHERE syn.syndrome = $syndrome
                RETURN syn.syndrome AS 证候,
                       COUNT(f) AS 方剂数量
            """,
            "parameters": ["syndrome"],
            "example": {"syndrome": "脾胃气虚证"}
        },
    },

    # ========== 方剂查询 ==========
    "formula": {
        "by_name": {
            "description": "根据方剂名称查询详情及适用证候",
            "cypher": """
                MATCH (f:Formula)<-[:TREATS_WITH]-(syn:Syndrome)
                WHERE f.formula = $formula
                RETURN f.formula AS 方剂,
                       COLLECT(DISTINCT syn.syndrome) AS 适用证候
            """,
            "parameters": ["formula"],
            "example": {"formula": "四君子汤"}
        },

        "by_keyword": {
            "description": "根据关键词模糊查询方剂",
            "cypher": """
                MATCH (f:Formula)
                WHERE f.formula CONTAINS $keyword
                OPTIONAL MATCH (syn:Syndrome)-[:TREATS_WITH]->(f)
                RETURN f.formula AS 方剂,
                       COLLECT(DISTINCT syn.syndrome) AS 适用证候
                LIMIT $limit
            """,
            "parameters": ["keyword", "limit"],
            "defaults": {"limit": 10},
            "example": {"keyword": "四君", "limit": 10}
        },

        "formulas_for_syndromes": {
            "description": "查询多个证候的共同推荐方剂",
            "cypher": """
                MATCH (syn:Syndrome)-[:TREATS_WITH]->(f:Formula)
                WHERE syn.syndrome IN $syndromes
                WITH f, COLLECT(DISTINCT syn.syndrome) AS matched_syndromes
                WHERE SIZE(matched_syndromes) >= $min_match
                RETURN f.formula AS 方剂,
                       matched_syndromes AS 适用证候,
                       SIZE(matched_syndromes) AS 匹配证候数
                ORDER BY 匹配证候数 DESC
                LIMIT $limit
            """,
            "parameters": ["syndromes", "min_match", "limit"],
            "defaults": {"min_match": 1, "limit": 10},
            "example": {"syndromes": ["脾胃气虚证", "气血两虚证"], "min_match": 1, "limit": 10}
        },
    },

    # ========== 证候查询 ==========
    "syndrome": {
        "by_name": {
            "description": "根据证候名称查询关联的症状和方剂",
            "cypher": """
                MATCH (syn:Syndrome {syndrome: $syndrome})
                OPTIONAL MATCH (s:Symptom)-[:INDICATES]->(syn)
                OPTIONAL MATCH (syn)-[:TREATS_WITH]->(f:Formula)
                RETURN syn.syndrome AS 证候,
                       COLLECT(DISTINCT s.symptom) AS 相关症状,
                       COLLECT(DISTINCT f.formula) AS 推荐方剂
            """,
            "parameters": ["syndrome"],
            "example": {"syndrome": "脾胃气虚证"}
        },

        "by_keyword": {
            "description": "根据关键词模糊查询证候",
            "cypher": """
                MATCH (syn:Syndrome)
                WHERE syn.syndrome CONTAINS $keyword
                RETURN syn.syndrome AS 证候
                LIMIT $limit
            """,
            "parameters": ["keyword", "limit"],
            "defaults": {"limit": 20},
            "example": {"keyword": "气虚", "limit": 20}
        },

        "all_syndromes": {
            "description": "查询所有证候列表",
            "cypher": """
                MATCH (syn:Syndrome)
                RETURN syn.syndrome AS 证候
                ORDER BY syn.syndrome
            """,
            "parameters": [],
            "example": {}
        },

        "syndrome_stats": {
            "description": "查询证候的统计信息（症状数、方剂数）",
            "cypher": """
                MATCH (syn:Syndrome)
                WHERE syn.syndrome = $syndrome
                OPTIONAL MATCH (s:Symptom)-[:INDICATES]->(syn)
                OPTIONAL MATCH (syn)-[:TREATS_WITH]->(f:Formula)
                RETURN syn.syndrome AS 证候,
                       COUNT(DISTINCT s) AS 症状数量,
                       COUNT(DISTINCT f) AS 方剂数量
            """,
            "parameters": ["syndrome"],
            "example": {"syndrome": "脾胃气虚证"}
        },
    },

    # ========== 症状查询 ==========
    "symptom": {
        "by_name": {
            "description": "根据症状名称查询关联的证候",
            "cypher": """
                MATCH (s:Symptom {symptom: $symptom})-[:INDICATES]->(syn:Syndrome)
                RETURN s.symptom AS 症状,
                       COLLECT(syn.syndrome) AS 可能证候
            """,
            "parameters": ["symptom"],
            "example": {"symptom": "便血"}
        },

        "by_keyword": {
            "description": "根据关键词模糊查询症状",
            "cypher": """
                MATCH (s:Symptom)
                WHERE s.symptom CONTAINS $keyword
                RETURN s.symptom AS 症状
                LIMIT $limit
            """,
            "parameters": ["keyword", "limit"],
            "defaults": {"limit": 20},
            "example": {"keyword": "痛", "limit": 20}
        },

        "all_symptoms": {
            "description": "查询所有症状列表",
            "cypher": """
                MATCH (s:Symptom)
                RETURN s.symptom AS 症状
                ORDER BY s.symptom
            """,
            "parameters": [],
            "example": {}
        },

        "symptoms_for_syndrome": {
            "description": "查询某证候的所有相关症状",
            "cypher": """
                MATCH (s:Symptom)-[:INDICATES]->(syn:Syndrome {syndrome: $syndrome})
                RETURN syn.syndrome AS 证候,
                       COLLECT(s.symptom) AS 相关症状
            """,
            "parameters": ["syndrome"],
            "example": {"syndrome": "脾胃气虚证"}
        },
    },

    # ========== 路径查询（完整诊断链路）==========
    "diagnosis_path": {
        "full_path": {
            "description": "查询完整的诊断路径：症状 → 证候 → 方剂",
            "cypher": """
                MATCH (s:Symptom)-[:INDICATES]->(syn:Syndrome)-[:TREATS_WITH]->(f:Formula)
                WHERE s.symptom IN $symptoms
                WITH syn,
                     COLLECT(DISTINCT s.symptom) AS matched_symptoms,
                     COLLECT(DISTINCT f.formula) AS formulas
                RETURN syn.syndrome AS 证候,
                       matched_symptoms AS 匹配症状,
                       SIZE(matched_symptoms) AS 匹配数,
                       formulas[0..5] AS 推荐方剂
                ORDER BY 匹配数 DESC
                LIMIT $limit
            """,
            "parameters": ["symptoms", "limit"],
            "defaults": {"limit": 5},
            "example": {"symptoms": ["便血", "神疲乏力", "面色萎黄"], "limit": 5}
        },

        "reverse_path": {
            "description": "从方剂反查诊断路径：方剂 → 证候 → 症状",
            "cypher": """
                MATCH (f:Formula)<-[:TREATS_WITH]-(syn:Syndrome)<-[:INDICATES]-(s:Symptom)
                WHERE f.formula = $formula
                RETURN f.formula AS 方剂,
                       syn.syndrome AS 适用证候,
                       COLLECT(DISTINCT s.symptom) AS 典型症状
            """,
            "parameters": ["formula"],
            "example": {"formula": "四君子汤"}
        },

        "symptom_formula_direct": {
            "description": "症状直达方剂（跳过证候展示）",
            "cypher": """
                MATCH (s:Symptom)-[:INDICATES]->(:Syndrome)-[:TREATS_WITH]->(f:Formula)
                WHERE s.symptom IN $symptoms
                WITH f, COUNT(DISTINCT s) AS symptom_match
                ORDER BY symptom_match DESC
                RETURN f.formula AS 方剂,
                       symptom_match AS 症状匹配数
                LIMIT $limit
            """,
            "parameters": ["symptoms", "limit"],
            "defaults": {"limit": 10},
            "example": {"symptoms": ["便血", "神疲乏力"], "limit": 10}
        },
    },

    # ========== 统计分析查询 ==========
    "statistics": {
        "graph_overview": {
            "description": "图谱概览统计",
            "cypher": """
                MATCH (s:Symptom) WITH COUNT(s) AS symptom_count
                MATCH (syn:Syndrome) WITH symptom_count, COUNT(syn) AS syndrome_count
                MATCH (f:Formula) WITH symptom_count, syndrome_count, COUNT(f) AS formula_count
                MATCH ()-[r:INDICATES]->() WITH symptom_count, syndrome_count, formula_count, COUNT(r) AS indicates_count
                MATCH ()-[t:TREATS_WITH]->()
                RETURN symptom_count AS 症状数量,
                       syndrome_count AS 证候数量,
                       formula_count AS 方剂数量,
                       indicates_count AS 症状证候关系数,
                       COUNT(t) AS 证候方剂关系数
            """,
            "parameters": [],
            "example": {}
        },

        "top_syndromes": {
            "description": "查询关联症状最多的证候",
            "cypher": """
                MATCH (s:Symptom)-[:INDICATES]->(syn:Syndrome)
                WITH syn, COUNT(s) AS symptom_count
                ORDER BY symptom_count DESC
                LIMIT $limit
                RETURN syn.syndrome AS 证候,
                       symptom_count AS 关联症状数
            """,
            "parameters": ["limit"],
            "defaults": {"limit": 10},
            "example": {"limit": 10}
        },

        "top_formulas": {
            "description": "查询关联证候最多的方剂",
            "cypher": """
                MATCH (syn:Syndrome)-[:TREATS_WITH]->(f:Formula)
                WITH f, COUNT(syn) AS syndrome_count
                ORDER BY syndrome_count DESC
                LIMIT $limit
                RETURN f.formula AS 方剂,
                       syndrome_count AS 关联证候数
            """,
            "parameters": ["limit"],
            "defaults": {"limit": 10},
            "example": {"limit": 10}
        },

        "orphan_check": {
            "description": "查询孤立节点（无关系的症状/证候/方剂）",
            "cypher": """
                MATCH (s:Symptom)
                WHERE NOT (s)-[:INDICATES]->()
                WITH COLLECT(s.symptom) AS orphan_symptoms
                MATCH (syn:Syndrome)
                WHERE NOT ()-[:INDICATES]->(syn) AND NOT (syn)-[:TREATS_WITH]->()
                WITH orphan_symptoms, COLLECT(syn.syndrome) AS orphan_syndromes
                MATCH (f:Formula)
                WHERE NOT ()-[:TREATS_WITH]->(f)
                RETURN orphan_symptoms AS 孤立症状,
                       orphan_syndromes AS 孤立证候,
                       COLLECT(f.formula) AS 孤立方剂
            """,
            "parameters": [],
            "example": {}
        },
    },
}


# ============== 工具 Schema 定义 ==============

class DiagnosisCypherTool(BaseModel):
    """诊断 Cypher 查询工具"""
    category: Literal[
        "symptom_to_syndrome",  # 症状 → 证候
        "syndrome_to_formula",  # 证候 → 方剂
        "formula",              # 方剂查询
        "syndrome",             # 证候查询
        "symptom",              # 症状查询
        "diagnosis_path",       # 完整诊断路径
        "statistics",           # 统计分析
    ] = Field(description="查询类别")
    template_name: str = Field(description="模板名称")
    parameters: dict = Field(default_factory=dict, description="查询参数")


# ============== 辅助函数 ==============

def get_diagnosis_template(category: str, template_name: str) -> dict | None:
    """
    获取诊断 Cypher 模板

    Args:
        category: 模板类别
        template_name: 模板名称

    Returns:
        dict: 模板信息
    """
    if category in DIAGNOSIS_CYPHER_TEMPLATES:
        if template_name in DIAGNOSIS_CYPHER_TEMPLATES[category]:
            return DIAGNOSIS_CYPHER_TEMPLATES[category][template_name]
    return None


def list_diagnosis_templates() -> dict:
    """
    列出所有诊断模板

    Returns:
        dict: 按类别组织的模板列表
    """
    result = {}
    for category, templates in DIAGNOSIS_CYPHER_TEMPLATES.items():
        result[category] = {
            name: {
                "description": info["description"],
                "parameters": info["parameters"],
                "example": info.get("example", {})
            }
            for name, info in templates.items()
        }
    return result


def execute_diagnosis_query(
    driver,
    category: str,
    template_name: str,
    parameters: dict
) -> list[dict]:
    """
    执行诊断查询

    Args:
        driver: Neo4j driver
        category: 模板类别
        template_name: 模板名称
        parameters: 查询参数

    Returns:
        list: 查询结果
    """
    template = get_diagnosis_template(category, template_name)
    if not template:
        raise ValueError(f"Template not found: {category}/{template_name}")

    # 合并默认参数
    final_params = {**template.get("defaults", {}), **parameters}

    # 验证必需参数
    for param in template["parameters"]:
        if param not in final_params and param not in template.get("defaults", {}):
            raise ValueError(f"Missing required parameter: {param}")

    # 执行查询
    with driver.session() as session:
        result = session.run(template["cypher"], final_params)
        return [dict(record) for record in result]


# ============== 快捷诊断函数 ==============

def diagnose_by_symptoms(driver, symptoms: list[str], limit: int = 5) -> list[dict]:
    """
    根据症状列表进行诊断

    Args:
        driver: Neo4j driver
        symptoms: 症状列表
        limit: 返回结果数量

    Returns:
        list: 诊断结果（证候 + 推荐方剂）
    """
    return execute_diagnosis_query(
        driver,
        "symptom_to_syndrome",
        "symptoms_with_formulas",
        {
            "symptoms": symptoms,
            "syndrome_limit": limit,
            "formula_limit": 5,
        }
    )


def get_syndrome_info(driver, syndrome: str) -> dict | None:
    """
    获取证候详细信息

    Args:
        driver: Neo4j driver
        syndrome: 证候名称

    Returns:
        dict: 证候信息
    """
    results = execute_diagnosis_query(
        driver,
        "syndrome",
        "by_name",
        {"syndrome": syndrome}
    )
    return results[0] if results else None


def get_formula_info(driver, formula: str) -> dict | None:
    """
    获取方剂详细信息

    Args:
        driver: Neo4j driver
        formula: 方剂名称

    Returns:
        dict: 方剂信息
    """
    results = execute_diagnosis_query(
        driver,
        "formula",
        "by_name",
        {"formula": formula}
    )
    return results[0] if results else None


def search_symptoms(driver, keyword: str, limit: int = 20) -> list[str]:
    """
    搜索症状

    Args:
        driver: Neo4j driver
        keyword: 关键词
        limit: 返回数量

    Returns:
        list: 症状列表
    """
    results = execute_diagnosis_query(
        driver,
        "symptom",
        "by_keyword",
        {"keyword": keyword, "limit": limit}
    )
    return [r["症状"] for r in results]


def get_diagnosis_path(driver, symptoms: list[str], limit: int = 5) -> list[dict]:
    """
    获取完整诊断路径

    Args:
        driver: Neo4j driver
        symptoms: 症状列表
        limit: 返回数量

    Returns:
        list: 诊断路径（症状 → 证候 → 方剂）
    """
    return execute_diagnosis_query(
        driver,
        "diagnosis_path",
        "full_path",
        {"symptoms": symptoms, "limit": limit}
    )


# ============== 模板索引（供 LLM 选择使用）==============

TEMPLATE_INDEX = """
# 中医诊断 Cypher 查询模板索引

## 1. 症状 → 证候 查询 (symptom_to_syndrome)
- single_symptom: 根据单个症状查询可能的证候
- multiple_symptoms: 根据多个症状查询证候（核心诊断查询）
- symptoms_with_formulas: 根据症状直接查询证候及推荐方剂（完整诊断路径）

## 2. 证候 → 方剂 查询 (syndrome_to_formula)
- by_syndrome_name: 根据证候名称查询推荐方剂
- by_syndrome_keyword: 根据证候关键词模糊查询方剂
- syndrome_formula_count: 查询证候对应的方剂数量

## 3. 方剂查询 (formula)
- by_name: 根据方剂名称查询详情及适用证候
- by_keyword: 根据关键词模糊查询方剂
- formulas_for_syndromes: 查询多个证候的共同推荐方剂

## 4. 证候查询 (syndrome)
- by_name: 根据证候名称查询关联的症状和方剂
- by_keyword: 根据关键词模糊查询证候
- all_syndromes: 查询所有证候列表
- syndrome_stats: 查询证候的统计信息

## 5. 症状查询 (symptom)
- by_name: 根据症状名称查询关联的证候
- by_keyword: 根据关键词模糊查询症状
- all_symptoms: 查询所有症状列表
- symptoms_for_syndrome: 查询某证候的所有相关症状

## 6. 完整诊断路径 (diagnosis_path)
- full_path: 查询完整的诊断路径：症状 → 证候 → 方剂
- reverse_path: 从方剂反查诊断路径：方剂 → 证候 → 症状
- symptom_formula_direct: 症状直达方剂（跳过证候展示）

## 7. 统计分析 (statistics)
- graph_overview: 图谱概览统计
- top_syndromes: 查询关联症状最多的证候
- top_formulas: 查询关联证候最多的方剂
- orphan_check: 查询孤立节点
"""


if __name__ == "__main__":
    # 测试：打印所有模板
    import json
    templates = list_diagnosis_templates()
    print(json.dumps(templates, ensure_ascii=False, indent=2))
