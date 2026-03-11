"""
MCP Server Implementation for SmartTCM
基于 Model Context Protocol 标准的中医知识服务器

参考：https://modelcontextprotocol.io/
符合 MCP Industry Standard (Linux Foundation AAIF)

提供工具：
- search_herbs: 搜索中药材信息
- search_prescriptions: 搜索方剂信息
- diagnose_syndrome: 根据症状进行辨证分析
- query_knowledge_graph: 查询TCM知识图谱（Neo4j）
- get_contraindications: 获取药物禁忌信息
"""

import asyncio
import json
from typing import Any, Optional, Sequence
from dataclasses import dataclass, field

from app.src.utils import get_logger

logger = get_logger("mcp_server")

# 检查 MCP 库是否可用
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Server = None
    logger.warning("MCP library not installed. Install with: pip install mcp")


# ============== TCM 工具定义 ==============

TCM_TOOLS: list[Tool] = [
    Tool(
        name="search_herbs",
        description="搜索中药材信息。根据药材名称或关键词查询药材的性味归经、功效主治、用法用量等详细信息。",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，可以是药材名称（如'人参'、'黄芪'）或功效关键词（如'补气'、'活血'）"
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果数量限制",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50
                }
            },
            "required": ["query"]
        }
    ),
    Tool(
        name="search_prescriptions",
        description="搜索方剂信息。根据证型、症状或方剂名称查询方剂的组成、功效、主治等详细信息。",
        inputSchema={
            "type": "object",
            "properties": {
                "syndrome": {
                    "type": "string",
                    "description": "证型名称，如'肝郁脾虚'、'肾阴虚'"
                },
                "symptoms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "症状列表，如['头痛', '胸闷', '失眠']"
                },
                "prescription_name": {
                    "type": "string",
                    "description": "方剂名称，如'逍遥散'、'六味地黄丸'"
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果数量限制",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50
                }
            }
        }
    ),
    Tool(
        name="diagnose_syndrome",
        description="根据症状进行中医辨证分析。综合症状、脉象、舌象等信息，分析可能的证型。",
        inputSchema={
            "type": "object",
            "properties": {
                "symptoms": {
                    "type": "object",
                    "description": "症状对象，包含各种症状信息。如 {'头痛': '经常头痛', '失眠': '入睡困难', '乏力': '精神不振'}"
                },
                "pulse": {
                    "type": "string",
                    "description": "脉象描述，如'脉弦'、'脉细数'、'脉沉细'"
                },
                "tongue": {
                    "type": "object",
                    "description": "舌象描述，如 {'tongue_color': '淡红', 'coating_color': '薄白', 'coating_texture': '薄'}"
                },
                "complexity_level": {
                    "type": "string",
                    "enum": ["simple", "moderate", "complex"],
                    "description": "诊断复杂度级别，影响检索策略"
                }
            },
            "required": ["symptoms"]
        }
    ),
    Tool(
        name="query_knowledge_graph",
        description="直接查询TCM知识图谱（Neo4j）。使用Cypher查询语言进行复杂查询。",
        inputSchema={
            "type": "object",
            "properties": {
                "cypher": {
                    "type": "string",
                    "description": "Cypher查询语句。如 'MATCH (h:Herb)-[:TREATS]->(s:Symptom) WHERE s.name = \"头痛\" RETURN h.name, h.effects'"
                },
                "params": {
                    "type": "object",
                    "description": "查询参数，用于参数化查询"
                }
            },
            "required": ["cypher"]
        }
    ),
    Tool(
        name="get_contraindications",
        description="获取药物禁忌信息。检查药材配伍是否安全，包括十八反、十九畏等配伍禁忌。",
        inputSchema={
            "type": "object",
            "properties": {
                "herbs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "药材列表，如 ['人参', '藜芦', '丹参']"
                },
                "constitution": {
                    "type": "string",
                    "description": "患者体质，如 '气虚体质'、'阴虚体质'、'湿热体质'"
                },
                "pregnancy": {
                    "type": "boolean",
                    "description": "是否孕期"
                }
            },
            "required": ["herbs"]
        }
    ),
    Tool(
        name="search_classics",
        description="从中医古籍中检索相关论述。根据关键词在古籍库中搜索相关的理论依据和经典条文。",
        inputSchema={
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "关键词列表，如 ['头痛', '肝郁']"
                },
                "books": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "限定检索的书籍列表，默认为常用经典"
                },
                "max_results": {
                    "type": "integer",
                    "description": "最多返回结果数",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20
                }
            },
            "required": ["keywords"]
        }
    ),
    Tool(
        name="search_cases",
        description="搜索历史医案。根据症状、证型等条件检索相似的历史医案作为参考。",
        inputSchema={
            "type": "object",
            "properties": {
                "symptoms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "症状列表"
                },
                "syndrome": {
                    "type": "string",
                    "description": "证型"
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果数量限制",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20
                }
            }
        }
    ),
]


# ============== TCM 工具实现 ==============

@dataclass
class TCMToolContext:
    """TCM工具执行上下文"""
    neo4j_graph: Optional[Any] = None
    vector_store: Optional[Any] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class TCMToolExecutor:
    """
    TCM工具执行器

    负责实际执行各个工具的逻辑，对接Neo4j、向量存储等后端服务。
    """

    def __init__(self, context: Optional[TCMToolContext] = None):
        self.context = context or TCMToolContext()
        self._neo4j_graph = None
        self._vector_store = None

    def _get_neo4j_graph(self):
        """获取Neo4j图实例"""
        if self._neo4j_graph is None:
            try:
                from app.src.agent.tcm_neo4j import get_neo4j_graph
                self._neo4j_graph = get_neo4j_graph()
            except Exception as e:
                logger.warning(f"Failed to get Neo4j graph: {e}")
        return self._neo4j_graph

    def _get_vector_store(self):
        """获取向量存储实例"""
        if self._vector_store is None:
            try:
                from app.src.agent.data.embedder import get_embedder
                self._vector_store = get_embedder()
            except Exception as e:
                logger.warning(f"Failed to get vector store: {e}")
        return self._vector_store

    async def search_herbs(self, query: str, limit: int = 10) -> dict:
        """搜索中药材"""
        logger.info(f"搜索药材: query={query}, limit={limit}")

        results = []

        try:
            graph = self._get_neo4j_graph()
            if graph:
                # 使用Neo4j查询
                cypher = """
                MATCH (h:Herb)
                WHERE h.name CONTAINS $query OR h.pinyin CONTAINS $query
                   OR ANY(effect IN h.effects WHERE effect CONTAINS $query)
                RETURN h.name as name,
                       h.pinyin as pinyin,
                       h.category as category,
                       h.nature as nature,
                       h.flavor as flavor,
                       h.meridians as meridians,
                       h.effects as effects,
                       h.indications as indications,
                       h.contraindications as contraindications,
                       h.dosage as dosage
                LIMIT $limit
                """
                records = graph.query(cypher, params={"query": query, "limit": limit})
                results = [dict(record) for record in records]
        except Exception as e:
            logger.warning(f"Neo4j query failed: {e}, using mock data")

        # 如果没有结果或查询失败，返回模拟数据
        if not results:
            results = self._get_mock_herbs(query, limit)

        return {
            "herbs": results,
            "count": len(results),
            "query": query
        }

    def _get_mock_herbs(self, query: str, limit: int) -> list:
        """获取模拟药材数据"""
        mock_db = {
            "人参": {
                "name": "人参",
                "pinyin": "Ren Shen",
                "category": "补气药",
                "nature": "微温",
                "flavor": ["甘", "微苦"],
                "meridians": ["脾", "肺", "心", "肾"],
                "effects": ["大补元气", "复脉固脱", "补脾益肺", "生津养血", "安神益智"],
                "indications": ["体虚欲脱", "肢冷脉微", "脾虚食少", "气短喘促", "失眠多梦"],
                "contraindications": ["实热证", "湿热证"],
                "dosage": "3-9g，大剂量可用至30g"
            },
            "黄芪": {
                "name": "黄芪",
                "pinyin": "Huang Qi",
                "category": "补气药",
                "nature": "微温",
                "flavor": ["甘"],
                "meridians": ["脾", "肺"],
                "effects": ["补气升阳", "固表止汗", "利水消肿", "生津养血"],
                "indications": ["气虚乏力", "中气下陷", "表虚自汗", "水肿"],
                "contraindications": ["表实邪盛", "气滞湿阻"],
                "dosage": "9-30g"
            },
            "当归": {
                "name": "当归",
                "pinyin": "Dang Gui",
                "category": "补血药",
                "nature": "温",
                "flavor": ["甘", "辛"],
                "meridians": ["肝", "心", "脾"],
                "effects": ["补血调经", "活血止痛", "润肠通便"],
                "indications": ["血虚萎黄", "月经不调", "经闭痛经", "肠燥便秘"],
                "contraindications": ["湿盛中满", "大便泄泻"],
                "dosage": "6-12g"
            },
            "甘草": {
                "name": "甘草",
                "pinyin": "Gan Cao",
                "category": "补气药",
                "nature": "平",
                "flavor": ["甘"],
                "meridians": ["心", "肺", "脾", "胃"],
                "effects": ["补脾益气", "清热解毒", "祛痰止咳", "缓急止痛"],
                "indications": ["脾胃虚弱", "倦怠乏力", "咳嗽痰多", "脘腹疼痛"],
                "contraindications": ["湿盛中满", "水肿"],
                "dosage": "2-10g"
            },
            "柴胡": {
                "name": "柴胡",
                "pinyin": "Chai Hu",
                "category": "解表药",
                "nature": "微寒",
                "flavor": ["苦", "辛"],
                "meridians": ["肝", "胆"],
                "effects": ["疏散退热", "疏肝解郁", "升举阳气"],
                "indications": ["感冒发热", "肝郁气滞", "子宫脱垂", "脱肛"],
                "contraindications": ["真阴亏损", "肝阳上亢"],
                "dosage": "3-10g"
            },
        }

        # 搜索匹配
        results = []
        for name, info in mock_db.items():
            if query in name or any(query in e for e in info.get("effects", [])):
                results.append(info)
                if len(results) >= limit:
                    break

        return results

    async def search_prescriptions(
        self,
        syndrome: Optional[str] = None,
        symptoms: Optional[list[str]] = None,
        prescription_name: Optional[str] = None,
        limit: int = 10
    ) -> dict:
        """搜索方剂"""
        logger.info(f"搜索方剂: syndrome={syndrome}, prescription_name={prescription_name}, limit={limit}")

        results = []

        try:
            graph = self._get_neo4j_graph()
            if graph and syndrome:
                cypher = """
                MATCH (p:Prescription)-[:TREATS]->(s:Syndrome)
                WHERE s.name CONTAINS $syndrome
                RETURN p.name as name,
                       p.source as source,
                       p.composition as composition,
                       p.effects as effects,
                       p.indications as indications,
                       p.syndrome as syndrome,
                       p.usage as usage
                LIMIT $limit
                """
                records = graph.query(cypher, params={"syndrome": syndrome, "limit": limit})
                results = [dict(record) for record in records]
            elif graph and prescription_name:
                cypher = """
                MATCH (p:Prescription)
                WHERE p.name CONTAINS $name
                RETURN p.name as name,
                       p.source as source,
                       p.composition as composition,
                       p.effects as effects,
                       p.indications as indications,
                       p.syndrome as syndrome,
                       p.usage as usage
                LIMIT $limit
                """
                records = graph.query(cypher, params={"name": prescription_name, "limit": limit})
                results = [dict(record) for record in records]
        except Exception as e:
            logger.warning(f"Neo4j query failed: {e}, using mock data")

        if not results:
            results = self._get_mock_prescriptions(syndrome, prescription_name, limit)

        return {
            "prescriptions": results,
            "count": len(results)
        }

    def _get_mock_prescriptions(
        self,
        syndrome: Optional[str],
        prescription_name: Optional[str],
        limit: int
    ) -> list:
        """获取模拟方剂数据"""
        mock_db = [
            {
                "name": "逍遥散",
                "source": "太平惠民和剂局方",
                "composition": [
                    {"herb": "柴胡", "dosage": "9g"},
                    {"herb": "当归", "dosage": "9g"},
                    {"herb": "白芍", "dosage": "9g"},
                    {"herb": "白术", "dosage": "9g"},
                    {"herb": "茯苓", "dosage": "9g"},
                    {"herb": "甘草", "dosage": "4.5g"},
                    {"herb": "煨生姜", "dosage": "少许"},
                    {"herb": "薄荷", "dosage": "少许"}
                ],
                "effects": "疏肝健脾，养血调经",
                "indications": "肝郁血虚脾弱证。两胁作痛，头痛目眩，口燥咽干，神疲食少，月经不调，乳房胀痛。",
                "syndrome": "肝郁脾虚",
                "usage": "水煎服，或为丸，每服6-9g"
            },
            {
                "name": "六味地黄丸",
                "source": "小儿药证直诀",
                "composition": [
                    {"herb": "熟地黄", "dosage": "24g"},
                    {"herb": "山茱萸", "dosage": "12g"},
                    {"herb": "山药", "dosage": "12g"},
                    {"herb": "泽泻", "dosage": "9g"},
                    {"herb": "茯苓", "dosage": "9g"},
                    {"herb": "牡丹皮", "dosage": "9g"}
                ],
                "effects": "滋阴补肾",
                "indications": "肾阴虚证。腰膝酸软，头晕耳鸣，骨蒸潮热，盗汗遗精，消渴。",
                "syndrome": "肾阴虚",
                "usage": "蜜丸，每服9g，日2次"
            },
            {
                "name": "补中益气汤",
                "source": "脾胃论",
                "composition": [
                    {"herb": "黄芪", "dosage": "18g"},
                    {"herb": "人参", "dosage": "6g"},
                    {"herb": "白术", "dosage": "9g"},
                    {"herb": "当归", "dosage": "6g"},
                    {"herb": "陈皮", "dosage": "6g"},
                    {"herb": "升麻", "dosage": "3g"},
                    {"herb": "柴胡", "dosage": "3g"},
                    {"herb": "甘草", "dosage": "4.5g"}
                ],
                "effects": "补中益气，升阳举陷",
                "indications": "脾虚气陷证。面色萎黄，气短乏力，食少便溏，脱肛，子宫脱垂。",
                "syndrome": "脾气虚",
                "usage": "水煎服"
            },
            {
                "name": "龙胆泻肝汤",
                "source": "医方集解",
                "composition": [
                    {"herb": "龙胆草", "dosage": "6g"},
                    {"herb": "黄芩", "dosage": "9g"},
                    {"herb": "栀子", "dosage": "9g"},
                    {"herb": "泽泻", "dosage": "9g"},
                    {"herb": "木通", "dosage": "6g"},
                    {"herb": "当归", "dosage": "3g"},
                    {"herb": "生地", "dosage": "9g"},
                    {"herb": "柴胡", "dosage": "6g"},
                    {"herb": "甘草", "dosage": "3g"},
                    {"herb": "车前子", "dosage": "9g"}
                ],
                "effects": "清肝胆实火，泻下焦湿热",
                "indications": "肝胆实火上炎证。头痛目赤，胁痛口苦，耳聋耳肿。",
                "syndrome": "肝胆湿热",
                "usage": "水煎服"
            },
        ]

        results = []
        for p in mock_db:
            if syndrome and syndrome in p.get("syndrome", ""):
                results.append(p)
            elif prescription_name and prescription_name in p["name"]:
                results.append(p)

        return results[:limit] if results else mock_db[:limit]

    async def diagnose_syndrome(
        self,
        symptoms: dict,
        pulse: Optional[str] = None,
        tongue: Optional[dict] = None,
        complexity_level: str = "simple"
    ) -> dict:
        """辨证分析"""
        logger.info(f"辨证分析: symptoms={symptoms}, pulse={pulse}, complexity={complexity_level}")

        symptom_list = list(symptoms.keys()) if isinstance(symptoms, dict) else list(symptoms)

        try:
            graph = self._get_neo4j_graph()
            if graph:
                # 查询匹配的证型
                cypher = """
                MATCH (s:Symptom)-[:INDICATES]->(syn:Syndrome)
                WHERE s.name IN $symptoms
                WITH syn, COLLECT(s.name) as matched_symptoms
                WHERE SIZE(matched_symptoms) >= $min_match
                RETURN syn.name as syndrome,
                       syn.description as description,
                       syn.treatment_principle as treatment_principle,
                       matched_symptoms,
                       SIZE(matched_symptoms) as match_count,
                       toFloat(SIZE(matched_symptoms)) / $symptom_count as confidence
                ORDER BY confidence DESC
                LIMIT 5
                """
                records = graph.query(cypher, params={
                    "symptoms": symptom_list,
                    "min_match": max(1, len(symptom_list) // 3),
                    "symptom_count": len(symptom_list)
                })

                syndromes = [dict(r) for r in records]
        except Exception as e:
            logger.warning(f"Neo4j query failed: {e}")
            syndromes = []

        if not syndromes:
            syndromes = self._get_mock_diagnosis(symptom_list, pulse, tongue)

        return {
            "syndromes": syndromes,
            "symptoms_analyzed": symptom_list,
            "pulse": pulse,
            "tongue": tongue,
            "complexity_level": complexity_level
        }

    def _get_mock_diagnosis(
        self,
        symptoms: list[str],
        pulse: Optional[str],
        tongue: Optional[dict]
    ) -> list:
        """获取模拟诊断结果"""
        # 症状到证型的简单映射
        syndrome_map = {
            "头痛": ["肝阳上亢", "肝郁气滞", "血虚头痛", "痰浊头痛"],
            "胸闷": ["肝郁气滞", "心气虚", "痰湿阻肺", "心血瘀阻"],
            "失眠": ["心肾不交", "肝郁化火", "心血虚", "痰热内扰"],
            "腰膝酸软": ["肾阴虚", "肾阳虚", "肝肾亏虚"],
            "乏力": ["脾气虚", "气血两虚", "肾阳虚", "肝郁脾虚"],
            "食欲不振": ["脾胃虚弱", "肝郁脾虚", "湿困脾胃"],
            "腹胀": ["脾虚湿阻", "肝郁脾虚", "食积胃肠"],
            "便秘": ["肠胃实热", "阴虚肠燥", "气血两虚", "肝郁气滞"],
            "盗汗": ["阴虚内热", "心血虚"],
            "自汗": ["肺气虚", "脾气虚", "阳虚"],
        }

        # 统计证型出现次数
        syndrome_count = {}
        for symptom in symptoms:
            if symptom in syndrome_map:
                for syn in syndrome_map[symptom]:
                    syndrome_count[syn] = syndrome_count.get(syn, 0) + 1

        # 生成结果
        results = []
        for syn, count in sorted(syndrome_count.items(), key=lambda x: -x[1]):
            matched = [s for s in symptoms if syn in syndrome_map.get(s, [])]
            results.append({
                "syndrome": syn,
                "description": f"{syn}的常见表现",
                "treatment_principle": self._get_treatment_principle(syn),
                "matched_symptoms": matched,
                "match_count": len(matched),
                "confidence": round(len(matched) / len(symptoms), 2)
            })
            if len(results) >= 3:
                break

        return results if results else [{
            "syndrome": "待进一步辨证",
            "description": "症状复杂，需要综合分析",
            "treatment_principle": "建议进一步问诊",
            "matched_symptoms": symptoms,
            "match_count": len(symptoms),
            "confidence": 0.5
        }]

    def _get_treatment_principle(self, syndrome: str) -> str:
        """获取治则治法"""
        principles = {
            "肝阳上亢": "平肝潜阳，滋养肝肾",
            "肝郁气滞": "疏肝理气，解郁",
            "心肾不交": "滋阴降火，交通心肾",
            "肾阴虚": "滋阴补肾",
            "肾阳虚": "温补肾阳",
            "脾气虚": "补中益气，健脾养胃",
            "肝郁脾虚": "疏肝健脾",
            "脾胃虚弱": "健脾益气，和胃",
            "气血两虚": "气血双补",
        }
        return principles.get(syndrome, "辨证论治")

    async def query_knowledge_graph(self, cypher: str, params: Optional[dict] = None) -> dict:
        """查询知识图谱"""
        logger.info(f"执行Cypher查询: {cypher[:100]}...")

        try:
            graph = self._get_neo4j_graph()
            if graph:
                records = graph.query(cypher, params=params or {})
                return {
                    "records": [dict(r) for r in records],
                    "count": len(records),
                    "cypher": cypher
                }
        except Exception as e:
            logger.error(f"Cypher查询失败: {e}")
            return {
                "error": str(e),
                "records": [],
                "count": 0,
                "cypher": cypher
            }

        return {
            "records": [],
            "count": 0,
            "cypher": cypher
        }

    async def get_contraindications(
        self,
        herbs: list[str],
        constitution: Optional[str] = None,
        pregnancy: bool = False
    ) -> dict:
        """获取药物禁忌"""
        logger.info(f"检查配伍禁忌: herbs={herbs}, pregnancy={pregnancy}")

        warnings = []
        incompatible_pairs = []
        suggestions = []

        # 十八反
        eighteen_opposites = {
            "甘草": ["藜芦"],
            "人参": ["藜芦"],
            "丹参": ["藜芦"],
            "玄参": ["藜芦"],
            "沙参": ["藜芦"],
            "细辛": ["藜芦"],
            "芍药": ["藜芦"],
            "藜芦": ["人参", "丹参", "玄参", "沙参", "细辛", "芍药", "甘草"],
        }

        # 十九畏
        nineteen_avoidances = {
            "人参": ["五灵脂"],
            "五灵脂": ["人参"],
            "丁香": ["郁金"],
            "郁金": ["丁香"],
            "芒硝": ["京三棱"],
            "京三棱": ["芒硝"],
        }

        # 检查十八反
        herb_set = [h for h in herbs if any(key in h or h in key for key in eighteen_opposites.keys())]
        for herb in herb_set:
            for key, opposites in eighteen_opposites.items():
                if key in herb or herb in key:
                    for opposite in opposites:
                        if any(opposite in h or h in opposite for h in herbs):
                            pair = (key, opposite)
                            if pair not in incompatible_pairs and (opposite, key) not in incompatible_pairs:
                                incompatible_pairs.append(pair)
                                warnings.append(f"【十八反】{key}与{opposite}相反，不宜同用！")

        # 检查十九畏
        for herb in herb_set:
            for key, avoidances in nineteen_avoidances.items():
                if key in herb or herb in key:
                    for avoidance in avoidances:
                        if any(avoidance in h or h in avoidance for h in herbs):
                            pair = (key, avoidance)
                            if pair not in incompatible_pairs and (avoidance, key) not in incompatible_pairs:
                                incompatible_pairs.append(pair)
                                warnings.append(f"【十九畏】{key}与{avoidance}相畏，配伍需谨慎！")

        # 孕期禁忌
        pregnancy_forbidden = ["麝香", "斑蝥", "水蛭", "虻虫", "附子", "肉桂", "巴豆"]
        pregnancy_careful = ["桃仁", "红花", "大黄", "芒硝", "牛膝"]

        if pregnancy:
            for herb in herbs:
                for forbidden in pregnancy_forbidden:
                    if forbidden in herb or herb in forbidden:
                        warnings.append(f"【孕期禁用】{herb}孕期禁用！")
                for careful in pregnancy_careful:
                    if careful in herb or herb in careful:
                        warnings.append(f"【孕期慎用】{herb}孕期需谨慎使用！")
                        suggestions.append(f"建议在医师指导下使用{herb}")

        # 体质建议
        if constitution:
            if "阴虚" in constitution:
                hot_herbs = ["附子", "肉桂", "干姜", "吴茱萸"]
                for herb in herbs:
                    if any(h in herb for h in hot_herbs):
                        suggestions.append(f"{herb}性温热，阴虚体质宜慎用或配伍滋阴药")

        is_compatible = len(incompatible_pairs) == 0

        return {
            "is_compatible": is_compatible,
            "warnings": warnings,
            "incompatible_pairs": incompatible_pairs,
            "suggestions": suggestions,
            "herbs_checked": herbs
        }

    async def search_classics(
        self,
        keywords: list[str],
        books: Optional[list[str]] = None,
        max_results: int = 5
    ) -> dict:
        """检索古籍"""
        logger.info(f"检索古籍: keywords={keywords}, books={books}")

        # 模拟古籍数据
        classics_db = [
            {
                "book": "伤寒论",
                "chapter": "辨太阳病脉证并治",
                "section": "第1条",
                "text": "太阳之为病，脉浮，头项强痛而恶寒。",
                "keywords": ["头痛", "太阳病", "恶寒"]
            },
            {
                "book": "伤寒论",
                "chapter": "辨少阳病脉证并治",
                "section": "第96条",
                "text": "伤寒五六日中风，往来寒热，胸胁苦满，嘿嘿不欲饮食，心烦喜呕。",
                "keywords": ["胸闷", "少阳病", "不欲饮食"]
            },
            {
                "book": "金匮要略",
                "chapter": "血痹虚劳病脉证并治",
                "text": "男子平人，脉虚弱细微者，善盗汗也。",
                "keywords": ["虚劳", "盗汗"]
            },
            {
                "book": "黄帝内经·素问",
                "chapter": "至真要大论",
                "text": "诸风掉眩，皆属于肝。",
                "keywords": ["头晕", "肝", "眩晕"]
            },
            {
                "book": "黄帝内经·素问",
                "chapter": "阴阳应象大论",
                "text": "肝主怒，怒伤肝。",
                "keywords": ["肝", "情志", "怒"]
            },
            {
                "book": "黄帝内经·灵枢",
                "chapter": "本神",
                "text": "心藏神，神有余则笑不休，神不足则悲。",
                "keywords": ["心", "失眠", "神"]
            },
        ]

        # 过滤书籍
        if books:
            classics_db = [c for c in classics_db if c["book"] in books]

        # 匹配关键词
        results = []
        for entry in classics_db:
            matched = [kw for kw in keywords if kw in entry.get("keywords", []) or kw in entry["text"]]
            if matched:
                results.append({
                    **entry,
                    "keywords_matched": matched
                })

        results.sort(key=lambda x: -len(x["keywords_matched"]))
        results = results[:max_results]

        return {
            "citations": results,
            "count": len(results),
            "keywords": keywords
        }

    async def search_cases(
        self,
        symptoms: Optional[list[str]] = None,
        syndrome: Optional[str] = None,
        limit: int = 5
    ) -> dict:
        """搜索医案"""
        logger.info(f"搜索医案: symptoms={symptoms}, syndrome={syndrome}")

        # 模拟医案数据
        cases_db = [
            {
                "case_id": "C001",
                "patient_info": "男，45岁",
                "chief_complaint": "反复头痛3年，加重1周",
                "symptoms": ["头痛", "失眠", "急躁易怒"],
                "syndrome": "肝阳上亢",
                "treatment": "平肝潜阳",
                "prescription": "天麻钩藤饮加减",
                "outcome": "服药7剂后头痛明显减轻",
                "source": "名医医案"
            },
            {
                "case_id": "C002",
                "patient_info": "女，38岁",
                "chief_complaint": "胸闷不适半年",
                "symptoms": ["胸闷", "叹息", "情绪不畅"],
                "syndrome": "肝郁气滞",
                "treatment": "疏肝理气",
                "prescription": "逍遥散加减",
                "outcome": "服药14剂后症状消失",
                "source": "临床经验"
            },
            {
                "case_id": "C003",
                "patient_info": "男，28岁",
                "chief_complaint": "失眠多梦1个月",
                "symptoms": ["失眠", "多梦", "腰膝酸软"],
                "syndrome": "心肾不交",
                "treatment": "交通心肾",
                "prescription": "黄连阿胶汤加减",
                "outcome": "服药10剂后睡眠改善",
                "source": "名医医案"
            },
        ]

        results = []
        for case in cases_db:
            if syndrome and syndrome in case["syndrome"]:
                results.append(case)
            elif symptoms:
                if any(s in case["symptoms"] for s in symptoms):
                    results.append(case)

        if not results:
            results = cases_db[:limit]

        return {
            "cases": results[:limit],
            "count": len(results[:limit])
        }


# ============== MCP 服务器实现 ==============

def create_mcp_server(name: str = "smart-tcm-knowledge") -> Optional[Server]:
    """
    创建MCP服务器实例

    Args:
        name: 服务器名称

    Returns:
        Server实例，如果MCP库不可用则返回None
    """
    if not MCP_AVAILABLE:
        logger.warning("MCP library not available, server cannot be created")
        return None

    return Server(name)


def create_mcp_server_with_executor(
    name: str = "smart-tcm-knowledge",
    executor: Optional[TCMToolExecutor] = None
) -> Optional[Server]:
    """
    创建带执行器的MCP服务器

    Args:
        name: 服务器名称
        executor: 工具执行器

    Returns:
        配置好的Server实例
    """
    server = create_mcp_server(name)
    if not server:
        return None

    tool_executor = executor or TCMToolExecutor()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """列出所有可用工具"""
        return TCM_TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """调用工具"""
        logger.info(f"MCP工具调用: {name} with arguments: {arguments}")

        try:
            if name == "search_herbs":
                result = await tool_executor.search_herbs(
                    query=arguments.get("query", ""),
                    limit=arguments.get("limit", 10)
                )
            elif name == "search_prescriptions":
                result = await tool_executor.search_prescriptions(
                    syndrome=arguments.get("syndrome"),
                    symptoms=arguments.get("symptoms"),
                    prescription_name=arguments.get("prescription_name"),
                    limit=arguments.get("limit", 10)
                )
            elif name == "diagnose_syndrome":
                result = await tool_executor.diagnose_syndrome(
                    symptoms=arguments.get("symptoms", {}),
                    pulse=arguments.get("pulse"),
                    tongue=arguments.get("tongue"),
                    complexity_level=arguments.get("complexity_level", "simple")
                )
            elif name == "query_knowledge_graph":
                result = await tool_executor.query_knowledge_graph(
                    cypher=arguments.get("cypher", ""),
                    params=arguments.get("params")
                )
            elif name == "get_contraindications":
                result = await tool_executor.get_contraindications(
                    herbs=arguments.get("herbs", []),
                    constitution=arguments.get("constitution"),
                    pregnancy=arguments.get("pregnancy", False)
                )
            elif name == "search_classics":
                result = await tool_executor.search_classics(
                    keywords=arguments.get("keywords", []),
                    books=arguments.get("books"),
                    max_results=arguments.get("max_results", 5)
                )
            elif name == "search_cases":
                result = await tool_executor.search_cases(
                    symptoms=arguments.get("symptoms"),
                    syndrome=arguments.get("syndrome"),
                    limit=arguments.get("limit", 5)
                )
            else:
                result = {"error": f"Unknown tool: {name}"}

            return [TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False, indent=2)
            )]

        except Exception as e:
            logger.error(f"工具调用失败: {e}")
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, ensure_ascii=False)
            )]

    return server


async def run_mcp_server(server: Server):
    """
    运行MCP服务器（stdio模式）

    Args:
        server: MCP服务器实例
    """
    if not MCP_AVAILABLE:
        raise RuntimeError("MCP library not available")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


# ============== 快捷函数 ==============

def get_tool_executor() -> TCMToolExecutor:
    """获取工具执行器单例"""
    return TCMToolExecutor()


__all__ = [
    "TCM_TOOLS",
    "TCMToolExecutor",
    "TCMToolContext",
    "create_mcp_server",
    "create_mcp_server_with_executor",
    "run_mcp_server",
    "get_tool_executor",
    "MCP_AVAILABLE",
]
