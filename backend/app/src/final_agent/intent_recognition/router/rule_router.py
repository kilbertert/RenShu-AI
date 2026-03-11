"""
L1 Rule-Based Router - 规则引擎层
使用正则表达式和关键词匹配，快速识别高频确定意图

借鉴大厂做法：
- 阿里：AC自动机多模式匹配
- 美团：正则+关键词拦截高频场景
- 响应时间 < 5ms
"""

import re
from typing import Optional
from app.src.agent.intent_recognition.schemas import (
    IntentRouteResult,
    IntentClassification,
    IntentType,
    OOSResult,
    OOSReason,
    EnrichedContext,
    UserProfile,
    EnvironmentContext,
    ConversationContext, ExtractedEntities, SentimentAnalysis, WellnessLevel,
)


class RuleBasedRouter:
    """
    L1规则引擎层 - 高频意图快速匹配

    核心原则：
    1. 只处理高置信度、明确的意图
    2. 响应时间 < 5ms
    3. 匹配失败则交给下一层处理
    """

    # ============== 中医实体词库 ==============

    # 常见方剂名（可扩展）
    PRESCRIPTION_NAMES = [
        "桂枝汤", "麻黄汤", "小柴胡汤", "大柴胡汤", "四逆汤",
        "六味地黄丸", "金匮肾气丸", "补中益气汤", "四君子汤", "六君子汤",
        "四物汤", "八珍汤", "十全大补汤", "归脾汤", "逍遥散",
        "银翘散", "桑菊饮", "麻杏石甘汤", "白虎汤", "承气汤",
        "理中汤", "小建中汤", "当归补血汤", "生脉散", "参苓白术散",
        "二陈汤", "温胆汤", "半夏泻心汤", "黄连解毒汤", "龙胆泻肝汤",
        "天王补心丹", "酸枣仁汤", "安神定志丸", "柴胡疏肝散",
        "血府逐瘀汤", "少腹逐瘀汤", "膈下逐瘀汤", "通窍活血汤",
    ]

    # 常见药材名（可扩展）
    HERB_NAMES = [
        "黄芪", "人参", "党参", "西洋参", "当归", "熟地", "白芍", "川芎",
        "白术", "茯苓", "甘草", "陈皮", "半夏", "枸杞", "菊花", "金银花",
        "黄连", "黄芩", "黄柏", "栀子", "大黄", "芒硝", "附子", "干姜",
        "肉桂", "桂枝", "麻黄", "细辛", "柴胡", "葛根", "防风", "荆芥",
        "薄荷", "蝉蜕", "连翘", "板蓝根", "鱼腥草", "蒲公英", "紫花地丁",
        "三七", "丹参", "红花", "桃仁", "川牛膝", "赤芍", "益母草",
        "天麻", "钩藤", "石决明", "珍珠母", "酸枣仁", "远志", "合欢皮",
    ]

    # 古籍名
    CLASSIC_BOOKS = [
        "黄帝内经", "素问", "灵枢", "伤寒论", "金匮要略", "伤寒杂病论",
        "神农本草经", "难经", "温病条辨", "温热论", "本草纲目",
        "千金方", "外台秘要", "医宗金鉴", "景岳全书", "类经",
    ]

    # 节气关键词
    SOLAR_TERMS = [
        "立春", "雨水", "惊蛰", "春分", "清明", "谷雨",
        "立夏", "小满", "芒种", "夏至", "小暑", "大暑",
        "立秋", "处暑", "白露", "秋分", "寒露", "霜降",
        "立冬", "小雪", "大雪", "冬至", "小寒", "大寒",
    ]

    # ============== 意图匹配规则 ==============

    def __init__(self):
        """初始化规则路由器"""
        self._compile_patterns()

    def _compile_patterns(self):
        """编译正则表达式"""
        # 方剂名模式
        prescription_pattern = "|".join(re.escape(p) for p in self.PRESCRIPTION_NAMES)
        self._prescription_re = re.compile(f"({prescription_pattern})", re.IGNORECASE)

        # 药材名模式
        herb_pattern = "|".join(re.escape(h) for h in self.HERB_NAMES)
        self._herb_re = re.compile(f"({herb_pattern})", re.IGNORECASE)

        # 古籍名模式
        book_pattern = "|".join(re.escape(b) for b in self.CLASSIC_BOOKS)
        self._book_re = re.compile(f"({book_pattern})", re.IGNORECASE)

        # 节气模式
        solar_pattern = "|".join(re.escape(s) for s in self.SOLAR_TERMS)
        self._solar_re = re.compile(f"({solar_pattern})", re.IGNORECASE)

        # 意图模式规则
        self._intent_patterns = {
            # 方剂查询模式
            "prescription_query": [
                re.compile(r"(" + prescription_pattern + r").*?(是什么|介绍|组成|成分|药材|配方|功效|作用|主治)", re.IGNORECASE),
                re.compile(r"(什么是|介绍一下).*?(" + prescription_pattern + r")", re.IGNORECASE),
            ],
            # 方剂推荐模式
            "prescription_recommend": [
                re.compile(r"(风寒|风热|湿热|阴虚|阳虚|气虚|血虚|痰湿).*(吃什么方|用什么方|什么方子)", re.IGNORECASE),
                re.compile(r"(感冒|咳嗽|失眠|头痛).*(吃什么方|推荐.*方)", re.IGNORECASE),
            ],
            # 药材功效查询
            "herb_effect": [
                re.compile(r"(" + herb_pattern + r").*?(功效|作用|好处|有什么用|能.*?吗)", re.IGNORECASE),
                re.compile(r"(什么是|介绍一下).*?(" + herb_pattern + r")", re.IGNORECASE),
            ],
            # 药材配伍禁忌
            "herb_compatibility": [
                re.compile(r"(" + herb_pattern + r").*?(不能|禁忌|相克|能不能|可以.*一起)", re.IGNORECASE),
                re.compile(r"(" + herb_pattern + r").*?和.*?(" + herb_pattern + r").*?(一起|配伍|搭配)", re.IGNORECASE),
            ],
            # 古籍检索
            "classics_query": [
                re.compile(r"(" + book_pattern + r").*?(记载|怎么说|条文|原文)", re.IGNORECASE),
                re.compile(r"(查.*?|检索.*?)(" + book_pattern + r")", re.IGNORECASE),
            ],
            # 养生L1（简单季节养生）
            "wellness_l1": [
                re.compile(r"(春季|夏季|秋季|冬季|春天|夏天|秋天|冬天).*(养生|调理|保健|注意)", re.IGNORECASE),
                re.compile(r"(" + solar_pattern + r").*(吃什么|怎么养生|注意什么)", re.IGNORECASE),
                re.compile(r"日常.*(养生|保健|调理)", re.IGNORECASE),
            ],
            # OOS - 中医基础知识科普（应路由到 tcm-chat）
            "tcm_knowledge": [
                re.compile(r"(什么是|介绍一下|了解|讲解).*(中医|中药|五行|阴阳|八纲|六淫|七情|气血|经络|脏腑)", re.IGNORECASE),
                re.compile(r"(中医|中药).*(是什么|介绍|历史|发展|特点|理论|基础|区别|对比|和.*的关系)", re.IGNORECASE),
                re.compile(r"(五行|阴阳|八纲|六淫|七情|气血|经络|脏腑).*(是什么|有哪些|包括|含义|概念)", re.IGNORECASE),
                re.compile(r"(中医.*西医|西医.*中医).*(区别|对比|不同|差异)", re.IGNORECASE),
            ],
        }

    def route(self, query: str) -> Optional[IntentClassification]:
        """
        规则匹配路由

        Args:
            query: 用户输入

        Returns:
            IntentClassification: 匹配成功返回分类结果，失败返回None
        """
        if not query or not query.strip():
            return None

        # 提取实体
        entities = self._extract_entities(query)

        # 按优先级匹配意图
        result = self._match_intent(query, entities)

        return result

    def _extract_entities(self, query: str) -> ExtractedEntities:
        """提取中医实体"""
        entities = ExtractedEntities()

        # 提取方剂名
        prescription_matches = self._prescription_re.findall(query)
        if prescription_matches:
            entities.prescriptions = list(set(prescription_matches))

        # 提取药材名
        herb_matches = self._herb_re.findall(query)
        if herb_matches:
            entities.herbs = list(set(herb_matches))

        # 提取古籍名
        book_matches = self._book_re.findall(query)
        if book_matches:
            entities.books = list(set(book_matches))

        return entities

    def _match_intent(self, query: str, entities: ExtractedEntities) -> Optional[IntentClassification]:
        """匹配意图"""
        
        # 0. 中医基础知识科普
        for pattern in self._intent_patterns["tcm_knowledge"]:
            if pattern.search(query):
                return IntentClassification(
                    primary_intent=IntentType.GENERAL,
                    confidence=0.95,
                    sub_type="tcm_knowledge",
                    entities=entities,
                    reasoning="规则匹配：中医基础知识科普",
                    route_source="rule",
                    sentiment=SentimentAnalysis(),
                )
        # 1. 方剂查询（有方剂名 + 查询动作）
        for pattern in self._intent_patterns["prescription_query"]:
            if pattern.search(query):
                return IntentClassification(
                    primary_intent=IntentType.PRESCRIPTION,
                    confidence=0.95,
                    sub_type="query",  # 方剂查询
                    entities=entities,
                    reasoning="规则匹配：方剂查询模式",
                    route_source="rule",
                    sentiment=SentimentAnalysis(),
                )

        # 2. 方剂推荐（证型/症状 + 推荐方剂）
        for pattern in self._intent_patterns["prescription_recommend"]:
            if pattern.search(query):
                return IntentClassification(
                    primary_intent=IntentType.PRESCRIPTION,
                    confidence=0.90,
                    sub_type="recommend",  # 方剂推荐
                    entities=entities,
                    reasoning="规则匹配：方剂推荐模式",
                    route_source="rule",
                    sentiment=SentimentAnalysis(),
                )

        # 3. 药材配伍禁忌（优先级高于功效查询）
        for pattern in self._intent_patterns["herb_compatibility"]:
            if pattern.search(query):
                return IntentClassification(
                    primary_intent=IntentType.HERB,
                    confidence=0.95,
                    sub_type="compatibility",  # 配伍禁忌
                    entities=entities,
                    reasoning="规则匹配：药材配伍禁忌查询",
                    route_source="rule",
                    sentiment=SentimentAnalysis(),
                )

        # 4. 药材功效查询
        for pattern in self._intent_patterns["herb_effect"]:
            if pattern.search(query):
                return IntentClassification(
                    primary_intent=IntentType.HERB,
                    confidence=0.92,
                    sub_type="effect",  # 药材功效
                    entities=entities,
                    reasoning="规则匹配：药材功效查询",
                    route_source="rule",
                    sentiment=SentimentAnalysis(),
                )

        # 5. 养生L1（简单养生）
        for pattern in self._intent_patterns["wellness_l1"]:
            if pattern.search(query):
                return IntentClassification(
                    primary_intent=IntentType.WELLNESS,
                    confidence=0.88,
                    sub_type="seasonal",  # 季节养生
                    wellness_level=WellnessLevel.L1,
                    entities=entities,
                    reasoning="规则匹配：简单养生咨询",
                    route_source="rule",
                    sentiment=SentimentAnalysis(
                        polarity="neutral",
                        anxiety_score=0.1,
                        urgency="low"
                    ),
                )

        # 6. 纯实体匹配（有实体但无明确意图模式）
        if entities.prescriptions and not entities.herbs:
            return IntentClassification(
                primary_intent=IntentType.PRESCRIPTION,
                confidence=0.75,
                sub_type="query",  # 默认方剂查询
                entities=entities,
                reasoning="规则匹配：检测到方剂名",
                route_source="rule",
                sentiment=SentimentAnalysis(),
            )

        if entities.herbs and not entities.prescriptions:
            return IntentClassification(
                primary_intent=IntentType.HERB,
                confidence=0.75,
                sub_type="effect",  # 默认药材功效
                entities=entities,
                reasoning="规则匹配：检测到药材名",
                route_source="rule",
                sentiment=SentimentAnalysis(),
            )

        # 古籍名实体 -> 交给LLM处理（可能是养生、方剂或问诊相关）
        # 不在规则层做硬性分类

        # 未匹配
        return None

    def add_prescription(self, name: str):
        """添加方剂名（支持动态扩展）"""
        if name not in self.PRESCRIPTION_NAMES:
            self.PRESCRIPTION_NAMES.append(name)
            self._compile_patterns()

    def add_herb(self, name: str):
        """添加药材名（支持动态扩展）"""
        if name not in self.HERB_NAMES:
            self.HERB_NAMES.append(name)
            self._compile_patterns()

    def is_tcm_knowledge_query(self, query: str) -> bool:
        """
        检测是否为中医基础知识科普问题
        
        这类问题应该路由到 tcm-chat（一般性回答），而不是 tcm-wellness（养生）
        
        Args:
            query: 用户查询
            
        Returns:
            bool: 是否为中医基础知识问题
        """
        for pattern in self._intent_patterns.get("tcm_knowledge", []):
            if pattern.search(query):
                return True
        return False


# 单例实例
_router_instance: Optional[RuleBasedRouter] = None


def get_rule_router() -> RuleBasedRouter:
    """获取规则路由器单例"""
    global _router_instance
    if _router_instance is None:
        _router_instance = RuleBasedRouter()
    return _router_instance
