"""
Context Enricher - 上下文增强模块
融合用户画像、节气、地域等中医特有上下文特征

借鉴大厂做法：
- 阿里小蜜：结合上下文+领域模型进行意图推理
- 美团：RT特征（用户停留页面、最近点击、用户画像）

数据来源：
- 用户画像：Patient.base_profile + Patient基础字段
- 会话画像：Conversation.session_metadata（由 analyze_persona API 实时更新）
- 对话历史：Message 表
"""

import asyncio
from datetime import datetime, date
from typing import Optional, List
from uuid import UUID

from .schemas import (
    UserProfile,
    EnvironmentContext,
    ConversationContext,
    EnrichedContext,
)

from app.src.utils import get_logger

logger = get_logger("context_enricher")


class TCMContextEnricher:
    """
    中医上下文增强器

    核心功能：
    1. 获取用户画像（体质、年龄、性别、病史）- 来源：Patient.base_profile
    2. 获取会话画像（主诉、疑似诊断）- 来源：Conversation.session_metadata
    3. 获取环境特征（节气、季节、地域）
    4. 获取对话上下文（历史症状、问诊阶段）- 来源：Message 表

    设计原则：
    - 使用独立的只读 session，避免与主事务冲突
    - 纯读取操作，不会造成事务竞争
    - 支持 Redis 缓存减少数据库压力
    """

    # 二十四节气对应的日期范围（简化版，实际需要更精确的计算）
    SOLAR_TERMS = [
        ("立春", (2, 4)), ("雨水", (2, 19)), ("惊蛰", (3, 6)), ("春分", (3, 21)),
        ("清明", (4, 5)), ("谷雨", (4, 20)), ("立夏", (5, 6)), ("小满", (5, 21)),
        ("芒种", (6, 6)), ("夏至", (6, 21)), ("小暑", (7, 7)), ("大暑", (7, 23)),
        ("立秋", (8, 8)), ("处暑", (8, 23)), ("白露", (9, 8)), ("秋分", (9, 23)),
        ("寒露", (10, 8)), ("霜降", (10, 24)), ("立冬", (11, 8)), ("小雪", (11, 22)),
        ("大雪", (12, 7)), ("冬至", (12, 22)), ("小寒", (1, 6)), ("大寒", (1, 20)),
    ]

    # 季节对应月份
    SEASONS = {
        "春季": [2, 3, 4],
        "夏季": [5, 6, 7],
        "秋季": [8, 9, 10],
        "冬季": [11, 12, 1],
    }

    # 地域与体质/养生的关联（简化版）
    REGION_CHARACTERISTICS = {
        "北方": {"climate": "干燥寒冷", "common_issues": ["阴虚燥热", "脾胃虚寒"]},
        "南方": {"climate": "湿热", "common_issues": ["湿热", "痰湿"]},
        "沿海": {"climate": "潮湿", "common_issues": ["风湿", "痰湿"]},
        "西北": {"climate": "干燥", "common_issues": ["阴虚", "燥证"]},
        "西南": {"climate": "湿润", "common_issues": ["湿热", "脾虚湿盛"]},
    }

    def __init__(self, db_session=None, redis_client=None):
        """
        初始化上下文增强器

        Args:
            db_session: 数据库会话（可选，如果不传则使用独立 session）
            redis_client: Redis客户端（用于缓存）
        """
        self.db_session = db_session
        self.redis_client = redis_client
        self._use_independent_session = db_session is None

    async def enrich(
        self,
        user_id: str,
        query: str,
        conversation_id: Optional[str] = None,
    ) -> EnrichedContext:
        """
        增强上下文

        Args:
            user_id: 用户ID
            query: 用户查询
            conversation_id: 会话ID

        Returns:
            EnrichedContext: 增强后的上下文
        """
        # 如果需要使用独立 session
        if self._use_independent_session:
            return await self._enrich_with_independent_session(user_id, query, conversation_id)
        else:
            return await self._enrich_with_existing_session(user_id, query, conversation_id)

    async def _enrich_with_independent_session(
        self,
        user_id: str,
        query: str,
        conversation_id: Optional[str] = None,
    ) -> EnrichedContext:
        """使用独立 session 进行上下文增强（避免事务冲突）"""
        from app.src.common.config.prosgresql_config import async_db_manager

        try:
            async with async_db_manager.get_session() as session:
                # 并行获取所有上下文
                results = await asyncio.gather(
                    self._get_user_profile_from_db(session, user_id),
                    self._get_session_metadata_from_db(session, conversation_id),
                    self._get_conversation_history_from_db(session, conversation_id),
                    return_exceptions=True
                )

                user_profile, session_metadata, conversation_context = results

                # 处理异常情况
                if isinstance(user_profile, Exception):
                    logger.warning(f"获取用户画像失败: {user_profile}")
                    user_profile = UserProfile(user_id=user_id)
                if isinstance(session_metadata, Exception):
                    logger.warning(f"获取会话画像失败: {session_metadata}")
                    session_metadata = {}
                if isinstance(conversation_context, Exception):
                    logger.warning(f"获取对话历史失败: {conversation_context}")
                    conversation_context = ConversationContext()

                # 合并 session_metadata 到 user_profile（如果有更新的信息）
                user_profile = self._merge_session_metadata(user_profile, session_metadata)

                # 获取环境上下文（不需要数据库）
                environment = await self._get_environment_context()

                return EnrichedContext(
                    user_profile=user_profile,
                    environment=environment,
                    conversation=conversation_context
                )
        except Exception as e:
            logger.error(f"上下文增强失败: {e}")
            # 降级返回基础上下文
            return EnrichedContext(
                user_profile=UserProfile(user_id=user_id),
                environment=await self._get_environment_context(),
                conversation=ConversationContext()
            )

    async def _enrich_with_existing_session(
        self,
        user_id: str,
        query: str,
        conversation_id: Optional[str] = None,
    ) -> EnrichedContext:
        """使用已有 session 进行上下文增强"""
        # 并行获取所有上下文
        results = await asyncio.gather(
            self._get_user_profile_from_db(self.db_session, user_id),
            self._get_session_metadata_from_db(self.db_session, conversation_id),
            self._get_conversation_history_from_db(self.db_session, conversation_id),
            return_exceptions=True
        )

        user_profile, session_metadata, conversation_context = results

        # 处理异常情况
        if isinstance(user_profile, Exception):
            logger.warning(f"获取用户画像失败: {user_profile}")
            user_profile = UserProfile(user_id=user_id)
        if isinstance(session_metadata, Exception):
            logger.warning(f"获取会话画像失败: {session_metadata}")
            session_metadata = {}
        if isinstance(conversation_context, Exception):
            logger.warning(f"获取对话历史失败: {conversation_context}")
            conversation_context = ConversationContext()

        # 合并 session_metadata 到 user_profile
        user_profile = self._merge_session_metadata(user_profile, session_metadata)

        # 获取环境上下文
        environment = await self._get_environment_context()

        return EnrichedContext(
            user_profile=user_profile,
            environment=environment,
            conversation=conversation_context
        )

    async def _get_user_profile_from_db(self, session, user_id: str) -> UserProfile:
        """
        从数据库获取用户画像

        数据来源：
        - Patient.base_profile: 体质类型、既往病史、家族病史、过敏信息
        - Patient 基础字段: gender, birth_date
        """
        # 1. 尝试从缓存获取
        if self.redis_client:
            cached = await self._get_cached_profile(user_id)
            if cached:
                logger.debug(f"从缓存获取用户画像: user_id={user_id}")
                return cached

        # 2. 从数据库获取
        try:
            from sqlmodel import select
            from app.src.model.account_model import Patient

            stmt = select(Patient).where(Patient.account_id == UUID(user_id))
            result = await session.exec(stmt)
            patient = result.first()

            if patient:
                base_profile = patient.base_profile or {}

                # 计算年龄
                age_group = None
                if patient.birth_date:
                    age_group = self._calculate_age_group_from_date(patient.birth_date)

                # 性别映射
                gender_map = {"male": "男", "female": "女", "other": "其他"}
                gender = gender_map.get(patient.gender, patient.gender)

                profile = UserProfile(
                    user_id=user_id,
                    gender=gender,
                    age_group=age_group,
                    constitution=base_profile.get("constitution_type"),
                    chronic_conditions=base_profile.get("medical_history", []),
                    allergies=base_profile.get("allergy_info", []),
                )

                # 3. 缓存结果
                if self.redis_client:
                    await self._cache_profile(user_id, profile)

                logger.debug(f"从数据库获取用户画像成功: user_id={user_id}, constitution={profile.constitution}")
                return profile

        except Exception as e:
            logger.warning(f"获取用户画像失败: {e}")

        return UserProfile(user_id=user_id)

    async def _get_session_metadata_from_db(self, session, conversation_id: Optional[str]) -> dict:
        """
        从数据库获取会话画像

        数据来源：Conversation.session_metadata
        包含：age, gender, healthScore, chiefComplaint, suspectedDiagnosis, recommendedTreatment
        """
        if not conversation_id:
            return {}

        try:
            from app.src.model.conversation_models import Conversation

            conversation = await session.get(Conversation, UUID(conversation_id))
            if conversation and conversation.session_metadata:
                logger.debug(f"获取会话画像成功: conversation_id={conversation_id}")
                return conversation.session_metadata

        except Exception as e:
            logger.warning(f"获取会话画像失败: {e}")

        return {}

    async def _get_conversation_history_from_db(
        self,
        session,
        conversation_id: Optional[str]
    ) -> ConversationContext:
        """
        从数据库获取对话历史上下文

        数据来源：Message 表
        提取：最近症状、提及的药材、提及的方剂
        """
        if not conversation_id:
            return ConversationContext()

        try:
            from sqlmodel import select
            from app.src.model.conversation_models import Message

            # 获取最近 10 条消息
            stmt = select(Message).where(
                Message.conversation_id == UUID(conversation_id),
                Message.is_deleted == False
            ).order_by(Message.created_at.desc()).limit(10)

            result = await session.exec(stmt)
            messages = result.all()

            # 提取上下文信息
            recent_symptoms = []
            mentioned_herbs = []
            mentioned_prescriptions = []

            # 简单的关键词提取（后续可以用 NER 模型增强）
            symptom_keywords = [
                "头痛", "头晕", "发热", "咳嗽", "乏力", "失眠", "腹痛", "腹泻",
                "便秘", "恶心", "呕吐", "胸闷", "心悸", "气短", "盗汗", "自汗",
                "口干", "口苦", "食欲不振", "怕冷", "怕热", "腰痛", "关节痛"
            ]

            herb_keywords = [
                "黄芪", "人参", "党参", "白术", "茯苓", "甘草", "当归", "川芎",
                "白芍", "熟地", "枸杞", "菊花", "金银花", "连翘", "板蓝根"
            ]

            prescription_keywords = [
                "六味地黄丸", "逍遥散", "四君子汤", "四物汤", "八珍汤",
                "补中益气汤", "归脾汤", "小柴胡汤", "桂枝汤", "麻黄汤"
            ]

            for msg in messages:
                if msg.role == "user":
                    content = msg.content
                    # 提取症状
                    for symptom in symptom_keywords:
                        if symptom in content and symptom not in recent_symptoms:
                            recent_symptoms.append(symptom)
                    # 提取药材
                    for herb in herb_keywords:
                        if herb in content and herb not in mentioned_herbs:
                            mentioned_herbs.append(herb)
                    # 提取方剂
                    for prescription in prescription_keywords:
                        if prescription in content and prescription not in mentioned_prescriptions:
                            mentioned_prescriptions.append(prescription)

            logger.debug(
                f"提取对话上下文: symptoms={recent_symptoms}, herbs={mentioned_herbs}, "
                f"prescriptions={mentioned_prescriptions}, turn_count={len(messages)}"
            )

            return ConversationContext(
                recent_symptoms=recent_symptoms,
                mentioned_herbs=mentioned_herbs,
                mentioned_prescriptions=mentioned_prescriptions,
                diagnosis_stage=None,  # 可以从 session_metadata 获取
                turn_count=len(messages),
            )

        except Exception as e:
            logger.warning(f"获取对话历史失败: {e}")

        return ConversationContext()

    def _merge_session_metadata(self, profile: UserProfile, session_metadata: dict) -> UserProfile:
        """
        合并会话画像到用户画像

        session_metadata 中的实时信息优先级更高（如果有的话）
        """
        if not session_metadata:
            return profile

        # 如果 session_metadata 有更新的年龄/性别信息，使用它
        if session_metadata.get("age") and not profile.age_group:
            try:
                age = int(session_metadata["age"])
                profile.age_group = self._calculate_age_group(age)
            except (ValueError, TypeError):
                pass

        if session_metadata.get("gender") and not profile.gender:
            profile.gender = session_metadata["gender"]

        return profile

    def _calculate_age_group_from_date(self, birth_date: date) -> Optional[str]:
        """从出生日期计算年龄段"""
        if not birth_date:
            return None

        today = date.today()
        age = today.year - birth_date.year - (
            (today.month, today.day) < (birth_date.month, birth_date.day)
        )
        return self._calculate_age_group(age)

    async def _get_cached_profile(self, user_id: str) -> Optional[UserProfile]:
        """从Redis缓存获取用户画像"""
        if not self.redis_client:
            return None

        try:
            import json
            cache_key = f"tcm:user_profile:{user_id}"
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                return UserProfile(**data)
        except Exception:
            pass

        return None

    async def _cache_profile(self, user_id: str, profile: UserProfile):
        """缓存用户画像"""
        if not self.redis_client:
            return

        try:
            import json
            cache_key = f"tcm:user_profile:{user_id}"
            await self.redis_client.setex(
                cache_key,
                3600,  # 1小时过期
                json.dumps(profile.model_dump())
            )
        except Exception:
            pass

    def _calculate_age_group(self, age) -> Optional[str]:
        """计算年龄段"""
        if not age:
            return None

        try:
            age = int(age)
            if age < 18:
                return "青少年"
            elif age < 40:
                return "青年"
            elif age < 60:
                return "中年"
            else:
                return "老年"
        except (ValueError, TypeError):
            return None

    async def _get_environment_context(self) -> EnvironmentContext:
        """获取环境上下文（节气、季节等）"""
        now = datetime.now()

        return EnvironmentContext(
            solar_term=self._get_current_solar_term(now),
            season=self._get_current_season(now),
            region=None,  # 可以通过IP定位获取
            weather=None,  # 可以通过天气API获取
        )

    def _get_current_solar_term(self, dt: datetime) -> str:
        """获取当前节气"""
        month, day = dt.month, dt.day

        for i, (term, (m, d)) in enumerate(self.SOLAR_TERMS):
            if month == m and day >= d:
                return term
            if month == m and day < d and i > 0:
                return self.SOLAR_TERMS[i - 1][0]

        # 处理年末年初的情况
        if month == 1 and day < 6:
            return "冬至"

        return "未知"

    def _get_current_season(self, dt: datetime) -> str:
        """获取当前季节"""
        month = dt.month
        for season, months in self.SEASONS.items():
            if month in months:
                return season
        return "未知"

    def get_seasonal_advice(self, season: str, constitution: Optional[str] = None) -> str:
        """获取季节养生建议"""
        base_advice = {
            "春季": "春季宜养肝，多食青色蔬菜，保持心情舒畅，适当运动。",
            "夏季": "夏季宜养心，清淡饮食，避免贪凉，适当午休。",
            "秋季": "秋季宜养肺，滋阴润燥，多食白色食物，注意保暖。",
            "冬季": "冬季宜养肾，适当进补，早睡晚起，避风寒。",
        }

        advice = base_advice.get(season, "")

        # 根据体质调整建议
        if constitution:
            constitution_adjustments = {
                "气虚": "您属气虚体质，建议加强补气，可适当食用黄芪、党参炖汤。",
                "阴虚": "您属阴虚体质，建议滋阴为主，多食百合、银耳等。",
                "阳虚": "您属阳虚体质，建议温补阳气，可适当食用羊肉、生姜。",
                "痰湿": "您属痰湿体质，建议健脾祛湿，少食肥甘厚腻。",
                "湿热": "您属湿热体质，建议清热利湿，饮食清淡。",
                "血瘀": "您属血瘀体质，建议活血化瘀，适当运动。",
                "气郁": "您属气郁体质，建议疏肝理气，保持心情舒畅。",
                "特禀": "您属特禀体质，建议避免过敏原，增强体质。",
                "平和": "您属平和体质，保持良好生活习惯即可。",
            }
            if constitution in constitution_adjustments:
                advice += "\n" + constitution_adjustments[constitution]

        return advice


# 工厂函数
def create_context_enricher(db_session=None, redis_client=None) -> TCMContextEnricher:
    """创建上下文增强器"""
    return TCMContextEnricher(db_session=db_session, redis_client=redis_client)
