"""
TCM 安全守卫中间件

功能：
1. 紧急情况检测（规则匹配）
2. 超范围问题拦截
3. 服务边界判断
4. 输出安全检查
5. LLM 兜底判断（规则未命中时）

架构设计（两层检测）：
- Layer 1: 规则检测（毫秒级，覆盖 80%+ 场景）
  - 紧急情况关键词 + 正则模式
  - 西医/超范围关键词 + 正则模式
  - 闲聊/非医学关键词
  - 中医实体关键词（确认在服务范围内）
- Layer 2: LLM 兜底（~200ms，处理模糊地带）
  - 规则未命中时调用轻量 LLM 判断
  - 失败时默认放行（保证可用性）
"""

import re
from typing import Any, Optional, Dict, List, Set
from dataclasses import dataclass
from enum import Enum
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.src.agent.middleware import BaseMiddleware
from app.src.agent.safety import (
    detect_psychological_crisis,
    psychological_crisis_response,
)




class GuardrailAction(Enum):
    """守卫动作"""
    ALLOW = "allow"           # 允许通过
    BLOCK_CRISIS = "block_crisis"  # 自杀/自伤等心理危机阻断
    BLOCK_EMERGENCY = "block_emergency"  # 紧急情况阻断
    BLOCK_MEDICATION_SAFETY = "block_medication_safety"  # 高风险用药阻断
    BLOCK_OOS = "block_oos"   # 超范围阻断
    WARN = "warn"             # 警告但允许
    CLARIFY = "clarify"       # 需要澄清


@dataclass
class GuardrailResult:
    """守卫检查结果"""
    action: GuardrailAction
    reason: str = ""
    matched_rule: str = ""
    response: str = ""
    confidence: float = 1.0


class TCMGuardrailsMiddleware(BaseMiddleware):
    """
    TCM 安全守卫中间件

    架构设计（两层检测）：
    - Layer 1: 规则检测（毫秒级，80%+ 覆盖率）
      - 紧急情况：关键词 + 正则模式
      - 超范围：西医/闲聊关键词 + 正则
      - 服务范围内：中医实体关键词确认
    - Layer 2: LLM 兜底（~200ms，处理模糊地带）
      - 规则未命中时调用轻量 LLM
      - 失败时默认放行（保证可用性）

    服务边界定义：
    - IN SCOPE: 中医养生、体质调理、症状分析、药材咨询、方剂查询
    - OUT OF SCOPE: 西医诊疗、急症处理、心理咨询、其他非医学话题
    """

    # ==================== 规则定义 ====================

    # 紧急情况关键词（需要立即就医）
    EMERGENCY_KEYWORDS = {
        # 心脑血管急症
        "剧烈胸痛", "胸闷气短", "心梗", "心肌梗死", "心绞痛",
        "中风", "脑梗", "脑出血", "半身不遂", "口眼歪斜",

        # 呼吸急症
        "呼吸困难", "喘不上气", "窒息", "呼吸衰竭",

        # 意识障碍
        "意识不清", "昏迷", "昏厥", "晕倒不醒", "抽搐",

        # 出血急症
        "大出血", "大量吐血", "大量便血", "咯血不止",

        # 休克/高热
        "休克", "高烧40度", "高烧不退超过3天",

        # 外伤/中毒
        "骨折", "严重外伤", "中毒", "药物过量",
    }

    # 紧急情况正则模式（症状组合）
    EMERGENCY_PATTERNS = [
        r"(突然|剧烈).{0,5}(头痛|胸痛|腹痛)",
        r"(?:胸口|胸部|心前区).{0,10}(?:压榨|紧缩|压迫|撕裂).{0,8}(?:痛|疼|不适)",
        r"(?:胸痛|胸口疼|胸部疼痛).{0,60}(?:左肩|左臂|下颌|后背|冷汗|气短|恶心)",
        r"(?:左肩|左臂|下颌|后背).{0,40}(?:放射|牵扯).{0,30}(?:胸痛|胸口疼|胸部疼痛)",
        r"(?:胸痛|胸口疼|胸部疼痛).{0,30}(?:持续|超过|已有).{0,8}(?:十五|15|二十|20|三十|30)(?:分钟|分)",
        r"(持续|反复).{0,5}(高烧|高热).{0,5}(不退|三天|3天)",
        r"(大量|不止).{0,5}(出血|吐血|便血)",
        r"(?:呕血|吐血|黑便|柏油样便).{0,30}(?:头晕|心慌|乏力|冷汗|晕厥)",
        r"(意识|神志).{0,5}(不清|模糊|丧失)",
        r"(呼吸|喘).{0,5}(困难|不上来|费力)",
        r"(?:突然|急性).{0,12}(?:一侧肢体无力|说话不清|口角歪斜|视物不清)",
        r"(?:嘴唇|舌头|咽喉).{0,8}(?:肿胀|肿起来).{0,20}(?:喘|呼吸困难|窒息)",
        r"(?:孕期|怀孕|妊娠).{0,20}(?:大量出血|剧烈腹痛|晕厥)",
    ]

    # 西医/超范围关键词
    OUT_OF_SCOPE_KEYWORDS = {
        # 西医检查
        "CT", "MRI", "核磁", "X光", "B超", "彩超",
        "心电图", "脑电图", "胃镜", "肠镜",

        # 西医治疗
        "手术", "开刀", "化疗", "放疗", "透析",
        "输液", "打点滴", "打针", "静脉注射",

        # 西药
        "抗生素", "消炎药", "头孢", "阿莫西林",
        "激素", "胰岛素", "降压药", "他汀",

        # 医院相关
        "挂号", "住院", "急诊", "ICU", "手术室",

        # 非中医领域
        "整容", "美容手术", "隆鼻", "双眼皮",
    }

    # 西医正则模式
    OUT_OF_SCOPE_PATTERNS = [
        r"需要.{0,5}(手术|开刀|做CT|做核磁)",
        r"(吃|用|开).{0,5}(抗生素|消炎药|西药)",
        r"(要不要|需不需要).{0,5}(去医院|挂号|住院)",
    ]

    # 闲聊/非医学关键词
    CHITCHAT_KEYWORDS = {
        # 闲聊
        "天气怎么样", "今天几号", "现在几点", "讲个笑话",
        "唱首歌", "讲个故事", "你是谁", "你叫什么",

        # 其他领域
        "炒股", "买房", "贷款", "理财",
        "游戏", "电影", "明星", "八卦",
        "编程", "代码", "Python", "Java",
    }

    NON_MEDICAL_CONTEXT_KEYWORDS = (
        "电脑", "手机", "代码", "程序", "软件", "操作系统", "电脑系统",
        "网络", "网页", "浏览器", "打印机", "服务器", "数据库", "汽车",
        "机器", "设备", "路由器", "天气",
    )
    HUMAN_HEALTH_CONTEXT_TERMS = (
        "头痛", "头晕", "眩晕", "咳嗽", "气短", "胸闷", "胸痛", "心悸",
        "恶心", "呕吐", "腰痛", "耳鸣", "腹痛", "胃痛", "腹泻", "便秘",
        "尿频", "夜尿", "失眠", "多梦", "食欲", "口干", "口苦", "盗汗",
        "自汗", "月经", "白带", "怀孕", "眼睛疼", "咽痛", "皮疹",
    )

    # 中医相关关键词（服务范围内）
    IN_SCOPE_KEYWORDS = {
        # 中医基础知识
        "中医", "中医药", "中华医学", "传统医学", "国医",
        "黄帝内经", "伤寒论", "金匮要略", "本草纲目",
        "中医理论", "中医基础", "中医知识", "中医科普",
        "阴阳", "五行", "八纲", "六淫", "七情",
        
        # 中医诊断
        "体质", "辨证", "证型", "脉象", "舌象", "舌苔",
        "望诊", "闻诊", "问诊", "切诊", "四诊",

        # 中医治疗
        "中药", "方剂", "汤药", "中成药", "药膳",
        "针灸", "艾灸", "拔罐", "刮痧", "推拿", "按摩",
        "经络", "穴位", "气血", "脏腑", "精气神",

        # 养生
        "养生", "调理", "食疗", "保健", "节气养生",
        "春季养生", "夏季养生", "秋季养生", "冬季养生",

        # 常见症状
        "失眠", "头痛", "头晕", "乏力", "疲劳",
        "便秘", "腹泻", "胃痛", "食欲不振",
        "感冒", "咳嗽", "发烧", "上火", "湿气重",
    }

    # 需要警告但允许的关键词（建议就医）
    WARN_KEYWORDS = {
        "长期", "反复", "加重", "越来越重",
        "半年", "一年", "好几年",
    }

    # 敏感输出关键词（需过滤）
    SENSITIVE_OUTPUT_KEYWORDS = {
        "保证治愈", "100%有效", "包好", "替代西医",
        "不用去医院", "不用看医生", "西医没用",
    }

    # ==================== 初始化 ====================

    def __init__(
        self,
        reject_out_of_scope: bool = True,
        warn_emergency: bool = True,
        check_output: bool = True,
        use_llm_fallback: bool = True,  # 默认启用 LLM 兖底
        llm_fallback_threshold: float = 0.5,  # 置信度低于此值时触发 LLM
    ):
        """
        初始化守卫中间件
    
        Args:
            reject_out_of_scope: 是否拒绝超范围问题
            warn_emergency: 是否警告紧急情况
            check_output: 是否检查输出安全
            use_llm_fallback: 是否启用 LLM 兖底判断
            llm_fallback_threshold: LLM 兖底的置信度阈值
        """

        super().__init__()
        self.reject_out_of_scope = reject_out_of_scope
        self.warn_emergency = warn_emergency
        self.check_output = check_output
        self.use_llm_fallback = use_llm_fallback
        self.llm_fallback_threshold = llm_fallback_threshold
    
        # 编译正则表达式
        self._emergency_patterns = [re.compile(p, re.IGNORECASE) for p in self.EMERGENCY_PATTERNS]
        self._oos_patterns = [re.compile(p, re.IGNORECASE) for p in self.OUT_OF_SCOPE_PATTERNS]
    
        # 预处理关键词为 set 和小写版本（性能优化）
        self._emergency_keywords_lower = {kw.lower() for kw in self.EMERGENCY_KEYWORDS}
        self._oos_keywords_lower = {kw.lower() for kw in self.OUT_OF_SCOPE_KEYWORDS}
        self._chitchat_keywords_lower = {kw.lower() for kw in self.CHITCHAT_KEYWORDS}
        self._in_scope_keywords_lower = {kw.lower() for kw in self.IN_SCOPE_KEYWORDS}
    
        # LLM 懒加载（仅在需要时初始化）
        self._llm = None

    # ==================== 辅助方法 ====================

    def _get_state_value(self, state: Any, key: str, default: Any = None) -> Any:
        """
        从状态中获取值（兼容字典和 Pydantic 模型）

        Args:
            state: 状态对象
            key: 键名
            default: 默认值

        Returns:
            对应的值
        """
        if isinstance(state, dict):
            return state.get(key, default)
        else:
            return getattr(state, key, default)

    # ==================== 主要接口 ====================

    def before_model(
        self,
        state: Dict[str, Any],
        runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """
        模型调用前：检查输入是否在服务范围内

        Args:
            state: 当前状态（可以是字典或 Pydantic 模型）
            runtime: 运行时上下文

        Returns:
            None: 允许通过
            Dict: 包含拦截响应或状态更新
        """
        # 兼容字典和 Pydantic 模型
        messages = self._get_state_value(state, "messages", [])
        if not messages:
            return None

        last_message = messages[-1]
        if not isinstance(last_message, HumanMessage):
            return None

        user_input = last_message.content

        # 执行规则检查
        result = self._check_input(user_input, state)

        # 根据检查结果决定动作
        if result.action == GuardrailAction.ALLOW:
            return None

        elif result.action == GuardrailAction.BLOCK_CRISIS:
            return {
                "messages": [AIMessage(content=result.response)],
                "answer": result.response,
                "should_seek_doctor": True,
                "steps": [f"安全检查: 心理危机拦截 ({result.matched_rule})"],
                "jump_to": "end",
            }

        elif result.action == GuardrailAction.BLOCK_EMERGENCY:
            return {
                "messages": [AIMessage(content=result.response)],
                "answer": result.response,
                "should_seek_doctor": True,
                "steps": [f"安全检查: 紧急情况拦截 ({result.matched_rule})"],
                "jump_to": "end",
            }

        elif result.action == GuardrailAction.BLOCK_OOS:
            return {
                "messages": [AIMessage(content=result.response)],
                "answer": result.response,
                "steps": [f"安全检查: 超范围拦截 ({result.matched_rule})"],
                "jump_to": "end",
            }

        elif result.action == GuardrailAction.BLOCK_MEDICATION_SAFETY:
            return {
                "messages": [AIMessage(content=result.response)],
                "answer": result.response,
                "should_seek_doctor": True,
                "steps": [f"安全检查: 高风险用药拦截 ({result.matched_rule})"],
                "jump_to": "end",
            }

        elif result.action == GuardrailAction.WARN:
            # 警告但允许继续，记录到 steps
            return {
                "steps": [f"安全检查: 警告 ({result.reason})"],
                "should_seek_doctor": True,
            }

        elif result.action == GuardrailAction.CLARIFY:
            return {
                "messages": [AIMessage(content=result.response)],
                "answer": result.response,
                "steps": [f"安全检查: 需要澄清"],
                "jump_to": "end",
            }

        return None

    def after_model(
        self,
        state: Dict[str, Any],
        runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """
        模型调用后：检查输出安全性

        Args:
            state: 当前状态
            runtime: 运行时上下文

        Returns:
            None: 输出安全
            Dict: 包含过滤后的输出
        """
        if not self.check_output:
            return None

        messages = self._get_state_value(state, "messages", [])
        if not messages:
            return None

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            return None

        content = last_message.content
        filtered_content, was_filtered = self._filter_output(content)

        if was_filtered:
            return {
                "messages": [AIMessage(content=filtered_content)],
                "steps": ["安全检查: 输出内容已过滤"],
            }

        return None

    # ==================== 输入检查逻辑 ====================

    def _check_input(self, user_input: str, state: Dict[str, Any]) -> GuardrailResult:
        """
        检查用户输入

        判断顺序：
        1. 心理危机检测（自杀/自伤/轻生）
        2. 紧急情况检测（关键词 + 正则）
        3. 超范围检测（关键词 + 正则）
        4. 闲聊检测（关键词）
        5. 服务范围内确认（关键词）
        6. 默认允许（删除了对路由结果的依赖和 LLM 兜底）
        """
        # 预处理
        text = user_input.lower().strip()

        # ========== 1. 心理危机检测 ==========
        crisis = detect_psychological_crisis(user_input)
        if crisis.is_crisis:
            return GuardrailResult(
                action=GuardrailAction.BLOCK_CRISIS,
                reason="检测到自杀、自伤或轻生风险表达",
                matched_rule=crisis.matched_text,
                response=psychological_crisis_response(),
                confidence=crisis.confidence,
            )

        # ========== 2. 紧急情况检测 ==========
        if self.warn_emergency:
            result = self._check_emergency(text, user_input)
            if result.action != GuardrailAction.ALLOW:
                return result

        # ========== 3. 孕期禁忌药/有毒药材剂量请求 ==========
        result = self._check_medication_safety(text, user_input)
        if result.action != GuardrailAction.ALLOW:
            return result

        # ========== 4. 超范围检测（仅检测明确的西医关键词） ==========
        if self.reject_out_of_scope:
            result = self._check_out_of_scope(text, user_input)
            if result.action != GuardrailAction.ALLOW:
                return result

        # ========== 5. 闲聊检测 ==========
        result = self._check_chitchat(text, user_input)
        if result.action != GuardrailAction.ALLOW:
            return result

        # ========== 6. 服务范围内确认 ==========
        if self._is_in_scope(text):
            # 检查是否需要警告
            warn_result = self._check_should_warn(text, user_input)
            if warn_result.action == GuardrailAction.WARN:
                return warn_result
            # 明确在服务范围内，允许通过
            return GuardrailResult(action=GuardrailAction.ALLOW)

        # ========== 7. 默认允许 ==========
        # 如果没有命中拦截规则，且包含中医相关内容，默认允许
        # 让后续的意图识别来判断具体如何处理
        return GuardrailResult(action=GuardrailAction.ALLOW)

    def _check_medication_safety(self, text: str, original: str) -> GuardrailResult:
        """在意图分类前拦截妊娠禁忌药和有毒药材的个体化剂量请求。"""
        from app.src.agent.tcm_validators import PREGNANCY_CONTRAINDICATED_HERBS

        pregnancy_markers = ("怀孕", "孕妇", "妊娠", "孕期", "孕早期", "孕中期", "孕晚期")
        dosage_markers = (
            "多少克", "剂量", "用量", "每天吃", "每日吃", "怎么服", "怎么吃",
            "服用多少", "直接告诉我", "开个方", "给我开", "完整中药处方",
            "完整处方", "写清楚每味药", "自己购买服用", "自行购买服用",
        )
        prohibited = PREGNANCY_CONTRAINDICATED_HERBS["禁用"]
        caution = PREGNANCY_CONTRAINDICATED_HERBS["慎用"]
        mentioned_prohibited = [herb for herb in prohibited if herb in original]
        mentioned_caution = [herb for herb in caution if herb in original]
        is_pregnant = any(marker in original for marker in pregnancy_markers)
        requests_dosage = any(marker in original for marker in dosage_markers)

        if is_pregnant and (mentioned_prohibited or mentioned_caution):
            herbs = [*mentioned_prohibited, *mentioned_caution]
            return GuardrailResult(
                action=GuardrailAction.BLOCK_MEDICATION_SAFETY,
                reason="妊娠期涉及禁用或慎用药材",
                matched_rule="妊娠用药:" + "、".join(herbs),
                response=self._get_medication_safety_response(
                    prohibited=mentioned_prohibited,
                    caution=mentioned_caution,
                    pregnancy=True,
                ),
                confidence=1.0,
            )

        if is_pregnant and requests_dosage:
            return GuardrailResult(
                action=GuardrailAction.BLOCK_MEDICATION_SAFETY,
                reason="妊娠期个体化处方或剂量请求",
                matched_rule="妊娠期处方剂量",
                response=self._get_high_risk_dosage_response("妊娠期"),
                confidence=1.0,
            )

        if requests_dosage and mentioned_prohibited:
            return GuardrailResult(
                action=GuardrailAction.BLOCK_MEDICATION_SAFETY,
                reason="有毒或高风险药材个体化剂量请求",
                matched_rule="高风险剂量:" + "、".join(mentioned_prohibited),
                response=self._get_medication_safety_response(
                    prohibited=mentioned_prohibited,
                    caution=[],
                    pregnancy=False,
                ),
                confidence=1.0,
            )

        minor_markers = (
            "婴儿", "婴幼儿", "宝宝", "新生儿", "儿童", "小孩", "孩子",
        )
        if requests_dosage and any(marker in original for marker in minor_markers):
            return GuardrailResult(
                action=GuardrailAction.BLOCK_MEDICATION_SAFETY,
                reason="儿童个体化处方或剂量请求",
                matched_rule="儿童处方剂量",
                response=self._get_high_risk_dosage_response("儿童"),
                confidence=1.0,
            )

        interaction_markers = (
            "华法林", "利伐沙班", "阿哌沙班", "达比加群", "氯吡格雷",
            "正在服用多种药", "服用多种药物", "多药联用",
        )
        if requests_dosage and any(marker in original for marker in interaction_markers):
            return GuardrailResult(
                action=GuardrailAction.BLOCK_MEDICATION_SAFETY,
                reason="抗凝或多药联用下的个体化中药请求",
                matched_rule="合并用药处方剂量",
                response=self._get_high_risk_dosage_response("合并用药"),
                confidence=1.0,
            )

        return GuardrailResult(action=GuardrailAction.ALLOW)

    def _check_emergency(self, text: str, original: str) -> GuardrailResult:
        """检测紧急情况"""
        # 关键词匹配（使用预处理的 set）
        for keyword in self.EMERGENCY_KEYWORDS:
            start = original.find(keyword)
            if start >= 0 and not self._is_negated_symptom(original, start):
                return GuardrailResult(
                    action=GuardrailAction.BLOCK_EMERGENCY,
                    reason="紧急情况关键词匹配",
                    matched_rule=keyword,
                    response=self._get_emergency_response(keyword),
                    confidence=1.0,
                )

        # 正则匹配
        for pattern in self._emergency_patterns:
            match = pattern.search(original)
            if match and not self._is_negated_symptom(original, match.start()):
                matched_text = match.group()
                return GuardrailResult(
                    action=GuardrailAction.BLOCK_EMERGENCY,
                    reason="紧急情况模式匹配",
                    matched_rule=matched_text,
                    response=self._get_emergency_response(matched_text),
                    confidence=0.9,
                )

        return GuardrailResult(action=GuardrailAction.ALLOW)

    @staticmethod
    def _is_negated_symptom(original: str, symptom_start: int) -> bool:
        """识别“无/否认/不伴某症状”，避免把明确否定描述当作急症。

        只处理紧邻前缀或同一顿号枚举中的明确否定；遇到“但/却/现”等转折，
        以及“不是没有/不能排除”等双重否定时保持急症拦截。
        """
        prefix = original[max(0, symptom_start - 12):symptom_start]
        compact = re.sub(r"\s+", "", prefix)
        if any(token in compact for token in ("不是没有", "并非没有", "不能排除", "不排除")):
            return False
        if re.search(r"(?:无|没有|未出现|否认|不伴|并无|未见|未发生)(?:明显|任何|持续|剧烈)?$", compact):
            return True

        clause_start = max(
            original.rfind(mark, 0, symptom_start)
            for mark in ("，", ",", "。", "！", "!", "？", "?", "；", ";", "\n")
        )
        clause_prefix = re.sub(r"\s+", "", original[clause_start + 1:symptom_start])
        if any(token in clause_prefix for token in ("但", "却", "然而", "现", "出现", "转为")):
            return False
        return bool(
            re.match(
                r"^(?:无|没有|未出现|否认|不伴|并无|未见|未发生)",
                clause_prefix,
            )
        )

    def _check_out_of_scope(self, text: str, original: str) -> GuardrailResult:
        """检测超范围问题"""
        # 关键词匹配
        for keyword in self.OUT_OF_SCOPE_KEYWORDS:
            keyword_lower = keyword.lower()
            if keyword_lower in text or keyword in original:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK_OOS,
                    reason="超范围关键词匹配",
                    matched_rule=keyword,
                    response=self._get_out_of_scope_response(keyword),
                    confidence=1.0,
                )

        # 正则匹配
        for pattern in self._oos_patterns:
            match = pattern.search(original)
            if match:
                matched_text = match.group()
                return GuardrailResult(
                    action=GuardrailAction.BLOCK_OOS,
                    reason="超范围模式匹配",
                    matched_rule=matched_text,
                    response=self._get_out_of_scope_response(matched_text),
                    confidence=0.9,
                )

        return GuardrailResult(action=GuardrailAction.ALLOW)

    def _check_chitchat(self, text: str, original: str) -> GuardrailResult:
        """检测闲聊/非医学话题"""
        # 先检查是否是问候语（允许）
        greetings = ["你好", "您好", "hi", "hello", "嗨", "早上好", "晚上好", "下午好", "早安", "晚安"]
        if any(g in text for g in greetings):
            return GuardrailResult(action=GuardrailAction.ALLOW)

        non_medical_match = next(
            (
                keyword for keyword in self.NON_MEDICAL_CONTEXT_KEYWORDS
                if keyword.lower() in text or keyword in original
            ),
            None,
        )
        if non_medical_match:
            has_human_health_context = any(
                term in original for term in self.HUMAN_HEALTH_CONTEXT_TERMS
            )
            if has_human_health_context:
                return GuardrailResult(action=GuardrailAction.ALLOW)
            return GuardrailResult(
                action=GuardrailAction.CLARIFY,
                reason="纯非医疗设备或技术语境",
                matched_rule=non_medical_match,
                response=self._get_chitchat_response(),
                confidence=0.98,
            )

        # 检查闲聊关键词
        for keyword in self.CHITCHAT_KEYWORDS:
            keyword_lower = keyword.lower()
            if keyword_lower in text or keyword in original:
                return GuardrailResult(
                    action=GuardrailAction.CLARIFY,
                    reason="非医学话题",
                    matched_rule=keyword,
                    response=self._get_chitchat_response(),
                    confidence=0.8,
                )

        return GuardrailResult(action=GuardrailAction.ALLOW)

    def _is_in_scope(self, text: str) -> bool:
        """检查是否在服务范围内（使用预处理的 set 优化性能）"""
        # 先检查小写版本（快速路径）
        if any(kw in text for kw in self._in_scope_keywords_lower):
            return True
        # 再检查原始关键词（处理大小写混合情况）
        return any(kw in text for kw in self.IN_SCOPE_KEYWORDS)

    def _check_should_warn(self, text: str, original: str) -> GuardrailResult:
        """检查是否需要警告（建议就医）"""
        # 检查警告关键词
        warn_count = sum(1 for kw in self.WARN_KEYWORDS if kw in original)

        if warn_count >= 2:
            return GuardrailResult(
                action=GuardrailAction.WARN,
                reason="症状持续/加重，建议就医",
                confidence=0.7,
            )

        return GuardrailResult(action=GuardrailAction.ALLOW)

    def _check_from_intent(self, router: Any) -> GuardrailResult:
        """从意图识别结果判断"""
        # 检查 OOS (Out of Scope)
        if hasattr(router, 'classification') and router.classification:
            classification = router.classification
            # 如果意图识别置信度很低
            if hasattr(classification, 'confidence') and classification.confidence < 0.3:
                return GuardrailResult(
                    action=GuardrailAction.CLARIFY,
                    reason="意图识别置信度过低",
                    response=self._get_clarify_response(),
                    confidence=classification.confidence,
                )

        return GuardrailResult(action=GuardrailAction.ALLOW)

    def _llm_fallback_check(self, user_input: str) -> GuardrailResult:
        """
        LLM 兜底判断（规则未命中时）

        使用轻量级 LLM 判断用户输入是否在服务范围内
        """
        if self._llm is None:
            try:
                from app.src.agent import get_llm
                self._llm=get_llm()

            except Exception as e:
                # LLM 初始化失败，默认允许
                return GuardrailResult(action=GuardrailAction.ALLOW)

        prompt = f"""判断以下用户输入是否在中医养生服务范围内。

用户输入: {user_input}

服务范围：
- 中医养生、体质调理、症状分析
- 中药材、方剂知识查询
- 中医经典古籍解读
- 日常保健建议

超出范围：
- 西医诊疗（手术、CT、西药等）
- 紧急医疗情况
- 非医学话题（天气、娱乐、编程等）

请直接返回以下选项之一（只返回选项，不要解释）：
- ALLOW: 在服务范围内
- BLOCK_EMERGENCY: 紧急情况
- BLOCK_OOS: 超出范围
- CLARIFY: 需要澄清"""

        try:
            response = self._llm.invoke([SystemMessage(content=prompt)])
            action_str = response.content.strip().upper().split()[0]

            if action_str == "BLOCK_EMERGENCY":
                return GuardrailResult(
                    action=GuardrailAction.BLOCK_EMERGENCY,
                    reason="LLM判断：紧急情况",
                    response=self._get_emergency_response("您描述的情况"),
                    confidence=0.7,
                )
            elif action_str == "BLOCK_OOS":
                return GuardrailResult(
                    action=GuardrailAction.BLOCK_OOS,
                    reason="LLM判断：超出服务范围",
                    response=self._get_out_of_scope_response("您的问题"),
                    confidence=0.7,
                )
            elif action_str == "CLARIFY":
                return GuardrailResult(
                    action=GuardrailAction.CLARIFY,
                    reason="LLM判断：需要澄清",
                    response=self._get_clarify_response(),
                    confidence=0.6,
                )
            else:  # ALLOW or unknown
                return GuardrailResult(action=GuardrailAction.ALLOW)

        except Exception as e:
            # LLM 调用失败，默认允许
            return GuardrailResult(action=GuardrailAction.ALLOW)

    # ==================== 输出过滤 ====================

    def _filter_output(self, content: str) -> tuple[str, bool]:
        """过滤输出中的敏感内容"""
        filtered = content
        was_filtered = False

        for keyword in self.SENSITIVE_OUTPUT_KEYWORDS:
            if keyword in filtered:
                filtered = filtered.replace(keyword, "[内容已调整]")
                was_filtered = True

        return filtered, was_filtered

    # ==================== 响应模板 ====================

    def _get_emergency_response(self, matched: str) -> str:
        """紧急情况响应"""
        return f"""
⚠️ **紧急情况提醒**

检测到您描述的情况（{matched}）可能是紧急医疗状况，请立即：

1. **拨打 120 急救电话** 或前往最近医院急诊
2. 保持冷静，避免剧烈活动
3. 如有家人请立即通知陪同

**重要提示：**
中医调理适用于慢性病调养和日常保健，**不适用于急症处理**。
您目前的情况需要专业医疗救治，请立即就医！

---
如果情况稳定后，您仍有中医调理方面的问题，欢迎继续咨询。
"""

    def _get_medication_safety_response(
        self,
        *,
        prohibited: list[str],
        caution: list[str],
        pregnancy: bool,
    ) -> str:
        prohibited_text = "、".join(prohibited) if prohibited else "无"
        caution_text = "、".join(caution) if caution else "无"
        context = "妊娠期" if pregnancy else "当前咨询"
        return f"""
⚠️ **高风险用药提醒**

您提到的药材中，禁用或高风险药材包括：**{prohibited_text}**；慎用药材包括：**{caution_text}**。

在{context}情况下，我不能提供自行服用的克数、频次或配伍方案。请立即停止自行尝试，并携带药名、药品包装和已经服用的时间/剂量，尽快咨询产科医生、中医师或药师。

如果已经误服后出现腹痛、阴道出血、剧烈呕吐腹泻、心悸、头晕或其他明显不适，请立即前往急诊或拨打 120。
"""

    @staticmethod
    def _get_high_risk_dosage_response(group: str) -> str:
        return f"""
⚠️ **高风险用药提醒**

您当前描述涉及**{group}**。在没有面诊、完整病史、检查结果和现用药清单的情况下，
我不能提供可自行执行的中药处方、每味克数、服用频次或加减方案。

请不要自行购药或停改现有治疗，建议携带全部药品包装和检查资料，尽快由执业中医师、
相关专科医生或药师进行面对面评估。若已经服用后出现明显不适，请及时就医；症状严重时拨打 120。
"""

    def _get_out_of_scope_response(self, matched: str) -> str:
        """超范围响应"""
        return f"""
您的问题涉及「{matched}」，这属于西医诊疗范畴。

**我是中医养生助手，主要服务范围包括：**
- 🌿 中医养生调理建议
- 🔍 体质辨识与调养方案
- 💊 中药材、方剂知识查询
- 🩺 常见症状的中医分析
- 📚 中医经典古籍解读

**关于您的问题，建议您：**
- 咨询西医医生或前往医院相关科室
- 如需中医辅助调理，请在西医诊断明确后再咨询

如有中医相关问题，欢迎继续咨询！
"""

    def _get_chitchat_response(self) -> str:
        """闲聊响应"""
        return """
您好！我是中医养生助手 🌿

我主要为您提供中医相关的健康咨询服务，包括：
- 日常养生调理建议
- 体质辨识与调养
- 常见症状的中医分析
- 药材、方剂知识查询

请问有什么中医养生或健康方面的问题需要帮助吗？
"""

    def _get_clarify_response(self) -> str:
        """澄清响应"""
        return """
您的问题我不太确定理解对了，能否再具体描述一下？

**例如：**
- 如果是身体不适，请描述具体症状、持续时间、伴随表现
- 如果是咨询药材，请告诉我药材名称或您想了解的方面
- 如果是养生问题，请说明您的具体需求或关注点

这样我能更准确地为您提供帮助！
"""


# ==================== 工厂函数 ====================

def get_tcm_guardrails_middleware(
    use_llm_fallback: bool = True,  # 默认启用 LLM 兜底
    **kwargs
) -> TCMGuardrailsMiddleware:
    """
    获取 TCM 守卫中间件实例

    Args:
        use_llm_fallback: 是否启用 LLM 兜底（默认 True）
        **kwargs: 其他配置参数

    Returns:
        TCMGuardrailsMiddleware 实例
    """
    return TCMGuardrailsMiddleware(
        use_llm_fallback=use_llm_fallback,
        **kwargs
    )
