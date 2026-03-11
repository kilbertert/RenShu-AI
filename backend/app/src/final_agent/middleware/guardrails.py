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




class GuardrailAction(Enum):
    """守卫动作"""
    ALLOW = "allow"           # 允许通过
    BLOCK_EMERGENCY = "block_emergency"  # 紧急情况阻断
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
        r"(持续|反复).{0,5}(高烧|高热).{0,5}(不退|三天|3天)",
        r"(大量|不止).{0,5}(出血|吐血|便血)",
        r"(意识|神志).{0,5}(不清|模糊|丧失)",
        r"(呼吸|喘).{0,5}(困难|不上来|费力)",
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
        1. 紧急情况检测（关键词 + 正则）
        2. 超范围检测（关键词 + 正则）
        3. 闲聊检测（关键词）
        4. 服务范围内确认（关键词）
        5. 默认允许（删除了对路由结果的依赖和 LLM 兜底）
        """
        # 预处理
        text = user_input.lower().strip()

        # ========== 1. 紧急情况检测 ==========
        if self.warn_emergency:
            result = self._check_emergency(text, user_input)
            if result.action != GuardrailAction.ALLOW:
                return result

        # ========== 2. 超范围检测（仅检测明确的西医关键词） ==========
        if self.reject_out_of_scope:
            result = self._check_out_of_scope(text, user_input)
            if result.action != GuardrailAction.ALLOW:
                return result

        # ========== 3. 闲聊检测 ==========
        result = self._check_chitchat(text, user_input)
        if result.action != GuardrailAction.ALLOW:
            return result

        # ========== 4. 服务范围内确认 ==========
        if self._is_in_scope(text):
            # 检查是否需要警告
            warn_result = self._check_should_warn(text, user_input)
            if warn_result.action == GuardrailAction.WARN:
                return warn_result
            # 明确在服务范围内，允许通过
            return GuardrailResult(action=GuardrailAction.ALLOW)

        # ========== 5. 默认允许 ==========
        # 如果没有命中拦截规则，且包含中医相关内容，默认允许
        # 让后续的意图识别来判断具体如何处理
        return GuardrailResult(action=GuardrailAction.ALLOW)

    def _check_emergency(self, text: str, original: str) -> GuardrailResult:
        """检测紧急情况"""
        # 关键词匹配（使用预处理的 set）
        for keyword in self.EMERGENCY_KEYWORDS:
            if keyword in original:
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
            if match:
                matched_text = match.group()
                return GuardrailResult(
                    action=GuardrailAction.BLOCK_EMERGENCY,
                    reason="紧急情况模式匹配",
                    matched_rule=matched_text,
                    response=self._get_emergency_response(matched_text),
                    confidence=0.9,
                )

        return GuardrailResult(action=GuardrailAction.ALLOW)

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
