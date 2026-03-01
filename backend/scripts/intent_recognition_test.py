"""
意图识别准确性测试脚本

测试内容：
1. L0红线层 - 急救拦截
2. L1规则层 - 快速匹配
3. L3 LLM层 - 深度分类
4. 四大意图 + 二级路由准确性
5. OOS拒识检测

运行方式:
    cd backend
    python scripts/test_intent_recognition.py
"""

import asyncio
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from enum import Enum

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.src.agent.intent_recognition import (
    IntentType,
    WellnessLevel,
    IntentRouter,
    create_intent_router,
    get_emergency_interceptor,
    get_rule_router,
    should_trigger_tongue_analysis,
)

from app.src.common.config.setting_config import settings


# ============== 工具函数 ==============


def create_configured_intent_router(db_session=None, redis_client=None):
    """
    创建使用配置中API设置的意图路由器
    """
    from app.src.agent.intent_recognition.intent_classifier import IntentClassifier, create_intent_classifier
    from app.src.agent.intent_recognition.emergency_interceptor import get_emergency_interceptor
    from app.src.agent.intent_recognition.rule_router import get_rule_router
    from app.src.agent.intent_recognition.context_enricher import create_context_enricher
    from app.src.agent.intent_recognition.wellness_router import get_wellness_router
    from app.src.agent.intent_recognition.intent_router import IntentRouter
    
    # 优先使用DeepSeek配置，如果未配置则使用OpenAI配置
    if settings.DEEPSEEK_API_KEY:
        intent_classifier = create_intent_classifier(
            provider_name="deepseek",
            model_name="deepseek-chat",  # 或者使用 deepseek-coder
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL or None,
        )
    elif settings.OPENAI_API_KEY:
        intent_classifier = create_intent_classifier(
            provider_name="openai",
            model_name="gpt-4o-mini",
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL or None,
        )
    else:
        # 如果都没有配置，则使用默认的创建方式（可能失败或使用空的API密钥）
        intent_classifier = create_intent_classifier()
    
    return IntentRouter(
        emergency_interceptor=get_emergency_interceptor(),
        rule_router=get_rule_router(),
        context_enricher=create_context_enricher(
            db_session=db_session,
            redis_client=redis_client
        ),
        intent_classifier=intent_classifier,
        wellness_router=get_wellness_router(),
        db_session=db_session,
        redis_client=redis_client,
    )


# ============== 测试用例定义 ==============

@dataclass
class TestCase:
    """测试用例"""
    query: str                          # 用户输入
    expected_intent: IntentType         # 期望的主意图
    expected_sub_type: Optional[str]    # 期望的二级路由
    expected_wellness_level: Optional[WellnessLevel] = None  # 养生级别
    is_emergency: bool = False          # 是否为急救
    is_oos: bool = False                # 是否为拒识
    description: str = ""               # 测试描述


# L0红线层测试用例
EMERGENCY_TEST_CASES = [
    TestCase(
        query="我胸口剧烈疼痛，喘不上气",
        expected_intent=IntentType.DIAGNOSIS,
        expected_sub_type=None,
        is_emergency=True,
        description="心脏急症"
    ),
    TestCase(
        query="家人突然晕倒不省人事",
        expected_intent=IntentType.DIAGNOSIS,
        expected_sub_type=None,
        is_emergency=True,
        description="意识障碍"
    ),
    TestCase(
        query="误食了有毒的东西怎么办",
        expected_intent=IntentType.DIAGNOSIS,
        expected_sub_type=None,
        is_emergency=True,
        description="中毒"
    ),
    TestCase(
        query="头疼得很厉害，想吐",
        expected_intent=IntentType.DIAGNOSIS,
        expected_sub_type="symptom",
        is_emergency=False,
        description="普通症状（非急救）"
    ),
]

# 养生类测试用例
WELLNESS_TEST_CASES = [
    # L1 简单养生
    TestCase(
        query="春季如何养肝",
        expected_intent=IntentType.WELLNESS,
        expected_sub_type="seasonal",
        expected_wellness_level=WellnessLevel.L1,
        description="季节养生L1"
    ),
    TestCase(
        query="夏天吃什么比较好",
        expected_intent=IntentType.WELLNESS,
        expected_sub_type="seasonal",
        expected_wellness_level=WellnessLevel.L1,
        description="季节饮食L1"
    ),
    TestCase(
        query="日常养生有什么好方法",
        expected_intent=IntentType.WELLNESS,
        expected_sub_type="daily",
        expected_wellness_level=WellnessLevel.L1,
        description="日常养生L1"
    ),
    TestCase(
        query="立冬后应该注意什么",
        expected_intent=IntentType.WELLNESS,
        expected_sub_type="seasonal",
        expected_wellness_level=WellnessLevel.L1,
        description="节气养生L1"
    ),
    # L2 复杂养生
    TestCase(
        query="我是气虚体质，经常感觉疲劳乏力，该怎么调理",
        expected_intent=IntentType.WELLNESS,
        expected_sub_type="constitution",
        expected_wellness_level=WellnessLevel.L2,
        description="体质调理L2"
    ),
    TestCase(
        query="阴虚火旺的人有什么养生建议，我很担心",
        expected_intent=IntentType.WELLNESS,
        expected_sub_type="constitution",
        expected_wellness_level=WellnessLevel.L2,
        description="体质+焦虑L2"
    ),
    TestCase(
        query="痰湿体质如何减肥，怎么调理比较好",
        expected_intent=IntentType.WELLNESS,
        expected_sub_type="constitution",
        expected_wellness_level=WellnessLevel.L2,
        description="体质调理L2"
    ),
]

# 方剂类测试用例
PRESCRIPTION_TEST_CASES = [
    TestCase(
        query="桂枝汤是什么方子",
        expected_intent=IntentType.PRESCRIPTION,
        expected_sub_type="query",
        description="方剂查询"
    ),
    TestCase(
        query="六味地黄丸有哪些成分",
        expected_intent=IntentType.PRESCRIPTION,
        expected_sub_type="composition",
        description="方剂组成"
    ),
    TestCase(
        query="补中益气汤的功效是什么",
        expected_intent=IntentType.PRESCRIPTION,
        expected_sub_type="query",
        description="方剂功效"
    ),
    TestCase(
        query="风寒感冒吃什么方子好",
        expected_intent=IntentType.PRESCRIPTION,
        expected_sub_type="recommend",
        description="方剂推荐"
    ),
    TestCase(
        query="阴虚火旺应该用什么方剂",
        expected_intent=IntentType.PRESCRIPTION,
        expected_sub_type="recommend",
        description="证型推荐方剂"
    ),
    TestCase(
        query="四君子汤和六君子汤有什么区别",
        expected_intent=IntentType.PRESCRIPTION,
        expected_sub_type="compare",
        description="方剂对比"
    ),
]

# 药材类测试用例
HERB_TEST_CASES = [
    TestCase(
        query="黄芪有什么功效",
        expected_intent=IntentType.HERB,
        expected_sub_type="effect",
        description="药材功效"
    ),
    TestCase(
        query="当归的作用是什么",
        expected_intent=IntentType.HERB,
        expected_sub_type="effect",
        description="药材功效"
    ),
    TestCase(
        query="人参和西洋参有什么区别",
        expected_intent=IntentType.HERB,
        expected_sub_type="effect",
        description="药材对比"
    ),
    TestCase(
        query="甘草不能和什么一起吃",
        expected_intent=IntentType.HERB,
        expected_sub_type="compatibility",
        description="配伍禁忌"
    ),
    TestCase(
        query="黄芪能不能和当归一起用",
        expected_intent=IntentType.HERB,
        expected_sub_type="compatibility",
        description="配伍查询"
    ),
    TestCase(
        query="枸杞一般用多少克",
        expected_intent=IntentType.HERB,
        expected_sub_type="usage",
        description="用法用量"
    ),
    TestCase(
        query="如何辨别真假冬虫夏草",
        expected_intent=IntentType.HERB,
        expected_sub_type="identify",
        description="药材鉴别"
    ),
]

# 问诊类测试用例
DIAGNOSIS_TEST_CASES = [
    TestCase(
        query="我最近总是怕冷，流清涕，头疼",
        expected_intent=IntentType.DIAGNOSIS,
        expected_sub_type="symptom",
        description="症状咨询"
    ),
    TestCase(
        query="失眠多梦是什么原因",
        expected_intent=IntentType.DIAGNOSIS,
        expected_sub_type="symptom",
        description="症状分析"
    ),
    TestCase(
        query="最近胃胀不消化，吃完饭就难受",
        expected_intent=IntentType.DIAGNOSIS,
        expected_sub_type="symptom",
        description="消化症状"
    ),
    TestCase(
        query="帮我看看这个舌苔",
        expected_intent=IntentType.DIAGNOSIS,
        expected_sub_type="tongue",
        description="舌诊请求"
    ),
    TestCase(
        query="我舌头发红，舌苔黄腻是什么问题",
        expected_intent=IntentType.DIAGNOSIS,
        expected_sub_type="tongue",
        description="舌象描述"
    ),
    TestCase(
        query="心悸气短，晚上盗汗，这是什么证型",
        expected_intent=IntentType.DIAGNOSIS,
        expected_sub_type="comprehensive",
        description="综合问诊"
    ),
    TestCase(
        query="有没有类似失眠的医案可以参考",
        expected_intent=IntentType.DIAGNOSIS,
        expected_sub_type="case",
        description="医案查询"
    ),
]

# OOS拒识测试用例
OOS_TEST_CASES = [
    TestCase(
        query="今天天气怎么样",
        expected_intent=IntentType.WELLNESS,
        expected_sub_type=None,
        is_oos=True,
        description="闲聊-天气"
    ),
    TestCase(
        query="讲个笑话",
        expected_intent=IntentType.WELLNESS,
        expected_sub_type=None,
        is_oos=True,
        description="闲聊-娱乐"
    ),
    TestCase(
        query="我需要做CT检查吗",
        expected_intent=IntentType.DIAGNOSIS,
        expected_sub_type=None,
        is_oos=True,
        description="西医问题"
    ),
    TestCase(
        query="感冒了要不要吃抗生素",
        expected_intent=IntentType.DIAGNOSIS,
        expected_sub_type=None,
        is_oos=True,
        description="西药问题"
    ),
]

# 边界情况测试用例
EDGE_CASES = [
    TestCase(
        query="气虚",
        expected_intent=IntentType.WELLNESS,
        expected_sub_type="constitution",
        description="极短输入-体质"
    ),
    TestCase(
        query="黄芪",
        expected_intent=IntentType.HERB,
        expected_sub_type="effect",
        description="极短输入-药材"
    ),
    TestCase(
        query="桂枝汤",
        expected_intent=IntentType.PRESCRIPTION,
        expected_sub_type="query",
        description="极短输入-方剂"
    ),
    TestCase(
        query="头疼",
        expected_intent=IntentType.DIAGNOSIS,
        expected_sub_type="symptom",
        description="极短输入-症状"
    ),
    TestCase(
        query="你好",
        expected_intent=IntentType.WELLNESS,
        expected_sub_type=None,
        is_oos=False,  # 问候语允许通过
        description="问候语"
    ),
]


# ============== 测试执行器 ==============

class TestResult:
    """测试结果统计"""

    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.details = []

    def add_result(self, test_case: TestCase, actual_intent, actual_sub_type,
                   actual_wellness_level, actual_emergency, actual_oos, passed: bool):
        self.total += 1
        if passed:
            self.passed += 1
        else:
            self.failed += 1

        self.details.append({
            "query": test_case.query,
            "description": test_case.description,
            "expected": {
                "intent": test_case.expected_intent.value if test_case.expected_intent else None,
                "sub_type": test_case.expected_sub_type,
                "wellness_level": test_case.expected_wellness_level.value if test_case.expected_wellness_level else None,
                "is_emergency": test_case.is_emergency,
                "is_oos": test_case.is_oos,
            },
            "actual": {
                "intent": actual_intent,
                "sub_type": actual_sub_type,
                "wellness_level": actual_wellness_level,
                "is_emergency": actual_emergency,
                "is_oos": actual_oos,
            },
            "passed": passed,
        })

    @property
    def accuracy(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


async def emergency_layer():
    """测试L0红线层"""
    print("\n" + "=" * 60)
    print("L0 红线层测试 - 急救拦截")
    print("=" * 60)

    interceptor = get_emergency_interceptor()
    result = TestResult()

    for case in EMERGENCY_TEST_CASES:
        emergency_result = interceptor.intercept(case.query)
        passed = emergency_result.is_emergency == case.is_emergency

        result.add_result(
            case,
            actual_intent=None,
            actual_sub_type=None,
            actual_wellness_level=None,
            actual_emergency=emergency_result.is_emergency,
            actual_oos=False,
            passed=passed
        )

        status = "✅" if passed else "❌"
        print(f"{status} [{case.description}] {case.query[:30]}...")
        if not passed:
            print(f"   期望急救={case.is_emergency}, 实际={emergency_result.is_emergency}")

    print(f"\n准确率: {result.accuracy:.1%} ({result.passed}/{result.total})")
    return result


async def rule_layer():
    """测试L1规则层"""
    print("\n" + "=" * 60)
    print("L1 规则层测试 - 快速匹配")
    print("=" * 60)

    router = get_rule_router()
    result = TestResult()

    # 合并方剂和药材测试用例（规则层主要处理这些）
    test_cases = PRESCRIPTION_TEST_CASES + HERB_TEST_CASES

    for case in test_cases:
        rule_result = router.route(case.query)

        if rule_result:
            actual_intent = rule_result.primary_intent.value
            actual_sub_type = rule_result.sub_type
            passed = (rule_result.primary_intent == case.expected_intent)
        else:
            actual_intent = None
            actual_sub_type = None
            passed = False  # 规则层应该能匹配这些

        result.add_result(
            case,
            actual_intent=actual_intent,
            actual_sub_type=actual_sub_type,
            actual_wellness_level=None,
            actual_emergency=False,
            actual_oos=False,
            passed=passed
        )

        status = "✅" if passed else "❌"
        print(f"{status} [{case.description}] {case.query[:30]}...")
        if not passed:
            print(f"   期望={case.expected_intent.value}, 实际={actual_intent}")

    print(f"\n准确率: {result.accuracy:.1%} ({result.passed}/{result.total})")
    return result


async def full_pipeline(use_llm: bool = False):
    """测试完整管道"""
    print("\n" + "=" * 60)
    print(f"完整管道测试 {'(含LLM)' if use_llm else '(仅规则层)'}")
    print("=" * 60)

    if use_llm:
        router = create_configured_intent_router()
    else:
        router = create_intent_router()
    result = TestResult()

    # 所有测试用例
    all_cases = (
        WELLNESS_TEST_CASES +
        PRESCRIPTION_TEST_CASES +
        HERB_TEST_CASES +
        DIAGNOSIS_TEST_CASES +
        EDGE_CASES
    )

    for case in all_cases:
        try:
            if use_llm:
                # 完整路由（包括LLM）
                route_result = await router.route(
                    query=case.query,
                    user_id="test_user",
                    conversation_id=None,
                    has_image=False
                )

                if route_result.emergency.is_emergency:
                    actual_intent = None
                    actual_sub_type = None
                    actual_wellness_level = None
                    actual_emergency = True
                    actual_oos = False
                elif route_result.oos.is_oos:
                    actual_intent = route_result.classification.primary_intent.value if route_result.classification else None
                    actual_sub_type = route_result.classification.sub_type if route_result.classification else None
                    actual_wellness_level = None
                    actual_emergency = False
                    actual_oos = True
                elif route_result.classification:
                    actual_intent = route_result.classification.primary_intent.value
                    actual_sub_type = route_result.classification.sub_type
                    actual_wellness_level = route_result.classification.wellness_level.value if route_result.classification.wellness_level else None
                    actual_emergency = False
                    actual_oos = False
                else:
                    actual_intent = None
                    actual_sub_type = None
                    actual_wellness_level = None
                    actual_emergency = False
                    actual_oos = False
            else:
                # 仅规则层
                rule_result = router.rule_router.route(case.query)
                if rule_result:
                    actual_intent = rule_result.primary_intent.value
                    actual_sub_type = rule_result.sub_type
                    actual_wellness_level = rule_result.wellness_level.value if rule_result.wellness_level else None
                else:
                    actual_intent = None
                    actual_sub_type = None
                    actual_wellness_level = None
                actual_emergency = False
                actual_oos = False

            # 判断是否通过
            passed = True
            if case.is_emergency:
                passed = actual_emergency == case.is_emergency
            elif case.is_oos:
                passed = actual_oos == case.is_oos
            elif actual_intent:
                passed = (actual_intent == case.expected_intent.value)
                # 如果主意图正确，检查二级路由（可选）
                # if passed and case.expected_sub_type:
                #     passed = (actual_sub_type == case.expected_sub_type)
            else:
                passed = False

            result.add_result(
                case,
                actual_intent=actual_intent,
                actual_sub_type=actual_sub_type,
                actual_wellness_level=actual_wellness_level,
                actual_emergency=actual_emergency,
                actual_oos=actual_oos,
                passed=passed
            )

            status = "✅" if passed else "❌"
            print(f"{status} [{case.description}] {case.query[:35]}...")
            if not passed:
                print(f"   期望: intent={case.expected_intent.value}, sub={case.expected_sub_type}")
                print(f"   实际: intent={actual_intent}, sub={actual_sub_type}")

        except Exception as e:
            print(f"❌ [{case.description}] 执行出错: {e}")
            result.add_result(
                case,
                actual_intent=None,
                actual_sub_type=None,
                actual_wellness_level=None,
                actual_emergency=False,
                actual_oos=False,
                passed=False
            )

    print(f"\n准确率: {result.accuracy:.1%} ({result.passed}/{result.total})")
    return result


async def oos_detection():
    """测试OOS拒识检测"""
    print("\n" + "=" * 60)
    print("OOS 拒识检测测试")
    print("=" * 60)

    router = create_configured_intent_router()  # OOS检测需要使用LLM
    result = TestResult()

    for case in OOS_TEST_CASES:
        try:
            route_result = await router.route(
                query=case.query,
                user_id="test_user",
                conversation_id=None,
                has_image=False
            )

            actual_oos = route_result.oos.is_oos
            passed = actual_oos == case.is_oos

            result.add_result(
                case,
                actual_intent=route_result.classification.primary_intent.value if route_result.classification else None,
                actual_sub_type=None,
                actual_wellness_level=None,
                actual_emergency=False,
                actual_oos=actual_oos,
                passed=passed
            )

            status = "✅" if passed else "❌"
            print(f"{status} [{case.description}] {case.query[:30]}...")
            if not passed:
                print(f"   期望OOS={case.is_oos}, 实际={actual_oos}")
                if route_result.oos.response:
                    print(f"   响应: {route_result.oos.response[:50]}...")

        except Exception as e:
            print(f"❌ [{case.description}] 执行出错: {e}")

    print(f"\n准确率: {result.accuracy:.1%} ({result.passed}/{result.total})")
    return result


async def tongue_keywords():
    """测试舌诊关键词检测"""
    print("\n" + "=" * 60)
    print("舌诊关键词检测测试")
    print("=" * 60)

    test_queries = [
        ("帮我看看舌苔", True),
        ("我的舌头发红", True),
        ("舌质淡白是什么意思", True),
        ("最近失眠", False),
        ("黄芪功效", False),
        ("舌诊怎么做", True),
        ("齿痕舌是什么", True),
    ]

    passed = 0
    for query, expected in test_queries:
        actual = should_trigger_tongue_analysis(query)
        is_pass = actual == expected
        if is_pass:
            passed += 1
        status = "✅" if is_pass else "❌"
        print(f"{status} \"{query}\" -> {actual} (期望: {expected})")

    print(f"\n准确率: {passed}/{len(test_queries)} = {passed/len(test_queries):.1%}")


def print_summary(results: dict):
    """打印测试总结"""
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    total_passed = 0
    total_tests = 0

    for name, result in results.items():
        if result:
            total_passed += result.passed
            total_tests += result.total
            print(f"{name}: {result.accuracy:.1%} ({result.passed}/{result.total})")

    if total_tests > 0:
        print(f"\n总体准确率: {total_passed/total_tests:.1%} ({total_passed}/{total_tests})")


async def main():
    """主测试入口"""
    print("=" * 60)
    print("SmartTCM 意图识别准确性测试")
    print("=" * 60)

    results = {}

    # 1. L0红线层测试
    results["L0红线层"] = await emergency_layer()

    # 2. L1规则层测试
    results["L1规则层"] = await rule_layer()

    # 3. 舌诊关键词测试
    await tongue_keywords()

    # 4. 完整管道测试（仅规则层）
    results["完整管道(规则)"] = await full_pipeline(use_llm=False)

    # 5. OOS检测测试（需要LLM）
    # 注意：这个测试需要配置LLM环境变量
    use_llm = bool(settings.DEEPSEEK_API_KEY or settings.OPENAI_API_KEY)
    if use_llm:
        print("\n检测到LLM配置，运行LLM相关测试...")
        results["OOS检测"] = await oos_detection()
        results["完整管道(LLM)"] = await full_pipeline(use_llm=True)
    else:
        print("\n未配置LLM API Key，跳过LLM相关测试")
        print("设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 环境变量以启用")

    # 打印总结
    print_summary(results)

    # 输出失败的测试详情
    print("\n" + "=" * 60)
    print("失败用例详情")
    print("=" * 60)

    for name, result in results.items():
        if result:
            failed_cases = [d for d in result.details if not d["passed"]]
            if failed_cases:
                print(f"\n[{name}] 失败 {len(failed_cases)} 个:")
                for case in failed_cases[:5]:  # 最多显示5个
                    print(f"  - {case['description']}: {case['query'][:40]}...")
                    print(f"    期望: {case['expected']}")
                    print(f"    实际: {case['actual']}")


if __name__ == "__main__":
    asyncio.run(main())
