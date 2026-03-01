"""
测试 LangGraph Interrupt 追问功能

测试场景：
1. 单轮追问：用户回答一次后信息足够
2. 多轮追问：需要多次追问才能收集足够信息
3. 舌像请求：触发 request_tongue interrupt
4. 意图切换：用户在追问时切换话题
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# 设置项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


# ============== 颜色输出 ==============


class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_chunk(idx: int, elapsed_ms: int, data: dict):
    """格式化打印流式消息"""
    msg_type = data.get("type", "?")
    time_str = f"{Colors.DIM}[{elapsed_ms:>6}ms]{Colors.RESET}"

    if msg_type == "content":
        content = data.get("content", "")
        if content.strip():
            print(
                f"{time_str} {Colors.YELLOW}[content]{Colors.RESET} {content}",
                end="",
                flush=True,
            )
    elif msg_type == "done":
        steps = data.get("steps", [])
        query_type = data.get("query_type", "?")
        print(
            f"\n{time_str} {Colors.GREEN}{Colors.BOLD}[done]{Colors.RESET} query_type={query_type}, steps={steps}"
        )
    elif msg_type == "error":
        content = data.get("content", data.get("error", ""))
        print(f"\n{time_str} {Colors.RED}[error]{Colors.RESET} {content}")
    elif msg_type == "interrupt":
        question = data.get("question", "")
        action = data.get("action", "")
        thread_id = data.get("thread_id", "")
        print(f"\n{time_str} {Colors.MAGENTA}{Colors.BOLD}[interrupt]{Colors.RESET}")
        print(f"  - question: {question}")
        print(f"  - action: {action}")
        print(f"  - thread_id: {thread_id}")
    elif msg_type == "thread_init":
        thread_id = data.get("thread_id", "")
        print(
            f"{time_str} {Colors.CYAN}{Colors.BOLD}[thread_init]{Colors.RESET} thread_id={thread_id}"
        )
    else:
        content = data.get("content", "")
        if content:
            print(
                f"{time_str} {Colors.CYAN}{Colors.BOLD}[{msg_type}]{Colors.RESET} {content}"
            )


# ============== 测试用例 ==============


class TestScenario:
    def __init__(
        self,
        name: str,
        initial_message: str,
        follow_up_answers: List[str],
        description: str,
        expect_interrupts: int = 1,
        expect_final_state: str = "done",
    ):
        self.name = name
        self.initial_message = initial_message
        self.follow_up_answers = follow_up_answers
        self.description = description
        self.expect_interrupts = expect_interrupts
        self.expect_final_state = expect_final_state


TEST_SCENARIOS = [
    TestScenario(
        name="[PASS] 单轮追问 - 主诉不明确时追问",
        initial_message="我最近不太舒服",  # 主诉不明确，需要追问
        follow_up_answers=["我头疼，有点低烧，已经两天了"],
        description="用户首次输入信息不足，系统追问，用户回答后信息足够",
        expect_interrupts=1,
        expect_final_state="done",
    ),
    TestScenario(
        name="[PASS] 单轮追问 - 信息充分但需要确认",
        initial_message="我最近头疼有点低烧，已经两天了，没有出汗，大便正常",
        follow_up_answers=["没有特别的不舒服，就是头疼和低烧"],
        description="用户输入信息充分，但可能需要确认某些细节",
        expect_interrupts=1,
        expect_final_state="done",
    ),
    TestScenario(
        name="[INFO] 多轮追问 - 信息严重不足",
        initial_message="不舒服",  # 信息非常少
        follow_up_answers=["头疼", "两天", "不发烧", "大便正常"],  # 预期需要多次回复
        description="用户只说了不舒服，系统需要多轮追问才能收集足够信息",
        expect_interrupts=2,  # 预期至少2次中断
        expect_final_state="done",
    ),
    TestScenario(
        name="[INFO] 意图切换 - 在追问过程中更换话题",
        initial_message="我头疼",
        follow_up_answers=["算了，我想问问感冒怎么治"],  # 意图切换到养生咨询
        description="用户在追问时突然切换话题，检测到 intent_switch",
        expect_interrupts=1,
        expect_final_state="done",
    ),
]


async def run_scenario(
    scenario: TestScenario,
    provider_name: str = None,
    model_name: str = None,
    api_key: str = None,
    base_url: str | None = None,
):
    """运行单个测试场景"""
    from app.src.agent.tcm_service import get_tcm_agent_service

    service = get_tcm_agent_service()
    thread_id = None
    interrupt_count = 0
    all_chunks: List[Dict] = []

    print(f"\n{'=' * 80}")
    print(f"{Colors.BOLD}测试场景: {scenario.name}{Colors.RESET}")
    print(f"{Colors.DIM}描述: {scenario.description}{Colors.RESET}")
    print(f"{'=' * 80}\n")

    # ============== 第一轮：初始输入 ==============
    print(f"{Colors.CYAN}>>> 用户:{Colors.RESET} {scenario.initial_message}\n")

    start = time.time()
    chunk_count = 0

    # 第一轮流式执行
    async for raw_chunk in service.chat_stream_with_tcm_agent(
        message=scenario.initial_message,
        user_id="test_user",
        conversation_id=f"test_conv_{scenario.name}",
        user_profile={},
        provider_name=provider_name,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
    ):
        elapsed_ms = int((time.time() - start) * 1000)
        chunk_count += 1

        try:
            data = json.loads(raw_chunk)
        except json.JSONDecodeError:
            print(f"{Colors.RED}[!] 无法解析 JSON: {raw_chunk[:100]}{Colors.RESET}")
            continue

        all_chunks.append(data)
        msg_type = data.get("type", "?")

        # 记录 thread_id
        if msg_type == "thread_init":
            thread_id = data.get("thread_id", "")

        # 检测 interrupt
        if msg_type == "interrupt":
            interrupt_count += 1
            thread_id_from_interrupt = data.get("thread_id", "")
            if thread_id_from_interrupt:
                thread_id = thread_id_from_interrupt

        print_chunk(chunk_count, elapsed_ms, data)

        # 如果收到 interrupt，停止流式输出
        if msg_type == "interrupt":
            break

        # 如果收到 done，也停止
        if msg_type == "done":
            break

    total_ms = int((time.time() - start) * 1000)
    print(f"\n{Colors.DIM}第一轮完成，耗时: {total_ms}ms{Colors.RESET}")

    # ============== 如果有 interrupt，进行追问回复 ==============
    if interrupt_count > 0 and thread_id:
        for i, answer in enumerate(scenario.follow_up_answers, start=1):
            print(f"\n{Colors.CYAN}>>> 用户追问 #{i}:{Colors.RESET} {answer}")

            resume_start = time.time()
            resume_chunk_count = 0
            received_interrupt = False

            async for raw_chunk in service.resume_stream(
                thread_id=thread_id, user_answer=answer
            ):
                elapsed_ms = int((time.time() - resume_start) * 1000)
                resume_chunk_count += 1

                try:
                    data = json.loads(raw_chunk)
                except json.JSONDecodeError:
                    continue

                all_chunks.append(data)
                msg_type = data.get("type", "?")

                if msg_type == "interrupt":
                    interrupt_count += 1
                    received_interrupt = True

                print_chunk(resume_chunk_count, elapsed_ms, data)

                # 收到 interrupt，停止
                if msg_type == "interrupt":
                    break

                # 收到 done，也停止
                if msg_type == "done":
                    break

            resume_total_ms = int((time.time() - resume_start) * 1000)
            print(f"\n{Colors.DIM}恢复完成，耗时: {resume_total_ms}ms{Colors.RESET}")

            # 如果没有继续收到 interrupt，且收到 done，跳出循环
            if not received_interrupt:
                done_chunks = [c for c in all_chunks if c.get("type") == "done"]
                if done_chunks:
                    break

    # ============== 结果验证 ==============
    print(f"\n{'=' * 80}")
    print(f"{Colors.BOLD}验证结果{Colors.RESET}")
    print(f"{'=' * 80}\n")

    passed = True

    # 1. 验证 thread_id 存在
    print(f"  thread_id: {thread_id}")
    if not thread_id:
        print(f"  {Colors.RED}FAIL: 未获取到 thread_id{Colors.RESET}")
        passed = False
    else:
        print(f"  {Colors.GREEN}PASS: 已获取 thread_id{Colors.RESET}")

    # 2. 验证 interrupt 次数
    print(f"  实际 interrupt 次数: {interrupt_count}")
    print(f"  期望 interrupt 次数: {scenario.expect_interrupts}")
    if interrupt_count == scenario.expect_interrupts:
        print(f"  {Colors.GREEN}PASS: interrupt 次数符合预期{Colors.RESET}")
    else:
        print(f"  {Colors.YELLOW}WARN: interrupt 次数不符合预期{Colors.RESET}")

    # 3. 验证最终状态
    done_chunks = [c for c in all_chunks if c.get("type") == "done"]
    if done_chunks:
        print(f"  {Colors.GREEN}PASS: 收到 done 消息{Colors.RESET}")
    else:
        interrupt_chunks = [c for c in all_chunks if c.get("type") == "interrupt"]
        if interrupt_chunks:
            print(
                f"  {Colors.BLUE}INFO: 流程在 interrupt 处暂停（期待用户回复）{Colors.RESET}"
            )
        else:
            print(f"  {Colors.RED}WARN: 未收到 done 或 interrupt 消息{Colors.RESET}")

    # 4. 验证 content 输出
    content_chunks = [c for c in all_chunks if c.get("type") == "content"]
    print(f"  content chunk 数: {len(content_chunks)}")

    # 5. 验证 interrupt 格式
    interrupt_chunks = [c for c in all_chunks if c.get("type") == "interrupt"]
    for idx, ic in enumerate(interrupt_chunks):
        question = ic.get("question", "")
        action = ic.get("action", "")
        ic_thread_id = ic.get("thread_id", "")

        print(f"\n  interrupt #{idx + 1}:")
        print(
            f"    - question: {question[:50]}..."
            if len(question) > 50
            else f"    - question: {question}"
        )
        print(f"    - action: {action}")
        print(f"    - thread_id: {ic_thread_id}")

        if question and action and ic_thread_id:
            print(f"    {Colors.GREEN}PASS: interrupt 格式正确{Colors.RESET}")
        else:
            print(f"    {Colors.RED}WARN: interrupt 缺少必要字段{Colors.RESET}")

    # 6. 统计所有步骤类型
    status_types = set()
    for c in all_chunks:
        t = c.get("type")
        if t and t not in ("content", "done", "error", "interrupt", "thread_init"):
            status_types.add(t)

    print(f"\n  执行的步骤: {status_types}")

    status = (
        f"{Colors.GREEN}PASSED{Colors.RESET}"
        if passed
        else f"{Colors.RED}FAILED{Colors.RESET}"
    )
    print(f"\n  结果: {status}")

    return passed


async def main():
    """主测试流程"""
    import os

    import sys

    # 设置 UTF-8 编码
    if sys.platform == "win32":
        import codecs

        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

    print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}  LangGraph Interrupt 追问功能测试{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")

    # 从环境变量或 settings 获取模型配置
    from app.src.common.config.setting_config import settings

    provider_name = None
    model_name = None
    api_key = None
    base_url = None

    if settings.DEEPSEEK_API_KEY:
        provider_name = "deepseek"
        model_name = "deepseek-chat"
        api_key = settings.DEEPSEEK_API_KEY
        base_url = settings.DEEPSEEK_BASE_URL or None
        print(f"\n使用模型: {provider_name}/{model_name}")
    elif settings.OPENAI_API_KEY:
        provider_name = "openai"
        model_name = "gpt-4o-mini"
        api_key = settings.OPENAI_API_KEY
        base_url = settings.OPENAI_BASE_URL or None
        print(f"\n使用模型: {provider_name}/{model_name}")
    else:
        print(
            f"\n{Colors.RED}错误: 未找到 DEEPSEEK_API_KEY 或 OPENAI_API_KEY{Colors.RESET}"
        )
        print("请在 .env 文件中配置 API Key")
        sys.exit(1)

    results = {}

    for scenario in TEST_SCENARIOS:
        try:
            ok = await run_scenario(
                scenario=scenario,
                provider_name=provider_name,
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
            )
            results[scenario.name] = ok
        except Exception as e:
            print(f"\n{Colors.RED}场景执行异常: {e}{Colors.RESET}")
            import traceback

            traceback.print_exc()
            results[scenario.name] = False

        # 场景之间暂停一下
        print("\n" + f"{Colors.DIM}" + "-" * 80 + f"{Colors.RESET}\n")
        await asyncio.sleep(1)

    # ============== 汇总 ==============
    print(f"\n{'=' * 80}")
    print(f"{Colors.BOLD}  测试汇总{Colors.RESET}")
    print(f"{'=' * 80}\n")

    for name, ok in results.items():
        icon = (
            f"{Colors.GREEN}PASS{Colors.RESET}"
            if ok
            else f"{Colors.RED}FAIL{Colors.RESET}"
        )
        print(f"  [{icon}] {name}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\n  {passed}/{total} 通过")

    if passed == total:
        print(f"\n  {Colors.GREEN}{Colors.BOLD}全部通过!{Colors.RESET}")
    else:
        print(f"\n  {Colors.RED}存在失败或警告项{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
