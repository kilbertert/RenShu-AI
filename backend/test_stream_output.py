"""
测试流式输出

验证：
1. 状态消息格式正确：{"type": "意图识别", "content": "正在识别您的意图..."}
2. 内容消息格式正确：{"type": "content", "content": "..."}
3. 子图节点事件能正确传播（诊断子图、养生子图）
4. 完成消息包含 steps 列表
5. SSE 不存在双重包装
"""

import asyncio
import json
import sys
import time
from pathlib import Path

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
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_chunk(idx: int, elapsed_ms: int, data: dict):
    """格式化打印流式消息"""
    msg_type = data.get("type", "?")
    time_str = f"{Colors.DIM}[{elapsed_ms:>6}ms]{Colors.RESET}"

    if msg_type == "content":
        # LLM token — 黄色，不换行累积
        content = data.get("content", "")
        print(f"{time_str} {Colors.YELLOW}[content]{Colors.RESET} {content}", end="", flush=True)
    elif msg_type == "done":
        steps = data.get("steps", [])
        query_type = data.get("query_type", "?")
        print(f"\n{time_str} {Colors.GREEN}{Colors.BOLD}[done]{Colors.RESET} query_type={query_type}, steps={steps}")
    elif msg_type == "error":
        content = data.get("content", data.get("error", ""))
        print(f"\n{time_str} {Colors.RED}[error]{Colors.RESET} {content}")
    else:
        # 状态消息 — 青色
        content = data.get("content", "")
        print(f"{time_str} {Colors.CYAN}{Colors.BOLD}[{msg_type}]{Colors.RESET} {content}")


# ============== 测试用例 ==============

TEST_CASES = [
    {
        "name": "一般性闲聊（走 general 流）",
        "message": "你好，请问什么是中医？",
        "expect_types": {"安全检查", "意图识别"},
    },
    {
        "name": "诊断问诊（走 diagnose 子图，验证子图事件传播）",
        "message": "我最近头晕乏力，还经常失眠多梦，食欲也不太好",
        "expect_types": {"安全检查", "意图识别", "辨证推理"},
    },
    {
        "name": "养生咨询（走 wellness 子图）",
        "message": "冬天应该怎么养生？有什么饮食建议吗？",
        "expect_types": {"安全检查", "意图识别"},
    },
]


async def run_stream_test(
    test_name: str,
    message: str,
    expect_types: set[str],
    provider_name: str = None,
    model_name: str = None,
    api_key: str = None,
    base_url: str = None,
):
    """运行单个流式测试"""
    from app.src.agent.tcm_service import get_tcm_agent_service

    service = get_tcm_agent_service()

    print(f"\n{'=' * 70}")
    print(f"{Colors.BOLD}测试: {test_name}{Colors.RESET}")
    print(f"输入: {message}")
    print(f"{'=' * 70}\n")

    start = time.time()
    chunk_count = 0
    content_chunks = 0
    status_types_seen: set[str] = set()
    all_chunks: list[dict] = []
    first_content_time = None

    try:
        async for raw_chunk in service.chat_stream_with_tcm_agent(
            message=message,
            user_id="test_user",
            conversation_id="test_conv_stream",
            user_profile={},
            provider_name=provider_name,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
        ):
            elapsed_ms = int((time.time() - start) * 1000)
            chunk_count += 1

            # 解析 JSON
            try:
                data = json.loads(raw_chunk)
            except json.JSONDecodeError:
                print(f"{Colors.RED}[!] 无法解析 JSON: {raw_chunk[:100]}{Colors.RESET}")
                continue

            all_chunks.append(data)
            msg_type = data.get("type", "?")

            # 统计
            if msg_type == "content":
                content_chunks += 1
                if first_content_time is None:
                    first_content_time = elapsed_ms
            elif msg_type not in ("done", "error"):
                status_types_seen.add(msg_type)

            print_chunk(chunk_count, elapsed_ms, data)

    except Exception as e:
        print(f"\n{Colors.RED}异常: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        return False

    total_ms = int((time.time() - start) * 1000)

    # ============== 结果验证 ==============
    print(f"\n\n{'-' * 70}")
    print(f"{Colors.BOLD}验证结果{Colors.RESET}")
    print(f"{'-' * 70}")

    passed = True

    # 1. 检查总 chunk 数
    print(f"  总 chunk 数: {chunk_count}")
    if chunk_count == 0:
        print(f"  {Colors.RED}FAIL: 没有收到任何 chunk{Colors.RESET}")
        passed = False

    # 2. 检查状态消息的 type 是中文
    print(f"  状态类型: {status_types_seen}")
    for t in status_types_seen:
        if t.isascii():
            print(f"  {Colors.RED}FAIL: 状态类型 '{t}' 不是中文{Colors.RESET}")
            passed = False

    # 3. 检查期望的状态类型
    missing_types = expect_types - status_types_seen
    if missing_types:
        print(f"  {Colors.YELLOW}WARN: 缺少期望的状态类型: {missing_types}{Colors.RESET}")
    else:
        print(f"  {Colors.GREEN}PASS: 期望的状态类型都已收到{Colors.RESET}")

    # 4. 检查 content chunk 数量（应该 > 0 表示 LLM 在流式输出）
    print(f"  内容 chunk 数: {content_chunks}")
    if content_chunks > 1:
        print(f"  {Colors.GREEN}PASS: LLM 确实在逐 token 流式输出{Colors.RESET}")
    elif content_chunks == 1:
        print(f"  {Colors.YELLOW}WARN: 只收到 1 个内容 chunk，可能不是真正的流式{Colors.RESET}")
    else:
        print(f"  {Colors.YELLOW}WARN: 没有收到内容 chunk（可能被子图吞了或节点返回 dict）{Colors.RESET}")

    # 5. 检查 done 消息
    done_chunks = [c for c in all_chunks if c.get("type") == "done"]
    if done_chunks:
        steps = done_chunks[0].get("steps", [])
        print(f"  完成消息 steps: {steps}")
        print(f"  {Colors.GREEN}PASS: 收到 done 消息{Colors.RESET}")
    else:
        print(f"  {Colors.RED}FAIL: 未收到 done 消息{Colors.RESET}")
        passed = False

    # 6. 检查没有双重包装（确认 raw_chunk 能直接解析，不是嵌套 JSON）
    double_wrapped = any(
        isinstance(c.get("content"), str) and c.get("content", "").startswith("{")
        and c.get("type") not in ("content", "done", "error")
        for c in all_chunks
    )
    if double_wrapped:
        print(f"  {Colors.RED}FAIL: 检测到双重 JSON 包装{Colors.RESET}")
        passed = False
    else:
        print(f"  {Colors.GREEN}PASS: 无双重包装{Colors.RESET}")

    # 时间统计
    print(f"\n  总耗时: {total_ms}ms")
    if first_content_time:
        print(f"  首 token 延迟 (TTFT): {first_content_time}ms")

    status = f"{Colors.GREEN}PASSED{Colors.RESET}" if passed else f"{Colors.RED}FAILED{Colors.RESET}"
    print(f"\n  结果: {status}")

    return passed


async def main():
    """主测试流程"""
    import os

    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}  TCM Agent 流式输出测试{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}")

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
        print(f"\n{Colors.RED}错误: 未找到 DEEPSEEK_API_KEY 或 OPENAI_API_KEY{Colors.RESET}")
        print("请在 .env 文件中配置 API Key")
        sys.exit(1)

    results = {}

    for case in TEST_CASES:
        ok = await run_stream_test(
            test_name=case["name"],
            message=case["message"],
            expect_types=case["expect_types"],
            provider_name=provider_name,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
        )
        results[case["name"]] = ok

    # ============== 汇总 ==============
    print(f"\n\n{'=' * 70}")
    print(f"{Colors.BOLD}  测试汇总{Colors.RESET}")
    print(f"{'=' * 70}\n")

    for name, ok in results.items():
        icon = f"{Colors.GREEN}PASS{Colors.RESET}" if ok else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  [{icon}] {name}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\n  {passed}/{total} 通过")

    if passed == total:
        print(f"\n  {Colors.GREEN}{Colors.BOLD}全部通过!{Colors.RESET}")
    else:
        print(f"\n  {Colors.RED}存在失败项{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
