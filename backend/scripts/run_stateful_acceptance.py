"""验收持久多轮状态、跨重启恢复和会话授权边界。

用法：

1. 重启前制造一个真实 LangGraph interrupt：
   uv run python scripts/run_stateful_acceptance.py prepare
2. 重启后使用上一步输出继续：
   uv run python scripts/run_stateful_acceptance.py resume \
       --conversation-id <conversation_id> --thread-id <thread_id>
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any
from uuid import uuid4

import httpx
from pydantic import BaseModel

from run_e2e_acceptance import (
    E2EConfig,
    configure_user_provider,
    find_longcat_ids,
    login_or_register,
    read_sse_events,
    response_data,
)


class StatefulAcceptanceConfig(E2EConfig):
    attacker_email: str = "renshu-e2e-attacker-20260720@example.com"
    attacker_password: str = "RenShu-E2E-Attacker-2026!"
    attacker_username: str = "renshu_e2e_attacker"


class StatefulTarget(BaseModel):
    conversation_id: str
    thread_id: str


def model_configuration(
    provider_id: str,
    model_id: str,
    config: StatefulAcceptanceConfig,
) -> dict[str, Any]:
    return {
        "provider_id": provider_id,
        "model_id": model_id,
        "model_name": config.model_name,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 1600,
    }


async def get_conversation(
    client: httpx.AsyncClient,
    conversation_id: str,
) -> dict[str, Any]:
    response = await client.get("/api/v1/conversations/me")
    response.raise_for_status()
    conversations = response_data(response.json()) or []
    conversation = next(
        (item for item in conversations if str(item.get("id")) == conversation_id),
        None,
    )
    if conversation is None:
        raise RuntimeError(f"未找到会话: {conversation_id}")
    return conversation


async def get_messages(
    client: httpx.AsyncClient,
    conversation_id: str,
) -> list[dict[str, Any]]:
    response = await client.post(
        "/api/v1/conversations/messages",
        json={"conversation_id": conversation_id},
    )
    response.raise_for_status()
    return response_data(response.json()) or []


async def assert_forbidden(response: httpx.Response, operation: str) -> None:
    if response.status_code not in {401, 403, 404}:
        raise RuntimeError(
            f"{operation} 未被拒绝: status={response.status_code}, "
            f"body={response.text[:500]}"
        )


async def prepare() -> None:
    config = StatefulAcceptanceConfig()
    timeout = httpx.Timeout(connect=20, read=900, write=60, pool=20)
    async with httpx.AsyncClient(base_url=config.api_base_url, timeout=timeout) as client:
        token, user_id = await login_or_register(client, config)
        client.headers["Authorization"] = f"Bearer {token}"
        provider_id, model_id = await find_longcat_ids(client, config)
        await configure_user_provider(client, provider_id, config)

        conversation_id = str(uuid4())
        payload = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query": "我最近身体不舒服，想做一次中医问诊，请先帮我辨证。",
            "model_configuration": model_configuration(provider_id, model_id, config),
            "stream": True,
            "enable_thinking": False,
        }
        async with client.stream("POST", "/api/v1/chat/generate", json=payload) as response:
            response.raise_for_status()
            events = await read_sse_events(response)

        interrupts = [event for event in events if event.get("type") == "interrupt"]
        if not interrupts:
            raise RuntimeError(f"信息不足的问诊未触发 interrupt: {events[-5:]}")
        if any(event.get("type") == "done" for event in events):
            raise RuntimeError("触发 interrupt 后不应同时返回 done")

        interrupt_event = interrupts[-1]
        conversation = await get_conversation(client, conversation_id)
        thread_id = str(conversation.get("agent_thread_id") or "")
        persisted_interrupt = conversation.get("agent_interrupt") or {}
        if not thread_id or thread_id != str(interrupt_event.get("thread_id")):
            raise RuntimeError("SSE thread 与服务端会话绑定不一致")
        if not persisted_interrupt.get("pending"):
            raise RuntimeError(f"interrupt 未持久化到会话: {persisted_interrupt}")

        messages = await get_messages(client, conversation_id)
        question = str(interrupt_event.get("question") or "").strip()
        if not any(
            message.get("role") == "assistant" and message.get("content") == question
            for message in messages
        ):
            raise RuntimeError("interrupt 问题未作为 assistant 消息保存")

        forged_response = await client.post(
            "/api/v1/chat/resume",
            json={
                "conversation_id": conversation_id,
                "thread_id": str(uuid4()),
                "query": "伪造线程恢复",
                "model_configuration": model_configuration(provider_id, model_id, config),
            },
        )
        await assert_forbidden(forged_response, "同用户伪造 thread resume")

    attacker_config = config.model_copy(update={
        "email": config.attacker_email,
        "password": config.attacker_password,
        "username": config.attacker_username,
    })
    async with httpx.AsyncClient(
        base_url=config.api_base_url,
        timeout=timeout,
    ) as attacker_client:
        attacker_token, _ = await login_or_register(attacker_client, attacker_config)
        attacker_client.headers["Authorization"] = f"Bearer {attacker_token}"
        messages_response = await attacker_client.post(
            "/api/v1/conversations/messages",
            json={"conversation_id": conversation_id},
        )
        await assert_forbidden(messages_response, "跨用户读取会话消息")
        resume_response = await attacker_client.post(
            "/api/v1/chat/resume",
            json={
                "conversation_id": conversation_id,
                "thread_id": thread_id,
                "query": "跨用户恢复",
                "model_configuration": model_configuration(provider_id, model_id, config),
            },
        )
        await assert_forbidden(resume_response, "跨用户 resume")

    print("STATEFUL_PREPARE=PASS")
    print(f"conversation_id={conversation_id}")
    print(f"thread_id={thread_id}")
    print(f"interrupt_question={question}")
    print(f"message_count={len(messages)}")


async def verify_second_turn(
    client: httpx.AsyncClient,
    target: StatefulTarget,
    user_id: str,
    provider_id: str,
    model_id: str,
    config: StatefulAcceptanceConfig,
    expected_history: str = "青竹七号",
) -> tuple[str, int, int]:
    """验证已完成问诊后的新回合复用 thread 和消息历史。"""
    messages_before = await get_messages(client, target.conversation_id)
    history_query = (
        "请介绍一下中医的特点，并在回答开头先复述我上一条补充信息中的"
        "确认口令。"
    )
    async with client.stream(
        "POST",
        "/api/v1/chat/generate",
        json={
            "user_id": user_id,
            "conversation_id": target.conversation_id,
            "query": history_query,
            "model_configuration": model_configuration(provider_id, model_id, config),
            "stream": True,
            "enable_thinking": False,
        },
    ) as response:
        response.raise_for_status()
        second_turn_events = await read_sse_events(response)

    if any(event.get("type") == "interrupt" for event in second_turn_events):
        raise RuntimeError("已完成问诊后的历史追问不应被旧状态短路为 interrupt")
    if not any(event.get("type") == "done" for event in second_turn_events):
        raise RuntimeError("同会话第二个正常回合未完成")
    second_answer = "".join(
        str(event.get("content") or "")
        for event in second_turn_events
        if event.get("type") == "content"
    )
    if expected_history not in second_answer:
        raise RuntimeError(f"第二回合未使用病程历史: {second_answer}")

    after_second_turn = await get_conversation(client, target.conversation_id)
    if str(after_second_turn.get("agent_thread_id")) != target.thread_id:
        raise RuntimeError("同会话第二回合未复用服务端 thread")

    messages_after = await get_messages(client, target.conversation_id)
    for _ in range(20):
        if len(messages_after) >= len(messages_before) + 2:
            break
        await asyncio.sleep(0.2)
        messages_after = await get_messages(client, target.conversation_id)
    if len(messages_after) < len(messages_before) + 2:
        raise RuntimeError(
            f"第二回合消息未完整保存: before={len(messages_before)}, "
            f"after={len(messages_after)}"
        )
    return second_answer, len(messages_before), len(messages_after)


async def resume(target: StatefulTarget) -> None:
    config = StatefulAcceptanceConfig()
    timeout = httpx.Timeout(connect=20, read=900, write=60, pool=20)
    async with httpx.AsyncClient(base_url=config.api_base_url, timeout=timeout) as client:
        token, user_id = await login_or_register(client, config)
        client.headers["Authorization"] = f"Bearer {token}"
        provider_id, model_id = await find_longcat_ids(client, config)
        await configure_user_provider(client, provider_id, config)

        before = await get_conversation(client, target.conversation_id)
        if str(before.get("agent_thread_id")) != target.thread_id:
            raise RuntimeError("后端重启后会话 thread_id 发生变化")
        if not (before.get("agent_interrupt") or {}).get("pending"):
            raise RuntimeError("后端重启后未恢复待回答的 interrupt")
        messages_before = await get_messages(client, target.conversation_id)

        answers = [
            (
                "补充信息：症状持续近两个月，主要是乏力、头晕、心悸、气短、"
                "腰痛、耳鸣，明显怕冷，无汗；饮食和饭量正常，大便正常，"
                "小便偏多，晚上多梦易醒，情绪基本稳定。"
                "为核对会话记忆，我的确认口令是青竹七号。"
            ),
            "没有其他明显症状，也没有发热、胸痛或晕厥，请继续辨证。",
            "舌淡、苔白，脉沉细，请基于以上完整信息继续。",
        ]
        events: list[dict[str, Any]] = []
        for answer in answers:
            async with client.stream(
                "POST",
                "/api/v1/chat/resume",
                json={
                    "conversation_id": target.conversation_id,
                    "thread_id": target.thread_id,
                    "query": answer,
                    "model_configuration": model_configuration(provider_id, model_id, config),
                },
            ) as response:
                response.raise_for_status()
                events = await read_sse_events(response)
            if any(event.get("type") == "done" for event in events):
                break
            if not any(event.get("type") == "interrupt" for event in events):
                raise RuntimeError(f"resume 既未完成也未再次追问: {events[-5:]}")
        else:
            raise RuntimeError("三次 resume 后仍未完成问诊")

        if not any(event.get("type") == "diagnosis_result" for event in events):
            raise RuntimeError("跨重启 resume 完成后缺少结构化诊断结果")
        after_resume = await get_conversation(client, target.conversation_id)
        if after_resume.get("agent_interrupt") is not None:
            raise RuntimeError(f"完成后 interrupt 未清理: {after_resume.get('agent_interrupt')}")

        second_answer, turn_messages_before, messages_after = await verify_second_turn(
            client,
            target,
            user_id,
            provider_id,
            model_id,
            config,
        )
        if messages_after < len(messages_before) + 4:
            raise RuntimeError(
                f"恢复与第二回合消息未完整保存: before={len(messages_before)}, "
                f"turn_before={turn_messages_before}, after={messages_after}"
            )

    print("STATEFUL_RESUME=PASS")
    print(f"conversation_id={target.conversation_id}")
    print(f"thread_id={target.thread_id}")
    print(f"messages_before={len(messages_before)}")
    print(f"messages_after={messages_after}")
    print(f"second_turn_answer_chars={len(second_answer)}")


async def turn(target: StatefulTarget, expected_history: str) -> None:
    config = StatefulAcceptanceConfig()
    timeout = httpx.Timeout(connect=20, read=900, write=60, pool=20)
    async with httpx.AsyncClient(base_url=config.api_base_url, timeout=timeout) as client:
        token, user_id = await login_or_register(client, config)
        client.headers["Authorization"] = f"Bearer {token}"
        provider_id, model_id = await find_longcat_ids(client, config)
        await configure_user_provider(client, provider_id, config)
        second_answer, messages_before, messages_after = await verify_second_turn(
            client,
            target,
            user_id,
            provider_id,
            model_id,
            config,
            expected_history=expected_history,
        )

    print("STATEFUL_TURN=PASS")
    print(f"conversation_id={target.conversation_id}")
    print(f"thread_id={target.thread_id}")
    print(f"messages_before={messages_before}")
    print(f"messages_after={messages_after}")
    print(f"second_turn_answer_chars={len(second_answer)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--conversation-id", required=True)
    resume_parser.add_argument("--thread-id", required=True)
    turn_parser = subparsers.add_parser("turn")
    turn_parser.add_argument("--conversation-id", required=True)
    turn_parser.add_argument("--thread-id", required=True)
    turn_parser.add_argument("--expected-history", default="青竹七号")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        await prepare()
        return
    target = StatefulTarget(
        conversation_id=args.conversation_id,
        thread_id=args.thread_id,
    )
    if args.command == "turn":
        await turn(target, args.expected_history)
        return
    await resume(target)


if __name__ == "__main__":
    asyncio.run(main())
