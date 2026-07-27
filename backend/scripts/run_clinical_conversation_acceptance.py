"""重放真实“粘贴血常规 → 追问 → 复杂病例安全降级”问诊链路。"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import uuid4

import httpx

from run_e2e_acceptance import (
    E2EConfig,
    configure_user_provider,
    find_longcat_ids,
    login_or_register,
    read_sse_events,
    response_data,
)


REPORT_QUERY = """血常规检验报告
血红蛋白：92 g/L，参考范围 115-150，偏低
白细胞：6.2×10^9/L，参考范围 3.5-9.5，正常
血小板：235×10^9/L，参考范围 125-350，正常

请解读这份报告，并结合我最近乏力、心悸、头晕的情况给出健康建议。"""

FOLLOW_UP_QUERY = """我既怕冷又经常手足心热，白天乏力自汗，夜间又盗汗，口干但不想喝水，
大便有时干、有时稀，胸闷心悸，失眠多梦，病程一年。
我有高血压和糖尿病，目前正在服用多种药物。"""


def model_configuration(provider_id: str, model_id: str, config: E2EConfig) -> dict[str, Any]:
    return {
        "provider_id": provider_id,
        "model_id": model_id,
        "model_name": config.model_name,
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 1600,
    }


async def get_conversation(client: httpx.AsyncClient, conversation_id: str) -> dict[str, Any]:
    response = await client.get("/api/v1/conversations/me")
    response.raise_for_status()
    items = response_data(response.json()) or []
    conversation = next(
        (item for item in items if str(item.get("id")) == conversation_id),
        None,
    )
    if conversation is None:
        raise RuntimeError(f"未找到会话 {conversation_id}")
    return conversation


async def get_messages(client: httpx.AsyncClient, conversation_id: str) -> list[dict[str, Any]]:
    response = await client.post(
        "/api/v1/conversations/messages",
        json={"conversation_id": conversation_id},
    )
    response.raise_for_status()
    return response_data(response.json()) or []


async def find_case(client: httpx.AsyncClient, conversation_id: str) -> dict[str, Any]:
    for _ in range(60):
        response = await client.get("/api/v1/cases", params={"limit": 100, "offset": 0})
        response.raise_for_status()
        items = (response_data(response.json()) or {}).get("items", [])
        matched = next(
            (item for item in items if item.get("conversation_id") == conversation_id),
            None,
        )
        if matched:
            detail = await client.get(f"/api/v1/cases/{matched['id']}")
            detail.raise_for_status()
            return response_data(detail.json())
        await asyncio.sleep(0.5)
    raise RuntimeError("问诊完成后未找到病例")


def assert_report_event(events: list[dict[str, Any]]) -> dict[str, Any]:
    report_event = next(
        (event for event in events if event.get("type") == "report_analysis"),
        None,
    )
    if report_event is None:
        raise RuntimeError("首轮未返回 report_analysis")
    report = report_event.get("data") or {}
    hemoglobin = next(
        (
            item
            for item in report.get("metrics", [])
            if item.get("name") == "血红蛋白"
        ),
        None,
    )
    if not hemoglobin or hemoglobin.get("value") != "92" or hemoglobin.get("abnormal_flag") != "low":
        raise RuntimeError(f"血红蛋白未被正确识别: {hemoglobin}")
    if "贫血" not in str(report.get("summary") or ""):
        raise RuntimeError(f"报告未给出贫血原因评估边界: {report}")
    return report


async def main() -> None:
    config = E2EConfig()
    timeout = httpx.Timeout(connect=20, read=900, write=60, pool=20)
    async with httpx.AsyncClient(base_url=config.api_base_url, timeout=timeout) as client:
        token, user_id = await login_or_register(client, config)
        client.headers["Authorization"] = f"Bearer {token}"
        provider_id, model_id = await find_longcat_ids(client, config)
        await configure_user_provider(client, provider_id, config)

        persona_response = await client.post(
            "/api/v1/chat/analyze_persona",
            json={
                "user_id": user_id,
                "text": "最近乏力、心悸、头晕，血红蛋白92 g/L偏低。",
                "current_persona": None,
                "conversation_id": None,
                "model_configuration": model_configuration(provider_id, model_id, config),
            },
        )
        persona_response.raise_for_status()
        persona = response_data(persona_response.json()) or {}
        if not persona.get("chiefComplaint") or not persona.get("suspectedDiagnosis"):
            raise RuntimeError(f"Persona 结构化分析未返回有效结果: {persona}")

        conversation_id = str(uuid4())
        generate_payload = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query": REPORT_QUERY,
            "model_configuration": model_configuration(provider_id, model_id, config),
            "stream": True,
            "enable_thinking": False,
        }

        async with client.stream(
            "POST",
            "/api/v1/chat/generate",
            json=generate_payload,
        ) as response:
            response.raise_for_status()
            duplicate = await client.post(
                "/api/v1/chat/generate",
                json={**generate_payload, "query": "并发重复发送，不应落库。"},
            )
            if duplicate.status_code != 409:
                raise RuntimeError(
                    f"同会话并发请求未返回 409: {duplicate.status_code} {duplicate.text[:500]}"
                )
            first_events = await read_sse_events(response)

        report = assert_report_event(first_events)
        interrupts = [event for event in first_events if event.get("type") == "interrupt"]
        if len(interrupts) != 1:
            raise RuntimeError(f"首轮追问数量异常: {interrupts}")
        first_interrupt = interrupts[0]
        first_question = str(first_interrupt.get("question") or "")
        if first_interrupt.get("action") != "ask_symptom":
            raise RuntimeError(f"首轮未进入症状追问: {first_interrupt}")
        if "血红蛋白 92 g/L" not in first_question or "舌" in first_question:
            raise RuntimeError(f"首轮追问未正确解释报告或仍索要舌照: {first_question}")

        conversation = await get_conversation(client, conversation_id)
        thread_id = str(conversation.get("agent_thread_id") or "")
        if not thread_id or not (conversation.get("agent_interrupt") or {}).get("pending"):
            raise RuntimeError(f"首轮 interrupt 未持久化: {conversation}")

        first_messages = await get_messages(client, conversation_id)
        if len(first_messages) != 2:
            raise RuntimeError(f"并发请求产生了重复消息: {len(first_messages)}")
        first_user = next(message for message in first_messages if message.get("role") == "user")
        metadata = first_user.get("message_metadata") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        if (metadata.get("report_analysis") or {}).get("report_type") != "血常规":
            raise RuntimeError(f"用户消息未保存报告结构: {metadata}")

        async with client.stream(
            "POST",
            "/api/v1/chat/resume",
            json={
                "conversation_id": conversation_id,
                "thread_id": thread_id,
                "query": FOLLOW_UP_QUERY,
                "model_configuration": model_configuration(provider_id, model_id, config),
            },
        ) as response:
            response.raise_for_status()
            second_events = await read_sse_events(response)

        second_interrupts = [
            event for event in second_events if event.get("type") == "interrupt"
        ]
        if second_interrupts:
            raise RuntimeError(f"信息充分后仍触发中断: {second_interrupts}")
        diagnosis_event = next(
            (event for event in second_events if event.get("type") == "diagnosis_result"),
            None,
        )
        if diagnosis_event is None:
            raise RuntimeError("第二轮未返回 diagnosis_result")
        diagnosis = diagnosis_event.get("data") or {}
        if diagnosis.get("prescriptions"):
            raise RuntimeError(f"复杂慢病多药病例仍返回具体方剂: {diagnosis['prescriptions']}")
        if not diagnosis.get("should_seek_doctor"):
            raise RuntimeError("复杂慢病多药病例未强制线下复核")

        content = "".join(
            str(event.get("content") or "")
            for event in second_events
            if event.get("type") == "content"
        )
        if content.lstrip().startswith("{"):
            raise RuntimeError("SSE content 泄漏了内部结构化 JSON")
        for expected in ("血红蛋白 92 g/L", "高血压", "糖尿病", "多种药物", "不能自行停药"):
            if expected not in content:
                raise RuntimeError(f"患者回答缺少安全信息 {expected}: {content}")

        detail = await find_case(client, conversation_id)
        case = detail.get("case") or {}
        if case.get("complexity_level") != "complex":
            raise RuntimeError(f"病例复杂度不是 complex: {case}")
        if (detail.get("report_analysis") or {}).get("report_type") != "血常规":
            raise RuntimeError("病例未保存 report_analysis")
        if detail.get("prescriptions"):
            raise RuntimeError(f"病例仍落库方剂: {detail['prescriptions']}")
        evidence = (detail.get("diagnosis_payload") or {}).get("input_evidence") or {}
        if evidence.get("medical_history") != ["高血压", "糖尿病"]:
            raise RuntimeError(f"病例未保存慢病证据: {evidence}")
        if not evidence.get("current_medications"):
            raise RuntimeError(f"病例未保存用药证据: {evidence}")

        profile_response = await client.get("/api/v1/cases/profile")
        profile_response.raise_for_status()
        profile = response_data(profile_response.json()) or {}
        chronic = profile.get("chronic_conditions") or []
        if not {"高血压", "糖尿病"}.issubset(set(chronic)):
            raise RuntimeError(f"健康档案未刷新慢病: {profile}")

    print("CLINICAL_CONVERSATION_ACCEPTANCE=PASS")
    print(f"conversation_id={conversation_id}")
    print(f"case_id={case.get('id')}")
    print(f"complexity_level={case.get('complexity_level')}")
    print(f"report_type={report.get('report_type')}")
    print("hemoglobin=92 g/L low")
    print(f"first_question={first_question}")
    print(f"patient_answer_chars={len(content)}")
    print(f"chronic_conditions={','.join(chronic)}")
    print("concurrent_generate_status=409")
    print("persona_analysis=PASS")


if __name__ == "__main__":
    asyncio.run(main())
