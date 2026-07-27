"""运行“问诊 → Neo4j → 病例落库 → 健康档案”真实端到端验收。"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from neo4j import AsyncGraphDatabase
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# override=True：.env 是验收配置的唯一事实来源，避免 shell 残留的旧
# ANTHROPIC_BASE_URL（如已失效的本地 cli-proxy 8317）把供应商错配到死端点。
load_dotenv(Path(__file__).resolve().parents[2] / ".env", encoding="utf-8", override=True)


class E2EConfig(BaseModel):
    api_base_url: str = Field(
        default_factory=lambda: os.environ.get(
            "E2E_API_BASE_URL",
            "http://127.0.0.1:8094",
        )
    )
    email: str = "renshu-e2e-20260720@example.com"
    password: str = "RenShu-E2E-2026!"
    username: str = "renshu_e2e"
    provider_name: str = "anthropic"
    model_name: str = Field(default_factory=lambda: os.environ.get("LONGCAT_MODEL", "LongCat-2.0"))
    api_key: str = Field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""), min_length=1)
    base_url: str = Field(default_factory=lambda: os.environ.get("ANTHROPIC_BASE_URL", ""), min_length=8)
    neo4j_uri: str = Field(default_factory=lambda: os.environ.get("NEO4J_URI", ""), min_length=8)
    neo4j_user: str = Field(default_factory=lambda: os.environ.get("NEO4J_USER", ""), min_length=1)
    neo4j_password: str = Field(default_factory=lambda: os.environ.get("NEO4J_PASSWORD", ""), min_length=1)
    neo4j_database: str = Field(default_factory=lambda: os.environ.get("NEO4J_DB", "neo4j"))


def llm_endpoint_hint(base_url: str) -> str:
    """对解析到的 LLM 端点给出可见性提示。

    若指向本地回环网关（如 cli-proxy），往往是 shell 残留的旧 ANTHROPIC_BASE_URL，
    需在运行前确认其会话/Key 仍有效，否则会出现“远端 Key + 本地网关”的 401 伪业务失败。
    """
    if "127.0.0.1" in base_url or "localhost" in base_url:
        return "（本地网关，请确认会话/Key 有效；若非预期请检查 shell 的 ANTHROPIC_BASE_URL）"
    return ""


class Neo4jEvidence(BaseModel):
    syndrome_count: int
    formula_count: int
    syndrome_name: str
    formula_name: str


async def verify_neo4j(config: E2EConfig) -> Neo4jEvidence:
    """显式验证验收图谱可查询，避免仅依赖后端日志判断 Neo4j 已参与。"""
    driver = AsyncGraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_user, config.neo4j_password),
    )
    try:
        async with driver.session(database=config.neo4j_database) as session:
            result = await session.run(
                """
                MATCH (sy:Syndrome {source_db: 'renshu_e2e'})
                WITH count(sy) AS syndrome_count
                MATCH (f:Formula {source_db: 'renshu_e2e'})
                WITH syndrome_count, count(f) AS formula_count
                MATCH (known_sy:Syndrome {id: 'E2E-SY-001'}),
                      (known_f:Formula {id: 'E2E-F-001'})
                RETURN syndrome_count, formula_count,
                       known_sy.name_zh AS syndrome_name,
                       known_f.name_zh AS formula_name
                """
            )
            record = await result.single()
    finally:
        await driver.close()

    if record is None:
        raise RuntimeError("Neo4j 验收图谱查询未返回结果")
    evidence = Neo4jEvidence(**dict(record))
    if evidence.syndrome_count < 2 or evidence.formula_count < 3:
        raise RuntimeError(f"Neo4j 验收图谱数据不足: {evidence.model_dump()}")
    return evidence


def response_data(payload: dict[str, Any]) -> Any:
    return payload.get("Data", payload.get("data"))


def response_success(payload: dict[str, Any]) -> bool:
    return bool(payload.get("Success", payload.get("success", False)))


def _canonical_name(value: str) -> str:
    normalized = re.sub(r"[\s，,。；;、（）()【】\[\]·—_\-]+", "", str(value or "")).lower()
    return normalized[:-1] if normalized.endswith("证") and len(normalized) > 1 else normalized


def assert_prescription_grounding(
    diagnosis_payload: dict[str, Any],
    persisted_prescriptions: list[dict[str, Any]],
) -> None:
    """有方剂时必须具备最终主证到方剂的显式关系证据。"""
    structured = diagnosis_payload.get("prescriptions") or []
    persisted_names = {
        str(item.get("name") or item.get("prescription_name") or "")
        for item in persisted_prescriptions
    }
    structured_names = {str(item.get("name") or "") for item in structured}
    if persisted_names != structured_names:
        raise RuntimeError(
            f"病例方剂与结构化结果不一致: persisted={persisted_names}, structured={structured_names}"
        )

    diagnosed_syndrome = _canonical_name(diagnosis_payload.get("syndrome") or "")
    for item in structured:
        evidence = item.get("relation_evidence") or {}
        if (
            evidence.get("relationship_type") != "TREATS_WITH"
            or not evidence.get("relationship_id")
            or not evidence.get("relationship_path")
            or not evidence.get("formula_name")
            or not evidence.get("syndrome_name")
        ):
            raise RuntimeError(f"方剂缺少可追溯证方关系: {item}")
        if _canonical_name(evidence["syndrome_name"]) != diagnosed_syndrome:
            raise RuntimeError(f"方剂关系证型与最终主证不一致: {item}")
        if _canonical_name(evidence["formula_name"]) != _canonical_name(item.get("name") or ""):
            raise RuntimeError(f"方剂关系节点与推荐方剂不一致: {item}")


async def login_or_register(client: httpx.AsyncClient, config: E2EConfig) -> tuple[str, str]:
    login_payload = {"email": config.email, "password": config.password}
    login = await client.post("/api/v1/users/login", json=login_payload)

    if login.status_code >= 400 or not response_success(login.json()):
        register = await client.post(
            "/api/v1/users/register",
            json={
                "username": config.username,
                "role": "patient",
                "email": config.email,
                "password": config.password,
                "real_name": "端到端验收用户",
                "gender": "other",
                "constitution_type": "气虚质",
            },
        )
        register.raise_for_status()
        if not response_success(register.json()):
            raise RuntimeError(f"注册失败: {register.text[:500]}")
        login = await client.post("/api/v1/users/login", json=login_payload)

    login.raise_for_status()
    payload = login.json()
    if not response_success(payload):
        raise RuntimeError(f"登录失败: {login.text[:500]}")
    data = response_data(payload)
    return str(data["access_token"]), str(data["user_id"])


async def find_longcat_ids(
    client: httpx.AsyncClient,
    config: E2EConfig,
) -> tuple[str, str]:
    response = await client.get("/api/v1/builtin/providers_with_models")
    response.raise_for_status()
    providers = response_data(response.json()) or []
    provider = next(p for p in providers if p.get("name") == config.provider_name)
    model = next(m for m in provider.get("models", []) if m.get("model_name") == config.model_name)
    return str(provider["id"]), str(model["id"])


async def configure_user_provider(
    client: httpx.AsyncClient,
    provider_id: str,
    config: E2EConfig,
) -> None:
    response = await client.post(
        "/api/v1/provider/update",
        json={
            "provider_id": provider_id,
            "api_key": config.api_key,
            "base_url": config.base_url,
            "is_enabled": True,
        },
    )
    response.raise_for_status()
    if not response_success(response.json()):
        raise RuntimeError(f"供应商配置失败: {response.text[:500]}")


async def read_sse_events(response: httpx.Response) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    async for line in response.aiter_lines():
        if not line.startswith("data: "):
            continue
        raw = line[6:]
        if raw == "[DONE]":
            continue
        event = json.loads(raw)
        if event.get("type") == "error":
            raise RuntimeError(f"问诊流返回错误: {event.get('content')}")
        events.append(event)
    return events


async def run_consultation(
    client: httpx.AsyncClient,
    user_id: str,
    provider_id: str,
    model_id: str,
    config: E2EConfig,
) -> tuple[str, list[dict[str, Any]]]:
    conversation_id = str(uuid4())
    request_payload = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "query": (
            "我近四个月一直乏力、头晕、心悸、气短、腰痛、耳鸣，"
            "明显怕冷，无汗，饮食正常、饭量正常、无特殊不适，"
            "大便基本正常但小便偏多，晚上多梦易醒。"
            "请按中医辨证分析。"
        ),
        "model_configuration": {
            "provider_id": provider_id,
            "model_id": model_id,
            "model_name": config.model_name,
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 1600,
        },
        "stream": True,
        "enable_thinking": False,
    }

    async with client.stream("POST", "/api/v1/chat/generate", json=request_payload) as response:
        response.raise_for_status()
        events = await read_sse_events(response)

    interrupts = [event for event in events if event.get("type") == "interrupt"]
    if interrupts:
        raise RuntimeError(f"本次验收输入仍触发追问: {interrupts[-1].get('question')}")
    if not any(event.get("type") == "done" for event in events):
        raise RuntimeError("问诊流未返回 done 事件")
    done_event = next(event for event in reversed(events) if event.get("type") == "done")
    if done_event.get("query_type") != "tcm-diagnose":
        raise RuntimeError(f"问诊未进入诊断流程: {done_event.get('query_type')}")
    return conversation_id, events


async def verify_case_and_profile(
    client: httpx.AsyncClient,
    conversation_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for _ in range(60):
        cases_response = await client.get("/api/v1/cases", params={"limit": 50, "offset": 0})
        cases_response.raise_for_status()
        cases_payload = response_data(cases_response.json()) or {}
        cases = cases_payload.get("items", [])
        matched = next(
            (case for case in cases if case.get("conversation_id") == conversation_id),
            None,
        )
        if matched:
            detail_response = await client.get(f"/api/v1/cases/{matched['id']}")
            profile_response = await client.get("/api/v1/cases/profile")
            detail_response.raise_for_status()
            profile_response.raise_for_status()
            detail = response_data(detail_response.json())
            profile = response_data(profile_response.json())
            if not profile or profile.get("total_cases", 0) < 1:
                raise RuntimeError(f"健康档案未聚合病例: {profile}")
            return detail, profile
        await asyncio.sleep(1)
    raise RuntimeError("问诊完成后未找到对应病例")


async def main() -> None:
    config = E2EConfig()
    _hint = llm_endpoint_hint(config.base_url)
    print(
        f"# E2E 验收 LLM 端点: ANTHROPIC_BASE_URL={config.base_url} "
        f"model={config.model_name}" + (f" {_hint}" if _hint else "")
    )
    neo4j_evidence = await verify_neo4j(config)
    timeout = httpx.Timeout(connect=20, read=900, write=60, pool=20)
    async with httpx.AsyncClient(base_url=config.api_base_url, timeout=timeout) as client:
        token, user_id = await login_or_register(client, config)
        client.headers["Authorization"] = f"Bearer {token}"

        provider_id, model_id = await find_longcat_ids(client, config)
        await configure_user_provider(client, provider_id, config)
        conversation_id, events = await run_consultation(
            client, user_id, provider_id, model_id, config
        )
        detail, profile = await verify_case_and_profile(client, conversation_id)

    content = "".join(
        event.get("content", "")
        for event in events
        if event.get("type") == "content" and isinstance(event.get("content"), str)
    )
    case = detail["case"]
    if not case.get("syndrome_name"):
        raise RuntimeError("病例已落库，但未提取到证型名称")
    structured_events = [event for event in events if event.get("type") == "diagnosis_result"]
    if not structured_events:
        raise RuntimeError("问诊流未返回 diagnosis_result 结构化事件")
    diagnosis_payload = case.get("diagnosis_payload") or {}
    if diagnosis_payload.get("syndrome") != case.get("syndrome_name"):
        raise RuntimeError(f"结构化主证与病例主证不一致: {diagnosis_payload}")
    complexity_level = case.get("complexity_level")
    if complexity_level == "complex" and detail.get("prescriptions"):
        raise RuntimeError("复杂病例安全降级路径不应落库具体方剂")
    assert_prescription_grounding(
        diagnosis_payload,
        detail.get("prescriptions") or [],
    )
    stream_diagnosis = structured_events[-1].get("data") or {}
    graph_citations = [
        citation
        for citation in diagnosis_payload.get("citations", [])
        if citation.get("source_type") == "graph_path"
    ]
    stream_graph_citations = [
        citation
        for citation in stream_diagnosis.get("citations", [])
        if citation.get("source_type") == "graph_path"
    ]
    if not graph_citations or not stream_graph_citations:
        raise RuntimeError("真实问诊未返回并落库 GraphRAG 关系证据")
    if not all(
        citation.get("source") == "med_tcm"
        and citation.get("node_ids")
        and citation.get("relationship_ids")
        and citation.get("relationship_path")
        and citation.get("symptom_role")
        and citation.get("evidence_weight") is not None
        for citation in graph_citations
    ):
        raise RuntimeError(f"GraphRAG 引用缺少来源或路径字段: {graph_citations}")
    print("E2E_ACCEPTANCE=PASS")
    print(f"conversation_id={conversation_id}")
    print(f"case_id={case['id']}")
    print(f"complexity_level={case.get('complexity_level')}")
    print(f"syndrome_name={case.get('syndrome_name')}")
    print(f"symptom_count={len(detail.get('symptoms', []))}")
    print(f"diagnosis_text_present={bool(detail.get('diagnosis_text'))}")
    print(f"diagnosis_payload_present={bool(diagnosis_payload)}")
    print(f"case_prescription_count={len(detail.get('prescriptions', []))}")
    print(f"graph_citation_count={len(graph_citations)}")
    print(f"health_profile_total_cases={profile.get('total_cases')}")
    print(f"stream_content_chars={len(content)}")
    print(f"neo4j_syndrome_count={neo4j_evidence.syndrome_count}")
    print(f"neo4j_formula_count={neo4j_evidence.formula_count}")
    print(f"neo4j_seed_nodes={neo4j_evidence.syndrome_name},{neo4j_evidence.formula_name}")


if __name__ == "__main__":
    asyncio.run(main())
