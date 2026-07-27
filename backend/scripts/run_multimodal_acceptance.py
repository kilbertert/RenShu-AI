"""真实验收 M1 舌像与 M2 医疗报告图片/PDF 主路径。

前置条件：
- 后端已启动；
- 验收用户已配置经过语义图片探测的 qwen3.6-flash；
- 不从脚本或命令行接收、打印模型 API 密钥。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx
import pymupdf
from pydantic import BaseModel


PUBLIC_TONGUE_FIXTURE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/4/4f/"
    "Oral_pseudomembranous_candidiasis_on_the_dorsum_of_the_tongue_of_a_59-year-old_male_"
    "%28third_week_of_RT_for_a_squamous_cell_carcinoma_of_the_floor_of_the_mouth%2C_"
    "T3N0M0%2C_daily_dose_1.8_Gy%29.png"
)
PUBLIC_TONGUE_FIXTURE_SOURCE = (
    "Wikimedia Commons, Ourania Nicolatou-Galitis et al., CC0; "
    "仅用于验证是否能识别真实人体舌部，不用于评价中医诊断准确性。"
)


class MultimodalAcceptanceConfig(BaseModel):
    api_base_url: str = "http://127.0.0.1:8094"
    email: str = "renshu-e2e-20260720@example.com"
    password: str = "RenShu-E2E-2026!"
    username: str = "renshu_e2e"
    provider_name: str = "qwen"
    model_name: str = "qwen3.6-flash"


def response_data(payload: dict[str, Any]) -> Any:
    return payload.get("Data", payload.get("data"))


def response_success(payload: dict[str, Any]) -> bool:
    return bool(payload.get("Success", payload.get("success", False)))


async def login(client: httpx.AsyncClient, config: MultimodalAcceptanceConfig) -> tuple[str, str]:
    response = await client.post(
        "/api/v1/users/login",
        json={"email": config.email, "password": config.password},
    )
    response.raise_for_status()
    if not response_success(response.json()):
        raise RuntimeError("验收用户登录失败，请先运行基础 E2E 用户初始化")
    data = response_data(response.json())
    return str(data["access_token"]), str(data["user_id"])


async def find_model_ids(
    client: httpx.AsyncClient,
    config: MultimodalAcceptanceConfig,
) -> tuple[str, str]:
    response = await client.get("/api/v1/builtin/providers_with_models")
    response.raise_for_status()
    providers = response_data(response.json()) or []
    provider = next(
        (item for item in providers if item.get("name") == config.provider_name),
        None,
    )
    if provider is None:
        raise RuntimeError("未找到 Qwen 视觉供应商，请先运行 seed_qwen_vision_provider.py")
    model = next(
        (
            item for item in provider.get("models", [])
            if item.get("model_name") == config.model_name
        ),
        None,
    )
    if model is None or "image_input" not in (model.get("features") or []):
        raise RuntimeError("目标模型不存在或尚未通过 image_input 语义能力验收")
    return str(provider["id"]), str(model["id"])


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
            raise RuntimeError(f"多模态流返回错误: {event.get('content')}")
        events.append(event)
    return events


def create_synthetic_report_files(directory: Path) -> tuple[Path, Path]:
    """生成带明确语义的非临床验收报告，不用于医学准确性评价。"""
    pdf_path = directory / "synthetic_medical_report.pdf"
    png_path = directory / "synthetic_medical_report.png"
    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    lines = [
        "SYNTHETIC TEST DATA - NOT A REAL MEDICAL REPORT",
        "Laboratory Test Report",
        "Report Date: 2026-07-20",
        "WBC 12.0 x10^9/L   Reference 3.5-9.5   HIGH",
        "CRP 18 mg/L         Reference 0-10      HIGH",
        "Hemoglobin 132 g/L  Reference 115-150   NORMAL",
        "Glucose 5.2 mmol/L  Reference 3.9-6.1   NORMAL",
        "For software pipeline acceptance only.",
    ]
    y = 72
    for index, line in enumerate(lines):
        page.insert_text(
            (60, y),
            line,
            fontsize=16 if index < 2 else 12,
        )
        y += 44 if index < 2 else 32
    document.save(pdf_path)
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0), alpha=False)
    pixmap.save(png_path)
    document.close()
    return png_path, pdf_path


async def download_public_tongue_fixture(directory: Path) -> Path:
    """下载无身份信息、CC0 的公开人体舌部图片作为真实视觉正向夹具。"""
    path = directory / "public_real_tongue_cc0.png"
    async with httpx.AsyncClient(
        timeout=60,
        follow_redirects=True,
        headers={"User-Agent": "RenShu-AI-E2E/1.0 (local clinical pipeline test)"},
    ) as client:
        response = await client.get(PUBLIC_TONGUE_FIXTURE_URL)
        response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "image/png" not in content_type or len(response.content) < 10_000:
        raise RuntimeError("公开舌像夹具下载结果不是有效 PNG")
    path.write_bytes(response.content)
    return path


async def upload_attachment(
    client: httpx.AsyncClient,
    *,
    conversation_id: str,
    path: Path,
    kind: str,
    mime_type: str,
) -> dict[str, Any]:
    with path.open("rb") as stream:
        response = await client.post(
            "/api/v1/attachments",
            data={"conversation_id": conversation_id, "kind": kind},
            files={"file": (path.name, stream, mime_type)},
        )
    response.raise_for_status()
    payload = response_data(response.json())
    if not payload:
        raise RuntimeError(f"附件上传失败: {response.text[:500]}")
    return payload


def model_configuration(provider_id: str, model_id: str, model_name: str) -> dict[str, Any]:
    return {
        "provider_id": provider_id,
        "model_id": model_id,
        "model_name": model_name,
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 1800,
    }


async def run_chat(
    client: httpx.AsyncClient,
    *,
    user_id: str,
    conversation_id: str,
    attachment_id: str,
    provider_id: str,
    model_id: str,
    model_name: str,
    query: str,
) -> list[dict[str, Any]]:
    async with client.stream(
        "POST",
        "/api/v1/chat/generate",
        json={
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query": query,
            "model_configuration": model_configuration(provider_id, model_id, model_name),
            "stream": True,
            "enable_thinking": False,
            "attachments": [{"id": attachment_id}],
        },
    ) as response:
        response.raise_for_status()
        events = await read_sse_events(response)

    all_events = list(events)
    follow_up_answers = [
        (
            "补充：症状持续近四个月，主要是乏力、头晕，偶有心悸，明显怕冷，"
            "没有发热；平时无汗，饮食和饭量正常，大便正常，小便偏多，"
            "晚上多梦易醒，情绪基本稳定。"
        ),
        "没有其他明显不适，请基于已经提供的症状和报告继续分析。",
        "舌淡、苔白，脉象暂未测，请在信息边界内完成辨证。",
    ]
    for answer in follow_up_answers:
        done = next(
            (event for event in reversed(all_events) if event.get("type") == "done"),
            None,
        )
        if done:
            break
        interrupt_event = next(
            (event for event in reversed(all_events) if event.get("type") == "interrupt"),
            None,
        )
        if interrupt_event is None:
            raise RuntimeError(f"问诊既未完成也未返回 interrupt: {all_events[-5:]}")
        async with client.stream(
            "POST",
            "/api/v1/chat/resume",
            json={
                "conversation_id": conversation_id,
                "thread_id": interrupt_event["thread_id"],
                "query": answer,
                "model_configuration": model_configuration(provider_id, model_id, model_name),
            },
        ) as response:
            response.raise_for_status()
            all_events.extend(await read_sse_events(response))

    done = next(
        (event for event in reversed(all_events) if event.get("type") == "done"),
        None,
    )
    if not done or done.get("query_type") != "tcm-diagnose":
        raise RuntimeError(f"附件未进入诊断结合主路径: {done}")
    if not any(event.get("type") == "diagnosis_result" for event in all_events):
        raise RuntimeError("诊断流缺少结构化 diagnosis_result")
    return all_events


async def get_messages(client: httpx.AsyncClient, conversation_id: str) -> list[dict[str, Any]]:
    response = await client.post(
        "/api/v1/conversations/messages",
        json={"conversation_id": conversation_id},
    )
    response.raise_for_status()
    return response_data(response.json()) or []


async def wait_for_case(client: httpx.AsyncClient, conversation_id: str) -> dict[str, Any]:
    for _ in range(90):
        response = await client.get("/api/v1/cases", params={"limit": 100, "offset": 0})
        response.raise_for_status()
        items = (response_data(response.json()) or {}).get("items", [])
        case = next(
            (item for item in items if item.get("conversation_id") == conversation_id),
            None,
        )
        if case:
            detail_response = await client.get(f"/api/v1/cases/{case['id']}")
            detail_response.raise_for_status()
            return response_data(detail_response.json())
        await asyncio.sleep(1)
    raise RuntimeError("多模态诊断完成后未找到病例")


async def verify_attachment_and_message(
    client: httpx.AsyncClient,
    *,
    conversation_id: str,
    attachment_id: str,
    metadata_key: str,
) -> dict[str, Any]:
    response = await client.get(f"/api/v1/attachments/{attachment_id}")
    response.raise_for_status()
    attachment = response_data(response.json())
    if attachment.get("status") != "analyzed" or not attachment.get("analysis_result"):
        raise RuntimeError(f"附件分析状态不完整: {attachment}")

    messages = await get_messages(client, conversation_id)
    user_message = next(
        (
            item for item in reversed(messages)
            if item.get("role") == "user" and attachment_id in str(item.get("message_metadata"))
        ),
        None,
    )
    if user_message is None:
        raise RuntimeError("用户消息没有绑定来源附件")
    metadata = user_message.get("message_metadata") or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if not metadata.get(metadata_key):
        raise RuntimeError(f"用户消息缺少 {metadata_key}")
    return attachment


async def accept_tongue(
    client: httpx.AsyncClient,
    *,
    path: Path,
    user_id: str,
    provider_id: str,
    model_id: str,
    config: MultimodalAcceptanceConfig,
) -> None:
    conversation_id = str(uuid4())
    attachment = await upload_attachment(
        client,
        conversation_id=conversation_id,
        path=path,
        kind="tongue_image",
        mime_type="image/png",
    )
    events = await run_chat(
        client,
        user_id=user_id,
        conversation_id=conversation_id,
        attachment_id=attachment["id"],
        provider_id=provider_id,
        model_id=model_id,
        model_name=config.model_name,
        query=(
            "我近四个月乏力、头晕、心悸、气短、腰酸、耳鸣，明显怕冷，无汗，"
            "饮食正常，大便正常，小便偏多，晚上多梦易醒。请结合舌像辨证。"
        ),
    )
    analysis = next(event["data"] for event in events if event.get("type") == "tongue_analysis")
    resume_count = sum(event.get("type") == "interrupt" for event in events)
    if str(analysis.get("source_attachment_id")) != attachment["id"]:
        raise RuntimeError("舌像结果没有保留来源附件 ID")
    if not analysis.get("tongue_color") or not analysis.get("coating_color"):
        raise RuntimeError(f"视觉模型未产生可核验舌像语义: {analysis}")
    await verify_attachment_and_message(
        client,
        conversation_id=conversation_id,
        attachment_id=attachment["id"],
        metadata_key="tongue_analysis",
    )
    detail = await wait_for_case(client, conversation_id)
    case = detail["case"]
    if not case.get("tongue_analysis") or not case.get("diagnosis_payload"):
        raise RuntimeError("病例未保存舌像和结构化诊断")
    print("M1_POSITIVE_E2E=PASS")
    print(f"M1_CONVERSATION_ID={conversation_id}")
    print(f"M1_ATTACHMENT_ID={attachment['id']}")
    print(f"M1_CASE_ID={case['id']}")
    print(f"M1_TONGUE_COLOR={analysis['tongue_color']}")
    print(f"M1_COATING_COLOR={analysis['coating_color']}")
    print(f"M1_RESUME_COUNT={resume_count}")
    print(f"M1_FIXTURE_SOURCE={PUBLIC_TONGUE_FIXTURE_SOURCE}")


async def accept_non_tongue(
    client: httpx.AsyncClient,
    *,
    path: Path,
    user_id: str,
    provider_id: str,
    model_id: str,
    config: MultimodalAcceptanceConfig,
) -> None:
    """把明确的检验报告图片伪装成“舌像”，系统必须拒识且不得编造舌象。"""
    conversation_id = str(uuid4())
    attachment = await upload_attachment(
        client,
        conversation_id=conversation_id,
        path=path,
        kind="tongue_image",
        mime_type="image/png",
    )
    async with client.stream(
        "POST",
        "/api/v1/chat/generate",
        json={
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query": "请结合我上传的舌像进行中医问诊分析。",
            "model_configuration": model_configuration(
                provider_id, model_id, config.model_name
            ),
            "stream": True,
            "enable_thinking": False,
            "attachments": [{"id": attachment["id"]}],
        },
    ) as response:
        response.raise_for_status()
        events = await read_sse_events(response)

    if any(event.get("type") == "tongue_analysis" for event in events):
        raise RuntimeError("非舌像图片被错误输出为结构化舌象")
    status_text = "".join(
        str(event.get("content") or "")
        for event in events
        if event.get("type") in {"舌像分析", "content"}
    )
    if "未识别为有效舌像" not in status_text or "不会据此编造舌象" not in status_text:
        raise RuntimeError(f"非舌像拒识提示不明确: {status_text}")

    response = await client.get(f"/api/v1/attachments/{attachment['id']}")
    response.raise_for_status()
    stored = response_data(response.json()) or {}
    if stored.get("status") != "analysis_failed" or stored.get("analysis_result"):
        raise RuntimeError(f"非舌像附件状态不符合预期: {stored}")
    print("M1_NEGATIVE_E2E=PASS")
    print(f"M1_NEGATIVE_CONVERSATION_ID={conversation_id}")
    print(f"M1_NEGATIVE_ATTACHMENT_ID={attachment['id']}")


async def accept_report(
    client: httpx.AsyncClient,
    *,
    path: Path,
    mime_type: str,
    expected_method: Literal["vision", "text"],
    label: str,
    user_id: str,
    provider_id: str,
    model_id: str,
    config: MultimodalAcceptanceConfig,
) -> None:
    conversation_id = str(uuid4())
    attachment = await upload_attachment(
        client,
        conversation_id=conversation_id,
        path=path,
        kind="medical_report",
        mime_type=mime_type,
    )
    initial_query = (
        "我最近有些乏力，请结合报告进行中医问诊。"
        if label == "IMAGE"
        else (
            "我近两周乏力、轻微头晕，偶有咽干，体温正常，饮食和睡眠尚可，"
            "大小便正常，无胸痛、呼吸困难或晕厥。请先忠实解读报告，再结合症状作中医问诊。"
        )
    )
    events = await run_chat(
        client,
        user_id=user_id,
        conversation_id=conversation_id,
        attachment_id=attachment["id"],
        provider_id=provider_id,
        model_id=model_id,
        model_name=config.model_name,
        query=initial_query,
    )
    analysis = next(event["data"] for event in events if event.get("type") == "report_analysis")
    resume_count = sum(event.get("type") == "interrupt" for event in events)
    if label == "IMAGE" and resume_count < 1:
        raise RuntimeError("M2 报告中断恢复验收未实际触发 interrupt")
    if str(analysis.get("source_attachment_id")) != attachment["id"]:
        raise RuntimeError("报告结果没有保留来源附件 ID")
    if analysis.get("extraction_method") != expected_method:
        raise RuntimeError(f"报告解析方式不符合预期: {analysis}")
    if not analysis.get("summary") or not analysis.get("metrics"):
        raise RuntimeError(f"模型未提取报告语义: {analysis}")
    if "不能替代" not in str(analysis.get("warning")):
        raise RuntimeError("报告结构化结果缺少医生诊断边界提示")
    await verify_attachment_and_message(
        client,
        conversation_id=conversation_id,
        attachment_id=attachment["id"],
        metadata_key="report_analysis",
    )
    detail = await wait_for_case(client, conversation_id)
    case = detail["case"]
    if not case.get("report_analysis") or not case.get("diagnosis_payload"):
        raise RuntimeError("病例未保存报告证据和结构化诊断")
    if not case.get("syndrome_name") or case.get("syndrome_name") == "未明确":
        raise RuntimeError(f"报告结合问诊完成，但病例主证仍未明确: {case}")
    persisted_symptoms = {
        str(item.get("name") if isinstance(item, dict) else item)
        for item in detail.get("symptoms") or []
    }
    expected_positive = {"乏力", "头晕", "咽干"}
    if label == "PDF" and not expected_positive.issubset(persisted_symptoms):
        raise RuntimeError(
            "报告问诊阳性症状未完整进入病例: "
            f"expected={sorted(expected_positive)}, actual={sorted(persisted_symptoms)}"
        )
    forbidden_observations = {
        "体温正常", "饮食和睡眠尚可", "大小便正常", "无胸痛", "无呼吸困难", "无晕厥",
    }
    if persisted_symptoms.intersection(forbidden_observations):
        raise RuntimeError(
            f"正常或否定观察被错误写入病例症状: {sorted(persisted_symptoms)}"
        )
    if (case.get("diagnosis_payload") or {}).get("syndrome") != case.get("syndrome_name"):
        raise RuntimeError("报告病例主证与结构化诊断主证不一致")
    print(f"M2_{label}_E2E=PASS")
    print(f"M2_{label}_CONVERSATION_ID={conversation_id}")
    print(f"M2_{label}_ATTACHMENT_ID={attachment['id']}")
    print(f"M2_{label}_CASE_ID={case['id']}")
    print(f"M2_{label}_REPORT_TYPE={analysis.get('report_type')}")
    print(f"M2_{label}_METRIC_COUNT={len(analysis.get('metrics') or [])}")
    print(f"M2_{label}_RESUME_COUNT={resume_count}")


async def main(mode: str) -> None:
    config = MultimodalAcceptanceConfig()
    timeout = httpx.Timeout(connect=20, read=900, write=120, pool=30)
    async with httpx.AsyncClient(base_url=config.api_base_url, timeout=timeout) as client:
        token, user_id = await login(client, config)
        client.headers["Authorization"] = f"Bearer {token}"
        provider_id, model_id = await find_model_ids(client, config)
        if mode in {"all", "m1"}:
            with tempfile.TemporaryDirectory(prefix="renshu-real-tongue-") as directory:
                tongue_path = await download_public_tongue_fixture(Path(directory))
                await accept_tongue(
                    client,
                    path=tongue_path,
                    user_id=user_id,
                    provider_id=provider_id,
                    model_id=model_id,
                    config=config,
                )
        if mode in {"all", "m1", "m1-negative"}:
            with tempfile.TemporaryDirectory(prefix="renshu-non-tongue-") as directory:
                image_path, _ = create_synthetic_report_files(Path(directory))
                await accept_non_tongue(
                    client,
                    path=image_path,
                    user_id=user_id,
                    provider_id=provider_id,
                    model_id=model_id,
                    config=config,
                )
        if mode in {"all", "m2", "m2-image", "m2-pdf"}:
            with tempfile.TemporaryDirectory(prefix="renshu-report-") as directory:
                image_path, pdf_path = create_synthetic_report_files(Path(directory))
                if mode in {"all", "m2", "m2-image"}:
                    await accept_report(
                        client,
                        path=image_path,
                        mime_type="image/png",
                        expected_method="vision",
                        label="IMAGE",
                        user_id=user_id,
                        provider_id=provider_id,
                        model_id=model_id,
                        config=config,
                    )
                if mode in {"all", "m2", "m2-pdf"}:
                    await accept_report(
                        client,
                        path=pdf_path,
                        mime_type="application/pdf",
                        expected_method="text",
                        label="PDF",
                        user_id=user_id,
                        provider_id=provider_id,
                        model_id=model_id,
                        config=config,
                    )
    print("MULTIMODAL_ACCEPTANCE=PASS")
    print(f"MODEL={config.model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["all", "m1", "m1-negative", "m2", "m2-image", "m2-pdf"],
    )
    args = parser.parse_args()
    asyncio.run(main(args.mode))
