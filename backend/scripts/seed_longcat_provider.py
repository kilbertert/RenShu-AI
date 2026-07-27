"""为本地端到端验收写入 LongCat Anthropic 兼容供应商与模型元数据。"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from pydantic import BaseModel, Field
from sqlmodel import select

from app.src.common.config.prosgresql_config import async_db_manager, create_db_tables
from app.src.model.model_config_models import SystemModelDefinition, SystemModelProvider


class LongCatSeedConfig(BaseModel):
    provider_name: str = "anthropic"
    provider_label: str = "LongCat (Anthropic Compatible)"
    base_url: str = Field(min_length=8)
    model_name: str = Field(min_length=1)


async def seed(config: LongCatSeedConfig) -> tuple[str, str]:
    await async_db_manager.init()
    await create_db_tables()

    try:
        async with async_db_manager.get_session() as session:
            provider_result = await session.exec(
                select(SystemModelProvider).where(
                    SystemModelProvider.name == config.provider_name
                )
            )
            provider = provider_result.one_or_none()
            if provider is None:
                provider = SystemModelProvider(
                    name=config.provider_name,
                    label=config.provider_label,
                    description="LongCat Anthropic-compatible API for RenShu-AI",
                    default_base_url=config.base_url,
                    supported_model_types=["chat"],
                    position=1,
                )
            else:
                provider.label = config.provider_label
                provider.default_base_url = config.base_url
            session.add(provider)
            await session.flush()

            model_result = await session.exec(
                select(SystemModelDefinition).where(
                    SystemModelDefinition.provider_id == provider.id,
                    SystemModelDefinition.model_name == config.model_name,
                )
            )
            model = model_result.one_or_none()
            if model is None:
                model = SystemModelDefinition(
                    provider_id=provider.id,
                    model_name=config.model_name,
                    label=config.model_name,
                    description="LongCat 2.0 reasoning and tool-use model",
                    model_type="llm",
                    features=["thinking", "tool_call", "structured_output", "streaming"],
                    context_window=128000,
                    default_max_tokens=4096,
                    default_parameters={"temperature": 0.2, "top_p": 0.9},
                    position=1,
                    is_enabled=True,
                )
            else:
                model.is_enabled = True
                model.features = ["thinking", "tool_call", "structured_output", "streaming"]
            session.add(model)
            await session.commit()
            return str(provider.id), str(model.id)
    finally:
        await async_db_manager.close()


async def main() -> None:
    config = LongCatSeedConfig(
        base_url=os.environ.get("ANTHROPIC_BASE_URL", ""),
        model_name=os.environ.get("LONGCAT_MODEL", "LongCat-2.0"),
    )
    provider_id, model_id = await seed(config)
    print(f"provider_id={provider_id}")
    print(f"model_id={model_id}")


if __name__ == "__main__":
    asyncio.run(main())
