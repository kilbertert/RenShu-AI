"""写入 Qwen OpenAI 兼容视觉模型，并为指定验收用户加密保存凭据。

必需环境变量：
    QWEN_VISION_BASE_URL
    QWEN_VISION_API_KEY

可选环境变量：
    QWEN_VISION_MODEL（默认 qwen3.6-flash）
    QWEN_VISION_USER_EMAIL（默认端到端验收用户）
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from pydantic import BaseModel, Field, SecretStr
from sqlmodel import select

from app.src.common.config.prosgresql_config import async_db_manager, create_db_tables
from app.src.model.account_model import Account
from app.src.model.model_config_models import (
    SystemModelDefinition,
    SystemModelProvider,
    UserProviderConfig,
)
from app.src.utils.auth_utils import encrypt_api_key


class QwenVisionSeedConfig(BaseModel):
    provider_name: str = "qwen"
    provider_label: str = "Qwen MaaS (OpenAI Compatible)"
    base_url: str = Field(min_length=8)
    model_name: str = Field(default="qwen3.6-flash", min_length=1)
    user_email: str = Field(min_length=3)
    api_key: SecretStr


async def seed(config: QwenVisionSeedConfig) -> tuple[str, str, str]:
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
                    description="Qwen OpenAI-compatible multimodal endpoint",
                    default_base_url=config.base_url.rstrip("/"),
                    supported_model_types=["chat", "vision"],
                    position=2,
                )
            else:
                provider.label = config.provider_label
                provider.default_base_url = config.base_url.rstrip("/")
                provider.supported_model_types = ["chat", "vision"]
            session.add(provider)
            await session.flush()

            model_result = await session.exec(
                select(SystemModelDefinition).where(
                    SystemModelDefinition.provider_id == provider.id,
                    SystemModelDefinition.model_name == config.model_name,
                )
            )
            model = model_result.one_or_none()
            features = [
                "image_input",
                "structured_output",
                "streaming",
            ]
            if model is None:
                model = SystemModelDefinition(
                    provider_id=provider.id,
                    model_name=config.model_name,
                    label="Qwen3.6 Flash",
                    description="Qwen3.6 Flash multimodal model verified by semantic image probe",
                    model_type="multimodal",
                    features=features,
                    context_window=128000,
                    default_max_tokens=4096,
                    default_parameters={"temperature": 0.1, "top_p": 0.9},
                    position=1,
                    is_enabled=True,
                )
            else:
                model.label = "Qwen3.6 Flash"
                model.model_type = "multimodal"
                model.features = features
                model.is_enabled = True
            session.add(model)
            await session.flush()

            account_result = await session.exec(
                select(Account).where(Account.email == config.user_email)
            )
            account = account_result.one_or_none()
            if account is None:
                raise RuntimeError(f"未找到目标用户: {config.user_email}")

            user_config_result = await session.exec(
                select(UserProviderConfig).where(
                    UserProviderConfig.user_id == account.id,
                    UserProviderConfig.provider_id == provider.id,
                )
            )
            user_config = user_config_result.one_or_none()
            if user_config is None:
                user_config = UserProviderConfig(
                    user_id=account.id,
                    provider_id=provider.id,
                )
            user_config.api_key = encrypt_api_key(config.api_key.get_secret_value())
            user_config.base_url_override = config.base_url.rstrip("/")
            user_config.is_enabled = True
            session.add(user_config)
            await session.commit()
            return str(provider.id), str(model.id), str(account.id)
    finally:
        await async_db_manager.close()


async def main() -> None:
    config = QwenVisionSeedConfig(
        base_url=os.environ.get("QWEN_VISION_BASE_URL", ""),
        api_key=SecretStr(os.environ.get("QWEN_VISION_API_KEY", "")),
        model_name=os.environ.get("QWEN_VISION_MODEL", "qwen3.6-flash"),
        user_email=os.environ.get(
            "QWEN_VISION_USER_EMAIL",
            "renshu-e2e-20260720@example.com",
        ),
    )
    provider_id, model_id, user_id = await seed(config)
    print("QWEN_VISION_SEED=PASS")
    print(f"provider_id={provider_id}")
    print(f"model_id={model_id}")
    print(f"user_id={user_id}")


if __name__ == "__main__":
    asyncio.run(main())
