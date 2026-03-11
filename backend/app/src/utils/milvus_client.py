"""
Milvus 客户端封装。

基于 Milvus 官方 `pymilvus.MilvusClient` 2.6.x 文档实现，
默认采用官方推荐的高层 CRUD 接口，并支持自定义 schema + index_params 建表。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from app.src.utils.logs.logger import get_logger


logger = get_logger(__name__)


try:
    from pymilvus import DataType, MilvusClient

    PYMILVUS_AVAILABLE = True
except ImportError:
    DataType = None
    MilvusClient = None
    PYMILVUS_AVAILABLE = False


@dataclass
class MilvusConnectionConfig:
    """Milvus 连接配置。"""

    uri: str = "http://localhost:19530"
    token: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    db_name: str = "default"
    timeout: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "MilvusConnectionConfig":
        uri = os.getenv("MILVUS_URL") or os.getenv("MILVUS_URI") or "http://localhost:19530"
        timeout = os.getenv("MILVUS_TIMEOUT")
        return cls(
            uri=uri,
            token=os.getenv("MILVUS_TOKEN") or None,
            user=os.getenv("MILVUS_USER") or None,
            password=os.getenv("MILVUS_PASSWORD") or None,
            db_name=os.getenv("MILVUS_DB_NAME", "default"),
            timeout=float(timeout) if timeout else None,
        )


@dataclass
class MilvusFieldConfig:
    """Milvus 标量/向量字段配置。"""

    name: str
    datatype: str
    description: str = ""
    is_primary: bool = False
    auto_id: Optional[bool] = None
    is_partition_key: bool = False
    nullable: Optional[bool] = None
    default_value: Optional[Any] = None
    max_length: Optional[int] = None
    dim: Optional[int] = None
    element_type: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_schema_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "field_name": self.name,
            "datatype": self.datatype,
        }
        if self.description:
            kwargs["description"] = self.description
        if self.is_primary:
            kwargs["is_primary"] = True
        if self.auto_id is not None:
            kwargs["auto_id"] = self.auto_id
        if self.is_partition_key:
            kwargs["is_partition_key"] = True
        if self.nullable is not None:
            kwargs["nullable"] = self.nullable
        if self.default_value is not None:
            kwargs["default_value"] = self.default_value
        if self.max_length is not None:
            kwargs["max_length"] = self.max_length
        if self.dim is not None:
            kwargs["dim"] = self.dim
        if self.element_type:
            kwargs["element_type"] = self.element_type
        if self.extra:
            kwargs.update(self.extra)
        return kwargs


@dataclass
class MilvusIndexConfig:
    """Milvus 向量索引配置。"""

    field_name: str = "vector"
    index_type: str = "IVF_FLAT"
    metric_type: str = "COSINE"
    index_name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=lambda: {"nlist": 1024})


@dataclass
class MilvusCollectionConfig:
    """Milvus 集合配置。"""

    collection_name: str
    dimension: int
    id_field_name: str = "id"
    id_type: str = "VARCHAR"
    id_max_length: int = 128
    vector_field_name: str = "vector"
    auto_id: bool = False
    enable_dynamic_field: bool = True
    scalar_fields: List[MilvusFieldConfig] = field(default_factory=list)
    vector_index: Optional[MilvusIndexConfig] = None

    def build_primary_field(self) -> MilvusFieldConfig:
        datatype = self.id_type.upper()
        max_length = self.id_max_length if datatype == "VARCHAR" else None
        return MilvusFieldConfig(
            name=self.id_field_name,
            datatype=datatype,
            is_primary=True,
            auto_id=self.auto_id,
            max_length=max_length,
        )

    def build_vector_field(self) -> MilvusFieldConfig:
        return MilvusFieldConfig(
            name=self.vector_field_name,
            datatype="FLOAT_VECTOR",
            dim=self.dimension,
        )

    def build_index(self) -> MilvusIndexConfig:
        if self.vector_index is not None:
            return self.vector_index
        return MilvusIndexConfig(field_name=self.vector_field_name)


class MilvusVectorClient:
    """基于官方 `MilvusClient` 的轻量封装。"""

    def __init__(self, connection: Optional[MilvusConnectionConfig] = None):
        self.connection = connection or MilvusConnectionConfig.from_env()
        self._client = None

    @staticmethod
    def _require_pymilvus() -> None:
        if not PYMILVUS_AVAILABLE:
            raise ImportError(
                "未安装 pymilvus，请先执行: pip install pymilvus 或 uv add pymilvus"
            )

    @staticmethod
    def _resolve_data_type(datatype: Any) -> Any:
        if not PYMILVUS_AVAILABLE:
            return datatype
        if datatype is None:
            raise ValueError("datatype 不能为空")
        if not isinstance(datatype, str):
            return datatype
        resolved = getattr(DataType, datatype.upper(), None)
        if resolved is None:
            raise ValueError(f"不支持的 Milvus DataType: {datatype}")
        return resolved

    @property
    def client(self):
        if self._client is None:
            self.connect()
        return self._client

    def connect(self) -> None:
        self._require_pymilvus()
        kwargs: Dict[str, Any] = {
            "uri": self.connection.uri,
            "db_name": self.connection.db_name,
        }
        if self.connection.token:
            kwargs["token"] = self.connection.token
        elif self.connection.user and self.connection.password:
            kwargs["user"] = self.connection.user
            kwargs["password"] = self.connection.password
        if self.connection.timeout is not None:
            kwargs["timeout"] = self.connection.timeout
        if self.connection.extra:
            kwargs.update(self.connection.extra)

        self._client = MilvusClient(**kwargs)
        logger.info("Milvus connected: %s", self.connection.uri)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def list_collections(self) -> List[str]:
        return list(self.client.list_collections())

    def has_collection(self, collection_name: str) -> bool:
        return bool(self.client.has_collection(collection_name=collection_name))

    def describe_collection(self, collection_name: str) -> Dict[str, Any]:
        return self.client.describe_collection(collection_name=collection_name)

    def drop_collection(self, collection_name: str) -> None:
        if self.has_collection(collection_name):
            self.client.drop_collection(collection_name=collection_name)

    def load_collection(
        self,
        collection_name: str,
        replica_number: int = 1,
        timeout: Optional[float] = None,
    ) -> None:
        self.client.load_collection(
            collection_name=collection_name,
            replica_number=replica_number,
            timeout=timeout,
        )

    def get_load_state(self, collection_name: str) -> Dict[str, Any]:
        return self.client.get_load_state(collection_name=collection_name)

    def ensure_collection(self, config: MilvusCollectionConfig) -> Dict[str, Any]:
        if self.has_collection(config.collection_name):
            return self.describe_collection(config.collection_name)

        schema = MilvusClient.create_schema(
            auto_id=config.auto_id,
            enable_dynamic_field=config.enable_dynamic_field,
        )

        fields = [config.build_primary_field(), config.build_vector_field(), *config.scalar_fields]
        seen_fields = set()
        for field in fields:
            if field.name in seen_fields:
                raise ValueError(f"重复的 Milvus 字段名: {field.name}")
            seen_fields.add(field.name)

            field_kwargs = field.to_schema_kwargs()
            field_kwargs["datatype"] = self._resolve_data_type(field_kwargs["datatype"])
            if "element_type" in field_kwargs:
                field_kwargs["element_type"] = self._resolve_data_type(field_kwargs["element_type"])
            schema.add_field(**field_kwargs)

        index = config.build_index()
        index_params = MilvusClient.prepare_index_params()
        index_kwargs: Dict[str, Any] = {
            "field_name": index.field_name,
            "index_type": index.index_type,
            "metric_type": index.metric_type,
            "params": dict(index.params),
        }
        if index.index_name:
            index_kwargs["index_name"] = index.index_name
        index_params.add_index(**index_kwargs)

        self.client.create_collection(
            collection_name=config.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info("Milvus collection created: %s", config.collection_name)
        return self.describe_collection(config.collection_name)

    def insert(
        self,
        collection_name: str,
        data: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        partition_name: str = "",
        timeout: Optional[float] = None,
    ) -> List[Any]:
        return self.client.insert(
            collection_name=collection_name,
            data=data,
            partition_name=partition_name,
            timeout=timeout,
        )

    def upsert(
        self,
        collection_name: str,
        data: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        partition_name: str = "",
        timeout: Optional[float] = None,
        partial_update: bool = False,
    ) -> List[Any]:
        return self.client.upsert(
            collection_name=collection_name,
            data=data,
            partition_name=partition_name,
            timeout=timeout,
            partial_update=partial_update,
        )

    def search(
        self,
        collection_name: str,
        data: Optional[Sequence[Sequence[float]]] = None,
        ids: Optional[Sequence[str] | Sequence[int]] = None,
        anns_field: str = "vector",
        limit: int = 10,
        filter: str = "",
        output_fields: Optional[List[str]] = None,
        search_params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Any]:
        kwargs: Dict[str, Any] = {
            "collection_name": collection_name,
            "anns_field": anns_field,
            "limit": limit,
            "filter": filter,
            "output_fields": output_fields,
            "timeout": timeout,
        }
        if search_params:
            kwargs["search_params"] = search_params
        if data is not None:
            kwargs["data"] = data
        if ids is not None:
            kwargs["ids"] = ids
        return self.client.search(**kwargs)

    def query(
        self,
        collection_name: str,
        filter: str = "",
        output_fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
        timeout: Optional[float] = None,
        partition_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        kwargs: Dict[str, Any] = {
            "collection_name": collection_name,
            "filter": filter,
            "output_fields": output_fields,
            "timeout": timeout,
        }
        if partition_names is not None:
            kwargs["partition_names"] = partition_names
        if limit is not None:
            kwargs["limit"] = limit
        return self.client.query(**kwargs)

    def get(
        self,
        collection_name: str,
        ids: Sequence[str] | Sequence[int],
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        return self.client.get(
            collection_name=collection_name,
            ids=ids,
            output_fields=output_fields,
            timeout=timeout,
        )

    def delete(
        self,
        collection_name: str,
        ids: Optional[Sequence[str] | Sequence[int] | str | int] = None,
        filter: str = "",
        partition_name: str = "",
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        return self.client.delete(
            collection_name=collection_name,
            ids=ids,
            filter=filter,
            partition_name=partition_name,
            timeout=timeout,
        )


__all__ = [
    "MilvusConnectionConfig",
    "MilvusFieldConfig",
    "MilvusIndexConfig",
    "MilvusCollectionConfig",
    "MilvusVectorClient",
]

