"""
TCM Classic Ingestor
中医古籍导入器

从JSON文件批量导入古籍条文到Neo4j
"""

import json
from typing import Optional

from .models import ClassicRecord, IngestResult
from .embedder import get_embedder, BaseEmbedder
from ..tcm_neo4j import get_tcm_neo4j_connection


# Cypher语句模板
CREATE_CLASSIC_CYPHER = """
UNWIND $records AS record
MERGE (c:Classic {book_name: record.book_name, chapter: record.chapter, title: record.title})
SET c.content = record.content,
    c.interpretation = record.interpretation,
    c.keywords = record.keywords,
    c.content_embedding = record.content_embedding
WITH c, record
UNWIND record.related_syndromes AS syndrome_name
MERGE (s:Syndrome {name: syndrome_name})
MERGE (c)-[:DISCUSSES]->(s)
WITH c, record
UNWIND record.related_prescriptions AS prescription_name
MERGE (p:Prescription {name: prescription_name})
MERGE (c)-[:MENTIONS]->(p)
"""

CREATE_CLASSIC_SIMPLE_CYPHER = """
UNWIND $records AS record
MERGE (c:Classic {book_name: record.book_name, chapter: record.chapter, title: record.title})
SET c.content = record.content,
    c.interpretation = record.interpretation,
    c.keywords = record.keywords,
    c.content_embedding = record.content_embedding
"""

CREATE_SYNDROME_RELATIONS_CYPHER = """
UNWIND $relations AS rel
MATCH (c:Classic {book_name: rel.book_name, chapter: rel.chapter, title: rel.title})
MERGE (s:Syndrome {name: rel.syndrome_name})
MERGE (c)-[:DISCUSSES]->(s)
"""

CREATE_PRESCRIPTION_RELATIONS_CYPHER = """
UNWIND $relations AS rel
MATCH (c:Classic {book_name: rel.book_name, chapter: rel.chapter, title: rel.title})
MERGE (p:Prescription {name: rel.prescription_name})
MERGE (c)-[:MENTIONS]->(p)
"""


class ClassicIngestor:
    """古籍导入器"""

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        batch_size: int = 50
    ):
        """
        初始化古籍导入器

        Args:
            embedder: 嵌入器实例，如果为None则自动创建
            batch_size: 批量处理大小
        """
        self.embedder = embedder or get_embedder()
        self.batch_size = batch_size
        self.conn = get_tcm_neo4j_connection()

    async def ingest_from_json(self, file_path: str) -> IngestResult:
        """
        从JSON文件导入古籍数据

        Args:
            file_path: JSON文件路径

        Returns:
            IngestResult: 导入结果
        """
        result = IngestResult(
            success=True,
            total_records=0,
            imported_count=0,
            failed_count=0,
            errors=[],
            message=""
        )

        try:
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            records = data if isinstance(data, list) else data.get("records", [])
            result.total_records = len(records)

            if not records:
                result.message = "JSON文件中没有记录"
                return result

            # 转换为ClassicRecord对象
            classic_records = []
            for item in records:
                try:
                    record = ClassicRecord(**item)
                    classic_records.append(record)
                except Exception as e:
                    result.errors.append(f"记录解析失败: {str(e)}")
                    result.failed_count += 1

            # 批量生成向量嵌入
            texts_to_embed = [
                f"{r.book_name} {r.chapter} {r.content}"
                for r in classic_records
            ]

            try:
                embeddings = await self.embedder.embed_texts(texts_to_embed)
                for i, record in enumerate(classic_records):
                    if i < len(embeddings):
                        record.content_embedding = embeddings[i]
            except Exception as e:
                result.errors.append(f"向量嵌入生成失败: {str(e)}")
                # 继续导入，但没有向量

            # 批量导入到Neo4j
            for i in range(0, len(classic_records), self.batch_size):
                batch = classic_records[i:i + self.batch_size]
                try:
                    await self._import_batch(batch)
                    result.imported_count += len(batch)
                except Exception as e:
                    result.errors.append(f"批次 {i//self.batch_size + 1} 导入失败: {str(e)}")
                    result.failed_count += len(batch)

            result.message = f"导入完成: {result.imported_count}/{result.total_records} 成功"

        except FileNotFoundError:
            result.success = False
            result.message = f"文件不存在: {file_path}"
        except json.JSONDecodeError as e:
            result.success = False
            result.message = f"JSON解析失败: {str(e)}"
        except Exception as e:
            result.success = False
            result.message = f"导入失败: {str(e)}"

        return result

    async def _import_batch(self, records: list[ClassicRecord]):
        """
        批量导入记录

        Args:
            records: 古籍记录列表
        """
        # 准备数据
        record_dicts = []
        syndrome_relations = []
        prescription_relations = []

        for record in records:
            record_dict = {
                "book_name": record.book_name,
                "chapter": record.chapter,
                "title": record.title,
                "content": record.content,
                "interpretation": record.interpretation,
                "keywords": record.keywords,
                "content_embedding": record.content_embedding,
            }
            record_dicts.append(record_dict)

            # 收集关系数据
            for syndrome in record.related_syndromes:
                syndrome_relations.append({
                    "book_name": record.book_name,
                    "chapter": record.chapter,
                    "title": record.title,
                    "syndrome_name": syndrome
                })

            for prescription in record.related_prescriptions:
                prescription_relations.append({
                    "book_name": record.book_name,
                    "chapter": record.chapter,
                    "title": record.title,
                    "prescription_name": prescription
                })

        # 执行导入
        self.conn.execute_query(CREATE_CLASSIC_SIMPLE_CYPHER, {"records": record_dicts})

        # 创建关系
        if syndrome_relations:
            self.conn.execute_query(
                CREATE_SYNDROME_RELATIONS_CYPHER,
                {"relations": syndrome_relations}
            )

        if prescription_relations:
            self.conn.execute_query(
                CREATE_PRESCRIPTION_RELATIONS_CYPHER,
                {"relations": prescription_relations}
            )

    async def ingest_single(self, record: ClassicRecord) -> bool:
        """
        导入单条记录

        Args:
            record: 古籍记录

        Returns:
            bool: 是否成功
        """
        try:
            # 生成向量嵌入
            if not record.content_embedding:
                text = f"{record.book_name} {record.chapter} {record.content}"
                record.content_embedding = await self.embedder.embed_text(text)

            await self._import_batch([record])
            return True
        except Exception:
            return False
