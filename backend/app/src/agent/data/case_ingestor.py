"""
TCM Case Ingestor
中医医案导入器

从JSON文件批量导入医案到Neo4j
"""

import json
from typing import Optional

from .models import CaseRecord, IngestResult
from .embedder import get_embedder, BaseEmbedder
from ..tcm_neo4j import get_tcm_neo4j_connection


# Cypher语句模板
CREATE_CASE_SIMPLE_CYPHER = """
UNWIND $records AS record
MERGE (ca:Case {case_id: record.case_id})
SET ca.source = record.source,
    ca.doctor_name = record.doctor_name,
    ca.patient_info = record.patient_info,
    ca.chief_complaint = record.chief_complaint,
    ca.symptoms = record.symptoms,
    ca.tongue = record.tongue,
    ca.pulse = record.pulse,
    ca.syndrome = record.syndrome,
    ca.treatment_principle = record.treatment_principle,
    ca.prescription = record.prescription,
    ca.prescription_herbs = record.prescription_herbs,
    ca.outcome = record.outcome,
    ca.notes = record.notes,
    ca.case_embedding = record.case_embedding
"""

CREATE_CASE_SYNDROME_RELATIONS_CYPHER = """
UNWIND $relations AS rel
MATCH (ca:Case {case_id: rel.case_id})
MERGE (s:Syndrome {name: rel.syndrome_name})
MERGE (ca)-[:DIAGNOSED_AS]->(s)
"""

CREATE_CASE_PRESCRIPTION_RELATIONS_CYPHER = """
UNWIND $relations AS rel
MATCH (ca:Case {case_id: rel.case_id})
MERGE (p:Prescription {name: rel.prescription_name})
MERGE (ca)-[:USED]->(p)
"""

CREATE_CASE_HERB_RELATIONS_CYPHER = """
UNWIND $relations AS rel
MATCH (ca:Case {case_id: rel.case_id})
MERGE (h:Herb {name: rel.herb_name})
MERGE (ca)-[:CONTAINS_HERB]->(h)
"""


class CaseIngestor:
    """医案导入器"""

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        batch_size: int = 50
    ):
        """
        初始化医案导入器

        Args:
            embedder: 嵌入器实例，如果为None则自动创建
            batch_size: 批量处理大小
        """
        self.embedder = embedder or get_embedder()
        self.batch_size = batch_size
        self.conn = get_tcm_neo4j_connection()

    async def ingest_from_json(self, file_path: str) -> IngestResult:
        """
        从JSON文件导入医案数据

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

            # 转换为CaseRecord对象
            case_records = []
            for item in records:
                try:
                    record = CaseRecord(**item)
                    case_records.append(record)
                except Exception as e:
                    result.errors.append(f"记录解析失败: {str(e)}")
                    result.failed_count += 1

            # 批量生成向量嵌入
            texts_to_embed = [
                f"{r.chief_complaint} {' '.join(r.symptoms)} {r.syndrome} {r.prescription}"
                for r in case_records
            ]

            try:
                embeddings = await self.embedder.embed_texts(texts_to_embed)
                for i, record in enumerate(case_records):
                    if i < len(embeddings):
                        record.case_embedding = embeddings[i]
            except Exception as e:
                result.errors.append(f"向量嵌入生成失败: {str(e)}")
                # 继续导入，但没有向量

            # 批量导入到Neo4j
            for i in range(0, len(case_records), self.batch_size):
                batch = case_records[i:i + self.batch_size]
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

    async def _import_batch(self, records: list[CaseRecord]):
        """
        批量导入记录

        Args:
            records: 医案记录列表
        """
        # 准备数据
        record_dicts = []
        syndrome_relations = []
        prescription_relations = []
        herb_relations = []

        for record in records:
            record_dict = {
                "case_id": record.case_id,
                "source": record.source,
                "doctor_name": record.doctor_name,
                "patient_info": record.patient_info,
                "chief_complaint": record.chief_complaint,
                "symptoms": record.symptoms,
                "tongue": record.tongue,
                "pulse": record.pulse,
                "syndrome": record.syndrome,
                "treatment_principle": record.treatment_principle,
                "prescription": record.prescription,
                "prescription_herbs": record.prescription_herbs,
                "outcome": record.outcome,
                "notes": record.notes,
                "case_embedding": record.case_embedding,
            }
            record_dicts.append(record_dict)

            # 收集证型关系
            if record.syndrome:
                syndrome_relations.append({
                    "case_id": record.case_id,
                    "syndrome_name": record.syndrome
                })

            # 收集方剂关系
            if record.prescription:
                prescription_relations.append({
                    "case_id": record.case_id,
                    "prescription_name": record.prescription
                })

            # 收集药材关系
            for herb_info in record.prescription_herbs:
                herb_name = herb_info.get("herb", "")
                if herb_name:
                    herb_relations.append({
                        "case_id": record.case_id,
                        "herb_name": herb_name
                    })

        # 执行导入
        self.conn.execute_query(CREATE_CASE_SIMPLE_CYPHER, {"records": record_dicts})

        # 创建关系
        if syndrome_relations:
            self.conn.execute_query(
                CREATE_CASE_SYNDROME_RELATIONS_CYPHER,
                {"relations": syndrome_relations}
            )

        if prescription_relations:
            self.conn.execute_query(
                CREATE_CASE_PRESCRIPTION_RELATIONS_CYPHER,
                {"relations": prescription_relations}
            )

        if herb_relations:
            self.conn.execute_query(
                CREATE_CASE_HERB_RELATIONS_CYPHER,
                {"relations": herb_relations}
            )

    async def ingest_single(self, record: CaseRecord) -> bool:
        """
        导入单条记录

        Args:
            record: 医案记录

        Returns:
            bool: 是否成功
        """
        try:
            # 生成向量嵌入
            if not record.case_embedding:
                text = f"{record.chief_complaint} {' '.join(record.symptoms)} {record.syndrome}"
                record.case_embedding = await self.embedder.embed_text(text)

            await self._import_batch([record])
            return True
        except Exception:
            return False
