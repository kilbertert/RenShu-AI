# 仁术AI · Neo4j/GraphRAG 落地任务规划

> 蓝图文档。用于新会话中按 P0 → P1 → P2 顺序实施。
> 编写依据：两轮对话（README 审查 + 代码现状调查） + 实际代码状态。

---

## 0. 项目现状基线（实施前必读）

### 0.1 README 描述 vs 代码实情的差距

| 维度 | README 声称 | 代码实情 | 差距 |
|------|-------------|----------|------|
| 核心定位 | 融合 GraphRAG 的多智能体中医系统 | 多智能体框架真实可用，GraphRAG 是装饰 | 一半空转 |
| 意图识别 | 6 类核心场景，准确率 90%+ | 5 类（缺"图文解析"），准确率未量化 | 文档虚标 |
| Cypher 模板 | 100+ 中医专属模板 | kg_tools.py 仅 2 个函数 + tcm_neo4j.py 通用 wrapper | 数量虚报 50 倍 |
| PostgreSQL → Neo4j 映射 | 打通结构化与图数据 | migrate_neo4j_schema.py 只定义 schema，无同步链路 | 文档化层 |
| 数据源 | 20+ 古籍、卫健委指南、三甲医案 | origin_data/ 里是 Northwind 演示数据（Product/Order/Customer） | **错位** |
| 工具校验 | 五重校验 | moderate_diagnosis.py 两个查询标 TODO 返回 mock | 核心特性未实现 |
| 自我纠正子图 | 自动纠正证型-方剂错误 | 代码中无对应子图 | 设计文档级 |
| MCP 服务 | 国家中医药管理局等 | 无任何 MCP 接入代码 | 纯文档 |
| 多模态（TCM-CV） | 接入中医多模态模型 | multimodal_prompts.py 存在但未调模型 | 占位 |
| 生产部署 | 字段级加密、Redis、K8s | SECRET_KEY 默认值、CORS 未收敛 | demo 级别 |

**结论**：README 是产品立项书 / 投标技术方案，代码是多智能体对话框架的早期原型。**真正能用的部分约 20-25%**。

### 0.2 Neo4j / GraphRAG 实际状态

**Neo4j：**
- ✅ `backend/app/src/core/tcm_neo4j.py`：基于 `langchain_community.graphs.Neo4jGraph` 的单例实现，能正常连接
- ✅ `backend/scripts/migrate_neo4j_schema.py`：定义 Symptom / Syndrome / Formula / Organ 节点和 INDICATES / TREATS_WITH 关系
- ❌ `backend/app/src/core/graph_db.py`：**不存在**
- ❌ `kg_tools.py:47,121` `from app.src.core.graph_db import get_neo4j_graph` → **ModuleNotFoundError 必然抛**
- ❌ `origin_data/create_neo4j_import.py` 导入 Northwind 演示数据，与中医无关
- ❌ `moderate_diagnosis/moderate_diagnosis.py:303,345` 的两个查询 `# TODO: 实现知识图谱查询`，直接返回 mock

**GraphRAG 目录：**
- ✅ `backend/graphrag/dev/graphrag_api.py`：真实 FastAPI 服务，封装微软 GraphRAG 的 `local_search` / `global_search` / `drift_search`
- ❌ `PROJECT_DIR` 硬编码为 `D:\code\SmartTCM-Agent-SYSTEM\graphrag`（**项目外路径**）
- ❌ 索引数据也是 Northwind

**PostgreSQL：**
- ✅ 完整的 SQLAlchemy 模型、migration 脚本、CRUD service
- ⚠️ 只用于"用户/会话/对话/模型配置"通用元数据
- ❌ README 第 56 行承诺的"病例表（id / 症状 / 辨证结果）"缺失
- ❌ 病例信息全部塞在 LangGraph checkpointer state 里，无法跨会话关联

### 0.3 当前调用链断点

```
moderate_diagnosis 节点
  └─ _query_similar_syndromes()      → TODO，返回 mock
  └─ _query_related_prescriptions()  → TODO，返回 mock

complex_diagnosis 节点
  └─ kg_syndrome_search (kg_tools.py)
       └─ from app.src.core.graph_db import get_neo4j_graph
            └─ ModuleNotFoundError
                 └─ except 分支返回 mock 数据
```

**任何走到 moderate / complex 路径的请求，实际收不到任何真实图谱知识增强。**

---

## 1. 统一优先级（与两轮对话收敛后的最终口径）

### 1.1 排序逻辑

不是反复，是**价值密度的重新计算**：

| 单独项 | 用户感知 | 结论 |
|--------|----------|------|
| 只恢复 graph_db.py（无数据） | 几乎无（查询返回空 ≈ mock） | 单独做没意义 |
| 恢复 + 真实数据 | 立刻看到辨证更准 | 合并为同一里程碑 |
| + 病例库落库 | 用户感知要几周才显现 | 排到下个月 |

### 1.2 三阶段优先级

**P0 — 第一天（半天）**：恢复调用链
**P1 — 第一周（2-3 天）**：灌入 TCM 真实数据
**P2 — 第一月（5-7 天）**：病例库结构化

> 答用户的统一口径问题：**先做 P0 + P1（合并成第一个里程碑），P2 排到下个月**。

---

## 2. P0 — 恢复 graph_db 模块 + 解开调用链

### 2.1 目标

让 moderate / complex 诊断路径**不再因模块缺失崩溃**，能跑通到"尝试查询 Neo4j"这一步。**不要求 Neo4j 有数据**（这是 P1 的事）。

### 2.2 交付物

| 序号 | 交付物 | 路径 |
|------|--------|------|
| P0-1 | `graph_db.py` 模块（get_neo4j_graph 单例） | `backend/app/src/core/graph_db.py` |
| P0-2 | `kg_tools.py` 改用真实路径，不再引用缺失模块 | `backend/app/src/agent/components/diagnose/nodes/complex/tools/kg_tools.py` |
| P0-3 | `moderate_diagnosis` 两个 TODO 改为真实 Cypher（可返回空） | `backend/app/src/agent/components/diagnose/nodes/moderate_diagnosis/moderate_diagnosis.py` |
| P0-4 | 单元测试：模块能 import、Cypher 优雅降级 | `backend/tests/core/test_graph_db.py` |
| P0-5 | 集成验证：现有测试套件 + 手动跑一次简单问诊 | — |

### 2.3 P0-1：graph_db.py 实现规格

```python
# backend/app/src/core/graph_db.py
"""
Neo4j 图数据库连接管理（单例模式）

提供 get_neo4j_graph() 全局访问点，供 kg_tools.py 等模块使用。
环境变量：NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB
"""
from functools import lru_cache
from typing import Optional
from langchain_community.graphs import Neo4jGraph
from app.src.utils import get_logger

logger = get_logger("graph_db")

@lru_cache(maxsize=1)
def get_neo4j_graph(database: Optional[str] = None) -> Optional[Neo4jGraph]:
    """
    获取 Neo4j 图连接（单例）

    Returns:
        Neo4jGraph: 连接对象
        None: 配置缺失时返回 None，调用方应做降级处理
    """
    import os
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    db = database or os.getenv("NEO4J_DB", "neo4j")

    if not all([uri, user, password]):
        logger.warning("Neo4j 配置缺失，graph_db 将不可用")
        return None

    try:
        return Neo4jGraph(
            url=uri,
            username=user,
            password=password,
            database=db,
        )
    except Exception as e:
        logger.error(f"Neo4j 连接失败: {e}")
        return None
```

**关键决策**：
- 用 `lru_cache` 实现单例（tcm_neo4j.py 也是这个模式，保持一致）
- 配置缺失时返回 `None` 而非抛异常 → 调用方做降级
- 不强制 `NEO4J_DB` 环境变量（兼容现有部署）

### 2.4 P0-2：kg_tools.py 修复

**当前代码**（kg_tools.py:47, 121）：
```python
from app.src.core.graph_db import get_neo4j_graph  # ModuleNotFoundError
```

**修改后**：
```python
try:
    from app.src.core.graph_db import get_neo4j_graph
    GRAPH_DB_AVAILABLE = True
except ImportError:
    GRAPH_DB_AVAILABLE = False
    def get_neo4j_graph(database=None):
        return None
```

并把 `kg_syndrome_search` / `kg_organ_query` 的 try/except 从"包住 Cypher 执行"改为"包住模块导入"，降级时返回空列表而非 mock 数据（让上层 decide）。

### 2.5 P0-3：moderate_diagnosis TODO 替换

**moderate_diagnosis.py:303 `_query_similar_syndromes()`**：

```python
async def _query_similar_syndromes(
    symptoms: list[str], top_k: int = 5
) -> list[dict]:
    """
    从 Neo4j 查询与症状最相似的证型

    Cypher:
        MATCH (s:Symptom)-[r:INDICATES]->(sy:Syndrome)
        WHERE s.name IN $symptoms
        RETURN sy.name AS syndrome,
               count(r) AS match_count,
               collect(s.name) AS matched_symptoms
        ORDER BY match_count DESC
        LIMIT $top_k
    """
    graph = get_neo4j_graph(database="tcm_graph")
    if graph is None:
        logger.warning("Neo4j 不可用，跳过证型查询")
        return []

    try:
        result = graph.query(
            """
            MATCH (s:Symptom)-[r:INDICATES]->(sy:Syndrome)
            WHERE s.name IN $symptoms
            RETURN sy.name AS syndrome,
                   count(r) AS match_count,
                   collect(s.name) AS matched_symptoms
            ORDER BY match_count DESC
            LIMIT $top_k
            """,
            params={"symptoms": symptoms, "top_k": top_k},
        )
        return [dict(record) for record in result]
    except Exception as e:
        logger.error(f"证型查询失败: {e}")
        return []
```

**moderate_diagnosis.py:345 `_query_related_prescriptions()`**：

```python
async def _query_related_prescriptions(
    syndrome: str, top_k: int = 5
) -> list[dict]:
    """
    从 Neo4j 查询治疗某证型的方剂

    Cypher:
        MATCH (p:Prescription)-[r:TREATS]->(sy:Syndrome {name: $syndrome})
        RETURN p.name AS prescription,
               p.composition AS composition,
               p.usage AS usage
        LIMIT $top_k
    """
    graph = get_neo4j_graph(database="tcm_graph")
    if graph is None:
        logger.warning("Neo4j 不可用，跳过方剂查询")
        return []

    try:
        result = graph.query(
            """
            MATCH (p:Prescription)-[:TREATS]->(sy:Syndrome {name: $syndrome})
            RETURN p.name AS prescription,
                   p.composition AS composition,
                   p.usage AS usage
            LIMIT $top_k
            """,
            params={"syndrome": syndrome, "top_k": top_k},
        )
        return [dict(record) for record in result]
    except Exception as e:
        logger.error(f"方剂查询失败: {e}")
        return []
```

**关键决策**：
- P0 阶段**不校验 Neo4j 是否有数据**，查询不到就返回空数组
- 上层 LLM 看到空数据应能自然降级到"仅基于症状推理"
- 把 try/except 兜底在 Cypher 执行层，不让单次查询失败影响整个诊断流程

### 2.6 P0-4：单元测试

```python
# backend/tests/core/test_graph_db.py
import pytest
from unittest.mock import patch, MagicMock
from app.src.core.graph_db import get_neo4j_graph

@pytest.fixture(autouse=True)
def reset_cache():
    get_neo4j_graph.cache_clear()
    yield
    get_neo4j_graph.cache_clear()

def test_returns_none_when_config_missing(monkeypatch):
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USER", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    assert get_neo4j_graph() is None

def test_returns_graph_when_config_present(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test")

    with patch("app.src.core.graph_db.Neo4jGraph") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        result = get_neo4j_graph()
        assert result is mock_instance
        mock_cls.assert_called_once()

def test_returns_none_on_connection_failure(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://invalid:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test")

    with patch("app.src.core.graph_db.Neo4jGraph", side_effect=Exception("Connection failed")):
        assert get_neo4j_graph() is None
```

### 2.7 P0-5：验证清单

- [ ] `python -c "from app.src.core.graph_db import get_neo4j_graph; print(get_neo4j_graph())"` 在无配置时输出 `None`
- [ ] `pytest backend/tests/core/test_graph_db.py` 全通过
- [ ] 启动 backend 服务，跑一次简单问诊（日志中 `analyze_follow_up` 走到 `assess_complexity` → moderate_diagnosis）
- [ ] moderate_diagnosis 日志中看到 `Neo4j 不可用，跳过证型查询` 警告（这是预期，因为 P0 阶段 Neo4j 没数据）
- [ ] 整个流程不因 `ModuleNotFoundError` 崩溃

---

## 3. P1 — 灌入真实 TCM 数据

### 3.1 目标

让 Neo4j 中**真的有可查询的 TCM 图数据**，中等复杂度辨证能真正利用图谱增强。

### 3.2 数据规模（最小可用）

| 实体 | 数量 | 来源 |
|------|------|------|
| 证型（Syndrome） | ~50 | 《中医内科学》核心证型 |
| 症状（Symptom） | ~200 | 覆盖 50 证型的典型症状 |
| 方剂（Prescription） | ~100 | 《方剂学》经典方剂 |
| 药材（Herb） | ~200 | 方剂中涉及的核心药材 |
| 关系 INDICATES（症状→证型） | ~500 | 教材中"该证型可见...症状" |
| 关系 TREATS（方剂→证型） | ~120 | 教材中"该方主治..." |
| 关系 CONTAINS（方剂→药材） | ~400 | 教材中"该方由...组成" |
| 关系 INCOMPATIBLE_WITH（药材→药材） | ~30 | "十八反""十九畏" |

### 3.3 交付物

| 序号 | 交付物 | 路径 |
|------|--------|------|
| P1-1 | 证型 CSV | `backend/data/tcm/syndromes.csv` |
| P1-2 | 症状 CSV | `backend/data/tcm/symptoms.csv` |
| P1-3 | 方剂 CSV | `backend/data/tcm/prescriptions.csv` |
| P1-4 | 药材 CSV | `backend/data/tcm/herbs.csv` |
| P1-5 | 关系 CSV（4 个） | `backend/data/tcm/relationships/*.csv` |
| P1-6 | 导入脚本 | `backend/scripts/import_tcm_knowledge.py` |
| P1-7 | 验证查询脚本 | `backend/scripts/verify_tcm_data.py` |
| P1-8 | 更新 origin_data/（删除 Northwind 残留） | `backend/graphrag/origin_data/` |
| P1-9 | 数据来源说明文档 | `backend/data/tcm/README.md` |

### 3.4 CSV Schema 设计

**syndromes.csv**：
```csv
id,name,category,description,treatment
S001,风寒感冒,表证,风寒袭表，肺气失宣所表现的证候,辛温解表，宣肺散寒
S002,风热感冒,表证,风热袭表，肺失清肃所表现的证候,辛凉解表，宣肺清热
```

**symptoms.csv**：
```csv
id,name,body_part,nature
SY001,恶寒,全身,寒
SY002,发热,全身,热
SY003,咽痛,咽喉,热
```

**prescriptions.csv**：
```csv
id,name,source,composition,usage,contraindication
P001,桂枝汤,伤寒论,桂枝9g 芍药9g 甘草6g 生姜9g 大枣3枚,水煎服，温覆微汗,外感热病禁服
P002,桑菊饮,温病条辨,桑叶7.5g 菊花3g 杏仁6g 连翘5g...,水煎服,风寒感冒禁服
```

**herbs.csv**：
```csv
id,name,nature,taste,meridian,effect,contraindication
H001,桂枝,温,辛甘,心肺膀胱,发汗解肌 温通经脉,阴虚火旺禁服
H002,芍药,微寒,苦酸,肝脾,养血敛阴 柔肝止痛,阳衰虚寒慎用
```

**relationships/symptom_indicates_syndrome.csv**：
```csv
symptom_id,syndrome_id,weight
SY001,S001,0.9
SY002,S002,0.8
```

**relationships/prescription_treats_syndrome.csv**：
```csv
prescription_id,syndrome_id
P001,S001
P002,S002
```

**relationships/prescription_contains_herb.csv**：
```csv
prescription_id,herb_id,dosage
P001,H001,9g
P001,H002,9g
```

**relationships/herb_incompatible_herb.csv**：
```csv
herb1_id,herb2_id,reason
H003,H004,十八反：甘草反甘遂
```

### 3.5 P1-6：导入脚本

```python
# backend/scripts/import_tcm_knowledge.py
"""
TCM 知识库导入脚本

将 backend/data/tcm/ 下的 CSV 导入 Neo4j
执行：python -m scripts.import_tcm_knowledge
"""
import os
import csv
from pathlib import Path
from app.src.core.graph_db import get_neo4j_graph
from app.src.utils import get_logger

logger = get_logger("import_tcm")
DATA_DIR = Path(__file__).parent.parent / "data" / "tcm"

def import_nodes(graph, csv_path: Path, label: str, id_field: str = "id"):
    """导入节点"""
    if not csv_path.exists():
        logger.warning(f"文件不存在: {csv_path}")
        return 0

    count = 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 构造 Cypher
            props = {k: v for k, v in row.items() if v}
            props[id_field] = props.pop(id_field)  # 确保 id 字段存在

            cypher = f"""
            MERGE (n:{label} {{{id_field}: ${id_field}}})
            SET n += $props
            """
            graph.query(cypher, params={id_field: props[id_field], "props": props})
            count += 1
    logger.info(f"导入 {label} 节点 {count} 条")
    return count

def import_relationship(
    graph, csv_path: Path, rel_type: str,
    from_label: str, from_id: str,
    to_label: str, to_id: str,
    extra_fields: list[str] = None,
):
    """导入关系"""
    if not csv_path.exists():
        logger.warning(f"文件不存在: {csv_path}")
        return 0

    extra_fields = extra_fields or []
    count = 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            params = {
                "from_id": row[from_id],
                "to_id": row[to_id],
            }
            extra_props = {f: row[f] for f in extra_fields if row.get(f)}

            cypher = f"""
            MATCH (a:{from_label} {{{from_id}: $from_id}})
            MATCH (b:{to_label} {{{to_id}: $to_id}})
            MERGE (a)-[r:{rel_type}]->(b)
            """
            if extra_props:
                set_clauses = ", ".join([f"r.{k} = ${k}" for k in extra_props.keys()])
                cypher += f"SET {set_clauses}"
                params.update(extra_props)

            graph.query(cypher, params=params)
            count += 1
    logger.info(f"导入关系 {rel_type} {count} 条")
    return count

def main():
    graph = get_neo4j_graph(database="tcm_graph")
    if graph is None:
        logger.error("Neo4j 不可用，请检查环境变量")
        return

    # 清空（开发环境用，生产环境慎用）
    # graph.query("MATCH (n) DETACH DELETE n")

    # 导入节点
    import_nodes(graph, DATA_DIR / "syndromes.csv", "Syndrome")
    import_nodes(graph, DATA_DIR / "symptoms.csv", "Symptom")
    import_nodes(graph, DATA_DIR / "prescriptions.csv", "Prescription")
    import_nodes(graph, DATA_DIR / "herbs.csv", "Herb")

    # 导入关系
    rel_dir = DATA_DIR / "relationships"
    import_relationship(
        graph, rel_dir / "symptom_indicates_syndrome.csv",
        "INDICATES", "Symptom", "symptom_id",
        "Syndrome", "syndrome_id",
        extra_fields=["weight"],
    )
    import_relationship(
        graph, rel_dir / "prescription_treats_syndrome.csv",
        "TREATS", "Prescription", "prescription_id",
        "Syndrome", "syndrome_id",
    )
    import_relationship(
        graph, rel_dir / "prescription_contains_herb.csv",
        "CONTAINS", "Prescription", "prescription_id",
        "Herb", "herb_id",
        extra_fields=["dosage"],
    )
    import_relationship(
        graph, rel_dir / "herb_incompatible_herb.csv",
        "INCOMPATIBLE_WITH", "Herb", "herb1_id",
        "Herb", "herb2_id",
        extra_fields=["reason"],
    )

    logger.info("TCM 知识库导入完成")

if __name__ == "__main__":
    main()
```

### 3.6 P1-7：验证脚本

```python
# backend/scripts/verify_tcm_data.py
"""
验证 Neo4j 中 TCM 数据完整性

执行：python -m scripts.verify_tcm_data
"""
from app.src.core.graph_db import get_neo4j_graph
from app.src.utils import get_logger

logger = get_logger("verify_tcm")

def main():
    graph = get_neo4j_graph(database="tcm_graph")
    if graph is None:
        logger.error("Neo4j 不可用")
        return

    checks = [
        ("证型数量", "MATCH (n:Syndrome) RETURN count(n) AS c", 50),
        ("症状数量", "MATCH (n:Symptom) RETURN count(n) AS c", 200),
        ("方剂数量", "MATCH (n:Prescription) RETURN count(n) AS c", 100),
        ("药材数量", "MATCH (n:Herb) RETURN count(n) AS c", 200),
        ("INDICATES 关系", "MATCH ()-[r:INDICATES]->() RETURN count(r) AS c", 500),
        ("TREATS 关系", "MATCH ()-[r:TREATS]->() RETURN count(r) AS c", 120),
        ("CONTAINS 关系", "MATCH ()-[r:CONTAINS]->() RETURN count(r) AS c", 400),
        ("INCOMPATIBLE_WITH 关系", "MATCH ()-[r:INCOMPATIBLE_WITH]->() RETURN count(r) AS c", 30),
    ]

    for name, cypher, expected in checks:
        result = graph.query(cypher)
        actual = result[0]["c"] if result else 0
        status = "✓" if actual >= expected * 0.8 else "✗"
        logger.info(f"{status} {name}: {actual} (期望 ≥ {expected})")

    # 实际查询测试
    logger.info("=" * 50)
    logger.info("实际查询测试：")

    # 测试 1：根据症状查证型
    result = graph.query("""
        MATCH (s:Symptom)-[r:INDICATES]->(sy:Syndrome)
        WHERE s.name IN ['恶寒', '发热', '咽痛']
        RETURN sy.name AS syndrome, count(r) AS match_count
        ORDER BY match_count DESC
        LIMIT 5
    """)
    logger.info(f"症状[恶寒、发热、咽痛] → 证型: {result}")

    # 测试 2：根据证型查方剂
    result = graph.query("""
        MATCH (p:Prescription)-[:TREATS]->(sy:Syndrome {name: '风寒感冒'})
        RETURN p.name AS prescription
    """)
    logger.info(f"证型[风寒感冒] → 方剂: {result}")

if __name__ == "__main__":
    main()
```

### 3.7 P1 实施步骤

1. **数据采集**（1 天）：从《中医内科学》《方剂学》提取数据，整理为 CSV
   - 可以用 LLM 辅助提取（GPT/Claude 对教材做结构化），人工校验关键证型
2. **CSV 编写**（0.5 天）：按 schema 写 8 个 CSV 文件
3. **导入脚本开发**（0.5 天）：完成 `import_tcm_knowledge.py`
4. **导入 + 验证**（0.5 天）：运行导入 + 验证脚本，确认数据完整
5. **集成测试**（0.5 天）：跑一次中等复杂度问诊，验证 moderate_diagnosis 真的能查到图谱数据

---

## 4. P2 — 病例库结构化

### 4.1 目标

把 LangGraph state 中的病例信息**定期落库到 PostgreSQL**，为后续"群体分析""用户健康档案""个性化养生"打基础。

### 4.2 交付物

| 序号 | 交付物 | 路径 |
|------|--------|------|
| P2-1 | 数据库 migration（cases / case_symptoms / case_syndromes / case_prescriptions） | `backend/app/src/model/migrations/versions/xxxx_add_case_tables.py` |
| P2-2 | SQLAlchemy 模型 | `backend/app/src/model/case_models.py` |
| P2-3 | 病例落库 service | `backend/app/src/service/case_service.py` |
| P2-4 | 与 LangGraph checkpointer 集成（事件钩子） | `backend/app/src/agent/tcm_builder.py` |
| P2-5 | 跨会话健康档案查询 API | `backend/app/src/controller/case_controller.py` |
| P2-6 | 前端健康档案展示页 | `frontend/src/views/Public/components/HealthProfile.tsx` |

### 4.3 P2-1：数据库 Schema

```sql
-- cases 表
CREATE TABLE cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    conversation_id UUID NOT NULL,
    thread_id UUID,
    chief_complaint TEXT NOT NULL,
    complexity_level VARCHAR(20),  -- simple / moderate / complex
    syndrome_id VARCHAR(50) REFERENCES tcm_syndromes(id),  -- 辨证结果
    syndrome_name VARCHAR(100),
    diagnosis_text TEXT,  -- 完整辨证文本
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_cases_user_id ON cases(user_id);
CREATE INDEX idx_cases_created_at ON cases(created_at DESC);

-- case_symptoms 表（一个病例对应多个症状）
CREATE TABLE case_symptoms (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    symptom_name VARCHAR(100) NOT NULL,
    body_part VARCHAR(50),
    nature VARCHAR(20),  -- 寒/热/虚/实
    severity SMALLINT  -- 1-5
);

CREATE INDEX idx_case_symptoms_case_id ON case_symptoms(case_id);

-- case_syndromes 表（一个病例可能辨为多个证型）
CREATE TABLE case_syndromes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    syndrome_name VARCHAR(100) NOT NULL,
    confidence DECIMAL(3,2),  -- 0.00-1.00
    is_primary BOOLEAN DEFAULT FALSE
);

-- case_prescriptions 表（推荐方剂）
CREATE TABLE case_prescriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    prescription_name VARCHAR(100) NOT NULL,
    composition TEXT,
    usage TEXT,
    source VARCHAR(200)
);

-- 用户健康档案（聚合）
CREATE TABLE user_health_profiles (
    user_id UUID PRIMARY KEY REFERENCES users(id),
    constitution VARCHAR(50),  -- 气虚/阳虚/阴虚/痰湿...
    chronic_conditions TEXT[],
    allergies TEXT[],
    last_case_at TIMESTAMP,
    total_cases INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### 4.4 P2-3：case_service 核心逻辑

```python
# backend/app/src/service/case_service.py
"""
病例结构化服务
"""
from uuid import UUID
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from app.src.model.case_models import Case, CaseSymptom, CaseSyndrome
from app.src.utils import get_logger

logger = get_logger("case_service")

class CaseService:
    """病例服务"""

    async def save_case_from_state(
        self,
        db: AsyncSession,
        user_id: UUID,
        conversation_id: UUID,
        thread_id: str,
        state: dict,
    ) -> Case:
        """
        从 LangGraph state 提取病例信息并落库

        Args:
            db: 数据库会话
            user_id: 用户 ID
            conversation_id: 会话 ID
            thread_id: LangGraph 线程 ID
            state: LangGraph 状态字典，包含 collected_info, syndrome 等

        Returns:
            Case: 保存的病例对象
        """
        # 1. 提取主诉
        chief_complaint = self._extract_chief_complaint(state)

        # 2. 提取症状列表
        symptoms = self._extract_symptoms(state)

        # 3. 提取辨证结果
        syndrome = self._extract_syndrome(state)

        # 4. 提取复杂度
        complexity = state.get("complexity_level", "simple")

        # 5. 创建 case
        case = Case(
            user_id=user_id,
            conversation_id=conversation_id,
            thread_id=thread_id,
            chief_complaint=chief_complaint,
            complexity_level=complexity,
            syndrome_id=syndrome.get("id"),
            syndrome_name=syndrome.get("name"),
            diagnosis_text=state.get("answer", ""),
        )
        db.add(case)
        await db.flush()  # 获取 case.id

        # 6. 批量插入症状
        for sym in symptoms:
            db.add(CaseSymptom(
                case_id=case.id,
                symptom_name=sym["name"],
                body_part=sym.get("body_part"),
                nature=sym.get("nature"),
                severity=sym.get("severity"),
            ))

        await db.commit()
        await db.refresh(case)
        logger.info(f"病例已落库: case_id={case.id}, user_id={user_id}")
        return case

    def _extract_chief_complaint(self, state: dict) -> str:
        """提取主诉"""
        collected = state.get("collected_info", {})
        if isinstance(collected, dict):
            return collected.get("chief_complaint", "未明确")
        return "未明确"

    def _extract_symptoms(self, state: dict) -> list[dict]:
        """提取症状列表"""
        collected = state.get("collected_info", {})
        symptoms = []

        # 从 collected_info 提取
        if isinstance(collected, dict):
            for key in ["head_body", "cold_heat", "sweat", "urine_stool",
                       "diet", "chest_abdomen", "sleep", "emotion"]:
                value = collected.get(key)
                if value and isinstance(value, str):
                    # 简单按逗号分隔
                    parts = [p.strip() for p in value.replace("、", ",").split(",") if p.strip()]
                    for p in parts:
                        symptoms.append({"name": p, "body_part": key})

        return symptoms

    def _extract_syndrome(self, state: dict) -> dict:
        """提取辨证结果"""
        return {
            "id": state.get("syndrome_id"),
            "name": state.get("syndrome_name"),
        }
```

### 4.5 P2-4：与 LangGraph 集成

在 `tcm_builder.py` 的 diagnose subgraph 完成后，添加 case_service 调用：

```python
# 在 diagnose subgraph 的最终节点
async def save_case_node(state: DiagnoseOverallState) -> Dict[str, Any]:
    """诊断完成后落库病例"""
    try:
        user_id = state.get("user_id")  # 需要从 input 传递
        conversation_id = state.get("conversation_id")
        thread_id = state.get("thread_id")

        if not all([user_id, conversation_id]):
            logger.warning("缺少 user_id 或 conversation_id，跳过病例落库")
            return {}

        async with get_db() as db:
            case_service = CaseService()
            await case_service.save_case_from_state(
                db, user_id, conversation_id, thread_id, state
            )
    except Exception as e:
        logger.error(f"病例落库失败: {e}", exc_info=True)

    return {}  # 不影响后续流程
```

### 4.6 P2 实施步骤

1. **Schema 设计 + migration**（1 天）：完成 P2-1, P2-2
2. **case_service 开发**（1-2 天）：完成 P2-3，包含充分的单元测试
3. **LangGraph 集成**（1 天）：完成 P2-4
4. **API + 前端**（2 天）：完成 P2-5, P2-6
5. **集成测试**（1 天）：跑一次完整流程，验证病例真的落库

---

## 5. 风险与约束

### 5.1 P0 风险
- **Neo4j 服务依赖**：如果用户环境没有 Neo4j，`get_neo4j_graph()` 返回 `None`，moderate/complex 走空数据路径。**降级要优雅**。
- **测试环境**：CI 流水线没有 Neo4j，单元测试要 mock 掉 `Neo4jGraph`。

### 5.2 P1 风险
- **数据质量**：教材数据可能存在版本差异、学术争议。**优先选权威教材**（《中医内科学》第九版、《方剂学》第十版）。
- **导入幂等性**：脚本要支持重复执行（用 `MERGE` 而非 `CREATE`）。
- **数据规模权衡**：50/200/100/200 是最小可用规模，不是上限。后续可以扩到 500+ 证型。

### 5.3 P2 风险
- **写库性能**：每次问诊都写库，QPS 高时需要异步队列。**P2 阶段先同步写，后续优化**。
- **数据隐私**：医疗数据合规要求高，**P2 阶段必须确认 PII 处理策略**（脱敏？加密？）。

---

## 6. 验证标准

### P0 完成标准
- [ ] `graph_db.py` 模块存在且可导入
- [ ] 现有测试套件不破坏
- [ ] 手动跑一次问诊，不因 `ModuleNotFoundError` 崩溃
- [ ] 单元测试覆盖率 ≥ 80%

### P1 完成标准
- [ ] Neo4j 中实际有 ≥ 50 证型 / 200 症状 / 100 方剂 / 200 药材
- [ ] 验证脚本输出全 ✓
- [ ] 实际查询能命中真实数据
- [ ] 一次中等复杂度问诊的日志中能看到 Neo4j 查询结果

### P2 完成标准
- [ ] 数据库表创建成功
- [ ] 一次问诊后 `cases` 表有对应记录
- [ ] 前端能看到用户的历次问诊记录
- [ ] 跨会话健康档案能查询到

---

## 7. 文档维护

- 本文档为**蓝图文档**，新会话开始时先读它
- 实施过程中如有调整，**先改本文档再改代码**
- 完成后归档到 `docs/postmortem/YYYY-MM-DD-neo4j-implementation.md`

---

**最后更新**：2026-06-05
**当前阶段**：P0 准备中
**下次会话切入点**：从 P0-1（实现 `graph_db.py`）开始
