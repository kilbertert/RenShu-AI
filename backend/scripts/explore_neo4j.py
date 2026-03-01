"""
Neo4j 知识图谱探索脚本
用于查看图谱结构、节点、关系、示例数据
"""

from neo4j import GraphDatabase

# Neo4j 连接配置
NEO4J_URI = "bolt://localhost:7687"  # bolt 协议端口
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "200102242519PyL"


def explore_graph():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        print("=" * 60)
        print("Neo4j 知识图谱探索")
        print("=" * 60)

        # 1. 查看所有节点标签
        print("\n【1. 节点标签 (Labels)】")
        result = session.run("CALL db.labels()")
        labels = [record["label"] for record in result]
        for label in labels:
            # 统计每种标签的节点数量
            count_result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
            count = count_result.single()["count"]
            print(f"  - {label}: {count} 个节点")

        # 2. 查看所有关系类型
        print("\n【2. 关系类型 (Relationship Types)】")
        result = session.run("CALL db.relationshipTypes()")
        rel_types = [record["relationshipType"] for record in result]
        for rel_type in rel_types:
            # 统计每种关系的数量
            count_result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
            count = count_result.single()["count"]
            print(f"  - {rel_type}: {count} 条关系")

        # 3. 查看节点属性
        print("\n【3. 节点属性 (Properties)】")
        for label in labels:
            result = session.run(f"MATCH (n:{label}) RETURN keys(n) as props LIMIT 1")
            record = result.single()
            if record:
                props = record["props"]
                print(f"  - {label}: {props}")

        # 4. 查看图谱 Schema（节点-关系-节点 模式）
        print("\n【4. 图谱 Schema（关系模式）】")
        result = session.run("""
            CALL db.schema.visualization()
        """)
        # 备用方案：手动查询关系模式
        result = session.run("""
            MATCH (a)-[r]->(b)
            RETURN DISTINCT labels(a)[0] AS from_label,
                   type(r) AS relationship,
                   labels(b)[0] AS to_label,
                   count(*) AS count
            ORDER BY count DESC
        """)
        for record in result:
            print(f"  ({record['from_label']}) -[:{record['relationship']}]-> ({record['to_label']})  [{record['count']}条]")

        # 5. 示例数据
        print("\n【5. 示例数据】")

        for label in labels[:5]:  # 最多展示5种标签
            print(f"\n  --- {label} 示例 (前3条) ---")
            result = session.run(f"MATCH (n:{label}) RETURN n LIMIT 3")
            for i, record in enumerate(result, 1):
                node = record["n"]
                props = dict(node)
                # 只显示前几个属性
                display_props = {k: v for k, v in list(props.items())[:5]}
                print(f"    {i}. {display_props}")

        # 6. 查看症状->证候->方剂 的完整路径示例
        print("\n【6. 症状→证候→方剂 路径示例】")
        result = session.run("""
            MATCH path = (s)-[r1]->(syn)-[r2]->(p)
            WHERE any(label IN labels(s) WHERE label CONTAINS '症' OR label CONTAINS 'Symptom')
              AND any(label IN labels(syn) WHERE label CONTAINS '证' OR label CONTAINS 'Syndrome')
              AND any(label IN labels(p) WHERE label CONTAINS '方' OR label CONTAINS 'Prescription')
            RETURN s, type(r1) as rel1, syn, type(r2) as rel2, p
            LIMIT 5
        """)
        records = list(result)
        if records:
            for record in records:
                s_name = dict(record["s"]).get("name", dict(record["s"]))
                syn_name = dict(record["syn"]).get("name", dict(record["syn"]))
                p_name = dict(record["p"]).get("name", dict(record["p"]))
                print(f"  {s_name} -[:{record['rel1']}]-> {syn_name} -[:{record['rel2']}]-> {p_name}")
        else:
            print("  未找到完整路径，尝试查找任意路径...")
            # 尝试查找任意两跳路径
            result = session.run("""
                MATCH (a)-[r1]->(b)-[r2]->(c)
                RETURN labels(a)[0] as a_label, a, type(r1) as rel1,
                       labels(b)[0] as b_label, b, type(r2) as rel2,
                       labels(c)[0] as c_label, c
                LIMIT 5
            """)
            for record in result:
                a_name = dict(record["a"]).get("name", "?")
                b_name = dict(record["b"]).get("name", "?")
                c_name = dict(record["c"]).get("name", "?")
                print(f"  [{record['a_label']}]{a_name} -[:{record['rel1']}]-> [{record['b_label']}]{b_name} -[:{record['rel2']}]-> [{record['c_label']}]{c_name}")

        print("\n" + "=" * 60)
        print("探索完成")
        print("=" * 60)

    driver.close()


if __name__ == "__main__":
    try:
        explore_graph()
    except Exception as e:
        print(f"连接失败: {e}")
        print("\n请检查:")
        print("1. Neo4j 是否已启动")
        print("2. 端口是否正确 (bolt://localhost:7687)")
        print("3. 用户名密码是否正确")
