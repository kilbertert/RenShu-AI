"""
Neo4j 知识图谱关系改造脚本
将图谱关系改为符合中医诊断逻辑：

原结构（错误）：
  (Syndrome) -[:Caused]-> (Symptom) -[:treated]-> (Formula)

新结构（正确）：
  (Symptom) -[:INDICATES]-> (Syndrome) -[:TREATS_WITH]-> (Formula)
  症状 → 指示 → 证候 → 治以 → 方剂
"""

from neo4j import GraphDatabase

# Neo4j 连接配置
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "200102242519PyL"


def migrate_schema():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        print("=" * 60)
        print("Neo4j 图谱关系改造")
        print("=" * 60)

        # Step 0: 查看改造前的状态
        print("\n【Step 0】改造前状态")
        result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS count
        """)
        for record in result:
            print(f"  {record['rel_type']}: {record['count']} 条")

        # Step 1: 创建 INDICATES 关系（反转 Caused）
        print("\n【Step 1】创建 INDICATES 关系（症状 → 证候）")
        print("  从 (Syndrome)-[:Caused]->(Symptom) 反转为 (Symptom)-[:INDICATES]->(Syndrome)")

        result = session.run("""
            MATCH (syn:Syndrome)-[r:Caused]->(s:Symptom)
            MERGE (s)-[:INDICATES]->(syn)
            RETURN count(r) AS count
        """)
        count = result.single()["count"]
        print(f"  ✓ 创建了 {count} 条 INDICATES 关系")

        # Step 2: 创建 TREATS_WITH 关系（证候 → 方剂）
        # 逻辑：如果一个证候的症状被某个方剂治疗，则该证候与该方剂建立 TREATS_WITH 关系
        print("\n【Step 2】创建 TREATS_WITH 关系（证候 → 方剂）")
        print("  推导逻辑：证候的症状被方剂治疗 → 证候与方剂建立治疗关系")

        result = session.run("""
            MATCH (syn:Syndrome)-[:Caused]->(s:Symptom)-[:treated]->(f:Formula)
            WITH syn, f, count(s) AS symptom_count
            MERGE (syn)-[r:TREATS_WITH]->(f)
            SET r.symptom_count = symptom_count
            RETURN count(DISTINCT syn) AS syndrome_count,
                   count(DISTINCT f) AS formula_count,
                   count(*) AS relation_count
        """)
        record = result.single()
        print(f"  ✓ 涉及 {record['syndrome_count']} 个证候")
        print(f"  ✓ 涉及 {record['formula_count']} 个方剂")
        print(f"  ✓ 创建了 {record['relation_count']} 条 TREATS_WITH 关系")

        # Step 3: 删除旧关系（可选，先注释掉，确认无误后再删除）
        print("\n【Step 3】删除旧关系")

        # 删除 Caused 关系
        result = session.run("""
            MATCH ()-[r:Caused]->()
            DELETE r
            RETURN count(r) AS count
        """)
        # count = result.single()["count"]
        print(f"  ✓ 删除了 Caused 关系")

        # 删除 treated 关系
        result = session.run("""
            MATCH ()-[r:treated]->()
            DELETE r
            RETURN count(r) AS count
        """)
        # count = result.single()["count"]
        print(f"  ✓ 删除了 treated 关系")

        # Step 4: 验证改造结果
        print("\n【Step 4】改造后状态")
        result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS count
        """)
        for record in result:
            print(f"  {record['rel_type']}: {record['count']} 条")

        # Step 5: 验证新的路径
        print("\n【Step 5】验证新路径：症状 → 证候 → 方剂")
        result = session.run("""
            MATCH (s:Symptom)-[:INDICATES]->(syn:Syndrome)-[:TREATS_WITH]->(f:Formula)
            RETURN s.symptom AS 症状, syn.syndrome AS 证候, f.formula AS 方剂
            LIMIT 5
        """)
        for record in result:
            print(f"  {record['症状']} → {record['证候']} → {record['方剂']}")

        # Step 6: 测试诊断查询
        print("\n【Step 6】测试诊断查询")
        print("  测试症状：便血、神疲乏力、面色萎黄")
        result = session.run("""
            MATCH (s:Symptom)-[:INDICATES]->(syn:Syndrome)
            WHERE s.symptom IN ['便血', '神疲乏力', '面色萎黄']
            WITH syn, COUNT(s) AS match_count, COLLECT(s.symptom) AS matched_symptoms
            ORDER BY match_count DESC
            LIMIT 3
            OPTIONAL MATCH (syn)-[:TREATS_WITH]->(f:Formula)
            RETURN syn.syndrome AS 证候,
                   matched_symptoms AS 匹配症状,
                   match_count AS 匹配数,
                   COLLECT(f.formula) AS 推荐方剂
        """)
        for record in result:
            print(f"\n  证候: {record['证候']}")
            print(f"  匹配症状: {record['匹配症状']}")
            print(f"  匹配数: {record['匹配数']}")
            print(f"  推荐方剂: {record['推荐方剂'][:5]}...")  # 只显示前5个

        print("\n" + "=" * 60)
        print("改造完成！")
        print("=" * 60)
        print("\n新的图谱结构：")
        print("  (Symptom) -[:INDICATES]-> (Syndrome) -[:TREATS_WITH]-> (Formula)")
        print("     症状        指示          证候          治以         方剂")

    driver.close()


if __name__ == "__main__":
    confirm = input("此操作将改造图谱关系结构，是否继续？(y/n): ")
    if confirm.lower() == 'y':
        try:
            migrate_schema()
        except Exception as e:
            print(f"改造失败: {e}")
    else:
        print("已取消")
