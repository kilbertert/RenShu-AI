#!/usr/bin/env python
"""
TCM Data Import Script
中医数据导入脚本

用法:
    python import_tcm_data.py --init-schema                    # 初始化Schema
    python import_tcm_data.py --type classics --file data.json # 导入古籍
    python import_tcm_data.py --type cases --file data.json    # 导入医案
    python import_tcm_data.py --info                           # 查看Schema信息
"""

import argparse
import asyncio
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def print_banner():
    """打印横幅"""
    print("=" * 60)
    print("  SmartTCM 数据导入工具")
    print("  TCM Knowledge Graph Data Importer")
    print("=" * 60)
    print()


async def init_schema(vector_dimension: int = 1024):
    """初始化Neo4j Schema"""
    from app.src.agent.data.schema_initializer import init_schema as do_init_schema

    print("正在初始化Neo4j Schema...")
    print(f"向量维度: {vector_dimension}")
    print()

    result = do_init_schema(vector_dimension)

    print("约束创建结果:")
    for msg in result["constraints"]:
        print(f"  {msg}")

    print("\n全文索引创建结果:")
    for msg in result["fulltext_indexes"]:
        print(f"  {msg}")

    print("\n向量索引创建结果:")
    for msg in result["vector_indexes"]:
        print(f"  {msg}")

    print(f"\n{result['message']}")
    return result["success"]


async def import_classics(file_path: str):
    """导入古籍数据"""
    from app.src.agent.data.classic_ingestor import ClassicIngestor

    print(f"正在导入古籍数据: {file_path}")
    print()

    ingestor = ClassicIngestor()
    result = await ingestor.ingest_from_json(file_path)

    print(f"总记录数: {result.total_records}")
    print(f"成功导入: {result.imported_count}")
    print(f"失败数量: {result.failed_count}")

    if result.errors:
        print("\n错误信息:")
        for err in result.errors[:10]:  # 只显示前10条错误
            print(f"  - {err}")

    print(f"\n{result.message}")
    return result.success


async def import_cases(file_path: str):
    """导入医案数据"""
    from app.src.agent.data.case_ingestor import CaseIngestor

    print(f"正在导入医案数据: {file_path}")
    print()

    ingestor = CaseIngestor()
    result = await ingestor.ingest_from_json(file_path)

    print(f"总记录数: {result.total_records}")
    print(f"成功导入: {result.imported_count}")
    print(f"失败数量: {result.failed_count}")

    if result.errors:
        print("\n错误信息:")
        for err in result.errors[:10]:
            print(f"  - {err}")

    print(f"\n{result.message}")
    return result.success


async def show_schema_info():
    """显示Schema信息"""
    from app.src.agent.data.schema_initializer import get_schema_info

    print("当前Neo4j Schema信息:")
    print()

    info = get_schema_info()

    if "error" in info:
        print(f"获取Schema信息失败: {info['error']}")
        return False

    print("约束:")
    for c in info["constraints"]:
        print(f"  - {c}")

    print("\n索引:")
    for idx in info["indexes"]:
        print(f"  - {idx['name']} ({idx['type']})")

    print("\n节点标签:")
    for label in info["node_labels"]:
        print(f"  - {label}")

    print("\n关系类型:")
    for rel in info["relationship_types"]:
        print(f"  - {rel}")

    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="SmartTCM 数据导入工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --init-schema                              初始化Schema
  %(prog)s --init-schema --dimension 1536             使用1536维向量初始化
  %(prog)s --type classics --file classics.json      导入古籍数据
  %(prog)s --type cases --file cases.json            导入医案数据
  %(prog)s --info                                     查看Schema信息
        """
    )

    parser.add_argument(
        "--init-schema",
        action="store_true",
        help="初始化Neo4j Schema（约束和索引）"
    )

    parser.add_argument(
        "--dimension",
        type=int,
        default=1024,
        help="向量维度，默认1024（DashScope text-embedding-v3）"
    )

    parser.add_argument(
        "--type",
        choices=["classics", "cases"],
        help="导入数据类型"
    )

    parser.add_argument(
        "--file",
        type=str,
        help="JSON数据文件路径"
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="显示当前Schema信息"
    )

    args = parser.parse_args()

    print_banner()

    # 检查环境变量
    neo4j_uri = os.getenv("NEO4J_URI")
    if not neo4j_uri:
        print("警告: 未设置 NEO4J_URI 环境变量，将使用默认值 bolt://localhost:7687")

    # 执行操作
    if args.init_schema:
        success = asyncio.run(init_schema(args.dimension))
        sys.exit(0 if success else 1)

    elif args.info:
        success = asyncio.run(show_schema_info())
        sys.exit(0 if success else 1)

    elif args.type and args.file:
        if not os.path.exists(args.file):
            print(f"错误: 文件不存在 - {args.file}")
            sys.exit(1)

        if args.type == "classics":
            success = asyncio.run(import_classics(args.file))
        else:
            success = asyncio.run(import_cases(args.file))

        sys.exit(0 if success else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
