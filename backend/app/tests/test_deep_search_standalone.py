"""
DeepSearch Agent 独立测试脚本

这个脚本独立运行，不需要依赖完整的项目导入链。
使用方法：
    cd backend
    python app/tests/test_deep_search_standalone.py
"""

import ast
import sys
from pathlib import Path


def test_file_structure():
    """测试文件基本结构"""
    file_path = (
        Path(__file__).parent.parent
        / "app/src/agent/components/diagnose/nodes/complex/deep_search_agent.py"
    )

    assert file_path.exists(), f"文件不存在: {file_path}"

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 解析 AST
    tree = ast.parse(content)

    # 检查关键函数
    functions = [
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ]
    assert "create_deep_search_agent" in functions, "缺少 create_deep_search_agent 函数"
    assert "run_deep_search_diagnosis" in functions, (
        "缺少 run_deep_search_diagnosis 函数"
    )

    # 检查常量
    constants = [
        node.targets[0].id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    ]
    assert "DEEP_SEARCH_SYSTEM_PROMPT" in constants, (
        "缺少 DEEP_SEARCH_SYSTEM_PROMPT 常量"
    )

    print("✓ 文件结构检查通过")
    print(f"  - 发现函数: {', '.join(functions)}")
    return True


def test_imports():
    """测试关键导入"""
    file_path = (
        Path(__file__).parent.parent
        / "app/src/agent/components/diagnose/nodes/complex/deep_search_agent.py"
    )

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 检查关键导入
    key_imports = [
        "from typing import Dict, Any, Optional, List",
        "from langchain_core.language_models import BaseChatModel",
        "from langgraph.checkpoint.memory import MemorySaver",
        "from langgraph.store.memory import InMemoryStore",
        "from deepagents import create_deep_agent",
    ]

    for import_stmt in key_imports:
        assert import_stmt in content, f"缺少导入: {import_stmt}"

    print("✓ 关键导入检查通过")
    return True


def test_middleware_imports():
    """测试中间件导入"""
    file_path = (
        Path(__file__).parent.parent
        / "app/src/agent/components/diagnose/nodes/complex/deep_search_agent.py"
    )

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 检查中间件
    middlewares = [
        "ModelRetryMiddleware",
        "ToolRetryMiddleware",
        "TodoListMiddleware",
        "FilesystemMiddleware",
        "SubAgentMiddleware",
        "SummarizationMiddleware",
        "ModelCallLimitMiddleware",
        "ToolCallLimitMiddleware",
    ]

    for middleware in middlewares:
        assert middleware in content, f"缺少中间件: {middleware}"

    print(f"✓ 中间件检查通过 (共 {len(middlewares)} 个)")
    return True


def test_tools_import():
    """测试工具导入"""
    file_path = (
        Path(__file__).parent.parent
        / "app/src/agent/components/diagnose/nodes/complex/deep_search_agent.py"
    )

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    tools = [
        "kg_syndrome_search",
        "case_vector_search",
        "classics_search",
        "web_search",
    ]

    for tool in tools:
        assert tool in content, f"缺少工具: {tool}"

    print(f"✓ 工具检查通过 (共 {len(tools)} 个)")
    return True


def test_subagents_import():
    """测试子 Agent 导入"""
    file_path = (
        Path(__file__).parent.parent
        / "app/src/agent/components/diagnose/nodes/complex/deep_search_agent.py"
    )

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    subagents = [
        "create_differential_expert",
        "create_treatment_expert",
        "create_prescription_expert",
        "create_prognosis_expert",
        "create_verification_expert",
    ]

    for subagent in subagents:
        assert subagent in content, f"缺少子 Agent: {subagent}"

    print(f"✓ 子 Agent 检查通过 (共 {len(subagents)} 个)")
    return True


def test_system_prompt_content():
    """测试系统提示词内容"""
    file_path = (
        Path(__file__).parent.parent
        / "app/src/agent/components/diagnose/nodes/complex/deep_search_agent.py"
    )

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 检查关键内容
    key_contents = [
        "中医智能诊断助手",
        "DeepSearch Agent",
        "并行数据查询",
        "并行专家咨询",
        "综合决策",
    ]

    for key in key_contents:
        assert key in content, f"系统提示词缺少: {key}"

    print(f"✓ 系统提示词内容检查通过")
    return True


def test_configuration_values():
    """测试配置值"""
    file_path = (
        Path(__file__).parent.parent
        / "app/src/agent/components/diagnose/nodes/complex/deep_search_agent.py"
    )

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 检查关键配置
    configs = [
        "max_retries=2",  # ModelRetryMiddleware
        "max_retries=3",  # ToolRetryMiddleware
        "tool_token_limit_before_evict=15000",  # FilesystemMiddleware
        "thread_limit=30",  # ModelCallLimitMiddleware
        "thread_limit=3",  # web_search limit
        "thread_limit=5",  # kg_syndrome_search limit
    ]

    for config in configs:
        assert config in content, f"缺少配置: {config}"

    print(f"✓ 配置值检查通过 (共 {len(configs)} 项)")
    return True


def test_docstrings():
    """测试文档字符串"""
    file_path = (
        Path(__file__).parent.parent
        / "app/src/agent/components/diagnose/nodes/complex/deep_search_agent.py"
    )

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 检查函数文档
    assert '"""' in content, "文件缺少文档字符串"
    assert "创建 DeepSearch Agent" in content or "创建 DeepSearch Agent" in content, (
        "缺少主函数说明"
    )

    print("✓ 文档字符串检查通过")
    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("DeepSearch Agent 独立测试")
    print("=" * 60)

    tests = [
        ("文件结构", test_file_structure),
        ("关键导入", test_imports),
        ("中间件", test_middleware_imports),
        ("工具", test_tools_import),
        ("子 Agent", test_subagents_import),
        ("系统提示词", test_system_prompt_content),
        ("配置值", test_configuration_values),
        ("文档字符串", test_docstrings),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {name} 测试失败: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {name} 测试出错: {e}")
            failed += 1

    print("=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
