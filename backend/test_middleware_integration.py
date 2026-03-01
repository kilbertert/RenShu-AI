"""
中间件集成测试脚本

逐步测试每个中间件的集成，找出冲突的中间件组合
"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.src.agent.tcm_builder import get_llm
from app.src.agent.components.diagnose.config import diagnose_config
from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# 导入所有中间件
from langchain.agents.middleware import (
    TodoListMiddleware,
    SummarizationMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ModelFallbackMiddleware,
    ToolRetryMiddleware,
    ModelRetryMiddleware,
)
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

from app.src.agent.components.diagnose.nodes.complex.subagents import (
    create_differential_expert,
)
from app.src.agent.components.diagnose.prompts.deepsearch_prompts import (
    DIFFERENTIAL_DIAGNOSIS_EXPERT_PROMPT,
)

print("=" * 60)
print("中间件集成测试")
print("=" * 60)


def test_middleware_combination(name: str, middleware_list: list):
    """测试特定的中间件组合"""
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"中间件数量: {len(middleware_list)}")
    print(f"{'='*60}")
    
    try:
        llm = get_llm(temperature=diagnose_config.DIAGNOSIS_TEMPERATURE)
        
        agent = create_deep_agent(
            model=llm,
            checkpointer=MemorySaver(),
            store=InMemoryStore(),
            tools=[],
            middleware=middleware_list,
            system_prompt="测试 Agent"
        )
        
        print(f"✅ {name} - 创建成功")
        return True
    except Exception as e:
        print(f"❌ {name} - 失败: {e}")
        return False


# ============================================================
# 测试 1: 不使用中间件（基线测试）
# ============================================================

result_baseline = test_middleware_combination(
    "基线测试（无中间件）",
    []
)

# ============================================================
# 测试 2: 单独测试每个中间件
# ============================================================

print("\n" + "=" * 60)
print("阶段 1: 单独测试每个中间件")
print("=" * 60)

llm = get_llm(temperature=diagnose_config.DIAGNOSIS_TEMPERATURE)
differential_expert = create_differential_expert(llm)

results = {}

# 2.1 ModelRetryMiddleware
results['ModelRetryMiddleware'] = test_middleware_combination(
    "ModelRetryMiddleware",
    [ModelRetryMiddleware(max_retries=2, backoff_factor=1.5)]
)

# 2.2 ToolRetryMiddleware
results['ToolRetryMiddleware'] = test_middleware_combination(
    "ToolRetryMiddleware",
    [ToolRetryMiddleware(max_retries=3, backoff_factor=2.0)]
)

# 2.3 ModelCallLimitMiddleware
results['ModelCallLimitMiddleware'] = test_middleware_combination(
    "ModelCallLimitMiddleware",
    [ModelCallLimitMiddleware(thread_limit=30, run_limit=15, exit_behavior="end")]
)

# 2.4 ToolCallLimitMiddleware
results['ToolCallLimitMiddleware'] = test_middleware_combination(
    "ToolCallLimitMiddleware",
    [ToolCallLimitMiddleware(tool_name="test_tool", thread_limit=10, exit_behavior="end")]
)

# 2.5 TodoListMiddleware
results['TodoListMiddleware'] = test_middleware_combination(
    "TodoListMiddleware",
    [TodoListMiddleware(system_prompt="测试任务列表")]
)

# 2.6 FilesystemMiddleware
results['FilesystemMiddleware'] = test_middleware_combination(
    "FilesystemMiddleware",
    [FilesystemMiddleware(
        backend=lambda rt: CompositeBackend(
            default=StateBackend(rt),
            routes={"/test/": StateBackend(rt)}
        ),
        tool_token_limit_before_evict=15000,
        system_prompt="测试文件系统"
    )]
)

# 2.7 SubAgentMiddleware
results['SubAgentMiddleware'] = test_middleware_combination(
    "SubAgentMiddleware",
    [SubAgentMiddleware(
        default_model=llm,
        subagents=[{
            "name": "test_expert",
            "description": "测试专家",
            "agent": differential_expert,
            "system_prompt": DIFFERENTIAL_DIAGNOSIS_EXPERT_PROMPT
        }]
    )]
)

# 2.8 SummarizationMiddleware
results['SummarizationMiddleware'] = test_middleware_combination(
    "SummarizationMiddleware",
    [SummarizationMiddleware(
        model=get_llm(model="deepseek-chat"),
        trigger=[("tokens", 4000), ("messages", 15)],
        summary_prompt="测试总结"
    )]
)

# ============================================================
# 测试 3: 两两组合测试（找出冲突）
# ============================================================

print("\n" + "=" * 60)
print("阶段 2: 两两组合测试")
print("=" * 60)

successful_singles = [k for k, v in results.items() if v]
print(f"\n单独测试成功的中间件: {', '.join(successful_singles)}")

combinations_results = {}

middleware_instances = {
    'ModelRetryMiddleware': ModelRetryMiddleware(max_retries=2, backoff_factor=1.5),
    'ToolRetryMiddleware': ToolRetryMiddleware(max_retries=3, backoff_factor=2.0),
    'ModelCallLimitMiddleware': ModelCallLimitMiddleware(thread_limit=30, run_limit=15, exit_behavior="end"),
    'ToolCallLimitMiddleware': ToolCallLimitMiddleware(tool_name="test_tool", thread_limit=10, exit_behavior="end"),
    'TodoListMiddleware': TodoListMiddleware(system_prompt="测试任务列表"),
    'FilesystemMiddleware': FilesystemMiddleware(
        backend=lambda rt: CompositeBackend(
            default=StateBackend(rt),
            routes={"/test/": StateBackend(rt)}
        ),
        tool_token_limit_before_evict=15000,
        system_prompt="测试文件系统"
    ),
    'SubAgentMiddleware': SubAgentMiddleware(
        default_model=llm,
        subagents=[{
            "name": "test_expert",
            "description": "测试专家",
            "agent": differential_expert,
            "system_prompt": DIFFERENTIAL_DIAGNOSIS_EXPERT_PROMPT
        }]
    ),
}

# 测试所有两两组合
for i, name1 in enumerate(successful_singles):
    for name2 in successful_singles[i+1:]:
        combo_name = f"{name1} + {name2}"
        
        # 重新创建中间件实例（避免重用）
        m1 = middleware_instances[name1]
        m2 = middleware_instances[name2]
        
        combinations_results[combo_name] = test_middleware_combination(
            combo_name,
            [m1, m2]
        )

# ============================================================
# 测试 4: 逐步添加中间件（找出第一个失败的组合）
# ============================================================

print("\n" + "=" * 60)
print("阶段 3: 逐步添加中间件")
print("=" * 60)

# 按原始顺序逐步添加
middleware_order = [
    ('ModelRetryMiddleware', ModelRetryMiddleware(max_retries=2, backoff_factor=1.5)),
    ('ToolRetryMiddleware', ToolRetryMiddleware(max_retries=3, backoff_factor=2.0)),
    ('TodoListMiddleware', TodoListMiddleware(system_prompt="测试")),
    ('FilesystemMiddleware', FilesystemMiddleware(
        backend=lambda rt: CompositeBackend(default=StateBackend(rt), routes={}),
        tool_token_limit_before_evict=15000,
        system_prompt="测试"
    )),
    ('SubAgentMiddleware', SubAgentMiddleware(
        default_model=llm,
        subagents=[{
            "name": "test_expert",
            "description": "测试专家",
            "agent": differential_expert,
            "system_prompt": DIFFERENTIAL_DIAGNOSIS_EXPERT_PROMPT
        }]
    )),
]

progressive_results = {}
current_stack = []

for name, middleware in middleware_order:
    current_stack.append(middleware)
    stack_names = " -> ".join([m[0] for m in middleware_order[:len(current_stack)]])
    
    progressive_results[stack_names] = test_middleware_combination(
        f"渐进测试 ({len(current_stack)}个): {stack_names}",
        current_stack.copy()
    )
    
    if not progressive_results[stack_names]:
        print(f"\n⚠️ 发现问题！添加 {name} 后失败")
        break

# ============================================================
# 测试结果汇总
# ============================================================

print("\n" + "=" * 60)
print("测试结果汇总")
print("=" * 60)

print("\n📊 单独测试结果:")
for name, success in results.items():
    status = "✅" if success else "❌"
    print(f"  {status} {name}")

print("\n📊 两两组合测试结果:")
failed_combinations = []
for combo, success in combinations_results.items():
    status = "✅" if success else "❌"
    print(f"  {status} {combo}")
    if not success:
        failed_combinations.append(combo)

print("\n📊 渐进测试结果:")
for stack, success in progressive_results.items():
    status = "✅" if success else "❌"
    print(f"  {status} {stack}")

if failed_combinations:
    print("\n" + "=" * 60)
    print("⚠️ 发现冲突的中间件组合:")
    print("=" * 60)
    for combo in failed_combinations:
        print(f"  ❌ {combo}")
else:
    print("\n✅ 所有两两组合测试通过！")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
