"""
DeepSearch Agent 测试脚本

测试内容：
1. Agent 创建测试
2. 中间件配置测试
3. 子 Agent 创建测试
4. 工具调用测试
5. 完整工作流测试（mock）

运行方式：
    cd backend
    python -m pytest app/tests/test_deep_search_agent.py -v
    python -m pytest app/tests/test_deep_search_agent.py::TestDeepSearchAgent::test_import -v

"""

import pytest
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any

# 确保可以导入 app 模块
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))


class TestDeepSearchAgent:
    """DeepSearch Agent 基础测试类"""

    def test_import_module(self):
        """测试模块导入"""
        try:
            from app.src.agent.components.diagnose.nodes.complex.deep_search_agent import (
                create_deep_search_agent,
                run_deep_search_diagnosis,
                DEEP_SEARCH_SYSTEM_PROMPT,
            )

            assert create_deep_search_agent is not None
            assert run_deep_search_diagnosis is not None
            assert isinstance(DEEP_SEARCH_SYSTEM_PROMPT, str)
            assert len(DEEP_SEARCH_SYSTEM_PROMPT) > 0
        except ImportError as e:
            pytest.fail(f"导入失败: {e}")


class TestSystemPrompt:
    """系统提示词测试类"""

    def test_system_prompt_content(self):
        """测试系统提示词内容"""
        from app.src.agent.components.diagnose.nodes.complex.deep_search_agent import (
            DEEP_SEARCH_SYSTEM_PROMPT,
        )

        assert isinstance(DEEP_SEARCH_SYSTEM_PROMPT, str)
        assert len(DEEP_SEARCH_SYSTEM_PROMPT) > 0

        # 验证关键内容存在
        assert "中医智能诊断助手" in DEEP_SEARCH_SYSTEM_PROMPT
        assert "DeepSearch Agent" in DEEP_SEARCH_SYSTEM_PROMPT
        assert "并行数据查询" in DEEP_SEARCH_SYSTEM_PROMPT
        assert "并行专家咨询" in DEEP_SEARCH_SYSTEM_PROMPT
        assert "综合决策" in DEEP_SEARCH_SYSTEM_PROMPT


class TestAgentCreation:
    """Agent 创建测试类"""

    @pytest.fixture
    def mock_llm(self):
        """模拟 LLM"""
        llm = Mock()
        llm.temperature = 0.7
        return llm

    @pytest.fixture
    def mock_checkpointer(self):
        """模拟 Checkpointer"""
        return Mock()

    @pytest.fixture
    def mock_store(self):
        """模拟 Store"""
        return Mock()

    @patch("app.src.agent.components.diagnose.nodes.complex.deep_search_agent.get_llm")
    @patch(
        "app.src.agent.components.diagnose.nodes.complex.deep_search_agent.create_deep_agent"
    )
    @patch(
        "app.src.agent.components.diagnose.nodes.complex.deep_search_agent.create_differential_expert"
    )
    @patch(
        "app.src.agent.components.diagnose.nodes.complex.deep_search_agent.create_treatment_expert"
    )
    @patch(
        "app.src.agent.components.diagnose.nodes.complex.deep_search_agent.create_prescription_expert"
    )
    @patch(
        "app.src.agent.components.diagnose.nodes.complex.deep_search_agent.create_prognosis_expert"
    )
    @patch(
        "app.src.agent.components.diagnose.nodes.complex.deep_search_agent.create_verification_expert"
    )
    def test_create_agent_with_defaults(
        self,
        mock_create_verification,
        mock_create_prognosis,
        mock_create_prescription,
        mock_create_treatment,
        mock_create_differential,
        mock_create_deep_agent,
        mock_get_llm,
        mock_llm,
    ):
        """测试使用默认参数创建 Agent"""
        # 设置 mock
        mock_get_llm.return_value = mock_llm
        mock_create_deep_agent.return_value = Mock()

        # 模拟子 Agent
        mock_subagent = Mock()
        mock_create_differential.return_value = mock_subagent
        mock_create_treatment.return_value = mock_subagent
        mock_create_prescription.return_value = mock_subagent
        mock_create_prognosis.return_value = mock_subagent
        mock_create_verification.return_value = mock_subagent

        # 执行
        from app.src.agent.components.diagnose.nodes.complex.deep_search_agent import (
            create_deep_search_agent,
        )

        agent = create_deep_search_agent()

        # 验证
        assert agent is not None
        mock_get_llm.assert_called_once()
        mock_create_deep_agent.assert_called_once()

        # 验证所有子 Agent 都被创建
        mock_create_differential.assert_called_once()
        mock_create_treatment.assert_called_once()
        mock_create_prescription.assert_called_once()
        mock_create_prognosis.assert_called_once()
        mock_create_verification.assert_called_once()

    @patch(
        "app.src.agent.components.diagnose.nodes.complex.deep_search_agent.create_deep_agent"
    )
    def test_create_agent_with_custom_params(
        self, mock_create_deep_agent, mock_llm, mock_checkpointer, mock_store
    ):
        """测试使用自定义参数创建 Agent"""
        mock_create_deep_agent.return_value = Mock()

        from app.src.agent.components.diagnose.nodes.complex.deep_search_agent import (
            create_deep_search_agent,
        )

        agent = create_deep_search_agent(
            llm=mock_llm,
            enable_doctor_approval=False,
            checkpointer=mock_checkpointer,
            store=mock_store,
        )

        assert agent is not None

        # 验证调用参数
        call_kwargs = mock_create_deep_agent.call_args[1]
        assert call_kwargs["model"] == mock_llm
        assert call_kwargs["checkpointer"] == mock_checkpointer
        assert call_kwargs["store"] == mock_store


class TestMiddleware:
    """中间件配置测试类"""

    @pytest.fixture
    def mock_llm(self):
        """模拟 LLM"""
        return Mock()

    @patch(
        "app.src.agent.components.diagnose.nodes.complex.deep_search_agent.create_deep_agent"
    )
    def test_middleware_stack_configuration(self, mock_create_deep_agent, mock_llm):
        """测试中间件栈配置"""
        mock_create_deep_agent.return_value = Mock()

        from app.src.agent.components.diagnose.nodes.complex.deep_search_agent import (
            create_deep_search_agent,
        )

        create_deep_search_agent(llm=mock_llm)

        call_kwargs = mock_create_deep_agent.call_args[1]
        middleware_stack = call_kwargs.get("middleware", [])

        # 验证中间件栈不为空
        assert len(middleware_stack) > 0

        # 验证关键中间件存在
        middleware_types = [type(m).__name__ for m in middleware_stack]
        assert any("ModelRetry" in t for t in middleware_types), (
            "缺少 ModelRetryMiddleware"
        )
        assert any("ToolRetry" in t for t in middleware_types), (
            "缺少 ToolRetryMiddleware"
        )
        assert any("TodoList" in t for t in middleware_types), "缺少 TodoListMiddleware"
        assert any("Filesystem" in t for t in middleware_types), (
            "缺少 FilesystemMiddleware"
        )
        assert any("SubAgent" in t for t in middleware_types), "缺少 SubAgentMiddleware"
        assert any("Summarization" in t for t in middleware_types), (
            "缺少 SummarizationMiddleware"
        )

    @patch(
        "app.src.agent.components.diagnose.nodes.complex.deep_search_agent.create_deep_agent"
    )
    def test_tools_configuration(self, mock_create_deep_agent, mock_llm):
        """测试工具配置"""
        mock_create_deep_agent.return_value = Mock()

        from app.src.agent.components.diagnose.nodes.complex.deep_search_agent import (
            create_deep_search_agent,
        )

        create_deep_search_agent(llm=mock_llm)

        call_kwargs = mock_create_deep_agent.call_args[1]
        tools = call_kwargs.get("tools", [])

        # 验证工具有 4 个
        assert len(tools) == 4

        # 验证工具名称
        tool_names = [t.__name__ if hasattr(t, "__name__") else str(t) for t in tools]
        assert any("kg_syndrome_search" in name for name in tool_names)
        assert any("case_vector_search" in name for name in tool_names)
        assert any("classics_search" in name for name in tool_names)
        assert any("web_search" in name for name in tool_names)

    @patch(
        "app.src.agent.components.diagnose.nodes.complex.deep_search_agent.create_deep_agent"
    )
    def test_subagent_configuration(self, mock_create_deep_agent, mock_llm):
        """测试子 Agent 配置"""
        mock_create_deep_agent.return_value = Mock()

        from app.src.agent.components.diagnose.nodes.complex.deep_search_agent import (
            create_deep_search_agent,
        )

        create_deep_search_agent(llm=mock_llm)

        call_kwargs = mock_create_deep_agent.call_args[1]
        middleware_stack = call_kwargs.get("middleware", [])

        # 查找 SubAgentMiddleware
        subagent_middleware = None
        for m in middleware_stack:
            if "SubAgent" in type(m).__name__:
                subagent_middleware = m
                break

        assert subagent_middleware is not None, "SubAgentMiddleware 未找到"

        # 验证子 Agent 列表
        assert hasattr(subagent_middleware, "subagents")
        subagents = subagent_middleware.subagents
        assert len(subagents) == 5, f"应该有 5 个子 Agent，实际有 {len(subagents)}"


class TestAsyncOperations:
    """异步操作测试类"""

    @pytest.fixture
    def mock_llm(self):
        return Mock()

    @pytest.fixture
    def sample_patient_info(self) -> Dict[str, Any]:
        """样本患者信息"""
        return {
            "chief_complaint": "头痛",
            "symptoms": {
                "main": ["头痛", "眩晕"],
                "secondary": ["失眠", "心悸"],
                "duration": "2周",
            },
            "tongue": "淡红，苔薄白",
            "pulse": "弦细",
            "medical_history": ["高血压病史3年"],
            "age": 55,
            "gender": "男",
        }

    @pytest.mark.anyio
    @patch(
        "app.src.agent.components.diagnose.nodes.complex.deep_search_agent.create_deep_agent"
    )
    @patch("app.src.agent.components.diagnose.nodes.complex.deep_search_agent.get_llm")
    @patch(
        "app.src.agent.components.diagnose.nodes.complex.deep_search_agent.create_differential_expert"
    )
    @patch(
        "app.src.agent.components.diagnose.nodes.complex.deep_search_agent.create_treatment_expert"
    )
    @patch(
        "app.src.agent.components.diagnose.nodes.complex.deep_search_agent.create_prescription_expert"
    )
    @patch(
        "app.src.agent.components.diagnose.nodes.complex.deep_search_agent.create_prognosis_expert"
    )
    @patch(
        "app.src.agent.components.diagnose.nodes.complex.deep_search_agent.create_verification_expert"
    )
    async def test_run_deep_search_diagnosis(
        self,
        mock_create_verification,
        mock_create_prognosis,
        mock_create_prescription,
        mock_create_treatment,
        mock_create_differential,
        mock_get_llm,
        mock_create_deep_agent,
        mock_llm,
        sample_patient_info,
    ):
        """测试深度辨证分析执行"""
        # 设置 mock
        mock_get_llm.return_value = mock_llm
        mock_subagent = Mock()
        mock_create_differential.return_value = mock_subagent
        mock_create_treatment.return_value = mock_subagent
        mock_create_prescription.return_value = mock_subagent
        mock_create_prognosis.return_value = mock_subagent
        mock_create_verification.return_value = mock_subagent

        # 模拟 Agent 返回结果
        mock_agent = Mock()
        mock_agent.ainvoke = AsyncMock(
            return_value={
                "messages": [Mock(content="辨证结果：肝阳上亢证")],
                "steps": ["数据收集", "专家咨询", "综合决策"],
            }
        )
        mock_create_deep_agent.return_value = mock_agent

        # 执行
        from app.src.agent.components.diagnose.nodes.complex.deep_search_agent import (
            run_deep_search_diagnosis,
        )

        result = await run_deep_search_diagnosis(sample_patient_info)

        # 验证
        assert result is not None
        assert "answer" in result
        assert "steps" in result
        assert "thread_id" in result
        assert result["answer"] == "辨证结果：肝阳上亢证"
        assert len(result["steps"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
