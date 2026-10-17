import pytest
from pytest_mock import MockerFixture

from evalharness.models import AgentRunOutput, JudgedMetrics, TaskCase, TaskCategory
from evalharness.orchestrator import run_suite


class MockAdapter:
    agent_name = "mock-agent"

    async def run_task(self, task: TaskCase) -> AgentRunOutput:
        return AgentRunOutput(
            task_id=task.task_id,
            final_response="Mock response",
            total_tokens_in=100,
            total_tokens_out=50,
            total_duration_ms=1000,
        )


@pytest.mark.asyncio
async def test_run_suite_aggregates_correctly(mocker: MockerFixture):
    mock_adapter = MockAdapter()
    mock_judge = mocker.patch(
        "evalharness.orchestrator.judge_task",
        return_value=JudgedMetrics(
            task_success=True,
            task_success_confidence=0.9,
            hallucination_detected=False,
        ),
    )

    task_cases = [
        TaskCase(
            task_id="TC-001",
            description="Test 1",
            input={},
            expected_outcome="Success",
            category=TaskCategory.happy_path,
        ),
        TaskCase(
            task_id="TC-002",
            description="Test 2",
            input={},
            expected_outcome="Success",
            category=TaskCategory.happy_path,
        ),
    ]

    results, summary = await run_suite(mock_adapter, task_cases, "test-suite")

    assert len(results) == 2
    assert summary.task_count == 2
    assert summary.agent_name == "mock-agent"
    assert summary.suite_id == "test-suite"
    assert summary.avg_task_success_rate == 1.0
    assert summary.avg_tool_call_accuracy == 1.0
    assert summary.hallucination_rate == 0.0
    assert summary.total_cost_usd > 0
    assert summary.avg_latency_ms == 1000
    assert mock_judge.call_count == 2


@pytest.mark.asyncio
async def test_run_suite_empty_task_list():
    mock_adapter = MockAdapter()
    results, summary = await run_suite(mock_adapter, [], "test-suite")

    assert len(results) == 0
    assert summary.task_count == 0
    assert summary.avg_task_success_rate == 0.0
    assert summary.avg_tool_call_accuracy == 0.0
