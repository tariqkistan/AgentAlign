import pytest
from pytest_mock import MockerFixture

from evalharness.judge import judge_task
from evalharness.models import (
    AgentRunOutput,
    JudgedMetrics,
    TaskCase,
    TaskCategory,
)


@pytest.mark.asyncio
async def test_judge_task_valid_parse(mocker: MockerFixture):
    mock_result = JudgedMetrics(
        task_success=True,
        task_success_confidence=0.95,
        hallucination_detected=False,
        hallucination_detail="",
        judge_rationale="Task completed successfully",
    )
    mock_judge = mocker.AsyncMock()
    mock_judge.ainvoke = mocker.AsyncMock(return_value=mock_result)
    mocker.patch("evalharness.judge.structured_judge", mock_judge)

    task = TaskCase(
        task_id="TC-001",
        description="Test task",
        input={"key": "value"},
        expected_outcome="Success",
        category=TaskCategory.happy_path,
    )
    output = AgentRunOutput(
        task_id="TC-001",
        final_response="Task completed",
    )

    result = await judge_task(task, output)
    assert result.task_success is True
    assert result.task_success_confidence == 0.95
    assert result.hallucination_detected is False


@pytest.mark.asyncio
async def test_judge_task_invalid_json_raises(mocker: MockerFixture):
    mock_judge = mocker.AsyncMock()
    mock_judge.ainvoke = mocker.AsyncMock(side_effect=Exception("Invalid JSON"))
    mocker.patch("evalharness.judge.structured_judge", mock_judge)

    task = TaskCase(
        task_id="TC-001",
        description="Test task",
        input={"key": "value"},
        expected_outcome="Success",
        category=TaskCategory.happy_path,
    )
    output = AgentRunOutput(
        task_id="TC-001",
        final_response="Task completed",
    )

    with pytest.raises(ValueError, match="Judge failed for task TC-001"):
        await judge_task(task, output)
