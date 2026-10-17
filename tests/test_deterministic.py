from evalharness.metrics.deterministic import (
    compute_deterministic_metrics,
    score_schema_validity,
    score_tool_calls,
)
from evalharness.models import (
    AgentRunOutput,
    TaskCase,
    TaskCategory,
    ToolCallExpectation,
    ToolCallRecord,
)


def test_score_tool_calls_perfect_match():
    task = TaskCase(
        task_id="TC-001",
        description="Test",
        input={},
        expected_tool_calls=[
            ToolCallExpectation(tool_name="tool_a"),
            ToolCallExpectation(tool_name="tool_b"),
        ],
        expected_outcome="Success",
        category=TaskCategory.happy_path,
    )
    output = AgentRunOutput(
        task_id="TC-001",
        final_response="Done",
        tool_calls=[
            ToolCallRecord(tool_name="tool_a"),
            ToolCallRecord(tool_name="tool_b"),
        ],
    )
    accuracy, missing, unexpected = score_tool_calls(task, output)
    assert accuracy == 1.0
    assert missing == []
    assert unexpected == []


def test_score_tool_calls_missing_tool():
    task = TaskCase(
        task_id="TC-001",
        description="Test",
        input={},
        expected_tool_calls=[
            ToolCallExpectation(tool_name="tool_a"),
            ToolCallExpectation(tool_name="tool_b"),
        ],
        expected_outcome="Success",
        category=TaskCategory.happy_path,
    )
    output = AgentRunOutput(
        task_id="TC-001",
        final_response="Done",
        tool_calls=[ToolCallRecord(tool_name="tool_a")],
    )
    accuracy, missing, unexpected = score_tool_calls(task, output)
    assert accuracy == 0.5
    assert missing == ["tool_b"]
    assert unexpected == []


def test_score_tool_calls_unexpected_tool():
    task = TaskCase(
        task_id="TC-001",
        description="Test",
        input={},
        expected_tool_calls=[ToolCallExpectation(tool_name="tool_a")],
        expected_outcome="Success",
        category=TaskCategory.happy_path,
    )
    output = AgentRunOutput(
        task_id="TC-001",
        final_response="Done",
        tool_calls=[
            ToolCallRecord(tool_name="tool_a"),
            ToolCallRecord(tool_name="tool_c"),
        ],
    )
    accuracy, missing, unexpected = score_tool_calls(task, output)
    assert accuracy == 1.0
    assert missing == []
    assert unexpected == ["tool_c"]


def test_score_schema_validity():
    output = AgentRunOutput(
        task_id="TC-001",
        final_response="Done",
        total_tokens_in=100,
        total_tokens_out=50,
        total_duration_ms=1000,
    )
    assert score_schema_validity(output) is True


def test_compute_deterministic_metrics():
    task = TaskCase(
        task_id="TC-001",
        description="Test",
        input={},
        expected_tool_calls=[ToolCallExpectation(tool_name="tool_a")],
        expected_outcome="Success",
        category=TaskCategory.happy_path,
    )
    output = AgentRunOutput(
        task_id="TC-001",
        final_response="Done",
        tool_calls=[ToolCallRecord(tool_name="tool_a")],
        total_tokens_in=1000,
        total_tokens_out=500,
        total_duration_ms=2000,
    )
    metrics = compute_deterministic_metrics(task, output, "gpt-4o")
    assert metrics.tool_call_accuracy == 1.0
    assert metrics.missing_tool_calls == []
    assert metrics.unexpected_tool_calls == []
    assert metrics.schema_valid is True
    assert metrics.cost_usd > 0
    assert metrics.latency_ms == 2000
