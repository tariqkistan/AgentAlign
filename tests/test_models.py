import pytest

from evalharness.models import (
    AgentRunOutput,
    CalibrationResult,
    DeterministicMetrics,
    HumanLabel,
    JudgedMetrics,
    RunSummary,
    TaskCase,
    TaskCategory,
    ToolCallExpectation,
    ToolCallRecord,
)


def test_tool_call_expectation():
    tce = ToolCallExpectation(tool_name="test_tool", required_args={"arg1": "value"})
    assert tce.tool_name == "test_tool"
    assert tce.required_args == {"arg1": "value"}


def test_task_case():
    tc = TaskCase(
        task_id="TC-001",
        description="Test task",
        input={"key": "value"},
        expected_outcome="Success",
        category=TaskCategory.happy_path,
    )
    assert tc.task_id == "TC-001"
    assert tc.category == TaskCategory.happy_path


def test_tool_call_record():
    tcr = ToolCallRecord(tool_name="test", args={"x": 1}, result="ok", duration_ms=100)
    assert tcr.tool_name == "test"
    assert tcr.duration_ms == 100


def test_agent_run_output():
    output = AgentRunOutput(
        task_id="TC-001",
        final_response="Done",
        total_tokens_in=100,
        total_tokens_out=50,
        total_duration_ms=1000,
    )
    assert output.task_id == "TC-001"
    assert output.tool_calls == []


def test_deterministic_metrics_bounds():
    with pytest.raises(ValueError):
        DeterministicMetrics(
            tool_call_accuracy=1.5,
            cost_usd=0.0,
            latency_ms=0,
        )


def test_judged_metrics_bounds():
    with pytest.raises(ValueError):
        JudgedMetrics(
            task_success=True,
            task_success_confidence=1.5,
            hallucination_detected=False,
        )


def test_calibration_result_bounds():
    cr = CalibrationResult(agreement_rate=0.85, total_compared=10, cohen_kappa=0.8)
    assert 0.0 <= cr.agreement_rate <= 1.0
    assert -1.0 <= cr.cohen_kappa <= 1.0


def test_human_label():
    hl = HumanLabel(task_id="TC-001", human_verdict=True, human_rationale="Good")
    assert hl.task_id == "TC-001"
    assert hl.human_verdict is True


def test_run_summary():
    rs = RunSummary(
        run_id="run-123",
        agent_name="test-agent",
        suite_id="golden-v1",
        task_count=5,
        avg_task_success_rate=0.8,
        avg_tool_call_accuracy=0.9,
        hallucination_rate=0.1,
        total_cost_usd=0.5,
        avg_latency_ms=1000,
        started_at="2024-01-01T00:00:00",
        finished_at="2024-01-01T01:00:00",
    )
    assert rs.run_id == "run-123"
    assert 0.0 <= rs.avg_task_success_rate <= 1.0
