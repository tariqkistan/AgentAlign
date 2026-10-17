from evalharness.models import (
    AgentRunOutput,
    DeterministicMetrics,
    TaskCase,
)
from evalharness.pricing import estimate_cost


def score_tool_calls(
    task: TaskCase, output: AgentRunOutput
) -> tuple[float, list[str], list[str]]:
    expected_names = {tc.tool_name for tc in task.expected_tool_calls}
    actual_names = {tc.tool_name for tc in output.tool_calls}
    correct = expected_names & actual_names
    missing = expected_names - actual_names
    unexpected = actual_names - expected_names
    accuracy = len(correct) / len(expected_names) if expected_names else 1.0
    return accuracy, sorted(missing), sorted(unexpected)


def score_schema_validity(output: AgentRunOutput) -> bool:
    try:
        output.model_validate(output.model_dump())
        return True
    except Exception:
        return False


def compute_deterministic_metrics(
    task: TaskCase, output: AgentRunOutput, model: str = "gpt-4o"
) -> DeterministicMetrics:
    accuracy, missing, unexpected = score_tool_calls(task, output)
    schema_valid = score_schema_validity(output)
    cost = estimate_cost(output.total_tokens_in, output.total_tokens_out, model)
    return DeterministicMetrics(
        tool_call_accuracy=accuracy,
        missing_tool_calls=missing,
        unexpected_tool_calls=unexpected,
        schema_valid=schema_valid,
        cost_usd=cost,
        latency_ms=output.total_duration_ms,
    )
