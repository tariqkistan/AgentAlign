import json
from pathlib import Path

from tabulate import tabulate

from evalharness.models import RunSummary, TaskEvalResult


def write_run_json(results: list[TaskEvalResult], summary: RunSummary) -> Path:
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    output_path = runs_dir / f"{summary.run_id}.json"

    data = {
        "summary": summary.model_dump(mode="json"),
        "results": [r.model_dump(mode="json") for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return output_path


def render_markdown_report(results: list[TaskEvalResult], summary: RunSummary) -> str:
    lines = [
        "# AgentAlign Evaluation Report",
        "",
        f"**Run ID:** {summary.run_id}",
        f"**Agent:** {summary.agent_name}",
        f"**Suite:** {summary.suite_id}",
        f"**Started:** {summary.started_at.isoformat()}",
        f"**Finished:** {summary.finished_at.isoformat()}",
        "",
        "## Summary",
        "",
        f"- **Tasks Evaluated:** {summary.task_count}",
        f"- **Avg Task Success Rate:** {summary.avg_task_success_rate:.2%}",
        f"- **Avg Tool Call Accuracy:** {summary.avg_tool_call_accuracy:.2%}",
        f"- **Hallucination Rate:** {summary.hallucination_rate:.2%}",
        f"- **Total Cost:** ${summary.total_cost_usd:.6f}",
        f"- **Avg Latency:** {summary.avg_latency_ms:.0f}ms",
        "",
        "## Per-Task Results",
        "",
    ]

    table_data = []
    for r in results:
        table_data.append(
            [
                r.task_id,
                "✓" if r.judged.task_success else "✗",
                f"{r.judged.task_success_confidence:.2f}",
                "✓" if r.judged.hallucination_detected else "✗",
                f"{r.deterministic.tool_call_accuracy:.2%}",
                f"${r.deterministic.cost_usd:.6f}",
                f"{r.deterministic.latency_ms}ms",
            ]
        )

    table = tabulate(
        table_data,
        headers=[
            "Task ID",
            "Success",
            "Confidence",
            "Hallucination",
            "Tool Accuracy",
            "Cost",
            "Latency",
        ],
        tablefmt="github",
    )
    lines.append(table)

    return "\n".join(lines)


def compare_runs(
    baseline: RunSummary, current: RunSummary, threshold: float = 0.05
) -> list[str]:
    regressions: list[str] = []

    if current.avg_task_success_rate < baseline.avg_task_success_rate - threshold:
        regressions.append(
            f"Task success rate dropped from "
            f"{baseline.avg_task_success_rate:.2%} to "
            f"{current.avg_task_success_rate:.2%}"
        )

    if current.avg_tool_call_accuracy < baseline.avg_tool_call_accuracy - threshold:
        regressions.append(
            f"Tool call accuracy dropped from "
            f"{baseline.avg_tool_call_accuracy:.2%} to "
            f"{current.avg_tool_call_accuracy:.2%}"
        )

    if current.hallucination_rate > baseline.hallucination_rate + threshold:
        regressions.append(
            f"Hallucination rate increased from "
            f"{baseline.hallucination_rate:.2%} to "
            f"{current.hallucination_rate:.2%}"
        )

    return regressions
