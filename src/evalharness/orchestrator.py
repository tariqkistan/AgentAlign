import uuid
from datetime import datetime

from evalharness.adapters.base import AgentAdapter
from evalharness.judge import judge_task
from evalharness.logger import logger
from evalharness.metrics.deterministic import compute_deterministic_metrics
from evalharness.models import (
    RunSummary,
    TaskCase,
    TaskEvalResult,
)


async def run_suite(
    adapter: AgentAdapter, task_cases: list[TaskCase], suite_id: str, model: str = "gpt-4o"
) -> tuple[list[TaskEvalResult], RunSummary]:
    run_id = str(uuid.uuid4())
    started_at = datetime.utcnow()
    results: list[TaskEvalResult] = []

    logger.info(
        "Starting evaluation run",
        run_id=run_id,
        agent_name=adapter.agent_name,
        suite_id=suite_id,
        task_count=len(task_cases),
    )

    for task in task_cases:
        logger.info("Running task", task_id=task.task_id, category=task.category)
        output = await adapter.run_task(task)
        deterministic = compute_deterministic_metrics(task, output, model)
        judged = await judge_task(task, output)

        result = TaskEvalResult(
            task_id=task.task_id,
            run_id=run_id,
            agent_name=adapter.agent_name,
            deterministic=deterministic,
            judged=judged,
        )
        results.append(result)

        logger.info(
            "Task completed",
            task_id=task.task_id,
            success=judged.task_success,
            tool_accuracy=deterministic.tool_call_accuracy,
            cost_usd=deterministic.cost_usd,
        )

    finished_at = datetime.utcnow()
    summary = _compute_summary(
        results, run_id, adapter.agent_name, suite_id, started_at, finished_at
    )

    logger.info(
        "Evaluation run completed",
        run_id=run_id,
        avg_success_rate=summary.avg_task_success_rate,
        avg_tool_accuracy=summary.avg_tool_call_accuracy,
        hallucination_rate=summary.hallucination_rate,
        total_cost_usd=summary.total_cost_usd,
    )

    return results, summary


def _compute_summary(
    results: list[TaskEvalResult],
    run_id: str,
    agent_name: str,
    suite_id: str,
    started_at: datetime,
    finished_at: datetime,
) -> RunSummary:
    if not results:
        return RunSummary(
            run_id=run_id,
            agent_name=agent_name,
            suite_id=suite_id,
            task_count=0,
            avg_task_success_rate=0.0,
            avg_tool_call_accuracy=0.0,
            hallucination_rate=0.0,
            total_cost_usd=0.0,
            avg_latency_ms=0.0,
            started_at=started_at,
            finished_at=finished_at,
        )

    total_success = sum(1 for r in results if r.judged.task_success)
    total_tool_accuracy = sum(r.deterministic.tool_call_accuracy for r in results)
    total_hallucinations = sum(1 for r in results if r.judged.hallucination_detected)
    total_cost = sum(r.deterministic.cost_usd for r in results)
    total_latency = sum(r.deterministic.latency_ms for r in results)

    return RunSummary(
        run_id=run_id,
        agent_name=agent_name,
        suite_id=suite_id,
        task_count=len(results),
        avg_task_success_rate=total_success / len(results),
        avg_tool_call_accuracy=total_tool_accuracy / len(results),
        hallucination_rate=total_hallucinations / len(results),
        total_cost_usd=total_cost,
        avg_latency_ms=total_latency / len(results),
        started_at=started_at,
        finished_at=finished_at,
    )
