import time

from evalharness.adapters.base import AgentAdapter
from evalharness.models import AgentRunOutput, TaskCase, ToolCallRecord


class CallscoreAdapter(AgentAdapter):
    agent_name = "callscore"

    async def run_task(self, task: TaskCase) -> AgentRunOutput:
        try:
            from callscore.scoring import score_transcript
        except ImportError:
            raise ImportError(
                "callscore package not found. Install it to use CallscoreAdapter."
            )

        start_time = time.time()
        transcript = task.input.get("transcript", "")
        score_result = score_transcript(transcript)
        duration_ms = int((time.time() - start_time) * 1000)

        tool_calls = []
        for criterion, score in score_result.get("criteria_scores", {}).items():
            tool_calls.append(
                ToolCallRecord(
                    tool_name="score_criterion",
                    args={"criterion": criterion},
                    result=score,
                    duration_ms=0,
                )
            )

        return AgentRunOutput(
            task_id=task.task_id,
            final_response=score_result.get("summary", ""),
            tool_calls=tool_calls,
            total_tokens_in=score_result.get("tokens_in", 0),
            total_tokens_out=score_result.get("tokens_out", 0),
            total_duration_ms=duration_ms,
            raw_trace=score_result,
        )
