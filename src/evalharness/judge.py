import os

from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

from evalharness.logger import logger
from evalharness.models import AgentRunOutput, JudgedMetrics, TaskCase

JUDGE_DEPLOYMENT = os.getenv("JUDGE_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

structured_judge = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=SecretStr(AZURE_OPENAI_KEY) if AZURE_OPENAI_KEY else None,
    api_version=AZURE_OPENAI_API_VERSION,
    model=JUDGE_DEPLOYMENT,
    temperature=0.0,
).with_structured_output(JudgedMetrics)

JUDGE_SYSTEM_PROMPT = """You are a strict and impartial evaluator of AI agent performance.

Your task is to assess whether the agent successfully completed the given task
and whether it hallucinated (made unsupported claims).

Task Success: The agent succeeded if it achieved the expected outcome described
in the task. The final response should directly address the task requirements.

Hallucination Detection: The agent hallucinated if it made any claims not
supported by the provided input, tool results, or context. You must quote the
specific unsupported claims in the hallucination_detail field.

Be precise and evidence-based. If the agent fails or hallucinates, explain why
clearly in the rationale."""


def _build_judge_messages(
    task: TaskCase, output: AgentRunOutput
) -> list[tuple[str, str]]:
    return [
        ("system", JUDGE_SYSTEM_PROMPT),
        (
            "human",
            f"Task ID: {task.task_id}\n"
            f"Task Description: {task.description}\n"
            f"Expected Outcome: {task.expected_outcome}\n\n"
            f"Agent Input: {task.input}\n\n"
            f"Agent Final Response: {output.final_response}\n\n"
            f"Tool Calls Made: {len(output.tool_calls)}\n"
            f"Raw Trace: {output.raw_trace}",
        ),
    ]


async def judge_task(task: TaskCase, output: AgentRunOutput) -> JudgedMetrics:
    messages = _build_judge_messages(task, output)
    logger.info("Invoking judge", task_id=task.task_id)
    try:
        result = await structured_judge.ainvoke(messages)
        assert isinstance(result, JudgedMetrics)
        logger.info("Judge completed", task_id=task.task_id, success=result.task_success)
        return result
    except Exception as e:
        logger.error("Judge invocation failed", task_id=task.task_id, error=str(e))
        raise ValueError(f"Judge failed for task {task.task_id}: {e}") from e
