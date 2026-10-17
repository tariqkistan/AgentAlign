from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TaskCategory(StrEnum):
    happy_path = "happy_path"
    edge_case = "edge_case"
    adversarial = "adversarial"
    hallucination_probe = "hallucination_probe"


class ToolCallExpectation(BaseModel):
    tool_name: str
    required_args: dict[str, Any] = Field(default_factory=dict)
    optional: dict[str, Any] = Field(default_factory=dict)


class TaskCase(BaseModel):
    task_id: str
    description: str
    input: dict[str, Any]
    expected_tool_calls: list[ToolCallExpectation] = Field(default_factory=list)
    expected_outcome: str
    category: TaskCategory


class ToolCallRecord(BaseModel):
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    duration_ms: int = 0


class AgentRunOutput(BaseModel):
    task_id: str
    final_response: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_duration_ms: int = 0
    raw_trace: dict[str, Any] = Field(default_factory=dict)


class DeterministicMetrics(BaseModel):
    tool_call_accuracy: float = Field(ge=0.0, le=1.0)
    missing_tool_calls: list[str] = Field(default_factory=list)
    unexpected_tool_calls: list[str] = Field(default_factory=list)
    schema_valid: bool = True
    cost_usd: float = Field(ge=0.0)
    latency_ms: int = Field(ge=0)


class JudgedMetrics(BaseModel):
    task_success: bool
    task_success_confidence: float = Field(ge=0.0, le=1.0)
    hallucination_detected: bool
    hallucination_detail: str = ""
    judge_rationale: str = ""


class TaskEvalResult(BaseModel):
    task_id: str
    run_id: str
    agent_name: str
    deterministic: DeterministicMetrics
    judged: JudgedMetrics
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


class HumanLabel(BaseModel):
    task_id: str
    human_verdict: bool
    human_rationale: str


class CalibrationResult(BaseModel):
    agreement_rate: float = Field(ge=0.0, le=1.0)
    total_compared: int = Field(ge=0)
    disagreements: list[str] = Field(default_factory=list)
    cohen_kappa: float = Field(ge=-1.0, le=1.0)


class RunSummary(BaseModel):
    run_id: str
    agent_name: str
    suite_id: str
    task_count: int = Field(ge=0)
    avg_task_success_rate: float = Field(ge=0.0, le=1.0)
    avg_tool_call_accuracy: float = Field(ge=0.0, le=1.0)
    hallucination_rate: float = Field(ge=0.0, le=1.0)
    total_cost_usd: float = Field(ge=0.0)
    avg_latency_ms: float = Field(ge=0.0)
    started_at: datetime
    finished_at: datetime
