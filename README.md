# AgentAlign

[![CI](https://img.shields.io/github/actions/workflow/status/tariqkistan/AgentAlign/ci.yml?branch=master)](https://github.com/tariqkistan/AgentAlign/actions)

A systematic evaluation framework for AI agents, built on Azure OpenAI and LangChain. AgentAlign measures task success rate, tool-call accuracy, hallucination rate, cost per task, and latency per step.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Orchestrator                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Adapter    │  │ Deterministic│  │    Judge     │          │
│  │  (Agent)     │──│   Metrics    │  │ (LLM-as-Judge)│          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                  │                  │                  │
│         v                  v                  v                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │AgentRunOutput│  │Deterministic │  │JudgedMetrics │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                            │    │
│                         TaskEvalResult                      │    │
└────────────────────────────────────────────────────────────┼────┘
                                                             │
                                                             v
                                                    ┌──────────────┐
                                                    │   Report     │
                                                    │  (JSON+MD)   │
                                                    └──────────────┘

Calibration:
  Judge Results + Human Labels → Cohen's Kappa → Agreement Rate
```

## Quickstart

1. Install dependencies:
```bash
pip install -e ".[dev]"
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

3. Seed the golden dataset:
```bash
python scripts/seed_golden_set.py
```

4. Run an evaluation:
```bash
python scripts/run_eval.py run --agent callscore --suite golden-v1
```

## Test Commands

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_deterministic.py

# Type checking
mypy src

# Linting
ruff check .
```

## Adding a New Agent Adapter

1. Create a new adapter in `src/evalharness/adapters/`:
```python
from evalharness.adapters.base import AgentAdapter
from evalharness.models import AgentRunOutput, TaskCase

class MyAdapter(AgentAdapter):
    agent_name = "my-agent"

    async def run_task(self, task: TaskCase) -> AgentRunOutput:
        # Run your agent on the task
        # Return AgentRunOutput with tool_calls, tokens, etc.
        pass
```

2. Register the adapter in `scripts/run_eval.py`:
```python
if agent == "my-agent":
    adapter = MyAdapter()
```

3. Run evaluation:
```bash
python scripts/run_eval.py run --agent my-agent --suite golden-v1
```

## Status

**Current Calibration Numbers:**
- Agreement Rate: Not yet calibrated (run evaluation with real key)
- Cohen's Kappa: Not yet calibrated (run evaluation with real key)

The judge requires calibration against the human-labeled golden dataset. Calibration should be run manually with a real Azure OpenAI key, not in CI.

## Design Decisions

### Deterministic vs Judged Metrics Split

Deterministic metrics (tool-call accuracy, cost, latency) are computed directly from the agent's output without LLM calls. This makes them fast, reliable, and cheap. Judged metrics (task success, hallucination detection) require an LLM-as-judge because they involve semantic evaluation that cannot be reduced to simple rules. This split allows for rapid iteration on deterministic metrics while using expensive LLM judgment only where necessary.

### Why Calibrate Against Humans

LLM-as-judge systems can have systematic biases or failure modes. Calibrating against a small human-labeled golden dataset using agreement rate and Cohen's kappa provides a quantitative measure of judge reliability. This makes it possible to detect when the judge is misaligned and to compare different judge prompts or models objectively.

### Why Cohen's Kappa

Cohen's kappa measures inter-rater agreement while accounting for chance agreement. Simple agreement rate can be misleading when tasks are easy (high agreement by chance) or when there's class imbalance. Kappa provides a more robust measure of judge quality, especially important when comparing different judge configurations.

### Why the Adapter Pattern

The adapter pattern allows the evaluation harness to be agent-agnostic. Any agent can be evaluated by implementing the `AgentAdapter` interface with a single `run_task` method. This makes it easy to compare different agents, add new ones, and swap implementations without modifying the core evaluation logic.

### Why Store Full Run JSON Not Just Summaries

Storing the full JSON run (including per-task results, raw traces, and all metrics) enables deeper analysis, debugging, and historical comparison. Summaries are useful for quick status checks, but the full data is necessary for:
- Investigating failures and edge cases
- Comparing runs over time to detect regressions
- Replaying specific tasks for debugging
- Computing additional metrics post-hoc without re-running
- Building dashboards and visualizations
