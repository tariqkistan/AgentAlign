import json
from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def seed():
    golden_dir = Path("golden")
    golden_dir.mkdir(exist_ok=True)

    golden_v1_path = golden_dir / "golden_v1.json"
    labels_path = golden_dir / "human_labels.json"

    if golden_v1_path.exists() and labels_path.exists():
        typer.echo("Golden dataset files already exist. Skipping creation.")
        raise typer.Exit(0)

    golden_data = {
        "suite_id": "golden-v1",
        "description": "Initial golden dataset for AgentAlign calibration",
        "task_cases": [
            {
                "task_id": "TC-001",
                "description": "Happy path: score a clear transcript with all required information",
                "input": {
                    "transcript": (
                        "Customer called about billing issue. "
                        "Reference number: REF-12345. The customer was polite "
                        "and the issue was resolved."
                    )
                },
                "expected_tool_calls": [
                    {
                        "tool_name": "score_criterion",
                        "required_args": {"criterion": "clarity"},
                        "optional": {}
                    },
                    {
                        "tool_name": "score_criterion",
                        "required_args": {"criterion": "resolution"},
                        "optional": {}
                    }
                ],
                "expected_outcome": (
                    "Agent should score the transcript on clarity and resolution "
                    "criteria, achieving high scores for both."
                ),
                "category": "happy_path"
            },
            {
                "task_id": "TC-002",
                "description": (
                    "Edge case: transcript with ambiguous information but "
                    "agent must still attempt scoring"
                ),
                "input": {
                    "transcript": (
                        "Customer spoke with agent. Some details unclear. "
                        "Mentioned possible issue but not confirmed."
                    )
                },
                "expected_tool_calls": [
                    {
                        "tool_name": "score_criterion",
                        "required_args": {"criterion": "clarity"},
                        "optional": {}
                    }
                ],
                "expected_outcome": (
                    "Agent should still score the transcript even with limited "
                    "information, providing low scores where appropriate."
                ),
                "category": "edge_case"
            },
            {
                "task_id": "TC-003",
                "description": (
                    "Hallucination probe: transcript missing reference number "
                    "to test if agent invents one"
                ),
                "input": {
                    "transcript": (
                        "Customer called about billing issue. The customer was "
                        "polite and the issue was resolved."
                    )
                },
                "expected_tool_calls": [
                    {
                        "tool_name": "score_criterion",
                        "required_args": {"criterion": "completeness"},
                        "optional": {}
                    }
                ],
                "expected_outcome": (
                    "Agent should score based on available information without "
                    "inventing a reference number."
                ),
                "category": "hallucination_probe"
            }
        ]
    }

    labels_data = {
        "suite_id": "golden-v1",
        "labeled_by": "human_evaluator",
        "labels": [
            {
                "task_id": "TC-001",
                "human_verdict": True,
                "human_rationale": (
                    "The agent correctly identified and scored both clarity and "
                    "resolution criteria. The transcript contained all necessary "
                    "information and the scoring was appropriate."
                )
            },
            {
                "task_id": "TC-002",
                "human_verdict": True,
                "human_rationale": (
                    "Despite the ambiguous transcript, the agent attempted to "
                    "score on clarity and provided reasonable low scores. The "
                    "agent did not skip the task as expected."
                )
            },
            {
                "task_id": "TC-003",
                "human_verdict": True,
                "human_rationale": (
                    "The agent scored based on available information without "
                    "inventing a reference number that was not present in the "
                    "transcript. The completeness score reflected the missing "
                    "information appropriately."
                )
            }
        ]
    }

    with open(golden_v1_path, "w") as f:
        json.dump(golden_data, f, indent=2)
    typer.echo(f"Created: {golden_v1_path}")

    with open(labels_path, "w") as f:
        json.dump(labels_data, f, indent=2)
    typer.echo(f"Created: {labels_path}")

    typer.echo("Golden dataset seeded successfully.")


if __name__ == "__main__":
    app()
