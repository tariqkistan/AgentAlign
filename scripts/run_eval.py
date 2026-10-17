import json
from pathlib import Path

import typer
from dotenv import load_dotenv

from evalharness.adapters.callscore_adapter import CallscoreAdapter
from evalharness.calibration import calibrate_judge
from evalharness.models import HumanLabel, TaskCase
from evalharness.orchestrator import run_suite
from evalharness.report import render_markdown_report, write_run_json

load_dotenv()

app = typer.Typer()


@app.command()
def run(
    agent: str = typer.Option(..., help="Agent adapter name (e.g., 'callscore')"),
    suite: str = typer.Option(..., help="Golden suite ID (e.g., 'golden-v1')"),
    model: str = typer.Option("gpt-4o", help="Model name for cost estimation"),
):
    golden_path = Path("golden") / f"{suite}.json"
    labels_path = Path("golden") / "human_labels.json"

    if not golden_path.exists():
        typer.echo(f"Golden suite not found: {golden_path}")
        raise typer.Exit(1)

    with open(golden_path) as f:
        golden_data = json.load(f)

    task_cases = [TaskCase(**tc) for tc in golden_data["task_cases"]]

    if agent == "callscore":
        adapter = CallscoreAdapter()
    else:
        typer.echo(f"Unknown agent: {agent}")
        raise typer.Exit(1)

    typer.echo(f"Running evaluation for agent '{agent}' on suite '{suite}'...")
    results, summary = run_suite(adapter, task_cases, suite, model)

    json_path = write_run_json(results, summary)
    typer.echo(f"JSON report written to: {json_path}")

    markdown = render_markdown_report(results, summary)
    markdown_path = Path("runs") / f"{summary.run_id}.md"
    with open(markdown_path, "w") as f:
        f.write(markdown)
    typer.echo(f"Markdown report written to: {markdown_path}")

    if labels_path.exists():
        with open(labels_path) as f:
            labels_data = json.load(f)
        human_labels = [HumanLabel(**label) for label in labels_data["labels"]]
        judge_results = {r.task_id: r.judged for r in results}
        calibration = calibrate_judge(judge_results, human_labels)
        typer.echo("\nCalibration Results:")
        typer.echo(f"  Agreement Rate: {calibration.agreement_rate:.2%}")
        typer.echo(f"  Cohen's Kappa: {calibration.cohen_kappa:.4f}")
        typer.echo(f"  Disagreements: {len(calibration.disagreements)}")
        if calibration.disagreements:
            typer.echo(f"  Disagreement Task IDs: {', '.join(calibration.disagreements)}")


if __name__ == "__main__":
    app()
