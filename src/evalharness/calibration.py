from collections import Counter

from evalharness.models import CalibrationResult, HumanLabel, JudgedMetrics


def compute_cohen_kappa(judge_verdicts: list[bool], human_verdicts: list[bool]) -> float:
    if len(judge_verdicts) != len(human_verdicts):
        raise ValueError("Judge and human verdict lists must have the same length")
    if not judge_verdicts:
        return 0.0

    n = len(judge_verdicts)
    judge_counts = Counter(judge_verdicts)
    human_counts = Counter(human_verdicts)

    observed_agreement = sum(1 for j, h in zip(judge_verdicts, human_verdicts) if j == h) / n

    expected_agreement = (
        (judge_counts[True] / n) * (human_counts[True] / n)
        + (judge_counts[False] / n) * (human_counts[False] / n)
    )

    if expected_agreement == 1.0:
        return 1.0

    kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
    return round(kappa, 4)


def calibrate_judge(
    judge_results: dict[str, JudgedMetrics], human_labels: list[HumanLabel]
) -> CalibrationResult:
    human_by_task = {label.task_id: label for label in human_labels}
    compared: list[tuple[str, bool, bool]] = []
    disagreements: list[str] = []

    for task_id, judged in judge_results.items():
        if task_id not in human_by_task:
            continue
        human = human_by_task[task_id]
        compared.append((task_id, judged.task_success, human.human_verdict))
        if judged.task_success != human.human_verdict:
            disagreements.append(task_id)

    if not compared:
        return CalibrationResult(
            agreement_rate=0.0, total_compared=0, disagreements=[], cohen_kappa=0.0
        )

    judge_verdicts = [j for _, j, _ in compared]
    human_verdicts = [h for _, _, h in compared]

    agreement_count = sum(1 for _, j, h in compared if j == h)
    agreement_rate = agreement_count / len(compared)
    cohen_kappa = compute_cohen_kappa(judge_verdicts, human_verdicts)

    return CalibrationResult(
        agreement_rate=round(agreement_rate, 4),
        total_compared=len(compared),
        disagreements=disagreements,
        cohen_kappa=cohen_kappa,
    )
