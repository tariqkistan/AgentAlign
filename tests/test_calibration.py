from evalharness.calibration import calibrate_judge, compute_cohen_kappa
from evalharness.models import HumanLabel, JudgedMetrics


def test_compute_cohen_kappa_full_agreement():
    judge_verdicts = [True, True, False, False]
    human_verdicts = [True, True, False, False]
    kappa = compute_cohen_kappa(judge_verdicts, human_verdicts)
    assert kappa == 1.0


def test_compute_cohen_kappa_no_agreement():
    judge_verdicts = [True, True, False, False]
    human_verdicts = [False, False, True, True]
    kappa = compute_cohen_kappa(judge_verdicts, human_verdicts)
    assert kappa < 0


def test_compute_cohen_kappa_partial_agreement():
    judge_verdicts = [True, True, False, False]
    human_verdicts = [True, False, False, True]
    kappa = compute_cohen_kappa(judge_verdicts, human_verdicts)
    assert -1.0 <= kappa <= 1.0


def test_compute_cohen_kappa_empty():
    kappa = compute_cohen_kappa([], [])
    assert kappa == 0.0


def test_calibrate_judge_full_agreement():
    judge_results = {
        "TC-001": JudgedMetrics(
            task_success=True,
            task_success_confidence=0.9,
            hallucination_detected=False,
        ),
        "TC-002": JudgedMetrics(
            task_success=False,
            task_success_confidence=0.8,
            hallucination_detected=False,
        ),
    }
    human_labels = [
        HumanLabel(task_id="TC-001", human_verdict=True, human_rationale="Good"),
        HumanLabel(task_id="TC-002", human_verdict=False, human_rationale="Bad"),
    ]
    result = calibrate_judge(judge_results, human_labels)
    assert result.agreement_rate == 1.0
    assert result.total_compared == 2
    assert result.disagreements == []
    assert result.cohen_kappa == 1.0


def test_calibrate_judge_disagreements():
    judge_results = {
        "TC-001": JudgedMetrics(
            task_success=True,
            task_success_confidence=0.9,
            hallucination_detected=False,
        ),
        "TC-002": JudgedMetrics(
            task_success=True,
            task_success_confidence=0.8,
            hallucination_detected=False,
        ),
    }
    human_labels = [
        HumanLabel(task_id="TC-001", human_verdict=True, human_rationale="Good"),
        HumanLabel(task_id="TC-002", human_verdict=False, human_rationale="Bad"),
    ]
    result = calibrate_judge(judge_results, human_labels)
    assert result.agreement_rate == 0.5
    assert result.total_compared == 2
    assert "TC-002" in result.disagreements
    assert result.cohen_kappa < 1.0


def test_calibrate_judge_no_overlap():
    judge_results = {
        "TC-001": JudgedMetrics(
            task_success=True,
            task_success_confidence=0.9,
            hallucination_detected=False,
        )
    }
    human_labels = [
        HumanLabel(task_id="TC-002", human_verdict=True, human_rationale="Good")
    ]
    result = calibrate_judge(judge_results, human_labels)
    assert result.agreement_rate == 0.0
    assert result.total_compared == 0
    assert result.disagreements == []
