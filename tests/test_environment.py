from qwen_hotword.diagnostics.environment import overall_status


def test_overall_status_fails_on_failed_check() -> None:
    report = {"checks": [{"status": "pass"}, {"status": "fail"}]}
    assert overall_status(report) == "fail"


def test_overall_status_warns_without_failure() -> None:
    report = {"checks": [{"status": "pass"}, {"status": "warn"}]}
    assert overall_status(report) == "warn"


def test_overall_status_passes_when_all_pass() -> None:
    report = {"checks": [{"status": "pass"}]}
    assert overall_status(report) == "pass"
