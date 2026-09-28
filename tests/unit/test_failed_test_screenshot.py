from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import conftest
import main


class DummyCapturer:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def capture_full_screen(self):
        return np.zeros((40, 60, 3), dtype=np.uint8)


def test_capture_failure_screenshot_for_validation_error(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.capture.screen_capture.ScreenCapturer", DummyCapturer)
    monkeypatch.setattr(
        main,
        "settings",
        main.settings.model_copy(update={"reports_dir": tmp_path / "reports"}),
    )

    screenshot_path = main._capture_failure_screenshot("item_sword_tooltip", "en")

    assert screenshot_path is not None
    assert screenshot_path.exists()
    assert screenshot_path.parent.name == "screenshots"
    assert "item_sword_tooltip_en" in screenshot_path.name


def test_capture_failed_test_screenshot_creates_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.capture.screen_capture.ScreenCapturer", DummyCapturer)

    item = SimpleNamespace(nodeid="tests/unit/test_example.py::test_fails")

    screenshot_path = conftest.capture_failed_test_screenshot(item)

    assert screenshot_path is not None
    assert screenshot_path.exists()
    assert screenshot_path.parent.name == "pytest_failures"
    assert screenshot_path.parent.parent.name == "screenshots"
    assert "test_example.py__test_fails" in screenshot_path.name
