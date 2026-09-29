from types import SimpleNamespace

import numpy as np
import pytest
import main

from conftest import _find_start_client_line
from main import (
    _wait_for_start_client_load,
    _get_locale_option_label,
    _find_locale_label_line,
    _find_locale_label_reference,
    _get_combo_click_target,
    _match_language_option,
)


class _FakeLine:
    def __init__(self, text: str, left: int = 0, top: int = 0, right: int = 0, bottom: int = 0):
        self.text = text
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom


def test_find_start_client_line_returns_the_matching_ocr_line():
    result = SimpleNamespace(
        text="Launch game\nStart Client",
        lines=[
            _FakeLine("Launch game", 10, 20, 80, 40),
            _FakeLine("Start Client", 100, 150, 220, 180),
        ],
    )

    line = _find_start_client_line(result)

    assert line is not None
    assert line.text == "Start Client"


def test_find_start_client_line_returns_none_when_missing():
    result = SimpleNamespace(
        text="Launch game",
        lines=[_FakeLine("Launch game", 10, 20, 80, 40)],
    )

    assert _find_start_client_line(result) is None


def test_wait_for_start_client_load_waits_for_the_minimum_game_load_time(monkeypatch):
    elapsed = {"time": 0.0}

    def fake_monotonic():
        return elapsed["time"]

    def fake_sleep(seconds):
        elapsed["time"] += seconds

    monkeypatch.setattr("main.time.monotonic", fake_monotonic)
    monkeypatch.setattr("main.time.sleep", fake_sleep)

    result = _wait_for_start_client_load(minimum_wait_seconds=34.0, poll_interval=1.0)

    assert result is True
    assert elapsed["time"] >= 34.0


def test_main_skips_wait_when_launcher_never_started(monkeypatch, tmp_path):
    calls = []

    class DummyReport:
        total = 1
        passed_count = 0
        failed_count = 0
        error_count = 0
        skipped_count = 0

        def to_dict(self):
            return {"status": "ok"}

    object.__setattr__(main.settings, "launcher_dir", tmp_path / "missing_launcher")
    object.__setattr__(main.settings, "reports_dir", tmp_path / "reports")
    monkeypatch.setattr(main, "_resolve_languages", lambda codes: [SimpleNamespace(code="es")])
    monkeypatch.setattr(main, "_write_report", lambda report, run_id: tmp_path / "report.json")
    monkeypatch.setattr(main, "run_validation", lambda languages, capture_cropped=False: DummyReport())
    monkeypatch.setattr(main, "_wait_for_start_client_load", lambda **kwargs: calls.append("wait") or True)
    monkeypatch.setattr(main, "configure_logging", lambda: tmp_path / "logs" / "run.log")

    result = main.main(["--language", "es"])

    assert result == 0
    assert calls == []


def test_main_uses_launcher_exe_name_from_settings(monkeypatch, tmp_path):
    exe_name = "CustomLauncher.exe"
    exe_path = tmp_path / exe_name
    exe_path.write_bytes(b"fake")

    launched = {}

    def fake_popen(cmd, cwd=None):
        launched["cmd"] = cmd
        launched["cwd"] = cwd
        return SimpleNamespace(pid=123)

    object.__setattr__(main.settings, "launcher_dir", tmp_path)
    object.__setattr__(main.settings, "launcher_exe_name", exe_name)
    object.__setattr__(main.settings, "reports_dir", tmp_path / "reports")
    monkeypatch.setattr(main, "_resolve_languages", lambda codes: [SimpleNamespace(code="es")])
    monkeypatch.setattr(main, "_write_report", lambda report, run_id: tmp_path / "report.json")
    monkeypatch.setattr(main, "run_validation", lambda languages, capture_cropped=False: SimpleNamespace(total=1, passed_count=0, failed_count=0, error_count=0, skipped_count=0, to_dict=lambda: {"status": "ok"}))
    monkeypatch.setattr(main, "configure_logging", lambda: tmp_path / "logs" / "run.log")
    monkeypatch.setattr(main.subprocess, "Popen", fake_popen)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delitem(main.sys.modules, "pytest", raising=False)

    result = main.main(["--language", "es"])

    assert result == 0
    assert launched["cmd"][0] == str(exe_path)


def test_should_click_start_client_requires_detection_to_stabilize():
    assert main._should_click_start_client(detected_at=100.0, now=100.0, min_stability_seconds=1.0) is False
    assert main._should_click_start_client(detected_at=100.0, now=100.9, min_stability_seconds=1.0) is False
    assert main._should_click_start_client(detected_at=100.0, now=101.0, min_stability_seconds=1.0) is True


def test_get_locale_option_label_returns_the_listbox_value_for_supported_languages():
    assert _get_locale_option_label("en") == "English (en-us)"
    assert _get_locale_option_label("es") == "Español (es-es)"
    assert _get_locale_option_label("pt") == "Português (pt-br)"


def test_should_skip_start_client_click_when_cli_or_setting_disables_it(monkeypatch):
    monkeypatch.setattr(main, "settings", SimpleNamespace(launcher_click_start_client=False), raising=False)
    assert main._should_skip_start_client_click(skip_start_click=False) is True
    assert main._should_skip_start_client_click(skip_start_click=True) is True

    monkeypatch.setattr(main, "settings", SimpleNamespace(launcher_click_start_client=True), raising=False)
    assert main._should_skip_start_client_click(skip_start_click=False) is False
    assert main._should_skip_start_client_click(skip_start_click=True) is True


def test_find_locale_label_line_uses_label_bbox_and_clicks_right_of_the_text():
    result = SimpleNamespace(
        text="Locale",
        lines=[_FakeLine("Locale", 60, 40, 140, 70)],
    )

    locale_line = _find_locale_label_line(result)
    click_target = _get_combo_click_target(locale_line)

    assert locale_line is not None
    assert click_target == (200, 55)
    assert _match_language_option("English (en-us)", "en") is True
    assert _match_language_option("Español (es-es)", "es") is True
    assert _match_language_option("Português (pt-br)", "pt") is True


def test_match_language_option_rejects_ocr_noise_that_only_contains_the_code_fragment():
    assert main._match_language_option("English (en-us)", "en") is True
    assert main._match_language_option("Workstation Environment", "en") is False
    assert main._match_language_option("Workstation Eniemedsan", "en") is False


def test_load_english_xml_config_uses_ctrl_o_and_language_path(monkeypatch, tmp_path):
    xml_path = tmp_path / "English.settings.xml"
    xml_path.write_text("<config />", encoding="utf-8")

    calls = []

    class _FakePyautogui:
        def hotkey(self, *keys):
            calls.append(("hotkey", keys))

        def press(self, key):
            calls.append(("press", key))

        def write(self, text, interval=0.0):
            calls.append(("write", text, interval))

        def keyUp(self, key):
            calls.append(("keyUp", key))

    def fake_sleep(seconds):
        calls.append(("sleep", seconds))

    monkeypatch.setattr(main, "_resolve_xml_config_path", lambda language_code=None: str(xml_path))
    monkeypatch.setattr(main.time, "sleep", fake_sleep)
    monkeypatch.setattr("pyautogui.hotkey", _FakePyautogui().hotkey)
    monkeypatch.setattr("pyautogui.press", _FakePyautogui().press)
    monkeypatch.setattr("pyautogui.write", _FakePyautogui().write)
    monkeypatch.setattr("pyautogui.keyUp", _FakePyautogui().keyUp)

    ok = main.load_english_xml_config(hwnd=None, language_code="en")

    assert ok is True
    assert calls[0] == ("keyUp", "alt")
    assert ("hotkey", ("ctrl", "o")) in calls
    assert any(call[0] == "write" and str(xml_path) in call[1] for call in calls)
    assert ("press", "enter") in calls
    assert ("sleep", 2.0) in calls


def test_get_locale_scan_region_keeps_the_crop_tight_to_the_combo():
    frame = np.zeros((800, 1200, 3), dtype=np.uint8)
    locale_line = SimpleNamespace(left=300, right=500, top=520, bottom=560)

    crop, left, top = main._get_locale_scan_region(frame, locale_line)

    assert left >= 280
    assert left <= 300
    assert top >= 500
    assert top <= 520
    assert crop.shape[0] < 200
    assert crop.shape[1] < 500


def test_force_open_locale_dropdown_is_disabled(monkeypatch):
    calls = []

    def fake_move_to(x, y, duration=0):
        calls.append(("moveTo", x, y, duration))

    def fake_click(x, y=None):
        calls.append(("click", x, y))

    def fake_scroll(value):
        calls.append(("scroll", value))

    monkeypatch.setattr("pyautogui.moveTo", fake_move_to)
    monkeypatch.setattr("pyautogui.click", fake_click)
    monkeypatch.setattr("pyautogui.scroll", fake_scroll)

    main._force_open_locale_dropdown(150, 320)

    assert calls == []


def test_find_locale_label_reference_uses_template_match(monkeypatch):
    class _FakeRegion:
        left = 120
        top = 40
        width = 80
        height = 18

    class _FakeMatch:
        matched = True
        region = _FakeRegion()

    monkeypatch.setattr("main.TemplateMatcher.find", lambda self, frame, template_path, threshold=None: _FakeMatch())

    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    region = _find_locale_label_reference(frame)

    assert region is not None
    assert region.left == 120
    assert region.top == 40


def test_main_no_start_click_opens_launcher_and_loads_settings(monkeypatch, tmp_path):
    calls = []
    xml_path = tmp_path / "English.settings.xml"
    xml_path.write_text("<config />", encoding="utf-8")

    monkeypatch.setattr(main, "configure_logging", lambda: tmp_path / "run.log")
    monkeypatch.setattr(main, "_find_launcher_hwnd", lambda: 42)
    monkeypatch.setattr(main, "_resolve_languages", lambda codes: [SimpleNamespace(code="en", display_name="English")])
    monkeypatch.setattr(main, "_resolve_xml_config_path", lambda language_code=None: str(xml_path))
    monkeypatch.setattr(main.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(pid=123))
    monkeypatch.setattr(main, "_find_launcher_hwnd", lambda: 42)
    monkeypatch.setattr(main, "load_english_xml_config", lambda hwnd=None, *, language_code=None: calls.append(("xml", hwnd, language_code)) or True)
    monkeypatch.setattr(main, "run_validation", lambda *args, **kwargs: pytest.fail("run_validation should not execute when --no-start-click is set"))

    exit_code = main.main(["--no-start-click"])

    assert exit_code == 0
    assert any(call[0] == "xml" and call[1] == 42 and call[2] == "en" for call in calls)


def test_resolve_xml_config_path_finds_nested_variants(monkeypatch, tmp_path):
    launcher_dir = tmp_path / "launcher" / "config"
    xml_path = launcher_dir / "en" / "English.settings.xml"
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text("<config />", encoding="utf-8")

    monkeypatch.setattr(main, "settings", SimpleNamespace(test_data_dir=tmp_path, launcher_dir=tmp_path / "launcher", settings_fallback_dir=None), raising=False)
    resolved = main._resolve_xml_config_path("en")

    assert resolved == str(xml_path)


def test_resolve_xml_config_path_prefers_project_launcher_settings(monkeypatch, tmp_path):
    project_dir = tmp_path / "test_data" / "launcher_settings"
    project_dir.mkdir(parents=True, exist_ok=True)
    project_xml = project_dir / "English.settings.xml"
    project_xml.write_text("<config />", encoding="utf-8")

    monkeypatch.setattr(main, "settings", SimpleNamespace(test_data_dir=tmp_path / "test_data", launcher_dir=tmp_path / "launcher", settings_fallback_dir=None), raising=False)
    resolved = main._resolve_xml_config_path("en")

    assert resolved == str(project_xml)


def test_find_locale_label_reference_falls_back_to_bak_template(monkeypatch, tmp_path):
    class _FakeRegion:
        left = 240
        top = 55
        width = 90
        height = 24

    class _FakeMatch:
        def __init__(self, matched: bool, region):
            self.matched = matched
            self.region = region

    calls = []

    def fake_find(self, frame, template_path, threshold=None):
        calls.append(str(template_path.name))
        if template_path.name == "locale_label.png":
            return _FakeMatch(False, None)
        if template_path.name == "locale_label.bak.png":
            return _FakeMatch(True, _FakeRegion())
        raise AssertionError(f"Unexpected template: {template_path.name}")

    object.__setattr__(main.settings, "reference_images_dir", tmp_path)
    monkeypatch.setattr("main.TemplateMatcher.find", fake_find)

    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    region = _find_locale_label_reference(frame)

    assert region is not None
    assert region.left == 240
    assert region.top == 55
    assert calls == ["locale_label.bak.png"]
