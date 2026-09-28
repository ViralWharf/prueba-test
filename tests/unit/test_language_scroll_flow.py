"""
Tests unitarios de LanguageScrollFlow.

Mockean `WindowLocator` y `MouseController`/`ScreenCapturer` para
verificar el patrón "clickear -> chequear -> repetir" (retry.iterate_until)
sin depender de mouse/pantalla reales: cuántas veces se clickea, en qué
posiciones, y qué pasa cuando nunca se encuentra la ventana.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.capture.screen_capture import Region
from src.config.language_config import LanguageDefinition
from src.flows.language_scroll_flow import (
    LanguageScrollFlow,
    ScrollPosition,
    WindowNotFoundError,
)
from src.flows.tooltip_flow import TooltipFlow, TooltipTarget
from src.locators.window_locator import WindowLocation
from src.vision.template_matcher import MatchResult


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    # iterate_until (retry.py) duerme settings.click_delay_seconds entre
    # cada intento; no queremos que estos tests tarden segundos reales.
    monkeypatch.setattr("src.utils.retry.time.sleep", lambda *_a, **_k: None)


def _fake_location(left=10, top=10):
    return WindowLocation(
        anchor_match=MatchResult(
            template_name="sword_icon",
            confidence=0.95,
            region=Region(left=left, top=top, width=32, height=32),
            matched=True,
        ),
        tooltip_region=Region(left=left + 20, top=top, width=320, height=180),
    )


def _capturer_stub():
    capturer = MagicMock()
    capturer.capture_full_screen.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    return capturer


def test_find_window_stops_at_first_successful_position():
    locator = MagicMock()
    locator.locate.side_effect = [None, None, _fake_location()]

    flow = LanguageScrollFlow(capturer=_capturer_stub(), mouse=MagicMock(), locator=locator)
    positions = [ScrollPosition(x=0, y=i) for i in range(5)]

    target = flow.find_window(
        window_id="item_sword_tooltip",
        anchor_template_path=Path("icon.png"),
        scroll_positions=positions,
        tooltip_width=320,
        tooltip_height=180,
    )

    assert target.window_id == "item_sword_tooltip"
    assert target.hover_point == (26, 26)  # centro del anchor_match: (10+32//2, 10+32//2)
    assert locator.locate.call_count == 3


def test_find_window_raises_when_never_found():
    locator = MagicMock()
    locator.locate.return_value = None

    flow = LanguageScrollFlow(capturer=_capturer_stub(), mouse=MagicMock(), locator=locator)
    positions = [ScrollPosition(x=0, y=i) for i in range(3)]

    with pytest.raises(WindowNotFoundError):
        flow.find_window(
            window_id="item_sword_tooltip",
            anchor_template_path=Path("icon.png"),
            scroll_positions=positions,
            tooltip_width=320,
            tooltip_height=180,
        )

    assert locator.locate.call_count == 3


def test_find_window_clicks_each_position_in_order_before_checking():
    mouse = MagicMock()
    locator = MagicMock()
    locator.locate.side_effect = [None, _fake_location()]

    flow = LanguageScrollFlow(capturer=_capturer_stub(), mouse=mouse, locator=locator)
    positions = [ScrollPosition(x=100, y=200), ScrollPosition(x=100, y=240)]

    flow.find_window(
        window_id="item_sword_tooltip",
        anchor_template_path=Path("icon.png"),
        scroll_positions=positions,
        tooltip_width=320,
        tooltip_height=180,
    )

    assert mouse.click.call_args_list == [
        ((100, 200),),
        ((100, 240),),
    ]


def test_find_window_waits_on_current_page_before_advancing():
    mouse = MagicMock()
    locator = MagicMock()
    locator.locate.side_effect = [None, _fake_location()]

    flow = LanguageScrollFlow(capturer=_capturer_stub(), mouse=mouse, locator=locator)

    flow.find_window(
        window_id="item_sword_tooltip",
        anchor_template_path=Path("icon.png"),
        scroll_positions=[],
        tooltip_width=320,
        tooltip_height=180,
        next_page_position=ScrollPosition(x=500, y=300),
        max_pages=2,
    )

    assert mouse.click.call_args_list == []


def test_find_window_forwards_tooltip_geometry_to_locator():
    locator = MagicMock()
    locator.locate.return_value = _fake_location()

    flow = LanguageScrollFlow(capturer=_capturer_stub(), mouse=MagicMock(), locator=locator)

    flow.find_window(
        window_id="item_sword_tooltip",
        anchor_template_path=Path("icon.png"),
        scroll_positions=[ScrollPosition(x=0, y=0)],
        tooltip_width=320,
        tooltip_height=180,
        tooltip_offset=(15, 5),
        anchor_threshold=0.7,
    )

    _, kwargs = locator.locate.call_args
    assert kwargs["tooltip_width"] == 320
    assert kwargs["tooltip_height"] == 180
    assert kwargs["tooltip_offset"] == (15, 5)
    assert kwargs["threshold"] == 0.7


def test_find_window_raises_with_empty_scroll_positions():
    flow = LanguageScrollFlow(capturer=_capturer_stub(), mouse=MagicMock(), locator=MagicMock())

    with pytest.raises(WindowNotFoundError):
        flow.find_window(
            window_id="item_sword_tooltip",
            anchor_template_path=Path("icon.png"),
            scroll_positions=[],
            tooltip_width=320,
            tooltip_height=180,
        )


def test_tooltip_flow_skips_cropped_evidence_by_default():
    capturer = MagicMock()
    capturer.capture_full_screen.return_value = np.zeros((20, 20, 3), dtype=np.uint8)
    capturer.capture_region.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    preprocessor = MagicMock()
    preprocessor.preprocess.return_value = MagicMock(
        ocr_ready=np.zeros((20, 20, 3), dtype=np.uint8),
        comparison_ready=np.zeros((20, 20, 3), dtype=np.uint8),
    )
    ocr = MagicMock()
    ocr.extract_text.return_value = "texto"
    image_validator = MagicMock()
    image_validator.has_reference.return_value = False

    flow = TooltipFlow(
        capturer=capturer,
        mouse=MagicMock(),
        preprocessor=preprocessor,
        ocr_engine=ocr,
        image_validator=image_validator,
        save_evidence=True,
    )

    target = TooltipTarget(
        hover_point=(10, 10),
        tooltip_region=Region(left=0, top=0, width=10, height=10),
        window_id="item_sword_tooltip",
    )
    language = LanguageDefinition(code="xx", display_name="Test", tesseract_lang_code="eng")

    result = flow.run(target, language)

    capturer.capture_region.assert_called_once_with(target.tooltip_region)
    assert result.evidence.cropped_region_path is None


def test_tooltip_flow_can_save_cropped_evidence_when_requested():
    capturer = MagicMock()
    capturer.capture_full_screen.return_value = np.zeros((20, 20, 3), dtype=np.uint8)
    capturer.capture_region.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    preprocessor = MagicMock()
    preprocessor.preprocess.return_value = MagicMock(
        ocr_ready=np.zeros((20, 20, 3), dtype=np.uint8),
        comparison_ready=np.zeros((20, 20, 3), dtype=np.uint8),
    )
    ocr = MagicMock()
    ocr.extract_text.return_value = "texto"
    image_validator = MagicMock()
    image_validator.has_reference.return_value = False

    flow = TooltipFlow(
        capturer=capturer,
        mouse=MagicMock(),
        preprocessor=preprocessor,
        ocr_engine=ocr,
        image_validator=image_validator,
        save_evidence=True,
        capture_cropped=True,
    )

    target = TooltipTarget(
        hover_point=(10, 10),
        tooltip_region=Region(left=0, top=0, width=10, height=10),
        window_id="item_sword_tooltip",
    )
    language = LanguageDefinition(code="xx", display_name="Test", tesseract_lang_code="eng")

    result = flow.run(target, language)

    assert result.evidence.cropped_region_path is not None