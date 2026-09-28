"""Tests unitarios de WindowLocator: solo lógica de cálculo de región,
delegando el matching real a un TemplateMatcher mockeado."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from src.capture.screen_capture import Region
from src.locators.window_locator import WindowLocator
from src.vision.template_matcher import MatchResult, TemplateMatcher


def _fake_match(matched: bool, confidence: float = 0.95, left=100, top=200, w=32, h=32):
    return MatchResult(
        template_name="sword_icon",
        confidence=confidence,
        region=Region(left=left, top=top, width=w, height=h),
        matched=matched,
    )


def test_locate_returns_none_when_anchor_not_matched():
    matcher = MagicMock()
    matcher.find.return_value = _fake_match(matched=False)
    locator = WindowLocator(matcher=matcher)

    result = locator.locate(
        np.zeros((10, 10, 3), dtype=np.uint8),
        Path("icon.png"),
        tooltip_width=320,
        tooltip_height=180,
    )

    assert result is None


def test_locate_computes_tooltip_region_from_anchor_and_offset():
    matcher = MagicMock()
    matcher.find.return_value = _fake_match(matched=True, left=100, top=200)
    locator = WindowLocator(matcher=matcher)

    result = locator.locate(
        np.zeros((10, 10, 3), dtype=np.uint8),
        Path("icon.png"),
        tooltip_width=320,
        tooltip_height=180,
        tooltip_offset=(20, -10),
    )

    assert result is not None
    assert result.tooltip_region == Region(left=120, top=190, width=320, height=180)
    assert result.anchor_match.confidence == 0.95


def test_locate_forwards_threshold_to_matcher():
    matcher = MagicMock()
    matcher.find.return_value = _fake_match(matched=True)
    locator = WindowLocator(matcher=matcher)

    locator.locate(
        np.zeros((10, 10, 3), dtype=np.uint8),
        Path("icon.png"),
        tooltip_width=10,
        tooltip_height=10,
        threshold=0.75,
    )

    _, kwargs = matcher.find.call_args
    assert kwargs["threshold"] == 0.75


def test_template_matcher_rejects_full_button_template_as_wrong_anchor(tmp_path):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    template_path = tmp_path / "wrong_button.png"
    cv2.imwrite(str(template_path), np.zeros((300, 500, 3), dtype=np.uint8))

    matcher = TemplateMatcher(default_threshold=0.8)

    with pytest.raises(ValueError, match="icono|ancla|botón|tooltip"):
        matcher.find(frame, template_path)