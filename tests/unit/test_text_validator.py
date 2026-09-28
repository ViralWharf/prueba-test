"""Tests unitarios de text_validator: carga de golden data + comparación."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.config.language_config import LanguageDefinition
from src.ocr.ocr_engine import OcrLine, OcrResult
from src.validators import text_validator as tv


def _patch_expected_dir(monkeypatch, expected_dir):
    monkeypatch.setattr(
        "src.config.language_config.settings",
        SimpleNamespace(expected_texts_dir=expected_dir),
    )


def test_load_expected_text_success(tmp_path, monkeypatch):
    expected_dir = tmp_path / "expected_texts"
    expected_dir.mkdir()
    (expected_dir / "en.json").write_text(
        json.dumps({"item_sword_tooltip": "Sharp Sword Deals 10 damage"}),
        encoding="utf-8",
    )
    _patch_expected_dir(monkeypatch, expected_dir)
    language = LanguageDefinition(code="en", display_name="English", tesseract_lang_code="eng")

    expected = tv.load_expected_text(language, "item_sword_tooltip")

    assert expected.text == "Sharp Sword Deals 10 damage"


def test_load_expected_text_missing_file_raises(tmp_path, monkeypatch):
    _patch_expected_dir(monkeypatch, tmp_path)
    language = LanguageDefinition(code="fr", display_name="Français", tesseract_lang_code="fra")

    with pytest.raises(tv.ExpectedTextNotFoundError):
        tv.load_expected_text(language, "item_sword_tooltip")


def test_load_expected_text_missing_key_raises(tmp_path, monkeypatch):
    (tmp_path / "en.json").write_text(json.dumps({"other_window": "..."}), encoding="utf-8")
    _patch_expected_dir(monkeypatch, tmp_path)
    language = LanguageDefinition(code="en", display_name="English", tesseract_lang_code="eng")

    with pytest.raises(tv.ExpectedTextNotFoundError):
        tv.load_expected_text(language, "item_sword_tooltip")


def test_load_expected_text_sections_success(tmp_path, monkeypatch):
    expected_dir = tmp_path / "expected_texts"
    expected_dir.mkdir()
    (expected_dir / "en.json").write_text(
        json.dumps(
            {
                "item_sword_tooltip": {
                    "title": "Main title",
                    "description": "Detailed body text",
                    "button_text": "Continue",
                }
            }
        ),
        encoding="utf-8",
    )
    _patch_expected_dir(monkeypatch, expected_dir)
    language = LanguageDefinition(code="en", display_name="English", tesseract_lang_code="eng")

    expected = tv.load_expected_text(language, "item_sword_tooltip")

    assert expected.sections["title"] == "Main title"
    assert expected.sections["description"] == "Detailed body text"
    assert expected.sections["button_text"] == "Continue"


class TestTextValidatorValidate:
    def test_exact_match_passes(self):
        validator = tv.TextValidator(similarity_threshold=0.9)
        ocr_result = OcrResult(text="Sharp Sword\nDeals 10 damage", confidence=85.0, word_count=5)
        expected = tv.ExpectedText(
            window_id="item_sword_tooltip", text="sharp sword deals 10 damage"
        )

        diff = validator.validate(ocr_result, expected)

        assert diff.matched is True
        assert diff.similarity_score == pytest.approx(1.0)

    def test_low_similarity_does_not_match(self):
        validator = tv.TextValidator(similarity_threshold=0.9)
        ocr_result = OcrResult(text="Completely different text", confidence=85.0, word_count=3)
        expected = tv.ExpectedText(
            window_id="item_sword_tooltip", text="Sharp Sword Deals 10 damage"
        )

        diff = validator.validate(ocr_result, expected)

        assert diff.matched is False

    def test_low_ocr_confidence_fails_even_with_matching_text(self):
        # settings.ocr_min_confidence real (default 60.0) sigue aplicando
        # dentro de OcrResult.meets_min_confidence().
        validator = tv.TextValidator(similarity_threshold=0.5)
        ocr_result = OcrResult(text="Sharp Sword Deals 10 damage", confidence=10.0, word_count=5)
        expected = tv.ExpectedText(
            window_id="item_sword_tooltip", text="Sharp Sword Deals 10 damage"
        )

        diff = validator.validate(ocr_result, expected)

        assert diff.matched is False
        assert diff.similarity_score == pytest.approx(1.0)

    def test_empty_ocr_result_does_not_match(self):
        validator = tv.TextValidator(similarity_threshold=0.1)
        ocr_result = OcrResult(text="", confidence=0.0, word_count=0)
        expected = tv.ExpectedText(window_id="item_sword_tooltip", text="Sharp Sword")

        diff = validator.validate(ocr_result, expected)

        assert diff.matched is False

    def test_sectioned_expected_text_matches_each_block(self):
        validator = tv.TextValidator(similarity_threshold=0.8)
        ocr_result = OcrResult(
            text="Main title\nDetailed body text\nContinue",
            confidence=90.0,
            word_count=7,
            lines=(
                OcrLine(text="Main title", top=10, bottom=30, left=0, right=200),
                OcrLine(text="Detailed body text", top=60, bottom=90, left=0, right=260),
                OcrLine(text="Continue", top=110, bottom=140, left=90, right=220),
            ),
        )
        expected = tv.ExpectedText(
            window_id="item_sword_tooltip",
            text="Main title\nDetailed body text\nContinue",
            sections={
                "title": "Main title",
                "description": "Detailed body text",
                "button_text": "Continue",
            },
        )

        diff = validator.validate(ocr_result, expected)

        assert diff.matched is True
        assert diff.section_matches is not None
        assert diff.section_matches["title"]["matched"] is True
        assert diff.section_matches["description"]["matched"] is True
        assert diff.section_matches["button_text"]["matched"] is True

    def test_left_aligned_description_tail_is_not_button(self):
        validator = tv.TextValidator(similarity_threshold=0.8)
        ocr_result = OcrResult(
            text="FLASHBACK TO SCHOOL\nVisit Marketplace for new drops featuring\nY2K styles.",
            confidence=95.0,
            word_count=10,
            lines=(
                OcrLine(text="FLASHBACK TO SCHOOL", top=10, bottom=30, left=40, right=320),
                OcrLine(
                    text="Visit Marketplace for new drops featuring",
                    top=60,
                    bottom=90,
                    left=40,
                    right=380,
                ),
                OcrLine(text="Y2K styles.", top=110, bottom=140, left=40, right=140),
            ),
        )
        expected = tv.ExpectedText(
            window_id="item_sword_tooltip",
            text="FLASHBACK TO SCHOOL\nVisit Marketplace for new drops featuring Y2K styles.\nGO Now",
            sections={
                "title": "FLASHBACK TO SCHOOL",
                "description": "Visit Marketplace for new drops featuring Y2K styles.",
                "button_text": "GO Now",
            },
        )

        diff = validator.validate(ocr_result, expected)

        assert diff.section_matches is not None
        assert diff.section_matches["description"]["actual"] == (
            "Visit Marketplace for new drops featuring Y2K styles."
        )
        assert diff.section_matches["button_text"]["actual"] == ""