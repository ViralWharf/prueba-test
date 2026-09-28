"""Tests unitarios de image_validator: comparación SSIM contra golden data."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.validators import image_validator as iv


def _solid_image(color, size=(60, 60)):
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:] = color
    return img


@pytest.fixture
def reference_dir(tmp_path, monkeypatch):
    ref_dir = tmp_path / "reference_images"
    ref_dir.mkdir()
    monkeypatch.setattr(
        "src.validators.image_validator.settings",
        SimpleNamespace(reference_images_dir=ref_dir, ssim_similarity_threshold=0.9),
    )
    return ref_dir


def test_has_reference_false_when_missing(reference_dir):
    validator = iv.ImageValidator()
    assert validator.has_reference("item_sword_tooltip") is False


def test_validate_identical_images_matches(reference_dir):
    image = _solid_image((10, 20, 30))
    cv2.imwrite(str(reference_dir / "item_sword_tooltip.png"), image)

    validator = iv.ImageValidator(similarity_threshold=0.95)
    diff = validator.validate(image.copy(), "item_sword_tooltip")

    assert diff.matched is True
    assert diff.ssim_score == pytest.approx(1.0, abs=1e-3)


def test_validate_very_different_images_does_not_match(reference_dir):
    reference = _solid_image((10, 20, 30))
    cv2.imwrite(str(reference_dir / "item_sword_tooltip.png"), reference)

    noisy = np.random.default_rng(0).integers(0, 255, size=reference.shape, dtype=np.uint8)

    validator = iv.ImageValidator(similarity_threshold=0.9)
    diff = validator.validate(noisy, "item_sword_tooltip")

    assert diff.matched is False


def test_validate_missing_reference_raises(reference_dir):
    validator = iv.ImageValidator()
    with pytest.raises(iv.ReferenceImageNotFoundError):
        validator.validate(_solid_image((1, 2, 3)), "unknown_window")


def test_validate_resizes_mismatched_candidate_size(reference_dir):
    reference = _solid_image((50, 50, 50), size=(60, 60))
    cv2.imwrite(str(reference_dir / "item_sword_tooltip.png"), reference)
    candidate = _solid_image((50, 50, 50), size=(30, 30))  # distinto tamaño, mismo contenido

    validator = iv.ImageValidator(similarity_threshold=0.9)
    diff = validator.validate(candidate, "item_sword_tooltip")

    assert diff.matched is True