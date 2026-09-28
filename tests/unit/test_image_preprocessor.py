import cv2
import numpy as np
import pytesseract

from src.ocr.tesseract_engine import TesseractOcrEngine
from src.vision.image_preprocessor import ImagePreprocessor


def test_dark_text_mask_detects_text_on_light_background():
    image = np.full((200, 500, 3), 245, dtype=np.uint8)
    cv2.putText(
        image,
        "HELLO",
        (60, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (30, 30, 30),
        4,
        cv2.LINE_AA,
    )

    mask = ImagePreprocessor().build_text_mask(image)

    assert np.count_nonzero(mask) > 100
    assert np.count_nonzero(mask[80:150, 40:260]) > 0

    ocr_ready = ImagePreprocessor().isolate_text_for_ocr(image, mask)
    assert ocr_ready.shape == (image.shape[0] * 2, image.shape[1] * 2)
    assert np.count_nonzero(ocr_ready < 200) > 100


def test_white_text_on_colored_button_is_detected():
    image = np.full((200, 500, 3), 90, dtype=np.uint8)
    image[:, :] = (80, 60, 200)
    cv2.rectangle(image, (80, 120), (420, 180), (60, 80, 180), thickness=-1)
    cv2.putText(
        image,
        "GO NOW",
        (110, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    mask = ImagePreprocessor().build_text_mask(image)

    assert np.count_nonzero(mask) > 50
    assert np.count_nonzero(mask[120:180, 100:430]) > 0


def test_button_region_is_ocr_readable():
    image = np.full((220, 500, 3), 90, dtype=np.uint8)
    image[:, :] = (80, 60, 200)
    cv2.rectangle(image, (80, 150), (420, 200), (60, 80, 180), thickness=-1)
    cv2.putText(
        image,
        "GO NOW",
        (120, 182),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    text = TesseractOcrEngine(psm=7)._detect_button_text(image, "eng")

    assert "GO" in text.upper()


def test_button_region_ignores_description_text_above_button():
    image = np.full((300, 600, 3), 245, dtype=np.uint8)
    cv2.putText(
        image,
        "FLASHBACK TO SCHOOL",
        (50, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (30, 30, 30),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Visit Marketplace for new drops featuring Y2K styles.",
        (50, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(image, (120, 220), (470, 280), (60, 80, 180), thickness=-1)
    cv2.putText(
        image,
        "GO NOW",
        (220, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    text = TesseractOcrEngine(psm=7)._detect_button_text(image, "eng")

    assert "GO" in text.upper()
    assert "VISIT" not in text.upper()


def test_button_region_ignores_description_tail_in_lower_band():
    image = np.full((350, 600, 3), 245, dtype=np.uint8)
    cv2.putText(
        image,
        "FLASHBACK TO SCHOOL",
        (50, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (30, 30, 30),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Visit Marketplace for new drops featuring Y2K styles.",
        (50, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(image, (110, 245), (470, 300), (60, 80, 180), thickness=-1)
    cv2.putText(
        image,
        "GO NOW",
        (210, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    text = TesseractOcrEngine(psm=7)._detect_button_text(image, "eng")

    assert "GO" in text.upper()
    assert "Y2K" not in text.upper()


def test_tooltip_crop_remains_readable_after_preprocessing():
    image = np.full((240, 560, 3), 245, dtype=np.uint8)
    cv2.putText(
        image,
        "FLASHBACK TO SCHOOL",
        (60, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (40, 40, 40),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Visit Marketplace for new drops featuring Y2K styles.",
        (60, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (60, 60, 60),
        2,
        cv2.LINE_AA,
    )

    ocr_ready = ImagePreprocessor().isolate_text_for_ocr(image)
    text = pytesseract.image_to_string(ocr_ready, config="--psm 6", lang="eng")

    assert "FLASHBACK TO SCHOOL" in text.upper()
