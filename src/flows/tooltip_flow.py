"""
tooltip_flow.py

Orquesta el flujo completo de validación de UN tooltip ya localizado:

    hover -> esperar settle -> capturar -> preprocesar (separar
    texto/ícono) -> OCR -> validar texto -> validar imagen ->
    ValidationResult

Este módulo asume que la ventana YA fue ubicada en pantalla (ver
`window_locator.py`): recibe directamente el punto de hover y la región
del tooltip a capturar como un `TooltipTarget`. No sabe nada de "buscar
la posición correcta según idioma" -- esa responsabilidad es de
`language_scroll_flow.py`, que usa `TooltipFlow.run()` como paso final
una vez que encontró la ventana.

Punto clave del hover: el juego muestra primero solo el título del
tooltip y recién ~1s después la descripción completa (ver
`settings.hover_settle_seconds` y el docstring de
`MouseController.hover_and_wait`). Por eso el primer paso del flow
siempre es hacer hover y ESPERAR antes de capturar -- capturar antes de
tiempo produce una imagen con la descripción incompleta, que no es un
error técnico sino un falso negativo silencioso si no se respeta ese
settle time.

Manejo de errores: este flow nunca deja propagar una excepción hacia el
caller (`language_scroll_flow` / `main.py`). Cualquier fallo técnico
(timeout de hover a nivel de sistema operativo, OCR roto, imagen
inválida) se traduce a `ValidationStatus.ERROR` con el mensaje
correspondiente, para que una corrida multi-idioma pueda seguir con el
resto de los idiomas aunque uno falle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.capture.screen_capture import Region, ScreenCapturer
from src.config.language_config import LanguageDefinition
from src.config.settings import settings
from src.input.mouse_controller import MouseController
from src.models.validation_result import Evidence, ValidationResult, ValidationStatus
from src.ocr.ocr_engine import OcrEngine, OcrExtractionError
from src.ocr.tesseract_engine import TesseractOcrEngine
from src.utils.logger import get_logger
from src.validators.image_validator import ImageValidator, ReferenceImageNotFoundError
from src.validators.text_validator import (
    ExpectedTextNotFoundError,
    TextValidator,
    load_expected_text,
)
from src.vision.image_preprocessor import ImagePreprocessor, PreprocessedTooltip

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TooltipTarget:
    """
    Todo lo que `TooltipFlow` necesita para validar un tooltip puntual,
    ya resuelto por `window_locator.py` / `language_scroll_flow.py`:

    - hover_point: dónde poner el mouse (típicamente el centro del
      ancla del ícono) para que el juego muestre el tooltip.
    - tooltip_region: región de pantalla a capturar una vez que el
      tooltip terminó de renderizar (título + descripción).
    - window_id: identificador lógico de la ventana (ver
      `validation_result.py`), usado para buscar golden data de texto
      e imagen.
    """

    hover_point: tuple[int, int]
    tooltip_region: Region
    window_id: str


class TooltipFlow:
    """
    Orquesta un único ciclo de validación de tooltip.

    Todas las dependencias son inyectables por constructor para poder
    testear con fakes/mocks sin tocar mouse ni pantalla reales (ver
    tests/unit/test_tooltip_flow.py). Solo `capturer` es obligatorio: no
    tiene un default razonable porque `ScreenCapturer` abre un recurso
    del sistema (mss) que conviene compartir entre flows en vez de crear
    uno nuevo por instancia.
    """

    def __init__(
        self,
        *,
        capturer: ScreenCapturer,
        mouse: MouseController | None = None,
        preprocessor: ImagePreprocessor | None = None,
        ocr_engine: OcrEngine | None = None,
        text_validator: TextValidator | None = None,
        image_validator: ImageValidator | None = None,
        save_evidence: bool = True,
        capture_cropped: bool = False,
    ) -> None:
        self._capturer = capturer
        self._mouse = mouse or MouseController()
        self._preprocessor = preprocessor or ImagePreprocessor()
        self._ocr = ocr_engine or TesseractOcrEngine()
        self._text_validator = text_validator or TextValidator()
        self._image_validator = image_validator or ImageValidator()
        self._save_evidence = save_evidence
        self._capture_cropped = capture_cropped

    def run(self, target: TooltipTarget, language: LanguageDefinition) -> ValidationResult:
        """
        Ejecuta el flujo completo para un target/idioma ya localizados.

        Nunca lanza -- ver docstring del módulo sobre por qué los
        errores técnicos se traducen a `ValidationStatus.ERROR` en vez
        de propagar.
        """
        started = time.monotonic()

        try:
            self._mouse.hover_and_wait(*target.hover_point)
            full_screen = self._capturer.capture_full_screen()
            raw_capture = self._capturer.capture_region(target.tooltip_region)
        except Exception as exc:  # noqa: BLE001 - frontera del flow: se traduce a ERROR
            logger.exception("Fallo durante hover/captura de '%s'", target.window_id)
            return self._error_result(target, language, None, str(exc), started)

        evidence = self._build_evidence(full_screen, raw_capture, target, language)

        try:
            # Asegurarse de usar la imagen preprocesada para OCR y aplicar
            # un ligero upscale consistente con lo que funcionó en debug.
            mask = self._preprocessor.build_text_mask(raw_capture)
            ocr_ready = self._preprocessor.isolate_text_for_ocr(raw_capture, mask, upscale_factor=1.8)
            # También generar comparison_ready para validación de imagen
            comparison_ready = self._preprocessor.remove_text_via_inpaint(raw_capture, mask)
            preprocessed = PreprocessedTooltip(ocr_ready=ocr_ready, comparison_ready=comparison_ready, text_mask=mask)
            ocr_result = self._ocr.extract_text(preprocessed.ocr_ready, lang=language.tesseract_lang_code)

            # Fallback: si el OCR no detectó el botón, intentar detectarlo
            # en la captura en color (raw_capture) y anexarlo al resultado.
            try:
                if hasattr(self._ocr, "_detect_button_text"):
                    detected_btn = self._ocr._detect_button_text(raw_capture, language.tesseract_lang_code)
                    if detected_btn:
                        ocr_result = self._ocr._append_button_line(ocr_result, detected_btn)
            except Exception:
                # No bloquear la corrida por un fallback de detección de botón
                pass
        except OcrExtractionError as exc:
            return self._error_result(target, language, evidence, str(exc), started)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Fallo durante preprocesamiento/OCR de '%s'", target.window_id)
            return self._error_result(target, language, evidence, str(exc), started)

        text_diff = self._validate_text(ocr_result, language, target.window_id)
        image_diff = self._validate_image(preprocessed.comparison_ready, target.window_id)

        status = self._resolve_status(text_diff, image_diff)
        duration = time.monotonic() - started

        return ValidationResult(
            language=language.code,
            window_id=target.window_id,
            status=status,
            evidence=evidence,
            text_diff=text_diff,
            image_diff=image_diff,
            duration_seconds=duration,
        )

    def _validate_text(self, ocr_result, language: LanguageDefinition, window_id: str):
        if not language.has_expected_text:
            logger.debug(
                "Sin golden data de texto para idioma '%s': se omite validación de texto.",
                language.code,
            )
            return None

        try:
            expected = load_expected_text(language, window_id)
        except ExpectedTextNotFoundError as exc:
            logger.warning("%s", exc)
            return None

        return self._text_validator.validate(ocr_result, expected)

    def _validate_image(self, comparison_ready: np.ndarray, window_id: str):
        if not self._image_validator.has_reference(window_id):
            logger.debug(
                "Sin imagen de referencia para '%s': se omite validación de imagen.", window_id
            )
            return None

        try:
            return self._image_validator.validate(comparison_ready, window_id)
        except ReferenceImageNotFoundError as exc:
            logger.warning("%s", exc)
            return None

    @staticmethod
    def _resolve_status(text_diff, image_diff) -> ValidationStatus:
        """
        SKIPPED solo si NINGUNA de las dos validaciones pudo ejecutarse
        (sin golden data de texto ni de imagen -- típicamente un idioma
        recién agregado a `language_config.py`, ver su docstring). Si al
        menos una corrió, PASSED exige que TODAS las que corrieron hayan
        matcheado.
        """
        diffs = [d for d in (text_diff, image_diff) if d is not None]
        if not diffs:
            return ValidationStatus.SKIPPED
        return ValidationStatus.PASSED if all(d.matched for d in diffs) else ValidationStatus.FAILED

    def _build_evidence(
        self,
        full_screen: np.ndarray,
        raw_capture: np.ndarray | None,
        target: TooltipTarget,
        language: LanguageDefinition,
    ) -> Evidence:
        if not self._save_evidence:
            return Evidence(screenshot_path=Path())

        run_dir = settings.reports_dir / "screenshots"
        run_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")

        screenshot_path = run_dir / f"{target.window_id}_{language.code}_{stamp}_full.png"
        cv2.imwrite(str(screenshot_path), full_screen)

        cropped_path = None
        if self._capture_cropped and raw_capture is not None:
            cropped_path = run_dir / f"{target.window_id}_{language.code}_{stamp}_crop.png"
            cv2.imwrite(str(cropped_path), raw_capture)

        return Evidence(
            screenshot_path=screenshot_path,
            cropped_region_path=cropped_path,
        )

    def _error_result(
        self,
        target: TooltipTarget,
        language: LanguageDefinition,
        evidence: Evidence | None,
        message: str,
        started: float,
    ) -> ValidationResult:
        return ValidationResult(
            language=language.code,
            window_id=target.window_id,
            status=ValidationStatus.ERROR,
            evidence=evidence or Evidence(screenshot_path=Path()),
            error_message=message,
            duration_seconds=time.monotonic() - started,
        )