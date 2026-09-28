"""
Validación de imagen: compara la vista "lista para comparación"
(`ImagePreprocessor.remove_text_via_inpaint`, ver
`vision/image_preprocessor.py` -- el ícono sin el texto sobrepuesto)
contra la imagen de referencia (golden) del `window_id`, vía SSIM.

Comparar DESPUÉS de quitar el texto es la razón de ser de
`image_preprocessor.py`: si se comparara el recorte crudo, la validación
de imagen fallaría por diferencias de TEXTO entre idiomas aunque el
ÍCONO sea idéntico (falso negativo), porque el texto ya se valida aparte
en `text_validator.py`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from src.config.settings import settings
from src.models.validation_result import ImageDiff
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReferenceImageNotFoundError(FileNotFoundError):
    """No existe imagen de referencia (golden) para el window_id pedido."""


class ImageValidator:
    """
    Compara una imagen "lista para comparación" contra el golden data en
    `reference_images/<window_id>.png` vía SSIM (Structural Similarity),
    más robusto que una diferencia píxel a píxel frente a variaciones
    menores de compresión/antialiasing entre capturas.
    """

    def __init__(self, similarity_threshold: float | None = None) -> None:
        self._threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.ssim_similarity_threshold
        )

    def reference_path(self, window_id: str) -> Path:
        return settings.reference_images_dir / f"{window_id}.png"

    def has_reference(self, window_id: str) -> bool:
        """
        Permite a `tooltip_flow.py` degradar a ValidationStatus.SKIPPED
        (mismo patrón que `LanguageDefinition.has_expected_text`) en vez
        de romper cuando todavía no se generó el golden de un ítem/idioma.
        """
        return self.reference_path(window_id).is_file()

    def validate(self, comparison_ready: np.ndarray, window_id: str) -> ImageDiff:
        """
        Raises:
            ReferenceImageNotFoundError: si no hay golden data para
                `window_id` (revisar `has_reference` antes de llamar).
        """
        ref_path = self.reference_path(window_id)
        reference = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
        if reference is None:
            raise ReferenceImageNotFoundError(
                f"No se pudo leer la imagen de referencia: {ref_path}"
            )

        candidate = comparison_ready
        if candidate.shape[:2] != reference.shape[:2]:
            # Puede pasar si el recorte del tooltip varía un par de
            # píxeles entre corridas (ej: el ancla matcheó con un offset
            # ligeramente distinto). Redimensionar en vez de fallar duro
            # evita falsos ERROR por una diferencia de tamaño trivial que
            # el propio SSIM va a seguir penalizando si el contenido no
            # coincide.
            candidate = cv2.resize(
                candidate,
                (reference.shape[1], reference.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            logger.debug(
                "ImageValidator: redimensionando candidato %s -> %s para comparar",
                comparison_ready.shape[:2], reference.shape[:2],
            )

        candidate_gray = (
            cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY) if candidate.ndim == 3 else candidate
        )
        reference_gray = (
            cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if reference.ndim == 3 else reference
        )

        score, _ = ssim(reference_gray, candidate_gray, full=True)
        # bool(...) explícito: `ssim` devuelve np.float64, y compararlo
        # produce np.bool_, no un bool de Python -- ImageDiff.matched debe
        # ser un bool "real" para serializar limpio a JSON (to_dict) y
        # para comparar con `is True/False` en tests/reportes.
        matched = bool(score >= self._threshold)

        logger.debug(
            "ImageValidator: ssim=%.4f (threshold=%.2f) -> matched=%s",
            score, self._threshold, matched,
        )

        return ImageDiff(
            ssim_score=float(score),
            threshold=self._threshold,
            matched=matched,
            reference_image_path=ref_path,
        )