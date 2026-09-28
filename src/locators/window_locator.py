"""
Localización de la ventana/tooltip de un ítem específico en pantalla.

Responde una única pregunta a nivel más alto que `template_matcher.py`:
"dado un frame de pantalla completa, ¿dónde está la ventana del ítem que
me interesa (identificada por su ícono, estable entre idiomas) y qué
región tengo que recortar para leerla?"

No sabe nada de "recorrer posiciones hasta encontrarla" -- eso es
`language_scroll_flow.py`, que llama a `locate()` una vez por cada
posición candidata a través de `retry.iterate_until`, tratando un
`None` como "en esta posición todavía no está, seguí probando".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.capture.screen_capture import Region
from src.utils.logger import get_logger
from src.vision.template_matcher import MatchResult, TemplateMatcher

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class WindowLocation:
    """
    Resultado de localizar con éxito la ventana/tooltip buscada.

    `anchor_match` guarda el match del ícono (con su `center`, útil para
    saber dónde hacer hover). `tooltip_region` es la región completa a
    recortar para OCR/comparación de imagen, derivada del ancla más un
    offset fijo -- no se detecta el borde del tooltip por separado
    porque el layout relativo ícono->tooltip es constante en el juego.
    """

    anchor_match: MatchResult
    tooltip_region: Region


class WindowLocator:
    """
    Ubica la ventana de un ítem dentro de un frame de pantalla completa,
    usando su ícono como anchor (ver docstring de template_matcher.py:
    el ícono es estable entre idiomas, el texto no).
    """

    def __init__(self, matcher: TemplateMatcher | None = None) -> None:
        self._matcher = matcher or TemplateMatcher()

    def locate(
        self,
        frame: np.ndarray,
        anchor_template_path: Path,
        *,
        tooltip_width: int,
        tooltip_height: int,
        tooltip_offset: tuple[int, int] = (0, 0),
        threshold: float | None = None,
    ) -> WindowLocation | None:
        """
        Intenta ubicar el ancla del ítem en `frame` (pantalla completa) y,
        si se encuentra por encima del threshold, calcula la región del
        tooltip a partir de esa posición.

        Args:
            frame: captura de pantalla completa (BGR), típicamente de
                `ScreenCapturer.capture_full_screen`.
            anchor_template_path: PNG del ícono del ítem a buscar.
            tooltip_width / tooltip_height: dimensiones del recorte del
                tooltip a extraer, calibradas contra el juego real.
            tooltip_offset: (dx, dy) desde la esquina superior izquierda
                del ancla encontrada hasta la esquina superior izquierda
                del tooltip.
            threshold: confianza mínima. Si no se pasa, usa el default
                del `TemplateMatcher` inyectado (a su vez
                `settings.template_match_threshold` si tampoco se
                configuró ahí).

        Returns:
            `WindowLocation` si el ancla superó el threshold, `None` si
            no se encontró en este frame puntual (el caller decide si
            reintentar en otra posición de scroll).
        """
        match = self._matcher.find(frame, anchor_template_path, threshold=threshold)
        if not match.matched:
            logger.debug(
                "Ancla '%s' no encontrada en este frame (confidence=%.4f)",
                anchor_template_path.stem, match.confidence,
            )
            return None

        left = match.region.left + tooltip_offset[0]
        top = match.region.top + tooltip_offset[1]
        tooltip_region = Region(
            left=left,
            top=top,
            width=tooltip_width,
            height=tooltip_height,
        )

        logger.debug(
            "Ventana localizada vía ancla '%s' (confidence=%.4f) en %s",
            anchor_template_path.stem, match.confidence, tooltip_region,
        )
        return WindowLocation(anchor_match=match, tooltip_region=tooltip_region)