"""
Ubicación de elementos en pantalla vía template matching (OpenCV).

Este módulo responde una única pregunta: "¿dónde está este template
(anchor) dentro de este frame, y con qué confianza?". No sabe nada de
mouse, de idiomas ni de OCR -- eso es responsabilidad de
`window_locator.py` / `language_scroll_flow.py`, que lo consumen.

Contexto del problema (ver docstring de `language_config.py`): el ítem que
queremos validar siempre tiene el mismo ícono sin importar el idioma, pero
puede aparecer en distinta posición dentro de la ventana según el idioma
activo (el texto de otros ítems cambia de ancho y reordena el layout). Por
eso el anchor de template matching debe ser el ÍCONO del ítem (estable
entre idiomas), nunca el texto.

Se cachean los templates leídos desde disco porque `iterate_until`
(retry.py) llama al predicate --y por lo tanto a este matcher-- una vez
por cada posición candidata mientras se recorre el scroll de idiomas;
releer y decodificar el mismo PNG en cada intento sería trabajo repetido
innecesario.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.capture.screen_capture import Region
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TemplateLoadError(RuntimeError):
    """El template no existe en disco o no se pudo decodificar como imagen."""


@dataclass(frozen=True, slots=True)
class MatchResult:
    """
    Resultado de intentar ubicar un template dentro de un frame.

    `region` está en las mismas coordenadas que se le haya pasado a
    `TemplateMatcher` como `offset` (ver docstring de `find`): si el frame
    buscado era una captura de pantalla completa, `region` ya son
    coordenadas absolutas de pantalla, listas para pasarle a
    `MouseController`. Si el frame era un recorte, hay que sumarle el
    offset de ese recorte para volverlas absolutas -- por eso `find`
    pide ese offset en vez de dejarlo implícito.
    """

    template_name: str
    confidence: float
    region: Region
    matched: bool

    @property
    def center(self) -> tuple[int, int]:
        """Punto central de la región encontrada, útil para hover/click."""
        return (
            self.region.left + self.region.width // 2,
            self.region.top + self.region.height // 2,
        )


class TemplateMatcher:
    """
    Wrapper sobre `cv2.matchTemplate` para ubicar templates (anchors, ej:
    el ícono de un ítem) dentro de un frame capturado con
    `ScreenCapturer`.

    No implementa búsqueda multi-escala: se asume que la resolución/zoom
    del juego es estable durante una corrida. Si en el futuro hace falta
    tolerar distintos zoom levels, el punto de extensión es `find()`
    (probar el template redimensionado a una lista de escalas y quedarse
    con el mejor score), sin tocar el resto del proyecto.
    """

    def __init__(self, default_threshold: float | None = None) -> None:
        self._default_threshold = (
            default_threshold
            if default_threshold is not None
            else settings.template_match_threshold
        )
        self._cache: dict[str, np.ndarray] = {}

    def clear_cache(self) -> None:
        """Libera los templates cacheados (útil en tests entre casos)."""
        self._cache.clear()

    def _load_template(self, template_path: Path) -> np.ndarray:
        key = str(template_path)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if not template_path.is_file():
            raise TemplateLoadError(f"Template no encontrado: {template_path}")

        # IMREAD_UNCHANGED preserva el canal alpha si el PNG del ícono
        # tiene transparencia -- se usa más abajo como máscara para que
        # el fondo transparente del template no compita en el matching
        # contra el fondo real (variable) detrás del ícono en el juego.
        template = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
        if template is None:
            raise TemplateLoadError(
                f"No se pudo decodificar el template como imagen: {template_path}"
            )

        self._cache[key] = template
        logger.debug("Template cacheado: %s (shape=%s)", template_path, template.shape)
        return template

    @staticmethod
    def _split_bgr_and_mask(
        template: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Si el template tiene 4 canales (BGRA), separa color y alpha.
        El alpha se usa como máscara de `matchTemplate` para ignorar el
        fondo transparente del recorte del ícono. Si el template es BGR
        plano (3 canales), no hay máscara.
        """
        if template.ndim == 3 and template.shape[2] == 4:
            bgr = template[:, :, :3]
            alpha = template[:, :, 3]
            return bgr, alpha
        return template, None

    def find(
        self,
        frame: np.ndarray,
        template_path: Path,
        *,
        threshold: float | None = None,
        offset: tuple[int, int] = (0, 0),
    ) -> MatchResult:
        """
        Busca `template_path` dentro de `frame` y devuelve el mejor match,
        aunque su confianza no supere el threshold (revisar
        `MatchResult.matched` antes de usar la región).

        Args:
            frame: imagen BGR (o BGRA) donde buscar, típicamente salida de
                `ScreenCapturer.capture_full_screen` o `capture_region`.
            template_path: PNG del anchor, normalmente bajo
                `settings.templates_dir`.
            threshold: confianza mínima para considerar el match válido.
                Si no se pasa, usa `settings.template_match_threshold`.
            offset: (left, top) a sumarle a la posición encontrada para
                convertirla a coordenadas absolutas de pantalla, cuando
                `frame` es un recorte y no la pantalla completa.

        Returns:
            MatchResult con la mejor posición encontrada y si superó el
            threshold.
        """
        effective_threshold = threshold if threshold is not None else self._default_threshold
        template = self._load_template(template_path)
        template_bgr, mask = self._split_bgr_and_mask(template)

        frame_bgr = frame[:, :, :3] if frame.ndim == 3 and frame.shape[2] == 4 else frame

        if (
            template_bgr.shape[0] > frame_bgr.shape[0]
            or template_bgr.shape[1] > frame_bgr.shape[1]
        ):
            raise ValueError(
                f"Template '{template_path.name}' ({template_bgr.shape[:2]}) es más "
                f"grande que el frame donde se busca ({frame_bgr.shape[:2]}); "
                "revisar si se está buscando en el recorte de región correcto."
            )

        # El ancla de un tooltip es un icono pequeño y estable. Si el PNG
        # es casi del tamaño de la ventana o del botón completo, muy
        # probablemente se usó el template equivocado (botón/tooltip del
        # juego en vez del icono del item). Eso produce matches nulos o
        # falsos negativos sin aportar información útil. Rechazarlo de forma
        # explícita ayuda a corregir la plantilla antes de que la búsqueda
        # quede "silenciosamente" rota.
        template_height, template_width = template_bgr.shape[:2]
        frame_height, frame_width = frame_bgr.shape[:2]
        if (
            template_width >= max(80, frame_width * 0.25)
            or template_height >= max(80, frame_height * 0.25)
        ):
            raise ValueError(
                f"Template '{template_path.name}' ({template_bgr.shape[:2]}) parece un "
                "botón o tooltip completo, no un icono de ancla. Usa el PNG del "
                "ícono del item en `test_data/templates/<window_id>_icon.png`."
            )

        # TM_CCOEFF_NORMED: normalizado por brillo/contraste local, más
        # robusto que TM_SQDIFF frente a variaciones de iluminación del
        # juego entre corridas. Con máscara (ícono con transparencia),
        # OpenCV exige este método o TM_SQDIFF; TM_CCOEFF_NORMED es el
        # que ya usamos sin máscara, así que se mantiene consistente.
        method = cv2.TM_CCOEFF_NORMED
        if mask is not None:
            result = cv2.matchTemplate(frame_bgr, template_bgr, method, mask=mask)
        else:
            result = cv2.matchTemplate(frame_bgr, template_bgr, method)

        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        height, width = template_bgr.shape[:2]
        left = max_loc[0] + offset[0]
        top = max_loc[1] + offset[1]

        matched = max_val >= effective_threshold
        logger.debug(
            "Template '%s': confidence=%.4f (threshold=%.2f) -> matched=%s en (%d, %d)",
            template_path.name, max_val, effective_threshold, matched, left, top,
        )

        return MatchResult(
            template_name=template_path.stem,
            confidence=float(max_val),
            region=Region(left=left, top=top, width=width, height=height),
            matched=matched,
        )

    def find_best(
        self,
        frame: np.ndarray,
        template_paths: list[Path],
        *,
        threshold: float | None = None,
        offset: tuple[int, int] = (0, 0),
    ) -> MatchResult | None:
        """
        Busca varios templates candidatos en el mismo frame y devuelve el
        de mayor confianza que haya superado el threshold, o `None` si
        ninguno lo hizo.

        Pensado para `window_locator`: en una posición dada del scroll de
        idiomas puede aparecer cualquier ítem de la pila, así que se
        prueban todos los anchors conocidos y se toma el que mejor matchea
        en esa posición puntual.
        """
        best: MatchResult | None = None
        for template_path in template_paths:
            result = self.find(frame, template_path, threshold=threshold, offset=offset)
            if result.matched and (best is None or result.confidence > best.confidence):
                best = result

        if best is None:
            logger.debug(
                "find_best: ninguno de %d templates superó el threshold",
                len(template_paths),
            )
        return best