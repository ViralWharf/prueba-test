"""
Captura de pantalla usando mss.

mss se prefiere sobre pyautogui.screenshot()/PIL.ImageGrab para este proyecto
porque:
- Es significativamente más rápido (no pasa por PIL para cada frame), lo cual
  importa porque tooltip_flow captura repetidamente durante el hover/scroll.
- Permite capturar únicamente una región (el área del tooltip) sin capturar
  la pantalla completa primero.

Este módulo no depende de vision/ ni de ocr/: solo sabe tomar screenshots y
devolverlos como arrays de numpy en formato BGR (el que espera OpenCV
nativamente). Al no tener dependencias pesadas, se testea con mocks simples
sobre `mss.mss()`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Region:
    """
    Región de pantalla a capturar, en coordenadas absolutas de pantalla
    (no relativas a la ventana del juego).
    """

    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Region con dimensiones inválidas: width={self.width}, "
                f"height={self.height}"
            )

    def to_mss_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


class ScreenCapturer:
    """
    Wrapper sobre mss para capturar pantalla completa o regiones puntuales.

    mss.mss() abre un recurso (contexto de captura sobre GDI/X11/etc.) que
    conviene reutilizar en vez de recrear en cada llamada. Por eso esta clase
    mantiene una única instancia viva durante su ciclo de vida. Se recomienda
    usarla como context manager o instanciarla una sola vez por flow y
    pasarla por dependency injection a quien la necesite.

    Ejemplo:
        with ScreenCapturer() as capturer:
            frame = capturer.capture_region(region)
    """

    def __init__(self) -> None:
        self._sct = mss.mss()

    def __enter__(self) -> "ScreenCapturer":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._sct.close()

    def capture_full_screen(self, monitor_index: int = 1) -> np.ndarray:
        """
        Captura el monitor completo.

        En mss, el índice 0 representa "todos los monitores combinados" y
        1 es el monitor primario. Se usa 1 por defecto porque es el caso
        común de un único monitor de juego; si el setup de captura usa un
        monitor secundario, pasar el índice correspondiente.
        """
        monitor = self._sct.monitors[monitor_index]
        logger.debug("Capturando pantalla completa (monitor_index=%d)", monitor_index)
        return self._grab_as_bgr(monitor)

    def capture_region(self, region: Region) -> np.ndarray:
        """Captura únicamente la región indicada, en coordenadas absolutas."""
        logger.debug("Capturando región: %s", region)
        return self._grab_as_bgr(region.to_mss_dict())

    def _grab_as_bgr(self, area: dict[str, int]) -> np.ndarray:
        """
        mss devuelve BGRA (4 canales). Se descarta el canal alpha y se deja
        en BGR de 3 canales, que es el formato que espera OpenCV de forma
        nativa, evitando conversiones repetidas en cada consumidor de
        vision/ (template_matcher, image_preprocessor, etc).
        """
        raw = self._sct.grab(area)
        frame = np.array(raw)  # shape: (h, w, 4) BGRA
        return frame[:, :, :3]