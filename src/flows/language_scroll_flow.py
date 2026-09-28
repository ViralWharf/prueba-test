"""
language_scroll_flow.py

Recorre posiciones candidatas (clickeando hacia abajo en la pila de
ítems) hasta encontrar la ventana del ítem que se quiere validar,
delegando en `WindowLocator` el chequeo "¿es esta la ventana correcta?"
en cada posición.

Ver docstring de `config/language_config.py`: cambiar de idioma
reordena el layout porque el ancho del texto de OTROS ítems cambia
(distintos idiomas, distinto largo de palabra), así que la posición del
ítem buscado no es estable entre idiomas. De ahí la necesidad de
recorrer posiciones clickeando en vez de ir directo a una coordenada
fija -- el patrón "hacer algo (click) -> chequear (locate) -> repetir"
es exactamente lo que ofrece `retry.iterate_until`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from src.capture.screen_capture import ScreenCapturer
from src.config.language_config import max_scroll_attempts
from src.config.settings import settings
from src.flows.tooltip_flow import TooltipTarget
from src.input.mouse_controller import MouseController
from src.locators.window_locator import WindowLocation, WindowLocator
from src.utils.logger import get_logger
from src.utils.retry import RetryTimeoutError, iterate_until

logger = get_logger(__name__)


class WindowNotFoundError(RuntimeError):
    """No se encontró la ventana del ítem tras agotar las posiciones de scroll."""


@dataclass(frozen=True, slots=True)
class ScrollPosition:
    """
    Una posición candidata dentro de la pila de ítems donde clickear
    para "bajar" a la siguiente, y luego chequear si ahí apareció el
    ítem buscado.
    """

    x: int
    y: int


class LanguageScrollFlow:
    """
    Recorre posiciones de scroll para un idioma dado hasta ubicar el
    ítem buscado (por su ancla de ícono), y arma el `TooltipTarget`
    correspondiente para que `TooltipFlow` haga el resto del trabajo.
    """

    def __init__(
        self,
        *,
        capturer: ScreenCapturer,
        mouse: MouseController | None = None,
        locator: WindowLocator | None = None,
    ) -> None:
        self._capturer = capturer
        self._mouse = mouse or MouseController()
        self._locator = locator or WindowLocator()

    def find_window(
        self,
        *,
        window_id: str,
        anchor_template_path: Path,
        scroll_positions: list[ScrollPosition],
        tooltip_width: int,
        tooltip_height: int,
        tooltip_offset: tuple[int, int] = (0, 0),
        anchor_threshold: float | None = None,
        next_page_position: ScrollPosition | None = None,
        max_pages: int = 1,
    ) -> TooltipTarget:
        """
        Clickea `scroll_positions` en orden hasta que el ancla de
        `window_id` aparezca en pantalla, y devuelve el `TooltipTarget`
        listo para `TooltipFlow.run`.

        Si no aparece en la primera página y hay un botón para pasar a la
        siguiente, se repite la búsqueda en cada página usando
        `next_page_position`.

        Args:
            scroll_positions: coordenadas a clickear en cada iteración,
                en el orden en que deben probarse. No es necesario que
                cubran las `settings.max_language_positions` completas:
                si la lista es más corta, se usa su longitud como tope.
            next_page_position: posición del botón para cambiar a la
                siguiente página.
            max_pages: cúantas páginas intentar antes de rendirse.

        Raises:
            WindowNotFoundError: si se agotaron las posiciones sin
                encontrar el ancla.
        """
        def predicate() -> WindowLocation | None:
            frame = self._capturer.capture_full_screen()
            return self._locator.locate(
                frame,
                anchor_template_path,
                tooltip_width=tooltip_width,
                tooltip_height=tooltip_height,
                tooltip_offset=tooltip_offset,
                threshold=anchor_threshold,
            )

        if next_page_position is not None:
            for page_index in range(max_pages):
                location = None
                for _ in range(2):
                    location = predicate()
                    if location is not None:
                        logger.info(
                            "Ventana '%s' encontrada en la página %d",
                            window_id,
                            page_index + 1,
                        )
                        return TooltipTarget(
                            hover_point=location.anchor_match.center,
                            tooltip_region=location.tooltip_region,
                            window_id=window_id,
                        )
                    time.sleep(settings.next_page_click_delay_seconds)

                if page_index == max_pages - 1:
                    break

                logger.info(
                    "Ventana '%s' no encontrada en la página %d; avanzando a la siguiente.",
                    window_id,
                    page_index + 1,
                )
                self._mouse.click(next_page_position.x, next_page_position.y)

            raise WindowNotFoundError(
                f"No se encontró la ventana '{window_id}' tras {max_pages} página(s)."
            )

        max_iterations = min(len(scroll_positions), max_scroll_attempts())
        if max_iterations == 0:
            raise WindowNotFoundError(
                f"No hay posiciones de scroll para buscar la ventana '{window_id}'."
            )

        def action(iteration: int) -> None:
            position = scroll_positions[iteration]
            self._mouse.click(position.x, position.y)

        try:
            location, iteration = iterate_until(
                action,
                predicate,
                max_iterations=max_iterations,
                delay_between_seconds=settings.click_delay_seconds,
                description=f"ventana '{window_id}'",
            )
        except RetryTimeoutError as exc:
            raise WindowNotFoundError(
                f"No se encontró la ventana '{window_id}' tras "
                f"{max_iterations} posiciones de scroll."
            ) from exc

        logger.info(
            "Ventana '%s' encontrada en iteración %d/%d",
            window_id, iteration + 1, max_iterations,
        )

        return TooltipTarget(
            hover_point=location.anchor_match.center,
            tooltip_region=location.tooltip_region,
            window_id=window_id,
        )