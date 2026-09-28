"""
Control de mouse para interactuar con la ventana del juego.

Se usa pyautogui como implementación base. IMPORTANTE: algunos juegos leen
input vía DirectInput/RawInput e ignoran los eventos sintéticos de
pyautogui (que usa SetCursorPos/mouse_event de Win32). Si al probar contra
el juego real el mouse "se mueve" pero el juego no reacciona (el tooltip no
aparece), el reemplazo recomendado es `pydirectinput`, que expone una API
casi idéntica (moveTo, click) pero simula input a nivel de driver via
SendInput. Por eso toda la interacción con el mouse está centralizada en
esta clase: el resto del proyecto (mouse_controller hacia arriba) no debería
necesitar cambios si se hace ese swap.

Este módulo no depende de vision/ ni ocr/, por lo que se testea con mocks
simples sobre pyautogui.
"""

from __future__ import annotations

import logging
import time

import pyautogui

from src.config.settings import settings

logger = logging.getLogger(__name__)

# Si el mouse se va a una esquina de la pantalla, pyautogui aborta la
# ejecución. Es una red de seguridad barata durante desarrollo/debugging
# (permite frenar un run que quedó clickeando en loop).
pyautogui.FAILSAFE = True


class MouseController:
    """Mueve y clickea el mouse, con soporte de hover con espera de settle."""

    def move(self, x: int, y: int, duration: float = 0.0) -> None:
        """Mueve el mouse a una posición absoluta de pantalla."""
        logger.debug("Moviendo mouse a (%d, %d)", x, y)
        pyautogui.moveTo(x, y, duration=duration)

    def click(self, x: int | None = None, y: int | None = None) -> None:
        """
        Clickea en (x, y) si se especifica, o en la posición actual del
        mouse si no.

        Incluye el delay configurado en settings.click_delay_seconds
        después del click: al recorrer posiciones de idioma
        (language_scroll_flow) es necesario darle tiempo al juego para
        reaccionar antes del siguiente click, o se pueden perder inputs.
        """
        logger.debug("Click en (%s, %s)", x, y)
        pyautogui.click(x=x, y=y)
        time.sleep(settings.click_delay_seconds)

    def hover_and_wait(
        self,
        x: int,
        y: int,
        settle_seconds: float | None = None,
    ) -> None:
        """
        Posiciona el mouse sobre (x, y) y espera a que el tooltip termine
        de renderizar la descripción completa.

        El juego muestra primero solo el título del tooltip y, recién
        ~1 segundo después, la descripción completa. Capturar antes de ese
        tiempo produce una imagen incompleta (falso negativo en la
        validación). `settle_seconds` permite override puntual (útil en
        tests); por defecto usa settings.hover_settle_seconds.
        """
        wait_time = (
            settle_seconds
            if settle_seconds is not None
            else settings.hover_settle_seconds
        )
        logger.debug(
            "Hover en (%d, %d), esperando %.2fs de settle", x, y, wait_time
        )
        self.move(x, y)
        time.sleep(wait_time)

    def is_mouse_within(
        self, left: int, top: int, right: int, bottom: int
    ) -> bool:
        """
        Confirma si el mouse quedó dentro de un rectángulo dado.
        Utilidad de diagnóstico para tests/debug, no usada en el flujo
        principal.
        """
        pos = pyautogui.position()
        return left <= pos.x <= right and top <= pos.y <= bottom