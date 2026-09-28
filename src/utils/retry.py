"""
Utilidades genéricas de espera activa (polling).

Estas funciones son la base de los patrones "wait-until" usados en todo el
proyecto: esperar a que aparezca un tooltip, esperar a que el hover
termine de renderizar, o iterar posiciones de idioma hasta encontrar la
ventana correcta.

No dependen de ningún módulo del dominio (capture, vision, ocr, etc.)
para poder testearse de forma aislada y reutilizarse en cualquier capa.

Nota: se mantienen deliberadamente dos funciones separadas (`wait_until`
e `iterate_until`) en vez de una sola genérica, porque representan
patrones distintos:
- `wait_until`: algo va a aparecer solo con el paso del tiempo (ej: el
  tooltip termina de renderizar tras el hover).
- `iterate_until`: hay que ejecutar una acción explícita en cada paso
  para avanzar (ej: clickear para pasar a la siguiente posición de
  idioma). Nada aparece si no se actúa.
Forzar ambos casos dentro de una sola función perdería claridad de
intención en el call site.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class RetryTimeoutError(TimeoutError):
    """Se agotó el tiempo o los intentos sin cumplir la condición esperada."""


@dataclass(frozen=True, slots=True)
class PollOutcome:
    """
    Metadata de cómo terminó un `wait_until`: cantidad de intentos y tiempo
    transcurrido. Útil para completar `duration_seconds` en un
    `ValidationResult` sin tener que medir el tiempo a mano en cada flow.
    """

    attempts: int
    elapsed_seconds: float


def wait_until(
    predicate: Callable[[], T | None],
    *,
    timeout_seconds: float,
    poll_interval_seconds: float = 0.3,
    description: str = "condición",
) -> tuple[T, PollOutcome]:
    """
    Ejecuta `predicate()` repetidamente hasta que devuelva un valor "verdadero"
    (no None, no False, no vacío) o hasta agotar `timeout_seconds`.

    Pensado para esperas donde el resultado útil ya viene del propio chequeo,
    ej: `predicate` intenta un template match y devuelve la posición
    encontrada o None si todavía no aparece.

    Args:
        predicate: función sin argumentos, barata de llamar (se invoca en
            loop). No debería lanzar excepciones esperables como parte del
            flujo normal.
        timeout_seconds: tiempo máximo total de espera.
        poll_interval_seconds: pausa entre intentos.
        description: texto human-readable para logs/mensaje de error.

    Returns:
        Tupla (valor_devuelto_por_predicate, PollOutcome).

    Raises:
        RetryTimeoutError: si se agota el tiempo sin que `predicate`
            devuelva un valor verdadero.
    """
    start = time.monotonic()
    attempts = 0

    while True:
        attempts += 1
        result = predicate()
        elapsed = time.monotonic() - start

        if result:
            logger.debug(
                "wait_until('%s') satisfecho en intento %d (%.2fs)",
                description, attempts, elapsed,
            )
            return result, PollOutcome(attempts=attempts, elapsed_seconds=elapsed)

        if elapsed >= timeout_seconds:
            raise RetryTimeoutError(
                f"Timeout esperando '{description}' tras {attempts} intentos "
                f"({elapsed:.2f}s >= {timeout_seconds:.2f}s)."
            )

        time.sleep(poll_interval_seconds)


def iterate_until(
    action: Callable[[int], None],
    predicate: Callable[[], T | None],
    *,
    max_iterations: int,
    delay_between_seconds: float = 0.3,
    description: str = "condición",
) -> tuple[T, int]:
    """
    Patrón "hacer algo -> chequear -> repetir", pensado específicamente
    para `language_scroll_flow`: hay que clickear para avanzar a la
    siguiente posición y recién después chequear si la ventana del idioma
    correcto apareció ahí. A diferencia de `wait_until`, acá nada aparece
    solo: cada iteración requiere una `action` explícita.

    Args:
        action: recibe el número de iteración actual (0-indexed) y ejecuta
            el paso para avanzar (ej: mover el mouse y clickear la
            siguiente posición).
        predicate: chequeo que devuelve el valor buscado o None/False
            (ej: intenta el hover + template match de la ventana esperada).
        max_iterations: tope de iteraciones antes de rendirse. Se puede
            pasar `settings.max_language_positions`.
        delay_between_seconds: pausa tras cada `action`, antes de chequear
            (ej: `settings.click_delay_seconds`).
        description: texto human-readable para logs/errores.

    Returns:
        Tupla (valor_devuelto_por_predicate, iteración_en_la_que_se_encontró).

    Raises:
        RetryTimeoutError: si se agotan las iteraciones sin éxito.
    """
    for iteration in range(max_iterations):
        action(iteration)
        time.sleep(delay_between_seconds)

        result = predicate()
        if result:
            logger.debug(
                "iterate_until('%s') satisfecho en iteración %d/%d",
                description, iteration + 1, max_iterations,
            )
            return result, iteration

    raise RetryTimeoutError(
        f"No se encontró '{description}' tras {max_iterations} iteraciones."
    )