"""
Configuración centralizada de logging.

Un único punto de entrada (`configure_logging`) que se llama una vez al
arrancar la aplicación (típicamente desde `main.py` o `conftest.py`), y
una factory (`get_logger`) que usan todos los módulos para obtener su
logger sin preocuparse de handlers/formato.

Diseño:
- Logger raíz de la app: "game_ui_validator". Todos los loggers de módulos
  cuelgan de él (ej: "game_ui_validator.vision.template_matcher"), así
  que basta configurar handlers una sola vez acá arriba.
- Consola: nivel según `settings.debug_mode` (INFO por defecto, DEBUG si
  está activado), pensado para seguir la corrida en vivo.
- Archivo: siempre a nivel DEBUG, uno por corrida, en
  `reports_dir/logs/`, para poder auditar después qué pasó exactamente
  (incluye evidencia de cada intento de hover/click/OCR).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from src.config.settings import settings

_ROOT_LOGGER_NAME = "game_ui_validator"
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def configure_logging(run_id: str | None = None) -> Path:
    """
    Configura handlers de consola y archivo para el logger raíz del proyecto.

    Idempotente: si ya se configuró en esta sesión, no vuelve a agregar
    handlers (evita logs duplicados si algo llama a esto más de una vez,
    ej: tests que importan `main` y también corren su propio setup).

    Args:
        run_id: identificador de la corrida, usado como nombre del archivo
            de log. Si no se pasa, se genera desde el timestamp actual
            (ej: "20260914_153000").

    Returns:
        Path al archivo de log de esta corrida (útil para adjuntarlo al
        reporte final).
    """
    global _configured

    log_dir = settings.reports_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{run_id}.log"

    root_logger = logging.getLogger(_ROOT_LOGGER_NAME)

    if _configured:
        get_logger(__name__).debug(
            "configure_logging() llamado de nuevo; se ignora (ya configurado)."
        )
        return log_file

    root_logger.setLevel(logging.DEBUG)  # el filtrado real lo hacen los handlers
    root_logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if settings.debug_mode else logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    _configured = True

    get_logger(__name__).info("Logging inicializado. Archivo: %s", log_file)
    return log_file


def get_logger(name: str) -> logging.Logger:
    """
    Devuelve un logger hijo del logger raíz del proyecto.

    Uso estándar en cualquier módulo:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)

    Si `configure_logging()` todavía no fue llamado (ej: un test unitario
    aislado), el logger igual funciona con el handler por defecto de
    logging; simplemente no vas a tener el archivo de la corrida hasta
    que `main.py`/`conftest.py` lo configuren.

    Args:
        name: normalmente `__name__` del módulo llamante.

    Returns:
        Logger bajo el namespace "game_ui_validator".
    """
    if name == _ROOT_LOGGER_NAME or name.startswith(f"{_ROOT_LOGGER_NAME}."):
        qualified_name = name
    else:
        qualified_name = f"{_ROOT_LOGGER_NAME}.{name}"

    return logging.getLogger(qualified_name)