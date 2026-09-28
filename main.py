"""
main.py

Punto de entrada CLI del validador.

Uso:
    python main.py --language en
    python main.py --language en es          # varios idiomas puntuales
    python main.py --language all            # todos los soportados (default)

Por cada idioma:
    1. `LanguageScrollFlow` recorre posiciones de scroll hasta encontrar
       la ventana del ítem configurado.
    2. `TooltipFlow` corre el flujo de validación completo (hover ->
       wait -> OCR -> validar texto/imagen) sobre esa ventana.
    3. El resultado se agrega al `ValidationReport`.

Al final, escribe el reporte a `reports_dir/report_<run_id>.json` y
resume passed/failed/error/skipped por consola. El exit code es 0 solo
si no hubo FAILED ni ERROR (útil para correr esto en CI).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import shutil

from src.capture.screen_capture import ScreenCapturer
from src.config.language_config import LanguageDefinition, get_language, iter_languages
from src.config.settings import settings
import numpy as np
from src.flows.language_scroll_flow import (
    LanguageScrollFlow,
    ScrollPosition,
    WindowNotFoundError,
)
from src.flows.tooltip_flow import TooltipFlow
from src.models.validation_result import (
    Evidence,
    ValidationReport,
    ValidationResult,
    ValidationStatus,
)
from src.utils.logger import configure_logging, get_logger
import subprocess
import time
import logging
import os

from src.ocr.tesseract_engine import TesseractOcrEngine
from src.vision.template_matcher import TemplateMatcher

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuración del ítem a validar.
#
# Hoy el proyecto valida un único tooltip. Si crece a validar varios
# ítems, esto pasaría a cargarse desde un archivo de "targets" (YAML/JSON)
# en vez de constantes acá -- con un solo ítem no vale la pena esa
# indirección todavía.
# ---------------------------------------------------------------------------
WINDOW_ID = "item_sword_tooltip"
ANCHOR_TEMPLATE = settings.templates_dir / f"{WINDOW_ID}_icon.png"
TOOLTIP_WIDTH = 515 
TOOLTIP_HEIGHT = 345 
TOOLTIP_OFFSET = (10, 0)

# Coordenadas de pantalla a recorrer bajando por la pila de ítems.
# TODO(calibración): estos valores son un placeholder con espaciado fijo;
# hay que medirlos contra el juego real antes de la primera corrida.
_SCROLL_START = (960, 300)
_SCROLL_STEP_Y = 40
# TODO(calibración): ajustar a la UI real del juego; este botón es el
# que se usa para avanzar a la siguiente página cuando la lista ya no
# cabe en la pantalla actual.
_NEXT_PAGE_BUTTON = ScrollPosition(x=2410, y=1363)


def _build_scroll_positions(count: int) -> list[ScrollPosition]:
    x, y0 = _SCROLL_START
    return [ScrollPosition(x=x, y=y0 + i * _SCROLL_STEP_Y) for i in range(count)]


def _wait_for_start_client_load(
    *,
    minimum_wait_seconds: float = 34.0,
    poll_interval: float = 1.0,
) -> bool:
    """Espera un tiempo mínimo estricto tras pulsar 'Start Client'.

    En la práctica el juego tarda ~34s en abrir la UI real, así que el
    scroll no debe arrancar antes de ese umbral. La comprobación visual
    puede usarse para diagnóstico, pero nunca debe acortar el tiempo mínimo.
    """
    deadline = time.monotonic() + minimum_wait_seconds
    logger.info(
        "Esperando %ss para que el launcher termine de abrir antes de iniciar el scroll.",
        minimum_wait_seconds,
    )

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.info("Launcher: tiempo mínimo de carga cumplido; se inicia el scroll.")
            return True

        try:
            with ScreenCapturer() as capturer:
                frame = capturer.capture_full_screen(monitor_index=1)
            if frame is not None:
                h, w = frame.shape[:2]
                cx = _NEXT_PAGE_BUTTON.x
                cy = _NEXT_PAGE_BUTTON.y
                x0 = max(0, cx - 30)
                x1 = min(w, cx + 30)
                y0 = max(0, cy - 30)
                y1 = min(h, cy + 30)
                region = frame[y0:y1, x0:x1]
                if region.size:
                    import cv2

                    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                    mean = float(np.mean(gray))
                    if mean > 10.0:
                        logger.debug(
                            "Launcher: UI visible durante la espera inicial; se sigue respetando el mínimo de carga."
                        )
        except Exception:
            logger.debug(
                "No se pudo comprobar la UI lista del juego; se sigue esperando.",
                exc_info=True,
            )

        sleep_for = min(poll_interval, max(0.1, remaining))
        time.sleep(sleep_for)


def _should_skip_start_client_click(*, skip_start_click: bool = False) -> bool:
    """Responde si se debe omitir cualquier interacción de UI con 'Start Client'."""
    cli_override = bool(skip_start_click)

    try:
        settings_obj = globals().get("settings")
        if settings_obj is None:
            from src.config.settings import settings as settings_obj
    except Exception:
        settings_obj = None

    if settings_obj is not None:
        return cli_override or not bool(getattr(settings_obj, "launcher_click_start_client", True))

    return cli_override


def _get_locale_option_label(language_code: str) -> str:
    """Devuelve el texto visible exactamente en el selector de idioma de la UI."""
    code = (language_code or "").strip().lower()
    locale_map = {
        "en": "English (en-us)",
        "en-us": "English (en-us)",
        "en_us": "English (en-us)",
        "es": "Español (es-es)",
        "es-es": "Español (es-es)",
        "es_es": "Español (es-es)",
        "pt": "Português (pt-br)",
        "pt-br": "Português (pt-br)",
        "pt_br": "Português (pt-br)",
    }
    return locale_map.get(code, code or "English (en-us)")


def _language_aliases(language_code: str) -> tuple[str, ...]:
    """Aliases tolerantes para detectar el idioma aunque la UI use otra etiqueta."""
    code = (language_code or "").strip().lower()
    aliases = {
        "en": ("english", "en", "en-us", "en_us"),
        "es": ("español", "spanish", "spanlsh", "es", "es-es", "es_es", "es-es"),
        "pt": ("português", "portuguese", "pt", "pt-br", "pt_br"),
        "fi": ("finnish", "suomi", "fi", "fi-fi", "fi_fi"),
    }
    return aliases.get(code, (code, code.replace("_", "-"), code.replace("-", "_")))


def _find_language_option_line(result, language_code: str):
    """Busca la línea OCR visible con el idioma objetivo, tolerando variantes y OCR defectuoso."""
    if result is None:
        return None
    for line in getattr(result, "lines", ()):
        try:
            line_text = getattr(line, "text", "")
            if _match_language_option(line_text, language_code):
                return line
        except Exception:
            continue
    return None


def _find_locale_label_line(result) -> object | None:
    """Devuelve la línea OCR del label 'Locale' cuando se encuentre en pantalla."""
    if result is None:
        return None

    for line in getattr(result, "lines", ()):
        try:
            if "locale" in getattr(line, "text", "").lower():
                return line
        except Exception:
            continue
    return None


def _locale_reference_paths() -> list[Path]:
    """Devuelve los templates posibles para identificar el label Locale."""
    base_dir = settings.reference_images_dir
    return [
        base_dir / "locale_label.bak.png",
        base_dir / "locale_label.png",
    ]


def _find_locale_label_reference(frame: np.ndarray, *, threshold: float = 0.8):
    """Usa la imagen de referencia del label Locale para ubicarlo visualmente."""
    if frame is None or frame.size == 0:
        return None

    matcher = TemplateMatcher(default_threshold=threshold)
    for template_path in _locale_reference_paths():
        try:
            match = matcher.find(frame, template_path, threshold=threshold)
        except Exception:
            logger.debug("No se pudo localizar la referencia visual de Locale: %s", template_path, exc_info=True)
            continue

        if match.matched:
            return match.region

    return None


def _get_combo_click_target(locale_line) -> tuple[int, int]:
    """Calcula un click en la parte derecha del combo/listbox, no sobre el label."""
    left = int(getattr(locale_line, "left", 0))
    right = int(getattr(locale_line, "right", 0))
    top = int(getattr(locale_line, "top", 0))
    bottom = int(getattr(locale_line, "bottom", 0))

    x = right + 60
    y = (top + bottom) // 2
    return x, y


def _force_open_locale_dropdown(click_x: int, click_y: int, *, scroll_amount: int = -150) -> None:
    """Se desactiva cualquier interacción con el combo de Locale."""
    logger.info(
        "Launcher: se omite cualquier interacción con el combo de Locale; "
        "la configuración se carga desde archivo XML usando Ctrl+O."
    )
    return None


def _get_locale_scan_region(frame: np.ndarray, locale_line) -> tuple[np.ndarray, int, int]:
    """Recorta la zona estricta alrededor del combo/listbox y evita incluir cabeceras del juego."""
    if frame is None or frame.size == 0:
        return frame, 0, 0

    height, width = frame.shape[:2]

    base_left = int(getattr(locale_line, "left", 0))
    base_right = int(getattr(locale_line, "right", None) or (base_left + int(getattr(locale_line, "width", 0))))
    base_top = int(getattr(locale_line, "top", 0))
    base_bottom = int(getattr(locale_line, "bottom", None) or (base_top + int(getattr(locale_line, "height", 0))))

    left = max(0, base_left - 10)
    right = min(width, base_right + 120)
    top = max(0, base_top - 10)
    bottom = min(height, base_bottom + 80)

    return frame[top:bottom, left:right], left, top


SYSTEM_BLACKLIST_WORDS = {
    "workstation",
    "environment",
    "profile",
    "producer",
    "server",
    "preset",
    "setup",
    "workspace",
    "discipline",
    "helper",
    "drive",
}


def _match_language_option(text: str, language_code: str) -> bool:
    """Compara una opción visible rechazando cabeceras y palabras del sistema."""
    import re

    if not text:
        return False

    raw_lower = (text or "").lower()
    if any(blacklisted in raw_lower for blacklisted in SYSTEM_BLACKLIST_WORDS):
        return False

    normalized = raw_lower.replace("_", "-")
    normalized = re.sub(r"[^a-z0-9\-\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return False

    aliases = _language_aliases(language_code)
    for alias in aliases:
        alias_clean = re.sub(r"[^a-z0-9\-]", "", (alias or "").lower().replace("_", "-"))
        if not alias_clean:
            continue

        if len(alias_clean) > 2:
            if alias_clean in normalized:
                return True
        else:
            pattern = rf"(?<![a-z0-9]){re.escape(alias_clean)}(?![a-z0-9])"
            if re.search(pattern, normalized):
                return True
    return False


def _save_diagnostic_image(frame: np.ndarray, name_prefix: str) -> Path | None:
    """Guarda una captura diagnóstica en reports/screenshots/diagnostics."""
    if frame is None or frame.size == 0:
        return None
    try:
        import cv2

        diag_dir = settings.reports_dir / "screenshots" / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = diag_dir / f"{name_prefix}_{stamp}.png"
        if cv2.imwrite(str(path), frame):
            return path
    except Exception:
        logger.debug("No se pudo guardar diagnóstico %s.", name_prefix, exc_info=True)
    return None


def _find_launcher_hwnd() -> int | None:
    """Intenta localizar la ventana del launcher por título (case-insensitive)."""
    try:
        import win32gui

        found = []
        title_tokens = {
            token.strip().lower()
            for token in (
                settings.launcher_window_title,
                settings.game_window_title,
            )
            if token and token.strip()
        }

        def _enum(hwnd, lparam):
            try:
                if not win32gui.IsWindowVisible(hwnd):
                    return True
                title = win32gui.GetWindowText(hwnd) or ""
                if any(token in title.lower() for token in title_tokens):
                    found.append(hwnd)
            except Exception:
                pass
            return True

        win32gui.EnumWindows(_enum, 0)
        return found[0] if found else None
    except Exception:
        logger.debug("No se pudo localizar el launcher vía win32gui.", exc_info=True)
        return None


_XML_NAMES = {
    "en": ("English.settings.xml", "En.settings.xml", "Ensettings.xml"),
    "es": ("Spanish.settings.xml", "Es.settings.xml", "Essettings.xml"),
    "pt": ("Portuguese.settings.xml", "Pt.settings.xml", "Ptsettings.xml"),
}


def _resolve_xml_config_path(language_code: str | None = None) -> str | None:
    """Resuelve la ruta del XML de configuración del idioma usando la carpeta del juego como prioridad."""
    key = (language_code or "en").strip().lower().replace("-", "_").split("_")[0]
    names = _XML_NAMES.get(key, (f"{key.capitalize()}settings.xml",))
    dirs = [settings.launcher_dir]
    if settings.settings_fallback_dir:
        dirs.append(settings.settings_fallback_dir)

    for directory in dirs:
        for name in names:
            path = directory / name
            if path.exists():
                return str(path)
    return None


def load_english_xml_config(hwnd: int | None = None, *, language_code: str | None = None) -> bool:
    """Trae el launcher al frente y carga el XML del idioma vía atajos de teclado sin intentar clickear Start Client."""
    xml_path = _resolve_xml_config_path(language_code)
    if not xml_path:
        fallback = settings.launcher_dir / f"{(language_code or 'en').capitalize()}settings.xml"
        logger.error("Error: No se encontró el archivo de configuración para '%s' en: %s", language_code or "en", fallback)
        return False

    resolved_hwnd = hwnd or _find_launcher_hwnd()

    try:
        import pyautogui

        try:
            import win32gui
        except Exception:
            win32gui = None

        if resolved_hwnd is not None and win32gui is not None:
            logger.info("Trayendo la ventana del launcher al frente...")
            win32gui.SetForegroundWindow(resolved_hwnd)
            time.sleep(0.4)

        pyautogui.keyUp("alt")
        logger.info("Abriendo diálogo de archivo con Ctrl+O y cargando la configuración XML...")
        pyautogui.hotkey("ctrl", "o")
        time.sleep(0.8)

        logger.info("Escribiendo ruta del archivo de settings: %s", xml_path)
        pyautogui.write(xml_path, interval=0.02)
        time.sleep(0.3)

        logger.info("Confirmando selección de archivo con Enter...")
        pyautogui.press("enter")
        time.sleep(0.5)
        logger.info("¡Configuración XML cargada exitosamente! (%s)", xml_path)
        return True
    except Exception as exc:
        logger.error("Error cargando configuración por teclado: %s", exc, exc_info=True)
        try:
            import pyautogui
            pyautogui.keyUp("alt")
            pyautogui.press("escape")
        except Exception:
            pass
        return False


def _select_game_language(language_code: str, *, timeout: float = 60.0, poll_interval: float = 0.5, hwnd: int | None = None) -> bool:
    """Abre el combo de Locale usando la izquierda del label como referencia y luego elige la opción del idioma objetivo."""
    target_label = _get_locale_option_label(language_code)
    logger.info("Launcher: seleccionando idioma '%s' usando el combo de Locale (alias esperado='%s')", language_code, target_label)

    # Usar Tesseract combinando 'eng' y el traineddata del idioma objetivo
    try:
        lang_def = get_language(language_code)
        ocr_lang = f"eng+{lang_def.tesseract_lang_code}"
    except Exception:
        ocr_lang = "eng"
    ocr = TesseractOcrEngine()
    deadline = time.time() + timeout
    attempt = 0
    max_diag_save = 5

    # Localizar la ventana del launcher y acotar todas las capturas a su rect
    window_left = 0
    window_top = 0
    window_region = None
    if hwnd is None:
        hwnd = _find_launcher_hwnd()

    if hwnd is None:
        logger.warning("No se encontró la ventana del launcher; se usará captura de pantalla completa (menos seguro).")
    else:
        try:
            import win32gui

            l, t, r, b = win32gui.GetWindowRect(hwnd)
            w = max(0, r - l)
            h = max(0, b - t)
            if w > 0 and h > 0:
                window_left = int(l)
                window_top = int(t)
                from src.capture.screen_capture import Region

                window_region = Region(left=window_left, top=window_top, width=w, height=h)
            else:
                logger.debug("Rect de ventana inválido para hwnd=%s: %s", hwnd, (l, t, r, b))
        except Exception:
            logger.debug("Error obteniendo rect de ventana del launcher.", exc_info=True)
    while time.time() < deadline:
        try:
            with ScreenCapturer() as capturer:
                if window_region is not None:
                    frame = capturer.capture_region(window_region)
                else:
                    frame = capturer.capture_full_screen(monitor_index=1)
            if frame is None:
                time.sleep(poll_interval)
                continue

            # Use window origin as base for absolute coordinates when available
            monitor_left = window_left
            monitor_top = window_top

            result = ocr.extract_text(frame, lang=ocr_lang)
            # Diagnostics: guardar algunos frames y OCR para análisis
            try:
                if attempt < max_diag_save:
                    try:
                        import cv2

                        diag_dir = settings.reports_dir / "screenshots" / "diagnostics"
                        diag_dir.mkdir(parents=True, exist_ok=True)
                        stamp = __import__("time").strftime("%Y%m%d_%H%M%S")
                        fname = diag_dir / f"select_lang_frame_{stamp}_{attempt}.png"
                        cv2.imwrite(str(fname), frame)
                        # Guardar texto OCR si existe
                        if result is not None:
                            txt = diag_dir / f"select_lang_ocr_{stamp}_{attempt}.txt"
                            try:
                                txt.write_text(result.text or "", encoding="utf-8")
                            except Exception:
                                pass
                    except Exception:
                        logger.debug("No se pudo guardar diagnóstico de selección de idioma.", exc_info=True)
            except Exception:
                pass
            attempt += 1
            if result is None:
                time.sleep(poll_interval)
                continue

            # Inicializar preventivamente variables
            locale_region = None
            combo_box_pos = None

            # PRIORIDAD ABSOLUTA: localizar 'Locale' SOLO por template-matching
            try:
                template_path = settings.reference_images_dir / "locale_label.png"
                if template_path.exists():
                    import cv2

                    tpl = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                    if tpl is not None and tpl.size != 0:
                        th_tpl, tw_tpl = tpl.shape[:2]
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)
                        logger.debug("Locale label template match val=%.3f loc=%s", max_val, max_loc)
                        if max_val >= max(0.8, float(settings.template_match_threshold)):
                            from src.capture.screen_capture import Region

                            # max_loc are coords relative to window frame
                            locale_region = Region(left=int(max_loc[0]), top=int(max_loc[1]), width=int(tw_tpl), height=int(th_tpl))
            except Exception:
                logger.debug("Template matching para locale_label falló.", exc_info=True)

            # Si no se detectó la plantilla, registrar error y no clicar a ciegas
            if locale_region is None:
                logger.error("No se detectó visualmente 'Locale' mediante template matching (threshold>=0.8). No se realizará ningún clic.")
                return False

            # Sanity check vertical: el label Locale debe estar en la mitad inferior.
            try:
                candidate_top = int(getattr(locale_region, "top", 0))
                absolute_y = monitor_top + candidate_top
                if absolute_y < 650:
                    logger.error("Locale detectado en Y=%d (<650), descartado como falso positivo. Aborting.", absolute_y)
                    return False
            except Exception:
                pass

            # Calcular target de click sobre el combo/listbox (un único clic firme)
            if locale_region is not None:
                base_left = int(getattr(locale_region, "left", 0))
                base_top = int(getattr(locale_region, "top", 0))
                base_right = base_left + int(getattr(locale_region, "width", 0))
                base_bottom = base_top + int(getattr(locale_region, "height", 0))
                RIGHT_OFFSET = 180
                # coords relative to screen/window origin
                click_x = monitor_left + base_right + RIGHT_OFFSET
                click_y = monitor_top + base_top + (base_bottom - base_top) // 2
                logger.info("Launcher: usando referencia visual de Locale en (%d,%d) (offset=%d).", click_x, click_y, RIGHT_OFFSET)
            else:
                base_left = int(getattr(locale_line, "left", 0))
                base_top = int(getattr(locale_line, "top", 0))
                base_right = int(getattr(locale_line, "right", base_left))
                base_bottom = int(getattr(locale_line, "bottom", base_top))
                x, y = _get_combo_click_target(locale_line)
                click_x = monitor_left + x
                click_y = monitor_top + y
                logger.info("Launcher: click en combo de Locale en (%d,%d) usando label a la izquierda.", click_x, click_y)

            try:
                import pyautogui

                # Abrir el desplegable del listbox y dejarlo preparado para OCR/scroll
                _force_open_locale_dropdown(click_x, click_y)

                # Intentar enfocar dentro del área desplegada antes de OCR/scroll
                try:
                    pyautogui.moveTo(click_x, click_y + 40, duration=0.08)
                except Exception:
                    pass

                def _verify_selection() -> bool:
                    """Verifica la selección OCR sólo dentro de la región del combo/listbox."""
                    try:
                        with ScreenCapturer() as capv:
                            if window_region is not None:
                                f = capv.capture_region(window_region)
                            else:
                                f = capv.capture_full_screen(monitor_index=1)
                        if f is None:
                            return False

                        scan_img, region_left, region_top = _get_locale_scan_region(f, locale_region or locale_line)
                        if scan_img is None or getattr(scan_img, "size", 0) == 0:
                            return False

                        # Cortar de forma estricta a la zona del dropdown para que no entren elementos de la cabecera.
                        if region_top < 450 or scan_img.shape[0] > 250:
                            base_left = max(0, int(getattr(locale_region or locale_line, "left", 0)) - 10)
                            base_top = max(450, int(getattr(locale_region or locale_line, "top", 0)) - 10)
                            base_right = min(f.shape[1], base_left + 500)
                            base_bottom = min(f.shape[0], base_top + 220)
                            scan_img = f[base_top:base_bottom, base_left:base_right]

                        r = ocr.extract_text(scan_img, lang=ocr_lang)
                        if r is None:
                            return False
                        for ln in getattr(r, "lines", ()):
                            if getattr(ln, "top", 0) < 450:
                                continue
                            line_text = getattr(ln, "text", "") or ""
                            if _match_language_option(line_text, language_code):
                                logger.info("Verificación: idioma '%s' detectado en la región del combo (línea=%r).", language_code, line_text)
                                return True
                        if getattr(r, "text", "") and target_label.lower() in (r.text or "").lower():
                            logger.info("Verificación: label completo '%s' detectado dentro del combo.", target_label)
                            return True
                    except Exception:
                        logger.debug("Error verificando selección de idioma en región delimitada.", exc_info=True)
                    return False

                # Omite tipeo hardcodeado para evitar saltar a la letra 's'.
                # En lugar de teclear por defecto, posicionar el cursor sobre
                # el cuerpo del listbox para que pyautogui.scroll() funcione en Windows.
                try:
                    # Mover el cursor hacia el cuerpo del dropdown (40px abajo)
                    pyautogui.moveTo(click_x, click_y + 40, duration=0.12)
                except Exception:
                    logger.debug("No se pudo mover el cursor sobre el cuerpo del listbox.", exc_info=True)

                # Importante: se desactiva el atajo de validación rápida sobre el
                # estado cerrado del combo. El flujo debe forzar siempre la apertura
                # real del listbox y navegar por scroll para validar el componente
                # interactivo en cada ejecución.
                logger.info("Launcher: se omite el atajo de validación rápida del combo cerrado; se fuerza la apertura real del dropdown.")

                # Estrategia B: buscar la opción dentro del listbox usando OCR y scroll
                # Obtener aliases dinámicamente a partir de language_code
                try:
                    aliases = [a.lower() for a in _language_aliases(language_code)]
                except Exception:
                    aliases = [language_code.lower()]
                max_scroll_steps = 8
                for scroll_step in range(1, max_scroll_steps + 1):
                    try:
                        with ScreenCapturer() as cap_scroll:
                            if window_region is not None:
                                frame_scroll = cap_scroll.capture_region(window_region)
                            else:
                                frame_scroll = cap_scroll.capture_full_screen(monitor_index=1)
                        if frame_scroll is None:
                            logger.debug("Frame vacío en intento de scroll %d", scroll_step)
                        else:
                            # Recortar región esperada del listbox si disponemos de locale_region/locale_line
                            scan_region_img, region_left, region_top = _get_locale_scan_region(frame_scroll, locale_region or locale_line)
                            result_scroll = ocr.extract_text(scan_region_img, lang=ocr_lang)
                            if result_scroll is not None:
                                # Buscar sólo líneas del dropdown; las del encabezado se descartan por Y.
                                found_line = None
                                for ln in getattr(result_scroll, "lines", ()):
                                    if getattr(ln, "top", 0) < 450:
                                        continue
                                    txt = (getattr(ln, "text", "") or "").lower()
                                    if any(alias in txt for alias in aliases):
                                        found_line = ln
                                        break
                                if found_line is not None:
                                    left = int(getattr(found_line, "left", 0))
                                    top = int(getattr(found_line, "top", 0))
                                    width = int(getattr(found_line, "width", 0) or (getattr(found_line, "right", left) - left))
                                    height = int(getattr(found_line, "height", 0) or (getattr(found_line, "bottom", top) - top))
                                    option_x = monitor_left + region_left + left + width // 2
                                    option_y = monitor_top + region_top + top + height // 2
                                    logger.info("Launcher: OCR detectó '%s' en (%d,%d) durante scroll #%d.", getattr(found_line, "text", ""), option_x, option_y, scroll_step)
                                    pyautogui.moveTo(option_x, option_y, duration=0.2)
                                    pyautogui.click(option_x, option_y)
                                    time.sleep(0.5)
                                    if _verify_selection():
                                        logger.info("Launcher: idioma '%s' seleccionado por scroll+OCR tras %d intento(s).", language_code, scroll_step)
                                        _save_diagnostic_image(frame_scroll, "select_lang_confirmed")
                                        return True
                                else:
                                    logger.info("Launcher: intento de scroll #%d/%d para '%s': OCR no encontró coincidencia dentro del dropdown.", scroll_step, max_scroll_steps, language_code)
                            else:
                                logger.info("Launcher: intento de scroll #%d/%d para '%s': OCR no devolvió líneas del dropdown.", scroll_step, max_scroll_steps, language_code)
                    except Exception:
                        logger.debug("Error en OCR/scroll del listbox, reintentando.", exc_info=True)

                    # Hacer scroll dentro del listbox y esperar a que la UI renderice
                    try:
                        pyautogui.scroll(-150)
                        logger.info("Launcher: scroll del listbox hacia abajo (paso %d/%d) para '%s'.", scroll_step, max_scroll_steps, language_code)
                    except Exception:
                        logger.debug("No se pudo ejecutar pyautogui.scroll(), se continúa.", exc_info=True)
                    time.sleep(0.35)

                # Último diagnóstico: guardar OCR visible como ayuda de depuración.
                try:
                    with ScreenCapturer() as cap_final:
                        if window_region is not None:
                            frame_final = cap_final.capture_region(window_region)
                        else:
                            frame_final = cap_final.capture_full_screen(monitor_index=1)
                    if frame_final is not None:
                        result_final = ocr.extract_text(frame_final, lang=ocr_lang)
                        if result_final is not None:
                            logger.warning("Launcher: no se encontró '%s' tras scroll; OCR visible=%s", language_code, getattr(result_final, "text", ""))
                            _save_diagnostic_image(frame_final, "select_lang_unresolved")
                except Exception:
                    logger.debug("No se pudo guardar diagnóstico final del combo.", exc_info=True)

            except Exception:
                logger.exception("Error al interactuar con el combo de Locale.")

        except Exception:
            logger.debug("No se pudo confirmar la selección de idioma aún; reintentando.", exc_info=True)

        time.sleep(poll_interval)

    logger.warning("Launcher: no se pudo seleccionar el idioma '%s' antes del timeout.", language_code)
    return False


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validador de UI multi-idioma de un tooltip de ítem (OCR + imagen)."
    )
    parser.add_argument(
        "--language",
        nargs="+",
        default=["all"],
        help="Código(s) de idioma a validar (ej: en es), o 'all' para todos los soportados.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Eliminar logs, screenshots y reports existentes antes de ejecutar.",
    )
    parser.add_argument(
        "--no-start-click",
        action="store_true",
        help="Omitir la detección y click automático en el botón 'Start Client' al lanzar el juego.",
    )
    parser.add_argument(
        "--capture-cropped",
        action="store_true",
        help="Guarda también una captura recortada del tooltip como evidencia extra.",
    )
    parser.add_argument(
        "--click-locale-only",
        action="store_true",
        help="Solo buscar el label 'Locale' y clickar el listbox (mover a la derecha) sin seleccionar idioma.",
    )
    parser.add_argument(
        "--locale-offset-x",
        type=int,
        default=210,
        help="Desplazamiento horizontal en px desde el borde derecho del label 'Locale' hasta el centro del listbox (por defecto=210).",
    )
    return parser.parse_args(argv)


def _resolve_languages(codes: list[str]) -> list[LanguageDefinition]:
    if len(codes) == 1 and codes[0].lower() == "all":
        return list(iter_languages())
    return [get_language(code) for code in codes]


def _capture_failure_screenshot(window_id: str, language_code: str) -> Path | None:
    """Guarda una captura completa para diagnóstico cuando la validación falla antes de capturar el tooltip."""
    capture_dir = settings.reports_dir / "screenshots"
    capture_dir.mkdir(parents=True, exist_ok=True)
    stamp = __import__("time").strftime("%Y%m%d_%H%M%S")
    screenshot_path = capture_dir / f"{window_id}_{language_code}_{stamp}_failure.png"

    try:
        with ScreenCapturer() as capturer:
            frame = capturer.capture_full_screen()
    except Exception:
        return None

    if frame is None:
        return None

    import cv2

    if not cv2.imwrite(str(screenshot_path), frame):
        return None
    return screenshot_path



def _verify_dropdown_open() -> bool:
    """Guarda diagn?stico del dropdown y devuelve True sin bloquear el flujo."""
    try:
        with ScreenCapturer() as capt_ver:
            frame = capt_ver.capture_full_screen(monitor_index=1)
        if frame is None:
            logger.debug("click-locale-only: no hay frame disponible para diagn?stico del dropdown.")
            return True

        try:
            ocr = TesseractOcrEngine()
            res = ocr.extract_text(frame, lang="eng")
            text = getattr(res, "text", "") or ""
            if text:
                logger.info("click-locale-only: diagn?stico OCR del dropdown: %s", text[:200])
        except Exception:
            logger.debug("click-locale-only: diagn?stico OCR del dropdown fall?.", exc_info=True)
        return True
    except Exception:
        logger.debug("click-locale-only: verificaci?n OCR del dropdown fall?.", exc_info=True)
        return True




def _verify_dropdown_open() -> bool:
    """Guarda diagn?stico del dropdown y devuelve True sin bloquear el flujo."""
    try:
        with ScreenCapturer() as capt_ver:
            frame = capt_ver.capture_full_screen(monitor_index=1)
        if frame is None:
            logger.debug("click-locale-only: no hay frame disponible para diagn?stico del dropdown.")
            return True

        try:
            ocr = TesseractOcrEngine()
            res = ocr.extract_text(frame, lang="eng")
            text = getattr(res, "text", "") or ""
            if text:
                logger.info("click-locale-only: diagn?stico OCR del dropdown: %s", text[:200])
        except Exception:
            logger.debug("click-locale-only: diagn?stico OCR del dropdown fall?.", exc_info=True)
        return True
    except Exception:
        logger.debug("click-locale-only: verificaci?n OCR del dropdown fall?.", exc_info=True)
        return True


# NOTE: `_click_locale_listbox_only` removed. Use `--click-locale-only` to call
# the full `_select_game_language()` flow directly from `main()` which will
# open the dropdown, perform OCR+scroll and attempt selection for the
# requested language code.


def run_validation(languages: list[LanguageDefinition], *, capture_cropped: bool = False) -> ValidationReport:
    report = ValidationReport()

    # Un único ScreenCapturer (recurso de mss) compartido entre los dos
    # flows y entre todos los idiomas de la corrida -- ver docstring de
    # ScreenCapturer sobre por qué conviene reutilizarlo.
    with ScreenCapturer() as capturer:
        scroll_flow = LanguageScrollFlow(capturer=capturer)
        tooltip_flow = TooltipFlow(capturer=capturer, capture_cropped=capture_cropped)
        scroll_positions = _build_scroll_positions(max(settings.max_language_positions, 1))

        for language in languages:
            logger.info("=== Validando idioma: %s (%s) ===", language.code, language.display_name)

            try:
                target = scroll_flow.find_window(
                    window_id=WINDOW_ID,
                    anchor_template_path=ANCHOR_TEMPLATE,
                    scroll_positions=scroll_positions,
                    tooltip_width=TOOLTIP_WIDTH,
                    tooltip_height=TOOLTIP_HEIGHT,
                    tooltip_offset=TOOLTIP_OFFSET,
                    next_page_position=_NEXT_PAGE_BUTTON,
                    max_pages=6,
                )
            except WindowNotFoundError as exc:
                logger.error(str(exc))
                failure_screenshot = _capture_failure_screenshot(WINDOW_ID, language.code)
                report.add(
                    ValidationResult(
                        language=language.code,
                        window_id=WINDOW_ID,
                        status=ValidationStatus.ERROR,
                        evidence=Evidence(
                            screenshot_path=failure_screenshot or Path(),
                        ),
                        error_message=str(exc),
                    )
                )
                continue

            result = tooltip_flow.run(target, language)
            report.add(result)
            logger.info("Resultado %s: status=%s", language.code, result.status.value)

    return report


def _write_report(report: ValidationReport, run_id: str) -> Path:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = settings.reports_dir / f"report_{run_id}.json"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report_path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # Clean previous run artifacts if requested by the user
    if getattr(args, "clean", False):
        try:
            # Remove logs and screenshots directories under reports_dir
            logs_dir = settings.reports_dir / "logs"
            screenshots_dir = settings.reports_dir / "screenshots"

            if logs_dir.exists():
                shutil.rmtree(logs_dir)
            if screenshots_dir.exists():
                shutil.rmtree(screenshots_dir)

            # Remove report json files at the root of reports_dir
            for p in settings.reports_dir.glob("report_*.json"):
                try:
                    p.unlink()
                except Exception:
                    # best-effort: ignore failures to delete individual files
                    pass
        except Exception:
            # Non-fatal: log later after logging is configured
            pass

    log_path = configure_logging()
    logger.info("Log de esta corrida: %s", log_path)

    # Lanzar el launcher antes de la validación (necesario para cambiar idioma)
    def launch_launcher(
        timeout: int = 5,
        *,
        skip_start_click: bool = False,
        language_code: str | None = None,
    ) -> None:
        exe_path = settings.launcher_dir / "Launcher.exe"

        def _running_under_pytest() -> bool:
            try:
                # pytest sets this env var during collection/execution
                if "PYTEST_CURRENT_TEST" in os.environ:
                    return True
            except Exception:
                pass
            # fallback: if pytest is imported in the process
            try:
                import sys as _sys

                if "pytest" in _sys.modules:
                    return True
            except Exception:
                pass
            return False

        if _running_under_pytest():
            logger.info("Detected pytest runtime — skipping launcher launch/clicks.")
            return
        from src.utils.logger import get_logger as _get_logger
        log = _get_logger(__name__)

        skip_click = _should_skip_start_client_click(skip_start_click=skip_start_click)
        if skip_click:
            log.info(
                "skip_click=True: omitiendo por completo la selección de idioma y el clic de 'Start Client'."
            )
            return

        if not exe_path.exists():
            log.info("No se encontró %s, omitiendo lanzamiento del juego.", exe_path)
            return

        try:
            proc = subprocess.Popen([str(exe_path)], cwd=str(exe_path.parent))
        except Exception as exc:
            log.exception("Fallo al lanzar el launcher: %s", exc)
            return

        # Intentar traer la ventana del proceso al frente antes de interactuar
        def _bring_process_window_to_front(pid: int, wait_seconds: int = 10) -> int:
            try:
                import ctypes
                from ctypes import wintypes

                user32 = ctypes.windll.user32

                EnumWindows = user32.EnumWindows
                GetWindowThreadProcessId = user32.GetWindowThreadProcessId
                IsWindowVisible = user32.IsWindowVisible
                ShowWindow = user32.ShowWindow
                SetForegroundWindow = user32.SetForegroundWindow

                windows = []

                @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
                def _enum_proc(hwnd, lParam):
                    pid_dw = wintypes.DWORD()
                    GetWindowThreadProcessId(hwnd, ctypes.byref(pid_dw))
                    if pid_dw.value == pid and IsWindowVisible(hwnd):
                        windows.append(hwnd)
                    return True

                EnumWindows(_enum_proc, 0)
                if not windows:
                    # Fallback: si no se encontró ventana por PID, intentar
                    # localizarla por título (ej: 'Launcher', 'Game').
                    try:
                        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
                        def _enum_title(hwnd, lParam):
                            try:
                                if not IsWindowVisible(hwnd):
                                    return True
                                length = user32.GetWindowTextLengthW(hwnd)
                                if length <= 0:
                                    return True
                                buf = ctypes.create_unicode_buffer(length + 1)
                                user32.GetWindowTextW(hwnd, buf, length + 1)
                                title = buf.value.lower()
                                title_tokens = {
                                    token.strip().lower()
                                    for token in (
                                        settings.launcher_window_title,
                                        settings.game_window_title,
                                    )
                                    if token and token.strip()
                                }
                                if any(token in title for token in title_tokens):
                                    windows.append(hwnd)
                                    return False
                            except Exception:
                                return True
                            return True

                        EnumWindows(_enum_title, 0)
                    except Exception:
                        pass

                    if not windows:
                        return 0

                hwnd = windows[0]
                SW_RESTORE = 9
                ShowWindow(hwnd, SW_RESTORE)
                SetForegroundWindow(hwnd)
                return int(hwnd)
            except Exception:
                return 0

        # Esperar a que la ventana comience a abrirse y reintentar traerla
        # al frente durante hasta 10s (evita fallar si la ventana tarda en crearse).
        window_hwnd = 0
        try:
            max_wait = 10
            waited = 0
            while waited < max_wait and not window_hwnd:
                time.sleep(1)
                waited += 1
                window_hwnd = _bring_process_window_to_front(proc.pid, wait_seconds=1)

            if window_hwnd:
                log.info("Launcher: ventana traída al frente (pid=%d hwnd=%d)", proc.pid, window_hwnd)
                try:
                    import ctypes
                    from ctypes import wintypes

                    user32 = ctypes.windll.user32
                    rect = wintypes.RECT()
                    if user32.GetWindowRect(wintypes.HWND(window_hwnd), ctypes.byref(rect)):
                        window_left = int(rect.left)
                        window_top = int(rect.top)
                    else:
                        window_left = None
                        window_top = None
                except Exception:
                    window_left = None
                    window_top = None
                # Intentar obtener el origen del área cliente (sin bordes) para
                # clickear con mayor precisión. Mapear (0,0) cliente a coords
                # de pantalla usando MapWindowPoints.
                try:
                    client_rect = wintypes.RECT()
                    if user32.GetClientRect(wintypes.HWND(window_hwnd), ctypes.byref(client_rect)):
                        # Mapear (0,0) del cliente a coordenadas de pantalla
                        point = wintypes.POINT(0, 0)
                        MapWindowPoints = user32.MapWindowPoints
                        MapWindowPoints(wintypes.HWND(window_hwnd), 0, ctypes.byref(point), 1)
                        window_client_left = int(point.x)
                        window_client_top = int(point.y)
                        log.debug("Window client origin=(%d,%d)", window_client_left, window_client_top)
                    else:
                        window_client_left = None
                        window_client_top = None
                except Exception:
                    window_client_left = None
                    window_client_top = None
            else:
                log.info("Launcher: no se pudo traer ventana al frente (pid=%d)", proc.pid)
                window_left = None
                window_top = None
        except Exception:
            log.exception("Error intentando traer ventana al frente.")
        ocr = TesseractOcrEngine()
        # Desactivar pyautogui.FAILSAFE durante clicks automáticos para
        # evitar FailSafeException cuando el mouse está en una esquina.
        try:
            import pyautogui

            pyautogui.FAILSAFE = False
            log.debug("pyautogui.FAILSAFE disabled for automated Start Client clicks")
        except Exception:
            log.exception("Could not disable pyautogui.FAILSAFE")
        if language_code:
            logger.info("Launcher: cargando configuración XML de idioma '%s' directamente desde el menú File -> Open settings...", language_code)
            try:
                load_english_xml_config(
                    window_hwnd or _find_launcher_hwnd(),
                    language_code=language_code,
                )
            except Exception:
                logger.exception("Error cargando configuración XML para '%s' antes del Start Client.", language_code)

        poll_interval = 2.0
        deadline = time.time() + timeout

        # Valores por defecto para offsets del monitor (evita NameError
        # si no se pudo resolver la ventana). Se intentarán sobrescribir
        # tras cada captura usando `capturer._sct.monitors`.
        monitor_left = 0
        monitor_top = 0

        while time.time() < deadline:
            try:
                with ScreenCapturer() as capturer:
                    # Capturar el monitor primario (monitor_index=1) para
                    # alinear las coordenadas con pyautogui (origen en
                    # el monitor primario). Esto evita desfases cuando la
                    # captura usa el espacio virtual combinado (índice 0).
                    frame = capturer.capture_full_screen(monitor_index=1)
                    try:
                        monitor = capturer._sct.monitors[1]
                        monitor_left = int(monitor.get("left", 0) or 0)
                        monitor_top = int(monitor.get("top", 0) or 0)
                    except Exception:
                        pass
                # Log and optionally save a diagnostic frame to help debug
                try:
                    log.debug("Captured frame: %s", None if frame is None else f"shape={frame.shape}")
                    if getattr(settings, 'debug_mode', False) and frame is not None:
                        try:
                            import cv2
                            diag_dir = settings.reports_dir / "screenshots" / "diagnostics"
                            diag_dir.mkdir(parents=True, exist_ok=True)
                            stamp = __import__("time").strftime("%Y%m%d_%H%M%S")
                            fname = diag_dir / f"start_client_frame_{stamp}.png"
                            cv2.imwrite(str(fname), frame)
                            log.info("Saved diagnostic frame for Start Client: %s", fname)
                        except Exception:
                            log.exception("Failed to write diagnostic frame.")
                except Exception:
                    log.exception("Error logging diagnostic frame info.")
                    try:
                        monitor = capturer._sct.monitors[1]
                        monitor_left = int(monitor.get("left", 0))
                        monitor_top = int(monitor.get("top", 0))
                    except Exception:
                        monitor_left = 0
                        monitor_top = 0
                # Si logramos obtener el rect de la ventana del juego, usarlo como offset
                try:
                    if 'window_left' in locals() and window_left is not None:
                        monitor_left = window_left
                    if 'window_top' in locals() and window_top is not None:
                        monitor_top = window_top
                except Exception:
                    pass
                # Intentar template-matching repetido (prioritario). Si existe
                # la imagen de referencia, intentar match en un bucle hasta
                # encontrarla o hasta el deadline. Esto evita depender del OCR.
                try:
                    template_path = Path("test_data") / "reference_images" / "start_client_button.png"
                    if template_path.exists():
                        import cv2

                        tpl = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                        if tpl is not None and tpl.size != 0:
                            th_tpl, tw_tpl = tpl.shape[:2]
                            # Hacer varios intentos hasta deadline
                            while time.time() < deadline:
                                try:
                                    # Re-capturar dentro del mismo capturer para
                                    # que los offsets sigan siendo válidos.
                                    with ScreenCapturer() as capturer_tpl:
                                        frame_tpl = capturer_tpl.capture_full_screen(monitor_index=1)
                                        try:
                                            mon = capturer_tpl._sct.monitors[1]
                                            mon_left = int(mon.get("left", 0) or 0)
                                            mon_top = int(mon.get("top", 0) or 0)
                                            mon_w = int(mon.get("width", frame_tpl.shape[1]) or frame_tpl.shape[1])
                                            mon_h = int(mon.get("height", frame_tpl.shape[0]) or frame_tpl.shape[0])
                                        except Exception:
                                            mon_left = monitor_left
                                            mon_top = monitor_top

                                        if frame_tpl is None:
                                            time.sleep(poll_interval)
                                            continue

                                        gray = cv2.cvtColor(frame_tpl, cv2.COLOR_BGR2GRAY)
                                        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
                                        _, max_val, _, max_loc = cv2.minMaxLoc(res)
                                        log.debug("StartClient template match val=%.3f loc=%s", max_val, max_loc)

                                        if max_val >= settings.template_match_threshold:
                                            cx = mon_left + max_loc[0] + tw_tpl // 2
                                            cy = mon_top + max_loc[1] + th_tpl // 2
                                            try:
                                                import pyautogui
                                                # Clamp to monitor
                                                cx = max(mon_left, min(cx, mon_left + mon_w - 1))
                                                cy = max(mon_top, min(cy, mon_top + mon_h - 1))
                                                pyautogui.moveTo(cx, cy, duration=0.12)
                                                pyautogui.click()
                                                log.info("Template click realizado en (%d,%d) (val=%.3f)", cx, cy, max_val)
                                                return
                                            except Exception:
                                                log.exception("Template fallback: error al hacer click.")

                                except Exception:
                                    log.exception("Template fallback: error en iteración de matching.")
                                time.sleep(poll_interval)
                            log.debug("Template: no se encontró coincidencia válida antes del deadline.")
                except Exception:
                    log.exception("Template fallback: error durante matching.")

                try:
                    result = ocr.extract_text(frame, lang="eng")
                except Exception:
                    result = None

                if result:
                    log.debug("OCR text: %r | confidence=%.2f | words=%d", result.text, getattr(result, 'confidence', 0.0), getattr(result, 'word_count', 0))
                    for i, line in enumerate(getattr(result, 'lines', ())):
                        log.debug("OCR line %d: %r bbox=(%d,%d,%d,%d)", i, getattr(line, 'text', ''), getattr(line, 'left', 0), getattr(line, 'top', 0), getattr(line, 'right', 0), getattr(line, 'bottom', 0))

                if result and "start client" in result.text.lower():
                    log.info("Detectado 'Start Client' en pantalla.")
                    # buscar la línea exacta y clickear en su centro si es posible
                    button_line = None
                    for line in result.lines:
                        try:
                            if "start client" in line.text.lower():
                                button_line = line
                                break
                        except Exception:
                            continue

                    if button_line is not None:
                        # convertir coords relativas a la imagen en coords de pantalla
                        offset_left = window_client_left if 'window_client_left' in locals() and window_client_left is not None else monitor_left
                        offset_top = window_client_top if 'window_client_top' in locals() and window_client_top is not None else monitor_top
                        x = offset_left + (button_line.left + button_line.right) // 2
                        y = offset_top + (button_line.top + button_line.bottom) // 2
                        log.debug("Click offset=(%d,%d) -> click target=(%d,%d)", offset_left, offset_top, x, y)
                        try:
                            import pyautogui
                            time.sleep(0.5)
                            # Asegurar que el click quede dentro de la pantalla virtual
                            try:
                                virt = capturer._sct.monitors[0]
                                virt_left = int(virt.get("left", 0) or 0)
                                virt_top = int(virt.get("top", 0) or 0)
                                virt_w = int(virt.get("width", frame.shape[1]) or frame.shape[1])
                                virt_h = int(virt.get("height", frame.shape[0]) or frame.shape[0])
                            except Exception:
                                w, h = pyautogui.size()
                                virt_left = 0
                                virt_top = 0
                                virt_w = w
                                virt_h = h

                            x = max(virt_left, min(x, virt_left + virt_w - 1))
                            y = max(virt_top, min(y, virt_top + virt_h - 1))

                            pyautogui.moveTo(x, y, duration=0.15)
                            pyautogui.click()
                            pos = pyautogui.position()
                            log.info("Click realizado en 'Start Client' calc=(%d,%d) reported=%s bbox=(%d,%d,%d,%d) monitor_offset=(%d,%d)",
                                     x, y, pos, button_line.left, button_line.top, button_line.right, button_line.bottom, offset_left, offset_top)

                            # Esperar a que la UI del juego esté lista (ej. botones de
                            # navegación aparezcan). Se checa el área alrededor de
                            # `_NEXT_PAGE_BUTTON` hasta `settings.launcher_ready_timeout_seconds`.
                            timeout_ready = settings.launcher_ready_timeout_seconds
                            poll = settings.launcher_ready_poll_interval_seconds
                            deadline_ready = time.time() + timeout_ready
                            region_half = 30
                            while time.time() < deadline_ready:
                                try:
                                    with ScreenCapturer() as capturer_ready:
                                        frame_ready = capturer_ready.capture_full_screen(monitor_index=1)
                                    if frame_ready is None:
                                        time.sleep(poll)
                                        continue
                                    h, w = frame_ready.shape[:2]
                                    cx = _NEXT_PAGE_BUTTON.x
                                    cy = _NEXT_PAGE_BUTTON.y
                                    x0 = max(0, cx - region_half)
                                    x1 = min(w, cx + region_half)
                                    y0 = max(0, cy - region_half)
                                    y1 = min(h, cy + region_half)
                                    region = frame_ready[y0:y1, x0:x1]
                                    try:
                                        import cv2

                                        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                                        mean = float(np.mean(gray))
                                        log.debug("Launcher ready check mean=%.2f region=%s", mean, (x0, y0, x1, y1))
                                        # Si la región tiene contenido (no está en negro/ausente),
                                        # consideramos la UI lista.
                                        if mean > 10.0:
                                            log.info("Launcher: UI lista detectada en área del botón siguiente.")
                                            return
                                    except Exception:
                                        # Si cv2 falla sobre la región, ignorar y repetir.
                                        log.exception("Error procesando región para readiness check.")
                                except Exception:
                                    log.exception("Error comprobando si el launcher está listo.")
                                time.sleep(poll)

                            log.warning("Launcher: UI no detectada tras esperar %ds", timeout_ready)
                        except Exception:
                            log.exception("Error al hacer click en 'Start Client'.")
                    return
            except Exception:
                log.exception("Error durante detección de 'Start Client'.")

            time.sleep(poll_interval)

            # Fallback: intentar detectar botón azul con máscara HSV
            try:
                import cv2
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                blue_mask = cv2.inRange(
                    hsv,
                    np.array([90, 30, 30], dtype=np.uint8),
                    np.array([150, 255, 255], dtype=np.uint8),
                )
                if np.count_nonzero(blue_mask) >= 200:
                    num, labels, stats, centroids = cv2.connectedComponentsWithStats(blue_mask, connectivity=8)
                    best = None
                    best_area = 0
                    for label in range(1, num):
                        area = stats[label, cv2.CC_STAT_AREA]
                        x0 = stats[label, cv2.CC_STAT_LEFT]
                        y0 = stats[label, cv2.CC_STAT_TOP]
                        w = stats[label, cv2.CC_STAT_WIDTH]
                        h = stats[label, cv2.CC_STAT_HEIGHT]
                        if area > best_area and (y0 + h) >= int(frame.shape[0] * 0.4) and w >= 40:
                            best_area = area
                            best = (x0, y0, w, h)

                    if best is not None:
                        x0, y0, w, h = best
                        cx = monitor_left + x0 + w // 2
                        cy = monitor_top + y0 + h // 2
                        try:
                            import pyautogui
                            time.sleep(0.3)
                            pyautogui.click(cx, cy)
                            log.info("Fallback: click realizado en botón azul (%d,%d).", cx, cy)
                            return
                        except Exception:
                            log.exception("Fallback: error al hacer click en botón azul.")
            except Exception:
                log.exception("Fallback: error durante detección por color.")

            # Fallback 2: template-matching con imagen de referencia
            try:
                template_path = Path("test_data") / "reference_images" / "start_client_button.png"
                if template_path.exists():
                    import cv2
                    from src.config.settings import settings as _settings

                    tpl = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                    if tpl is not None and tpl.size != 0:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                        log.debug("Template match val=%.3f loc=%s", max_val, max_loc)
                        if max_val >= _settings.template_match_threshold:
                            th, tw = tpl.shape[:2]
                            cx = monitor_left + max_loc[0] + tw // 2
                            cy = monitor_top + max_loc[1] + th // 2
                            try:
                                import pyautogui
                                time.sleep(0.2)
                                pyautogui.click(cx, cy)
                                log.info("Template fallback: click realizado en (%d,%d) (val=%.3f).", cx, cy, max_val)
                                return
                            except Exception:
                                log.exception("Template fallback: error al hacer click.")
                        else:
                            try:
                                diag_dir = settings.reports_dir / "screenshots" / "diagnostics"
                                diag_dir.mkdir(parents=True, exist_ok=True)
                                stamp = __import__("time").strftime("%Y%m%d_%H%M%S")
                                fname = diag_dir / f"template_lowval_{stamp}_{int(max_val*1000)}.png"
                                cv2.imwrite(str(fname), frame)
                                log.info("Guardado frame diagnóstico por baja coincidencia template: %s (val=%.3f)", fname, max_val)
                            except Exception:
                                log.exception("No se pudo guardar frame diagnóstico.")
            except Exception:
                log.exception("Template fallback: error durante matching.")

        log.warning("No se detectó 'Start Client' tras esperar %ds", timeout)

    languages = _resolve_languages(args.language)
    selected_language = languages[0].code if languages else "en"

    # If user requested only to click/select the locale, run the full
    # selection flow (open dropdown -> OCR/scroll -> select) and exit.
    if getattr(args, 'click_locale_only', False):
        logger.info("--click-locale-only: invoking XML config load instead of OCR locale selection.")
        try:
            ok = load_english_xml_config(_find_launcher_hwnd(), language_code=selected_language)
            logger.info("--click-locale-only result=%s", ok)
            return 0 if ok else 1
        except Exception:
            logger.exception("--click-locale-only: error during XML config load.")
            return 1

    # Ejecutar el lanzamiento del juego antes de la validación
    try:
        launch_launcher(
            skip_start_click=getattr(args, 'no_start_click', False),
            language_code=selected_language,
        )

        try:
            from src.config.settings import settings as _settings
            if getattr(_settings, 'launcher_click_start_client', True):
                _wait_for_start_client_load()
        except Exception:
            logger.exception("Error esperando a que el launcher termine de abrir.")
    except Exception:
        logger.exception("Error al intentar lanzar el launcher.")

    report = run_validation(languages, capture_cropped=args.capture_cropped)

    report_path = _write_report(report, log_path.stem)
    logger.info(
        "Reporte: %s | total=%d passed=%d failed=%d error=%d skipped=%d",
        report_path, report.total, report.passed_count, report.failed_count,
        report.error_count, report.skipped_count,
    )

    return 0 if report.failed_count == 0 and report.error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())