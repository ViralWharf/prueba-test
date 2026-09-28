"""
Fixtures compartidas de la suite de tests.

`disable_pyautogui_failsafe` evita que un test que corra en una máquina
sin ventana real (CI headless, o simplemente el mouse cerca de una
esquina) aborte con `pyautogui.FailSafeException` -- ver
`mouse_controller.py`, donde se activa `pyautogui.FAILSAFE = True` a
propósito para corridas manuales contra el juego real.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
import subprocess
import time
import logging
from src.utils.logger import get_logger

from src.capture.screen_capture import ScreenCapturer


def _safe_test_name(nodeid: str) -> str:
    return (
        nodeid.replace("/", "__")
        .replace("::", "__")
        .replace("[", "_")
        .replace("]", "_")
    )


def capture_failed_test_screenshot(item: pytest.Item | object) -> Path | None:
    """Guarda una captura de pantalla cuando una prueba falla."""
    nodeid = getattr(item, "nodeid", "unknown")
    report_dir = Path("reports") / "screenshots" / "pytest_failures"
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    screenshot_path = report_dir / f"{_safe_test_name(nodeid)}_{timestamp}.png"

    try:
        with ScreenCapturer() as capturer:
            frame = capturer.capture_full_screen()
    except Exception:
        return None

    if not isinstance(frame, np.ndarray):
        return None

    import cv2

    if not cv2.imwrite(str(screenshot_path), frame):
        return None
    return screenshot_path


def _find_start_client_line(result) -> object | None:
    """Devuelve la línea OCR que contiene el texto 'Start Client' si existe."""
    if result is None:
        return None

    for line in getattr(result, "lines", ()):
        try:
            if "start client" in line.text.lower():
                return line
        except Exception:
            continue
    return None


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_makereport(item, call):
    """Guarda una captura automática cuando la prueba falla."""
    if call.when != "call" or call.excinfo is None:
        return

    screenshot_path = capture_failed_test_screenshot(item)
    if screenshot_path is not None:
        setattr(item, "screenshot_path", str(screenshot_path))


@pytest.fixture(autouse=True)
def disable_pyautogui_failsafe():
    import pyautogui

    original = pyautogui.FAILSAFE
    pyautogui.FAILSAFE = False
    yield
    pyautogui.FAILSAFE = original


@pytest.fixture(scope="session", autouse=True)
def launch_launcher():
    """Lanza Launcher.exe al inicio de la sesión de tests y espera a que
    aparezca el botón "Start Client" en pantalla (OCR). Si no se encuentra
    después del timeout, continúa pero loguea un warning.

    Este fixture intenta terminar el proceso al final de la sesión.
    """
    from src.config.settings import settings as _settings

    exe_path = _settings.launcher_dir / "Launcher.exe"
    logger = get_logger(__name__)

    if not exe_path.exists():
        logger.info("No se encontró %s, omitiendo lanzamiento del juego.", exe_path)
        yield
        return

    try:
        proc = subprocess.Popen([str(exe_path)], cwd=str(exe_path.parent))
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.exception("Fallo al lanzar el launcher: %s", exc)
        yield
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
                return 0

            hwnd = windows[0]
            SW_RESTORE = 9
            ShowWindow(hwnd, SW_RESTORE)
            SetForegroundWindow(hwnd)
            return int(hwnd)
        except Exception:
            return 0

    # Dar tiempo inicial a que la ventana comience a abrirse
    time.sleep(2)
    window_hwnd = 0
    try:
        window_hwnd = _bring_process_window_to_front(proc.pid, wait_seconds=10)
        if window_hwnd:
            logger.info("Launcher: ventana traída al frente (pid=%d hwnd=%d)", proc.pid, window_hwnd)
            # obtener rect de la ventana para usar como offset en los clicks
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
        else:
            logger.info("Launcher: no se pudo traer ventana al frente (pid=%d)", proc.pid)
            window_left = None
            window_top = None
    except Exception:
        logger.exception("Error intentando traer ventana al frente.")
    except Exception:
        logger.exception("Error intentando traer ventana al frente.")

    # Usar OCR para detectar "Start Client" en pantalla
    from src.capture.screen_capture import ScreenCapturer
    from src.ocr.tesseract_engine import TesseractOcrEngine

    ocr = TesseractOcrEngine()
    timeout = 90  # segundos totales para esperar hasta que aparezca el botón
    poll_interval = 2.0
    deadline = time.time() + timeout
    # Respetar flag de configuración: permite deshabilitar intentos automáticos
    # de click en 'Start Client' cuando cause problemas en ciertas máquinas.
    try:
        from src.config.settings import settings as _settings
        if not _settings.launcher_click_start_client:
            logger.info("launcher_click_start_client disabled; skipping 'Start Client' detection/click.")
            # Ajustar deadline a tiempo actual para que el bucle no se ejecute
            deadline = time.time()
    except Exception:
        # Si falla la importación o acceso a settings, proceder normalmente.
        pass
    found = False

    while time.time() < deadline:
        try:
            with ScreenCapturer() as capturer:
                frame = capturer.capture_full_screen()
                # obtener offset del monitor para convertir coords relativas a absolutas
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

            # Intentar template-matching primero (más robusto para botones estáticos)
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
                        logger.debug("StartClient template match val=%.3f loc=%s", max_val, max_loc)
                        if max_val >= _settings.template_match_threshold:
                            th, tw = tpl.shape[:2]
                            cx = monitor_left + max_loc[0] + tw // 2
                            cy = monitor_top + max_loc[1] + th // 2
                            try:
                                import pyautogui
                                time.sleep(0.2)
                                pyautogui.click(cx, cy)
                                logger.info("Template fallback: click realizado en (%d,%d) (val=%.3f).", cx, cy, max_val)
                                found = True
                                break
                            except Exception:
                                logger.exception("Template fallback: error al hacer click.")
                        else:
                            logger.debug("StartClient template below threshold: %.3f", max_val)
                            try:
                                # guardar frame diagnóstico para inspección
                                diag_dir = logger.root.handlers[1].baseFilename if len(logger.root.handlers) > 1 else None
                            except Exception:
                                diag_dir = None
            except Exception:
                logger.exception("Template fallback: error durante matching.")

            try:
                result = ocr.extract_text(frame, lang="eng")
            except Exception:
                result = None

            if result:
                logger.debug("OCR text: %r | confidence=%.2f | words=%d", result.text, getattr(result, 'confidence', 0.0), getattr(result, 'word_count', 0))
                for i, line in enumerate(getattr(result, 'lines', ())):
                    logger.debug("OCR line %d: %r bbox=(%d,%d,%d,%d)", i, getattr(line, 'text', ''), getattr(line, 'left', 0), getattr(line, 'top', 0), getattr(line, 'right', 0), getattr(line, 'bottom', 0))

            if result and "start client" in result.text.lower():
                logger.info("Detectado 'Start Client' en pantalla.")
                button_line = _find_start_client_line(result)
                if button_line is not None:
                    # coords relativas a la imagen -> sumar offset del monitor
                    x = monitor_left + (button_line.left + button_line.right) // 2
                    y = monitor_top + (button_line.top + button_line.bottom) // 2
                    logger.debug("Monitor offset=(%d,%d) -> click target=(%d,%d)", monitor_left, monitor_top, x, y)
                    try:
                        import pyautogui

                        # pequeña espera para asegurar que la ventana está activa
                        time.sleep(0.5)
                        pyautogui.click(x, y)
                        logger.info("Click realizado en 'Start Client' (%d,%d).", x, y)
                    except Exception:
                        logger.exception("Error al hacer click en 'Start Client'.")

                found = True
                break
            # Si OCR no encontró, intentar template-matching con la imagen de referencia
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
                        logger.debug("Template match val=%.3f loc=%s", max_val, max_loc)
                        if max_val >= _settings.template_match_threshold:
                            th, tw = tpl.shape[:2]
                            cx = monitor_left + max_loc[0] + tw // 2
                            cy = monitor_top + max_loc[1] + th // 2
                            try:
                                import pyautogui
                                time.sleep(0.2)
                                pyautogui.click(cx, cy)
                                logger.info("Template fallback: click realizado en (%d,%d) (val=%.3f).", cx, cy, max_val)
                                found = True
                                break
                            except Exception:
                                logger.exception("Template fallback: error al hacer click.")
                        else:
                            try:
                                # guardar frame diagnóstico para inspección
                                from src.config.settings import settings as _s
                                diag_dir = _s.reports_dir / "screenshots" / "diagnostics"
                                diag_dir.mkdir(parents=True, exist_ok=True)
                                stamp = __import__("time").strftime("%Y%m%d_%H%M%S")
                                fname = diag_dir / f"template_lowval_{stamp}_{int(max_val*1000)}.png"
                                cv2.imwrite(str(fname), frame)
                                logger.info("Guardado frame diagnóstico por baja coincidencia template: %s (val=%.3f)", fname, max_val)
                            except Exception:
                                logger.exception("No se pudo guardar frame diagnóstico.")
            except Exception:
                logger.exception("Template fallback: error durante matching.")
        except Exception:
            logger.exception("Error durante detección de 'Start Client'.")

        time.sleep(poll_interval)

    if not found:
        logger.warning("No se detectó 'Start Client' tras esperar %ds", timeout)

    try:
        yield
    finally:
        try:  # intentar cerrar el juego al finalizar la sesión de tests
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                logger.exception("No se pudo terminar el proceso del launcher.")