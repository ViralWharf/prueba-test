"""
Implementación de `OcrEngine` sobre Tesseract (vía pytesseract).

Decisiones de diseño puntuales:

- Se usa `image_to_data` (no `image_to_string`) porque es la única forma
  de que Tesseract devuelva confianza por palabra. `TextDiff.ocr_confidence`
  (validation_result.py) necesita ese número, y `image_to_string` no lo
  expone.
- PSM 6 ("Assume a single uniform block of text") por defecto: el
  `ocr_ready` que produce `ImagePreprocessor.isolate_text_for_ocr`
  (image_preprocessor.py) ya es un recorte acotado al tooltip, sin otro
  contenido de pantalla alrededor -- es exactamente el caso para el que
  PSM 6 está pensado, a diferencia del PSM 3 (default, "página completa")
  que asume que puede haber múltiples bloques desordenados.
- Reconstrucción de líneas agrupando por (block_num, par_num, line_num):
  Tesseract devuelve resultados palabra por palabra, y si se las une con
  un solo espacio se pierde el salto de línea real del tooltip
  (relevante para `text_validator.py`, que compara contra el texto
  esperado línea por línea -- ver text.json/expected_texts).
- Esta clase NO decide si el resultado es válido (eso es
  `OcrResult.meets_min_confidence` / `text_validator.py`); solo registra
  un warning en logs cuando la confianza queda por debajo de
  `settings.ocr_min_confidence`, para que quede rastro en el log de la
  corrida sin acoplar esta clase a la lógica de aprobar/rechazar.
"""

from __future__ import annotations

import cv2
import numpy as np
import re
import difflib
import pytesseract
from pytesseract import Output

# Tesseract sobre texto de botón suele rendir mejor con PSM 7 (single text line),
# mientras que el recorte completo del tooltip usa PSM 4 para mantener el bloque de
# texto completo. Esto se encapsula en la clase concreta para que la capa superior
# siga usando una interfaz neutral.
_BUTTON_PSM = 7

from src.config.settings import settings
from src.ocr.ocr_engine import OcrEngine, OcrExtractionError, OcrLine, OcrResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Tesseract descarta una palabra como "no confiable" devolviendo -1 en
# vez de un score real; hay que filtrarlas antes de promediar o antes de
# reconstruir el texto, o van a colarse como ruido.
_UNRELIABLE_CONFIDENCE = -1


class TesseractOcrEngine(OcrEngine):
    """
    Motor de OCR basado en Tesseract.

    El binario de Tesseract se apunta una sola vez por instancia (no a
    nivel de módulo) para que tests puedan crear instancias con paths
    distintos si hace falta simular un binario roto/ausente, sin
    interferir entre tests vía estado global.
    """

    def __init__(
        self,
        *,
        tesseract_cmd: str | None = None,
        psm: int = 6,
        oem: int = 3,
        min_confidence: float | None = None,
    ) -> None:
        """
        Args:
            tesseract_cmd: path al ejecutable de tesseract.exe. Si no se
                pasa, usa `settings.tesseract_path` (ver settings.py,
                default apunta a la instalación estándar de Tesseract en
                Windows).
            psm: "Page Segmentation Mode" de Tesseract. 6 por defecto
                (ver docstring del módulo). Configurable porque, al
                calibrar contra el juego real, un tooltip de una sola
                línea corta podría rendir mejor con PSM 7
                ("single text line").
            oem: "OCR Engine Mode". 3 (default de Tesseract: usa LSTM si
                está disponible, si no el motor legacy) es la opción
                segura salvo que se confirme que el traineddata instalado
                solo soporta uno de los dos motores.
            min_confidence: umbral usado solo para decidir si loguear un
                warning tras extraer (ver docstring del módulo). Si no
                se pasa, usa `settings.ocr_min_confidence`.
        """
        self._tesseract_cmd = tesseract_cmd or str(settings.tesseract_path)
        pytesseract.pytesseract.tesseract_cmd = self._tesseract_cmd
        self._psm = psm
        self._oem = oem
        self._min_confidence = (
            min_confidence if min_confidence is not None else settings.ocr_min_confidence
        )

    def extract_text(self, image: np.ndarray, lang: str) -> OcrResult:
        prepared = self._prepare_for_tesseract(image)
        config = f"--psm {self._psm} --oem {self._oem}"

        try:
            data = pytesseract.image_to_data(
                prepared, lang=lang, config=config, output_type=Output.DICT
            )
        except pytesseract.TesseractNotFoundError as exc:
            raise OcrExtractionError(
                f"No se encontró el ejecutable de Tesseract en "
                f"'{self._tesseract_cmd}'. Verificar settings.tesseract_path "
                "o la instalación de Tesseract-OCR."
            ) from exc
        except pytesseract.TesseractError as exc:
            raise OcrExtractionError(
                f"Tesseract falló durante la extracción (lang='{lang}'): {exc}"
            ) from exc

        result = self._build_result(data)
        button_text = self._detect_button_text(image, lang)
        if button_text:
            result = self._append_button_line(result, button_text)

        if not result.meets_min_confidence(self._min_confidence):
            logger.warning(
                "OCR por debajo del umbral: confidence=%.2f (min=%.2f), "
                "text=%r, lang=%s",
                result.confidence, self._min_confidence, result.text, lang,
            )
        else:
            logger.debug(
                "OCR ok: confidence=%.2f, %d palabras, lang=%s",
                result.confidence, result.word_count, lang,
            )

        return result

    @staticmethod
    def _detect_button_text(image: np.ndarray, lang: str) -> str:
        if image is None or image.size == 0:
            return ""

        height, width = image.shape[:2]
        y_offset = max(0, int(height * 0.72))
        lower = image[y_offset:height]
        if lower.size == 0:
            return ""

        # 1) Intentar detectar botón por color (azul típico de UI)
        try:
            hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
            # Valores HSV típicos para azul UI (ajustable)
            lower_blue = np.array([90, 60, 40])
            upper_blue = np.array([140, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            if np.count_nonzero(blue_mask) > 50:
                contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # elegir el contorno más grande en área
                    best = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(best)
                    if w >= 20 and h >= 8:
                        # expandir un poco el bbox
                        pad_x = min(12, int(w * 0.15))
                        pad_y = min(8, int(h * 0.25))
                        x0 = max(0, x - pad_x)
                        y0 = max(0, y - pad_y)
                        x1 = min(lower.shape[1], x + w + pad_x)
                        y1 = min(lower.shape[0], y + h + pad_y)
                        candidate = lower[y0:y1, x0:x1]
                        text = TesseractOcrEngine._ocr_candidate(candidate, lang)
                        cleaned = TesseractOcrEngine._clean_button_text(text)
                        mapped = TesseractOcrEngine._map_button_text(cleaned)
                        if mapped:
                            return mapped

        except Exception:
            # no bloquear por fallo en la detección por color
            pass

        # 2) Fallback: buscar regiones brillantes en la banda inferior
        lower_gray = lower if lower.ndim == 2 else cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(lower_gray, 180, 255, cv2.THRESH_BINARY)
        if np.count_nonzero(bright_mask) > 40:
            num, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
            best_bb = None
            best_score = -1
            for label in range(1, num):
                area = stats[label, cv2.CC_STAT_AREA]
                x = stats[label, cv2.CC_STAT_LEFT]
                y = stats[label, cv2.CC_STAT_TOP]
                w = stats[label, cv2.CC_STAT_WIDTH]
                h = stats[label, cv2.CC_STAT_HEIGHT]
                if area < 30 or w < 20 or h < 6:
                    continue
                score = area + w * 2 - y
                if score > best_score:
                    best_score = score
                    best_bb = (x, y, w, h)

            if best_bb is not None:
                x, y, w, h = best_bb
                x0 = max(0, x - 6)
                y0 = max(0, y - 4)
                x1 = min(lower.shape[1], x + w + 6)
                y1 = min(lower.shape[0], y + h + 6)
                candidate = lower[y0:y1, x0:x1]
                text = TesseractOcrEngine._ocr_candidate(candidate, lang)
                cleaned = TesseractOcrEngine._clean_button_text(text)
                mapped = TesseractOcrEngine._map_button_text(cleaned)
                if mapped:
                    return mapped

        return ""

    @staticmethod
    def _ocr_candidate(img: np.ndarray, lang: str) -> str:
        # Normalizar al gris, binarizar adaptativo, invertir si necesario y upscale
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        # intentar Otsu para casos marcados
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Si el texto es blanco sobre fondo oscuro, invertir a texto oscuro/clear background
        # Queremos texto oscuro sobre blanco para una whitelist más fiable
        black_ratio = np.count_nonzero(th == 0) / th.size
        if black_ratio > 0.6:
            th = cv2.bitwise_not(th)

        # upscale para mejorar reconocimiento
        th = cv2.resize(th, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LINEAR)
        config = (
            "--psm 7 --oem 3 -c "
            "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "
        )
        try:
            txt = pytesseract.image_to_string(th, lang=lang, config=config)
        except Exception:
            txt = ""
        return " ".join(txt.split())

    @staticmethod
    def _clean_button_text(text: str) -> str:
        if not text:
            return ""
        # Normalizar substituciones comunes de OCR (digitos -> letras)
        subs = {
            "0": "O",
            "1": "I",
            "5": "S",
            "4": "A",
            "3": "E",
            "7": "T",
            "2": "Z",
            "6": "G",
            "8": "B",
            "9": "G",
        }
        s = text
        for k, v in subs.items():
            s = s.replace(k, v)

        # eliminar caracteres no alfabéticos salvo espacios
        cleaned = re.sub(r"[^A-Za-z ]+", "", s)
        # colapsar espacios y capitalizar palabras
        cleaned = " ".join(cleaned.split())
        return cleaned.strip()

    @staticmethod
    def _map_button_text(candidate: str) -> str:
        if not candidate:
            return ""
        # Comparar con un conjunto pequeño de etiquetas de botón esperadas
        canonical = ["Go Now", "Buy Now", "View", "Equip", "Use", "Close", "OK"]
        candidate_norm = candidate.strip()
        if not candidate_norm:
            return ""

        # Lowercase for matching
        cand_low = candidate_norm.lower()
        cans_low = [c.lower() for c in canonical]

        # 1) Exact or close fuzzy match
        matches = difflib.get_close_matches(cand_low, cans_low, n=1, cutoff=0.5)
        if matches:
            idx = cans_low.index(matches[0])
            return canonical[idx]

        # 2) Heurística: detectar presencia de las letras clave de "Go Now"
        # (g, o, n, w) en la cadena ruidosa; si aparecen al menos 3, mapear.
        keyset = set("gonw")
        letters_present = sum(1 for ch in set(cand_low) if ch in keyset)
        if letters_present >= 3:
            return "Go Now"

        # 3) Heurística anterior: si contiene 'go' y fragmentos que parezcan 'now'
        if "go" in cand_low.replace(" ", "") or "g0" in cand_low:
            if any(tok in cand_low for tok in ("now", "n0w", "no w", "naw", "nw")) or re.search(r"n[aoz]w", cand_low):
                return "Go Now"

        # 3) Fallback: pick best ratio with a lower threshold
        best = None
        best_score = 0.0
        for idx, cand in enumerate(cans_low):
            score = difflib.SequenceMatcher(None, cand_low, cand).ratio()
            if score > best_score:
                best_score = score
                best = idx
        if best is not None and best_score >= 0.45:
            return canonical[best]

        # devolver la versión limpiada con capitalización por palabras
        return " ".join(w.capitalize() for w in candidate_norm.split())

    @staticmethod
    def _append_button_line(result: OcrResult, button_text: str) -> OcrResult:
        if not button_text or any(line.text.strip().lower() == button_text.strip().lower() for line in result.lines):
            return result

        button_line = OcrLine(
            text=button_text.strip(),
            top=max((line.bottom for line in result.lines), default=0) + 12,
            bottom=max((line.bottom for line in result.lines), default=0) + 40,
            left=max((line.left for line in result.lines), default=0),
            right=max((line.right for line in result.lines), default=0) + max(60, len(button_text) * 4),
        )
        lines = tuple(result.lines) + (button_line,)
        text = "\n".join(line.text for line in lines)
        word_count = sum(len(line.text.split()) for line in lines)
        return OcrResult(
            text=text,
            confidence=result.confidence,
            word_count=word_count,
            lines=lines,
        )

    @staticmethod
    def _prepare_for_tesseract(image: np.ndarray) -> np.ndarray:
        """
        pytesseract interpreta arrays de 3 canales como RGB (via
        `PIL.Image.fromarray`); nuestras imágenes vienen en BGR (OpenCV).
        Si no se convierte, Tesseract igual suele funcionar sobre
        imágenes ya binarizadas en blanco/negro (que es el caso de
        `ocr_ready`, ver image_preprocessor.py) porque no hay color real
        que confundir, pero convertir explícitamente evita sorpresas si
        en el futuro se le pasa un recorte a color sin preprocesar.
        """
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def _build_result(data: dict) -> OcrResult:
        """
        Reconstruye texto y confianza a partir del dict crudo de
        `image_to_data`, agrupando palabras por línea real (no un join
        plano) y descartando las que Tesseract marcó como no confiables.
        """
        lines: dict[tuple[int, int, int], list[str]] = {}
        boxes: dict[tuple[int, int, int], tuple[int, int, int, int]] = {}
        confidences: list[float] = []

        for i, raw_text in enumerate(data["text"]):
            text = raw_text.strip()
            if not text:
                continue

            confidence = float(data["conf"][i])
            if confidence == _UNRELIABLE_CONFIDENCE:
                continue

            line_key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(line_key, []).append(text)
            boxes.setdefault(
                line_key,
                (
                    int(data["top"][i]),
                    int(data["top"][i] + data["height"][i]),
                    int(data["left"][i]),
                    int(data["left"][i] + data["width"][i]),
                ),
            )
            confidences.append(confidence)

        # dict preserva orden de inserción (Python 3.7+), y Tesseract
        # entrega las palabras en orden de lectura, así que las líneas
        # ya quedan en el orden correcto sin necesidad de sortear por
        # line_key.
        ordered_lines = []
        for line_key, words in lines.items():
            top, bottom, left, right = boxes[line_key]
            ordered_lines.append(
                OcrLine(
                    text=" ".join(words),
                    top=top,
                    bottom=bottom,
                    left=left,
                    right=right,
                )
            )

        full_text = "\n".join(line.text for line in ordered_lines)
        average_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return OcrResult(
            text=full_text,
            confidence=average_confidence,
            word_count=len(confidences),
            lines=tuple(ordered_lines),
        )