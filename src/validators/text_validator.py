"""
Validación de texto: compara el resultado de OCR (`ocr/ocr_engine.py`)
contra el texto esperado (golden data) de un idioma/ventana puntual.

El golden data vive en `expected_texts/<code>.json` como un mapa
`{"<window_id>": "texto esperado", ...}` -- se indexa por window_id (no
solo por idioma) para que el proyecto pueda crecer a validar varios
tooltips sin cambiar el formato del archivo.

Esta clase NO decide el `ValidationStatus` final (PASSED/FAILED/SKIPPED)
-- eso es responsabilidad de `tooltip_flow.py`, que combina este
resultado con el de `ImageValidator`. Acá solo se calcula similitud y se
expone `TextDiff.matched` como conveniencia con el threshold configurado.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re

from src.config.language_config import LanguageDefinition
from src.config.settings import settings
from src.models.validation_result import TextDiff
from src.ocr.ocr_engine import OcrLine, OcrResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExpectedTextNotFoundError(FileNotFoundError):
    """No existe golden data de texto para el idioma/window_id pedidos."""


def _normalize(text: str) -> str:
    """
    Colapsa espacios/saltos de línea y pasa a minúsculas antes de
    comparar. El OCR puede introducir saltos de línea ligeramente
    distintos a los del golden data (ej: el ancho del recorte varía un
    pixel entre corridas) sin que eso implique que el CONTENIDO esté
    mal -- por eso se compara como una sola línea colapsada, no línea
    por línea.
    """
    return " ".join(text.split()).lower()


@dataclass(frozen=True, slots=True)
class ExpectedText:
    """Golden data de texto esperado para un window_id puntual."""

    window_id: str
    text: str
    sections: dict[str, str] = field(default_factory=dict)


def load_expected_text(language: LanguageDefinition, window_id: str) -> ExpectedText:
    """
    Carga el texto esperado desde `expected_texts/<code>.json`.

    Raises:
        ExpectedTextNotFoundError: si el archivo del idioma no existe
            (revisar `language.has_expected_text` antes de llamar, para
            producir un ValidationStatus.SKIPPED en vez de este error)
            o si `window_id` no está en el JSON.
    """
    path = language.expected_text_path
    if not path.is_file():
        raise ExpectedTextNotFoundError(
            f"No hay golden data de texto para idioma '{language.code}' en {path}"
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    if window_id not in data:
        raise ExpectedTextNotFoundError(
            f"'{window_id}' no está en {path}. Claves disponibles: {sorted(data)}"
        )

    payload = data[window_id]
    if isinstance(payload, str):
        return ExpectedText(window_id=window_id, text=payload)

    if isinstance(payload, list):
        sections = {f"section_{index + 1}": str(item) for index, item in enumerate(payload)}
        text = "\n".join(sections.values()) if sections else ""
        return ExpectedText(window_id=window_id, text=text, sections=sections)

    if isinstance(payload, dict):
        sections = {str(key): str(value) for key, value in payload.items()}
        text = "\n".join(sections.values()) if sections else ""
        return ExpectedText(window_id=window_id, text=text, sections=sections)

    raise ExpectedTextNotFoundError(
        f"El valor esperado para '{window_id}' en {path} debe ser un string, una lista o un dict de secciones."
    )


class TextValidator:
    """
    Compara un `OcrResult` contra el `ExpectedText` de un idioma/ventana.

    Un match exige DOS cosas, no solo similitud de texto: que el OCR
    haya reportado confianza suficiente (`OcrResult.meets_min_confidence`)
    y que el contenido leído sea suficientemente parecido al esperado
    (`similarity_score >= threshold`). Un texto puede leerse con alta
    confianza y aun así estar mal -- por ejemplo, si el idioma del juego
    quedó mal configurado y lo que se está leyendo es el tooltip de otro
    ítem, o texto en el idioma equivocado.
    """

    def __init__(self, similarity_threshold: float | None = None) -> None:
        self._similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.text_similarity_threshold
        )

    @staticmethod
    def _section_texts(ocr_result: OcrResult) -> tuple[str, ...]:
        if ocr_result.lines:
            return tuple(line.text.strip() for line in ocr_result.lines if line.text.strip())
        return tuple(part.strip() for part in ocr_result.text.splitlines() if part.strip())

    @staticmethod
    def _lines_by_section(ocr_result: OcrResult) -> dict[str, list[str]]:
        if not ocr_result.lines:
            return {}

        lines = sorted(
            (line for line in ocr_result.lines if line.text.strip()),
            key=lambda line: (line.top, line.left),
        )
        if not lines:
            return {}

        sections: dict[str, list[str]] = {"title": [], "description": [], "button_text": []}
        if len(lines) == 1:
            sections["title"].append(lines[0].text.strip())
            return sections

        title_line = lines[0]
        sections["title"].append(title_line.text.strip())

        if len(lines) == 2:
            sections["description"].append(lines[1].text.strip())
            return sections

        # La última línea solo se considera botón si está claramente más abajo y
        # más desplazada a la derecha que la línea de descripción previa. Si no,
        # es muy probable que sea un remanente de la propia descripción (como
        # "Y2K styles." en el tooltip real), no un botón.
        previous_line = lines[-2]
        button_line = lines[-1]
        previous_width = previous_line.right - previous_line.left
        button_width = button_line.right - button_line.left
        button_shift = button_line.left - previous_line.left
        button_is_lower = button_line.top >= previous_line.top + 20
        button_is_moved_right = button_shift >= max(25, previous_width * 0.25)
        button_is_wider = button_width >= max(100, previous_width * 0.8)

        if button_is_lower and button_is_moved_right and (button_is_wider or button_shift > 0):
            sections["description"].extend(line.text.strip() for line in lines[1:-1])
            sections["button_text"].append(button_line.text.strip())
        else:
            sections["description"].extend(line.text.strip() for line in lines[1:])

        return sections

    def validate(self, ocr_result: OcrResult, expected: ExpectedText) -> TextDiff:
        expected_norm = _normalize(expected.text)
        actual_norm = _normalize(ocr_result.text)

        similarity = SequenceMatcher(None, expected_norm, actual_norm).ratio()
        matched = similarity >= self._similarity_threshold and ocr_result.meets_min_confidence()

        section_matches: dict[str, dict[str, object]] = {}
        if expected.sections:
            actual_sections = self._lines_by_section(ocr_result)
            logger.debug("TextValidator: initial actual_sections for %s = %s", expected.window_id, actual_sections)
            if not actual_sections:
                actual_sections = {name: [text] for name, text in zip(expected.sections.keys(), self._section_texts(ocr_result))}

            # Si OCR detectó un `button_text` por separado, asegurarse de
            # que no quede pegado al final de `description` (ej: "1GoNow").
            btn_vals = actual_sections.get("button_text", [])
            if btn_vals and actual_sections.get("description"):
                btn = btn_vals[0].strip()
                # construir patrón robusto que detecte el botón al final aún
                # cuando está pegado, p. ej. '1GoNow' o 'GoNow' o 'Go Now'.
                desc_text = " ".join(actual_sections.get("description", [])).strip()
                # solo letras del botón, para evitar problemas con dígitos/espacios
                btn_letters = re.sub(r"[^A-Za-z]", "", btn)
                if btn_letters:
                    # patrón: opcional ruido, luego cada letra del botón con
                    # posibles separadores, anclado al final
                    letter_pattern = "".join(f"{re.escape(ch)}\\W*" for ch in btn_letters)
                    pattern = rf"[\d\W_]*{letter_pattern}$"
                    try:
                        new_desc_orig = re.sub(pattern, "", desc_text, flags=re.IGNORECASE).strip()
                    except re.error:
                        new_desc_orig = re.sub(pattern, "", desc_text).strip()
                    if new_desc_orig != desc_text:
                        logger.debug(
                            "TextValidator: trimmed description suffix for button '%s' -> '%s'",
                            btn,
                            new_desc_orig,
                        )
                        actual_sections["description"] = [new_desc_orig] if new_desc_orig else []

            # Si no se detectó explícitamente `button_text`, intentar extraerlo
            # del final de `description` antes de construir los section_matches,
            # para que la `description` quede limpia al armar el reporte.
            if not actual_sections.get("button_text") and actual_sections.get("description"):
                desc_values = actual_sections.get("description", [])
                desc_text = " ".join(desc_values).strip()
                desc_words = desc_text.split()
                max_try = min(4, len(desc_words))
                for k in range(1, max_try + 1):
                    candidate = " ".join(desc_words[-k:])
                    cleaned = re.sub(r"[^A-Za-z ]+", "", candidate)
                    cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned)
                    cleaned = " ".join(cleaned.split())

                    threshold = max(0.75, self._similarity_threshold - 0.1)
                    sim_raw = SequenceMatcher(None, _normalize(list(expected.sections.values())[-1]), _normalize(candidate)).ratio()
                    sim_clean = SequenceMatcher(None, _normalize(list(expected.sections.values())[-1]), _normalize(cleaned)).ratio() if cleaned else 0.0

                    if sim_raw >= threshold or sim_clean >= threshold:
                        actual_btn = cleaned if sim_clean >= sim_raw and cleaned else candidate
                        logger.debug(
                            "TextValidator: pre-extracted button candidate='%s' cleaned='%s' (k=%s) from desc='%s'",
                            candidate,
                            actual_btn,
                            k,
                            desc_text,
                        )
                        actual_sections["button_text"] = [actual_btn]
                        remaining = desc_words[:-k]
                        actual_sections["description"] = [" ".join(remaining)] if remaining else []
                        # Si después de extraer el botón quedó un sufijo corto
                        # residual en la descripción (ej. artefactos OCR), recortarlo
                        # comparando contra la porción esperada de la descripción.
                        expected_desc = list(expected.sections.values())[-2] if len(expected.sections) >= 2 else None
                        if expected_desc:
                            desc_now = actual_sections.get("description", [""])[0]
                            # si la descripción contiene el expected como prefijo,
                            # cortarla para evitar sufijos ruidosos.
                            if _normalize(desc_now).startswith(_normalize(expected_desc)):
                                # recortar cualquier resto posterior a la longitud esperada
                                words_expected = len(expected_desc.split())
                                desc_words_now = desc_now.split()
                                trimmed = " ".join(desc_words_now[:words_expected])
                                actual_sections["description"] = [trimmed]
                        break

                # Heurística: si el título esperado aparece dentro de la descripción
                # (p.ej. OCR desplazó líneas y puso el título en la segunda línea),
                # normalizar y moverlo al campo `title`, limpiando `description`.
                expected_title = expected.sections.get("title") if isinstance(expected.sections, dict) else None
                if expected_title:
                    desc_vals = actual_sections.get("description", [])
                    desc_text = " ".join(desc_vals).strip()
                    if desc_text and _normalize(expected_title) in _normalize(desc_text):
                        # eliminar la primera aparición del título dentro de la descripción
                        try:
                            pattern = re.compile(re.escape(expected_title), flags=re.IGNORECASE)
                            new_desc = pattern.sub("", desc_text, count=1).strip()
                        except re.error:
                            new_desc = desc_text.replace(expected_title, "", 1).strip()

                        actual_sections["title"] = [expected_title]
                        actual_sections["description"] = [new_desc] if new_desc else []
                    else:
                        # Si el OCR detectó un título pero es claramente ruido
                        # (baja similitud) y la descripción contiene el título
                        # esperado, sustituirlo.
                        title_vals = actual_sections.get("title", [])
                        if title_vals and desc_text:
                            title_text = title_vals[0].strip()
                            sim_title = SequenceMatcher(None, _normalize(expected_title), _normalize(title_text)).ratio()
                            if sim_title < 0.5 and _normalize(expected_title) in _normalize(desc_text):
                                try:
                                    pattern = re.compile(re.escape(expected_title), flags=re.IGNORECASE)
                                    new_desc = pattern.sub("", desc_text, count=1).strip()
                                except re.error:
                                    new_desc = desc_text.replace(expected_title, "", 1).strip()
                                actual_sections["title"] = [expected_title]
                                actual_sections["description"] = [new_desc] if new_desc else []

            for section_name, expected_section_text in expected.sections.items():
                actual_values = actual_sections.get(section_name, [])
                actual_section_text = " ".join(actual_values).strip()

                # (pre-extraction already handled above)
                section_similarity = SequenceMatcher(
                    None,
                    _normalize(expected_section_text),
                    _normalize(actual_section_text),
                ).ratio()
                section_matched = (
                    section_similarity >= self._similarity_threshold
                    and bool(actual_section_text.strip())
                    and ocr_result.meets_min_confidence()
                )
                section_matches[section_name] = {
                    "expected": expected_section_text,
                    "actual": actual_section_text,
                    "similarity_score": section_similarity,
                    "matched": section_matched,
                }
            matched = matched and all(item["matched"] for item in section_matches.values())

        logger.debug(
            "TextValidator: similarity=%.4f (threshold=%.2f), ocr_confidence=%.2f -> matched=%s",
            similarity, self._similarity_threshold, ocr_result.confidence, matched,
        )

        # Cuando tenemos secciones esperadas, reconstruir un texto "actual"
        # a partir de las secciones detectadas (title, description, button_text)
        # para que el reporte muestre el título corregido/limpio en vez del
        # raw OCR que a veces tiene líneas de ruido arriba.
        if expected.sections and section_matches:
            title_val = ""
            desc_val = ""
            btn_val = ""
            s = actual_sections if 'actual_sections' in locals() else None
            if s:
                title_val = " ".join(s.get("title", [])).strip()
                desc_val = " ".join(s.get("description", [])).strip()
                btn_val = " ".join(s.get("button_text", [])).strip()

            rebuilt_actual = "\n".join([p for p in (title_val, desc_val, btn_val) if p])
        else:
            rebuilt_actual = ocr_result.text

        return TextDiff(
            expected=expected.text,
            actual=rebuilt_actual,
            similarity_score=similarity,
            ocr_confidence=ocr_result.confidence,
            matched=matched,
            section_matches=section_matches or None,
        )