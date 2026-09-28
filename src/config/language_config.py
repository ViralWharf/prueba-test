"""
Configuración de idiomas soportados.

Decisión de diseño clave: el idioma de la corrida NO se detecta
automáticamente. Se pasa como parámetro al ejecutar el validador
(ej: `main.py --language en`), porque el idioma del juego se define
al lanzarlo, y eso está fuera de nuestro control/código fuente. Por
lo tanto este módulo NO necesita anchors para "adivinar" en qué
idioma está la ventana visible: ya se sabe de antemano.

Lo que SÍ sigue siendo cierto (y la razón de que exista
`language_scroll_flow.py`) es que cambiar de idioma cambia el ancho
del texto y por lo tanto el layout: la ventana/tooltip del ítem que
queremos validar puede terminar en una posición distinta dentro de la
pila de elementos superpuestos. Identificar "llegué a la ventana
correcta" es responsabilidad de `window_locator.py`, matcheando un
anchor propio del ÍTEM (ej: su ícono), que es igual sin importar el
idioma. Este módulo no se mete en esa lógica.

Otro punto importante: hoy solo existe texto esperado (golden data)
para inglés. `expected_texts_dir` puede no tener un JSON para todos
los idiomas que el juego soporta. Este módulo expone
`has_expected_text` para que validators/flows puedan degradar
correctamente (status=SKIPPED, ver validation_result.py) en vez de
romperse con un FileNotFoundError cuando falta el golden data de un
idioma.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config.settings import settings


class LanguageDefinition(BaseModel):
    """
    Metadata de un idioma soportado por el validador.

    Inmutable (frozen): es configuración estática. No incluye nada de
    "posición en el scroll" a propósito -- esa posición no es estable
    entre corridas (ver docstring del módulo) y se resuelve en runtime
    en window_locator.py / language_scroll_flow.py, no acá.
    """

    model_config = ConfigDict(frozen=True)

    code: str = Field(
        description=(
            "Código corto del idioma. Debe coincidir con el nombre de "
            "archivo en test_data/expected_texts/<code>.json (ej: 'es', "
            "'en'). Es la clave usada para buscar el idioma en todo el "
            "proyecto y la que se pasa por parámetro al ejecutar."
        ),
    )
    display_name: str = Field(
        description="Nombre legible para logs y reportes (ej: 'Español')."
    )
    tesseract_lang_code: str = Field(
        description=(
            "Código que Tesseract espera vía --lang (ej: 'spa', 'eng'). "
            "No siempre coincide con `code`, por eso es un campo aparte."
        ),
    )

    @field_validator("code", "tesseract_lang_code")
    @classmethod
    def _normalize_lower(cls, v: str) -> str:
        v = v.strip().lower()
        if not v:
            raise ValueError("no puede ser un string vacío")
        return v

    @property
    def locale_option_label(self) -> str:
        """Texto que aparece en el selector de idioma de Launcher."""
        mapping = {
            "en": "English (en-us)",
            "es": "Español (es-es)",
            "pt": "Português (pt-br)",
        }
        return mapping.get(self.code, self.display_name)

    @property
    def expected_text_path(self):
        """Ruta al JSON con los textos esperados de este idioma."""
        return settings.expected_texts_dir / f"{self.code}.json"

    @property
    def has_expected_text(self) -> bool:
        """
        True si ya existe golden data de texto para este idioma.

        Hoy debería dar True solo para 'en'. A medida que se genere
        expected_texts para otros idiomas, esto empieza a devolver
        True sin tocar código: solo agregando el JSON correspondiente.
        """
        return self.expected_text_path.is_file()


# ---------------------------------------------------------------------------
# Idiomas soportados por el proyecto (no necesariamente todos con golden
# data de texto todavía -- ver has_expected_text).
#
# NOTA: esta lista es un punto de partida con 'en' y 'es'. Convendría
# completarla con todos los idiomas que el juego realmente soporta,
# aunque todavía no tengan expected_texts/<code>.json -- eso permite
# que window_locator/tooltip_flow corran igual y generen evidencia
# visual (screenshots) con status=SKIPPED en la parte de texto, en vez
# de no tener ninguna entrada de idioma configurada.
# ---------------------------------------------------------------------------
LANGUAGES: tuple[LanguageDefinition, ...] = (
    LanguageDefinition(code="en", display_name="English", tesseract_lang_code="eng"),
    LanguageDefinition(code="es", display_name="Español", tesseract_lang_code="spa"),
    LanguageDefinition(code="pt", display_name="Português", tesseract_lang_code="por"),
)


def _validate_no_duplicate_codes(languages: tuple[LanguageDefinition, ...]) -> None:
    codes = [lang.code for lang in languages]
    duplicates = {c for c in codes if codes.count(c) > 1}
    if duplicates:
        raise ValueError(
            f"Códigos de idioma duplicados en LANGUAGES: {sorted(duplicates)}"
        )


_validate_no_duplicate_codes(LANGUAGES)

_LANGUAGES_BY_CODE: dict[str, LanguageDefinition] = {
    lang.code: lang for lang in LANGUAGES
}


def get_language(code: str) -> LanguageDefinition:
    """
    Busca la definición de un idioma por código (el que se pasa como
    parámetro al ejecutar el validador).

    Lanza KeyError con mensaje claro (incluyendo los códigos
    disponibles) en lugar de dejar propagar un KeyError críptico.
    """
    try:
        return _LANGUAGES_BY_CODE[code.strip().lower()]
    except KeyError as exc:
        available = ", ".join(sorted(_LANGUAGES_BY_CODE))
        raise KeyError(
            f"Idioma '{code}' no está configurado en language_config.py. "
            f"Idiomas disponibles: {available}"
        ) from exc


def iter_languages() -> tuple[LanguageDefinition, ...]:
    """Devuelve todos los idiomas soportados."""
    return LANGUAGES


def languages_with_expected_text() -> tuple[LanguageDefinition, ...]:
    """
    Subconjunto de idiomas que ya tienen golden data de texto.

    Útil para, por ejemplo, correr una suite de CI que solo valide
    texto donde hay con qué comparar, sin tocar la lista de idiomas
    soportados por el juego.
    """
    return tuple(lang for lang in LANGUAGES if lang.has_expected_text)


def max_scroll_attempts() -> int:
    """
    Límite de clicks a intentar antes de considerar que no se encontró
    la ventana del ítem buscado, para el idioma actual.

    Delegado a `settings.max_language_positions` para no duplicar la
    fuente de verdad de este número en dos módulos distintos.
    """
    return settings.max_language_positions