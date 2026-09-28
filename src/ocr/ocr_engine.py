"""
Interfaz abstracta para motores de OCR.

Por qué una ABC en vez de llamar a pytesseract directo desde
`text_validator.py`:

1. Hoy usamos Tesseract, pero es un motor con precisión limitada en
   texto pequeño/estilizado como el de UI de videojuegos. No tenemos
   acceso al código fuente del juego para calibrar de antemano qué tan
   bien va a funcionar contra la fuente real -- si al probar contra el
   juego la precisión no alcanza, el reemplazo natural es un motor
   basado en un modelo (ej: EasyOCR, PaddleOCR, o incluso una llamada a
   un modelo multimodal) sin tener que tocar `text_validator.py` ni
   `tooltip_flow.py`, que solo conocen esta interfaz.
2. Permite testear `text_validator.py` con un `FakeOcrEngine` en tests
   unitarios, sin depender de que Tesseract esté instalado en la
   máquina que corre la suite (ver tests/unit).

Mismo patrón que `TextMaskStrategy` en `image_preprocessor.py`:
interfaz chica, una sola responsabilidad (texto crudo + confianza),
nada de decisiones de negocio acá (esas son de `text_validator.py`,
que decide si el `OcrResult` es aceptable comparándolo contra el texto
esperado y contra `settings.ocr_min_confidence`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.config.settings import settings


class OcrExtractionError(RuntimeError):
    """
    El motor de OCR no pudo ejecutar la extracción (binario no
    encontrado, idioma no instalado, imagen inválida, etc).

    Se distingue de "extrajo texto pero con baja confianza": ese caso
    NO es un error, es un `OcrResult` con `confidence` baja que
    `text_validator.py` decide cómo tratar (típicamente
    ValidationStatus.FAILED, no ERROR -- ver validation_result.py).
    """


@dataclass(frozen=True, slots=True)
class OcrLine:
    """Una línea OCR con su texto y su caja delimitadora en la imagen."""

    text: str
    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0


@dataclass(frozen=True, slots=True)
class OcrResult:
    """
    Salida normalizada de cualquier motor de OCR, independiente de la
    librería subyacente.

    `confidence` va de 0.0 a 100.0 para poder compararse directamente
    contra `settings.ocr_min_confidence` y volcarse tal cual en
    `TextDiff.ocr_confidence` (validation_result.py) sin conversiones
    adicionales en `text_validator.py`.
    """

    text: str
    confidence: float
    word_count: int
    lines: tuple[OcrLine, ...] = ()

    @property
    def is_empty(self) -> bool:
        """
        True si el motor no detectó ningún texto (recorte vacío, hover
        no llegó a tiempo, o el idioma configurado en Tesseract no
        coincide con el texto real de la ventana).
        """
        return self.word_count == 0 or not self.text.strip()

    def meets_min_confidence(self, min_confidence: float | None = None) -> bool:
        """
        Compara contra `settings.ocr_min_confidence` por defecto, o un
        umbral puntual si se pasa (ej: un idioma con fuente más chica
        que necesita relajar el umbral en un test específico).
        """
        threshold = (
            min_confidence if min_confidence is not None else settings.ocr_min_confidence
        )
        return not self.is_empty and self.confidence >= threshold


class OcrEngine(ABC):
    """
    Contrato mínimo que debe cumplir cualquier motor de OCR para poder
    conectarse a `text_validator.py`.

    Deliberadamente NO expone nada específico de Tesseract (como PSM o
    formatos de config string) en la firma: eso queda encapsulado en la
    implementación concreta (`TesseractOcrEngine`), para que cambiar de
    motor no filtre detalles de implementación hacia arriba.
    """

    @abstractmethod
    def extract_text(self, image: np.ndarray, lang: str) -> OcrResult:
        """
        Extrae texto de `image` (recorte ya preprocesado, típicamente
        el `ocr_ready` que devuelve `ImagePreprocessor.preprocess`, ver
        image_preprocessor.py) para el idioma `lang`.

        Args:
            image: recorte en escala de grises o BGR. Se asume que ya
                pasó por `ImagePreprocessor.isolate_text_for_ocr` (texto
                sobre fondo plano) -- este método no hace ningún
                preprocesamiento de imagen, solo OCR.
            lang: código de idioma en el formato que espere el motor
                concreto (para Tesseract, el
                `LanguageDefinition.tesseract_lang_code` de
                language_config.py, ej: "spa", "eng").

        Returns:
            OcrResult con el texto extraído y la confianza reportada.

        Raises:
            OcrExtractionError: si el motor no pudo ejecutar la
                extracción (no es lo mismo que "extrajo pero con baja
                confianza", ver docstring de OcrExtractionError).
        """
        raise NotImplementedError