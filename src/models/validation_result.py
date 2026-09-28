"""
Modelos de resultado de validación.

Estas dataclasses son el contrato entre las capas de validación
(text_validator, image_validator) y las capas de orquestación/reporting
(flows, reports). No contienen lógica de negocio, solo estructura,
invariantes básicas y serialización a dict/JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ValidationStatus(str, Enum):
    """Resultado posible de una validación individual."""

    PASSED = "passed"
    FAILED = "failed"
    # Fallo técnico: no se pudo ni comparar (OCR crasheó, la ventana nunca
    # apareció, timeout de hover, etc). Distinto de FAILED, que implica que
    # la comparación sí se ejecutó y el contenido no coincidió.
    ERROR = "error"
    # Validación deliberadamente omitida (ej: no hay imagen de referencia
    # todavía para ese idioma).
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class Evidence:
    """Rastro visual de una corrida: qué se capturó y dónde quedó guardado."""

    screenshot_path: Path
    cropped_region_path: Path | None = None
    diff_image_path: Path | None = None  # heatmap/overlay de diferencias
    captured_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "screenshot_path": str(self.screenshot_path),
            "cropped_region_path": (
                str(self.cropped_region_path) if self.cropped_region_path else None
            ),
            "diff_image_path": (
                str(self.diff_image_path) if self.diff_image_path else None
            ),
            "captured_at": self.captured_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class TextDiff:
    """Resultado de comparar el texto extraído por OCR contra el esperado."""

    expected: str
    actual: str
    similarity_score: float  # 0.0 - 1.0
    ocr_confidence: float    # Confianza promedio reportada por Tesseract (0-100)
    matched: bool
    section_matches: dict[str, dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected": self.expected,
            "actual": self.actual,
            "similarity_score": round(self.similarity_score, 4),
            "ocr_confidence": round(self.ocr_confidence, 2),
            "matched": self.matched,
            "section_matches": self.section_matches,
        }


@dataclass(frozen=True, slots=True)
class ImageDiff:
    """Resultado de comparar la imagen extraída contra la imagen golden."""

    ssim_score: float  # 0.0 - 1.0
    threshold: float
    matched: bool
    reference_image_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "ssim_score": round(self.ssim_score, 4),
            "threshold": self.threshold,
            "matched": self.matched,
            "reference_image_path": str(self.reference_image_path),
        }


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """
    Resultado completo de validar un tooltip/ventana para un idioma puntual.

    Es el objeto que produce `tooltip_flow.py` y que consume el reporter
    para generar el reporte final (HTML/JSON) en `reports/`.
    """

    language: str
    # Identificador lógico de la ventana/tooltip validado, ej: "item_sword_tooltip".
    # No es el idioma ni la posición: es el "qué" se está validando, útil cuando
    # el proyecto crezca a validar múltiples tooltips distintos.
    window_id: str
    status: ValidationStatus
    evidence: Evidence
    text_diff: TextDiff | None = None
    image_diff: ImageDiff | None = None
    error_message: str | None = None
    duration_seconds: float = 0.0
    executed_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if self.status is ValidationStatus.ERROR and not self.error_message:
            raise ValueError(
                "Un ValidationResult con status=ERROR debe incluir error_message."
            )

    @property
    def passed(self) -> bool:
        return self.status is ValidationStatus.PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "window_id": self.window_id,
            "status": self.status.value,
            "evidence": self.evidence.to_dict(),
            "text_diff": self.text_diff.to_dict() if self.text_diff else None,
            "image_diff": self.image_diff.to_dict() if self.image_diff else None,
            "error_message": self.error_message,
            "duration_seconds": round(self.duration_seconds, 3),
            "executed_at": self.executed_at.isoformat(),
        }


@dataclass(slots=True)
class ValidationReport:
    """
    Agrupa múltiples ValidationResult (ej: todos los idiomas de una corrida)
    para calcular un resumen y serializarlo a `reports/`.

    A diferencia de ValidationResult, esta clase NO es frozen: se va
    construyendo incrementalmente a medida que el flow procesa cada idioma.
    """

    results: list[ValidationResult] = field(default_factory=list)

    def add(self, result: ValidationResult) -> None:
        self.results.append(result)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.status is ValidationStatus.PASSED)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if r.status is ValidationStatus.FAILED)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.status is ValidationStatus.ERROR)

    @property
    def skipped_count(self) -> int:
        return sum(1 for r in self.results if r.status is ValidationStatus.SKIPPED)

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed_count / self.total

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "total": self.total,
                "passed": self.passed_count,
                "failed": self.failed_count,
                "error": self.error_count,
                "skipped": self.skipped_count,
                "success_rate": round(self.success_rate, 4),
            },
            "results": [r.to_dict() for r in self.results],
        }