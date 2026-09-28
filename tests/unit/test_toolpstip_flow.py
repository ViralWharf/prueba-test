"""
Configuración centralizada del proyecto.

Todos los paths son relativos a la raíz del proyecto, calculados
dinámicamente para que el proyecto sea portable entre máquinas.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Raíz del proyecto: game_ui_validator/
# settings.py está en src/config/settings.py -> subimos 2 niveles
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
    )

    # --- Paths ---
    tesseract_path: Path = Field(
        default=Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        description="Ruta al ejecutable de Tesseract OCR",
    )
    reports_dir: Path = Field(default=PROJECT_ROOT / "reports")
    test_data_dir: Path = Field(default=PROJECT_ROOT / "test_data")
    templates_dir: Path = Field(default=PROJECT_ROOT / "test_data" / "templates")
    reference_images_dir: Path = Field(
        default=PROJECT_ROOT / "test_data" / "reference_images"
    )
    expected_texts_dir: Path = Field(
        default=PROJECT_ROOT / "test_data" / "expected_texts"
    )
    launcher_dir: Path = Field(
        default=Path(r"C:\launcher"),
        description="Carpeta del launcher que se va a abrir y validar",
    )
    settings_fallback_dir: Path | None = Field(
        default=None,
        description="Carpeta alternativa donde buscar XML de configuración",
    )

    # --- Timeouts / esperas (en segundos) ---
    hover_settle_seconds: float = Field(
        default=1.2,
        description=(
            "Tiempo a esperar tras el hover antes de capturar, dado que "
            "el tooltip tarda ~1s en mostrarse completo. Se deja margen extra."
        ),
    )
    window_appear_timeout_seconds: float = Field(
        default=5.0,
        description="Timeout máximo esperando que aparezca una ventana/tooltip",
    )
    click_delay_seconds: float = Field(
        default=0.3,
        description="Delay entre clicks al recorrer posiciones de idioma",
    )
    next_page_click_delay_seconds: float = Field(
        default=0.15,
        description="Delay específico tras hacer click en el siguiente tooltip/page",
    )
    launcher_ready_timeout_seconds: float = Field(
        default=40.0,
        description="Tiempo máximo para esperar a que la UI del launcher esté lista",
    )
    launcher_ready_poll_interval_seconds: float = Field(
        default=1.0,
        description="Intervalo de polling para comprobar que la UI del launcher está lista",
    )
    launcher_click_start_client: bool = Field(
        default=True,
        description="Si es True, intenta detectar y hacer click en Start Client",
    )

    # --- Umbrales ---
    template_match_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confianza mínima para considerar un template match válido",
    )
    ssim_similarity_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Similitud mínima (SSIM) para considerar dos imágenes iguales",
    )
    ocr_min_confidence: float = Field(
        default=60.0,
        ge=0.0,
        le=100.0,
        description="Confianza mínima de Tesseract para aceptar texto extraído",
    )
    text_similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Similitud mínima (SequenceMatcher) entre el texto extraído por "
            "OCR y el texto esperado (golden data) para considerarlo un "
            "match. Separado de `ocr_min_confidence` a propósito: uno mide "
            "qué tan seguro está Tesseract de haber leído bien, el otro qué "
            "tan parecido es lo leído al contenido esperado. Un texto puede "
            "leerse con alta confianza y aun así estar mal (idioma "
            "incorrecto, texto de otro ítem)."
        ),
    )

    # --- Scroll de idiomas ---
    max_language_positions: int = Field(
        default=10,
        description="Cantidad máxima de posiciones a recorrer buscando la ventana",
    )

    # --- Debug ---
    debug_mode: bool = Field(
        default=False,
        description="Si es True, guarda capturas intermedias de cada paso",
    )


# Instancia única (patrón singleton simple) para importar en todo el proyecto
settings = Settings()