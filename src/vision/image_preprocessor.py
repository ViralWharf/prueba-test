"""
Separación de texto e imagen dentro del recorte del tooltip/ventana.

Problema que resuelve este módulo (ver README del proyecto): el texto del
tooltip está SOBREPUESTO al gráfico del ítem, no en un panel aparte, así
que el ícono puede tapar parcialmente al texto y viceversa. Esto rompe dos
cosas si se las hace directo sobre el recorte crudo:

1. OCR: Tesseract confunde bordes/detalles del ícono con caracteres, y
   pierde precisión si el texto no está sobre un fondo limpio y uniforme.
2. Comparación de imagen (SSIM contra el golden en `reference_images/`):
   el texto cambia de idioma a idioma (distinto ancho, distintos glifos),
   así que si se compara el recorte crudo, la validación de imagen fallaría
   por diferencias de TEXTO aunque el ÍCONO sea idéntico -- eso es un falso
   negativo, porque el texto ya se valida aparte con `text_validator.py`.

La solución es separar ambos canales antes de que sigan camino:
    - Un mask de "dónde hay texto" (`build_text_mask`).
    - A partir del mask: una versión "lista para OCR" (texto sobre fondo
      plano, sin el ícono estorbando) y una versión "lista para
      comparación" (ícono con el texto removido vía inpainting, para que
      el SSIM compare gráfico contra gráfico, no gráfico+texto contra
      gráfico+texto-en-otro-idioma).

No se asume de antemano el color exacto del texto del juego (no hay
acceso al código fuente para confirmarlo). Por eso la detección del mask
está detrás de una interfaz intercambiable (`TextMaskStrategy`): si al
calibrar contra el juego real la estrategia por defecto no separa bien,
se escribe una nueva implementación sin tocar `ImagePreprocessor` ni nada
río abajo (mismo patrón que `ocr_engine.py` con Tesseract).

IMPORTANTE sobre las imágenes de referencia (`test_data/reference_images/`):
para que la comparación de imagen tenga sentido, el golden data debería
generarse pasando también por `remove_text_via_inpaint` (o directamente
capturarse en un momento sin hover, si el juego lo permite). Si el golden
tiene texto "quemado" en la imagen, comparar contra un `comparison_ready`
sin texto va a bajar el SSIM artificialmente. Esto queda documentado acá
porque es una decisión que afecta a cómo se arma `test_data/`, no solo a
este módulo.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Estrategia de detección de texto (intercambiable)
# ---------------------------------------------------------------------------


class TextMaskStrategy(ABC):
    """
    Interfaz para "encontrar qué píxeles del recorte son texto".

    Se mantiene separada de `ImagePreprocessor` (que solo orquesta) para
    poder calibrar/reemplazar la heurística de detección sin tocar el
    resto del pipeline de vision/ una vez que se tengan capturas reales
    del juego para probar contra ellas.
    """

    @abstractmethod
    def build_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Devuelve una máscara binaria (uint8, 0 o 255) del mismo alto/ancho
        que `image`, con 255 en los píxeles que se consideran texto.
        """
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class BrightTextMaskStrategy(TextMaskStrategy):
    """
    Estrategia por defecto: asume que el texto del juego es claro
    (blanco/amarillento) con un contorno oscuro de alto contraste contra
    el fondo del ícono -- un estilo muy común en UI de videojuegos
    justamente porque tiene que leerse sobre fondos variables (el mismo
    motivo por el que nos está complicando la validación acá).

    La heurística combina dos señales en vez de un solo umbral de brillo,
    porque un umbral de brillo solo también marcaría zonas claras del
    ícono que no son texto:
        1. Píxeles "core" muy brillantes (el relleno de la letra).
        2. Alto contraste local (gradiente), del contorno oscuro que
           suele rodear el texto.
    Luego filtra por geometría de componente conexa (alto/ancho/aspect
    ratio típico de un glifo o línea de texto) para descartar blobs
    grandes y uniformes que no son letras (ej: un parche brillante del
    ícono).

    Todos los umbrales son parámetros con default razonable pero están
    pensados para recalibrarse empíricamente contra capturas reales del
    juego (no hay acceso al código fuente para conocer el estilo exacto
    de fuente/color de antemano). Una vez calibrados, promoverlos a
    `settings.py` para no tener números mágicos sueltos en el código.
    """

    min_brightness: int = 200
    dark_text_threshold: int = 180
    light_text_threshold: int = 140
    background_brightness_threshold: int = 220
    gradient_threshold: int = 40
    min_component_area: int = 4
    max_component_area_ratio: float = 0.25  # relativo al área total del recorte
    min_component_height: int = 4
    max_component_height_ratio: float = 0.5  # relativo al alto del recorte
    dilate_kernel_size: tuple[int, int] = (5, 3)  # (ancho, alto): fusiona glifos en líneas

    def build_mask(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        # 1) Texto oscuro sobre fondo claro: la mayoría del tooltip principal.
        _, dark_mask = cv2.threshold(
            gray, self.dark_text_threshold, 255, cv2.THRESH_BINARY_INV
        )
        _, background_mask = cv2.threshold(
            gray, self.background_brightness_threshold, 255, cv2.THRESH_BINARY
        )
        dark_text_mask = cv2.bitwise_and(dark_mask, cv2.bitwise_not(background_mask))

        # 2) Texto claro sobre fondo coloreado: el botón del tooltip usa
        # letras blancas sobre un fondo azul/coloreado. Aquí tomamos los
        # píxeles muy brillantes que NO son el fondo general del recorte.
        _, bright_mask = cv2.threshold(gray, self.light_text_threshold, 255, cv2.THRESH_BINARY)
        bright_text_mask = cv2.bitwise_and(bright_mask, cv2.bitwise_not(background_mask))

        # 3) Contorno de alto contraste alrededor del texto, útil para
        # reforzar ambas variantes (oscuro sobre claro y claro sobre color).
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(grad_x, grad_y)
        gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, edge_mask = cv2.threshold(
            gradient, self.gradient_threshold, 255, cv2.THRESH_BINARY
        )

        combined = cv2.bitwise_or(dark_text_mask, bright_text_mask)
        combined = cv2.bitwise_or(combined, edge_mask)

        # Cierra pequeños huecos dentro de una misma letra antes de filtrar
        # por componente conexa, para no descartar letras "agujereadas"
        # (ej: la "a", la "o") por tener el núcleo partido en dos blobs.
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel)

        filtered = self._filter_by_component_geometry(combined, image.shape[:2])

        # Dilata horizontalmente más que verticalmente para fusionar
        # letras individuales en bloques de línea de texto: el resto del
        # pipeline (inpainting, recorte para OCR) trabaja mejor con
        # regiones de texto contiguas que con glifos sueltos.
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, self.dilate_kernel_size
        )
        mask = cv2.dilate(filtered, dilate_kernel, iterations=1)

        logger.debug(
            "Text mask: %.2f%% de píxeles marcados como texto",
            100.0 * np.count_nonzero(mask) / mask.size,
        )
        return mask

    def _filter_by_component_geometry(
        self, mask: np.ndarray, image_shape: tuple[int, int]
    ) -> np.ndarray:
        image_height, image_width = image_shape
        total_area = image_height * image_width
        max_area = total_area * self.max_component_area_ratio
        max_height = image_height * self.max_component_height_ratio

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        filtered = np.zeros_like(mask)

        for label in range(1, num_labels):  # 0 es el fondo
            area = stats[label, cv2.CC_STAT_AREA]
            height = stats[label, cv2.CC_STAT_HEIGHT]

            if area < self.min_component_area or area > max_area:
                continue
            if height < self.min_component_height or height > max_height:
                continue

            filtered[labels == label] = 255

        return filtered


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PreprocessedTooltip:
    """
    Salida del preprocesamiento: las dos vistas derivadas del recorte
    crudo que consumen `text_validator.py` (ocr_ready) e
    `image_validator.py` (comparison_ready) respectivamente, más el mask
    intermedio por si hace falta para evidencia/debug
    (`settings.debug_mode`).
    """

    ocr_ready: np.ndarray
    comparison_ready: np.ndarray
    text_mask: np.ndarray


class ImagePreprocessor:
    """
    Orquesta la separación de texto/imagen sobre el recorte de un
    tooltip, delegando la detección del mask a una `TextMaskStrategy`
    (inyectada, swappable sin tocar esta clase).
    """

    def __init__(self, strategy: TextMaskStrategy | None = None) -> None:
        self._strategy = strategy or BrightTextMaskStrategy()

    def build_text_mask(self, image: np.ndarray) -> np.ndarray:
        """Expuesto aparte de `preprocess` por si algún caller solo necesita el mask (ej: tests de calibración de la estrategia)."""
        return self._strategy.build_mask(image)

    def isolate_text_for_ocr(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        *,
        upscale_factor: float = 2.0,
    ) -> np.ndarray:
        """
        Para esta UI, la ruta estable y verificable no es binarizar a lo loco,
        sino mantener la imagen en escala de grises y dejar que Tesseract
        lea el texto con el contraste real. Eso es lo que de hecho funciona
        con el tooltip real del juego, mientras que la binarización agresiva
        introduce artefactos y rompe letras.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        if mask is not None:
            mask_area = np.count_nonzero(mask)
            image_area = mask.shape[0] * mask.shape[1]
            coverage = mask_area / image_area
            if 0.005 <= coverage <= 0.35:
                mask = (mask > 0).astype(np.uint8) * 255
                mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
                text_only = np.full_like(gray, 255, dtype=np.uint8)
                text_only[mask > 0] = gray[mask > 0]
                gray = text_only

        if upscale_factor != 1.0:
            new_size = (
                int(gray.shape[1] * upscale_factor),
                int(gray.shape[0] * upscale_factor),
            )
            gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_CUBIC)

        return gray

    def remove_text_via_inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        *,
        dilate_iterations: int = 2,
        inpaint_radius: int = 3,
    ) -> np.ndarray:
        """
        Produce una versión del ícono sin el texto superpuesto, para que
        `image_validator.py` compare gráfico contra gráfico (ver docstring
        del módulo sobre por qué esto es necesario para que el SSIM no
        sea sensible al idioma).

        Se dilata el mask antes de inpaintear (`dilate_iterations`) para
        cubrir también el halo/antialiasing alrededor de las letras que
        el mask exacto puede dejar afuera; inpaintear de más un par de
        píxeles de fondo es preferible a dejar residuos de texto que
        arruinen la comparación.

        cv2.INPAINT_TELEA porque reconstruye mejor bordes/estructura que
        NS cuando el área a rellenar es angosta y alargada (el caso
        típico de una línea de texto), que es nuestro escenario.
        """
        if mask is None:
            mask = self.build_text_mask(image)

        if dilate_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)

        return cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)

    def preprocess(self, image: np.ndarray) -> PreprocessedTooltip:
        """
        Punto de entrada único recomendado para `tooltip_flow.py`: calcula
        el mask una sola vez y deriva ambas vistas a partir de él, en vez
        de que el caller tenga que orquestar los tres métodos sueltos.
        """
        mask = self.build_text_mask(image)
        return PreprocessedTooltip(
            ocr_ready=self.isolate_text_for_ocr(image, mask),
            comparison_ready=self.remove_text_via_inpaint(image, mask),
            text_mask=mask,
        )