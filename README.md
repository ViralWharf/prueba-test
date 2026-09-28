# Game Validator

Descripción
-----------

`Game Validator` es una herramienta de validación automática de UI orientada a validar tooltips de ítems en distintas localizaciones/idiomas. Combina captura de pantalla, búsqueda por template, preprocesado de imágenes, OCR (Tesseract) y validadores de imagen/texto para generar un reporte de validación que puede usarse en presentaciones o integrarse en CI.

Resumen del flujo
-----------------

1. `main.py` arranca la ejecución y resuelve los idiomas a validar.
2. Se crea un único `ScreenCapturer` (recurso compartido) para la sesión.
3. `LanguageScrollFlow` recorre posiciones de la UI (scroll/posiciones) buscando la ventana objetivo mediante `template matching` (ancla del icono del ítem).
4. Para cada ventana encontrada, `TooltipFlow` ejecuta el flujo de validación:
   - Hover / wait: simula hover para mostrar el tooltip y espera `hover_settle_seconds`.
   - Captura de pantalla: toma la región del tooltip.
   - Preprocesado: normaliza la imagen para OCR y comparación (ver `vision/image_preprocessor.py`).
   - OCR: extrae texto usando Tesseract (`ocr/tesseract_engine.py`).
   - Validación: usa `text_validator` para comparar OCR contra `test_data/expected_texts/<lang>.json` y `image_validator` / `template_matcher` para comparar evidencias gráficas.
   - El resultado se empaqueta en un `ValidationResult` y se añade al `ValidationReport`.
5. Al terminar todos los idiomas, `main.py` escribe `reports/report_<run_id>.json` y devuelve un código de salida 0 solamente si no hubo `FAILED` ni `ERROR` (útil para CI).

Estructura del proyecto
-----------------------

- `main.py`: Punto de entrada CLI. Opciones principales: `--language` y `--clean`.
- `src/capture/screen_capture.py`: Abstracción sobre la captura de pantalla (usa `mss`).
- `src/config/settings.py`: Ajustes globales (paths, timeouts, umbrales). Valores por defecto:
  - `reports_dir`: `reports/`
  - `templates_dir`: `test_data/templates/`
  - Umbrales: `template_match_threshold=0.8`, `ssim_similarity_threshold=0.9`, `ocr_min_confidence=60.0`, `text_similarity_threshold=0.85`.
- `src/config/language_config.py`: Definición de idiomas soportados y carga de textos esperados.
- `src/flows/language_scroll_flow.py`: Lógica para recorrer posiciones y localizar la ventana objetivo por template.
- `src/flows/tooltip_flow.py`: Flujo de validación por ventana (hover → captura → OCR → validadores).
- `src/ocr/`: Implementaciones de OCR; `tesseract_engine.py` es la implementación por defecto.
- `src/vision/`: Preprocesado y `template_matcher`.
- `src/validators/`: `text_validator.py` y `image_validator.py` que implementan las reglas de éxito/fallo.
- `src/utils/logger.py`: Configuración de logging; crea `reports/logs/<run_id>.log` por corrida.
- `test_data/`: Datos de prueba y golden data (`expected_texts/`, `templates/`, `reference_images/`).
- `reports/`: Salida por defecto de la herramienta: `report_*.json`, `logs/`, `screenshots/`.

Instalación y requisitos
------------------------

- Requisitos: Python 3.11+ recomendado.
- Depencias Python: instalar con:

```bash
pip install -r requirements.txt
```

- Requisito externo: Tesseract OCR instalado en la máquina. La ruta por defecto configurada en `src/config/settings.py` es:

```
C:\Program Files\Tesseract-OCR\tesseract.exe
```

Configuración
-------------

Los parámetros principales están en `src/config/settings.py` (singleton `settings`). Puntos relevantes:

- `reports_dir`: carpeta donde se guardan reportes, logs y capturas.
- Timeouts: `hover_settle_seconds`, `window_appear_timeout_seconds`.
- Umbrales de aceptación: `template_match_threshold`, `ssim_similarity_threshold`, `ocr_min_confidence`, `text_similarity_threshold`.
- `debug_mode`: si `True` activa logging más verborreico y guarda capturas intermedias.

Uso
---

Ejemplos:

```bash
python main.py --language en --clean
python main.py --language en es
python main.py --language all
```

- `--language`: códigos de idioma (ej: `en`, `es`) o `all` para correr todos los soportados.
- `--clean`: borra logs, screenshots y reportes previos en `reports/` antes de la corrida.

Salida y artefactos
-------------------

- Reporte JSON: `reports/report_<run_id>.json` — contiene la lista de `ValidationResult` con estado, mensajes de error y rutas a evidencia (capturas).
- Logs: `reports/logs/<run_id>.log` — registro detallado (DEBUG) de la corrida.
- Capturas: `reports/screenshots/` — capturas guardadas cuando `debug_mode` está activo o para la evidencia final.

Cómo funciona la validación (detalle técnico)
-------------------------------------------

1. Búsqueda de ventana objetivo
   - `LanguageScrollFlow` recorre posiciones calculadas por `_build_scroll_positions` hasta `max_language_positions` buscando la plantilla `templates/<WINDOW_ID>_icon.png`.
   - Si no encuentra la ventana lanza `WindowNotFoundError` y el resultado para ese idioma queda en `ERROR` con evidencia vacía.

2. Validación del tooltip
   - `TooltipFlow` realiza hover en la posición objetivo, espera `hover_settle_seconds` y captura la región de tamaño `TOOLTIP_WIDTH x TOOLTIP_HEIGHT`.
   - La imagen pasa por `vision/image_preprocessor.py` (normalización, escalado, binarización opcional) antes de OCR o comparación.
   - OCR: `ocr/tesseract_engine.py` invoca Tesseract y devuelve texto con confidencias.
   - `text_validator` compara el texto extraído con el `expected_texts/<lang>.json` usando una métrica de similitud (SequenceMatcher) y respeta `ocr_min_confidence` y `text_similarity_threshold`.
   - `image_validator` usa `vision/template_matcher.py` y métricas como SSIM para determinar si la imagen coincide con la esperada.

3. Resultado y reporte
   - Cada validación produce un `ValidationResult` con `status` ∈ {PASSED, FAILED, ERROR, SKIPPED} y campo `evidence` con rutas a las capturas.
   - `ValidationReport` agrega contadores `passed_count`, `failed_count`, `error_count`, `skipped_count` y se persiste al final.

Tests
-----

Las pruebas unitarias están en `tests/unit/` y las de integración en `tests/integration/`.

Ejecutar tests:

```bash
pytest -q
```

Notas de desarrollo y calibración
--------------------------------

- Valores de `_SCROLL_START`, `_SCROLL_STEP_Y`, `TOOLTIP_OFFSET` y `_NEXT_PAGE_BUTTON` son placeholders y requieren calibración contra la UI real antes de ejecutar en producción.
- Para depuración activa, establecer `debug_mode = True` en `src/config/settings.py` o exportar la variable de entorno correspondiente si se añade soporte.
- Los umbrales de OCR y similitud deben ajustarse al dataset real; los valores por defecto son conservadores para pruebas locales.

Integración en CI
-----------------

El `main.py` devuelve exit code 0 solo si no hubo `FAILED` ni `ERROR`. Esto permite usar la ejecución directa como step en pipelines CI y que falle si aparecen problemas.

Preguntas frecuentes rápidas
---------------------------

- ¿Dónde están los textos esperados? `test_data/expected_texts/`.
- ¿Cómo cambiar la plantilla del ítem? Sustituir el PNG en `test_data/templates/` con el nombre `item_sword_tooltip_icon.png` o ajustar `WINDOW_ID` en `main.py`.
- ¿Cómo ver los logs de una corrida? Abrir `reports/logs/<run_id>.log`.

Contacto
-------
