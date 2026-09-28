"""
scripts/select_locale.py

Script para seleccionar el locale "Spanish (es-es)" en un ListBox mediante GUI.

Requisitos:
- Python 3
- pyautogui
- pytesseract
- pillow

Uso:
    python scripts/select_locale.py

El script primero intenta una búsqueda rápida usando teclado (presionar 'S' y escribir),
luego como fallback realiza un bucle de scroll + OCR hasta 8 intentos para encontrar y
hacer clic sobre la opción deseada.

Nota: Ajustar las coordenadas de `LISTBOX_OPEN_COORDS` y la región `LIST_REGION` según
la interfaz objetivo.
"""

import time
import sys
from typing import Optional, Tuple, List

import pyautogui
from PIL import Image

try:
    import pytesseract
except Exception:
    pytesseract = None

# CONFIGURACIÓN (ajustar según la UI)
# Coordenadas para hacer clic y abrir el ListBox de locales (x, y)
LISTBOX_OPEN_COORDS: Tuple[int, int] = (100, 200)

# Región aproximada donde aparece la lista desplegable: (left, top, width, height)
# Ajustar para que cubra el área visible del listbox cuando está abierto.
LIST_REGION: Tuple[int, int, int, int] = (80, 230, 300, 400)

# Tiempos
SHORT_DELAY = 0.15
MID_DELAY = 0.3
LONG_DELAY = 0.6

# OCR keywords (minúsculas)
KEYWORDS = ["spanish", "español", "es-es", "spanish (es-es)"]


def abrir_listbox(coords: Tuple[int, int]) -> None:
    """Mueve el cursor y hace clic para abrir el ListBox.

    coords: tupla (x, y) con las coordenadas del centro del control.
    """
    pyautogui.moveTo(coords[0], coords[1], duration=0.2)
    time.sleep(SHORT_DELAY)
    pyautogui.click()
    time.sleep(MID_DELAY)


def quick_keyboard_search(guess_text: str = "spanish") -> bool:
    """Intenta seleccionar la opción tecleando 'S' + el resto de la palabra.

    Devuelve True si parece que se seleccionó (basado en una pequeña pausa),
    False en caso contrario. Esta función no valida visualmente la selección,
    por lo que se usa como intento rápido previo al OCR.
    """
    # Presionar 's' y escribir el resto rápidamente
    pyautogui.press('s')
    time.sleep(0.05)
    pyautogui.typewrite(guess_text[1:], interval=0.03)
    time.sleep(0.5)
    return True


def image_to_ocr_data(img: Image.Image) -> Optional[List[dict]]:
    """Convierte una imagen en datos OCR con coordenadas usando pytesseract.

    Retorna una lista de dicts con claves: text, left, top, width, height, conf
    o None si pytesseract no está disponible.
    """
    if pytesseract is None:
        return None

    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    results = []
    n = len(data.get('text', []))
    for i in range(n):
        text = data['text'][i].strip()
        if not text:
            continue
        results.append({
            'text': text,
            'left': int(data['left'][i]),
            'top': int(data['top'][i]),
            'width': int(data['width'][i]),
            'height': int(data['height'][i]),
            'conf': int(data['conf'][i]) if data['conf'][i] != '-1' else -1,
        })
    return results


def buscar_por_ocr_y_clicar(region: Tuple[int, int, int, int], keywords: List[str],
                             max_attempts: int = 8) -> bool:
    """Realiza scroll y OCR en la región dada buscando keywords. Si encuentra,
    hace clic sobre el centro del bounding box y retorna True.

    region: (left, top, width, height)
    """
    left, top, width, height = region
    for attempt in range(1, max_attempts + 1):
        # Capturar la región
        img = pyautogui.screenshot(region=region)
        # Convertir a RGB (Pillow)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        ocr_data = image_to_ocr_data(img)
        if ocr_data is None:
            raise RuntimeError("pytesseract no está disponible; instale pytesseract para usar OCR")

        # Buscar coincidencias (tolerante a mayúsculas/minúsculas)
        for item in ocr_data:
            txt = item['text'].lower()
            for kw in keywords:
                if kw in txt:
                    # Calcular centro absoluto en pantalla
                    cx = left + item['left'] + item['width'] // 2
                    cy = top + item['top'] + item['height'] // 2
                    pyautogui.moveTo(cx, cy, duration=0.25)
                    time.sleep(SHORT_DELAY)
                    pyautogui.click()
                    time.sleep(MID_DELAY)
                    return True

        # Si no se encontró, realizar scroll dentro del listado
        # Movemos el cursor al centro de la región antes de scrollear para
        # asegurarnos de que el scroll afecte al listbox.
        cx = left + width // 2
        cy = top + height // 2
        pyautogui.moveTo(cx, cy, duration=0.15)
        time.sleep(0.05)
        pyautogui.scroll(-120)  # scroll hacia abajo
        time.sleep(0.3)

    return False


def seleccionar_locale_por_nombre(target_names: List[str]) -> None:
    """Flujo principal: abre el listbox, intenta búsqueda rápida y luego OCR+scroll.

    target_names: lista de keywords para buscar (minúsculas preferible)
    """
    abrir_listbox(LISTBOX_OPEN_COORDS)

    # Intento rápido por teclado
    try:
        quick_keyboard_search('spanish')
        # Dar tiempo para que la UI seleccione si correspondiera
        time.sleep(0.5)
    except Exception:
        pass

    # Intentar OCR + scroll como fallback
    try:
        found = buscar_por_ocr_y_clicar(LIST_REGION, target_names)
    except RuntimeError as e:
        raise

    if not found:
        raise RuntimeError("No se pudo localizar el idioma Spanish (es-es) en la interfaz después de varios intentos")


def main():
    if pytesseract is None:
        print("Aviso: pytesseract no está disponible. Instale 'pytesseract' para habilitar OCR.")
        print("Aún así se intentará la búsqueda por teclado. Para mejor robustez, instale pytesseract.")

    print("Iniciando selección de Locale: Spanish (es-es)")
    try:
        seleccionar_locale_por_nombre(KEYWORDS)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("Idioma seleccionado (o acción completada).")


if __name__ == '__main__':
    main()
