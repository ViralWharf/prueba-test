import sys
from pathlib import Path
# Añadir raíz del repo al path para importar src
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import cv2
from src.ocr.tesseract_engine import TesseractOcrEngine
from src.vision.image_preprocessor import ImagePreprocessor

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("image", help="Path to cropped image to OCR")
args = parser.parse_args()

IMAGE_PATH = args.image

if __name__ == '__main__':
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"No se pudo abrir la imagen: {IMAGE_PATH}")
        sys.exit(2)

    pre = ImagePreprocessor()
    mask = pre.build_text_mask(img)

    out_dir = Path(repo_root) / "reports" / "ocr_debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Guardar máscara y preprocesado base (upscale=2.0)
    cv2.imwrite(str(out_dir / "mask.png"), mask)
    base_ocr_ready = pre.isolate_text_for_ocr(img, mask, upscale_factor=2.0)
    cv2.imwrite(str(out_dir / "ocr_ready_2x.png"), base_ocr_ready)

    psms = [3, 4, 6]
    upscales = [1.5, 2.0, 3.0]

    for psm in psms:
        for up in upscales:
            ocr_img = pre.isolate_text_for_ocr(img, mask, upscale_factor=up)
            cv2.imwrite(str(out_dir / f"ocr_ready_{psm}_x{int(up*10)}.png"), ocr_img)
            engine = TesseractOcrEngine(psm=psm)
            result = engine.extract_text(ocr_img, lang='eng')

            print(f"PSM={psm} upscale={up}: confidence={result.confidence:.2f} words={result.word_count}")
            print(result.text)
            print("-" * 30)
