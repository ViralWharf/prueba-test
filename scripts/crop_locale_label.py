import sys
from pathlib import Path
import cv2

diag_dir = Path("reports/screenshots/diagnostics")
if not diag_dir.exists():
    print("Diagnostics dir not found:", diag_dir)
    sys.exit(2)

files = sorted(diag_dir.glob('select_lang_frame_*.png'))
if not files:
    print("No diagnostic frames found")
    sys.exit(2)

latest = files[-1]
print('Using', latest)
img = cv2.imread(str(latest))
if img is None:
    print('Failed to read', latest)
    sys.exit(3)

h, w = img.shape[:2]
# Heuristic crop for left panel where the Locale control appears
# Adjusted to capture the top-left panel controls
x0, y0, x1, y1 = 0, 30, min(w, 480), min(h, 300)
print('Crop box', x0, y0, x1, y1)
crop = img[y0:y1, x0:x1]

out_dir = Path('test_data/reference_images')
out_dir.mkdir(parents=True, exist_ok=True)
dst = out_dir / 'locale_label.png'
# Backup existing
if dst.exists():
    bak = dst.with_suffix('.bak.png')
    print('Backing up', dst, '->', bak)
    dst.replace(bak)

ok = cv2.imwrite(str(dst), crop)
print('Wrote', dst, 'ok=', ok)
if not ok:
    sys.exit(4)
print('Done')
