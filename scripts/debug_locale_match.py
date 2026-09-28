import cv2
import sys
from pathlib import Path

frame_path = Path(r"reports/screenshots/diagnostics/select_lang_frame_20260923_224557_4.png")
template_path = Path(r"test_data/reference_images/locale_label.png")

if not frame_path.exists():
    print('Frame not found:', frame_path)
    sys.exit(2)
if not template_path.exists():
    print('Template not found:', template_path)
    sys.exit(2)

frame = cv2.imread(str(frame_path))
tpl = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
if frame is None or tpl is None:
    print('Failed to read images')
    sys.exit(2)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
th, tw = tpl.shape[:2]
print('template size:', tw, th)
print('min_val, max_val:', min_val, max_val)
print('max_loc:', max_loc)
# Save a visualization with rectangle
vis = frame.copy()
cv2.rectangle(vis, max_loc, (max_loc[0]+tw, max_loc[1]+th), (0,0,255), 2)
out = Path('reports/screenshots/diagnostics/locale_match_vis.png')
cv2.imwrite(str(out), vis)
print('Wrote visualization to', out)
