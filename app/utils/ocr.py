import pytesseract
from PIL import Image
from typing import Dict

def ocr_image(image_path: str) -> Dict:
    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    blocks = []
    n = len(data['level'])
    for i in range(n):
        txt = data['text'][i].strip()
        if not txt:
            continue
        blocks.append({
            'text': txt,
            'left': data['left'][i],
            'top': data['top'][i],
            'width': data['width'][i],
            'height': data['height'][i],
            'conf': data['conf'][i]
        })
    return {'blocks': blocks}
