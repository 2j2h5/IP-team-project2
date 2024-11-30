from ultralytics import YOLO
import cv2
import pytesseract
import matplotlib.pyplot as plt
from PIL import Image
import os

pytesseract.pytesseract.tesseract_cmd = r'./Tesseract-OCR/tesseract.exe'

def plate_focus(path):
    model = YOLO('best.pt')

    results = model(path)

    plate = cv2.imread(path)

    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate = plate[int(y1):int(y2), int(x1):int(x2)]
    
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    resized = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    cv2.imwrite("preprocessed_plate.jpg", resized)

    config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주아바사자배하허호국합육해공외교영준기협정대표'

    results = pytesseract.image_to_data(Image.fromarray(resized), lang='kor', output_type=pytesseract.Output.DICT, config=config)

    for i in range(len(results['text'])):
        if results['text'][i] == '':
            continue
        x, y, w, h = results['left'][i] // 2, results['top'][i] // 2, results['width'][i] // 2, results['height'][i] // 2
        cv2.rectangle(plate, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(results['text'][i])
        cv2.imwrite("plate.jpg", plate[y:y+h, x:x+w])
    
    if os.path.exists("preprocessed_plate.jpg"):
        os.remove("preprocessed_plate.jpg")

    if not os.path.exists("plate.jpg"):
        cv2.imwrite("plate.jpg", plate)
