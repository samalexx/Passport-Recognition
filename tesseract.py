import pytesseract
import cv2
import numpy as np
import re
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'



image = cv2.imread('C:/Games/STS/box/file_2.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
invert = 255 - opening

custom_config = r'-l eng+rus --psm 6 --oem 1'

text = pytesseract.image_to_string(invert, config=custom_config)
if re.match(r"(^\d\S+\s+.+)", text):
        print(text)
else:
        print("нету")
print(text)
cv2.imshow('C:/Games/123.jpg', invert)
cv2.waitKey(0)
cv2.destroyAllWindows()