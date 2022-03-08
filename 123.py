import re
import cv2
import numpy as np
import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import pandas as pd
from pytesseract import Output
import cv2
from scipy.ndimage import interpolation as inter

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

image = cv2.imread('C:/Games/STS/16.jpeg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image_blur = cv2.GaussianBlur(gray, (3,3), 1)

obr_image = cv2.Canny(image_blur, 100, 300, 4)

contours, _ = cv2.findContours(obr_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   

c = sorted(contours, key = cv2.contourArea, reverse = True)[0]

rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))
x,y,w,h = cv2.boundingRect(box)

crop = image[y-5:y+h+5, x-5:x+w+5]


def correct_skew(img, delta=0.2, limit=10):
    image = img
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        _, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
              borderMode=cv2.BORDER_REPLICATE)
    return rotated

img = crop[40:120, 0:310]

table = correct_skew(img)

table = cv2.resize(table, (500,300))

result = table.copy()
gray = cv2.cvtColor(table,cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35,1))
remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(result, [c], -1, (0,0,0), 4)

vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,70))
remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=3)
cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(result, [c], -1, (0,0,0), 4)

gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray_result, (3,3), 2)

cannys = cv2.Canny(blur, 50,350, 4)

cont, _ = cv2.findContours(cannys, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

idx = 0
for con in cont:
    x,y,w,h = cv2.boundingRect(con)
    if 30 < y < 200 and x < 300 and 1 < len(con) < 500 and w > 100:
        idx += 1
        cv2.rectangle(result, (x, y), (x + w, y + h), (36,255,12), 2)
        # cv2.putText(result, f'{idx}', (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255), 2)
        new_image = table[y:y+h, x:x+w]
        new_image = correct_skew(new_image, 0.5, 5)
        print(idx,len(con),x,y,w,h)
        img = cv2.resize(new_image, (350,80))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        invert = 255 - opening
        custom_config = r'-l eng+rus+lat --psm 7 --oem 1'
        text = pytesseract.image_to_string(invert, config=custom_config)
        print(text.capitalize())
        filename = "C:/Games/STS/box/file_%d.png"%idx
        cv2.imwrite(filename, invert)
cv2.imshow('C:/Games/STS/box/123.jpeg', invert) 
cv2.waitKey(0)
cv2.destroyAllWindows()