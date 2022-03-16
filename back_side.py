import re
import numpy as np
import pytesseract
import cv2
from kat_null import kat_is_zero
from PIL import Image
import io
from main import correct_skew1
from scipy.ndimage import interpolation as inter

async def side_main(contents):

    image = np.array(Image.open(io.BytesIO(contents)))

    image = cv2.resize(image, (790, 499))

    crop_image = find_rectangle(image)

    rectangles_contour, table = draw_contours(crop_image)

    text = recognize_text(rectangles_contour, table)

    ocr_text = text_correction(text,image)

    return ocr_text



def find_rectangle(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_blur = cv2.GaussianBlur(gray, (3,3), 1)

        obr_image = cv2.Canny(image_blur, 75, 280, 3)

        contours, _ = cv2.findContours(obr_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   

        c = sorted(contours, key = cv2.contourArea, reverse = True)[0]

        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        x,y,w,h = cv2.boundingRect(box)
        crop = img[y-5:y+h+5, x-5:x+w+5]
        return crop
    except cv2.error:
        return {"Image error":"Photo channels have been disrupted there are highlights in the photo"}


# contour find and draw in input image
def draw_contours(crop):
    img = crop[40:120, 0:350]

    table = correct_skew1(img, 0.2, 10)

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
    return result, table

def recognize_text(rectangles_contour, table):
    gray_result = cv2.cvtColor(rectangles_contour, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray_result, (3,3), 2)

    cannys = cv2.Canny(blur, 50,350, 4)

    cont, _ = cv2.findContours(cannys, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_ocr = []
    idx = 0
    for con in cont:
        x,y,w,h = cv2.boundingRect(con)
        if w > 100 and h > 10 and 20 < y < 200 and x < 300:
            idx += 1
            cv2.rectangle(table, (x, y), (x + w, y + h), (36,255,12), 2)
            # cv2.putText(table, f'{idx}', (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255), 2)
            new_image = table[y-2:y+h+2, x-2:x+w+2]
            new_image = correct_skew1(new_image, 0.5, 5)
            img = cv2.resize(new_image, (355, 75))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            invert = 255 - opening
            custom_config = r'-l rus+frn --psm 6 --oem 1'
            text = pytesseract.image_to_string(invert, config=custom_config)
            text = text.replace("\n", '')
            text_ocr.append(text.capitalize())
            print(text_ocr)
    return text_ocr
    

def text_correction(text, img):
    text_reverse = text[::-1]
    text_reverse = str(text_reverse)
    text_without_symbol = re.sub(r"[^\d\.\sBВв]", '', text_reverse)
    category_date = str(re.findall(r"[В|В|в|8]\s+\d{2}\.\d{2}.\d{4}", text_without_symbol))
    category_date_sub = re.sub(r"[\[\]|,|']",'', category_date)
    if re.match(r"[В|В|в|8]\s+\d{2}\.\d{2}.\d{4}", category_date_sub):
        data = {
                "Category and date": category_date_sub
        }
        return data
    else:   
        text_reverse1 = str(text)
        text_without_symbol1 = re.sub(r"[^\d\.\sBВв]", '', text_reverse1)
        category_date1 = str(re.findall(r"[В|В|в|8]\s+\d{2}\.\d{2}.\d{4}", text_without_symbol1))
        category_date_sub1 = re.sub(r"[\[\]|,|']",'', category_date1)
        if re.match(r"[В|В|в|8]\s+\d{2}\.\d{2}.\d{4}", category_date_sub1):
            data1 = {
                "Category and date": category_date_sub1
            }
            return data1
        else:
            data2 = kat_is_zero(img)
            return data2
