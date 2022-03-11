import re
import cv2
import pytesseract
import numpy as np
import cv2
import imutils
from scipy.ndimage import interpolation as inter


def kat_is_zero(img):

    crop = preprocess(img)

    thresh = text_block(crop)

    blocks = find_text(crop,thresh)

    replace_text = recognize_text_kat(blocks)

    return replace_text


def correct_skew(img, delta=0.2, limit=10):
    image = img
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        return {"Image error":"Photo channels have been disrupted there are highlights in the photo"}
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

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    image_blur = cv2.GaussianBlur(gray, (3,3), 1)

    obr_image = cv2.Canny(image_blur, 100, 300, 4)

    contours, _ = cv2.findContours(obr_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   

    c = sorted(contours, key = cv2.contourArea, reverse = True)[0]

    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    x,y,w,h = cv2.boundingRect(box)

    crop = img[y-200:y+h+100, x:x+w-(int((x/100)*80))]
    crop = correct_skew(crop, 0.2)
    return crop

def text_block(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        return {"Image error":"Photo channels have been disrupted there are highlights in the photo"}
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 10))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad = np.absolute(grad)
    (minVal, maxVal) = (np.min(grad), np.max(grad))

    grad = (grad - minVal) / (maxVal - minVal)
    grad = (grad * 255).astype("uint8")

    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return thresh  

def find_text(img, thresh):
    image = img
    try:
        allContours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   
    except cv2.error:
        return {"Image error":"Photo channels have been disrupted there are highlights in the photo"}
    allContours = imutils.grab_contours(allContours)    
    idx = 0
    ocr_text = []
    for c in allContours:
        x,y,w,h = cv2.boundingRect(c)
        if x < 100 and y < 100:    
            idx+=1
            # cv2.rectangle(image, (x,y),(x+w+5,y+h+5),(0,255,0), 2)
            # cv2.putText(image, f'{idx}', (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255), 2)
            new_image = image[y:y+h+5, x:x+w+5]
            gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            invert = 255 - opening
            custom_config = r'-l rus+frn --psm 6 --oem 1'
            text = pytesseract.image_to_string(invert, config=custom_config)
            text = text.replace("\n", '')
            ocr_text.append(text.capitalize())
    return ocr_text


def recognize_text_kat(text):
    text_to_Str = str(text)
    text_without_symbol = re.sub(r"[\]\[\—\"§|!|'|©|®|_|№|`‘’|›|()|@|=|%|>|\/|-\|]", '', text_to_Str)
    text_without_kir = re.sub(r"[а-яёa-z]", '', text_without_symbol)
    fixed_text = re.sub(r"[С|C|O|О|A|А]", '', text_without_kir)
    del_letter = re.sub(r"[г-яёГ-Я]",'', fixed_text)
    category_date = str(re.findall(r"\s\d{4}\s", del_letter))
    category_date_sub = re.sub(r"[\[\]|,|']",'', category_date)  
    data = {
            "Category and date": "01.01."+category_date_sub
    }
    return data
