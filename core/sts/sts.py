import re
import numpy as np
import pytesseract
import cv2
from PIL import Image
import io
import imutils
from scipy.ndimage import interpolation as inter

def sts_main(data):
    image = np.array(Image.open(io.BytesIO(data)))
    bst_image= cv2.resize(image, (800, 1200))

    blur = cv2.medianBlur(bst_image, 9)

    edged = cv2.Canny(blur, 35, 210, 4)

    accumEdged = np.zeros(bst_image.shape[:2], dtype="uint8")

    accumEdged = cv2.bitwise_or(accumEdged, edged)

    contours, _ = cv2.findContours(accumEdged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    idx = 0 
    for con in contours:
        x,y,w,h = cv2.boundingRect(con)
        if len(con) > 300:
            idx+=1
            new_image = bst_image[y:y+h, x:x+w]

    new_image1 = new_image[0:250, 0:1200]

    gray = cv2.cvtColor(new_image1, cv2.COLOR_BGR2GRAY)
    #preprocess image for best result
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad = np.absolute(grad)
    (minVal, maxVal) = (np.min(grad), np.max(grad))
    grad = (grad - minVal) / (maxVal - minVal)
    grad = (grad * 255).astype("uint8")
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    allContours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   
    allContours = imutils.grab_contours(allContours)

    idx=0

    for contour in allContours:
        x,y,w,h = cv2.boundingRect(contour)
        if w > 250:
            idx+=1
            print(idx, ":", x,y,w,h)
            # cv2.rectangle(new_image1, (x, y-10), (x + w, (y-10) + h), (36,255,12), 2)
            # cv2.putText(new_image1, f'{idx}', (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
            new_image = new_image1[y-10:(y-10)+h, x-10:x+w]
            gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
            thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
            opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=1)
            invert = 255 - opening
            custom_config2 = r'-l eng+rus --psm 6 --oem 3'
            text2 = pytesseract.image_to_string(invert, config=custom_config2)
            text2 = text2.replace("\n", ' ')
            print(text2)
            text2 = re.sub(r"[^\d№]+", '', text2)
            print(text2)
    return {'series':text2}