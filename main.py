import cv2
import numpy as np
import imutils
import pytesseract

BAZIS_WIDTH = 1300
BASIZ_HEIGHT = 900

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

image = cv2.imread('C:/Games/STS/11.jpeg')

image = cv2.resize(image, (BAZIS_WIDTH,BASIZ_HEIGHT))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 10))

gray = cv2.GaussianBlur(gray, (5, 5), 0)

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
idx = 0
for contour in allContours:
    x,y,w,h = cv2.boundingRect(contour)
    lenght = len(contour)
    if lenght > 10 and w > 100:
        idx += 1
        color = list(np.random.random(size=3) * 256)
        new_image = image[y-15:y+h, x-110:x+w]
        custom_config = r'-l eng --psm 6 --oem 3'
        text = pytesseract.image_to_string(new_image, config=custom_config)
        print(idx,':',text)
cv2.imshow('C:/Games/123.jpg', image)
cv2.waitKey()
cv2.destroyAllWindows()