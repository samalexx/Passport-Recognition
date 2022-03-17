import re
import numpy as np
import pytesseract
import cv2
from PIL import Image
import io
from scipy.ndimage import interpolation as inter

img = cv2.imread('C:\Games\STS\sts/35.jpeg')

image= cv2.resize(img, (800, 1200))

blur = cv2.medianBlur(image, 9)

edged = cv2.Canny(blur, 35, 210, 4)

accumEdged = np.zeros(image.shape[:2], dtype="uint8")

accumEdged = cv2.bitwise_or(accumEdged, edged)

contours, _ = cv2.findContours(accumEdged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

idx = 0 
for con in contours:
    x,y,w,h = cv2.boundingRect(con)
    if len(con) > 300:
        idx+=1
        print(idx,':',len(con), x,y,w,h)
        new_image = image[y:y+h, x:x+w]

gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
#preprocess image for best result
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
gray = cv2.GaussianBlur(gray, (7, 7), 0)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
grad = np.absolute(grad)
(minVal, maxVal) = (np.min(grad), np.max(grad))
grad = (grad - minVal) / (maxVal - minVal)
grad = (grad * 255).astype("uint8")
grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

cv2.imwrite('C:\Games\STS/box/1.jpeg',thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()