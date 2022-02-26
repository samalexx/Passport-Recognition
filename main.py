import cv2
import numpy as np
import imutils
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

#Calculate degrees of image, and convert it
def skew_angle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    print(angle)
    if angle < -45:
        angle = -(90 + angle)
    elif angle == 90:
        angle = 0
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


#Resize image around contour rectangle
def resize_image(image):
    img = image
    img = cv2.resize(img, (1200,900))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    invGamma = 1.0 / 0.3
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype('uint8')

    gray = cv2.LUT(gray, table)

    _, thresh1 = cv2.threshold(gray,75,255,cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def biggestRectangle(contours):
        max_area = 0
        indexReturn = -1
        for index in range(len(contours)):
                i = contours[index]
                area = cv2.contourArea(i)
                if area > 100:
                    peri = cv2.arcLength(i,True)
                    approx = cv2.approxPolyDP(i,0.1*peri,True)
                    if area > max_area: #and len(approx)==4:
                            max_area = area
                            indexReturn = index
        return indexReturn

    indexReturn = biggestRectangle(contours)
    hull = cv2.convexHull(contours[indexReturn])
    #photo simplification
    if len(hull) > 10:
        ROIdimensions = hull.reshape(len(hull),2)

        rect = np.zeros((4,2), dtype='float32')
        
        s = np.sum(ROIdimensions, axis=1)
        rect[0] = ROIdimensions[np.argmin(s)]
        rect[2] = ROIdimensions[np.argmax(s)]

    
        diff = np.diff(ROIdimensions, axis=1)
        rect[1] = ROIdimensions[np.argmin(diff)]
        rect[3] = ROIdimensions[np.argmax(diff)]

        (tl, tr, br, bl) = rect

        widthA = np.sqrt((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2 )
        widthB = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2 )
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2 )
        heightB = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2 )
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0,0],
            [maxWidth-1, 0],
            [maxWidth-1, maxHeight-1],
            [0, maxHeight-1]], dtype="float32")


        transformMatrix = cv2.getPerspectiveTransform(rect, dst)

        scan = cv2.warpPerspective(img, transformMatrix, (maxWidth, maxHeight))
        return scan
    else:
        return img


# marking up rectangles
def rectangle_image(img):
    image = img
    image = cv2.resize(image, (1200,900))
    #to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #preprocess image for best 
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 10))
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad = np.absolute(grad)
    (minVal, maxVal) = (np.min(grad), np.max(grad))
    grad = (grad - minVal) / (maxVal - minVal)
    grad = (grad * 255).astype("uint8")
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #find contour
    allContours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   
    allContours = imutils.grab_contours(allContours)    
    idx = 0
    ocr_text = []
    #find block of text
    for contour in allContours:
        x,y,w,h = cv2.boundingRect(contour)
        lenght = len(contour)
        if 10 < lenght < 200 and w > 100:
            idx += 1
            print(idx, ':',x,y,w,h, len(contour))
            new_image = image[y-10:y+(h+1), x-102:x+(w+10)]
            gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            invert = 255 - opening
            custom_config = r'-l eng+rus --psm 6 --oem 1'
            text = pytesseract.image_to_string(invert, config=custom_config)
            if re.match(r"(^\d+\D+.+)", text):
                 ocr_text.append(text)
            else:
                print(text)
    return ocr_text

#for correct text (1.,2. ... n+1)
def correct_text(text):
    text.reverse()
    text = str(text)
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r"'", '', text)
    text = re.sub(r"\[", '', text)
    text = re.sub(r"\]", '', text)
    return text

# main 
if __name__ == '__main__':
    image = cv2.imread('C:/Games/STS/13.jpg')
    angled = skew_angle(image)
    image_resize = resize_image(angled)
    text = rectangle_image(image_resize)
    end_text = correct_text(text)
    print(end_text)