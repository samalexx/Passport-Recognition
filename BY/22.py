import cv2
import numpy as np
import imutils
import pytesseract
import uvicorn
from fastapi import FastAPI, File, Query, UploadFile
import io
import os
from PIL import Image
import uuid
from pdf2image import convert_from_bytes
import re
import back_side 
import platform
from scipy.ndimage import interpolation as inter

app = FastAPI()

#select platform
plt = platform.system()
if plt == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
    poppler_path = r'C:/Python38/Lib/poppler/Library/bin'
    temp_path = r'app/tmp'
elif plt == "Linux":
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    poppler_path = None
    temp_path = r'app/tmp'
else:
    print("Unidentified system")


@app.post("/predict/", tags=["Predict"], summary="Predict")
async def upload(file: UploadFile = File(..., description='Выберите файл для загрузки',), 
                 mode: str = Query("front", enum=["front", "back", "sts"], description='Choice doc template')):
    ext = file.filename
    if mode == 'front':
        # search extension in filename
        input_filename = re.findall(r"(jpg|png|jpeg|pdf)$", ext)
        data = await file.read()
        # if filename have extension like pdf
        if input_filename == ['pdf']:
            images = convert_from_bytes(data, dpi=300, single_file=True)
            for page in images:
                image = np.array(page)
                # aligning the photo to the contour
                angled = skew_angle(image)
                # perspective correction 
                image_resize = perspective_correction(angled)
                # preprocessing pictures for text blocks
                tresh_image = preprocess_image(image_resize)
                # get text from text box 
                text = recognize_text(image_resize, tresh_image)
                # get the recognized text and return it with a response from the server
                end_text = correct_text(text)
            return end_text
        # else filename have extension like .jpg,.jpeg,.png
        else:    
            image = np.array(Image.open(io.BytesIO(data)))
            # aligning the photo to the contour
            angled = skew_angle(image)
            # perspective correction 
            image_resize = perspective_correction(angled)
            # preprocessing pictures for text blocks
            tresh_image = preprocess_image(image_resize)
            # get text from text box 
            text = recognize_text(image_resize, tresh_image)
            # get the recognized text and return it with a response from the server
            end_text = correct_text(text)
            return end_text
    elif mode == 'back': 
        file.filename = f"{uuid.uuid4()}.jpg"
        contents = await file.read()
        result = await back_side.side_main(contents)
        return result


#Calculate degrees of image, and convert it
def skew_angle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
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
def perspective_correction(image):
    img = image
    img = cv2.resize(img, (1200,900))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    invGamma = 1.0 / 0.3
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype('uint8')

    gray = cv2.LUT(gray, table)

    _, thresh1 = cv2.threshold(gray,75,255,cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    for index in range(len(contours)):
            i = contours[index]
            area = cv2.contourArea(i)
            if area > 100:
                if area > max_area: #and len(approx)==4:
                        max_area = area
                        indexReturn = index

    hull = cv2.convexHull(contours[indexReturn])
    print(len(hull), indexReturn)
    x,y,w,h = cv2.boundingRect(hull)
    new_image = image[y:y+h, x:x+w]
    #photo simplification
    if 10 < len(hull) < 40 and indexReturn < 10:
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


#2 function for rotate image in text block
def correct_skew1(image, delta=0.6, limit=5):
    #score
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    #find image angle
    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        _, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    #rotate image 
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)

    return rotated


# marking up rectangles
def preprocess_image(img):
    image = img
    image = cv2.resize(image, (1200,900))
    #to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #preprocess image for best result
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
    return thresh


#find contour and text block
def recognize_text(img,thresh):
    image = img
    allContours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   
    allContours = imutils.grab_contours(allContours)    
    ocr_text = []
    #find block of text
    idx = 0
    for contour in allContours:
        x,y,w,h = cv2.boundingRect(contour)
        lenght = len(contour)
        if 50 < w < 500:
            idx += 1
            new_image = image[y-10:(y+5)+(h+1), x-102:x+(w+10)]
            # new_image = cv2.resize(new_image, ())
            try:
                rotated = correct_skew1(new_image, 0.5)
            except:
                pass
            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
            opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=1)
            invert = 255 - opening
            custom_config = r'-l rus+eng --psm 11 --oem 3'
            custom_config2 = r'-l eng+rus --psm 6 --oem 3'
            text = pytesseract.image_to_string(invert, config=custom_config)
            text2 = pytesseract.image_to_string(invert, config=custom_config2)
            text = text.replace("\n", ' ')
            text2 = text2.replace("\n", ' ')
            text_mass = text+text2
            if re.match(r"^\w", text_mass):
                 ocr_text.append(text_mass)
            else:
                print(text)

    return ocr_text


#for correct text (1.,2. ... n+1)
def correct_text(text):
    text.reverse()
    text1 = str(text)
    main_text = re.sub(r'[^\d\wа-яА-Я\s),\.\}]', '', text1)
    main_text = re.sub(r'}', ')', main_text)
    text_without_8 = re.sub(r'\s8{1}[\.\s].+', '', main_text)
    print(text_without_8)
    surname = str(re.findall(r"(1\.\s+\S+[\s|\w\s]+)", text_without_8))
    surname_rus = (re.sub(r"[^а-яА-Я\s]", '', surname)).strip()
    surname_eng = str(re.findall(r"[A-Z]+", surname)[0])

    name_father = str(re.findall(r"2\.\s+\D+,", text_without_8))
    name_father_rus = (re.sub(r"[^а-яА-Я\s]", '', name_father)).strip()
    namefather_eng = str(re.findall(r"[A-Z]+", name_father))
    namefather_eng = re.sub(r"[,|\[\]'\"]", '', namefather_eng)

    date_birth = str(re.findall(r"\s3[\.\s]+\d{2}\.\d{2}\.\d{4}", text_without_8))
    date_birth1 = re.sub(r"[,|\[\]'\"]", '', date_birth)
    date_birth = str(re.findall(r"\d{2}.\d{2}.\d{4}", date_birth1))
    date_birth = re.sub(r"[,|\[\]'\"]", '', date_birth)

    license_date = str(re.findall(r"4[aа]\)\s\d+\.\d+.\d+", text_without_8))
    license_date1 = re.sub(r"[,|\[\]'\"]", '', license_date)
    license_date = str(re.findall(r"\d{2}.\d{2}.\d{4}", license_date1))
    license_date = re.sub(r"[,|\[\]'\"]", '', license_date)
    
    license_number = str(re.findall(r"\d{10}", text_without_8))
    license_number = (re.sub(r"[,|\[\]'\"]", '', license_number)).strip()

    data  = {"surname_rus": surname_rus,
        "surname_eng": surname_eng,
        "firtsname_rus": name_father_rus,
        "firstname_eng": namefather_eng,
        "birthday": date_birth,
        "date_license": license_date,
        "series_number": license_number
    }

    return data



# main 
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8005))
    uvicorn.run("main:app", host='0.0.0.0', port=port,
                  log_level="info", reload=True, workers=1)