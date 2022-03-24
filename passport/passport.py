import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import io
import os
from fastapi import FastAPI, File, Query, UploadFile
import uvicorn
from scipy.ndimage import interpolation as inter
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)
refine_net = load_refinenet_model(cuda=False)
craft_net = load_craftnet_model(cuda=False)
path = "models/FSRCNN_x4.pb"
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

app = FastAPI()

@app.post("/predict/", tags=["Predict"], summary="Predict")
async def upload(file: UploadFile = File(..., description='Выберите файл для загрузки')):
    data = await file.read()

    image = np.array(Image.open(io.BytesIO(data))) # can be filepath, PIL image or numpy array # can be filepath, PIL image or numpy array
    result = text_block(image)
    return result
def text_block(image):
    img = cv2.resize(image, (2700, 3200), fx=0.5, fy=0.3)
    # read image
    image = read_image(img)

    # load models
    refine_net = load_refinenet_model(cuda=False)
    craft_net = load_craftnet_model(cuda=False)

    # perform prediction
    prediction_result = get_prediction(
                image=image,
                craft_net=craft_net,
                refine_net=refine_net,
                text_threshold=0.7,
                link_threshold=0.4,
                low_text=0.4,
                cuda=False,
                long_size=600,
            )



    ocr_text = []
    ocr_text1 = []
    print('start ocr')
    idx = 0
    for contour in prediction_result["boxes"]:
        idx +=1
        x,y,w,h = cv2.boundingRect(contour)
        if y < 1000:
            new_image = img[y-10:y+h+5, x-10:x+w]
            height, width = new_image.shape[:2]
            if height > width:
                new_image = cv2.rotate(new_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            new_image1 = correct_skew(new_image)
            img2 = super_res(new_image1) 
            gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
            opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=1)
            invert = 255 - opening
            bordur = cv2.copyMakeBorder(invert, 10,10,10,10, cv2.BORDER_CONSTANT, 1, (255,255,255))
            cstm_config = r"-l rus+eng --psm 6 --oem 3"
            text = pytesseract.image_to_string(bordur, config=cstm_config)
            text = text.replace("\n", ' ')
            ocr_text.append(text)
        else:
            new_image = img[y-10:y+h+5, x-10:x+w]
            height, width = new_image.shape[:2]
            if height > width:
                new_image = cv2.rotate(new_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            new_image1 = correct_skew(new_image)
            img2 = super_res(new_image1) 
            gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
            opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=1)
            invert = 255 - opening
            bordur = cv2.copyMakeBorder(invert, 10,10,10,10, cv2.BORDER_CONSTANT, 1, (255,255,255))
            cstm_config = r"-l rus+eng --psm 6 --oem 3"
            text = pytesseract.image_to_string(bordur, config=cstm_config)
            text = text.replace("\n", ' ')
            ocr_text1.append(text)
    recognize_text_1 = text_recognize(ocr_text, ocr_text1)
    return recognize_text_1
    
def text_recognize(ocr_text, ocr_text1):
    del ocr_text[0]
    text = str(ocr_text)
    text = re.sub(r"[^А-Яа-яA-Z\d\-\s\—\.]",'', text)
    text1 = str(ocr_text1)
    text1 = re.sub(r"[^А-Яа-яA-Z\d\-\s\—\.]",'', text1)
    try:
        gender = re.findall(r"(муж|МУЖ|ЖЕН|жен)+", text1)[0]
    except:
        gender = 'Not found'
    try:
        issued_number = re.findall(r"\d{3}[—|-]\S{3}", text)[0]
    except:
        issued_number = 'Not found'
    try:
        issued_date = re.findall(r"(\d{2}\.\d{2}\.\d{4})", text)[0]
    except:
        issued_date = 'Not found'
    try:
        date_birth = re.findall(r"(\d{2}\.\d{2}\.\d{4})", text1)[0]
    except:
        date_birth = 'Not found'
    try:
        data_number = re.sub(r"['\[\]]", '', str(ocr_text))
        series = re.findall(r"\s\d{2}\s", data_number)
        series = re.sub(r"[^\d]",'',str(series))
        number = re.findall(r"\d{6}", str(data_number))[0]
    except:
        series = 'Not Found'
        number = 'Not Found'
    text1 = re.sub(r"[^А-Я\.]+", ' ', text)
    FIO = recognize_fio(ocr_text1)
    data = {
        "issued_date": issued_date, 
        "division": issued_number,
        "birthday": date_birth,
        "gender": gender,
        "issued": text1,
        "fio" : FIO ,
        "series": series,
        "number":number
    }
    return data

def recognize_fio(ocr_text):
    def recognize(data):
        text = re.sub(r"[^А-Я\s]",'',data)
        text = re.sub(r"\s{2,}", '',text)
        return text
    data = ocr_text
    del data[0]
    result = list(map(recognize, data))
    result = list(filter(None, result))
    result = [x for x in result if len(x) > 2]
    def lower(text):
        text = text.title()
        return text
    result = list(map(lower, result))

    FIO = (' ').join(result[0:3])

    return FIO

def super_res(img):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()

    sr.readModel(path)

    sr.setModel("fsrcnn",4)

    result = sr.upsample(img)

    return result

def correct_skew(image, delta=0.6, limit=10):
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
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv2.BORDER_REPLICATE)

    return rotated

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run("main:app", host='0.0.0.0', port=port,
                  log_level="info", reload=True, workers=1)