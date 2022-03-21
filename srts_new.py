from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction
)
import requests
import json
import imutils
from matplotlib.pyplot import text
import pytesseract
import numpy as np
from scipy.ndimage import interpolation as inter
from PIL import Image
import os
import io
import cv2
from fastapi import FastAPI, File, Query, UploadFile
import uvicorn
import re



pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'


app = FastAPI()

@app.post("/predict/", tags=["Predict"], summary="Predict")
async def upload(file: UploadFile = File(..., description='Выберите файл для загрузки'),
mode: str = Query("front", enum=["front", "back"], description='Choice doc template')):
    data = await file.read()

    image = np.array(Image.open(io.BytesIO(data))) # can be filepath, PIL image or numpy array
    if mode == 'front':
        img = cv2.resize(image, (2700, 3000), fx=0.5, fy=0.3)
        image = read_image(img)


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
            long_size=1280,
        )

        # export heatmap, detection points, box visualization
        ocr_text = []
        print('start ocr')
        idx = 0
        for contour in prediction_result["boxes"]:
            idx +=1
            x,y,w,h = cv2.boundingRect(contour)
            new_image = img[y-10:y+h+5, x-20:x+w]
            new_image1 = correct_skew(new_image)
            img2 = super_res(new_image1)
            gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
            opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=1)
            invert = 255 - opening
            invert = cv2.copyMakeBorder(invert, 20,20,20,20, cv2.BORDER_CONSTANT, 3, (255,255,255))
            text = pytesseract.image_to_string(invert, config = '--psm 11 --oem 3 -l eng+rus')
            text = text.replace("\n", ' ')
            ocr_text.append(text)
        print(ocr_text)
        result = fix_text(ocr_text)
        return result
    elif mode == 'back':
        result = lic_str(image)
        return result

def fix_text(ocr_text):
    text1 = str(ocr_text)
    main_text = re.sub(r'[^А-Яа-я\w\,]', '', text1)
    print(main_text)
    vin_number = re.findall(r"VIN\,(.{0,17})", main_text)[0]
    print(vin_number)
    number_plate = re.findall(r"знак(.{0,9})", main_text)
    data = get_vehicle(vin_number, number_plate)

    return data

def get_vehicle(vin_number, number_plate):
    print(vin_number)
    params = {'vin': vin_number,
        'checkType':'history'}
    
    resp = requests.post("https://сервис.гибдд.рф/proxy/check/auto/history", data=params)

    data = json.loads(resp.text)

    items = data['RequestResult']['vehicle']

    dict_keys = {"01":"Грузовой автомобиль бортовой",
        "02":"Грузовой автомобиль шасси",
        "03":"Грузовой автомобиль фургоны",
        "04":"Грузовой автомобиль тягачи седельной",
        "05":"Грузовой автомобиль самосвалы",
        "06":"Грузовой автомобиль рефрижераторы",
        "07":"Грузовой автомобиль цистерны",
        "08":"Грузовой автомобиль с гидроманипулятором",
        "09":"Грузовой автомобиль прочие",
        "21":"Легковой автомобиль универсал",
        "22":"Легковой автомобиль комби (хэтчбек)",
        "23":"Легковой автомобиль седан",
        "24":"Легковой автомобиль лимузин",
        "25":"Легковой автомобиль купе",
        "26":"Легковой автомобиль кабриолет",
        "27":"Легковой автомобиль фаэтон",
        "28":"Легковой автомобиль пикап",
        "29":"Легковой автомобиль прочие",
        "41":"Автобус длиной не более 5 м",
        "42":"Автобус длиной более 5 м, но не более 8 м",
        "43":"Автобус длиной более 8 м, но не более 12 м",
        "44":"Автобус сочлененные длиной более 12 м",
        "49":"Автобус прочие",
        "51":"Специализированные автомобили автоцистерны",
        "52":"Специализированные автомобили санитарные",
        "53":"Специализированные автомобили автокраны",
        "54":"Специализированные автомобили заправщики",
        "55":"Специализированные автомобили мастерские",
        "56":"Специализированные автомобили автопогрузчики",
        "57":"Специализированные автомобили эвакуаторы",
        "58":"Специализированные пассажирские транспортные средства",
        "59":"Специализированные автомобили прочие",
        "71":"Мотоциклы",
        "72":"Мотороллеры и мотоколяски",
        "73":"Мотовелосипеды и мопеды",
        "74":"Мотонарты",
        "80":"Прицепы самосвалы",
        "81":"Прицепы к легковым автомобилям",
        "82":"Прицепы общего назначения к грузовым автомобилям",
        "83":"Прицепы цистерны",
        "84":"Прицепы тракторные",
        "85":"Прицепы вагоны-дома передвижные",
        "86":"Прицепы со специализированными кузовами",
        "87":"Прицепы трейлеры",
        "88":"Прицепы автобуса",
        "89":"Прицепы прочие",
        "91":"Полуприцепы с бортовой платформой",
        "92":"Полуприцепы самосвалы",
        "93":"Полуприцепы фургоны",
        "95":"Полуприцепы цистерны",
        "99":"Полуприцепы прочие",
        "31":"Трактора",
        "32":"Самоходные машины и механизмы",
        "33":"Трамваи",
        "34":"Троллейбусы",
        "35":"Велосипеды",
        "36":"Гужевой транспорт",
        "38":"Подвижной состав железных дорог",
        "39":"Иной"}  

    model = items['model']
    vin = items['vin']
    year = items['year']
    category = items['category']
    type = items['type']
    type_ts = dict_keys.get(type)
    engine_Hp = items['powerHp']
    engine_powerKwt = items['powerKwt']
    engine = f'{engine_powerKwt}/{engine_Hp}'
    data = {
        'number_plate':number_plate,
        'vin': vin,
        'model':model,
        'type':type_ts,
        'categoty':category,
        'year':year,
        'engine':engine
    }
    return data

def correct_skew(image, delta=0.3, limit=10):
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
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)

    return rotated

def super_res(img):
    path = "models/FSRCNN_x4.pb"
    print('start super res')
    sr = cv2.dnn_superres.DnnSuperResImpl_create()

    sr.readModel(path)

    sr.setModel("fsrcnn",4)

    result = sr.upsample(img)

    return result

def lic_str(img):
    bst_image= cv2.resize(img, (800, 1200))

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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run("main:app", host='0.0.0.0', port=port,
                  log_level="info", reload=True, workers=1)