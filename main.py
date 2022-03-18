from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction
)

import matplotlib.pyplot as plt
import torch
from matplotlib.pyplot import text
import pytesseract
import numpy as np
from PIL import Image
import os
import io
import cv2
from fastapi import FastAPI, File, Query, UploadFile
import uvicorn



device = torch.device('cpu')
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'


app = FastAPI()

@app.post("/predict/", tags=["Predict"], summary="Predict")
async def upload(file: UploadFile = File(..., description='Выберите файл для загрузки')):
    data = await file.read()

    image = np.array(Image.open(io.BytesIO(data))) # can be filepath, PIL image or numpy array

    img = cv2.resize(image, (1900, 2300), fx=0.5, fy=0.3)

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
    print(prediction_result["boxes"])
    ocr_text = []
    print('start ocr')
    for contour in prediction_result["boxes"]:
        x,y,w,h = cv2.boundingRect(contour)
        new_image = img[y-10:y+h+5, x-20:x+w]
        img2 = super_res(new_image)
        img_res = cv2.resize(img2, (700, 95))
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=1)
        invert = 255 - opening
        invert = cv2.copyMakeBorder(invert, 20,20,20,20, cv2.BORDER_CONSTANT, 3, (255,255,255))
        text = pytesseract.image_to_string(invert, config = f'--psm 6 --oem 3 -l rus+eng')
        text = text.replace("\n", ' ')
        print(text)
        ocr_text.append(text)
    return ocr_text

def super_res(img):
    path = "models/FSRCNN_x3.pb"
    print('start super res')
    sr = cv2.dnn_superres.DnnSuperResImpl_create()

    sr.readModel(path)

    sr.setModel("fsrcnn",3)

    result = sr.upsample(img)

    return result

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run("main:app", host='0.0.0.0', port=port,
                  log_level="info", reload=True, workers=1)