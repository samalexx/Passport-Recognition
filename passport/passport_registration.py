import cv2
from matplotlib.pyplot import gray
from scipy.ndimage import interpolation as inter
import numpy as np
import io
import os
import uvicorn
from PIL import Image
from fastapi import FastAPI, File, Query, UploadFile
import pytesseract
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
)
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
path = "models/FSRCNN_x4.pb"
sr = cv2.dnn_superres.DnnSuperResImpl_create()
refine_net = load_refinenet_model(cuda=False)
craft_net = load_craftnet_model(cuda=False)

app = FastAPI()

@app.post("/predict/", tags=["Predict"], summary="Predict")
async def upload(file: UploadFile = File(..., description='Выберите файл для загрузки')):
    data = await file.read()

    image11 = np.array(Image.open(io.BytesIO(data)))

    result = find_cont(image11)

    return result

def find_cont(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.medianBlur(gray, 3)

    edges = cv2.Canny(blur, 75,300)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    idx = 0
    for c in contours:
        idx += 1
        x,y,w,h = cv2.boundingRect(c)
        if 300 < w < 1000 and 100 < h < 800 and y > 50 and x < 100:
            print(idx, x,y,w,h)
            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            # cv2.putText(img, f'{idx}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
            new_image = img[y:y+h, x:x+w]
            img2 = super_res(new_image)
            result = recognize_box(img2)
    return result



def recognize_box(new_image):
    image = read_image(new_image)

    # perform prediction 
    prediction_result = get_prediction(
                image=image,
                craft_net=craft_net,
                refine_net=refine_net,
                text_threshold=0.7,
                link_threshold=0.4,
                low_text=0.4,
                cuda=False,
                long_size=550,
            )
    idx = 0
    ocr_text = []
    for contour in prediction_result["boxes"]:
        idx +=1
        x,y,w,h = cv2.boundingRect(contour)
        resize_image = image[y:y+h, x:x+w]
        # new_image1 = correct_skew(resize_image)
        gray = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)
    # Apply dilation and erosion to remove some noise
        # thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        # opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=1)
        # invert = 255 - opening
        filename  = 'C:\Games\crop_pass/file_%i.jpeg'%idx 
        cv2.imwrite(filename, gray)
        cstm_config = r"-l rus --psm 6 --oem 3" 
        text = pytesseract.image_to_string(gray, config=cstm_config)
        text = text.replace("\n", ' ')
        print(text)
        ocr_text.append(text)
    return ocr_text


def super_res(img):

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