import cv2
from matplotlib.pyplot import text
import numpy as np
import pytesseract
from PIL import Image
import re
import io
import imutils
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
    print('start ocr')
    idx = 0
    for contour in prediction_result["boxes"]:
        idx +=1
        x,y,w,h = cv2.boundingRect(contour)
        new_image = img[y-10:y+h+5, x-10:x+w]
        new_image1 = correct_skew(new_image)
        img2 = super_res(new_image1)
        height, width = img.shape[:2]     
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        filename = 'C:\Games\passport\crop_pass/filename_%i.jpeg'%idx
        cv2.imwrite(filename, gray)
        thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=1)
        invert = 255 - opening
        bordur = cv2.copyMakeBorder(invert, 10,10,10,10, cv2.BORDER_CONSTANT, 1, (255,255,255))
        cstm_config = r"-l rus+eng --psm 6 --oem 3"
        text = pytesseract.image_to_string(bordur, config=cstm_config)
        text = text.replace("\n", ' ')
        ocr_text.append(text)
    result = data_to_text(ocr_text)
    print(result)
    return result

    

def data_to_text(ocr_text):
    text = str(ocr_text)
    print(text)
    text = re.sub(r"[^А-Яа-яA-Z\d\-\s\—\.]",'', text)
    text = re.sub(r"\s+", ' ', text)
    print(text)

    try:
        gender = re.findall(r"(муж|МУЖ|ЖЕН|жен)+", text)[0]
    except:
        gender = 'not find'
    try:
       issued_number = re.findall(r"\d{3}[—|-]\S{3}", text)[0]
    except:
        issued_number = 'not find'
    try:
        date_birth = re.findall(r"(\d{2}\.\d{2}\.\d{4})", text)[1]
    except:
        date_birth = 'not find'
    try:
        issued_date = re.findall(r"(\d{2}\.\d{2}\.\d{4})", text)[0]
    except:
        issued_date = ''

    
    data = {
        "issued_date": issued_date, 
        "division": issued_number,
        "birthday": date_birth,
        "gender": gender
    }
    return data

def super_res(img):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()

    sr.readModel(path)

    sr.setModel("fsrcnn",4)

    result = sr.upsample(img)

    return result

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
        _, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)

    return rotated

image = np.array(Image.open('C:\Games\passport/27.jpeg')) # can be filepath, PIL image or numpy array

result = text_block(image)