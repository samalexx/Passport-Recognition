import cv2
import numpy as np
import pytesseract
import re
from scipy.ndimage import interpolation as inter
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
)
refine_net = load_refinenet_model(cuda=False)
craft_net = load_craftnet_model(cuda=False)
path = "models/FSRCNN_x4.pb"
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(path)
sr.setModel("fsrcnn",4)

def main_pass_first(data):

    result = text_block(data)
    
    return result

def text_block(image):
    img = cv2.resize(image, (2700, 3200))
    image = read_image(img)

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
    idx = 0
    for contour in prediction_result["boxes"]:
        idx +=1
        x,y,w,h = cv2.boundingRect(contour)
        new_image = img[y-10:y+h+5, x-10:x+w]
        if y < 1000:
            text = preprocess_text_bocks(new_image)
            ocr_text.append(text)
        else:
            text = preprocess_text_bocks(new_image)
            ocr_text1.append(text)
    recognize_text_1 = text_recognize(ocr_text, ocr_text1)
    return recognize_text_1
    

def preprocess_text_bocks(new_image):
    height, width = new_image.shape[:2]
    if height > width:
        new_image = cv2.rotate(new_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    new_image1 = correct_skew(new_image)
    img2 = sr.upsample(new_image1) 
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening
    border = cv2.copyMakeBorder(invert, 10,10,10,10, cv2.BORDER_CONSTANT, 1, (255,255,255))
    cstm_config = r"-l rus+eng --psm 6 --oem 3"
    text = pytesseract.image_to_string(border, config=cstm_config)
    text = text.replace("\n", ' ')
    return text


def text_recognize(ocr_text, ocr_text1):
    print(ocr_text, ocr_text1)
    text = str(ocr_text[1:])
    text = re.sub(r"[^А-Яа-яA-Z\d\-\s\—\.]",'', text)
    text1 = str(ocr_text1)
    text1 = re.sub(r"[^А-Яа-яA-Z\d\-\s\—\.]",'', text1)
    try:
        gender = re.findall(r"(муж|МУЖ|ЖЕН|жен|ЖеН)+", text1)[0]
    except IndexError:
        gender = 'Not found'
    try:
        issued_number = re.findall(r"\S{0,3}[—|-]\S{0,3}", text)[0]
    except IndexError:
        issued_number = 'Not found'
    try:
        issued_date = re.findall(r"(\d{2}\.\d{2}\.\d{4})", text)[0]
    except IndexError:
        issued_date = 'Not found'
    try:
        date_birth = re.findall(r"(\d{2}\.\d{2}\.\d{4})", text1)[0]
    except IndexError:
        date_birth = 'Not found'
    try:
        data_number = re.sub(r"['\[\]]", '', str(ocr_text))
        series = re.findall(r"\s\d{2}\s", data_number)
        series = re.sub(r"[^\d]",'',str(series))
        number = re.findall(r"\d{6}", str(data_number))[0]
    except IndexError:
        series = 'Not Found'
        number = 'Not Found'
    text1 = re.sub(r"[^А-Я\.]+", ' ', text)
    text1 = re.sub(r"\.", "", text1)
    
    FIO = recognize_fio(ocr_text1)[0:3]
    FIO = re.sub(r'[^А-Яа-яёЁA-Za-z\s]', '', str(FIO))
    FIO = re.sub(r"(Муж|Мух|Жен)+", '', str(FIO))
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
    result = list(map(recognize, data))
    result = list(filter(None, result))
    result = [x for x in result if len(x) > 2]

    def lower(text):
        text = text.title()
        return text
    result = list(map(lower, result))
    print(result)
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