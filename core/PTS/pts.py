from craft_text_detector import (read_image, load_craftnet_model, load_refinenet_model, get_prediction,)
from PIL import Image
import numpy as np
import io
from scipy.ndimage import interpolation as inter
from core.PTS.text_recognizer import get_data
import cv2
import easyocr
import pytesseract
path = "models/FSRCNN_x4.pb"
sr = cv2.dnn_superres.DnnSuperResImpl_create()
reader = easyocr.Reader(['en'], gpu=False)
cstm_config = r"-l eng+frn --psm 10 --oem 3 tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
sr.readModel(path)
sr.setModel("fsrcnn",4)
refine_net = load_refinenet_model(cuda=False)
craft_net = load_craftnet_model(cuda=False)


def pts_start(data):
    image = data

    h,w = image.shape[:2]
    print(h,w)
    if w > 800 and h > 1200:
        img = cv2.resize(image, (3100, 3500))
        image = read_image(img)
        prediction_result = get_prediction(
            image=image,
            craft_net=craft_net,
            refine_net=refine_net,
            text_threshold=0.9,
            link_threshold=0.4,
            low_text=0.4,
            cuda=False,
            long_size=1280
        )
        ocr_text = []
        ocr_text1 = []
        boxes = prediction_result["boxes"]
        for box in boxes[1:]:
            x,y,w,h = cv2.boundingRect(box)
            resize_image = img[y:y+h, x:x+w]
            new_image1 = correct_skew(resize_image)
            new_image_sr = sr.upsample(new_image1)
            gray = cv2.cvtColor(new_image_sr, cv2.COLOR_BGR2GRAY)
            result_easy = reader.readtext(gray, detail=0)
            result_tes = pytesseract.image_to_string(gray,config=cstm_config)
            ocr_text.append(result_easy)
            ocr_text1.append(result_tes)

        result_text = ocr_text+ocr_text1
        result = get_data(result_text)
        return result
    else:
        return {'Size error':'Image dimension is too small'}


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