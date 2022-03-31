from craft_text_detector import (read_image, load_craftnet_model, load_refinenet_model, get_prediction,
)
from PIL import Image
import numpy as np
import io
from core.PTS.text_recognizer import get_data
import cv2
import pytesseract
path = "models/FSRCNN_x4.pb"
sr = cv2.dnn_superres.DnnSuperResImpl_create()
CSTM_CONFIG = r'-l eng+rus --psm 6 --oem 1'
refine_net = load_refinenet_model(cuda=False)
craft_net = load_craftnet_model(cuda=False)


def pts_start(data):
    image = np.array(Image.open(io.BytesIO(data)))

    image = read_image(image)

    prediction_result = get_prediction(
        image=image,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4,
        cuda=False,
        long_size=1280
    )
 
    idx = 0
    ocr_text = []
    boxes = prediction_result["boxes"]
    for box in boxes[1:]:
        idx+=1
        x,y,w,h = cv2.boundingRect(box)
        resize_image = image[y:y+h, x:x+w]
        super_image = super_res(resize_image)
        text = pytesseract.image_to_string(super_image, config=CSTM_CONFIG)
        ocr_text.append(text.replace("\n", ' '))
    # result = get_data(ocr_text)
    return ocr_text

def super_res(img):
    sr.readModel(path)
    sr.setModel("fsrcnn",4)
    result = sr.upsample(img)
    return result
