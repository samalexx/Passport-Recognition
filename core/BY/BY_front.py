from craft_text_detector import (read_image, load_craftnet_model, load_refinenet_model, get_prediction,
)
import cv2
import pytesseract
from core.BY.regex_BY import regex_main
path = "models/FSRCNN_x4.pb"
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(path)
sr.setModel("fsrcnn",4)
refine_net = load_refinenet_model(cuda=False)
craft_net = load_craftnet_model(cuda=False)


def by_start(data):
    image = data

    h,w = image.shape[:2]
    print(h,w)
    image = cv2.resize(image, (1100,650))
    image = read_image(image)


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
    boxes = prediction_result["boxes"]
    for box in boxes[1:]:
        x,y,w,h = cv2.boundingRect(box)
        if y > 130:
            resize_image = image[y:y+h, x:x+w]
            custom_config = r'-l rus+eng --psm 6 --oem 3'
            text = pytesseract.image_to_string(resize_image, config=custom_config)
            text = text.replace("\n", ' ')
            ocr_text.append(text)
    result = regex_main(ocr_text)
    return result
