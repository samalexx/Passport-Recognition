from craft_text_detector import (read_image, load_craftnet_model, load_refinenet_model, get_prediction,
)
from PIL import Image
import numpy as np
import io
import cv2
import pytesseract
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
    
    for contour in prediction_result["boxes"]:
        idx+=1
        x,y,w,h = cv2.boundingRect(contour)
        print(x,y,w,h)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 1)
        cv2.putText(image, f'{idx}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.imwrite('/content/1231232312.jpeg',image)
        resize_image = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(image, config=CSTM_CONFIG)
        print(text)
    return text