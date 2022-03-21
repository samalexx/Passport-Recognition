import cv2
from matplotlib.pyplot import text
import numpy as np
import pytesseract
from PIL import Image
import io
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
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'



def text_block(image):
    img = cv2.resize(image, (2700, 3000), fx=0.5, fy=0.3)

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
                long_size=1280,
            )
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
        text = pytesseract.image_to_string(invert, config = '--psm 11 --oem 3 -l rus')
        text = text.replace("\n", ' ')
        ocr_text.append(text)
    print(ocr_text)
    return(ocr_text)

def super_res(img):
    path = "models/FSRCNN_x4.pb"
    print('start super res')
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
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)

    return rotated

image = np.array(Image.open('C:\Games\passport/20.jpeg')) # can be filepath, PIL image or numpy array

result = text_block(image)