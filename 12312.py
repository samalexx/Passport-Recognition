from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction
)
import pytesseract
import cv2
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# set image path and export folder directory
img = cv2.imread('C:\Games\STS\sts/33.jpeg') # can be filepath, PIL image or numpy array
output_dir = 'C:\Games\STS/box/ss'

img = cv2.resize(img, (1500, 2000), fx=0.5, fy=0.3)

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
    long_size=1280
)

# export heatmap, detection points, box visualization
print(prediction_result["boxes"])
for contour in prediction_result["boxes"]:
    x,y,w,h = cv2.boundingRect(contour)
    new_image = img[y-10:y+h+5, x-20:x+w]
    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening
    print(pytesseract.image_to_string(invert,
                                    config = f'--psm 11 --oem 3 -l eng+rus'))