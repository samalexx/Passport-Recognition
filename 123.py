import re
import cv2
import numpy as np
import cv2
import pytesseract
import numpy as np
import pytesseract
import cv2
from scipy.ndimage import interpolation as inter

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def find_rectangle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_blur = cv2.GaussianBlur(gray, (3,3), 1)

    obr_image = cv2.Canny(image_blur, 100, 300, 4)

    contours, _ = cv2.findContours(obr_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   

    c = sorted(contours, key = cv2.contourArea, reverse = True)[0]

    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    x,y,w,h = cv2.boundingRect(box)

    crop = image[y-5:y+h+5, x-5:x+w+5]

    return crop


def correct_skew(img, delta=0.2, limit=10):
    image = img
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
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
              borderMode=cv2.BORDER_REPLICATE)
    return rotated

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

def draw_contour(image, c, i):
	# compute the center of the contour area and draw a circle
	# representing the center
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	# draw the countour number on the image
	cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (255, 255, 255), 2)
	# return the image with the contour number drawn on it
	return image

# contour find and draw in input image
def draw_contours(crop):
    img = crop[40:120, 0:320]

    table = correct_skew(img)

    table = cv2.resize(table, (500,300))

    result = table.copy()
    gray = cv2.cvtColor(table,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35,1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (0,0,0), 4)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,70))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=3)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (0,0,0), 4)

    return result, table

def recognize_text(result, table):
    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray_result, (3,3), 2)

    cannys = cv2.Canny(blur, 50,350, 4)

    cont, _ = cv2.findContours(cannys, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_ocr = []
    idx = 0
    for con in cont:
        x,y,w,h = cv2.boundingRect(con)
        if w > 100 and h > 10 and 20 < y < 200 and x < 300:
            idx += 1
            cv2.rectangle(table, (x, y), (x + w, y + h), (36,255,12), 2)
            # cv2.putText(table, f'{idx}', (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255), 2)
            new_image = table[y-2:y+h+2, x-2:x+w+2]
            new_image = correct_skew(new_image, 0.5, 5)
            img = cv2.resize(new_image, (355, 75))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            invert = 255 - opening
            custom_config = r'-l rus+frn --psm 6 --oem 1'
            text = pytesseract.image_to_string(invert, config=custom_config)
            text = text.replace("\n", '')
            text_ocr.append(text.capitalize())
            filename = "C:/Games/STS/box/file_%d.png"%idx
            cv2.imwrite(filename, invert)
    return text_ocr

image = cv2.imread('C:/Games/STS/box/.jpeg')

crop_image = find_rectangle(image)

rectangles_contour, table = draw_contours(crop_image)

text = recognize_text(rectangles_contour, table)

text = str(text)
text = re.sub(r"[\]\[\—\"§|!|'|©|®|_|№|`’]", '', text)
print(text)
cv2.imshow('C:/Games/STS/box/123.jpeg', table) 
cv2.waitKey(0)
cv2.destroyAllWindows()

