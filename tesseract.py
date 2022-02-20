import pytesseract
from PIL import Image
import cv2
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

img = cv2.imread('C:/Games/STS/box/file_6.png')



custom_config = r'-l eng --psm 6 --oem 3'

text = pytesseract.image_to_string(img, config=custom_config)

print(text)