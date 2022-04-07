from ast import Pass
from unittest import result
import numpy as np
import pytesseract
import uvicorn
from fastapi import FastAPI, File, Query, UploadFile
import io
import os
from PIL import Image
from core.BY import back_side, BY_front
from core.sts import sts, srts_front
from core.passport import pass_first, passport_registration
from core.PTS import pts
from pdf2image import convert_from_bytes
import platform



plt = platform.system()
if plt == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
    poppler_path = r'C:/Python38/Lib/poppler/Library/bin'
    temp_path = r'app/tmp'
elif plt == "Linux":
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    poppler_path = None
    temp_path = r'app/tmp'
else:
    print("Unidentified system")
app = FastAPI()

@app.post("/predict/", tags=["Predict"], summary="Predict")
async def upload(file: UploadFile = File(..., description='Выберите файл для загрузки',), 
        template: str = Query("Водительское удостворение", enum=[1,2,3,4], description='1 - Водительское удостоверение, 2 - СТС, 3 - Паспорт, 4 - ПТС'), 
        mode: str = Query("front", enum=["front", "back"], description='Choice doc template')):
    data = await file.read()
    if template == '1' and mode == 'front':
        result_by_front = BY_front.by_start(data) 
        return result_by_front
    if template == '1' and mode == 'back':
        result_by_back = back_side.side_main(data)
        return result_by_back
    if template == '2' and mode == 'front':
        print(file.filename)
        result_srts_front = srts_front.main_srts_front(data)
        return result_srts_front
    if template == '2' and mode == 'back':
        result_sts = sts.sts_main(data)
        return result_sts
    if template == '3' and mode == 'front':
        result_passport = pass_first.main_pass_first(data)
        return result_passport
    if template == '3' and mode == 'back':
        result_registration = passport_registration.passport_registration(data)
        return result_registration
    if template == '4':
        result_pts = pts.pts_start(data)
        return result_pts
# main 
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8005))
    uvicorn.run("main:app", host='0.0.0.0', port=port,
                  log_level="info", reload=True, workers=1) 