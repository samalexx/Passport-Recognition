# -*- coding: utf-8 -*-
import re
import requests
import json

dict_keys = {"01":"Грузовой автомобиль бортовой",
        "02":"Грузовой автомобиль шасси",
        "03":"Грузовой автомобиль фургоны",
        "04":"Грузовой автомобиль тягачи седельной",
        "05":"Грузовой автомобиль самосвалы",
        "06":"Грузовой автомобиль рефрижераторы",
        "07":"Грузовой автомобиль цистерны",
        "08":"Грузовой автомобиль с гидроманипулятором",
        "09":"Грузовой автомобиль прочие",
        "21":"Легковой автомобиль универсал",
        "22":"Легковой автомобиль комби (хэтчбек)",
        "23":"Легковой автомобиль седан",
        "24":"Легковой автомобиль лимузин",
        "25":"Легковой автомобиль купе",
        "26":"Легковой автомобиль кабриолет",
        "27":"Легковой автомобиль фаэтон",
        "28":"Легковой автомобиль пикап",
        "29":"Легковой автомобиль прочие",
        "41":"Автобус длиной не более 5 м",
        "42":"Автобус длиной более 5 м, но не более 8 м",
        "43":"Автобус длиной более 8 м, но не более 12 м",
        "44":"Автобус сочлененные длиной более 12 м",
        "49":"Автобус прочие",
        "51":"Специализированные автомобили автоцистерны",
        "52":"Специализированные автомобили санитарные",
        "53":"Специализированные автомобили автокраны",
        "54":"Специализированные автомобили заправщики",
        "55":"Специализированные автомобили мастерские",
        "56":"Специализированные автомобили автопогрузчики",
        "57":"Специализированные автомобили эвакуаторы",
        "58":"Специализированные пассажирские транспортные средства",
        "59":"Специализированные автомобили прочие",
        "71":"Мотоциклы",
        "72":"Мотороллеры и мотоколяски",
        "73":"Мотовелосипеды и мопеды",
        "74":"Мотонарты",
        "80":"Прицепы самосвалы",
        "81":"Прицепы к легковым автомобилям",
        "82":"Прицепы общего назначения к грузовым автомобилям",
        "83":"Прицепы цистерны",
        "84":"Прицепы тракторные",
        "85":"Прицепы вагоны-дома передвижные",
        "86":"Прицепы со специализированными кузовами",
        "87":"Прицепы трейлеры",
        "88":"Прицепы автобуса",
        "89":"Прицепы прочие",
        "91":"Полуприцепы с бортовой платформой",
        "92":"Полуприцепы самосвалы",
        "93":"Полуприцепы фургоны",
        "95":"Полуприцепы цистерны",
        "99":"Полуприцепы прочие",
        "31":"Трактора",
        "32":"Самоходные машины и механизмы",
        "33":"Трамваи",
        "34":"Троллейбусы",
        "35":"Велосипеды",
        "36":"Гужевой транспорт",
        "38":"Подвижной состав железных дорог",
        "39":"Иной"}
def get_data(ocr_text):
    texting = list(map(subchik, ocr_text))
    texting = list(filter(None, texting))

    find_vin_text = re.findall(r"[A-Z0-9]{10,17}", str(ocr_text))[0]
    print(find_vin_text)

    print(texting)

    data = get_vehicle(find_vin_text, texting)
    return data

def find_vin(ocr_text):
    vin = re.findall(r"[A-Z]\w{16}", ocr_text)
    return vin

def find_number_series(ocr_text):
    number = str(re.findall(r"\d{6}", ocr_text))
    return number

def subchik(ocr_text):
    text = re.sub(r"[^A-Za-z\d\-\s\—\.\(\)\,]", '', ocr_text)
    text = str(text)
    text = text.strip()
    print(text)
    return text

def get_number(ocr_text):
    str = ocr_text[0:3]
    summ = str[1]+str[2]+str[0]
    print(summ)
    return summ

def get_date(ocr_text):
    text = str(ocr_text)
    date = re.findall(r"[0-9]{2}\.[0-9]{2}\.[0-9]{4}", text)[0]
    return(date)

def get_vehicle(vin_number, ocr_text):
    params = {'vin': vin_number,
        'checkType':'history'}
    
    resp = requests.post("https://сервис.гибдд.рф/proxy/check/auto/history", data=params)

    data = json.loads(resp.text)
    try:
        items = data['RequestResult']['vehicle']
    except IndexError and KeyError:
        return {'In image not found VIN':'In image not found VIN number'}

    type = items['type']
    engine_Hp = items['powerHp']
    engine_powerKwt = items['powerKwt']
    engine = f'({engine_powerKwt}){engine_Hp}'
    data = {
        'number': get_number(ocr_text),
        'vin': items['vin'],
        'model':items['model'],
        'type':dict_keys.get(type),
        'categoty':items['category'],
        'year':items['year'],
        'engine':engine,
        'date':get_date(ocr_text)
    }
    return data