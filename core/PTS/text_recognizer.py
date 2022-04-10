# -*- coding: utf-8 -*-
import re
from fake_useragent import UserAgent
import requests
import json

ua = UserAgent()

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
def get_data(tes_easy):
    print('start recognize text')
    def get_vin(data):
        date_birth = re.findall(r"[A-Z0-9]{15,17}", str(data))
        return date_birth

    series_number = list(map(get_vin, tes_easy))
    texting = list(filter(None, series_number))
    result = get_vehicle(texting, tes_easy)
    return result


def get_number(ocr_text):
    str = ocr_text[0:3]
    summ = str[1]+str[2]+str[0]
    print(summ)
    return summ

def get_date(ocr_text):
    text = str(ocr_text)
    date = re.findall(r"[0-9]{2}\.[0-9]{2}\.[0-9]{4}", text)[0]
    return(date)

def get_vehicle(data, ocr_text):
    for vin in data:
        print(vin)
        params = {'vin': vin[0],
            'checkType':'history',
            'User-agent':ua.random}
        resp = requests.post("https://сервис.гибдд.рф/proxy/check/auto/history", data=params)
        if resp.status_code == 200:
            data = json.loads(resp.text)
            items = data['RequestResult']['vehicle']
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
        else:
            return {'In image not found VIN':'In image not found VIN number'}