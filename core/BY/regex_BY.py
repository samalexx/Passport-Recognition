import re


def regex_main(data):
    series_number = list(map(get_series, data))
    series_number = list(filter(None, series_number))

    date_birthday = list(map(get_birthday, data))
    date_birthday = list(filter(None, date_birthday))


    surname = list(map(get_name_surname, data))
    print(surname)
    data = {
        "surname_rus": surname[0],
        "surname_eng": surname[1],
        "firtsname": surname[2],
        'birthday':date_birthday[0][0],
        'series':date_birthday[1][0],
        'series': series_number[0][0]
    }
    
    return data

def get_name_surname(data):
    print(data)
    text = re.sub(r"[^А-Яа-яёЁA-Za-z\s]",'',data)
    print(text)
    return text

def get_birthday(data):
    date_birth = re.findall(r"\d{2}\.\d{2}\.\d{4}", data)
    return date_birth

def get_series(data):
    series = re.sub('\s', '', data)
    series = re.findall('\d{10}', series)
    return series

