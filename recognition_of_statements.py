import time

import pytesseract
from pytesseract import TesseractError
import spacy
import json
import os
import numpy as np
import re

from flask import Flask, send_file, request, send_from_directory, flash, redirect, render_template, jsonify, get_flashed_messages
from werkzeug.utils import secure_filename
from apscheduler.schedulers.background import BackgroundScheduler

import cv2
import shutil
import tempfile

from pdf2image import convert_from_path

from PIL import ImageEnhance
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

scheduler = BackgroundScheduler()
scheduler.start()

app.config['UPLOAD_FOLDER'] = '/app/uploads'
ALLOWED_EXTENSIONS = {'pdf'}

def clear_upload_folder():
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    app.logger.info("Папка загрузок очищена.")

scheduler.add_job(func=clear_upload_folder, trigger='interval', hours=24)

import numpy as np
from PIL import Image
import pytesseract

def is_blank_page(image, text_threshold=5, pixel_threshold=1):
    """
    Проверяет, является ли страница пустой, основываясь на количестве текста и яркости изображения.
    :param image: Изображение страницы
    :param text_threshold: Порог для количества символов, чтобы считать страницу пустой
    :param pixel_threshold: Порог для яркости изображения (0.99 означает почти белая страница)
    :return: True, если страница пустая
    """
    # Проверка яркости изображения
    grayscale_image = image.convert('L')  # Преобразование в градации серого
    np_image = np.array(grayscale_image)
    white_pixel_ratio = np.mean(np_image > 250)  # 250 — порог для белого цвета (почти белый пиксель)
    
    # Проверка количества текста на странице
    text = pytesseract.image_to_string(image, config=r'--oem 3 --psm 3 -l rus').strip()
    
    # Если на странице мало текста или она слишком белая, считаем страницу пустой
    if len(text) < text_threshold or white_pixel_ratio > pixel_threshold:
        return True
    
    return False

def convert_date(date_str):
    month_map = {
                        'января': '01', 'февраля': '02', 'марта': '03', 'апреля': '04', 'мая': '05', 'июня': '06',
                        'июля': '07', 'августа': '08', 'сентября': '09', 'октября': '10', 'ноября': '11', 'декабря': '12'
                    }
    parts = date_str.split()
    day = parts[0]
    month = month_map.get(parts[1].lower())  
    year = parts[2]
    return f"{day.zfill(2)}.{month}.{year}"

def process_pdf(pdf_path, type_juvenile, type_death,  type_jurisdiction, type_court_order, type_no_identifier):
    custom_config = r'--oem 3 --psm 3 -l rus'
    pages = convert_from_path(pdf_path)

    # Массивы для хранения данных
    documents = {'Сканы': [], 'Текст': [], 'Результат': []}
    type_text = {'Выписка ЕГРН': '', 'Определение': '', 'Заявление СП': '', 'Паспортный стол': '', 'Судебный приказ': ''}
    type_document = {'Выписка ЕГРН': [], 'Определение': [], 'Заявление СП': [], 'Паспортный стол': [], 'Судебный приказ': []}

    # Переменные для отслеживания состояния
    custom_type = None  # Текущий тип документа
    page_text = ''  # Для накопления текста текущего документа
    sum_page_text = ''  # Для суммирования текста текущего документа
    png_extract = []  # Для накопления изображений текущего документа

    for page in pages:
        try:
            if is_blank_page(page):
                # Добавляем текущие данные, если они есть, перед переходом к пустой странице
                if sum_page_text:
                    if custom_type == 'Выписка ЕГРН':
                        type_text['Выписка ЕГРН'] = sum_page_text
                        type_document['Выписка ЕГРН'] = png_extract
                    elif custom_type == 'Определение':
                        type_text['Определение'] = sum_page_text
                        type_document['Определение'] = png_extract
                    elif custom_type == 'Заявление СП':
                        type_text['Заявление СП'] = sum_page_text
                        type_document['Заявление СП'] = png_extract
                    elif custom_type == 'Паспортный стол':
                        type_text['Паспортный стол'] = sum_page_text
                        type_document['Паспортный стол'] = png_extract
                    elif custom_type == 'Судебный приказ':
                        type_text['Судебный приказ'] = sum_page_text
                        type_document['Судебный приказ'] = png_extract

                    documents['Сканы'].append(type_document)
                    documents['Текст'].append(type_text)
                
                    # Сброс данных для нового документа
                    type_text = {'Выписка ЕГРН': '', 'Определение': '', 'Заявление СП': '', 'Паспортный стол': '', 'Судебный приказ': ''}
                    type_document = {'Выписка ЕГРН': [], 'Определение': [], 'Заявление СП': [], 'Паспортный стол': [], 'Судебный приказ': []}
                    png_extract = []
                    sum_page_text = ''
                    custom_type = None

                continue

            # Преобразуем страницу в изображение
            png = page.convert('RGB')
            orientation = pytesseract.image_to_osd(png)
            data_img = {}
            pairs = [pair.split(": ") for pair in orientation.split("\n")]
            for pair in pairs:
                if len(pair) == 2:
                    key, value = pair
                    data_img[key.strip()] = value.strip()

            # Поворот изображения, если оно имеет неправильную ориентацию
            if data_img['Orientation in degrees'] != '0':
                png = png.rotate(-int(data_img['Rotate']), expand=True)

            # Извлекаем текст с помощью pytesseract
            page_text = pytesseract.image_to_string(png, config=custom_config).strip()
            # Определяем тип документа по ключевым словам
            if 'выписка из единого государственного реестра' in page_text.lower() or custom_type == 'Выписка ЕГРН':
                if custom_type != 'Выписка ЕГРН':
                    if sum_page_text:
                        type_text[custom_type] = sum_page_text
                        type_document[custom_type] = png_extract

                    sum_page_text = ''
                    png_extract = []
                    sum_page_text += ' ' + page_text
                    png_extract.append(png)
                    custom_type = 'Выписка ЕГРН'
                else:
                    sum_page_text += ' ' + page_text
                    png_extract.append(png)

            # Повторяем для остальных типов документов
            if 'ОПРЕДЕЛЕНИЕ' in page_text or custom_type == 'Определение':
                if custom_type != 'Определение':
                    if sum_page_text:
                        type_text[custom_type] = sum_page_text
                        type_document[custom_type] = png_extract

                    sum_page_text = ''
                    png_extract = []
                    sum_page_text += ' ' + page_text
                    png_extract.append(png)
                    custom_type = 'Определение'
                else:
                    sum_page_text += ' ' + page_text
                    png_extract.append(png)

            if 'Заявление о вынесении судебного приказа' in page_text or custom_type == 'Заявление СП':
                if custom_type != 'Заявление СП':
                    if sum_page_text:
                        type_text[custom_type] = sum_page_text
                        type_document[custom_type] = png_extract

                    sum_page_text = ''
                    png_extract = []
                    png_extract.append(png)
                    sum_page_text += ' ' + page_text
                    custom_type = 'Заявление СП'
                else:
                    sum_page_text += ' ' + page_text
                    png_extract.append(png)

            if 'адресная справка' in page_text.lower() or custom_type == 'Паспортный стол':
                if custom_type != 'Паспортный стол':
                    if sum_page_text:
                        type_text[custom_type] = sum_page_text
                        type_document[custom_type] = png_extract

                    sum_page_text = ''
                    png_extract = []
                    png_extract.append(png)
                    sum_page_text += ' ' + page_text
                    custom_type = 'Паспортный стол'
                else:
                    sum_page_text += ' ' + page_text
                    png_extract.append(png)

            if 'СУДЕБНЫЙ ПРИКАЗ' in page_text or custom_type == 'Судебный приказ':
                if custom_type != 'Судебный приказ':
                    if sum_page_text:
                        type_text[custom_type] = sum_page_text
                        type_document[custom_type] = png_extract


                    sum_page_text = ''
                    png_extract = []
                    png_extract.append(png)
                    sum_page_text += ' ' + page_text
                    custom_type = 'Судебный приказ'
                else:
                    sum_page_text += ' ' + page_text
                    png_extract.append(png)

        except pytesseract.TesseractError as e:
            print('Ошибка при распознавании страницы:', e)

    # В конце добавляем последний документ, если он есть
    if sum_page_text:
        if custom_type == 'Выписка ЕГРН':
            type_text['Выписка ЕГРН'] = sum_page_text
            type_document['Выписка ЕГРН'] = png_extract
        elif custom_type == 'Определение':
            type_text['Определение'] = sum_page_text
            type_document['Определение'] = png_extract
        elif custom_type == 'Заявление СП':
            type_text['Заявление СП'] = sum_page_text
            type_document['Заявление СП'] = png_extract
        elif custom_type == 'Паспортный стол':
            type_text['Паспортный стол'] = sum_page_text
            type_document['Паспортный стол'] = png_extract
        elif custom_type == 'Судебный приказ':
            type_text['Судебный приказ'] = sum_page_text
            type_document['Судебный приказ'] = png_extract

        documents['Сканы'].append(type_document)
        documents['Текст'].append(type_text)


    for i in range(len(documents['Текст'])):
        results = {'Определение': [], 'Заявление СП': [], 'Выписка ЕГРН': [], 'Паспортный стол': [], 'Судебный приказ': []}
        if len(documents['Текст'][i]['Выписка ЕГРН']) > 0:
            nlp = spacy.load("model-extracts")
            doc = nlp(documents['Текст'][i]['Выписка ЕГРН'])

            entities = []
            combined_entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_
                })
            current_person = None
            current_org = None
            current_goverment_agency = None
            cad_number = 'Не найден'
            location = 'Не найден'
            activate_person = False
            activate_goverment_agency = False
            activate_org = False
            for entity in entities:
                label = entity["label"]
                text_document = entity["text"].replace('|', '').replace('\n', ' ').strip().replace('\n', ' ')
                if label == 'CADASTRAL_NUMBER':
                    cad_number = text_document

                if label == 'LOC':
                    location = text_document
                
                if label == "ORG":
                    activate_person = False
                    activate_goverment_agency = False
                    activate_org = True
                    if current_org:  
                        combined_entities.append(current_org)
                    current_org = {
                        "Тип документа": "Выписки ЕГРН",
                        "Название": text_document,
                        "ИНН": 'Не найден',
                        "ОГРН": 'Не найден',
                        "Кадастровый номер": cad_number,
                        "Местоположение": location,
                        "Дата начала собственности": "Не найден",
                        "Долевая собственность": "Не найден"
                    }
                
                elif label == "INN" and current_org:
                        current_org["ИНН"] = text_document
                    
                elif label == "OGRN" and current_org:
                        current_org["ОГРН"] = text_document
                    
                elif label == "GOVERNMENT_AGENCY":
                    activate_person = False
                    activate_goverment_agency = True
                    activate_org = False
                    if current_goverment_agency:
                        combined_entities.append(current_goverment_agency)
                    current_goverment_agency = {
                        "Тип документа": "Выписки ЕГРН",
                        "Название": text_document,
                        "Кадастровый номер": cad_number,
                        "Местоположение": location,
                        "Дата начала собственности": "Не найден",
                        "Долевая собственность": "Не найден"
                    }

                elif label == "PER":
                    activate_person = True
                    activate_goverment_agency = False
                    activate_org = False
                    if current_person:
                        combined_entities.append(current_person)
                    current_person = {
                        "Тип документа": "Выписки ЕГРН",
                        "ФИО": text_document,
                        "Дата рождения": "Не найден",
                        "Паспорт": "Не найден",
                        "Свидетельство о рождении": "Не найден",
                        "Дата выдачи": "Не найден",
                        "Место выдачи": "Не найден",
                        "СНИЛС": "Не найден",
                        "Кадастровый номер": cad_number,
                        "Местоположение": location,
                        "Дата начала собственности": "Не найден",
                        "Долевая собственность": "Не найден"
                    }

                elif label == "BIRTHDAY" and current_person:
                        current_person["Дата рождения"] = text_document
                        date_pattern = r'\b\d{2}\.\d{2}\.\d{4}\b'
                        if isinstance(current_person["Дата рождения"], list):
                            current_person["Дата рождения"] = ' '.join(current_person["Дата рождения"])
                        current_person["Дата рождения"] = re.findall(date_pattern, current_person["Дата рождения"])

                    
                elif label == "PASSPORT":
                    text_document = re.sub(r"№", "", text_document)  # Убираем символ "№"
                    text_document = re.sub(r"\s+", "", text_document)  # Убираем все пробелы
                    text_document = re.sub(r"(\d{4})(\d{6})", r"\1 \2", text_document)  # Форматируем серию и номер паспорта
                    text_document = re.sub(r"[.,;!?]+$", "", text_document)
                    current_person["Паспорт"] = text_document
                
                elif label == "BIRTH_CERTIFICATE" and current_person:
                        current_person["Свидетельство о рождении"] = text_document
                    
                elif label == "PASSPORT_DATE" and current_person:
                        current_person["Дата выдачи"] = text_document
                        date_pattern = r'\b\d{2}\.\d{2}\.\d{4}\b'
                        if isinstance(current_person["Дата выдачи"], list):
                            current_person["Дата выдачи"] = ' '.join(current_person["Дата выдачи"])
                        current_person["Дата выдачи"] = re.findall(date_pattern, current_person["Дата выдачи"])

                elif label == "PLACE_PASSPORT" and current_person:
                        current_person["Место выдачи"] = text_document

                elif label == "SNILS" and current_person:
                        current_person["СНИЛС"] = re.sub(r'\D', '', text_document)

                elif label == "START_DATE_OF_OWNERSHIP":
                    if activate_person:
                        current_person["Дата начала собственности"] = text_document
                        date_pattern = r'\b\d{2}\.\d{2}\.\d{4}\b'
                        if isinstance(current_person["Дата начала собственности"], list):
                            current_person["Дата начала собственности"] = ' '.join(current_person["Дата начала собственности"])
                        current_person["Дата начала собственности"] = re.findall(date_pattern, current_person["Дата начала собственности"])
                    elif activate_goverment_agency:
                        current_goverment_agency["Дата начала собственности"] = text_document
                        date_pattern = r'\b\d{2}\.\d{2}\.\d{4}\b'
                        if isinstance(current_person["Дата начала собственности"], list):
                            current_person["Дата начала собственности"] = ' '.join(current_person["Дата начала собственности"])
                        current_person["Дата начала собственности"] = re.findall(date_pattern, current_person["Дата начала собственности"])
                    elif activate_org:
                        current_org["Дата начала собственности"] = text_document
                        date_pattern = r'\b\d{2}\.\d{2}\.\d{4}\b'
                        if isinstance(current_person["Дата начала собственности"], list):
                            current_person["Дата начала собственности"] = ' '.join(current_person["Дата начала собственности"])
                        current_person["Дата начала собственности"] = re.findall(date_pattern, current_person["Дата начала собственности"])

                elif label == "SHARED_OWNERSHIP":
                    if activate_person:
                        current_person["Долевая собственность"] = text_document
                    elif activate_goverment_agency:
                        current_goverment_agency["Долевая собственность"] = text_document
                    elif activate_org:
                        current_org["Долевая собственность"] = text_document
                
                elif label == "NOT_EGRN":
                    combined_entities.append({
                        "Текст": text_document,
                        "Кадастровый номер": cad_number,
                        "Местоположение": location,
                    })
            
            if current_person:
                combined_entities.append(current_person)

            if current_org:
                combined_entities.append(current_org)

            if current_goverment_agency:
                combined_entities.append(current_goverment_agency)

            results['Выписка ЕГРН'] = combined_entities
        
        if len(documents['Текст'][i]['Определение']) > 0:
            nlp = spacy.load("model-definitions")
            doc = nlp(documents['Текст'][i]['Определение'])
            entities = []
            combined_entities = []
            current_people = []
            for ent in doc.ents:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_
                    })
            
            if type_juvenile:
                current_person = {
                    "Тип документа": "Определение-несовершеннолетние",
                    "ФИО": "",
                    "Дата рождения": "",
                    "Место рождения": "",
                    "Паспорт": "",
                    "Адрес регистрации": ""
                }

                for entity in entities:
                    label = entity["label"]
                    text_document = entity["text"].replace('|', '').replace('\n', ' ').strip()
                    if label == "PER":
                        # Добавляем предыдущего человека в список, если ФИО не пустое
                        if current_person["ФИО"].strip():
                            current_people.append(current_person)
                        # Создаём новый объект для следующего человека
                        current_person = {
                            "Тип документа": "Определение-несовершеннолетние",
                            "ФИО": text_document,
                            "Дата рождения": "",
                            "Место рождения": "",
                            "Паспорт": "",
                            "Адрес регистрации": ""
                        }

                    elif label == "BIRTHDAY":
                        current_person["Дата рождения"] = text_document
                        date_pattern = r'\b\d{2}\.\d{2}\.\d{4}\b'
                        if isinstance(current_person["Дата рождения"], list):
                            current_person["Дата рождения"] = ' '.join(current_person["Дата рождения"])
                        current_person["Дата рождения"] = re.findall(date_pattern, current_person["Дата рождения"])

                    elif label == "PASSPORT":
                        text_document = re.sub(r"№", "", text_document)  # Убираем символ "№"
                        text_document = re.sub(r"\s+", "", text_document)  # Убираем все пробелы
                        text_document = re.sub(r"(\d{4})(\d{6})", r"\1 \2", text_document)  # Форматируем серию и номер паспорта
                        text_document = re.sub(r"[.,;!?]+$", "", text_document)
                        current_person["Паспорт"] = text_document

                    elif label == "PLACE_OF_RESIDENCE":
                        current_person["Место рождения"] = text_document

                    elif label == "REGISTRATION_ADDRESS":
                        current_person["Адрес регистрации"] = text_document

                # Добавляем последнего человека после выхода из цикла
                if current_person["ФИО"].strip():
                    current_people.append(current_person)

                # Если ни один человек не найден, добавить пустого
                if not current_people:
                    current_people.append(current_person)

                results['Определение'] = current_people


            elif type_death:
                current_person = {
                    "Тип документа": "Определение-смерть",
                    "ФИО": "",
                    "Дата рождения": "",
                    "Место рождения": "",
                    "Паспорт": "",
                    "Адрес регистрации": "",
                    "Дата смерти": ""
                }

                for entity in entities:
                    label = entity["label"]
                    text_document = entity["text"].replace('|', '').replace('\n', ' ').strip()
                    if label == "PER":
                        if current_person["ФИО"].strip():
                            current_people.append(current_person)
                        current_person = {
                            "Тип документа": "Определение-смерть",
                            "ФИО": text_document,
                            "Дата рождения": "",
                            "Место рождения": "",
                            "Паспорт": "",
                            "Адрес регистрации": "",
                            "Дата смерти": ""
                        }

                    elif label == "BIRTHDAY":
                        current_person["Дата рождения"] = text_document
                        date_pattern = r'\b\d{2}\.\d{2}\.\d{4}\b'
                        if isinstance(current_person["Дата рождения"], list):
                            current_person["Дата рождения"] = ' '.join(current_person["Дата рождения"])
                        current_person["Дата рождения"] = re.findall(date_pattern, current_person["Дата рождения"])

                    elif label == "PASSPORT":
                        text_document = re.sub(r"№", "", text_document)  # Убираем символ "№"
                        text_document = re.sub(r"\s+", "", text_document)  # Убираем все пробелы
                        text_document = re.sub(r"(\d{4})(\d{6})", r"\1 \2", text_document)  # Форматируем серию и номер паспорта
                        text_document = re.sub(r"[.,;!?]+$", "", text_document)
                        current_person["Паспорт"] = text_document

                    elif label == "PLACE_OF_RESIDENCE":
                        current_person["Место рождения"] = text_document

                    elif label == "REGISTRATION_ADDRESS":
                        current_person["Адрес регистрации"] = text_document

                    elif label == "DATE_DEATH":
                        current_person["Дата смерти"] = text_document
                        date_pattern = r'\b\d{2}\.\d{2}\.\d{4}\b'
                        if isinstance(current_person["Дата смерти"], list):
                            current_person["Дата смерти"] = ' '.join(current_person["Дата смерти"])
                        current_person["Дата смерти"] = re.findall(date_pattern, current_person["Дата смерти"])

                if current_person["ФИО"].strip():
                    current_people.append(current_person)

                if not current_people:
                    current_people.append(current_person)

                results['Определение'] = current_people


            elif type_jurisdiction:     
                current_person = {
                            "Тип документа": "Определение-Неверная подсудность",
                            "Старая подсудность": "",
                            "Новая подсудность": ""
                        }

                for entity in entities:
                    label = entity["label"]
                    text_document = entity["text"].replace('|', '').replace('\n', ' ').strip()
                    if label == "OLD_JURISDICTION":
                        current_person["Старая подсудность"] += " " + text_document
                        
                    elif label == "NEW_JURISDICTION":
                            current_person["Новая подсудность"] += " " + text_document
                    
                for indx, val in current_person.items():
                    if val == "":
                        current_person[indx] = "Не найден"
            
                results['Определение'] = current_person

            elif type_no_identifier:
                current_person = {
                    "Тип документа": "Определение-отсутствие идентификатора",
                    "ФИО": "",
                    "Дата рождения": "",
                    "Место рождения": "",
                    "Адрес регистрации": "",
                    "Паспорт": "",
                    "Дата выдачи паспорта": "",
                    "СНИЛС": "",
                    "ИНН": "",
                    "Долевая собственность": "",
                    "Дата начала собственности": "",

                }

                for entity in entities:
                    label = entity["label"]
                    text_document = entity["text"].replace('|', '').replace('\n', ' ').strip()
                    if label == "PER":
                        # Добавляем предыдущего человека в список, если ФИО не пустое
                        if current_person["ФИО"].strip():
                            current_people.append(current_person)
                        # Создаём новый объект для следующего человека
                        current_person = {
                            "Тип документа": "Определение-отсутствие идентификатора",
                            "ФИО": text_document,
                            "Дата рождения": "",
                            "Место рождения": "",
                            "Адрес регистрации": "",
                            "Паспорт": "",
                            "Дата выдачи паспорта": "",
                            "СНИЛС": "",
                            "ИНН": "",
                            "Долевая собственность": "",
                            "Дата начала собственности": "",
                        }

                    elif label == "BIRTHDAY":
                        current_person["Дата рождения"] = text_document
                        date_pattern = r'\b\d{2}\.\d{2}\.\d{4}\b'
                        if isinstance(current_person["Дата рождения"], list):
                            current_person["Дата рождения"] = ' '.join(current_person["Дата рождения"])
                        current_person["Дата рождения"] = re.findall(date_pattern, current_person["Дата рождения"])

                    elif label == "PLACE_OF_RESIDENCE":
                        current_person["Место рождения"] = text_document

                    elif label == "REGISTRATION_ADDRESS":
                        current_person["Адрес регистрации"] = text_document

                    elif label == "PASSPORT":
                        text_document = re.sub(r"№", "", text_document)  # Убираем символ "№"
                        text_document = re.sub(r"\s+", "", text_document)  # Убираем все пробелы
                        text_document = re.sub(r"(\d{4})(\d{6})", r"\1 \2", text_document)  # Форматируем серию и номер паспорта
                        text_document = re.sub(r"[.,;!?]+$", "", text_document)
                        current_person["Паспорт"] = text_document
                    
                    elif label == "DATE_PASSPORT":
                        current_person["Дата выдачи паспорта"] = text_document
                        date_pattern = r'\b\d{2}\.\d{2}\.\d{4}\b'
                        if isinstance(current_person["Дата выдачи паспорта"], list):
                            current_person["Дата выдачи паспорта"] = ' '.join(current_person["Дата выдачи паспорта"])
                        current_person["Дата выдачи паспорта"] = re.findall(date_pattern, current_person["Дата выдачи паспорта"])

                    elif label == "SNILS":
                        current_person["СНИЛС"] = re.sub(r'\D', '', text_document)
                    
                    elif label == "INN":
                        current_person["INN"] = text_document

                    elif label == "SHARED_OWNERSHIP":
                        current_person["Долевая собственность"] = text_document

                    elif label == "OWNERSHIP_START_DATE":
                        current_person["Дата начала собственности"] = text_document
                        date_pattern = r'\b\d{2}\.\d{2}\.\d{4}\b'
                        if isinstance(current_person["Дата начала собственности"], list):
                            current_person["Дата начала собственности"] = ' '.join(current_person["Дата начала собственности"])
                        current_person["Дата начала собственности"] = re.findall(date_pattern, current_person["Дата начала собственности"])

                # Добавляем последнего человека после выхода из цикла
                if current_person["ФИО"].strip():
                    current_people.append(current_person)

                # Если ни один человек не найден, добавить пустого
                if not current_people:
                    current_people.append(current_person)

                results['Определение'] = current_people

        if len(documents['Текст'][i]['Заявление СП']) > 0:
            x, y, w, h = 1100, 1, 7000, 200
            open_cv_image = np.array(documents['Сканы'][i]['Заявление СП'][0])
            open_cv_image = open_cv_image[:, :, ::-1].copy()

            cropped_image = open_cv_image[y:y+h, x:x+w]

            try:
                text = pytesseract.image_to_string(cropped_image, config=custom_config)
                text = re.findall(r'\b\d{6}\b', text)
                if len(text) == 0:
                    x, y, w, h = 1, 830, 1500, 150
                    cropped_image = open_cv_image[y:y+h, x:x+w]
                    try:
                        text = pytesseract.image_to_string(cropped_image, config=custom_config)
                        print('Текст:', text)
                        text = re.findall(r'\b\d{6}\b', text)
                        if len(text) == 0:
                            text = "Не найден"
                    except:
                        text = "Не найден"
            except:
                text = "Не найден"
            
            current_person = {
                "Тип документа": "Заявление СП",
                "Номер": text
            }
            results['Заявление СП'] = current_person
        
        if len(documents['Текст'][i]['Паспортный стол']) > 0:
            nlp = spacy.load("model-passport_desk")
            doc = nlp(documents['Текст'][i]['Паспортный стол'])

            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_
                })
            current_person = {
                        "Тип документа": "Паспортный стол",
                        "ФИО": "",
                        "Дата рождения": "",
                        "Место рождения": "",
                        "Адрес регистрации": "",
                        "Паспорт": "",
                        "Свидетельство о рождении": "",
                        "Дата выдачи": "",
                        "Место выдачи": "",
                        "Дата смерти": ""
                    }

            for entity in entities:
                label = entity["label"]
                text_document = entity["text"].replace('|', '').replace('\n', ' ').strip()
                if label == "PER":
                    current_person["ФИО"] += " " + text_document
                    
                elif label == "BIRTHDAY":
                        current_person["Дата рождения"] = text_document
                        if isinstance(current_person["Дата рождения"], list):
                                current_person["Дата рождения"] = " ".join(current_person["Дата рождения"])
                        date_pattern = r'\b\d{2}\.\d{2}\.\d{4}\b'
                        current_person["Дата рождения"] = re.findall(date_pattern, current_person["Дата рождения"])
                    
                elif label == "PASSPORT": 
                    current_person["Паспорт"] += " " + text_document
                    text_document = re.sub(r"№", "", text_document)  # Убираем символ "№"
                    text_document = re.sub(r"\s+", "", text_document)  # Убираем все пробелы
                    text_document = re.sub(r"(\d{4})(\d{6})", r"\1 \2", text_document)  # Форматируем серию и номер паспорта
                    text_document = re.sub(r"[.,;!?]+$", "", text_document)
                    current_person["Паспорт"] = text_document
                
                elif label == "BIRTH_CERTIFICATE":
                        current_person["Свидетельство о рождении"] += " " + text_document
                    
                elif label == "PASSPORT_DATE":
                        current_person["Дата выдачи"] += " " + text_document
                        if isinstance(current_person["Дата выдачи"], list):
                                current_person["Дата выдачи"] = " ".join(current_person["Дата выдачи"])
                        date_pattern = r'\b\d{2}\.\d{2}\.\d{4}\b'
                        current_person["Дата выдачи"] = re.findall(date_pattern, current_person["Дата выдачи"])

                elif label == "PLACE_PASSPORT":
                        current_person["Место выдачи"] += " " + text_document

                elif label == "REGISTRATION_ADDRESS":
                        current_person["Адрес регистрации"] += " " + text_document

                elif label == "PLACE_OF_RESIDENCE":
                        current_person["Место выдачи"] += " " + text_document
                
                elif label == "DEATH_DATE":
                        current_person["Дата смерти"] += " " + text_document
                        if isinstance(current_person["Дата смерти"], list):
                                current_person["Дата смерти"] = " ".join(current_person["Дата смерти"])
                        date_pattern = r'\b\d{2}\.\d{2}\.\d{4}\b'
                        current_person["Дата смерти"] = re.findall(date_pattern, current_person["Дата смерти"])
                
            for indx, val in current_person.items():
                if val == "":
                    current_person[indx] = "Не найден"
                    
            results['Паспортный стол'] = current_person

        if len(documents['Текст'][i]['Судебный приказ']) > 0 and type_court_order:
            nlp = spacy.load("model-judicial_distinction")
            doc = nlp(documents['Текст'][i]['Судебный приказ'])

            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_
                })
            current_person = {
                        "Тип документа": "Судебный приказ",
                        "Номер суда":"",
                        "Номер приказа":"",
                        "Дата судебного приказа":"",
                        "Адрес ОН": "",
                        "Период задолженности от":"",
                        "Период задолженности до":"",
                        "Долг по начислению":"",
                        "Долг по пени":"",
                        "Долг по Госпошлине":"",
                        "Общий долг":"",
                        "ФИО": "",
                        "Дата рождения": "",
                        "Место рождения": "",
                        "Паспорт": "",
                        "Адрес регистрации": "",
                        "Дата выдачи Паспорта": "",
                        "Место выдачи паспорта": "", 
                        "ИНН": "",
                        "СНИЛС": ""
                    }

            for entity in entities:
                label = entity["label"]
                text_document = entity["text"].replace('|', '').replace('\n', ' ').strip()
                if label == "PER":
                    current_person["ФИО"] = text_document
                    
                elif label == "BIRTHDAY":
                        current_person["Дата рождения"] = text_document
                        if isinstance(current_person["Дата рождения"], list):
                                current_person["Дата рождения"] = " ".join(current_person["Дата рождения"])
                        date_pattern = r'\b\d{2}\.\d{2}\.\d{4}\b'
                        current_person["Дата рождения"] = re.findall(date_pattern, current_person["Дата рождения"])
                
                elif label == "PASSPORT":
                    text_document = re.sub(r"№", "", text_document)  # Убираем символ "№"
                    text_document = re.sub(r"\s+", "", text_document)  # Убираем все пробелы
                    text_document = re.sub(r"(\d{4})(\d{6})", r"\1 \2", text_document)  # Форматируем серию и номер паспорта
                    text_document = re.sub(r"[.,;!?]+$", "", text_document)
                    current_person["Паспорт"] = text_document
                    
                elif label == "SNILS":
                    snils_pattern = r'\b\d{3}-\d{3}-\d{3} \d{2}\b'
                    if not text_document or not re.fullmatch(snils_pattern, text_document):
                        matches = re.findall(snils_pattern, documents['Текст'][i]['Судебный приказ'])
                        if matches:
                            current_person["СНИЛС"] = matches[0]
                        else:
                            current_person["СНИЛС"] = "" 
                    else:
                        current_person["СНИЛС"] = text_document
                elif label == "DATE_PASSPORT":
                        current_person["Дата выдачи Паспорта"] = text_document
                        if isinstance(current_person["Дата выдачи Паспорта"], list):
                                current_person["Дата выдачи Паспорта"] = " ".join(current_person["Дата выдачи Паспорта"])
                        date_pattern = r'\b\d{2}\.\d{2}\.\d{4}\b'
                        current_person["Дата выдачи Паспорта"] = re.findall(date_pattern, current_person["Дата выдачи Паспорта"])

                elif label == "COURT_ORDER_DATE":
                    if len(text_document.split()) > 3:
                        current_person["Дата судебного приказа"] = convert_date(text_document)

                elif label == "PLACE_PASSPORT":
                        current_person["Место выдачи паспорта"] = text_document

                elif label == "REGISTRATION_ADDRESS":
                        current_person["Адрес регистрации"] = text_document

                elif label == "PLACE_OF_RESIDENCE":
                        current_person["Место рождения"] = text_document
                
                elif label == "INN":
                    current_person["ИНН"] = text_document
                
                elif label == "ORDER_NUMBER":
                    current_person["Номер суда"] = text_document

                elif label == "COURT_NUMBER":
                    text_document = re.sub(r"№", "", text_document).strip()
                    if not text_document:
                        date_pattern = r'\d-\d{4}/\d{4}'
                        matches = re.findall(date_pattern, documents['Текст'][i]['Судебный приказ'])
                        if matches:
                            current_person["Номер приказа"] = matches[0]
                        else:
                            current_person["Номер приказа"] = "" 
                    else:
                        current_person["Номер приказа"] = text_document


                elif label == "OH_ADDRESS":
                    current_person["Адрес ОН"] = text_document
                
                elif label == "PERIOD_OF_INDEBTEDNESS":
                    if "-" in text_document or "по" in text_document:
                        text_document = re.split(r"-|по", text_document)
                        
                        if len(text_document) >= 2:
                            current_person["Период задолженности от"] = text_document[0].strip()
                            current_person["Период задолженности до"] = text_document[1].strip()
                        else:
                            current_person["Период задолженности от"] = text_document[0].strip()
                            current_person["Период задолженности до"] = "Не найден"
                    else:
                        current_person["Период задолженности от"] = text_document.strip()
                        current_person["Период задолженности до"] = "Не найден"
                
                elif label == "ACCRUAL_DEBT":
                    if text_document:
                        # Проверяем, является ли текст уже в формате 'XX,YY' или 'XX.YY'
                        if re.match(r'^\d+[,.]\d{2}$', text_document):
                            text_document = text_document.replace('.', ',')  # Заменяем точку на запятую, если нужно
                        else:
                            numbers = re.findall(r'\d+', text_document)  # Ищем все числа в тексте
                            
                            # Проверяем, что найдены и рубли, и копейки
                            if len(numbers) >= 2:
                                rubles, kopecks = map(int, numbers[:2])  # Берём первые два числа для рублей и копеек
                                text_document = f"{rubles},{kopecks:02d}"

                    current_person["Долг по начислению"] = text_document

                elif label == "PENALTY_DEBT":
                    if text_document:
                        # Проверяем, является ли текст уже в формате 'XX,YY' или 'XX.YY'
                        if re.match(r'^\d+[,.]\d{2}$', text_document):
                            text_document = text_document.replace('.', ',')  # Заменяем точку на запятую, если нужно
                        else:
                            numbers = re.findall(r'\d+', text_document)  # Ищем все числа в тексте
                            
                            # Проверяем, что найдены и рубли, и копейки
                            if len(numbers) >= 2:
                                rubles, kopecks = map(int, numbers[:2])  # Берём первые два числа для рублей и копеек
                                text_document = f"{rubles},{kopecks:02d}"

                    current_person["Долг по пени"] = text_document
                
                elif label == "STATE_DUTY_DEBT":
                    if text_document:
                        # Проверяем, является ли текст уже в формате 'XX,YY' или 'XX.YY'
                        if re.match(r'^\d+[,.]\d{2}$', text_document):
                            text_document = text_document.replace('.', ',')  # Заменяем точку на запятую, если нужно
                        else:
                            numbers = re.findall(r'\d+', text_document)  # Ищем все числа в тексте
                            
                            # Проверяем, что найдены и рубли, и копейки
                            if len(numbers) >= 2:
                                rubles, kopecks = map(int, numbers[:2])  # Берём первые два числа для рублей и копеек
                                text_document = f"{rubles},{kopecks:02d}"

                    current_person["Долг по Госпошлине"] = text_document

                elif label == "TOTAL_DEBT":
                    if text_document:
                        # Проверяем, является ли текст уже в формате 'XX,YY' или 'XX.YY'
                        if re.match(r'^\d+[,.]\d{2}$', text_document):
                            text_document = text_document.replace('.', ',')  # Заменяем точку на запятую, если нужно
                        else:
                            numbers = re.findall(r'\d+', text_document)  # Ищем все числа в тексте
                            
                            # Проверяем, что найдены и рубли, и копейки
                            if len(numbers) >= 2:
                                rubles, kopecks = map(int, numbers[:2])  # Берём первые два числа для рублей и копеек
                                text_document = f"{rubles},{kopecks:02d}"

                    current_person["Общий долг"] = text_document
                

            for indx, val in current_person.items():
                if val == "" or (isinstance(val, list) and not val):
                    current_person[indx] = "Не найден"
                    
            results['Судебный приказ'] = current_person
        documents['Результат'].append(results)

    return documents


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    # Получаем последние 6 файлов в папке загрузок
    files = sorted(
        [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.zip')],
        key=lambda x: os.path.getmtime(os.path.join(app.config['UPLOAD_FOLDER'], x)),
        reverse=True
    )[:6]
    
    # Генерируем HTML для отображения списка файлов
    file_links = ''.join([f'<li><a href="/download/{file}">{file}</a></li>' for file in files])

    return f'''
    <!doctype html>
    <html lang="ru">
    <head>
        <meta charset="utf-8">
        <title>Загрузка файла</title>
        <style>
            .container {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
            }}
            .form-container {{
                width: 50%;
            }}
            .files-container {{
                width: 40%;
                margin-left: 20px;
                padding: 10px;
                border: 1px solid #ddd;
                background-color: #f9f9f9;
            }}
            h2 {{
                font-size: 1.2em;
            }}
            ul {{
                list-style-type: none;
                padding-left: 0;
            }}
            li {{
                margin-bottom: 8px;
            }}
            li a {{
                text-decoration: none;
                color: #007bff;
            }}
            li a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        <h1>Загрузить PDF файл</h1>
        <div class="container">
            <div class="form-container">
                <form method="post" enctype="multipart/form-data" action="/upload">
                  <label for="doc_type">Выберите тип документа:</label><br>
                  <input type="radio" name="doc_type" value="type_1"> Определение несовершеннолетний<br>
                  <input type="radio" name="doc_type" value="type_2"> Определение смерть собственника<br>
                  <input type="radio" name="doc_type" value="type_3"> Определение неверная подсудность<br>
                  <input type="radio" name="doc_type" value="type_4"> Определение отсутствие идентификатора<br><br>
                  <input type="radio" name="doc_type" value="type_5"> Судебный приказ<br><br>
                  <input type="file" name="file">
                  <input type="submit" value="Загрузить">
                </form>
            </div>
            <div class="files-container">
                <h2>Последние загруженные файлы</h2>
                <ul>
                    {file_links}
                </ul>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/download/<filename>')
def download_file(filename):
    # Отправляем файл для скачивания
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
    
    if file and allowed_file(file.filename):
        # Создание папки uploads, если она не существует
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        selected_type = request.form.get('doc_type')
        if not selected_type:
            return jsonify({"error": "Не выбран тип документа"}), 400

        app.logger.debug(f"Selected document type: {selected_type}")
        type_juvenile = selected_type == 'type_1'
        type_death = selected_type == 'type_2'
        type_jurisdiction = selected_type == 'type_3'
        type_no_identifier = selected_type == 'type_4'
        type_court_order = selected_type == 'type_5'

        documents = process_pdf(file_path, type_juvenile, type_death,  type_jurisdiction, type_court_order, type_no_identifier)
        app.logger.debug(f"Processed documents: {documents}")
        tmpdirname = tempfile.mkdtemp()
        app.logger.debug(f"Temporary directory created at: {tmpdirname}")
        
        try:
            # Проходим по каждому человеку
            for i in range(len(documents['Результат'])):
                # Создаем папку для каждого человека
                person_folder = os.path.join(tmpdirname, f'распознавание_{i}')
                os.makedirs(person_folder, exist_ok=True)
                app.logger.debug(f"Creating folder for person {i}: {person_folder}")

                # Проходим по документам каждого человека
                keys_to_remove = [key for key, value in documents['Результат'][i].items() if value == []]

                for key in keys_to_remove:
                    del documents['Результат'][i][key]
                    
                json_path = os.path.join(person_folder, 'данные.json')
                with open(json_path, 'w', encoding='utf-8') as json_file:
                    json.dump(documents['Результат'][i], json_file, ensure_ascii=False, indent=4)
                    app.logger.debug(f"Saved JSON data for person {i} to {json_path}")

                for doc_type, images in documents['Сканы'][i].items():
                    if images:
                        # Сохраняем PDF напрямую в папку человека
                        images_rgb = [img.convert("RGB") for img in images]
                        pdf_output_path = os.path.join(person_folder, f'{doc_type}.pdf')
                        app.logger.debug(f"Saving PDF for {doc_type} to {pdf_output_path}")

                        if images_rgb:
                            images_rgb[0].save(pdf_output_path, save_all=True, append_images=images_rgb[1:])

            # Генерация уникального имени для ZIP архива
            timestamp = time.strftime("%d:%m:%Y:_%H:%M")
            final_zip_path = os.path.join(app.config['UPLOAD_FOLDER'], f'Документы_{timestamp}.zip')
            app.logger.debug(f"Creating zip archive at: {final_zip_path}")
            shutil.make_archive(final_zip_path[:-4], 'zip', tmpdirname)
            
            app.logger.debug("Zip archive created successfully.")
            
            # Удаляем изначально загруженный PDF файл
            if os.path.exists(file_path):
                os.remove(file_path)
                app.logger.debug(f"Deleted uploaded PDF file: {file_path}")

            return send_file(final_zip_path, as_attachment=True, download_name=f'Документы_{timestamp}.zip')

        finally:
            app.logger.debug(f"Cleaning up temporary directory: {tmpdirname}")
            shutil.rmtree(tmpdirname)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)