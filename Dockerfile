# Используйте официальный образ Python 3.10

FROM nvidia/cuda:11.2.2-base

# Установите необходимые системные пакеты и Tesseract
RUN apt-get update && apt-get install -y \
    ffmpeg \
    poppler-utils \
    tesseract-ocr \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Загрузка языковых данных для Tesseract
RUN wget -P /usr/share/tesseract-ocr/4.00/tessdata/ \
    https://github.com/tesseract-ocr/tessdata/raw/main/rus.traineddata && \
    wget -P /usr/share/tesseract-ocr/4.00/tessdata/ \
    https://github.com/tesseract-ocr/tessdata/raw/main/osd.traineddata

# Установка рабочего каталога
WORKDIR /app

# Установка переменной окружения для пути к языковым файлам Tesseract
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/

# Копирование файлов зависимостей и установка Python-зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование моделей и скриптов
COPY model-definitions/ model-definitions/
COPY model-extracts/ model-extracts/
COPY model-judicial_distinction/ model-judicial_distinction/
COPY model-passport_desk/ model-passport_desk/
COPY recognition_of_statements.py .

RUN mkdir -p /app/uploads

#Запуск приложения
CMD ["flask", "--app", "recognition_of_statements", "run", "--host=0.0.0.0", "--port=5000"]
