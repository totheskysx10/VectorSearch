# Используем официальный базовый образ Python
FROM python:3.12-slim

# Устанавливаем необходимые системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    python3-dev \
    wget \
    cmake \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем файл с зависимостями
COPY requirements.txt .

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код приложения в контейнер
COPY . .

# Загрузка и кэширование модели
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large'); AutoModel.from_pretrained('intfloat/multilingual-e5-large')"

# Определяем команду для запуска приложения
CMD ["python", "main.py"]
