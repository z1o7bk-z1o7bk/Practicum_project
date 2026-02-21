▎Text Autocomplete

Проект для автодополнения текста (next-token prediction) на основе LSTM с использованием BERT-токенизатора.

▎Структура проекта

text-autocomplete/
├── configs/                    # Конфигурационные файлы
├── data/                       # Данные и токенизатор
│   ├── bert_tokenizer/        # BERT-токенизатор
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── cleaned_dataset_TOY.pkl
│   ├── raw_dataset_TOY.txt
│   ├── train_texts_TOY.pkl
│   ├── val_texts_TOY.pkl
│   └── test_texts_TOY.pkl
├── src/                        # Исходный код
│   ├── data_utils.py          # Утилиты для обработки данных
│   ├── next_token_dataset.py  # Dataset для next-token prediction
│   ├── lstm_model.py          # Архитектура LSTM модели
│   ├── lstm_train.py          # Скрипт обучения модели
│   ├── eval_transformer_pipeline.py  # Оценка через pipeline
│   └── plt_lstm.py            # Визуализация результатов
├── models/                     # Сохраненные модели
│   └── best_lstm_model.pth    # Лучшая модель LSTM
├── best_lstm_model.pth        # Основная сохраненная модель
├── Logs.txt                   # Логи обучения
├── LSTM after 10 epoch.png    # График обучения
├── requirements.txt           # Зависимости Python
└── text-autocomplete.ipynb    # Jupyter ноутбук с анализом

▎Описание

Проект реализует модель автодополнения текста, которая предсказывает следующий токен на основе предыдущего контекста. Основные особенности:

• Модель: LSTM (Long Short-Term Memory) нейронная сеть
• Токенизатор: BERT-токенизатор для обработки текста
• Архитектура: Эмбеддинг + LSTM + Линейный слой для классификации токенов
• Обучение: Оптимизация с помощью Adam, функция потерь CrossEntropyLoss

▎Установка и запуск

▎1. Установка зависимостей

Bash

pip install -r requirements.txt

▎2. Обучение модели

Bash

python src/lstm_train.py

▎3. Использование в Jupyter Notebook

Откройте text-autocomplete.ipynb для интерактивного анализа и тестирования модели.

▎Основные компоненты

▎src/lstm_train.py

Основной скрипт обучения, который:
• Загружает и подготавливает данные
• Инициализирует модель LSTM
• Обучает модель с валидацией
• Сохраняет лучшую модель

▎src/lstm_model.py

Определяет архитектуру модели LSTM:
• LSTMModel: Основной класс модели с эмбеддингами и LSTM слоями
• generate_text: Функция для генерации текста

▎src/next_token_dataset.py

Кастомный Dataset для next-token prediction:
• Преобразует текст в последовательности токенов
• Создает пары (контекст, целевой токен)

▎src/data_utils.py

Утилиты для обработки данных:
• Очистка текста
• Разделение на train/val/test
• Загрузка токенизатора

▎src/eval_transformer_pipeline.py

Интеграция с Hugging Face Transformers для оценки через pipeline.

▎src/plt_lstm.py

Визуализация результатов обучения.

▎Использование модели

▎Генерация текста

Python

from src.lstm_model import LSTMModel
import torch

# Загрузка модели
model = LSTMModel(vocab_size=30522, embedding_dim=128, hidden_dim=256, num_layers=2)
model.load_state_dict(torch.load('best_lstm_model.pth'))
model.eval()

# Генерация текста
generated_text = model.generate_text(
    starting_text="Привет, как",
    max_length=50,
    temperature=0.8
)
print(generated_text)

▎Данные

Проект использует TOY-датасет для демонстрации. Данные включают:
• Очищенный текст в формате pickle
• Разделение на обучающую, валидационную и тестовую выборки
• Предобученный BERT-токенизатор

▎Результаты

• Модель сохраняется в best_lstm_model.pth
• Логи обучения записываются в Logs.txt
• График обучения сохраняется как LSTM after 10 epoch.png

▎Требования

Основные зависимости:
• Python 3.7+
• PyTorch
• Transformers (Hugging Face)
• NumPy, Pandas
• Matplotlib для визуализации

▎Лицензия

Проект предназначен для образовательных целей.
