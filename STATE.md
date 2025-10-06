# STATE.md - Детальная структура проекта

_Последнее обновление: 2025-09-29_

## 📋 Оглавление
1. [Обзор проекта](#обзор-проекта)
2. [Архитектура системы](#архитектура-системы)
3. [Структура файлов](#структура-файлов)
4. [Модульная система](#модульная-система)
5. [Поток выполнения](#поток-выполнения)
6. [Конфигурация и настройка](#конфигурация-и-настройка)
7. [API и интеграции](#api-и-интеграции)
8. [Технологический стек](#технологический-стек)
9. [Инструкции по запуску](#инструкции-по-запуску)
10. [Известные ограничения](#известные-ограничения)

---

## 🎯 Обзор проекта

**Название**: Система прогнозирования акций с анализом настроений
**Тип**: Streamlit веб-приложение
**Цель**: Комплексный анализ и прогнозирование движения акций на основе исторических данных, новостей из Telegram и LLM-анализа настроений

### Ключевые возможности:
- 📊 Анализ исторических данных акций с Yahoo Finance и MOEX
- 📱 Парсинг Telegram каналов для сбора финансовых новостей
- 🤖 LLM-анализ настроений новостей через OpenAI API
- 📈 ML-прогнозирование с использованием RandomForest
- 📉 Корреляционный анализ новостей и движения акций
- 🎨 Интерактивная визуализация через Plotly

---

## 🏗️ Архитектура системы

### Модульная архитектура
```
┌─────────────────┐
│    app.py       │ ← Точка входа Streamlit
└────────┬────────┘
         │
    ┌────▼────┐
    │ modules │
    └────┬────┘
         │
         ├── stock_analyzer.py    → Загрузка и анализ биржевых данных
         ├── telegram_parser.py   → Парсинг Telegram каналов
         ├── llm_analyzer.py      → LLM анализ настроений
         ├── forecaster.py        → ML прогнозирование
         └── visualizer.py        → Визуализация данных
```

### Поток данных
```
[Yahoo/MOEX API] → [StockAnalyzer] → [Исторические данные]
                                           ↓
[Telegram API] → [TelegramParser] → [Новости] → [LLMAnalyzer] → [Настроения]
                                           ↓
                                    [Forecaster] → [Прогноз]
                                           ↓
                                    [Visualizer] → [UI]
```

---

## 📁 Структура файлов

```
ИИ ЭКСПЕРТНЫЕ СИСТЕМЫ/
│
├── app.py                    # Главный файл приложения Streamlit
├── STATE.md                  # Этот файл - документация проекта
├── replit.md                 # Конфигурация для Replit
├── pyproject.toml            # Конфигурация проекта и зависимости
├── uv.lock                   # Lock-файл зависимостей
│
├── config/                   # Конфигурации источников
│   ├── channels.yml          # Список Telegram-каналов
│   └── keywords.yml          # Ключевые слова и фразы
│
├── modules/                  # Основные модули системы
│   ├── __init__.py          # Инициализация пакета
│   ├── stock_analyzer.py    # Анализ биржевых данных
│   ├── schemas.py           # Pydantic-схемы для конфигов и сообщений
│   ├── storage.py           # Работа с Parquet-хранилищем
│   ├── telegram_parser.py   # Асинхронный конвейер Telegram
│   ├── llm_analyzer.py      # LLM анализатор
│   ├── forecaster.py        # Прогнозирование
│   └── visualizer.py        # Визуализация
│
├── tests/                   # Модульные тесты
│   └── test_telegram_parser.py
│
├── .venv/                   # Виртуальное окружение Python
└── __pycache__/             # Скомпилированные Python файлы
```

---

## 🔧 Модульная система

### 1. StockAnalyzer (`modules/stock_analyzer.py`)
**Назначение**: Загрузка и анализ исторических данных акций

#### Класс: `StockAnalyzer`

**Методы**:
- `__init__()`: Инициализация с пустым словарем данных
- `fetch_stock_data(symbols, period="2y")`: Загрузка данных для списка символов
  - Сначала пытается Yahoo Finance
  - При неудаче использует MOEX ISS API для российских акций
- `_fetch_from_yahoo(symbol, period)`: Загрузка через yfinance
- `_fetch_from_moex(symbol, period)`: Резервная загрузка с MOEX
  - Поддерживает пагинацию (до 50 страниц)
  - Конвертирует данные в формат совместимый с Yahoo
- `_normalize_moex_symbol(symbol)`: Нормализация символов для MOEX
- `calculate_price_changes(symbol)`: Расчет изменений цены (1д, 7д, 30д)
- `find_top_movements(symbol, top_n=20)`: Поиск топ движений цены
- `calculate_statistics(symbol)`: Расчет статистики (среднее, медиана, std)
- `identify_patterns_before_movements(symbol, days_before=5)`: Анализ паттернов
- `get_recent_data(symbol, days=30)`: Получение последних данных

**Особенности**:
- Автоматическое переключение между Yahoo Finance и MOEX
- Кеширование загруженных данных в `self.stock_data`
- Обработка ошибок с информативными сообщениями

---

### 2. Telegram Parser (`modules/telegram_parser.py`)
**Назначение**: Надёжный асинхронный конвейер парсинга публичных Telegram-каналов с инкрементальными обновлениями.

#### Ключевые функции
- `initialize_client(api_id, api_hash, phone)` — настраивает Telethon user-сессию в `data/telegram/stock_forecasting_session.session`.
- `fetch_channel_messages(client, channel, since_id, since_days)` — догружает новые сообщения, нормализует дату в UTC и извлекает ссылки/тикеры/хэштеги.
- `parse_channels(channel_cfgs, stock_symbols, days_back, ...)` — параллельно обрабатывает каналы, выдерживает `FloodWait`, обновляет watermark `data/telegram/state.json` и сохраняет результаты.
- `filter_messages(df, stock_symbols, keywords)` — комбинированная фильтрация по тикерам и ключевым словам.
- `run_telegram_parser(...)` — синхронная оболочка для Streamlit/CLI, возвращающая готовый `pandas.DataFrame`.

#### Особенности
- Работа только с публичными каналами; авторизация через Telethon под user-аккаунт.
- Инкрементальность и дедупликация сообщений по `msg_hash` (sha256).
- Автоматическое сохранение сырых данных в `data/telegram/raw/{YYYY-MM-DD}.parquet` и агрегата `data/telegram/latest.parquet`.
- Мягкое восстановление после `FloodWaitError`, `UsernameInvalidError`, `ChannelPrivateError`.
- Настраиваемая конкуррентность (`max_concurrency`) и фильтры по allow-листу тикеров и ключевым словам.

#### Сопутствующие модули
- `modules/schemas.py` — Pydantic-модели `ChannelConfig`, `TelegramMessage` для валидации конфигов и данных.
- `modules/storage.py` — обёртки `save_parquet`/`load_parquet` с гарантией создания директорий.

---

### 3. LLMAnalyzer (`modules/llm_analyzer.py`)
**Назначение**: Анализ настроений через LLM

#### Класс: `LLMAnalyzer`

**Методы**:
- `__init__(api_key, base_url, model_name)`:
  - Поддержка кастомных OpenAI-совместимых endpoints
  - По умолчанию использует модель "gpt-5"
- `analyze_news_sentiment(text, stock_symbol=None)`:
  - Анализ одного текста
  - Возвращает JSON с sentiment, confidence, stock_impact
- `analyze_batch_messages(messages_df, stock_symbols)`:
  - Пакетный анализ множества сообщений
  - Progress bar для отслеживания
- `correlate_news_with_stock_movements(news_df, stock_data, symbol)`:
  - Корреляция настроений с движением цен
  - Расчет лаговых корреляций (до 7 дней)
- `generate_forecast_analysis(historical_patterns, news_sentiment, symbol)`:
  - Генерация текстового прогноза на основе данных
  - Структурированный JSON ответ

**Формат ответа LLM**:
```json
{
  "sentiment": "positive|negative|neutral",
  "confidence": 0.0-1.0,
  "stock_impact": "bullish|bearish|neutral",
  "impact_strength": 1-10,
  "key_factors": ["factor1", "factor2"],
  "summary": "краткое объяснение"
}
```

---

### 4. StockForecaster (`modules/forecaster.py`)
**Назначение**: ML прогнозирование с RandomForest

#### Класс: `StockForecaster`

**Методы**:
- `__init__()`: Инициализация модели и scaler
- `prepare_features(stock_data, sentiment_data=None)`:
  - Инженеринг признаков (SMA, RSI, volatility)
  - Интеграция sentiment данных если доступны
- `calculate_rsi(prices, window=14)`: Расчет RSI индикатора
- `aggregate_daily_sentiment(sentiment_data)`:
  - Агрегация sentiment по дням
  - Расчет средних значений и количества
- `merge_sentiment_data(stock_data, sentiment_data)`:
  - Объединение биржевых и sentiment данных
  - Добавление лаговых признаков
- `train_model(prepared_data)`:
  - Обучение RandomForestRegressor
  - 80/20 train/test split
  - Расчет метрик (MSE, R²)
- `generate_forecast(recent_data, days_ahead=90)`:
  - Итеративное прогнозирование
  - Добавление confidence bands
- `get_feature_importance()`: Важность признаков модели

**Признаки модели**:
- Технические индикаторы: SMA_5, SMA_20, RSI
- Изменения цен: 1д, 5д, 20д
- Объемы: Volume_SMA_5, Volume_Ratio
- Волатильность: 5д, 20д
- High-Low spread
- Sentiment метрики (если доступны)

---

### 5. Visualizer (`modules/visualizer.py`)
**Назначение**: Интерактивная визуализация данных

#### Класс: `Visualizer`

**Методы**:
- `__init__()`: Инициализация цветовой палитры
- `plot_stock_price_history(stock_data, symbol)`:
  - График цены и объема (2 подграфика)
- `plot_price_distribution(stock_data, symbol)`:
  - Гистограмма и box plot распределения
- `plot_top_movements(top_rises, top_falls, symbol)`:
  - Визуализация топ движений цены
- `plot_sentiment_analysis(sentiment_df)`:
  - Pie chart настроений
  - Timeline настроений
  - Heatmap силы воздействия
- `plot_correlation_analysis(correlation_df)`:
  - Корреляционная матрица
  - График лаговых корреляций
- `plot_forecast(historical_data, forecast_data, symbol)`:
  - Исторические данные + прогноз
  - Confidence bands
- `display_statistics_table(stats_dict, symbol)`:
  - Таблица статистики в Streamlit

**Библиотека**: Plotly для всех графиков
**Особенности**:
- Интерактивные графики
- Адаптивная компоновка
- Кастомная цветовая схема

---

## 🔄 Поток выполнения

### 1. Инициализация (`app.py`)
```python
1. Конфигурация Streamlit страницы
2. Инициализация session_state с дефолтными значениями
3. Создание sidebar с настройками
4. Инициализация сервисов (StockAnalyzer, LLMAnalyzer, etc.)
```

### 2. Вкладки интерфейса

#### Вкладка "Анализ акций":
```python
1. fetch_stock_data() для выбранных символов
2. calculate_price_changes() и calculate_statistics()
3. find_top_movements() для каждой акции
4. Визуализация через Visualizer методы
```

#### Вкладка "Настроения новостей":
```python
1. run_telegram_parser() с API credentials
2. filter_stock_related_messages()
3. analyze_batch_messages() через LLM
4. Отображение результатов анализа
```

#### Вкладка "Корреляция":
```python
1. Загрузка sentiment и stock данных
2. correlate_news_with_stock_movements()
3. Визуализация корреляционных матриц
4. Расчет лаговых эффектов
```

#### Вкладка "Прогноз":
```python
1. prepare_features() с историческими данными
2. train_model() на подготовленных признаках
3. generate_forecast() на 90 дней вперед
4. Визуализация прогноза с confidence bands
```

#### Вкладка "Отчет":
```python
1. Агрегация данных из всех вкладок
2. Генерация сводного отчета
3. Экспорт в JSON формат
4. Отображение ключевых метрик
```

---

## ⚙️ Конфигурация и настройка

### Переменные окружения
```bash
# OpenAI/LLM настройки
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL_NAME=gpt-5

# Telegram настройки
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_PHONE=your_phone
```

### Конфигурация через UI (sidebar)
- **Символы акций**: Multiselect + custom input
- **Telegram каналы**: Текстовое поле (по одному на строку)
- **Период анализа**: 1y, 2y, 5y
- **Дни новостей**: Slider 1-30 дней
- **API настройки**: Динамическая конфигурация через session_state

### Зависимости (`pyproject.toml`)
```toml
[project]
name = "repl-nix-workspace"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=2.3.3",
    "openai>=1.109.1",
    "pandas>=2.3.2",
    "pydantic>=2.0.0",
    "pyyaml>=6.0.2",
    "pytest>=8.3.4",
    "pytest-asyncio>=0.25.2",
    "plotly>=6.3.0",
    "scikit-learn>=1.7.2",
    "scipy>=1.16.2",
    "streamlit>=1.50.0",
    "telethon>=1.41.2",
    "yfinance>=0.2.66",
]
```

---

## 🌐 API и интеграции

### 1. Yahoo Finance API
- **Библиотека**: yfinance
- **Endpoint**: Автоматический через библиотеку
- **Данные**: OHLCV, дивиденды, сплиты
- **Ограничения**: Некоторые российские акции недоступны

### 2. MOEX ISS API
- **Endpoint**: `https://iss.moex.com/iss/engines/stock/markets/shares/boards/{board}/securities/{symbol}/candles.json`
- **Формат**: JSON
- **Особенности**:
  - Публичный API без авторизации
  - Поддержка пагинации
  - Интервалы: 1, 10, 60, 24 (часы)

### 3. Telegram API
- **Библиотека**: Telethon
- **Авторизация**: API ID + Hash + Phone
- **Сессия**: Сохраняется в `stock_forecasting_session`
- **Rate limits**: Стандартные Telegram ограничения

### 4. OpenAI API
- **Библиотека**: openai
- **Модель**: gpt-5 (по умолчанию)
- **Особенности**:
  - Поддержка кастомных endpoints
  - JSON mode для структурированных ответов
  - Температура: 0.3 для консистентности

---

## 💻 Технологический стек

### Backend
- **Python 3.11+**: Основной язык
- **Pandas**: Обработка данных
- **NumPy**: Численные вычисления
- **Scikit-learn**: Machine Learning
- **SciPy**: Статистический анализ

### Frontend
- **Streamlit**: Web интерфейс
- **Plotly**: Интерактивная визуализация

### API клиенты
- **yfinance**: Yahoo Finance данные
- **Telethon**: Telegram API
- **OpenAI**: LLM интеграция

### Инфраструктура
- **UV**: Управление зависимостями
- **Virtualenv**: Изоляция окружения

---

## 📨 Telegram-конвейер: настройка и запуск

1. **Получите Telegram API ID/Hash**: на [my.telegram.org/apps](https://my.telegram.org/apps) создайте приложение и сохраните `api_id`, `api_hash` и телефон аккаунта.
2. **Обновите конфиги**:
   - `config/channels.yml` — список публичных `username` и названий каналов; поле `enabled` управляет парсингом.
   - `config/keywords.yml` (опционально) — ключевые слова/регулярки для текстовой фильтрации.
3. **Первый запуск** сохранит user-сессию в `data/telegram/stock_forecasting_session.session`. При первом входе Telethon запросит код подтверждения в консоли.
4. **CLI**: 
   ```bash
   python -m modules.telegram_parser \
     --channels config/channels.yml \
     --days-back 7 \
     --symbols SBER,GAZP,AAPL \
     --api-id $TELEGRAM_API_ID \
     --api-hash $TELEGRAM_API_HASH \
     --phone $TELEGRAM_PHONE
   ```
5. **Инкрементальность и хранилище**:
   - `data/telegram/state.json` — watermark `message_id` по каналам.
   - `data/telegram/raw/{YYYY-MM-DD}.parquet` + `data/telegram/latest.parquet` — сырые и агрегированные сообщения с колонками `date_utc`, `channel_username`, `tickers`, `links`, `msg_hash`.
6. **Интеграция в Streamlit**: вкладка «Настроения новостей» использует `run_telegram_parser(...)` и отображает прогресс, сводку и экспорт в LLM-конвейер.

---

## 🚀 Инструкции по запуску

### 1. Установка зависимостей
```bash
# Создание виртуального окружения
python -m venv .venv

# Активация окружения
source .venv/bin/activate  # Linux/Mac
# или
.venv\Scripts\activate  # Windows

# Установка зависимостей
pip install -r requirements.txt
# или через uv
uv pip install -r pyproject.toml
```

### 2. Настройка окружения
```bash
# Создайте .env файл
cat > .env << EOF
OPENAI_API_KEY=your_key
TELEGRAM_API_ID=your_id
TELEGRAM_API_HASH=your_hash
TELEGRAM_PHONE=your_phone
EOF

# Экспорт переменных
export $(cat .env | xargs)
```

### 3. Запуск приложения
```bash
# Стандартный запуск
streamlit run app.py

# С кастомным портом
streamlit run app.py --server.port 8501

# Headless режим (для сервера)
streamlit run app.py --server.headless true
```

### 4. Первый запуск
1. Откройте http://localhost:8501
2. Введите API ключи в sidebar
3. Выберите акции для анализа
4. При первом использовании Telegram потребуется авторизация

---

## ⚠️ Известные ограничения

### Технические ограничения
1. **RandomForest прогноз**: Упрощенная рекурсивная модель, может дрейфовать на длинных горизонтах
2. **MOEX API**: Нет защиты от rate limiting, возможны сбои при частых запросах
3. **Telegram парсер**: Требует интерактивную авторизацию, не поддерживает 2FA автоматически
4. **Состояние сессии**: Все данные сбрасываются при перезапуске Streamlit

### Функциональные ограничения
1. **Нет персистентности**: Данные не сохраняются между сессиями
2. **LLM соответствие**: Предполагается JSON формат ответа, нет retry логики
3. **Обработка ошибок**: Базовая обработка, может требовать улучшения
4. **Масштабируемость**: Однопоточное выполнение, не оптимизировано для больших объемов

### Ограничения данных
1. **Yahoo Finance**: Российские акции могут быть недоступны из-за санкций
2. **MOEX**: Только публичные данные, задержка 15 минут
3. **Telegram**: Ограничения на количество сообщений и каналов
4. **LLM**: Зависит от качества модели и промптов

---

## 📊 Метрики производительности

### Типичное время выполнения
- Загрузка данных акций: 2-5 сек на символ
- Парсинг Telegram: 5-10 сек на канал
- LLM анализ: 1-2 сек на сообщение
- Обучение модели: 5-15 сек
- Генерация прогноза: 2-5 сек

### Требования к ресурсам
- RAM: Минимум 2GB, рекомендовано 4GB
- CPU: 2+ ядра для комфортной работы
- Диск: 500MB для зависимостей
- Сеть: Стабильное интернет соединение

---

## 🔍 Отладка и логирование

### Включение debug режима
```python
# В app.py
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('logger.level', 'debug')
```

### Проверка синтаксиса
```bash
python -m compileall app.py modules
```

### Типичные проблемы и решения
1. **"API key not found"**: Проверьте переменные окружения
2. **"No data for symbol"**: Попробуйте международные символы
3. **"Telegram auth failed"**: Удалите session файл и переавторизуйтесь
4. **"Model training failed"**: Проверьте наличие достаточных данных

---

## 📝 TODO и планы развития

### Краткосрочные улучшения
- [ ] Добавить unit тесты
- [ ] Реализовать retry логику для API
- [ ] Добавить кеширование данных
- [ ] Улучшить обработку ошибок

### Долгосрочные планы
- [ ] Миграция на async архитектуру
- [ ] Добавление других ML моделей (LSTM, XGBoost)
- [ ] Реализация backtesting
- [ ] Добавление real-time данных
- [ ] Интеграция с брокерскими API
- [ ] Мобильное приложение

---

## 📄 Лицензия и авторы

**Тип проекта**: Учебный проект
**Статус**: В активной разработке
**Последнее обновление**: 2025-09-29

---

*Этот документ автоматически обновляется при изменениях в структуре проекта*
