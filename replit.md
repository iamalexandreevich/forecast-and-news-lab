# Overview

This is a stock forecasting system built with Streamlit that combines technical analysis with sentiment analysis from Telegram channels. The application analyzes Russian stock market securities (MOEX symbols like SBER, YNDX, GAZP) using historical price data and news sentiment from Russian financial Telegram channels. It uses machine learning models to predict stock movements based on technical indicators and sentiment scores derived from financial news.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit Web Application**: Single-page application with sidebar configuration
- **Interactive Dashboard**: Real-time visualization of stock data, predictions, and sentiment analysis
- **Multi-tab Interface**: Separate sections for stock analysis, telegram parsing, sentiment analysis, forecasting, and visualization

## Backend Architecture
- **Modular Design**: Separated into distinct modules for different functionalities
  - `stock_analyzer.py`: Yahoo Finance data fetching and technical analysis
  - `telegram_parser.py`: Telegram channel message extraction using Telethon
  - `llm_analyzer.py`: OpenAI GPT-5 powered sentiment analysis
  - `forecaster.py`: Machine learning predictions using Random Forest
  - `visualizer.py`: Plotly-based interactive charts and graphs

## Data Processing Pipeline
- **Technical Analysis**: RSI, SMA, volatility, price change calculations
- **Sentiment Analysis**: LLM-based news sentiment scoring with confidence levels
- **Feature Engineering**: Combines technical indicators with sentiment scores
- **Machine Learning**: Random Forest regression for price prediction

## Authentication and Authorization
- **Telegram Authentication**: Session-based authentication using phone number verification
- **API Key Management**: Environment variable-based configuration for OpenAI and Telegram APIs

## Design Patterns
- **Factory Pattern**: Analyzer classes instantiated as needed
- **Pipeline Pattern**: Sequential data processing from raw data to predictions
- **Observer Pattern**: Streamlit reactive updates based on user inputs

# External Dependencies

## APIs and Services
- **Yahoo Finance (yfinance)**: Historical stock data retrieval for Russian securities
- **OpenAI GPT-5**: Financial news sentiment analysis and market impact assessment
- **Telegram API (Telethon)**: Real-time message parsing from Russian financial channels

## Data Sources
- **Russian Stock Market**: MOEX-listed securities (Sberbank, Yandex, Gazprom, etc.)
- **Telegram Channels**: Russian financial news channels (@russianinvestor, @finmarket, @rbc_news)

## Machine Learning Libraries
- **scikit-learn**: Random Forest regression, data preprocessing, model evaluation
- **pandas/numpy**: Data manipulation and numerical computations
- **scipy**: Statistical analysis for technical indicators

## Visualization
- **Plotly**: Interactive financial charts and sentiment visualization
- **Streamlit**: Web interface framework with built-in charting capabilities

## Environment Variables Required
- `OPENAI_API_KEY`: OpenAI API authentication
- `OPENAI_BASE_URL`: Optional OpenAI endpoint override
- `TELEGRAM_API_ID`: Telegram application ID
- `TELEGRAM_API_HASH`: Telegram application hash
- `TELEGRAM_PHONE`: Phone number for Telegram authentication