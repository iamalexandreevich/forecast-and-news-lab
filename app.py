import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import custom modules
from modules.stock_analyzer import StockAnalyzer
from modules.telegram_parser import run_telegram_parser
from modules.llm_analyzer import LLMAnalyzer
from modules.forecaster import StockForecaster
from modules.visualizer import Visualizer

# Page configuration
st.set_page_config(
    page_title="Система прогнозирования акций",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data(show_spinner=False)
def load_channel_config(path: str = "config/channels.yml") -> list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    channels = data.get("channels", data)
    return channels if isinstance(channels, list) else []


@st.cache_data(show_spinner=False)
def load_keywords_config(path: str = "config/keywords.yml") -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    keywords = data.get("keywords", data)
    return [str(keyword) for keyword in keywords] if isinstance(keywords, list) else []


def main():
    st.title("📈 Система прогнозирования акций")
    st.markdown("*Анализ исторических рыночных паттернов и настроений новостей Telegram*")

    # Initialize API/model configuration defaults once per session
    default_config = {
        'openai_api_key': os.getenv("LLM_API") or os.getenv("OPENAI_API_KEY", ""),
        'openai_base_url': os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        'openai_model_name': os.getenv("LLM_MODEL_NAME") or os.getenv("OPENAI_MODEL_NAME", "gpt-5"),
        'telegram_api_id': os.getenv("TELEGRAM_API_ID", ""),
        'telegram_api_hash': os.getenv("TELEGRAM_API_HASH", "").strip("'\""),
        'telegram_phone': os.getenv("TELEGRAM_PHONE", "")
    }
    for key, value in default_config.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Sidebar configuration
    with st.sidebar:
        st.header("Настройки")
        
        # Stock symbols input
        st.subheader("Символы акций")
        st.info("📝 Примечание: Российские акции могут быть недоступны через Yahoo Finance. Рекомендуем использовать международные символы.")
        # Use international alternatives that work with Yahoo Finance
        default_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
        stock_symbols = st.multiselect(
            "Выберите символы акций для анализа:",
            options=["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "META", "AMZN", "NFLX", "BABA", "DIS", "JPM", "V"],
            default=default_symbols
        )
        
        # Custom symbol input
        custom_symbol = st.text_input("Добавить свой символ:", placeholder="например, YNDX (для Яндекс ADR)")
        if custom_symbol and custom_symbol.upper() not in stock_symbols:
            stock_symbols.append(custom_symbol.upper())
        
        # Add expandable section with Russian stock guidance
        with st.expander("🔍 Как найти российские акции?"):
            st.markdown("""
            **Российские компании через ADR:**
            - **YNDX** - Яндекс (Yandex ADR)
            - **QIWI** - Киви (QIWI ADR)
            
            **Международные альтернативы:**
            - **BABA** - Alibaba (Китай)
            - **PDD** - PDD Holdings
            - **NIO** - NIO Inc.
            
            *Примечание: Прямые российские акции (.ME) могут быть недоступны через Yahoo Finance из-за геополитических ограничений.*
            """)
        
        # Telegram channels input
        st.subheader("Telegram каналы")
        config_channels_sidebar = load_channel_config()
        if config_channels_sidebar:
            st.caption("📁 Основной список каналов хранится в `config/channels.yml`")
        manual_channels_raw = st.text_area(
            "Дополнительные каналы (по одному на строку):",
            value="",
            placeholder="investfuture\ntbank_news",
            help="Каждый канал можно указать с @ или без"
        ).split('\n')
        manual_channels = [ch.strip().lstrip('@') for ch in manual_channels_raw if ch.strip()]
        
        # Time period settings
        st.subheader("Период анализа")
        analysis_period = st.selectbox(
            "Период исторических данных:",
            options=["1y", "2y", "5y"],
            index=1
        )
        
        news_days = st.slider(
            "Дни анализа новостей:",
            min_value=1,
            max_value=30,
            value=7
        )
        
        # API configuration inputs
        st.subheader("Настройки LLM")
        openai_model_name = st.text_input(
            "Название модели",
            value=st.session_state.get('openai_model_name', "gpt-5"),
            help="Например, gpt-5, gpt-4o, llama3 и т.д."
        )
        openai_base_url = st.text_input(
            "Базовый URL API",
            value=st.session_state.get('openai_base_url', "https://api.openai.com/v1"),
            help="Используйте кастомный эндпоинт, если применяете совместимую LLM-платформу"
        )
        openai_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.get('openai_api_key', ""),
            type="password",
            help="Ключ доступа к модели (не сохраняется на сервере)"
        )

        st.subheader("Telegram API")
        telegram_api_id = st.text_input(
            "API ID",
            value=st.session_state.get('telegram_api_id', ""),
            help="Числовой идентификатор приложения Telegram"
        )
        telegram_api_hash = st.text_input(
            "API Hash",
            value=st.session_state.get('telegram_api_hash', ""),
            help="Секретный hash приложения Telegram"
        )
        telegram_phone = st.text_input(
            "Телефон (необязательно)",
            value=st.session_state.get('telegram_phone', ""),
            help="Телефон, к которому привязан аккаунт Telegram"
        )

        if st.button("💾 Сохранить настройки API"):
            st.session_state.openai_model_name = openai_model_name.strip() or "gpt-5"
            st.session_state.openai_base_url = openai_base_url.strip() or "https://api.openai.com/v1"
            st.session_state.openai_api_key = openai_key.strip()
            st.session_state.telegram_api_id = telegram_api_id.strip()
            st.session_state.telegram_api_hash = telegram_api_hash.strip()
            st.session_state.telegram_phone = telegram_phone.strip()
            st.success("Настройки API обновлены для текущей сессии")

        # API status indicators
        if st.session_state.openai_api_key:
            st.success("✅ Ключ OpenAI API загружен")
            st.info(f"🔗 Базовый URL: {st.session_state.openai_base_url}")
            st.caption(f"🧠 Модель: {st.session_state.openai_model_name}")
        else:
            st.error("❌ Ключ OpenAI API отсутствует")

        if st.session_state.telegram_api_id and st.session_state.telegram_api_hash:
            st.success("✅ Telegram API подключен")
        else:
            st.warning("⚠️ Укажите Telegram API ID и API Hash")

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Анализ акций", 
        "📱 Настроения новостей", 
        "🔗 Корреляция", 
        "🔮 Прогноз", 
        "📋 Отчет"
    ])
    
    # Initialize components
    if 'stock_analyzer' not in st.session_state:
        st.session_state.stock_analyzer = StockAnalyzer()

    llm_config_signature = (
        st.session_state.get('openai_api_key'),
        st.session_state.get('openai_base_url'),
        st.session_state.get('openai_model_name')
    )
    if (
        'llm_analyzer' not in st.session_state or
        st.session_state.get('llm_config_signature') != llm_config_signature
    ):
        st.session_state.llm_analyzer = LLMAnalyzer(
            api_key=st.session_state.get('openai_api_key'),
            base_url=st.session_state.get('openai_base_url'),
            model_name=st.session_state.get('openai_model_name')
        )
        st.session_state.llm_config_signature = llm_config_signature

    if 'forecaster' not in st.session_state:
        st.session_state.forecaster = StockForecaster()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = Visualizer()
    
    # Tab 1: Stock Analysis
    with tab1:
        st.header("📊 Исторический анализ акций")

        if not stock_symbols:
            st.warning("Пожалуйста, выберите хотя бы один символ акции из боковой панели.")
        else:
            # Fetch stock data button
            if st.button("🔄 Загрузить данные о акциях", type="primary"):
                with st.spinner("Загрузка исторических данных о акциях..."):
                    success = st.session_state.stock_analyzer.fetch_stock_data(stock_symbols, analysis_period)
                    if success:
                        st.success(f"Успешно загружены данные по {len(stock_symbols)} акциям")
                        st.session_state.stock_data_loaded = True
                    else:
                        st.error("Ошибка загрузки данных о акциях")
        
        # Display stock analysis if data is loaded
        if hasattr(st.session_state, 'stock_data_loaded') and st.session_state.stock_data_loaded:
            
            # Stock selection for detailed analysis
            selected_stock = st.selectbox("Выберите акцию для детального анализа:", stock_symbols)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Price history chart
                if selected_stock in st.session_state.stock_analyzer.stock_data:
                    stock_data = st.session_state.stock_analyzer.stock_data[selected_stock]
                    fig_price = st.session_state.visualizer.plot_stock_price_history(stock_data, selected_stock)
                    st.plotly_chart(fig_price, use_container_width=True)
                    
                    # Price distribution analysis
                    fig_dist = st.session_state.visualizer.plot_price_distribution(stock_data, selected_stock)
                    st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                # Statistics table
                stats = st.session_state.stock_analyzer.calculate_statistics(selected_stock)
                if stats:
                    stats_df = st.session_state.visualizer.display_statistics_table(stats, selected_stock)
                    st.subheader(f"Статистика {selected_stock}")
                    st.dataframe(stats_df, hide_index=True)
            
            # Top movements analysis
            st.subheader("Топ 20 ценовых движений")
            top_rises, top_falls = st.session_state.stock_analyzer.find_top_movements(selected_stock)
            
            if top_rises is not None and top_falls is not None:
                fig_movements = st.session_state.visualizer.plot_top_movements(top_rises, top_falls, selected_stock)
                st.plotly_chart(fig_movements, use_container_width=True)
                
                # Display top movements tables
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📈 Топ 20 ростов")
                    rises_display = top_rises[['Daily_Change_Pct', 'Volume', 'Close']].round(4)
                    st.dataframe(rises_display)
                
                with col2:
                    st.subheader("📉 Топ 20 падений")
                    falls_display = top_falls[['Daily_Change_Pct', 'Volume', 'Close']].round(4)
                    st.dataframe(falls_display)
    
    # Tab 2: News Sentiment Analysis
    with tab2:
        st.header("📱 Анализ настроений новостей Telegram")

        config_channels_tab = load_channel_config()
        enabled_channels = [cfg for cfg in config_channels_tab if cfg.get("enabled", True)]
        channel_options = [cfg.get("username", "").lstrip('@') for cfg in enabled_channels if cfg.get("username")]
        channel_labels = {
            cfg.get("username", "").lstrip('@'): f"{cfg.get('title') or cfg.get('username')} (@{cfg.get('username', '').lstrip('@')})"
            for cfg in enabled_channels
            if cfg.get("username")
        }

        selected_config_channels = st.multiselect(
            "Каналы из конфигурации:",
            options=channel_options,
            default=channel_options[: min(5, len(channel_options))],
            format_func=lambda username: channel_labels.get(username, f"@{username}"),
            help="Выберите публичные Telegram-каналы из `config/channels.yml`"
        )

        if manual_channels:
            st.caption("➕ Добавлены дополнительные каналы из боковой панели: " + ", ".join(f"@{c}" for c in manual_channels))

        combined_usernames = set(selected_config_channels)
        combined_usernames.update(manual_channels)

        selected_channel_payload = []
        for cfg in enabled_channels:
            username = cfg.get("username", "").lstrip('@')
            if username and username in combined_usernames:
                payload = cfg.copy()
                payload["username"] = username
                selected_channel_payload.append(payload)

        existing_usernames = {item["username"] for item in selected_channel_payload}
        for manual in manual_channels:
            if manual not in existing_usernames:
                selected_channel_payload.append({"username": manual, "title": manual, "enabled": True})

        keywords_catalog = load_keywords_config()
        selected_keywords = st.multiselect(
            "Ключевые слова фильтрации:",
            options=keywords_catalog,
            default=keywords_catalog,
            help="Текстовый фильтр по сообщениям (регистронезависимый)"
        )
        custom_keywords_input = st.text_input(
            "Добавить ключевые слова через запятую:",
            value="",
            placeholder="дивиденды, buyback, guidance"
        )
        if custom_keywords_input:
            extra_keywords = [kw.strip() for kw in custom_keywords_input.replace('\n', ',').split(',') if kw.strip()]
            selected_keywords = sorted({*selected_keywords, *extra_keywords})

        st.caption(f"📆 Инкрементальный сбор за последние {news_days} дн.")

        api_id = st.session_state.get('telegram_api_id')
        api_hash = st.session_state.get('telegram_api_hash')
        phone = st.session_state.get('telegram_phone')

        # Кнопка для сброса сессии Telegram (если нужна повторная авторизация)
        col1, col2 = st.columns([3, 1])
        with col1:
            fetch_news_btn = st.button("🔍 Собрать новости Telegram", type="primary")
        with col2:
            if st.button("🔄 Сбросить сессию"):
                # Очищаем все флаги авторизации
                keys_to_delete = [
                    'telegram_entered_code',
                    'telegram_phone_code_hash',
                    'telegram_auth_in_progress'
                ]
                for key in keys_to_delete:
                    if key in st.session_state:
                        del st.session_state[key]

                # Удаляем файл сессии
                session_file = Path("data/telegram/stock_forecasting_session.session")
                if session_file.exists():
                    session_file.unlink()
                st.success("Сессия сброшена. Нажмите 'Собрать новости' для новой авторизации.")
                st.rerun()

        # Проверяем, находимся ли мы в процессе авторизации
        if 'telegram_auth_in_progress' not in st.session_state:
            st.session_state.telegram_auth_in_progress = False

        if fetch_news_btn:
            st.session_state.telegram_auth_in_progress = True

        if st.session_state.telegram_auth_in_progress:
            if not selected_channel_payload:
                st.warning("Выберите канал из списка или добавьте его вручную в боковой панели.")
                st.session_state.telegram_auth_in_progress = False
            elif not api_id or not api_hash:
                st.error("Укажите Telegram API ID и API Hash в настройках боковой панели.")
                st.session_state.telegram_auth_in_progress = False
            else:
                st.info("💡 Если увидите QR — отсканируйте его в Telegram: Settings → Devices → Link Desktop Device")
                channels_text = ", ".join(f"@{name}" for name in sorted(combined_usernames)) or "—"
                progress_bar = st.progress(0, text="Подготовка клиента…")

                # Панель для отображения логов
                log_container = st.expander("📋 Логи процесса авторизации", expanded=True)
                with log_container:
                    st.write("🔍 **Session state флаги:**")
                    st.write(f"- telegram_phone_code_hash: {'установлен ✅' if st.session_state.get('telegram_phone_code_hash') else 'не установлен ❌'}")
                    st.write(f"- telegram_entered_code: {'установлен ✅' if st.session_state.get('telegram_entered_code') else 'не установлен ❌'}")
                    st.write(f"- telegram_auth_in_progress: {st.session_state.get('telegram_auth_in_progress', False)}")

                with st.status("Сбор сообщений из Telegram", expanded=True) as status:
                    status.write(f"Каналы: {channels_text}")
                    progress_bar.progress(20, text="Инициализация клиента…")
                    try:
                        status.write("🔄 Запуск парсера...")
                        result_df = run_telegram_parser(
                            channel_list=selected_channel_payload,
                            stock_symbols=stock_symbols,
                            days_back=news_days,
                            api_id=api_id,
                            api_hash=api_hash,
                            phone=phone,
                            keywords=selected_keywords or None,
                            ui=st,
                        )
                        progress_bar.progress(85, text="Обработка результатов…")
                    except RuntimeError as exc:
                        # RuntimeError обычно означает "ожидание ввода кода"
                        if "Waiting for code input" in str(exc):
                            progress_bar.progress(50, text="⏳ Ожидание кода...")
                            status.update(label="Ожидание ввода кода", state="running")
                            status.write("⏳ Пожалуйста, введите код из Telegram выше")
                            with log_container:
                                st.warning("⏳ Ожидание ввода кода от пользователя")
                            # НЕ сбрасываем telegram_auth_in_progress, чтобы продолжить после ввода кода
                        else:
                            st.session_state.telegram_messages = pd.DataFrame()
                            st.session_state.telegram_auth_in_progress = False
                            progress_bar.progress(100, text="Ошибка")
                            status.update(label="Ошибка при авторизации", state="error")
                            status.write(str(exc))
                            st.error(f"❌ Ошибка: {exc}")
                            with log_container:
                                st.error(f"RuntimeError: {exc}")
                    except Exception as exc:
                        st.session_state.telegram_messages = pd.DataFrame()
                        st.session_state.telegram_auth_in_progress = False
                        progress_bar.progress(100, text="Ошибка")
                        status.update(label="Ошибка при сборе", state="error")
                        status.write(str(exc))
                        st.error(f"Ошибка парсинга Telegram: {exc}")
                        with log_container:
                            st.error(f"Exception: {type(exc).__name__}: {exc}")
                    else:
                        # Успешное завершение - сбрасываем флаг авторизации
                        st.session_state.telegram_auth_in_progress = False
                        if not result_df.empty:
                            messages_df = result_df.copy()
                            messages_df["date"] = pd.to_datetime(messages_df["date_utc"], utc=True)
                            messages_df["channel"] = messages_df["channel_title"].fillna(messages_df["channel_username"])
                            messages_df["message_id"] = messages_df["id"]
                            st.session_state.telegram_messages = messages_df
                            status.update(label="Сбор завершён", state="complete")
                            status.write(f"Получено {len(messages_df)} сообщений")
                            st.success(f"Обработано {len(messages_df)} сообщений из {len(combined_usernames)} каналов")
                        else:
                            st.session_state.telegram_messages = result_df
                            status.update(label="Сбор завершён", state="complete")
                            status.write("Фильтры не обнаружили новых сообщений. Попробуйте ослабить условия.")
                            st.warning("Сообщения не найдены. Увеличьте окно поиска или уберите фильтры.")
                        progress_bar.progress(100, text="Готово")

        news_available = hasattr(st.session_state, 'telegram_messages') and not st.session_state.telegram_messages.empty

        if news_available:
            news_df = st.session_state.telegram_messages
            st.subheader("Сводка собранных сообщений")
            col1, col2, col3 = st.columns(3)
            col1.metric("Сообщений", len(news_df))
            col2.metric("Уникальных каналов", news_df['channel_username'].nunique())
            if 'date_utc' in news_df.columns:
                date_min = pd.to_datetime(news_df['date_utc']).min()
                date_max = pd.to_datetime(news_df['date_utc']).max()
                if pd.notna(date_min) and pd.notna(date_max):
                    col3.metric("Диапазон дат", f"{date_min.strftime('%d.%m %H:%M')} – {date_max.strftime('%d.%m %H:%M')}")
            preview_cols = [col for col in ['date_utc', 'channel', 'text', 'tickers', 'links'] if col in news_df.columns]
            st.dataframe(news_df[preview_cols].tail(25), use_container_width=True)

        # Analyze sentiment if messages are available
        if news_available:
            if st.button("🧠 Анализ настроений с помощью LLM", type="secondary"):
                with st.spinner("Анализ настроений с использованием LLM..."):
                    sentiment_results = st.session_state.llm_analyzer.analyze_batch_messages(
                        st.session_state.telegram_messages,
                        stock_symbols
                    )
                    if not sentiment_results.empty:
                        st.session_state.sentiment_analysis = sentiment_results
                        st.success(f"Проанализировано настроения для {len(sentiment_results)} сообщений")
                    else:
                        st.error("Ошибка анализа настроений")

            if hasattr(st.session_state, 'sentiment_analysis') and not st.session_state.sentiment_analysis.empty:
                sentiment_df = st.session_state.sentiment_analysis

                fig_sentiment = st.session_state.visualizer.plot_sentiment_analysis(sentiment_df)
                if fig_sentiment:
                    st.plotly_chart(fig_sentiment, use_container_width=True)

                st.subheader("Сводка настроений")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    positive_count = len(sentiment_df[sentiment_df['sentiment'] == 'positive'])
                    st.metric("Позитивные новости", positive_count)

                with col2:
                    negative_count = len(sentiment_df[sentiment_df['sentiment'] == 'negative'])
                    st.metric("Негативные новости", negative_count)

                with col3:
                    neutral_count = len(sentiment_df[sentiment_df['sentiment'] == 'neutral'])
                    st.metric("Нейтральные новости", neutral_count)

                with col4:
                    avg_confidence = sentiment_df['confidence'].mean()
                    st.metric("Ср. уверенность", f"{avg_confidence:.2f}")

                st.subheader("Недавние проанализированные сообщения")
                display_cols = ['date', 'channel', 'sentiment', 'confidence', 'impact_strength', 'summary']
                available_cols = [col for col in display_cols if col in sentiment_df.columns]
                st.dataframe(sentiment_df[available_cols].head(10))

        else:
            st.info("Не загружено сообщений Telegram. Настройте каналы и выполните сбор данных.")
    
    # Tab 3: Correlation Analysis
    with tab3:
        st.header("🔗 Анализ корреляции новостей и акций")
        
        # Check if we have both stock data and sentiment analysis
        has_stock_data = hasattr(st.session_state, 'stock_data_loaded') and st.session_state.stock_data_loaded
        has_sentiment = hasattr(st.session_state, 'sentiment_analysis') and not st.session_state.sentiment_analysis.empty
        
        if not has_stock_data:
            st.warning("Пожалуйста, сначала загрузите данные о акциях (вкладка Анализ акций)")
        elif not has_sentiment:
            st.warning("Пожалуйста, сначала проанализируйте настроения новостей (вкладка Настроения новостей)")
        else:
            # Select stock for correlation analysis
            correlation_stock = st.selectbox("Выберите акцию для анализа корреляции:", stock_symbols, key="corr_stock")

            if st.button("📊 Анализировать корреляцию", type="primary"):
                with st.spinner("Анализ корреляции между новостями и движениями акций..."):
                    correlation_df = st.session_state.llm_analyzer.correlate_news_with_stock_movements(
                    st.session_state.sentiment_analysis,
                    st.session_state.stock_analyzer,
                    correlation_stock
                )
                
                if correlation_df is not None and not correlation_df.empty:
                    st.session_state.correlation_results = correlation_df
                    st.success(f"Найдено {len(correlation_df)} точек корреляции")
                else:
                    st.warning("Не найдено данных корреляции за выбранный период")
        
        # Display correlation results
        if hasattr(st.session_state, 'correlation_results') and not st.session_state.correlation_results.empty:
            correlation_df = st.session_state.correlation_results
            
            # Correlation visualization
            fig_correlation = st.session_state.visualizer.plot_correlation_analysis(correlation_df)
            if fig_correlation:
                st.plotly_chart(fig_correlation, use_container_width=True)
            
            # Correlation metrics
            st.subheader("Метрики корреляции")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                sentiment_stock_corr = pd.Series(correlation_df['sentiment_score']).corr(pd.Series(correlation_df['stock_change_pct']))
                st.metric("Корреляция настроение-цена", f"{sentiment_stock_corr:.3f}")
            
            with col2:
                avg_impact = correlation_df['impact_strength'].mean()
                st.metric("Средняя сила воздействия", f"{avg_impact:.1f}")
            
            with col3:
                same_day_corr = correlation_df[correlation_df['days_offset'] == 0]
                if not same_day_corr.empty:
                    same_day_strength = pd.Series(same_day_corr['sentiment_score']).corr(pd.Series(same_day_corr['stock_change_pct']))
                    st.metric("Корреляция в тот же день", f"{same_day_strength:.3f}")
                else:
                    st.metric("Корреляция в тот же день", "N/A")
            
            with col4:
                high_confidence = correlation_df[correlation_df['confidence'] > 0.7]
                st.metric("Точки высокой уверенности", len(high_confidence))
            
            # Correlation data table
            st.subheader("Данные корреляции")
            display_cols = ['news_date', 'stock_date', 'days_offset', 'sentiment_score', 
                           'stock_change_pct', 'confidence', 'impact_strength']
            available_cols = [col for col in display_cols if col in correlation_df.columns]
            st.dataframe(correlation_df[available_cols])
    
    # Tab 4: Forecasting
    with tab4:
        st.header("🔮 Прогнозирование акций")
        
        # Check if required data is available
        has_stock_data = hasattr(st.session_state, 'stock_data_loaded') and st.session_state.stock_data_loaded
        
        if not has_stock_data:
            st.warning("Пожалуйста, сначала загрузите данные о акциях (вкладка Анализ акций)")
        else:
            # Forecast parameters
            col1, col2 = st.columns([2, 1])

            with col1:
                forecast_stock = st.selectbox("Выберите акцию для прогноза:", stock_symbols, key="forecast_stock")

            with col2:
                forecast_days = st.slider("Дни прогноза:", min_value=30, max_value=365, value=90)

            # Train model and generate forecast
            if st.button("🚀 Сгенерировать прогноз", type="primary"):
                with st.spinner("Обучение модели прогнозирования..."):
                    # Prepare data for training
                    stock_data = st.session_state.stock_analyzer.stock_data[forecast_stock]

                    # Use sentiment data if available
                    sentiment_data = None
                    if hasattr(st.session_state, 'sentiment_analysis') and not st.session_state.sentiment_analysis.empty:
                        sentiment_data = st.session_state.sentiment_analysis

                    prepared_data = st.session_state.forecaster.prepare_features(stock_data, sentiment_data)

                    if prepared_data is not None:
                        # Train the model
                        training_success = st.session_state.forecaster.train_model(prepared_data)

                        if training_success:
                            # Generate forecast
                            with st.spinner("Генерация прогноза..."):
                                recent_data = st.session_state.stock_analyzer.get_recent_data(forecast_stock, 30)
                                recent_prepared = st.session_state.forecaster.prepare_features(recent_data, sentiment_data)

                                if recent_prepared is not None:
                                    forecast_results = st.session_state.forecaster.generate_forecast(
                                        recent_prepared, forecast_days
                                    )

                                    if forecast_results is not None:
                                        st.session_state.forecast_results = forecast_results
                                        st.session_state.forecast_stock_data = stock_data
                                        st.success("Прогноз успешно сгенерирован!")
                                    else:
                                        st.error("Ошибка генерации прогноза")
                        else:
                            st.error("Ошибка обучения модели прогнозирования")
                    else:
                        st.error("Ошибка подготовки данных для прогноза")
        
            # Display forecast results
            if hasattr(st.session_state, 'forecast_results') and st.session_state.forecast_results is not None:
                forecast_df = st.session_state.forecast_results
                historical_data = st.session_state.forecast_stock_data
            
                # Forecast visualization
                fig_forecast = st.session_state.visualizer.plot_forecast(historical_data, forecast_df, forecast_stock)
                st.plotly_chart(fig_forecast, use_container_width=True)
            
                # Forecast summary
                st.subheader("Forecast Summary")
                col1, col2, col3, col4 = st.columns(4)
            
                current_price = historical_data['Close'].iloc[-1]
                final_predicted_price = forecast_df['Predicted_Price'].iloc[-1]
                total_return = ((final_predicted_price - current_price) / current_price) * 100
            
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")

                with col2:
                    st.metric("Predicted Price", f"${final_predicted_price:.2f}")

                with col3:
                    st.metric("Expected Return", f"{total_return:.2f}%")

                with col4:
                    max_price = forecast_df['Predicted_Price'].max()
                    st.metric("Max Predicted", f"${max_price:.2f}")
            
                # Feature importance
                feature_importance = st.session_state.forecaster.get_feature_importance()
                if feature_importance is not None:
                    st.subheader("Model Feature Importance")
                    st.bar_chart(feature_importance.set_index('Feature')['Importance'])
            
                # LLM-based forecast analysis
            if st.button("🧠 Generate LLM Forecast Analysis"):
                with st.spinner("Generating AI-powered forecast analysis..."):
                    # Prepare historical patterns summary
                    stats = st.session_state.stock_analyzer.calculate_statistics(forecast_stock)
                    rise_patterns, fall_patterns = st.session_state.stock_analyzer.identify_patterns_before_movements(forecast_stock)
                    
                    # Prepare sentiment summary
                    sentiment_summary = None
                    if hasattr(st.session_state, 'sentiment_analysis') and not st.session_state.sentiment_analysis.empty:
                        sentiment_df = st.session_state.sentiment_analysis
                        sentiment_summary = {
                            'total_messages': len(sentiment_df),
                            'positive_ratio': len(sentiment_df[sentiment_df['sentiment'] == 'positive']) / len(sentiment_df),
                            'negative_ratio': len(sentiment_df[sentiment_df['sentiment'] == 'negative']) / len(sentiment_df),
                            'avg_confidence': sentiment_df['confidence'].mean(),
                            'avg_impact_strength': sentiment_df['impact_strength'].mean()
                        }
                    
                    llm_analysis = st.session_state.llm_analyzer.generate_forecast_analysis(
                        {'stats': stats, 'patterns': {'rises': rise_patterns, 'falls': fall_patterns}},
                        sentiment_summary,
                        forecast_stock
                    )
                    
                    if llm_analysis:
                        st.session_state.llm_forecast_analysis = llm_analysis
                        
                        # Display LLM analysis
                        st.subheader("🧠 AI Forecast Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Quarter Outlook:**", llm_analysis.get('quarter_outlook', 'N/A'))
                            st.write("**Confidence Level:**", f"{llm_analysis.get('confidence_level', 0):.2f}")
                            
                            target_range = llm_analysis.get('target_price_range', {})
                            if target_range:
                                st.write("**Target Price Range:**", f"${target_range.get('min', 0):.2f} - ${target_range.get('max', 0):.2f}")
                        
                        with col2:
                            correlation_strength = llm_analysis.get('sentiment_correlation_strength', 0)
                            st.write("**Sentiment Correlation:**", f"{correlation_strength:.2f}")
                            st.write("**Time Horizon:**", llm_analysis.get('time_horizon', 'Q1 2026'))
                        
                        # Risks and opportunities
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Key Risks")
                            risks = llm_analysis.get('key_risks', [])
                            for risk in risks:
                                st.write(f"• {risk}")
                        
                        with col2:
                            st.subheader("Key Opportunities")
                            opportunities = llm_analysis.get('key_opportunities', [])
                            for opp in opportunities:
                                st.write(f"• {opp}")
                        
                        # Forecast summary
                        st.subheader("Forecast Summary")
                        summary = llm_analysis.get('forecast_summary', '')
                        st.write(summary)
    
    # Tab 5: Summary Report
    with tab5:
        st.header("📋 Analysis Summary Report")
        
        # Check what analyses have been completed
        has_stock_data = hasattr(st.session_state, 'stock_data_loaded') and st.session_state.stock_data_loaded
        has_sentiment = hasattr(st.session_state, 'sentiment_analysis')
        has_correlation = hasattr(st.session_state, 'correlation_results')
        has_forecast = hasattr(st.session_state, 'forecast_results')
        has_llm_analysis = hasattr(st.session_state, 'llm_forecast_analysis')
        
        # Analysis completion status
        st.subheader("Analysis Status")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = "✅ Complete" if has_stock_data else "❌ Pending"
            st.write(f"**Stock Analysis:** {status}")
        
        with col2:
            status = "✅ Complete" if has_sentiment else "❌ Pending"
            st.write(f"**News Sentiment:** {status}")
        
        with col3:
            status = "✅ Complete" if has_correlation else "❌ Pending"
            st.write(f"**Correlation:** {status}")
        
        with col4:
            status = "✅ Complete" if has_forecast else "❌ Pending"
            st.write(f"**Forecasting:** {status}")
        
        st.divider()
        
        # Key findings summary
        if has_stock_data:
            st.subheader("📊 Key Stock Analysis Findings")
            
            for symbol in stock_symbols:
                if symbol in st.session_state.stock_analyzer.stock_data:
                    stats = st.session_state.stock_analyzer.calculate_statistics(symbol)
                    if stats:
                        with st.expander(f"{symbol} Summary"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Mean Daily Return", f"{stats['mean']:.4f}%")
                                st.metric("Volatility", f"{stats['std']:.4f}%")
                            with col2:
                                st.metric("95th Percentile", f"{stats['quantile_95']:.4f}%")
                                st.metric("5th Percentile", f"{stats['quantile_05']:.4f}%")
                            with col3:
                                st.metric("Skewness", f"{stats['skewness']:.4f}")
                                st.metric("Kurtosis", f"{stats['kurtosis']:.4f}")
        
        # Sentiment summary
        if has_sentiment and not st.session_state.sentiment_analysis.empty:
            st.subheader("📱 News Sentiment Summary")
            sentiment_df = st.session_state.sentiment_analysis
            
            col1, col2, col3 = st.columns(3)
            with col1:
                positive_pct = len(sentiment_df[sentiment_df['sentiment'] == 'positive']) / len(sentiment_df) * 100
                st.metric("Positive News", f"{positive_pct:.1f}%")
            
            with col2:
                negative_pct = len(sentiment_df[sentiment_df['sentiment'] == 'negative']) / len(sentiment_df) * 100
                st.metric("Negative News", f"{negative_pct:.1f}%")
            
            with col3:
                avg_confidence = sentiment_df['confidence'].mean()
                st.metric("Average Confidence", f"{avg_confidence:.2f}")
        
        # Correlation summary
        if has_correlation and not st.session_state.correlation_results.empty:
            st.subheader("🔗 Correlation Summary")
            correlation_df = st.session_state.correlation_results
            
            col1, col2 = st.columns(2)
            with col1:
                overall_corr = pd.Series(correlation_df['sentiment_score']).corr(pd.Series(correlation_df['stock_change_pct']))
                st.metric("Overall Correlation", f"{overall_corr:.3f}")
            
            with col2:
                high_confidence = len(correlation_df[correlation_df['confidence'] > 0.7])
                st.metric("High Confidence Points", high_confidence)
        
        # Forecast summary
        if has_forecast and st.session_state.forecast_results is not None:
            st.subheader("🔮 Forecast Summary")
            forecast_df = st.session_state.forecast_results
            
            if hasattr(st.session_state, 'forecast_stock_data'):
                current_price = st.session_state.forecast_stock_data['Close'].iloc[-1]
                final_price = forecast_df['Predicted_Price'].iloc[-1]
                expected_return = ((final_price - current_price) / current_price) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with col2:
                    st.metric("90-Day Target", f"${final_price:.2f}")
                with col3:
                    st.metric("Expected Return", f"{expected_return:.2f}%")
        
        # LLM Analysis summary
        if has_llm_analysis:
            st.subheader("🧠 AI Analysis Summary")
            llm_analysis = st.session_state.llm_forecast_analysis
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Quarter Outlook:**", llm_analysis.get('quarter_outlook', 'N/A'))
                st.write("**Confidence Level:**", f"{llm_analysis.get('confidence_level', 0):.2f}")
            
            with col2:
                correlation_strength = llm_analysis.get('sentiment_correlation_strength', 0)
                st.write("**Sentiment Correlation:**", f"{correlation_strength:.2f}")
            
            # Executive summary
            summary = llm_analysis.get('forecast_summary', '')
            if summary:
                st.subheader("Executive Summary")
                st.write(summary)
        
        # Recommendations
        st.subheader("📈 Investment Recommendations")
        
        if has_stock_data and has_sentiment and has_forecast:
            st.success("**Comprehensive Analysis Completed**")
            st.write("""
            Based on the complete analysis combining historical patterns, news sentiment, and forecasting models:
            
            1. **Historical Analysis**: Review the statistical patterns and top movements to understand volatility profiles
            2. **Sentiment Impact**: Consider the correlation between news sentiment and price movements
            3. **Forecast Reliability**: Evaluate model confidence and feature importance
            4. **Risk Management**: Use the quantile analysis for position sizing and stop-loss levels
            
            **Note**: This analysis is for educational purposes. Always conduct your own research and consider consulting with financial advisors before making investment decisions.
            """)
        else:
            st.info("Complete all analysis steps for comprehensive investment recommendations.")
        
        # Export functionality
        st.subheader("📄 Export Results")
        
        if st.button("📊 Generate Analysis Report"):
            # Create a comprehensive report
            report_data = {
                "analysis_date": datetime.now().isoformat(),
                "stocks_analyzed": stock_symbols,
                "analysis_period": analysis_period,
                "has_stock_data": has_stock_data,
                "has_sentiment": has_sentiment,
                "has_correlation": has_correlation,
                "has_forecast": has_forecast
            }
            
            st.json(report_data)
            st.success("Analysis report generated! Copy the JSON data above for your records.")

if __name__ == "__main__":
    main()
