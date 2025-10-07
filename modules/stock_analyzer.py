import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yfinance as yf
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import streamlit as st
from typing import Dict, List, Tuple
import numpy as np

class StockAnalyzer:
    def __init__(self):
        self.stock_data = {}
        self.pattern_analyzer = None
        
    def fetch_stock_data(self, symbols, period="2y"):
        """Fetch historical stock data for given symbols"""
        try:
            successful_loads = 0
            for symbol in symbols:
                yahoo_data = self._fetch_from_yahoo(symbol, period)
                if yahoo_data is not None and not yahoo_data.empty:
                    self.stock_data[symbol] = yahoo_data
                    successful_loads += 1
                    continue

                # Fallback: try MOEX ISS API for Russian equities
                moex_symbol = self._normalize_moex_symbol(symbol)
                moex_data = self._fetch_from_moex(moex_symbol, period)

                if moex_data is not None and not moex_data.empty:
                    self.stock_data[symbol] = moex_data
                    successful_loads += 1
                    st.info(f"Данные для {symbol} получены через MOEX ISS API")
                else:
                    # Попробуем автоматически добавить суффикс .ME для Yahoo Finance
                    alt_symbol = symbol.upper()
                    yahoo_alt = None
                    if not alt_symbol.endswith(".ME"):
                        yahoo_alt = self._fetch_from_yahoo(f"{alt_symbol}.ME", period)
                        if yahoo_alt is not None and not yahoo_alt.empty:
                            self.stock_data[symbol] = yahoo_alt
                            successful_loads += 1
                            st.info(f"Данные для {symbol} загружены как {alt_symbol}.ME (Yahoo Finance)")
                            continue

                    st.warning(
                        f"Данные не найдены для {symbol}. "
                        "Попробуйте указать тикер с суффиксом .ME или убедитесь, что актив торгуется на MOEX."
                    )
            
            if successful_loads > 0:
                return True
            else:
                st.error("Не удалось загрузить данные ни для одной акции")
                return False
        except Exception as e:
            st.error(f"Ошибка загрузки данных об акциях: {str(e)}")
            return False

    def _fetch_from_yahoo(self, symbol, period):
        """Try loading data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            if data is None or data.empty:
                return None

            # Ensure index naming consistency
            data.index.name = 'Date'
            return data
        except Exception as exc:
            st.warning(f"Yahoo Finance не вернул данные для {symbol}: {exc}")
            return None

    def _normalize_moex_symbol(self, symbol: str) -> str:
        """Prepare symbol for MOEX lookup (remove Yahoo-specific suffixes)."""
        if symbol.upper().endswith('.ME'):
            return symbol.upper().replace('.ME', '')
        return symbol.upper()

    def _fetch_from_moex(self, symbol: str, period: str):
        """Fetch history from MOEX ISS open API"""
        if not symbol:
            return None

        days_map = {"1y": 365, "2y": 730, "5y": 1825}
        days_back = days_map.get(period, 730)
        start_date = datetime.now() - timedelta(days=days_back)
        start_param = start_date.strftime('%Y-%m-%d')

        all_rows = []
        start = 0
        page_guard = 0
        base_url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{symbol}.json"

        while True:
            url = f"{base_url}?from={start_param}&start={start}"
            try:
                req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(req, timeout=10) as response:
                    payload = json.loads(response.read().decode('utf-8'))
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                st.warning(f"MOEX API недоступен для {symbol}: {exc}")
                return None

            history = payload.get('history', {})
            rows = history.get('data', [])
            columns = history.get('columns', [])

            if not rows:
                break

            chunk_df = pd.DataFrame(rows, columns=columns)
            all_rows.append(chunk_df)

            # Pagination handling: MOEX returns up to 100 rows per page
            if len(rows) < 100:
                break
            start += len(rows)
            page_guard += 1
            if page_guard > 50:
                st.warning(f"Превышен лимит страниц MOEX для {symbol}. Данные могут быть неполными.")
                break

        if not all_rows:
            return None

        data = pd.concat(all_rows, ignore_index=True)

        required_cols = ['TRADEDATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
        if not set(required_cols).issubset(set(data.columns)):
            st.warning(f"MOEX не вернул полный набор полей для {symbol}")
            return None

        data = data[required_cols]
        data = data.rename(columns={
            'TRADEDATE': 'Date',
            'OPEN': 'Open',
            'HIGH': 'High',
            'LOW': 'Low',
            'CLOSE': 'Close',
            'VOLUME': 'Volume'
        })

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        data.dropna(subset=['Close'], inplace=True)
        if data.empty:
            return None

        data['Date'] = pd.to_datetime(data['Date'])
        data.sort_values('Date', inplace=True)
        data.set_index('Date', inplace=True)
        data.index.name = 'Date'
        data['Adj Close'] = data['Close']

        return data
    
    def calculate_price_changes(self, symbol):
        """Calculate daily price changes for a stock"""
        if symbol not in self.stock_data:
            return None
            
        data = self.stock_data[symbol]
        data['Daily_Change_Pct'] = ((data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1)) * 100
        data['Daily_Change_Abs'] = data['Close'] - data['Close'].shift(1)
        
        return data
    
    def find_top_movements(self, symbol, top_n=20):
        """Find top N rises and falls for a stock"""
        data = self.calculate_price_changes(symbol)
        if data is None:
            return None, None
            
        # Remove NaN values
        clean_data = data.dropna()
        
        # Get top rises and falls
        top_rises = clean_data.nlargest(top_n, 'Daily_Change_Pct')
        top_falls = clean_data.nsmallest(top_n, 'Daily_Change_Pct')
        
        return top_rises, top_falls
    
    def calculate_statistics(self, symbol):
        """Calculate statistical measures including quantiles"""
        data = self.calculate_price_changes(symbol)
        if data is None:
            return None
            
        changes = data['Daily_Change_Pct'].dropna()
        
        stats_dict = {
            'mean': changes.mean(),
            'std': changes.std(),
            'median': changes.median(),
            'quantile_95': changes.quantile(0.95),
            'quantile_99': changes.quantile(0.99),
            'quantile_05': changes.quantile(0.05),
            'quantile_01': changes.quantile(0.01),
            'skewness': stats.skew(changes),
            'kurtosis': stats.kurtosis(changes)
        }
        
        return stats_dict
    
    def identify_patterns_before_movements(self, symbol, days_before=5):
        """Identify patterns in the days before major movements"""
        top_rises, top_falls = self.find_top_movements(symbol)
        if top_rises is None or top_falls is None:
            return None, None
            
        data = self.stock_data[symbol]
        
        rise_patterns = []
        fall_patterns = []
        
        # Analyze patterns before major rises
        for date in top_rises.index:
            start_date = date - timedelta(days=days_before)
            pattern_data = data[start_date:date]
            if len(pattern_data) >= days_before:
                pattern = {
                    'date': date,
                    'movement': top_rises.loc[date, 'Daily_Change_Pct'],
                    'volume_trend': pattern_data['Volume'].pct_change().mean(),
                    'price_volatility': pattern_data['Close'].pct_change().std(),
                    'avg_volume': pattern_data['Volume'].mean()
                }
                rise_patterns.append(pattern)
        
        # Analyze patterns before major falls
        for date in top_falls.index:
            start_date = date - timedelta(days=days_before)
            pattern_data = data[start_date:date]
            if len(pattern_data) >= days_before:
                pattern = {
                    'date': date,
                    'movement': top_falls.loc[date, 'Daily_Change_Pct'],
                    'volume_trend': pattern_data['Volume'].pct_change().mean(),
                    'price_volatility': pattern_data['Close'].pct_change().std(),
                    'avg_volume': pattern_data['Volume'].mean()
                }
                fall_patterns.append(pattern)
        
        return rise_patterns, fall_patterns
    
    def get_recent_data(self, symbol, days=30):
        """Get recent stock data for current analysis"""
        if symbol not in self.stock_data:
            return None

        data = self.stock_data[symbol]
        recent_data = data.tail(days)
        return recent_data

    def find_common_patterns_across_stocks(
        self,
        top_n: int = 20,
        days_before: int = 5,
        significance_level: float = 0.05
    ) -> Dict:
        """
        Анализирует топ-N движений по ВСЕМ выбранным акциям и находит общие закономерности.

        Args:
            top_n: Количество топ движений для каждой акции
            days_before: Количество дней до события для анализа
            significance_level: Уровень значимости для статистических тестов

        Returns:
            Словарь с агрегированными паттернами по всем акциям
        """
        try:
            # Импортируем PatternAnalyzer при первом использовании
            if self.pattern_analyzer is None:
                from modules.pattern_analyzer import PatternAnalyzer
                self.pattern_analyzer = PatternAnalyzer()

            all_rise_patterns = []
            all_fall_patterns = []

            # Собираем паттерны по всем акциям
            for symbol in self.stock_data.keys():
                top_rises, top_falls = self.find_top_movements(symbol, top_n=top_n)

                if top_rises is None or top_falls is None:
                    continue

                stock_data = self.stock_data[symbol]

                # Анализируем паттерны перед ростами
                rise_patterns = self.pattern_analyzer.analyze_pre_movement_patterns(
                    stock_data, top_rises, days_before=days_before, movement_type='rise'
                )
                rise_patterns['symbol'] = symbol
                all_rise_patterns.append(rise_patterns)

                # Анализируем паттерны перед падениями
                fall_patterns = self.pattern_analyzer.analyze_pre_movement_patterns(
                    stock_data, top_falls, days_before=days_before, movement_type='fall'
                )
                fall_patterns['symbol'] = symbol
                all_fall_patterns.append(fall_patterns)

            if not all_rise_patterns or not all_fall_patterns:
                st.warning("Недостаточно данных для анализа паттернов")
                return None

            # Объединяем паттерны всех акций
            combined_rise_patterns = pd.concat(all_rise_patterns, ignore_index=True)
            combined_fall_patterns = pd.concat(all_fall_patterns, ignore_index=True)

            # Находим общие закономерности
            common_patterns = self.pattern_analyzer.find_common_patterns(
                combined_rise_patterns,
                combined_fall_patterns,
                significance_level=significance_level
            )

            # Добавляем метаданные
            result = {
                'common_patterns': common_patterns,
                'rise_patterns_df': combined_rise_patterns,
                'fall_patterns_df': combined_fall_patterns,
                'total_stocks': len(self.stock_data),
                'total_rise_events': len(combined_rise_patterns),
                'total_fall_events': len(combined_fall_patterns),
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            st.success(f"Проанализировано {result['total_rise_events']} ростов и {result['total_fall_events']} падений по {result['total_stocks']} акциям")

            return result

        except Exception as e:
            st.error(f"Ошибка анализа паттернов: {str(e)}")
            return None

    def calculate_pattern_thresholds(
        self,
        pattern_results: Dict,
        quantiles: List[float] = [0.95, 0.99, 0.05, 0.01]
    ) -> Dict:
        """
        Определяет пороги для аномальных сигналов на основе квантилей.

        Args:
            pattern_results: Результаты анализа паттернов (из find_common_patterns_across_stocks)
            quantiles: Список квантилей для расчёта порогов

        Returns:
            Словарь с порогами для каждого признака
        """
        try:
            if pattern_results is None:
                st.error("Нет результатов анализа паттернов")
                return None

            rise_patterns = pattern_results['rise_patterns_df']
            fall_patterns = pattern_results['fall_patterns_df']

            thresholds = {
                'rises': {},
                'falls': {},
                'anomaly_thresholds': {}
            }

            # Числовые признаки для анализа
            numeric_features = [
                'volume_change_1d', 'volume_change_3d', 'volume_change_5d',
                'volatility_1d', 'volatility_3d', 'volatility_5d',
                'price_range_pct', 'price_trend', 'rsi_final', 'rsi_mean'
            ]

            for feature in numeric_features:
                if feature not in rise_patterns.columns or feature not in fall_patterns.columns:
                    continue

                rise_values = rise_patterns[feature].dropna()
                fall_values = fall_patterns[feature].dropna()

                if len(rise_values) == 0 or len(fall_values) == 0:
                    continue

                # Рассчитываем квантили для ростов
                thresholds['rises'][feature] = {}
                for q in quantiles:
                    thresholds['rises'][feature][f'q{int(q*100)}'] = rise_values.quantile(q)

                # Рассчитываем квантили для падений
                thresholds['falls'][feature] = {}
                for q in quantiles:
                    thresholds['falls'][feature][f'q{int(q*100)}'] = fall_values.quantile(q)

                # Определяем пороги аномалий (экстремальные значения)
                thresholds['anomaly_thresholds'][feature] = {
                    'rise_extreme_high': rise_values.quantile(0.99),
                    'rise_extreme_low': rise_values.quantile(0.01),
                    'fall_extreme_high': fall_values.quantile(0.99),
                    'fall_extreme_low': fall_values.quantile(0.01),
                    'rise_median': rise_values.median(),
                    'fall_median': fall_values.median()
                }

            # Добавляем статистику по объёму торгов
            if 'volume_mean' in rise_patterns.columns and 'volume_std' in rise_patterns.columns:
                thresholds['volume_statistics'] = {
                    'rise_volume_mean': rise_patterns['volume_mean'].mean(),
                    'rise_volume_std': rise_patterns['volume_std'].mean(),
                    'fall_volume_mean': fall_patterns['volume_mean'].mean(),
                    'fall_volume_std': fall_patterns['volume_std'].mean()
                }

            st.success(f"Рассчитаны пороги для {len(numeric_features)} признаков")

            return thresholds

        except Exception as e:
            st.error(f"Ошибка расчёта порогов: {str(e)}")
            return None

    def detect_current_anomalies(
        self,
        symbol: str,
        thresholds: Dict,
        days_window: int = 5
    ) -> Dict:
        """
        Обнаруживает текущие аномальные сигналы для конкретной акции.

        Args:
            symbol: Символ акции
            thresholds: Пороги из calculate_pattern_thresholds
            days_window: Окно для анализа текущих данных

        Returns:
            Словарь с обнаруженными сигналами и их оценкой
        """
        try:
            if symbol not in self.stock_data:
                st.error(f"Нет данных для {symbol}")
                return None

            if thresholds is None:
                st.error("Нет порогов для сравнения")
                return None

            # Импортируем PatternAnalyzer при первом использовании
            if self.pattern_analyzer is None:
                from modules.pattern_analyzer import PatternAnalyzer
                self.pattern_analyzer = PatternAnalyzer()

            current_data = self.stock_data[symbol]

            # Используем detect_current_signals из PatternAnalyzer
            signals = self.pattern_analyzer.detect_current_signals(
                current_data,
                thresholds,
                days_window=days_window
            )

            # Добавляем информацию о символе
            signals['symbol'] = symbol
            signals['analysis_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            return signals

        except Exception as e:
            st.error(f"Ошибка обнаружения аномалий для {symbol}: {str(e)}")
            return None
