"""
Pattern Analyzer - анализ закономерностей перед крупными движениями акций.

Этот модуль отвечает за:
1. Обнаружение общих паттернов перед топ-20 ростами и падениями
2. Расчет статистической значимости паттернов
3. Определение порогов аномалий на основе квантилей
4. Сравнение текущей ситуации с историческими паттернами
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import timedelta
from typing import Dict, List, Tuple, Optional
import streamlit as st


class PatternAnalyzer:
    """
    Класс для анализа паттернов перед крупными движениями акций.
    """

    def __init__(self):
        self.pattern_cache = {}
        self.thresholds = {}

    def analyze_pre_movement_patterns(
        self,
        stock_data: pd.DataFrame,
        top_movements: pd.DataFrame,
        days_before: int = 5,
        movement_type: str = 'rise'
    ) -> pd.DataFrame:
        """
        Анализирует паттерны перед крупными движениями.

        Args:
            stock_data: Исторические данные акций
            top_movements: Топ-N движений (росты или падения)
            days_before: Количество дней до события для анализа
            movement_type: 'rise' или 'fall'

        Returns:
            DataFrame с паттернами и признаками
        """
        patterns = []

        for date in top_movements.index:
            # Получаем данные за N дней до события
            start_date = date - timedelta(days=days_before)
            pattern_data = stock_data[start_date:date]

            if len(pattern_data) < days_before:
                continue

            # Извлекаем признаки паттерна
            pattern = {
                'event_date': date,
                'movement_pct': top_movements.loc[date, 'Daily_Change_Pct'],
                'movement_type': movement_type,

                # Объём торгов
                'volume_change_1d': self._calculate_volume_change(pattern_data, 1),
                'volume_change_3d': self._calculate_volume_change(pattern_data, 3),
                'volume_change_5d': self._calculate_volume_change(pattern_data, 5),
                'volume_mean': pattern_data['Volume'].mean(),
                'volume_std': pattern_data['Volume'].std(),

                # Волатильность
                'volatility_1d': pattern_data['Close'].pct_change().tail(1).std(),
                'volatility_3d': pattern_data['Close'].pct_change().tail(3).std(),
                'volatility_5d': pattern_data['Close'].pct_change().tail(5).std(),
                'price_range_pct': ((pattern_data['High'] - pattern_data['Low']) / pattern_data['Close']).mean(),

                # Ценовые паттерны
                'price_trend': self._calculate_price_trend(pattern_data),
                'sma5_cross': self._check_sma_cross(pattern_data, window=5),
                'sma20_cross': self._check_sma_cross(pattern_data, window=20),

                # RSI
                'rsi_final': self._calculate_rsi(pattern_data['Close']).iloc[-1] if len(pattern_data) >= 14 else np.nan,
                'rsi_mean': self._calculate_rsi(pattern_data['Close']).mean() if len(pattern_data) >= 14 else np.nan,

                # Разрыв цен (gap)
                'gap_up_count': self._count_gaps(pattern_data, gap_type='up'),
                'gap_down_count': self._count_gaps(pattern_data, gap_type='down'),
            }

            patterns.append(pattern)

        return pd.DataFrame(patterns)

    def find_common_patterns(
        self,
        rise_patterns: pd.DataFrame,
        fall_patterns: pd.DataFrame,
        significance_level: float = 0.05
    ) -> Dict[str, Dict]:
        """
        Находит общие закономерности в паттернах ростов и падений.

        Args:
            rise_patterns: Паттерны перед ростами
            fall_patterns: Паттерны перед падениями
            significance_level: Уровень значимости для статистических тестов

        Returns:
            Словарь с найденными закономерностями
        """
        common_patterns = {
            'rises': {},
            'falls': {},
            'differences': {}
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

            # Статистика для ростов
            rise_values = rise_patterns[feature].dropna()
            if len(rise_values) > 0:
                common_patterns['rises'][feature] = {
                    'mean': rise_values.mean(),
                    'median': rise_values.median(),
                    'std': rise_values.std(),
                    'q95': rise_values.quantile(0.95),
                    'q99': rise_values.quantile(0.99),
                    'q05': rise_values.quantile(0.05),
                    'q01': rise_values.quantile(0.01),
                }

            # Статистика для падений
            fall_values = fall_patterns[feature].dropna()
            if len(fall_values) > 0:
                common_patterns['falls'][feature] = {
                    'mean': fall_values.mean(),
                    'median': fall_values.median(),
                    'std': fall_values.std(),
                    'q95': fall_values.quantile(0.95),
                    'q99': fall_values.quantile(0.99),
                    'q05': fall_values.quantile(0.05),
                    'q01': fall_values.quantile(0.01),
                }

            # Тест на различия между ростами и падениями
            if len(rise_values) > 0 and len(fall_values) > 0:
                t_stat, p_value = stats.ttest_ind(rise_values, fall_values)

                common_patterns['differences'][feature] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < significance_level,
                    'rise_mean': rise_values.mean(),
                    'fall_mean': fall_values.mean(),
                    'difference': rise_values.mean() - fall_values.mean()
                }

        return common_patterns

    def detect_current_signals(
        self,
        current_data: pd.DataFrame,
        pattern_thresholds: Dict[str, Dict],
        days_window: int = 5
    ) -> Dict[str, any]:
        """
        Обнаруживает текущие сигналы на основе исторических паттернов.

        Args:
            current_data: Текущие данные акции
            pattern_thresholds: Пороги из исторических паттернов
            days_window: Окно для анализа текущих данных

        Returns:
            Словарь с обнаруженными сигналами
        """
        recent_data = current_data.tail(days_window)

        signals = {
            'rise_signals': [],
            'fall_signals': [],
            'neutral_signals': [],
            'overall_score': 0,
            'confidence': 0
        }

        # Проверяем каждый признак
        if 'rises' in pattern_thresholds and 'falls' in pattern_thresholds:
            # Объём
            current_volume_change = self._calculate_volume_change(recent_data, days_window)

            if 'volume_change_5d' in pattern_thresholds['rises']:
                rise_threshold = pattern_thresholds['rises']['volume_change_5d']['median']
                fall_threshold = pattern_thresholds['falls']['volume_change_5d']['median']

                if current_volume_change > rise_threshold:
                    signals['rise_signals'].append(f"Объём выше медианы перед ростами: {current_volume_change:.2f}% vs {rise_threshold:.2f}%")
                elif current_volume_change < fall_threshold:
                    signals['fall_signals'].append(f"Объём ниже медианы перед падениями: {current_volume_change:.2f}% vs {fall_threshold:.2f}%")

            # Волатильность
            current_volatility = recent_data['Close'].pct_change().std()

            if 'volatility_5d' in pattern_thresholds['rises']:
                rise_vol = pattern_thresholds['rises']['volatility_5d']['median']
                fall_vol = pattern_thresholds['falls']['volatility_5d']['median']

                if current_volatility > max(rise_vol, fall_vol):
                    signals['neutral_signals'].append(f"Высокая волатильность: {current_volatility:.4f}")

            # RSI
            current_rsi = self._calculate_rsi(current_data['Close']).iloc[-1]

            if not np.isnan(current_rsi):
                if current_rsi < 30:
                    signals['rise_signals'].append(f"RSI перепродан: {current_rsi:.2f}")
                elif current_rsi > 70:
                    signals['fall_signals'].append(f"RSI перекуплен: {current_rsi:.2f}")

        # Расчет общего скора
        signals['overall_score'] = len(signals['rise_signals']) - len(signals['fall_signals'])
        total_signals = len(signals['rise_signals']) + len(signals['fall_signals']) + len(signals['neutral_signals'])
        signals['confidence'] = min(total_signals / 5.0, 1.0)  # Нормализуем от 0 до 1

        return signals

    def calculate_pattern_similarity(
        self,
        current_pattern: Dict,
        historical_patterns: pd.DataFrame,
        top_n: int = 5
    ) -> pd.DataFrame:
        """
        Вычисляет схожесть текущего паттерна с историческими.

        Args:
            current_pattern: Текущий паттерн (словарь признаков)
            historical_patterns: Исторические паттерны
            top_n: Количество наиболее похожих паттернов

        Returns:
            DataFrame с топ-N наиболее похожими паттернами
        """
        similarities = []

        # Признаки для сравнения
        features = ['volume_change_5d', 'volatility_5d', 'price_trend', 'rsi_final']

        for idx, hist_pattern in historical_patterns.iterrows():
            similarity_score = 0
            valid_features = 0

            for feature in features:
                if feature in current_pattern and feature in hist_pattern and not pd.isna(hist_pattern[feature]):
                    # Нормализованная разница
                    diff = abs(current_pattern[feature] - hist_pattern[feature])
                    max_val = max(abs(current_pattern[feature]), abs(hist_pattern[feature]), 1e-10)
                    normalized_diff = 1 - (diff / max_val)
                    similarity_score += normalized_diff
                    valid_features += 1

            if valid_features > 0:
                avg_similarity = similarity_score / valid_features
                similarities.append({
                    'event_date': hist_pattern['event_date'],
                    'movement_pct': hist_pattern['movement_pct'],
                    'similarity': avg_similarity
                })

        similarity_df = pd.DataFrame(similarities)

        if not similarity_df.empty:
            similarity_df = similarity_df.sort_values('similarity', ascending=False).head(top_n)

        return similarity_df

    # Вспомогательные методы

    def _calculate_volume_change(self, data: pd.DataFrame, days: int) -> float:
        """Рассчитывает изменение объёма за N дней"""
        if len(data) < days + 1:
            return 0

        recent_volume = data['Volume'].tail(days).mean()
        previous_volume = data['Volume'].iloc[-(days+1):-(days-1)].mean()

        if previous_volume > 0:
            return ((recent_volume - previous_volume) / previous_volume) * 100
        return 0

    def _calculate_price_trend(self, data: pd.DataFrame) -> float:
        """Рассчитывает тренд цены (угол линейной регрессии)"""
        if len(data) < 2:
            return 0

        prices = data['Close'].values
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)

        # Нормализуем относительно средней цены
        return (slope / prices.mean()) * 100 if prices.mean() > 0 else 0

    def _check_sma_cross(self, data: pd.DataFrame, window: int = 5) -> int:
        """Проверяет пересечение цены с SMA"""
        if len(data) < window + 1:
            return 0

        sma = data['Close'].rolling(window=window).mean()
        price = data['Close']

        # 1 = цена выше SMA, -1 = цена ниже SMA, 0 = пересечение
        if price.iloc[-1] > sma.iloc[-1] and price.iloc[-2] <= sma.iloc[-2]:
            return 1  # Бычье пересечение
        elif price.iloc[-1] < sma.iloc[-1] and price.iloc[-2] >= sma.iloc[-2]:
            return -1  # Медвежье пересечение

        return 0

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Рассчитывает RSI индикатор"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _count_gaps(self, data: pd.DataFrame, gap_type: str = 'up') -> int:
        """Считает количество ценовых разрывов (gaps)"""
        if len(data) < 2:
            return 0

        gaps = 0
        for i in range(1, len(data)):
            prev_close = data['Close'].iloc[i-1]
            curr_open = data['Open'].iloc[i]

            if gap_type == 'up' and curr_open > prev_close * 1.01:  # Gap up > 1%
                gaps += 1
            elif gap_type == 'down' and curr_open < prev_close * 0.99:  # Gap down > 1%
                gaps += 1

        return gaps
