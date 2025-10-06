import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional

class Visualizer:
    def __init__(self):
        self.color_palette = {
            'positive': '#00ff00',
            'negative': '#ff0000',
            'neutral': '#808080',
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'rise': '#26a69a',
            'fall': '#ef5350',
            'warning': '#ffa726',
            'info': '#42a5f5'
        }
    
    def plot_stock_price_history(self, stock_data, symbol):
        """Plot stock price history with volume"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} Price History', 'Volume'),
            row_width=[0.2, 0.7]
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color=self.color_palette['primary'])
            ),
            row=1, col=1
        )
        
        # Volume chart
        fig.add_trace(
            go.Bar(
                x=stock_data.index,
                y=stock_data['Volume'],
                name='Volume',
                marker_color=self.color_palette['secondary'],
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'{symbol} Stock Analysis',
            xaxis_title='Date',
            height=600,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def plot_price_distribution(self, stock_data, symbol):
        """Plot price change distribution with box plot"""
        data = stock_data.copy()
        data['Daily_Change_Pct'] = ((data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1)) * 100
        daily_changes = data['Daily_Change_Pct'].dropna()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Daily Returns Distribution', 'Box Plot with Quantiles')
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=daily_changes,
                nbinsx=50,
                name='Daily Returns',
                marker_color=self.color_palette['primary'],
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                y=daily_changes,
                name='Daily Returns',
                boxpoints='outliers',
                marker_color=self.color_palette['secondary']
            ),
            row=1, col=2
        )
        
        # Add quantile lines
        q99 = daily_changes.quantile(0.99)
        q95 = daily_changes.quantile(0.95)
        q05 = daily_changes.quantile(0.05)
        q01 = daily_changes.quantile(0.01)
        
        for q_val, q_name, color in [(q99, '99th percentile', 'red'), 
                                     (q95, '95th percentile', 'orange'),
                                     (q05, '5th percentile', 'orange'),
                                     (q01, '1st percentile', 'red')]:
            fig.add_hline(
                y=q_val, 
                line_dash="dash", 
                line_color=color,
                annotation_text=q_name,
                row="all", col="all"
            )
        
        fig.update_layout(
            title=f'{symbol} Daily Returns Analysis',
            height=500,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Daily Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Daily Return (%)", row=1, col=2)
        
        return fig
    
    def plot_top_movements(self, top_rises, top_falls, symbol):
        """Plot top rises and falls"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Top 20 Rises', 'Top 20 Falls')
        )
        
        # Top rises
        fig.add_trace(
            go.Bar(
                x=list(range(len(top_rises))),
                y=top_rises['Daily_Change_Pct'],
                name='Top Rises',
                marker_color=self.color_palette['positive'],
                text=[f"{val:.2f}%" for val in top_rises['Daily_Change_Pct']],
                textposition='auto',
            ),
            row=1, col=1
        )
        
        # Top falls
        fig.add_trace(
            go.Bar(
                x=list(range(len(top_falls))),
                y=top_falls['Daily_Change_Pct'],
                name='Top Falls',
                marker_color=self.color_palette['negative'],
                text=[f"{val:.2f}%" for val in top_falls['Daily_Change_Pct']],
                textposition='auto',
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f'{symbol} Top 20 Price Movements',
            height=500,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Rank", row=1, col=1)
        fig.update_xaxes(title_text="Rank", row=1, col=2)
        fig.update_yaxes(title_text="Change (%)", row=1, col=1)
        fig.update_yaxes(title_text="Change (%)", row=1, col=2)
        
        return fig
    
    def plot_sentiment_analysis(self, sentiment_df):
        """Plot sentiment analysis results"""
        if sentiment_df.empty:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Distribution', 'Sentiment Over Time', 
                           'Impact Strength vs Confidence', 'Sentiment by Channel')
        )
        
        # Sentiment distribution
        sentiment_counts = sentiment_df['sentiment'].value_counts()
        colors = [self.color_palette.get(sent, '#808080') for sent in sentiment_counts.index]
        
        fig.add_trace(
            go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                name='Sentiment',
                marker_colors=colors
            ),
            row=1, col=1
        )
        
        # Sentiment over time
        daily_sentiment = sentiment_df.groupby(sentiment_df['date'].dt.date).agg({
            'sentiment': lambda x: (x == 'positive').sum() - (x == 'negative').sum(),
            'confidence': 'mean'
        }).reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=daily_sentiment['date'],
                y=daily_sentiment['sentiment'],
                mode='lines+markers',
                name='Daily Sentiment Score',
                line=dict(color=self.color_palette['primary'])
            ),
            row=1, col=2
        )
        
        # Impact vs Confidence scatter
        fig.add_trace(
            go.Scatter(
                x=sentiment_df['confidence'],
                y=sentiment_df['impact_strength'],
                mode='markers',
                name='Messages',
                marker=dict(
                    color=sentiment_df['sentiment'].map({
                        'positive': self.color_palette['positive'],
                        'negative': self.color_palette['negative'],
                        'neutral': self.color_palette['neutral']
                    }),
                    size=8
                )
            ),
            row=2, col=1
        )
        
        # Sentiment by channel
        channel_sentiment = sentiment_df.groupby(['channel', 'sentiment']).size().unstack(fill_value=0)
        
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in channel_sentiment.columns:
                fig.add_trace(
                    go.Bar(
                        x=channel_sentiment.index,
                        y=channel_sentiment[sentiment],
                        name=sentiment.title(),
                        marker_color=self.color_palette[sentiment]
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title='News Sentiment Analysis Dashboard',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def plot_correlation_analysis(self, correlation_df):
        """Plot correlation between news sentiment and stock movements"""
        if correlation_df is None or correlation_df.empty:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment vs Stock Change', 'Correlation by Day Offset',
                           'News Volume vs Stock Movement', 'Impact Strength Distribution')
        )
        
        # Sentiment vs Stock Change scatter
        colors = correlation_df['sentiment_score'].apply(
            lambda x: self.color_palette['positive'] if x > 0 
            else self.color_palette['negative'] if x < 0 
            else self.color_palette['neutral']
        )
        
        fig.add_trace(
            go.Scatter(
                x=correlation_df['sentiment_score'],
                y=correlation_df['stock_change_pct'],
                mode='markers',
                name='Correlation Points',
                marker=dict(color=colors, size=8)
            ),
            row=1, col=1
        )
        
        # Correlation by day offset
        offset_corr = correlation_df.groupby('days_offset').agg({
            'sentiment_score': 'mean',
            'stock_change_pct': 'mean'
        }).reset_index()
        
        fig.add_trace(
            go.Bar(
                x=offset_corr['days_offset'],
                y=offset_corr['stock_change_pct'],
                name='Avg Stock Change by Offset',
                marker_color=self.color_palette['secondary']
            ),
            row=1, col=2
        )
        
        # News volume vs stock movement
        fig.add_trace(
            go.Scatter(
                x=correlation_df['news_volume'],
                y=correlation_df['stock_change_pct'],
                mode='markers',
                name='Volume vs Change',
                marker=dict(
                    color=correlation_df['relevance_score'],
                    colorscale='Viridis',
                    size=8,
                    colorbar=dict(title="Relevance Score")
                )
            ),
            row=2, col=1
        )
        
        # Impact strength distribution
        fig.add_trace(
            go.Histogram(
                x=correlation_df['impact_strength'],
                nbinsx=20,
                name='Impact Strength',
                marker_color=self.color_palette['primary']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='News-Stock Correlation Analysis',
            height=800
        )
        
        fig.update_xaxes(title_text="Sentiment Score", row=1, col=1)
        fig.update_yaxes(title_text="Stock Change (%)", row=1, col=1)
        fig.update_xaxes(title_text="Days Offset", row=1, col=2)
        fig.update_yaxes(title_text="Avg Stock Change (%)", row=1, col=2)
        fig.update_xaxes(title_text="News Volume", row=2, col=1)
        fig.update_yaxes(title_text="Stock Change (%)", row=2, col=1)
        fig.update_xaxes(title_text="Impact Strength", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        return fig
    
    def plot_forecast(self, historical_data, forecast_data, symbol):
        """Plot stock forecast with confidence intervals"""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(
            go.Scatter(
                x=historical_data.index,
                y=historical_data['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color=self.color_palette['primary'])
            )
        )
        
        # Forecast
        fig.add_trace(
            go.Scatter(
                x=forecast_data['Date'],
                y=forecast_data['Predicted_Price'],
                mode='lines',
                name='Forecast',
                line=dict(color=self.color_palette['secondary'], dash='dash')
            )
        )
        
        # Confidence interval
        fig.add_trace(
            go.Scatter(
                x=forecast_data['Date'],
                y=forecast_data['Confidence_High'],
                mode='lines',
                name='Upper Bound',
                line=dict(color='rgba(255,127,14,0.3)'),
                showlegend=False
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast_data['Date'],
                y=forecast_data['Confidence_Low'],
                mode='lines',
                name='Lower Bound',
                line=dict(color='rgba(255,127,14,0.3)'),
                fill='tonexty',
                fillcolor='rgba(255,127,14,0.2)',
                showlegend=False
            )
        )
        
        fig.update_layout(
            title=f'{symbol} Price Forecast (90 Days)',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=600,
            hovermode='x unified'
        )
        
        return fig
    
    def display_statistics_table(self, stats_dict, symbol):
        """Display statistics in a formatted table"""
        if stats_dict is None:
            return None

        stats_data = [
            ['Mean Daily Return (%)', f"{stats_dict['mean']:.4f}"],
            ['Standard Deviation (%)', f"{stats_dict['std']:.4f}"],
            ['Median Daily Return (%)', f"{stats_dict['median']:.4f}"],
            ['95th Percentile (%)', f"{stats_dict['quantile_95']:.4f}"],
            ['99th Percentile (%)', f"{stats_dict['quantile_99']:.4f}"],
            ['5th Percentile (%)', f"{stats_dict['quantile_05']:.4f}"],
            ['1st Percentile (%)', f"{stats_dict['quantile_01']:.4f}"],
            ['Skewness', f"{stats_dict['skewness']:.4f}"],
            ['Kurtosis', f"{stats_dict['kurtosis']:.4f}"]
        ]
        stats_df = pd.DataFrame(stats_data, columns=['Metric', 'Value'])

        return stats_df

    def plot_pattern_analysis(
        self,
        pattern_results: Dict,
        thresholds: Dict,
        symbol: Optional[str] = None
    ) -> go.Figure:
        """
        Визуализирует анализ паттернов: распределения признаков перед ростами и падениями.

        Args:
            pattern_results: Результаты анализа паттернов
            thresholds: Пороги для каждого признака
            symbol: Символ акции (опционально)

        Returns:
            Plotly Figure
        """
        try:
            rise_patterns = pattern_results['rise_patterns_df']
            fall_patterns = pattern_results['fall_patterns_df']
            common_patterns = pattern_results.get('common_patterns', {})

            title_suffix = f" - {symbol}" if symbol else ""

            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Изменение объёма перед движением',
                    'Волатильность перед движением',
                    'RSI перед движением',
                    'Ценовой тренд',
                    'Распределение движений',
                    'Сравнение признаков (квантили)'
                ),
                vertical_spacing=0.12,
                horizontal_spacing=0.15
            )

            # 1. Изменение объёма
            if 'volume_change_5d' in rise_patterns.columns and 'volume_change_5d' in fall_patterns.columns:
                fig.add_trace(
                    go.Box(
                        y=rise_patterns['volume_change_5d'],
                        name='Рост',
                        marker_color=self.color_palette['rise'],
                        boxmean='sd'
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Box(
                        y=fall_patterns['volume_change_5d'],
                        name='Падение',
                        marker_color=self.color_palette['fall'],
                        boxmean='sd'
                    ),
                    row=1, col=1
                )

            # 2. Волатильность
            if 'volatility_5d' in rise_patterns.columns and 'volatility_5d' in fall_patterns.columns:
                fig.add_trace(
                    go.Box(
                        y=rise_patterns['volatility_5d'],
                        name='Рост',
                        marker_color=self.color_palette['rise'],
                        boxmean='sd',
                        showlegend=False
                    ),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Box(
                        y=fall_patterns['volatility_5d'],
                        name='Падение',
                        marker_color=self.color_palette['fall'],
                        boxmean='sd',
                        showlegend=False
                    ),
                    row=1, col=2
                )

            # 3. RSI
            if 'rsi_final' in rise_patterns.columns and 'rsi_final' in fall_patterns.columns:
                fig.add_trace(
                    go.Histogram(
                        x=rise_patterns['rsi_final'],
                        name='Рост',
                        marker_color=self.color_palette['rise'],
                        opacity=0.6,
                        nbinsx=20
                    ),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Histogram(
                        x=fall_patterns['rsi_final'],
                        name='Падение',
                        marker_color=self.color_palette['fall'],
                        opacity=0.6,
                        nbinsx=20
                    ),
                    row=2, col=1
                )

            # 4. Ценовой тренд
            if 'price_trend' in rise_patterns.columns and 'price_trend' in fall_patterns.columns:
                fig.add_trace(
                    go.Box(
                        y=rise_patterns['price_trend'],
                        name='Рост',
                        marker_color=self.color_palette['rise'],
                        boxmean='sd',
                        showlegend=False
                    ),
                    row=2, col=2
                )
                fig.add_trace(
                    go.Box(
                        y=fall_patterns['price_trend'],
                        name='Падение',
                        marker_color=self.color_palette['fall'],
                        boxmean='sd',
                        showlegend=False
                    ),
                    row=2, col=2
                )

            # 5. Распределение движений
            fig.add_trace(
                go.Histogram(
                    x=rise_patterns['movement_pct'] if 'movement_pct' in rise_patterns.columns else [],
                    name='Рост',
                    marker_color=self.color_palette['rise'],
                    opacity=0.6,
                    nbinsx=20
                ),
                row=3, col=1
            )
            fig.add_trace(
                go.Histogram(
                    x=fall_patterns['movement_pct'] if 'movement_pct' in fall_patterns.columns else [],
                    name='Падение',
                    marker_color=self.color_palette['fall'],
                    opacity=0.6,
                    nbinsx=20
                ),
                row=3, col=1
            )

            # 6. Сравнение квантилей признаков
            if thresholds and 'anomaly_thresholds' in thresholds:
                features = list(thresholds['anomaly_thresholds'].keys())[:5]  # Топ-5 признаков
                rise_q99 = [thresholds['anomaly_thresholds'][f]['rise_extreme_high'] for f in features]
                fall_q99 = [thresholds['anomaly_thresholds'][f]['fall_extreme_high'] for f in features]

                fig.add_trace(
                    go.Bar(
                        x=features,
                        y=rise_q99,
                        name='Рост (99%)',
                        marker_color=self.color_palette['rise']
                    ),
                    row=3, col=2
                )
                fig.add_trace(
                    go.Bar(
                        x=features,
                        y=fall_q99,
                        name='Падение (99%)',
                        marker_color=self.color_palette['fall']
                    ),
                    row=3, col=2
                )

            fig.update_layout(
                title=f'Анализ паттернов перед крупными движениями{title_suffix}',
                height=1000,
                showlegend=True
            )

            fig.update_yaxes(title_text="Изменение объёма (%)", row=1, col=1)
            fig.update_yaxes(title_text="Волатильность", row=1, col=2)
            fig.update_xaxes(title_text="RSI", row=2, col=1)
            fig.update_yaxes(title_text="Частота", row=2, col=1)
            fig.update_yaxes(title_text="Ценовой тренд", row=2, col=2)
            fig.update_xaxes(title_text="Движение (%)", row=3, col=1)
            fig.update_yaxes(title_text="Частота", row=3, col=1)
            fig.update_xaxes(title_text="Признак", row=3, col=2)
            fig.update_yaxes(title_text="Значение (99%)", row=3, col=2)

            return fig

        except Exception as e:
            st.error(f"Ошибка визуализации паттернов: {str(e)}")
            return go.Figure()

    def plot_quarterly_forecast(
        self,
        quarterly_summary: List[Dict],
        adjusted_forecast: pd.DataFrame,
        similar_patterns: pd.DataFrame,
        symbol: str
    ) -> go.Figure:
        """
        Визуализирует квартальный прогноз с учётом паттернов.

        Args:
            quarterly_summary: Сводка по кварталам
            adjusted_forecast: Скорректированный прогноз
            similar_patterns: Похожие исторические паттерны
            symbol: Символ акции

        Returns:
            Plotly Figure
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    f'Прогноз цены по кварталам - {symbol}',
                    'Ожидаемая доходность по кварталам',
                    'Диапазон уверенности',
                    'Похожие исторические события'
                ),
                specs=[
                    [{"secondary_y": False}, {"type": "bar"}],
                    [{"secondary_y": False}, {"type": "bar"}]
                ],
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )

            # 1. Прогноз цены
            if adjusted_forecast is not None and not adjusted_forecast.empty:
                fig.add_trace(
                    go.Scatter(
                        x=adjusted_forecast['Date'],
                        y=adjusted_forecast['Predicted_Price'],
                        mode='lines',
                        name='Прогноз',
                        line=dict(color=self.color_palette['primary'], width=2)
                    ),
                    row=1, col=1
                )

                # Доверительный интервал
                fig.add_trace(
                    go.Scatter(
                        x=adjusted_forecast['Date'],
                        y=adjusted_forecast['Confidence_High'],
                        mode='lines',
                        name='Верхняя граница',
                        line=dict(color=self.color_palette['rise'], width=0.5, dash='dash'),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=adjusted_forecast['Date'],
                        y=adjusted_forecast['Confidence_Low'],
                        mode='lines',
                        name='Нижняя граница',
                        line=dict(color=self.color_palette['fall'], width=0.5, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(31, 119, 180, 0.2)',
                        showlegend=False
                    ),
                    row=1, col=1
                )

            # 2. Ожидаемая доходность по кварталам
            if quarterly_summary:
                quarters = [q['quarter'] for q in quarterly_summary]
                returns = [q['expected_return_pct'] for q in quarterly_summary]
                colors = [self.color_palette['rise'] if r > 0 else self.color_palette['fall'] for r in returns]

                fig.add_trace(
                    go.Bar(
                        x=[f"Q{q}" for q in quarters],
                        y=returns,
                        name='Доходность',
                        marker_color=colors,
                        text=[f"{r:.2f}%" for r in returns],
                        textposition='auto'
                    ),
                    row=1, col=2
                )

            # 3. Диапазон уверенности по кварталам
            if quarterly_summary:
                quarters = [q['quarter'] for q in quarterly_summary]
                conf_ranges = [(q['confidence_range'][1] - q['confidence_range'][0]) / 2 for q in quarterly_summary]

                fig.add_trace(
                    go.Bar(
                        x=[f"Q{q}" for q in quarters],
                        y=conf_ranges,
                        name='Диапазон',
                        marker_color=self.color_palette['warning'],
                        text=[f"±{r:.2f}" for r in conf_ranges],
                        textposition='auto'
                    ),
                    row=2, col=1
                )

            # 4. Похожие исторические события
            if not similar_patterns.empty and 'movement_pct' in similar_patterns.columns:
                movements = similar_patterns['movement_pct'].head(10)
                colors = [self.color_palette['rise'] if m > 0 else self.color_palette['fall'] for m in movements]

                fig.add_trace(
                    go.Bar(
                        x=list(range(1, len(movements) + 1)),
                        y=movements,
                        name='Исторические исходы',
                        marker_color=colors,
                        text=[f"{m:.2f}%" for m in movements],
                        textposition='auto'
                    ),
                    row=2, col=2
                )

            fig.update_layout(
                title=f'Квартальный прогноз - {symbol}',
                height=900,
                showlegend=True
            )

            fig.update_xaxes(title_text="Дата", row=1, col=1)
            fig.update_yaxes(title_text="Цена", row=1, col=1)
            fig.update_xaxes(title_text="Квартал", row=1, col=2)
            fig.update_yaxes(title_text="Доходность (%)", row=1, col=2)
            fig.update_xaxes(title_text="Квартал", row=2, col=1)
            fig.update_yaxes(title_text="Диапазон уверенности", row=2, col=1)
            fig.update_xaxes(title_text="Топ события", row=2, col=2)
            fig.update_yaxes(title_text="Движение (%)", row=2, col=2)

            return fig

        except Exception as e:
            st.error(f"Ошибка визуализации квартального прогноза: {str(e)}")
            return go.Figure()

    def plot_current_signals(
        self,
        current_signals: Dict,
        thresholds: Dict,
        symbol: str
    ) -> go.Figure:
        """
        Визуализирует текущие обнаруженные сигналы относительно порогов.

        Args:
            current_signals: Текущие сигналы
            thresholds: Пороги
            symbol: Символ акции

        Returns:
            Plotly Figure
        """
        try:
            fig = go.Figure()

            if 'current_features' not in current_signals or not thresholds:
                st.warning("Недостаточно данных для визуализации сигналов")
                return fig

            current_features = current_signals['current_features']

            # Выбираем топ-10 признаков
            features_to_plot = [
                'volume_change_5d', 'volatility_5d', 'rsi_final',
                'price_trend', 'price_range_pct', 'volume_change_3d',
                'volatility_3d', 'rsi_mean', 'volume_change_1d', 'volatility_1d'
            ]

            features = []
            current_values = []
            rise_q99 = []
            fall_q99 = []

            for feat in features_to_plot:
                if feat in current_features and feat in thresholds.get('anomaly_thresholds', {}):
                    features.append(feat)
                    current_values.append(current_features[feat])
                    rise_q99.append(thresholds['anomaly_thresholds'][feat]['rise_extreme_high'])
                    fall_q99.append(thresholds['anomaly_thresholds'][feat]['fall_extreme_high'])

            # Текущие значения
            fig.add_trace(
                go.Bar(
                    x=features,
                    y=current_values,
                    name='Текущее значение',
                    marker_color=self.color_palette['info']
                )
            )

            # Пороги ростов
            fig.add_trace(
                go.Scatter(
                    x=features,
                    y=rise_q99,
                    mode='markers+lines',
                    name='Порог роста (99%)',
                    marker=dict(size=10, symbol='diamond', color=self.color_palette['rise']),
                    line=dict(dash='dash')
                )
            )

            # Пороги падений
            fig.add_trace(
                go.Scatter(
                    x=features,
                    y=fall_q99,
                    mode='markers+lines',
                    name='Порог падения (99%)',
                    marker=dict(size=10, symbol='diamond', color=self.color_palette['fall']),
                    line=dict(dash='dash')
                )
            )

            fig.update_layout(
                title=f'Текущие сигналы vs пороги топ-20 событий - {symbol}',
                xaxis_title='Признак',
                yaxis_title='Значение',
                height=600,
                showlegend=True
            )

            return fig

        except Exception as e:
            st.error(f"Ошибка визуализации текущих сигналов: {str(e)}")
            return go.Figure()
