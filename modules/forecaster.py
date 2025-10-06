import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class StockForecaster:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.trained = False
        self.pattern_analyzer = None
    
    def prepare_features(self, stock_data, sentiment_data=None, pattern_features=None):
        """Prepare features for forecasting model"""
        try:
            # Calculate technical indicators
            data = stock_data.copy()

            # Price-based features
            data['SMA_5'] = data['Close'].rolling(window=5).mean()
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['RSI'] = self.calculate_rsi(data['Close'])
            data['Price_Change_1d'] = data['Close'].pct_change(1)
            data['Price_Change_5d'] = data['Close'].pct_change(5)
            data['Price_Change_20d'] = data['Close'].pct_change(20)

            # Volume features
            data['Volume_SMA_5'] = data['Volume'].rolling(window=5).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_5']

            # Volatility features
            data['Volatility_5d'] = data['Close'].rolling(window=5).std()
            data['Volatility_20d'] = data['Close'].rolling(window=20).std()

            # High-Low features
            data['High_Low_Pct'] = (data['High'] - data['Low']) / data['Close']
            data['Close_High_Pct'] = (data['Close'] - data['High']) / data['High']

            # Sentiment features (if available)
            if sentiment_data is not None and not sentiment_data.empty:
                # Aggregate daily sentiment
                sentiment_daily = self.aggregate_daily_sentiment(sentiment_data)
                data = self.merge_sentiment_data(data, sentiment_daily)

            # Pattern-based features (if available)
            if pattern_features is not None:
                data = self._merge_pattern_features(data, pattern_features)

            # Target variable (next day's return)
            data['Target'] = data['Close'].shift(-1) / data['Close'] - 1

            return data

        except Exception as e:
            st.error(f"Error preparing features: {str(e)}")
            return None
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def aggregate_daily_sentiment(self, sentiment_data):
        """Aggregate sentiment data by day"""
        if sentiment_data.empty:
            return pd.DataFrame()
        
        # Convert sentiment to numerical scores
        sentiment_scores = {
            'positive': 1,
            'neutral': 0,
            'negative': -1
        }
        
        sentiment_data['sentiment_score'] = sentiment_data['sentiment'].map(sentiment_scores)
        
        # Group by date
        daily_sentiment = sentiment_data.groupby(sentiment_data['date'].dt.date).agg({
            'sentiment_score': ['mean', 'sum', 'count'],
            'confidence': 'mean',
            'impact_strength': 'mean',
            'views': 'sum',
            'relevance_score': 'mean'
        }).reset_index()
        
        # Flatten column names
        daily_sentiment.columns = [
            'date', 'sentiment_mean', 'sentiment_sum', 'sentiment_count',
            'confidence_mean', 'impact_strength_mean', 'total_views', 'relevance_mean'
        ]
        
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
        
        return daily_sentiment
    
    def merge_sentiment_data(self, stock_data, sentiment_data):
        """Merge sentiment data with stock data"""
        if sentiment_data.empty:
            # Add zero sentiment features if no sentiment data
            stock_data['sentiment_mean'] = 0
            stock_data['sentiment_sum'] = 0
            stock_data['sentiment_count'] = 0
            stock_data['confidence_mean'] = 0.5
            stock_data['impact_strength_mean'] = 5
            stock_data['total_views'] = 0
            stock_data['relevance_mean'] = 0
            return stock_data
        
        # Merge on date
        stock_data.reset_index(inplace=True)
        stock_data['date'] = pd.to_datetime(stock_data['Date'])
        
        merged_data = pd.merge(
            stock_data, sentiment_data, 
            left_on=stock_data['date'].dt.date,
            right_on=sentiment_data['date'].dt.date,
            how='left'
        )
        
        # Fill missing sentiment data with neutral values
        sentiment_columns = ['sentiment_mean', 'sentiment_sum', 'sentiment_count',
                           'confidence_mean', 'impact_strength_mean', 'total_views', 'relevance_mean']
        
        for col in sentiment_columns:
            if col in merged_data.columns:
                if col == 'confidence_mean':
                    merged_data[col].fillna(0.5, inplace=True)
                elif col == 'impact_strength_mean':
                    merged_data[col].fillna(5, inplace=True)
                else:
                    merged_data[col].fillna(0, inplace=True)
        
        return merged_data
    
    def _merge_pattern_features(self, stock_data, pattern_features):
        """Объединяет признаки паттернов с данными акций"""
        try:
            # Добавляем скалярные признаки паттернов
            if isinstance(pattern_features, dict):
                # Признаки на основе похожих исторических паттернов
                stock_data['pattern_similarity_score'] = pattern_features.get('avg_similarity', 0)
                stock_data['pattern_avg_movement'] = pattern_features.get('avg_movement', 0)
                stock_data['pattern_signal_strength'] = pattern_features.get('signal_strength', 0)
                stock_data['pattern_rise_probability'] = pattern_features.get('rise_probability', 0.5)
                stock_data['pattern_extreme_event_risk'] = pattern_features.get('extreme_event_risk', 0)

            return stock_data

        except Exception as e:
            st.warning(f"Ошибка объединения признаков паттернов: {str(e)}")
            return stock_data

    def train_model(self, prepared_data):
        """Train the forecasting model"""
        try:
            if prepared_data is None or prepared_data.empty:
                return False

            # Define feature columns (базовые + паттерны)
            base_features = [
                'SMA_5', 'SMA_20', 'RSI', 'Price_Change_1d', 'Price_Change_5d', 'Price_Change_20d',
                'Volume_Ratio', 'Volatility_5d', 'Volatility_20d', 'High_Low_Pct', 'Close_High_Pct',
                'sentiment_mean', 'sentiment_sum', 'sentiment_count', 'confidence_mean',
                'impact_strength_mean', 'total_views', 'relevance_mean'
            ]

            pattern_features = [
                'pattern_similarity_score', 'pattern_avg_movement', 'pattern_signal_strength',
                'pattern_rise_probability', 'pattern_extreme_event_risk'
            ]

            # Используем только те признаки, которые есть в данных
            self.feature_columns = [col for col in base_features + pattern_features if col in prepared_data.columns]
            
            # Remove rows with missing data
            clean_data = prepared_data[self.feature_columns + ['Target']].dropna()
            
            if len(clean_data) < 50:  # Minimum data requirement
                st.error("Insufficient data for training (need at least 50 data points)")
                return False
            
            X = clean_data[self.feature_columns]
            y = clean_data['Target']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Random Forest model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            
            self.model.fit(X_scaled, y)
            
            # Calculate training metrics
            y_pred = self.model.predict(X_scaled)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            st.success(f"Model trained successfully - R² Score: {r2:.4f}, MSE: {mse:.6f}")
            
            self.trained = True
            return True
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return False
    
    def generate_forecast(self, recent_data, days_ahead=90):
        """Generate forecast for specified number of days"""
        if not self.trained or self.model is None:
            st.error("Model not trained. Please train the model first.")
            return None
        
        try:
            forecasts = []
            current_data = recent_data.copy()
            
            for day in range(days_ahead):
                # Prepare features for current state
                features = current_data[self.feature_columns].iloc[-1:].values
                features_scaled = self.scaler.transform(features)
                
                # Make prediction
                predicted_return = self.model.predict(features_scaled)[0]
                
                # Calculate predicted price
                last_price = current_data['Close'].iloc[-1]
                predicted_price = last_price * (1 + predicted_return)
                
                # Create forecast entry
                forecast_date = current_data.index[-1] + timedelta(days=1)
                
                forecast_entry = {
                    'Date': forecast_date,
                    'Predicted_Price': predicted_price,
                    'Predicted_Return': predicted_return,
                    'Confidence_Low': predicted_price * 0.95,  # Simple confidence interval
                    'Confidence_High': predicted_price * 1.05
                }
                
                forecasts.append(forecast_entry)
                
                # Update current_data for next iteration (simplified approach)
                # In practice, you'd want more sophisticated feature updating
                new_row = current_data.iloc[-1:].copy()
                new_row.index = [forecast_date]
                new_row['Close'] = predicted_price
                current_data = pd.concat([current_data, new_row])
                
                # Recalculate some features
                current_data['SMA_5'] = current_data['Close'].rolling(window=5).mean()
                current_data['SMA_20'] = current_data['Close'].rolling(window=20).mean()
                current_data['Price_Change_1d'] = current_data['Close'].pct_change(1)
                current_data = current_data.fillna(method='ffill')
            
            return pd.DataFrame(forecasts)
            
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")
            return None
    
    def get_feature_importance(self):
        """Get feature importance from trained model"""
        if not self.trained or self.model is None:
            return None

        try:
            importance_df = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)

            return importance_df

        except Exception as e:
            st.error(f"Error getting feature importance: {str(e)}")
            return None

    def generate_quarterly_forecast(
        self,
        stock_analyzer,
        symbols: List[str],
        pattern_results: Dict,
        thresholds: Dict,
        quarters: int = 4
    ) -> Dict:
        """
        Генерирует квартальный прогноз на основе обнаруженных паттернов.

        Args:
            stock_analyzer: Экземпляр StockAnalyzer с загруженными данными
            symbols: Список символов акций для прогноза
            pattern_results: Результаты анализа паттернов (из find_common_patterns_across_stocks)
            thresholds: Пороги для обнаружения сигналов
            quarters: Количество кварталов для прогноза

        Returns:
            Словарь с прогнозами по каждой акции и общим анализом
        """
        try:
            # Импортируем PatternAnalyzer при первом использовании
            if self.pattern_analyzer is None:
                from modules.pattern_analyzer import PatternAnalyzer
                self.pattern_analyzer = PatternAnalyzer()

            quarterly_forecasts = {}
            days_per_quarter = 90
            total_days = quarters * days_per_quarter

            for symbol in symbols:
                if symbol not in stock_analyzer.stock_data:
                    st.warning(f"Нет данных для {symbol}, пропускаем")
                    continue

                stock_data = stock_analyzer.stock_data[symbol]

                # 1. Обнаруживаем текущие сигналы
                current_signals = stock_analyzer.detect_current_anomalies(
                    symbol, thresholds, days_window=5
                )

                if current_signals is None:
                    st.warning(f"Не удалось обнаружить сигналы для {symbol}")
                    continue

                # 2. Находим похожие исторические паттерны
                similar_patterns = self._find_similar_historical_patterns(
                    current_signals,
                    pattern_results,
                    top_n=10
                )

                # 3. Генерируем базовый прогноз ML моделью
                prepared_data = self.prepare_features(stock_data)
                if prepared_data is not None and not prepared_data.empty:
                    ml_forecast = self.generate_forecast(prepared_data, days_ahead=total_days)
                else:
                    ml_forecast = None

                # 4. Корректируем прогноз на основе паттернов
                adjusted_forecast = self._adjust_forecast_with_patterns(
                    ml_forecast,
                    similar_patterns,
                    current_signals,
                    quarters=quarters
                )

                # 5. Формируем квартальную сводку
                quarterly_summary = self._create_quarterly_summary(
                    adjusted_forecast,
                    similar_patterns,
                    current_signals,
                    quarters=quarters
                )

                quarterly_forecasts[symbol] = {
                    'current_signals': current_signals,
                    'similar_patterns': similar_patterns,
                    'ml_forecast': ml_forecast,
                    'adjusted_forecast': adjusted_forecast,
                    'quarterly_summary': quarterly_summary,
                    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

                st.success(f"Квартальный прогноз для {symbol} сгенерирован")

            # Общий анализ по портфелю
            portfolio_analysis = self._analyze_portfolio_forecast(
                quarterly_forecasts,
                pattern_results
            )

            return {
                'forecasts': quarterly_forecasts,
                'portfolio_analysis': portfolio_analysis,
                'quarters': quarters,
                'total_days': total_days,
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            st.error(f"Ошибка генерации квартального прогноза: {str(e)}")
            return None

    def _find_similar_historical_patterns(
        self,
        current_signals: Dict,
        pattern_results: Dict,
        top_n: int = 10
    ) -> pd.DataFrame:
        """Находит похожие исторические паттерны из топ-20 событий"""
        try:
            if 'current_features' not in current_signals:
                return pd.DataFrame()

            current_features = current_signals['current_features']

            # Объединяем паттерны ростов и падений
            rise_patterns = pattern_results['rise_patterns_df']
            fall_patterns = pattern_results['fall_patterns_df']
            all_patterns = pd.concat([rise_patterns, fall_patterns], ignore_index=True)

            # Вычисляем схожесть с каждым историческим паттерном
            similarities = self.pattern_analyzer.calculate_pattern_similarity(
                current_features,
                all_patterns,
                top_n=top_n
            )

            return similarities

        except Exception as e:
            st.warning(f"Ошибка поиска похожих паттернов: {str(e)}")
            return pd.DataFrame()

    def _adjust_forecast_with_patterns(
        self,
        ml_forecast: Optional[pd.DataFrame],
        similar_patterns: pd.DataFrame,
        current_signals: Dict,
        quarters: int = 4
    ) -> pd.DataFrame:
        """Корректирует ML прогноз на основе исторических паттернов"""
        try:
            if ml_forecast is None or ml_forecast.empty:
                # Создаём простой прогноз на основе паттернов
                days_per_quarter = 90
                total_days = quarters * days_per_quarter
                base_price = 100  # Нормализованная цена

                forecast_dates = pd.date_range(
                    start=datetime.now(),
                    periods=total_days,
                    freq='D'
                )

                ml_forecast = pd.DataFrame({
                    'Date': forecast_dates,
                    'Predicted_Price': base_price,
                    'Predicted_Return': 0.0,
                    'Confidence_Low': base_price * 0.95,
                    'Confidence_High': base_price * 1.05
                })

            adjusted_forecast = ml_forecast.copy()

            if similar_patterns.empty:
                return adjusted_forecast

            # Рассчитываем средний исход похожих паттернов
            avg_movement = similar_patterns['movement_pct'].mean()
            pattern_direction = 1 if avg_movement > 0 else -1

            # Корректируем прогноз
            adjustment_factor = 1 + (avg_movement / 100)  # Конвертируем процент в множитель

            # Применяем корректировку с затуханием со временем
            for i in range(len(adjusted_forecast)):
                days_from_now = i + 1
                decay = np.exp(-days_from_now / 60)  # Эффект затухает за ~60 дней

                adjusted_forecast.loc[i, 'Predicted_Price'] *= (1 + (adjustment_factor - 1) * decay)
                adjusted_forecast.loc[i, 'Predicted_Return'] = (
                    adjusted_forecast.loc[i, 'Predicted_Price'] /
                    adjusted_forecast.loc[max(0, i-1), 'Predicted_Price'] - 1
                )

                # Расширяем доверительный интервал на основе волатильности паттернов
                volatility_factor = similar_patterns['movement_pct'].std() / 100
                adjusted_forecast.loc[i, 'Confidence_Low'] *= (1 - volatility_factor * decay)
                adjusted_forecast.loc[i, 'Confidence_High'] *= (1 + volatility_factor * decay)

            return adjusted_forecast

        except Exception as e:
            st.warning(f"Ошибка корректировки прогноза: {str(e)}")
            return ml_forecast

    def _create_quarterly_summary(
        self,
        forecast: pd.DataFrame,
        similar_patterns: pd.DataFrame,
        current_signals: Dict,
        quarters: int = 4
    ) -> List[Dict]:
        """Создаёт сводку по каждому кварталу"""
        try:
            if forecast is None or forecast.empty:
                return []

            days_per_quarter = 90
            quarterly_summaries = []

            for q in range(quarters):
                start_idx = q * days_per_quarter
                end_idx = min((q + 1) * days_per_quarter, len(forecast))

                quarter_data = forecast.iloc[start_idx:end_idx]

                if quarter_data.empty:
                    continue

                start_price = quarter_data['Predicted_Price'].iloc[0]
                end_price = quarter_data['Predicted_Price'].iloc[-1]
                quarter_return = ((end_price - start_price) / start_price) * 100

                summary = {
                    'quarter': q + 1,
                    'start_date': quarter_data['Date'].iloc[0].strftime('%Y-%m-%d'),
                    'end_date': quarter_data['Date'].iloc[-1].strftime('%Y-%m-%d'),
                    'start_price': start_price,
                    'end_price': end_price,
                    'expected_return_pct': quarter_return,
                    'max_price': quarter_data['Predicted_Price'].max(),
                    'min_price': quarter_data['Predicted_Price'].min(),
                    'volatility': quarter_data['Predicted_Price'].std(),
                    'confidence_range': (
                        quarter_data['Confidence_Low'].min(),
                        quarter_data['Confidence_High'].max()
                    )
                }

                # Добавляем информацию о похожих паттернах для этого квартала
                if not similar_patterns.empty:
                    summary['pattern_insights'] = {
                        'similar_count': len(similar_patterns),
                        'avg_historical_movement': similar_patterns['movement_pct'].mean(),
                        'pattern_direction': 'рост' if similar_patterns['movement_pct'].mean() > 0 else 'падение',
                        'confidence': similar_patterns['similarity_score'].mean() if 'similarity_score' in similar_patterns.columns else 0.5
                    }

                quarterly_summaries.append(summary)

            return quarterly_summaries

        except Exception as e:
            st.warning(f"Ошибка создания квартальной сводки: {str(e)}")
            return []

    def _analyze_portfolio_forecast(
        self,
        quarterly_forecasts: Dict,
        pattern_results: Dict
    ) -> Dict:
        """Анализирует прогнозы по всему портфелю"""
        try:
            if not quarterly_forecasts:
                return {}

            total_stocks = len(quarterly_forecasts)
            bullish_count = 0
            bearish_count = 0
            neutral_count = 0

            avg_quarterly_returns = []

            for symbol, forecast_data in quarterly_forecasts.items():
                quarterly_summary = forecast_data.get('quarterly_summary', [])

                if not quarterly_summary:
                    neutral_count += 1
                    continue

                # Анализируем общую тенденцию
                total_return = sum(q['expected_return_pct'] for q in quarterly_summary)

                if total_return > 5:
                    bullish_count += 1
                elif total_return < -5:
                    bearish_count += 1
                else:
                    neutral_count += 1

                avg_quarterly_returns.append(total_return)

            portfolio_analysis = {
                'total_stocks': total_stocks,
                'bullish_stocks': bullish_count,
                'bearish_stocks': bearish_count,
                'neutral_stocks': neutral_count,
                'avg_expected_return': np.mean(avg_quarterly_returns) if avg_quarterly_returns else 0,
                'portfolio_sentiment': self._determine_portfolio_sentiment(
                    bullish_count, bearish_count, neutral_count
                ),
                'risk_level': self._assess_portfolio_risk(quarterly_forecasts, pattern_results)
            }

            return portfolio_analysis

        except Exception as e:
            st.warning(f"Ошибка анализа портфеля: {str(e)}")
            return {}

    def _determine_portfolio_sentiment(
        self,
        bullish: int,
        bearish: int,
        neutral: int
    ) -> str:
        """Определяет общий настрой портфеля"""
        total = bullish + bearish + neutral

        if total == 0:
            return 'неопределённый'

        bullish_pct = bullish / total
        bearish_pct = bearish / total

        if bullish_pct > 0.6:
            return 'сильно бычий'
        elif bullish_pct > 0.4:
            return 'умеренно бычий'
        elif bearish_pct > 0.6:
            return 'сильно медвежий'
        elif bearish_pct > 0.4:
            return 'умеренно медвежий'
        else:
            return 'нейтральный'

    def _assess_portfolio_risk(
        self,
        quarterly_forecasts: Dict,
        pattern_results: Dict
    ) -> str:
        """Оценивает уровень риска портфеля"""
        try:
            if not quarterly_forecasts:
                return 'неопределённый'

            volatilities = []
            signal_strengths = []

            for symbol, forecast_data in quarterly_forecasts.items():
                quarterly_summary = forecast_data.get('quarterly_summary', [])

                for quarter in quarterly_summary:
                    volatilities.append(quarter.get('volatility', 0))

                current_signals = forecast_data.get('current_signals', {})
                detected_signals = current_signals.get('detected_signals', {})

                # Подсчитываем количество сильных сигналов
                strong_signals = sum(
                    1 for signals_list in detected_signals.values()
                    for signal in signals_list
                    if 'extreme' in signal.lower()
                )
                signal_strengths.append(strong_signals)

            avg_volatility = np.mean(volatilities) if volatilities else 0
            avg_signal_strength = np.mean(signal_strengths) if signal_strengths else 0

            # Определяем уровень риска
            if avg_volatility > 5 or avg_signal_strength > 3:
                return 'высокий'
            elif avg_volatility > 2 or avg_signal_strength > 1:
                return 'средний'
            else:
                return 'низкий'

        except Exception as e:
            st.warning(f"Ошибка оценки риска: {str(e)}")
            return 'неопределённый'
