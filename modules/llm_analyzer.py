import json
import os
from openai import OpenAI
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional

class LLMAnalyzer:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model_name: str | None = None):
        self.openai_api_key = (api_key or os.getenv("OPENAI_API_KEY"))
        self.openai_base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        # By default we keep using the newest OpenAI model "gpt-5" (released August 7, 2025)
        # unless the пользователь явно указал другое имя модели в настройках
        self.model = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-5")

        if not self.openai_api_key:
            st.error("Ключ OpenAI API не найден. Укажите его в настройках приложения.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url)
    
    def analyze_news_sentiment(self, text, stock_symbol=None):
        """Analyze sentiment of news text using LLM"""
        if not self.client:
            return None
        
        try:
            context = f" related to {stock_symbol}" if stock_symbol else ""
            
            prompt = f"""
            Analyze the sentiment of this financial news text{context}.
            
            Text: {text}
            
            Provide analysis in the following JSON format:
            {{
                "sentiment": "positive|negative|neutral",
                "confidence": 0.0-1.0,
                "stock_impact": "bullish|bearish|neutral",
                "impact_strength": 1-10,
                "key_factors": ["factor1", "factor2", "factor3"],
                "summary": "brief explanation of sentiment and potential stock impact"
            }}
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial sentiment analysis expert. Analyze news sentiment and its potential impact on stock prices."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if content:
                result = json.loads(content)
                return result
            else:
                return None
            
        except Exception as e:
            st.error(f"Ошибка анализа настроений: {str(e)}")
            return None
    
    def analyze_batch_messages(self, messages_df, stock_symbols):
        """Analyze sentiment for batch of messages"""
        if messages_df.empty or not self.client:
            return messages_df
        
        results = []
        
        progress_bar = st.progress(0)
        total_messages = len(messages_df)
        
        for idx, row in messages_df.iterrows():
            # Find relevant stock symbol for this message
            relevant_symbol = None
            for symbol in stock_symbols:
                if symbol.lower() in row['text'].lower():
                    relevant_symbol = symbol
                    break
            
            # Analyze sentiment
            sentiment_result = self.analyze_news_sentiment(row['text'], relevant_symbol)
            
            if sentiment_result:
                result = {
                    'message_id': row.get('message_id', idx),
                    'date': row['date'],
                    'channel': row['channel'],
                    'text': row['text'],
                    'relevant_symbol': relevant_symbol,
                    'sentiment': sentiment_result.get('sentiment', 'neutral'),
                    'confidence': sentiment_result.get('confidence', 0.5),
                    'stock_impact': sentiment_result.get('stock_impact', 'neutral'),
                    'impact_strength': sentiment_result.get('impact_strength', 5),
                    'key_factors': sentiment_result.get('key_factors', []),
                    'summary': sentiment_result.get('summary', ''),
                    'views': row.get('views', 0),
                    'relevance_score': row.get('relevance_score', 0)
                }
                results.append(result)
            
            # Update progress
            progress_bar.progress((idx + 1) / total_messages)
        
        progress_bar.empty()
        
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame()
    
    def correlate_news_with_stock_movements(self, news_df, stock_data, symbol):
        """Correlate news sentiment with actual stock movements"""
        if news_df.empty:
            return None
        
        try:
            # Get stock price changes
            stock_changes = stock_data.calculate_price_changes(symbol)
            if stock_changes is None:
                return None
            
            correlations = []
            
            # Group news by date
            news_df['date_only'] = news_df['date'].dt.date
            daily_sentiment = news_df.groupby('date_only').agg({
                'sentiment': lambda x: (x == 'positive').sum() - (x == 'negative').sum(),
                'impact_strength': 'mean',
                'confidence': 'mean',
                'views': 'sum',
                'relevance_score': 'mean'
            }).reset_index()
            
            # Match with stock movements
            for _, news_row in daily_sentiment.iterrows():
                news_date = pd.to_datetime(news_row['date_only'])
                
                # Look for stock movement on same day and next few days
                for offset in range(0, 3):  # Check same day and next 2 days
                    target_date = news_date + pd.Timedelta(days=offset)
                    
                    if target_date.date() in stock_changes.index.date:
                        stock_row = stock_changes[stock_changes.index.date == target_date.date()]
                        if not stock_row.empty:
                            correlation = {
                                'news_date': news_row['date_only'],
                                'stock_date': target_date.date(),
                                'days_offset': offset,
                                'sentiment_score': news_row['sentiment'],
                                'impact_strength': news_row['impact_strength'],
                                'confidence': news_row['confidence'],
                                'news_volume': news_row['views'],
                                'relevance_score': news_row['relevance_score'],
                                'stock_change_pct': stock_row.iloc[0]['Daily_Change_Pct'],
                                'stock_volume': stock_row.iloc[0]['Volume'],
                                'stock_price': stock_row.iloc[0]['Close']
                            }
                            correlations.append(correlation)
            
            return pd.DataFrame(correlations) if correlations else None
            
        except Exception as e:
            st.error(f"Ошибка корреляции новостей с движениями акций: {str(e)}")
            return None
    
    def generate_forecast_analysis(self, historical_patterns, news_sentiment, symbol):
        """Generate quarterly forecast using LLM analysis of patterns and sentiment"""
        if not self.client:
            return None
        
        try:
            prompt = f"""
            Based on the following data for stock {symbol}, provide a quarterly forecast analysis:
            
            Historical Patterns:
            - Recent price movements and volatility patterns
            - Statistical indicators (quantiles, trends)
            
            Recent News Sentiment:
            - Overall sentiment trend
            - Key factors affecting the stock
            - News volume and relevance
            
            Provide forecast in JSON format:
            {{
                "quarter_outlook": "bullish|bearish|neutral",
                "confidence_level": 0.0-1.0,
                "target_price_range": {{"min": number, "max": number}},
                "key_risks": ["risk1", "risk2", "risk3"],
                "key_opportunities": ["opp1", "opp2", "opp3"],
                "sentiment_correlation_strength": 0.0-1.0,
                "forecast_summary": "detailed explanation of forecast reasoning",
                "time_horizon": "Q1 2026"
            }}
            
            Historical Patterns: {str(historical_patterns)}
            News Sentiment: {str(news_sentiment)}
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a quantitative analyst specializing in stock forecasting using historical patterns and news sentiment analysis."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if content:
                result = json.loads(content)
                return result
            else:
                return None
            
        except Exception as e:
            st.error(f"Ошибка генерации прогноза: {str(e)}")
            return None

    def analyze_historical_patterns(
        self,
        pattern_results: Dict,
        similar_patterns: pd.DataFrame,
        current_signals: Dict,
        symbol: str
    ) -> Optional[Dict]:
        """
        Анализирует исторические паттерны и объясняет их значение для текущей ситуации.

        Args:
            pattern_results: Результаты анализа паттернов по всем акциям
            similar_patterns: Похожие исторические паттерны (топ-N)
            current_signals: Текущие обнаруженные сигналы
            symbol: Символ акции

        Returns:
            Словарь с анализом паттернов от LLM
        """
        if not self.client:
            return None

        try:
            # Подготавливаем данные для анализа
            common_patterns = pattern_results.get('common_patterns', {})
            total_rise_events = pattern_results.get('total_rise_events', 0)
            total_fall_events = pattern_results.get('total_fall_events', 0)

            # Информация о похожих паттернах
            similar_patterns_summary = ""
            if not similar_patterns.empty:
                avg_movement = similar_patterns['movement_pct'].mean()
                direction = "роста" if avg_movement > 0 else "падения"
                similar_count = len(similar_patterns)

                similar_patterns_summary = f"""
                Найдено {similar_count} похожих исторических событий из топ-20 {direction}.
                Средний исход: {avg_movement:.2f}%
                Диапазон: от {similar_patterns['movement_pct'].min():.2f}% до {similar_patterns['movement_pct'].max():.2f}%
                """

            # Текущие сигналы
            detected_signals = current_signals.get('detected_signals', {})
            signals_summary = f"Обнаружено сигналов: {len(detected_signals)} категорий"

            prompt = f"""
            Проанализируй исторические паттерны перед крупными движениями акций и объясни их значение для {symbol}.

            ИСТОРИЧЕСКИЕ ДАННЫЕ:
            - Проанализировано {total_rise_events} топ-ростов и {total_fall_events} топ-падений
            - Выявлены статистически значимые закономерности перед крупными движениями

            ПОХОЖИЕ ИСТОРИЧЕСКИЕ СОБЫТИЯ:
            {similar_patterns_summary}

            ТЕКУЩИЕ СИГНАЛЫ:
            {signals_summary}
            Детали: {json.dumps(detected_signals, ensure_ascii=False, indent=2)}

            ОБЩИЕ ЗАКОНОМЕРНОСТИ:
            {json.dumps(common_patterns, ensure_ascii=False, indent=2)[:1000]}

            Предоставь анализ в JSON формате:
            {{
                "pattern_interpretation": "подробное объяснение того, что означают обнаруженные паттерны",
                "historical_context": "как часто подобные ситуации приводили к росту/падению в прошлом",
                "signal_strength": "слабый|средний|сильный|экстремальный",
                "risk_level": "низкий|средний|высокий|критический",
                "key_indicators": ["индикатор1", "индикатор2", "индикатор3"],
                "similar_events_analysis": "анализ похожих исторических событий и их исходов",
                "recommended_action": "держать|покупать|продавать|наблюдать",
                "confidence": 0.0-1.0,
                "detailed_reasoning": "подробное объяснение логики рекомендации",
                "time_horizon": "краткосрочный (недели)|среднесрочный (месяцы)|долгосрочный (кварталы)",
                "critical_thresholds": "какие пороговые значения были превышены или близки к превышению"
            }}
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """Ты - эксперт по техническому и квантовому анализу акций.
                        Твоя специализация - выявление паттернов перед крупными движениями и объяснение их значения
                        на основе статистических данных из топ-20 исторических событий."""
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            if content:
                result = json.loads(content)
                result['analysis_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                result['symbol'] = symbol
                return result
            else:
                return None

        except Exception as e:
            st.error(f"Ошибка анализа исторических паттернов: {str(e)}")
            return None

    def explain_quarterly_forecast(
        self,
        quarterly_summary: List[Dict],
        pattern_analysis: Optional[Dict],
        portfolio_analysis: Dict,
        symbol: str
    ) -> Optional[Dict]:
        """
        Объясняет квартальный прогноз простым языком с учётом паттернов.

        Args:
            quarterly_summary: Сводка по кварталам
            pattern_analysis: Анализ паттернов от analyze_historical_patterns
            portfolio_analysis: Общий анализ портфеля
            symbol: Символ акции

        Returns:
            Словарь с объяснением прогноза
        """
        if not self.client:
            return None

        try:
            # Формируем краткую сводку по кварталам
            quarters_info = []
            for q in quarterly_summary:
                quarters_info.append(f"""
                Квартал {q['quarter']}: {q['start_date']} - {q['end_date']}
                Ожидаемая доходность: {q['expected_return_pct']:.2f}%
                Волатильность: {q['volatility']:.2f}
                """)

            pattern_context = ""
            if pattern_analysis:
                pattern_context = f"""
                АНАЛИЗ ПАТТЕРНОВ:
                - Уровень сигнала: {pattern_analysis.get('signal_strength', 'неизвестно')}
                - Уровень риска: {pattern_analysis.get('risk_level', 'неизвестно')}
                - Рекомендация: {pattern_analysis.get('recommended_action', 'наблюдать')}
                - Обоснование: {pattern_analysis.get('detailed_reasoning', '')}
                """

            prompt = f"""
            Объясни квартальный прогноз для акции {symbol} простым и понятным языком.

            ПРОГНОЗ ПО КВАРТАЛАМ:
            {''.join(quarters_info)}

            {pattern_context}

            КОНТЕКСТ ПОРТФЕЛЯ:
            - Общий настрой портфеля: {portfolio_analysis.get('portfolio_sentiment', 'неопределённый')}
            - Уровень риска: {portfolio_analysis.get('risk_level', 'неопределённый')}

            Предоставь объяснение в JSON формате:
            {{
                "executive_summary": "краткое резюме прогноза на 2-3 предложения",
                "quarterly_breakdown": [
                    {{
                        "quarter": 1,
                        "outlook": "что ожидать в этом квартале",
                        "key_drivers": ["драйвер1", "драйвер2"],
                        "risk_factors": ["риск1", "риск2"]
                    }}
                ],
                "overall_recommendation": "общая рекомендация для инвестора",
                "confidence_explanation": "почему мы уверены/не уверены в этом прогнозе",
                "alternative_scenarios": [
                    {{
                        "scenario": "оптимистичный|пессимистичный",
                        "probability": 0.0-1.0,
                        "outcome": "что произойдёт в этом сценарии"
                    }}
                ],
                "action_items": ["что делать инвестору сейчас", "на что обратить внимание"],
                "plain_language_summary": "объяснение прогноза без жаргона для непрофессионала"
            }}
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """Ты - финансовый советник, который объясняет сложные прогнозы простым языком.
                        Твоя задача - сделать квартальный прогноз понятным для инвестора, используя анализ паттернов."""
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            if content:
                result = json.loads(content)
                result['generated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                result['symbol'] = symbol
                return result
            else:
                return None

        except Exception as e:
            st.error(f"Ошибка объяснения квартального прогноза: {str(e)}")
            return None
