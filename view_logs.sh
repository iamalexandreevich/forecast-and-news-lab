#!/bin/bash
# Скрипт для просмотра логов Streamlit в реальном времени

echo "📋 Логи Streamlit (нажмите Ctrl+C для выхода)"
echo "=============================================="
echo ""

# Запуск Streamlit с выводом логов
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
streamlit run app.py 2>&1 | grep -E '\[AUTH\]|\[CLIENT\]|ERROR|WARNING|Traceback' --line-buffered
