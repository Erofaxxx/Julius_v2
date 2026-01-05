#!/bin/bash

# Скрипт быстрой установки AI CSV Analysis Agent

echo "╔════════════════════════════════════════════════════════════╗"
echo "║      AI CSV Analysis Agent - Установка                     ║"
echo "║      Powered by Claude Sonnet 4.5                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Проверка Python
echo "🔍 Проверка Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 не найден. Установите Python 3.8 или выше."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Найден Python $PYTHON_VERSION"

# Создание виртуального окружения
echo ""
echo "📦 Создание виртуального окружения..."
if [ -d "venv" ]; then
    echo "⚠️  venv уже существует, пропускаем..."
else
    python3 -m venv venv
    echo "✓ Виртуальное окружение создано"
fi

# Активация виртуального окружения
echo ""
echo "🔧 Активация виртуального окружения..."
source venv/bin/activate

# Установка зависимостей
echo ""
echo "📥 Установка зависимостей..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Зависимости установлены"

# Настройка .env
echo ""
echo "🔑 Настройка API ключа..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✓ Файл .env создан"
    echo ""
    echo "⚠️  ВАЖНО: Отредактируйте файл .env и добавьте ваш OpenRouter API ключ:"
    echo "   nano .env"
    echo ""
    echo "   Получите ключ на: https://openrouter.ai/keys"
else
    echo "✓ Файл .env уже существует"
fi

# Проверка CSV файлов
echo ""
echo "📊 Проверка CSV файлов..."
CSV_COUNT=$(ls -1 *.csv 2>/dev/null | wc -l)
if [ $CSV_COUNT -gt 0 ]; then
    echo "✓ Найдено $CSV_COUNT CSV файл(ов)"
else
    echo "⚠️  CSV файлы не найдены"
    echo "   Используйте example_sales.csv для тестирования"
fi

# Финальные инструкции
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                 Установка завершена!                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Следующие шаги:"
echo ""
echo "1. Добавьте API ключ в .env:"
echo "   nano .env"
echo ""
echo "2. Активируйте виртуальное окружение:"
echo "   source venv/bin/activate"
echo ""
echo "3. Запустите агента:"
echo "   python csv_agent.py"
echo ""
echo "Или запустите примеры:"
echo "   python example_usage.py"
echo ""
echo "Документация:"
echo "   - README.md - полная документация"
echo "   - QUICKSTART_RU.md - быстрый старт на русском"
echo ""
echo "Удачи! 🚀"
