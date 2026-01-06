"""
FastAPI сервер для CSV Analysis Agent
Интеграция с Lovable и другими frontend приложениями
Упрощённая версия - только один файл, только Claude Sonnet 4.5
"""

import os
import os.path
import io
import traceback
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from csv_agent_api import CSVAnalysisAgentAPI, MODEL_NAME

# Загрузка переменных окружения
load_dotenv()

# Инициализация FastAPI
app = FastAPI(
    title="CSV Analysis Agent API",
    description="AI-powered CSV analysis and editing with Claude Sonnet 4.5",
    version="2.0.0"
)

# CORS для работы с Lovable и другими frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API ключ для OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY не найден в переменных окружения")

# Валидация формата API ключа
OPENROUTER_API_KEY = OPENROUTER_API_KEY.strip()

if not OPENROUTER_API_KEY.startswith("sk-"):
    raise ValueError(
        f"OPENROUTER_API_KEY имеет неверный формат. "
        f"Ключ должен начинаться с 'sk-'. "
        f"Текущий ключ начинается с: '{OPENROUTER_API_KEY[:10]}...'"
    )

print(f"✓ OpenRouter API ключ загружен: {OPENROUTER_API_KEY[:10]}...{OPENROUTER_API_KEY[-4:]}")


# Health check endpoint
@app.get("/")
async def root():
    """Проверка работы API"""
    return {
        "status": "online",
        "service": "CSV Analysis Agent API",
        "version": "2.0.0",
        "model": MODEL_NAME,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check для мониторинга"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/info")
async def get_api_info():
    """
    Получить информацию об API

    Returns:
        JSON с информацией о сервисе
    """
    return {
        "success": True,
        "service": "CSV Analysis Agent API",
        "version": "2.0.0",
        "model": MODEL_NAME,
        "features": [
            "Анализ CSV данных",
            "Редактирование данных (добавление/удаление строк и колонок)",
            "Автоматическая очистка данных",
            "Построение графиков и визуализаций",
            "Возврат изменённого CSV файла"
        ],
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/analyze")
async def analyze_csv(
    file: UploadFile = File(..., description="CSV файл для анализа"),
    query: Optional[str] = Form("", description="Запрос пользователя (пустой = автоочистка)"),
    chat_history: Optional[str] = Form(None, description="История чата в JSON формате")
):
    """
    Основной endpoint для анализа и редактирования CSV файла

    Args:
        file: Загруженный CSV файл
        query: Запрос пользователя. Если пустой - выполняется автоматическая очистка данных
        chat_history: JSON строка с историей предыдущих запросов (опционально)

    Returns:
        JSON с результатами анализа, включая:
        - text_output: текстовый результат анализа
        - plots: графики в base64
        - modified_csv: изменённый CSV в base64 (если данные были изменены)
        - was_modified: флаг изменения данных
    """
    agent = None
    try:
        # Проверка формата файла (CSV и Excel)
        allowed_extensions = ['.csv', '.xlsx', '.xls', '.xlsm']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый формат файла. Поддерживаются: {', '.join(allowed_extensions)}"
            )

        # Чтение CSV файла
        file_bytes = await file.read()

        # Парсинг истории если есть
        history = None
        if chat_history:
            import json
            try:
                history = json.loads(chat_history)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Неверный формат chat_history. Требуется валидный JSON."
                )

        # Создание агента
        agent = CSVAnalysisAgentAPI(api_key=OPENROUTER_API_KEY)

        # Загрузка CSV
        try:
            df = agent.load_csv_from_bytes(file_bytes, file.filename)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Ошибка при чтении CSV файла: {str(e)}"
            )

        # Выполнение анализа (или автоочистки если query пустой)
        result = agent.analyze(query, chat_history=history)

        # Добавляем информацию о файле
        result["file_info"] = {
            "filename": file.filename,
            "size_bytes": len(file_bytes),
            "rows": df.shape[0],
            "columns": df.shape[1]
        }
        result["model_info"] = {
            "model_name": MODEL_NAME
        }

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        error_detail = {
            "error": "Внутренняя ошибка сервера",
            "message": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.utcnow().isoformat()
        }
        return JSONResponse(
            status_code=500,
            content=error_detail
        )
    finally:
        # Очистка памяти после каждого запроса
        if agent is not None:
            agent.cleanup()
            del agent


@app.post("/api/auto-clean")
async def auto_clean_csv(
    file: UploadFile = File(..., description="CSV файл для автоматической очистки")
):
    """
    Endpoint для автоматической очистки CSV файла
    Эквивалентен вызову /api/analyze с пустым query

    Args:
        file: Загруженный CSV файл

    Returns:
        JSON с результатами очистки и изменённым CSV
    """
    return await analyze_csv(file=file, query="", chat_history=None)


@app.post("/api/schema")
async def get_csv_schema(
    file: UploadFile = File(..., description="CSV файл")
):
    """
    Получить информацию о структуре CSV файла

    Args:
        file: Загруженный CSV файл

    Returns:
        JSON со схемой данных (колонки, типы, статистика)
    """
    agent = None
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="Неподдерживаемый формат файла. Требуется CSV файл."
            )

        file_bytes = await file.read()

        # Создание агента
        agent = CSVAnalysisAgentAPI(api_key=OPENROUTER_API_KEY)

        # Загрузка CSV
        try:
            df = agent.load_csv_from_bytes(file_bytes, file.filename)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Ошибка при чтении CSV файла: {str(e)}"
            )

        # Получение схемы
        schema_info = agent.get_schema_info()

        # Добавляем имя файла
        schema_info["filename"] = file.filename

        return JSONResponse(content=schema_info)

    except HTTPException:
        raise
    except Exception as e:
        error_detail = {
            "error": "Внутренняя ошибка сервера",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
        return JSONResponse(
            status_code=500,
            content=error_detail
        )
    finally:
        # Очистка памяти
        if agent is not None:
            agent.cleanup()
            del agent


@app.post("/api/quick-analyze")
async def quick_analyze(
    file: UploadFile = File(...),
    query: str = Form(...)
):
    """
    Упрощенный endpoint без истории (для быстрых запросов)

    Args:
        file: CSV файл
        query: Запрос пользователя

    Returns:
        Результаты анализа
    """
    return await analyze_csv(file=file, query=query, chat_history=None)


# Запуск сервера
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"""
╔════════════════════════════════════════════════════════════╗
║         CSV Analysis Agent API Server v2.0                 ║
║         Powered by {MODEL_NAME}                       ║
╚════════════════════════════════════════════════════════════╝

Новые возможности:
✓ Редактирование данных (добавление/удаление строк и колонок)
✓ Автоматическая очистка данных при загрузке без запроса
✓ Возврат изменённого CSV файла в base64

Server starting...
- Host: {host}
- Port: {port}
- Docs: http://{host}:{port}/docs
- Health: http://{host}:{port}/health

Ready to accept requests!
    """)

    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
