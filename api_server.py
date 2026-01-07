"""
FastAPI сервер для CSV Analysis Agent
Интеграция с Lovable и другими frontend приложениями
Упрощённая версия - только один файл, только Claude Sonnet 4.5
"""

import os
import os.path
import io
import traceback
import uuid
import httpx
import asyncio
from typing import Optional
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
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
    version="2.1.0"
)

# Директория для временных файлов
TEMP_FILES_DIR = Path("./temp_files")
TEMP_FILES_DIR.mkdir(exist_ok=True)

# Размер файла для переключения на URL (10 МБ)
LARGE_FILE_THRESHOLD = 10 * 1024 * 1024  # 10 MB

# Время жизни временных файлов (1 час)
FILE_EXPIRY_TIME = timedelta(hours=1)


def cleanup_old_files():
    """Удаляет файлы старше 1 часа"""
    try:
        now = datetime.now()
        for file_path in TEMP_FILES_DIR.glob("*"):
            if file_path.is_file():
                file_age = now - datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_age > FILE_EXPIRY_TIME:
                    file_path.unlink()
                    print(f"🗑️ Удалён старый файл: {file_path.name}")
    except Exception as e:
        print(f"⚠️ Ошибка очистки файлов: {e}")


async def cleanup_file_after_delay(file_path: Path, delay: timedelta):
    """Удаляет файл после заданной задержки"""
    await asyncio.sleep(delay.total_seconds())
    try:
        if file_path.exists():
            file_path.unlink()
            print(f"🗑️ Удалён файл по истечении срока: {file_path.name}")
    except Exception as e:
        print(f"⚠️ Ошибка удаления файла {file_path.name}: {e}")


async def periodic_cleanup():
    """Периодическая очистка старых файлов каждые 30 минут"""
    while True:
        await asyncio.sleep(1800)  # 30 минут
        cleanup_old_files()


@app.on_event("startup")
async def startup_event():
    """Запуск фоновой задачи очистки при старте сервера"""
    asyncio.create_task(periodic_cleanup())
    print(f"✓ Директория для временных файлов: {TEMP_FILES_DIR.absolute()}")
    print(f"✓ Порог для больших файлов: {LARGE_FILE_THRESHOLD / (1024*1024):.0f} МБ")

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
        "version": "2.1.0",
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
        "version": "2.1.0",
        "model": MODEL_NAME,
        "features": [
            "Анализ CSV данных",
            "Поддержка больших файлов через signed URL",
            "Редактирование данных (добавление/удаление строк и колонок)",
            "Автоматическая очистка данных",
            "Построение графиков и визуализаций",
            "Возврат изменённого CSV файла (base64 или URL)"
        ],
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/analyze")
async def analyze_csv(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None, description="CSV/Excel файл для анализа"),
    file_url: Optional[str] = Form(None, description="Signed URL для загрузки большого файла"),
    file_name: Optional[str] = Form(None, description="Имя файла (используется с file_url)"),
    file_type: Optional[str] = Form(None, description="MIME тип файла (используется с file_url)"),
    query: Optional[str] = Form("", description="Запрос пользователя (пустой = автоочистка)"),
    chat_history: Optional[str] = Form(None, description="История чата в JSON формате")
):
    """
    Основной endpoint для анализа и редактирования CSV файла
    
    Поддерживает два режима:
    1. Прямая загрузка файла (file) - для файлов <10 МБ
    2. Загрузка по signed URL (file_url + file_name) - для больших файлов >10 МБ

    Args:
        file: Загруженный CSV/Excel файл (опционально)
        file_url: Signed URL для скачивания файла из Supabase Storage (опционально)
        file_name: Имя файла (обязательно при использовании file_url)
        file_type: MIME тип файла (опционально)
        query: Запрос пользователя. Если пустой - выполняется автоматическая очистка данных
        chat_history: JSON строка с историей предыдущих запросов (опционально)

    Returns:
        JSON с результатами анализа, включая:
        - text_output: текстовый результат анализа
        - plots: графики в base64
        - modified_csv: изменённый CSV в base64 (для маленьких файлов)
        - modified_file_url: URL для скачивания (для больших файлов)
        - modified_file_name: имя модифицированного файла
        - was_modified: флаг изменения данных
    """
    agent = None
    # Инициализация переменных для обоих режимов загрузки
    content_length = 0
    load_from_disk = False
    temp_download_path = None
    file_bytes = None

    try:
        # Определяем источник файла
        if file_url:
            # Режим 1: Скачивание по signed URL (для больших файлов)
            if not file_name:
                raise HTTPException(
                    status_code=400,
                    detail="file_name обязателен при использовании file_url"
                )
            
            print(f"📥 Скачивание большого файла по URL: {file_name}")
            print(f"🔗 URL: {file_url[:100]}...")
            
            try:
                # Проверка доступности URL перед скачиванием (HEAD запрос)
                content_length = 0
                async with httpx.AsyncClient(timeout=30.0) as client:
                    try:
                        head_response = await client.head(file_url)
                        content_length_str = head_response.headers.get('content-length')
                        if content_length_str:
                            content_length = int(content_length_str)
                            file_size_mb = content_length / (1024 * 1024)
                            print(f"📊 Размер файла: {file_size_mb:.2f} МБ")
                    except Exception as e:
                        print(f"⚠️ Не удалось получить размер файла: {e}")

                # Для больших файлов используем streaming в временный файл
                # чтобы не загружать всё в память (критично для серверов с 1 ГБ RAM)
                STREAM_THRESHOLD = 20 * 1024 * 1024  # 20 МБ - порог для streaming
                use_streaming = content_length > STREAM_THRESHOLD

                print(f"⏳ Начинаем загрузку... (таймаут: 600 сек, streaming: {use_streaming})")
                download_start = datetime.now()

                # Переменные для отслеживания режима загрузки
                temp_download_path = None
                file_bytes = None
                load_from_disk = False  # Флаг для загрузки напрямую с диска

                if use_streaming:
                    # Streaming загрузка в временный файл на диске
                    # ВАЖНО: НЕ читаем файл обратно в память!
                    temp_file_id = str(uuid.uuid4())
                    file_ext = os.path.splitext(file_name)[1].lower()
                    temp_download_path = TEMP_FILES_DIR / f"download_{temp_file_id}{file_ext}"

                    async with httpx.AsyncClient(
                        timeout=httpx.Timeout(600.0, connect=60.0)
                    ) as client:
                        async with client.stream("GET", file_url) as response:
                            response.raise_for_status()

                            downloaded_bytes = 0
                            chunk_size = 1024 * 1024  # 1 МБ chunks

                            with open(temp_download_path, 'wb') as f:
                                async for chunk in response.aiter_bytes(chunk_size):
                                    f.write(chunk)
                                    downloaded_bytes += len(chunk)
                                    # Логируем прогресс каждые 10 МБ
                                    if downloaded_bytes % (10 * 1024 * 1024) < chunk_size:
                                        print(f"   📥 Загружено: {downloaded_bytes / (1024*1024):.1f} МБ")

                    filename = file_name
                    load_from_disk = True  # Будем загружать напрямую с диска
                    file_size_mb = downloaded_bytes / (1024 * 1024)

                    download_time = (datetime.now() - download_start).total_seconds()
                    speed_mbps = file_size_mb / download_time if download_time > 0 else 0
                    print(f"✓ Файл скачан на диск: {file_size_mb:.2f} МБ за {download_time:.1f} сек ({speed_mbps:.2f} МБ/сек)")
                    print(f"💾 Режим: загрузка напрямую с диска (экономия RAM)")
                else:
                    # Прямая загрузка в память для маленьких файлов (<20 МБ)
                    async with httpx.AsyncClient(
                        timeout=httpx.Timeout(600.0, connect=60.0),
                        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
                    ) as client:
                        response = await client.get(file_url)
                        response.raise_for_status()
                        file_bytes = response.content
                        filename = file_name

                    download_time = (datetime.now() - download_start).total_seconds()
                    file_size_mb = len(file_bytes) / (1024 * 1024)
                    speed_mbps = file_size_mb / download_time if download_time > 0 else 0
                    print(f"✓ Файл скачан в память: {file_size_mb:.2f} МБ за {download_time:.1f} сек ({speed_mbps:.2f} МБ/сек)")
                
            except httpx.TimeoutException as e:
                error_msg = f"Таймаут при скачивании файла (>600 сек). Файл: {file_name}"
                print(f"❌ {error_msg}")
                raise HTTPException(
                    status_code=504,
                    detail=error_msg
                )
            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP ошибка {e.response.status_code} при скачивании файла"
                print(f"❌ {error_msg}")
                raise HTTPException(
                    status_code=400,
                    detail=f"{error_msg}: {str(e)}"
                )
            except httpx.RequestError as e:
                error_msg = f"Ошибка сети при скачивании файла"
                print(f"❌ {error_msg}: {e}")
                raise HTTPException(
                    status_code=503,
                    detail=f"{error_msg}. Проверьте доступность Supabase Storage."
                )
            except Exception as e:
                error_msg = f"Неизвестная ошибка при скачивании файла"
                print(f"❌ {error_msg}: {e}")
                print(f"📋 Traceback: {traceback.format_exc()}")
                raise HTTPException(
                    status_code=500,
                    detail=f"{error_msg}: {str(e)}"
                )
        elif file:
            # Режим 2: Прямая загрузка файла (для маленьких файлов)
            file_bytes = await file.read()
            filename = file.filename
            content_length = len(file_bytes)
            print(f"📤 Получен загруженный файл: {filename} ({len(file_bytes) / (1024*1024):.2f} МБ)")
        else:
            raise HTTPException(
                status_code=400,
                detail="Необходимо указать либо file, либо file_url + file_name"
            )

        # Проверка формата файла (CSV и Excel)
        allowed_extensions = ['.csv', '.xlsx', '.xls', '.xlsm']
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in allowed_extensions:
            # Удаляем временный файл если есть
            if temp_download_path and temp_download_path.exists():
                temp_download_path.unlink()
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый формат файла. Поддерживаются: {', '.join(allowed_extensions)}"
            )

        print(f"✓ Формат файла проверен: {file_ext}")
        if file_bytes:
            print(f"📊 Размер в памяти: {len(file_bytes) / (1024*1024):.2f} МБ")
        elif temp_download_path:
            print(f"📊 Размер на диске: {temp_download_path.stat().st_size / (1024*1024):.2f} МБ")

        # Парсинг истории если есть
        history = None
        if chat_history:
            import json
            try:
                history = json.loads(chat_history)
                print(f"✓ История чата загружена: {len(history)} сообщений")
            except json.JSONDecodeError:
                if temp_download_path and temp_download_path.exists():
                    temp_download_path.unlink()
                raise HTTPException(
                    status_code=400,
                    detail="Неверный формат chat_history. Требуется валидный JSON."
                )

        # Создание агента
        print(f"🤖 Создание AI агента...")
        agent = CSVAnalysisAgentAPI(api_key=OPENROUTER_API_KEY)
        print(f"✓ Агент создан")

        # Загрузка CSV - разные режимы для экономии памяти
        print(f"📂 Начинаем загрузку данных в pandas...")
        load_start = datetime.now()
        try:
            if load_from_disk and temp_download_path:
                # Большой файл - загружаем НАПРЯМУЮ С ДИСКА (экономия ~77+ МБ RAM!)
                print(f"💾 Режим загрузки: с диска (экономия памяти)")
                df = agent.load_csv_from_file(str(temp_download_path))
            else:
                # Маленький файл - загружаем из памяти (быстрее)
                print(f"💾 Режим загрузки: из памяти")
                df = agent.load_csv_from_bytes(file_bytes, filename)

            load_time = (datetime.now() - load_start).total_seconds()
            print(f"✓ Данные загружены за {load_time:.2f} сек: {df.shape[0]} строк × {df.shape[1]} колонок")
            print(f"💾 Память DataFrame: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} МБ")
        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            print(f"📋 Traceback: {traceback.format_exc()}")
            # Удаляем временный файл при ошибке
            if temp_download_path and temp_download_path.exists():
                temp_download_path.unlink()
            raise HTTPException(
                status_code=400,
                detail=f"Ошибка при чтении файла: {str(e)}"
            )
        finally:
            # Удаляем временный файл после загрузки в pandas
            if temp_download_path and temp_download_path.exists():
                temp_download_path.unlink()
                print(f"🗑️ Временный файл удалён")

        # Выполнение анализа (или автоочистки если query пустой)
        print(f"🧠 Начинаем AI анализ...")
        print(f"📝 Запрос: '{query[:100] if query else '[автоочистка]'}{'...' if len(query) > 100 else ''}')")
        analysis_start = datetime.now()
        
        try:
            result = agent.analyze(query, chat_history=history)
            analysis_time = (datetime.now() - analysis_start).total_seconds()
            print(f"✓ Анализ завершён за {analysis_time:.2f} сек")
        except Exception as e:
            print(f"❌ Ошибка AI анализа: {e}")
            print(f"📋 Traceback: {traceback.format_exc()}")
            raise

        # Добавляем информацию о файле
        print(f"📦 Формирование ответа...")
        # Вычисляем размер файла (file_bytes может быть None для больших файлов)
        file_size_bytes = len(file_bytes) if file_bytes else int(content_length) if content_length else 0
        result["file_info"] = {
            "filename": filename,
            "size_bytes": file_size_bytes,
            "rows": df.shape[0],
            "columns": df.shape[1]
        }
        result["model_info"] = {
            "model_name": MODEL_NAME
        }
        
        # Если данные были изменены и файл большой - сохраняем на диск
        if result.get("was_modified") and result.get("modified_csv"):
            print(f"💾 Обработка изменённых данных...")
            modified_csv_b64 = result["modified_csv"]
            
            # Оценка размера результата
            import base64
            estimated_size = len(base64.b64decode(modified_csv_b64))
            print(f"📏 Размер результата: {estimated_size / (1024*1024):.2f} МБ")
            
            if estimated_size > LARGE_FILE_THRESHOLD:
                # Большой файл - сохраняем на диск и возвращаем URL
                print(f"💾 Файл большой ({estimated_size / (1024*1024):.2f} МБ), сохраняем на диск...")
                
                # Генерируем уникальное имя файла
                unique_id = str(uuid.uuid4())
                base_name = os.path.splitext(filename)[0]
                result_filename = f"{base_name}_modified_{unique_id}.csv"
                result_path = TEMP_FILES_DIR / result_filename
                
                # Сохраняем файл
                csv_content = base64.b64decode(modified_csv_b64)
                result_path.write_bytes(csv_content)
                
                # Планируем удаление через 1 час
                background_tasks.add_task(lambda: cleanup_file_after_delay(result_path, FILE_EXPIRY_TIME))
                
                # Формируем URL для скачивания
                # Предполагаем что сервер доступен по HTTPS
                server_url = os.getenv("SERVER_URL", "https://server.asktab.ru")
                download_url = f"{server_url}/api/download/{result_filename}"
                
                # Заменяем base64 на URL
                result["modified_csv"] = None  # Убираем base64 для экономии трафика
                result["modified_file_url"] = download_url
                result["modified_file_name"] = result_filename
                result["file_delivery_mode"] = "url"  # Режим доставки
                
                print(f"✓ Файл сохранён: {result_filename}")
                print(f"✓ URL: {download_url}")
            else:
                # Маленький файл - оставляем base64
                result["file_delivery_mode"] = "base64"
                print(f"✓ Файл маленький ({estimated_size / 1024:.2f} КБ), возвращаем в base64")
        
        print(f"✅ Запрос обработан успешно")
        print(f"📤 Отправка ответа клиенту...")
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
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None, description="CSV файл для автоматической очистки"),
    file_url: Optional[str] = Form(None, description="Signed URL файла"),
    file_name: Optional[str] = Form(None, description="Имя файла")
):
    """
    Endpoint для автоматической очистки CSV файла
    Эквивалентен вызову /api/analyze с пустым query

    Args:
        file: Загруженный CSV файл (опционально)
        file_url: Signed URL для больших файлов (опционально)
        file_name: Имя файла (при использовании file_url)

    Returns:
        JSON с результатами очистки и изменённым CSV
    """
    return await analyze_csv(
        background_tasks=background_tasks,
        file=file,
        file_url=file_url,
        file_name=file_name,
        query="",
        chat_history=None
    )


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


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """
    Скачивание временного файла с результатами
    
    Args:
        filename: Имя файла для скачивания
    
    Returns:
        Файл для скачивания с CORS заголовками
    """
    try:
        file_path = TEMP_FILES_DIR / filename
        
        # Проверка существования файла
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Файл не найден или срок действия ссылки истёк (>1 часа)"
            )
        
        # Проверка что файл не слишком старый
        file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
        if file_age > FILE_EXPIRY_TIME:
            file_path.unlink()  # Удаляем старый файл
            raise HTTPException(
                status_code=410,
                detail="Срок действия ссылки истёк. Файл был удалён."
            )
        
        # Возвращаем файл с правильными заголовками
        return FileResponse(
            path=file_path,
            media_type="text/csv",
            filename=filename,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "Content-Disposition",
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-cache, no-store, must-revalidate"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при скачивании файла: {str(e)}"
        )


@app.post("/api/quick-analyze")
async def quick_analyze(
    background_tasks: BackgroundTasks,
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
    return await analyze_csv(
        background_tasks=background_tasks,
        file=file,
        query=query,
        chat_history=None
    )


# Запуск сервера
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"""
╔════════════════════════════════════════════════════════════╗
║         CSV Analysis Agent API Server v2.1                 ║
║         Powered by {MODEL_NAME}                       ║
╚════════════════════════════════════════════════════════════╝

Новые возможности v2.1:
✓ Поддержка больших файлов (50-200+ МБ) через signed URL
✓ Автоматическое переключение base64 ↔ URL в зависимости от размера
✓ Редактирование данных (добавление/удаление строк и колонок)
✓ Автоматическая очистка данных при загрузке без запроса
✓ Умный AI-агент с глубоким анализом данных

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
