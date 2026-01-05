# CSV Analysis Agent API v2.0 - Быстрый старт

## 🚀 Быстрая настройка

### 1. Установка зависимостей

```bash
pip install -r requirements_api.txt
```

### 2. Настройка API ключа

Создайте файл `.env`:
```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Получите ключ на https://openrouter.ai/keys

### 3. Запуск сервера

```bash
python api_server.py
```

Сервер запустится на `http://localhost:8000`

---

## 📝 Основные сценарии

### 1. Автоматическая очистка (загрузка без запроса)

При загрузке CSV без текстового запроса агент автоматически:
- Определяет структуру таблицы
- Удаляет пустые строки/колонки
- Исправляет заголовки
- Приводит типы данных

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@messy_data.csv"
```

### 2. Анализ данных

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@sales.csv" \
  -F "query=Покажи топ-10 стран по продажам"
```

### 3. Редактирование данных

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@data.csv" \
  -F "query=Удали все строки где Price = 0"
```

### 4. Построение графиков

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@data.csv" \
  -F "query=Построй график продаж по месяцам"
```

---

## 💻 JavaScript пример

```javascript
async function analyzeCSV(file, query = "") {
  const formData = new FormData();
  formData.append('file', file);
  if (query) formData.append('query', query);

  const response = await fetch('http://localhost:8000/api/analyze', {
    method: 'POST',
    body: formData
  });
  return response.json();
}

// Использование
const fileInput = document.querySelector('input[type="file"]');
const file = fileInput.files[0];

// Автоочистка
const result = await analyzeCSV(file);
console.log(result.text_output);

// Если данные изменены - скачать
if (result.was_modified) {
  const csvBytes = atob(result.modified_csv);
  const blob = new Blob([csvBytes], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  window.open(url);
}
```

---

## 🔧 Примеры запросов на редактирование

| Запрос | Действие |
|--------|----------|
| Удали строки где Sales < 100 | Фильтрация по условию |
| Удали колонку Notes | Удаление столбца |
| Добавь колонку Profit = Revenue - Cost | Новый вычисляемый столбец |
| Переименуй Date в OrderDate | Переименование |
| Отсортируй по Amount по убыванию | Сортировка |
| Удали дубликаты | Очистка дубликатов |
| Оставь только первые 100 строк | Ограничение |

---

## 📊 Структура ответа

```json
{
  "success": true,
  "query": "Запрос пользователя",
  "text_output": "Результат анализа в Markdown",
  "plots": ["data:image/png;base64,..."],
  "modified_csv": "base64-encoded-csv или null",
  "was_modified": true/false,
  "file_info": {
    "filename": "data.csv",
    "rows": 1000,
    "columns": 12
  }
}
```

---

## 🔗 Endpoints

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/health` | GET | Проверка работы |
| `/api/info` | GET | Информация о сервисе |
| `/api/analyze` | POST | Основной endpoint |
| `/api/auto-clean` | POST | Только автоочистка |
| `/api/schema` | POST | Схема CSV файла |
| `/api/quick-analyze` | POST | Быстрый анализ без истории |

---

## 📚 Полная документация

См. [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
