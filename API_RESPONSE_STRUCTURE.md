# CSV Analysis Agent API v2.0 - Структура ответов

## Основной ответ /api/analyze

```typescript
interface AnalyzeResponse {
  // Статус выполнения
  success: boolean;
  
  // Запрос пользователя или "[Автоматическая очистка]"
  query: string;
  
  // Попытки выполнения кода (для отладки)
  code_attempts: CodeAttempt[];
  
  // Финальный успешный код
  final_code: string | null;
  
  // Результат выполнения (если есть)
  result_data: any | null;
  
  // Текстовый вывод (логи + результат в Markdown)
  text_output: string | null;
  
  // Графики в base64
  plots: string[];
  
  // Ошибка (если success=false)
  error: string | null;
  error_details?: string;
  
  // Количество попыток
  attempts_count: number;
  
  // Время выполнения
  timestamp: string;
  
  // Метаданные загрузки
  load_info: LoadInfo;
  
  // === НОВОЕ В v2.0 ===
  
  // Изменённый CSV в base64 (если данные редактировались)
  modified_csv: string | null;
  
  // Флаг изменения данных
  was_modified: boolean;
  
  // Шаги очистки (для автоочистки)
  cleaning_steps?: string[];
  
  // Информация о файле
  file_info: FileInfo;
  
  // Информация о модели
  model_info: ModelInfo;
}

interface CodeAttempt {
  attempt: number;
  code: string;
  success: boolean;
  error?: string;
}

interface LoadInfo {
  has_unnamed_columns: boolean;
  first_row_is_header: boolean;
  columns_cleaned: boolean;
  rows_removed: number;
  cols_removed: number;
  was_edited: boolean;
}

interface FileInfo {
  filename: string;
  size_bytes: number;
  rows: number;
  columns: number;
}

interface ModelInfo {
  model_name: string;  // "Claude Sonnet 4.5"
}
```

---

## Примеры ответов

### 1. Автоматическая очистка (успех)

```json
{
  "success": true,
  "query": "[Автоматическая очистка]",
  "code_attempts": [],
  "final_code": "# Автоматическая очистка данных",
  "result_data": null,
  "text_output": "## 🧹 Автоматическая очистка данных\n\n### 📊 Исходные данные\n- **Файл:** sales.csv\n- **Размер:** 1002 строк × 14 колонок\n\n### ✅ Выполненные шаги очистки\n- 🗑️ Удалено 2 пустых строки в начале\n- 🎯 Первая строка преобразована в заголовки\n- 🗑️ Удалено 2 пустых колонок\n\n### 📈 Результат\n- **Размер после очистки:** 1000 строк × 12 колонок\n- **Колонки:** Country, Product, Sales, Date...\n\n### 📋 Первые строки очищенных данных\n| Country | Product | Sales |\n|---------|---------|-------|\n| USA | Widget | 1,234.50 |\n...",
  "plots": [],
  "error": null,
  "attempts_count": 1,
  "timestamp": "2024-01-01T12:00:00.000000",
  "load_info": {
    "has_unnamed_columns": true,
    "first_row_is_header": true,
    "columns_cleaned": true,
    "rows_removed": 2,
    "cols_removed": 2,
    "was_edited": true
  },
  "modified_csv": "Q291bnRyeSxQcm9kdWN0LFNhbGVzLERhdGUKVVNBLFdpZGdldCwxMjM0LjUsMjAyNC0wMS0wMQ==",
  "was_modified": true,
  "cleaning_steps": [
    "🗑️ Удалено 2 пустых строки в начале",
    "🎯 Первая строка преобразована в заголовки",
    "🗑️ Удалено 2 пустых колонок"
  ],
  "file_info": {
    "filename": "sales.csv",
    "size_bytes": 45678,
    "rows": 1000,
    "columns": 12
  },
  "model_info": {
    "model_name": "Claude Sonnet 4.5"
  }
}
```

### 2. Анализ данных (успех)

```json
{
  "success": true,
  "query": "Покажи топ-5 стран по продажам",
  "code_attempts": [
    {
      "attempt": 1,
      "code": "# === ШАГ 1: ПОНИМАНИЕ ДАННЫХ ===\nprint(\"🔍 ШАГ 1: Изучаю структуру данных...\")\n...",
      "success": true
    }
  ],
  "final_code": "# === ШАГ 1: ПОНИМАНИЕ ДАННЫХ ===\nprint(\"🔍 ШАГ 1: Изучаю структуру данных...\")\n...",
  "result_data": "## 📊 Топ-5 стран по продажам\n\n| Country | Total Sales |\n|---------|-------------|\n| USA | 1,234,567 |\n| Germany | 987,654 |\n...",
  "text_output": "🔍 ШАГ 1: Изучаю структуру данных...\nРазмер данных: 1000 строк, 12 колонок\n\n🧹 ШАГ 2: Проверяю качество данных...\n✅ Найдены колонки: Country, Sales\n\n📊 ШАГ 3: Выполняю анализ...\n✅ Агрегировано: 45 стран\n\n📈 ШАГ 4: Создаю визуализацию...\n✅ График создан\n\n✅ ШАГ 5: Формирую финальный отчет...\n✅ Анализ завершен успешно!",
  "plots": [
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA..."
  ],
  "error": null,
  "attempts_count": 1,
  "timestamp": "2024-01-01T12:00:00.000000",
  "load_info": {
    "has_unnamed_columns": false,
    "first_row_is_header": false,
    "columns_cleaned": false,
    "rows_removed": 0,
    "cols_removed": 0,
    "was_edited": false
  },
  "modified_csv": null,
  "was_modified": false,
  "file_info": {
    "filename": "sales.csv",
    "size_bytes": 45678,
    "rows": 1000,
    "columns": 12
  },
  "model_info": {
    "model_name": "Claude Sonnet 4.5"
  }
}
```

### 3. Редактирование данных (успех)

```json
{
  "success": true,
  "query": "Удали все строки где Sales < 100",
  "code_attempts": [
    {
      "attempt": 1,
      "code": "# Фильтрация данных\nprint(\"🔍 ШАГ 1: Изучаю структуру данных...\")\n...\ndf = df[df['Sales'] >= 100]\nmodified_df = df.copy()\n...",
      "success": true
    }
  ],
  "final_code": "# Фильтрация данных\n...",
  "result_data": "## ✅ Данные отредактированы\n\nУдалено 150 строк с Sales < 100\n\n| До | После |\n|----|-------|\n| 1000 | 850 |",
  "text_output": "🔍 ШАГ 1: Изучаю структуру данных...\nРазмер данных: 1000 строк, 12 колонок\n\n🧹 ШАГ 2: Проверяю качество данных...\n✅ Найдена колонка: Sales\n\n✏️ ШАГ 3: Редактирую данные...\n✅ Удалено 150 строк с Sales < 100\n✅ Осталось 850 строк\n\n✅ ШАГ 4: Сохраняю изменения...\n✅ Готово!",
  "plots": [],
  "error": null,
  "attempts_count": 1,
  "timestamp": "2024-01-01T12:00:00.000000",
  "load_info": {
    "has_unnamed_columns": false,
    "first_row_is_header": false,
    "columns_cleaned": false,
    "rows_removed": 0,
    "cols_removed": 0,
    "was_edited": true
  },
  "modified_csv": "Q291bnRyeSxQcm9kdWN0LFNhbGVzLERhdGUK...",
  "was_modified": true,
  "file_info": {
    "filename": "sales.csv",
    "size_bytes": 45678,
    "rows": 850,
    "columns": 12
  },
  "model_info": {
    "model_name": "Claude Sonnet 4.5"
  }
}
```

### 4. Ошибка

```json
{
  "success": false,
  "query": "Покажи данные из колонки XYZ",
  "code_attempts": [
    {
      "attempt": 1,
      "code": "...",
      "success": false,
      "error": "KeyError: 'XYZ'"
    },
    {
      "attempt": 2,
      "code": "...",
      "success": false,
      "error": "KeyError: 'XYZ'"
    },
    {
      "attempt": 3,
      "code": "...",
      "success": false,
      "error": "KeyError: 'XYZ'"
    }
  ],
  "final_code": null,
  "result_data": null,
  "text_output": null,
  "plots": [],
  "error": "Не удалось выполнить код после 3 попыток",
  "error_details": "KeyError: 'XYZ'\nTraceback...",
  "attempts_count": 3,
  "timestamp": "2024-01-01T12:00:00.000000",
  "load_info": {...},
  "modified_csv": null,
  "was_modified": false,
  "file_info": {...},
  "model_info": {
    "model_name": "Claude Sonnet 4.5"
  }
}
```

---

## Ответ /api/schema

```json
{
  "success": true,
  "schema": {
    "columns": ["Country", "Product", "Sales", "Date"],
    "dtypes": {
      "Country": "object",
      "Product": "object",
      "Sales": "float64",
      "Date": "datetime64[ns]"
    },
    "shape": {
      "rows": 1000,
      "columns": 4
    },
    "missing_values": {
      "Country": 0,
      "Product": 5,
      "Sales": 12,
      "Date": 0
    },
    "sample_data": [
      {"Country": "USA", "Product": "Widget", "Sales": 1234.5, "Date": "2024-01-01T00:00:00"},
      {"Country": "Germany", "Product": "Gadget", "Sales": 987.3, "Date": "2024-01-02T00:00:00"},
      ...
    ],
    "summary_stats": {
      "Sales": {
        "count": 988.0,
        "mean": 5432.1,
        "std": 2345.6,
        "min": 10.0,
        "25%": 2500.0,
        "50%": 5000.0,
        "75%": 7500.0,
        "max": 50000.0
      }
    },
    "metadata": {
      "has_unnamed_columns": false,
      "first_row_is_header": false,
      "columns_cleaned": true,
      "rows_removed": 2,
      "cols_removed": 0,
      "was_edited": false
    }
  },
  "filename": "sales.csv",
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

---

## Обработка modified_csv на фронтенде

### JavaScript

```javascript
function handleResponse(response) {
  // Показываем результат
  document.getElementById('output').innerHTML = response.text_output;
  
  // Показываем графики
  response.plots.forEach(plot => {
    const img = document.createElement('img');
    img.src = plot;
    document.getElementById('charts').appendChild(img);
  });
  
  // Если данные изменены - предлагаем скачать
  if (response.was_modified && response.modified_csv) {
    const downloadBtn = document.createElement('button');
    downloadBtn.textContent = '📥 Скачать изменённый CSV';
    downloadBtn.onclick = () => downloadCSV(response);
    document.getElementById('actions').appendChild(downloadBtn);
  }
}

function downloadCSV(response) {
  const csvBytes = atob(response.modified_csv);
  const blob = new Blob([csvBytes], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  
  const a = document.createElement('a');
  a.href = url;
  a.download = 'modified_' + response.file_info.filename;
  a.click();
  
  URL.revokeObjectURL(url);
}
```

### React

```tsx
function CSVAnalyzer() {
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  
  const handleDownload = () => {
    if (!result?.modified_csv) return;
    
    const csvBytes = atob(result.modified_csv);
    const blob = new Blob([csvBytes], { type: 'text/csv' });
    saveAs(blob, `modified_${result.file_info.filename}`);
  };
  
  return (
    <div>
      {result?.text_output && (
        <ReactMarkdown>{result.text_output}</ReactMarkdown>
      )}
      
      {result?.plots.map((plot, i) => (
        <img key={i} src={plot} alt={`Chart ${i + 1}`} />
      ))}
      
      {result?.was_modified && (
        <button onClick={handleDownload}>
          📥 Скачать изменённый CSV
        </button>
      )}
    </div>
  );
}
```
