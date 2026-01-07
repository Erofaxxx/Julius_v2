# API Integration Guide v2.1.0

## 🆕 Что изменилось в v2.1.0

### Основные изменения:

1. **Поддержка больших файлов (>10 МБ)** через signed URL из Supabase Storage
2. **Автоматическое переключение** между base64 и URL в зависимости от размера файла
3. **Новый endpoint** для скачивания результатов: `GET /api/download/{filename}`
4. **Обратная совместимость** - старый способ (прямая загрузка) продолжает работать

---

## 📤 Отправка запросов

### Режим 1: Маленькие файлы (<10 МБ) - КАК РАНЬШЕ

```typescript
// Прямая загрузка файла (без изменений)
const formData = new FormData();
formData.append('file', fileBlob);
formData.append('query', userQuery);
formData.append('chat_history', JSON.stringify(history));

const response = await fetch('https://server.asktab.ru/api/analyze', {
  method: 'POST',
  body: formData
});
```

### Режим 2: Большие файлы (>10 МБ) - НОВЫЙ СПОСОБ

```typescript
// 1. Загружаем файл в Supabase Storage
const { data: uploadData, error: uploadError } = await supabase.storage
  .from('user-files')
  .upload(`user-${userId}/${fileId}/${filename}`, fileBlob);

if (uploadError) throw uploadError;

// 2. Создаём signed URL (срок действия 1 час)
const { data: signedUrlData, error: signedUrlError } = await supabase.storage
  .from('user-files')
  .createSignedUrl(uploadData.path, 3600);

if (signedUrlError) throw signedUrlError;

// 3. Отправляем signed URL вместо файла
const formData = new FormData();
formData.append('file_url', signedUrlData.signedUrl);  // 🆕 НОВОЕ
formData.append('file_name', filename);                // 🆕 НОВОЕ
formData.append('file_type', fileBlob.type);           // 🆕 ОПЦИОНАЛЬНО
formData.append('query', userQuery);
formData.append('chat_history', JSON.stringify(history));

const response = await fetch('https://server.asktab.ru/api/analyze', {
  method: 'POST',
  body: formData
});
```

### Универсальная функция (автоматический выбор)

```typescript
async function analyzeFile(
  file: File, 
  query: string, 
  history?: any[]
): Promise<ApiResponse> {
  const LARGE_FILE_THRESHOLD = 10 * 1024 * 1024; // 10 MB
  const formData = new FormData();
  
  // Автоматический выбор режима по размеру файла
  if (file.size > LARGE_FILE_THRESHOLD) {
    console.log('📤 Большой файл, используем signed URL...');
    
    // Загрузка в Supabase Storage
    const fileId = crypto.randomUUID();
    const filePath = `user-${userId}/${fileId}/${file.name}`;
    
    const { data: uploadData, error: uploadError } = await supabase.storage
      .from('user-files')
      .upload(filePath, file);
    
    if (uploadError) throw new Error(`Upload failed: ${uploadError.message}`);
    
    // Создание signed URL
    const { data: signedUrlData, error: signedUrlError } = await supabase.storage
      .from('user-files')
      .createSignedUrl(uploadData.path, 3600);
    
    if (signedUrlError) throw new Error(`Signed URL failed: ${signedUrlError.message}`);
    
    // Отправка через signed URL
    formData.append('file_url', signedUrlData.signedUrl);
    formData.append('file_name', file.name);
    formData.append('file_type', file.type);
  } else {
    console.log('📤 Маленький файл, прямая загрузка...');
    
    // Прямая загрузка (как раньше)
    formData.append('file', file);
  }
  
  // Общие параметры
  formData.append('query', query);
  if (history) {
    formData.append('chat_history', JSON.stringify(history));
  }
  
  // Отправка запроса
  const response = await fetch('https://server.asktab.ru/api/analyze', {
    method: 'POST',
    body: formData
  });
  
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  
  return await response.json();
}
```

---

## 📥 Обработка ответов

### Структура ответа v2.1.0

```typescript
interface ApiResponse {
  success: boolean;
  text_output: string;
  plots?: string[];              // Графики в base64
  code?: string;
  
  // Изменённые данные
  was_modified: boolean;
  
  // 🆕 НОВЫЕ ПОЛЯ для режима доставки
  file_delivery_mode?: 'base64' | 'url';
  
  // Режим 1: Маленький файл (base64)
  modified_csv?: string;         // Base64 CSV данных
  
  // Режим 2: Большой файл (URL)
  modified_file_url?: string;    // 🆕 URL для скачивания
  modified_file_name?: string;   // 🆕 Имя файла
  
  // Метаинформация
  file_info?: {
    filename: string;
    size_bytes: number;
    rows: number;
    columns: number;
  };
  model_info?: {
    model_name: string;
  };
}
```

### Универсальная обработка результата

```typescript
async function handleAnalysisResult(response: ApiResponse) {
  console.log('📊 Результат анализа:', response.text_output);
  
  // Отображаем графики если есть
  if (response.plots && response.plots.length > 0) {
    response.plots.forEach(plotBase64 => {
      displayPlot(plotBase64); // Ваша функция отображения
    });
  }
  
  // Обрабатываем изменённые данные
  if (response.was_modified) {
    if (response.file_delivery_mode === 'url') {
      // Большой файл - скачивание по URL
      console.log('💾 Большой файл, скачивание по URL...');
      await downloadFileFromUrl(
        response.modified_file_url!,
        response.modified_file_name!
      );
    } else {
      // Маленький файл - использование base64
      console.log('💾 Маленький файл, конвертация из base64...');
      downloadBase64File(
        response.modified_csv!,
        response.file_info?.filename || 'modified.csv'
      );
    }
  }
}

// Скачивание файла по URL
async function downloadFileFromUrl(url: string, filename: string) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Download failed: ${response.status}`);
    }
    
    const blob = await response.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    window.URL.revokeObjectURL(downloadUrl);
    console.log('✅ Файл скачан:', filename);
  } catch (error) {
    console.error('❌ Ошибка скачивания:', error);
    throw error;
  }
}

// Скачивание файла из base64
function downloadBase64File(base64Data: string, filename: string) {
  try {
    const link = document.createElement('a');
    link.href = `data:text/csv;charset=utf-8-sig;base64,${base64Data}`;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    console.log('✅ Файл скачан:', filename);
  } catch (error) {
    console.error('❌ Ошибка скачивания:', error);
    throw error;
  }
}
```

---

## 🎯 Рекомендации по использованию

### Когда использовать signed URL:

✅ **Используйте signed URL когда:**
- Размер файла > 10 МБ
- Файл уже загружен в Supabase Storage
- Хотите избежать лимитов памяти в Edge Functions
- Обрабатываете файлы 50-200+ МБ

❌ **НЕ используйте signed URL когда:**
- Размер файла < 10 МБ (накладные расходы не оправданы)
- Файл создан динамически на клиенте
- Нужна максимальная скорость для маленьких файлов

### Пороговые значения:

```typescript
const FILE_SIZE_THRESHOLDS = {
  DIRECT_UPLOAD: 10 * 1024 * 1024,      // <10 MB - прямая загрузка
  SIGNED_URL: 10 * 1024 * 1024,         // >10 MB - signed URL
  MAX_FILE_SIZE: 200 * 1024 * 1024      // 200 MB - максимум
};
```

---

## 🔧 Настройки Supabase Storage

### Конфигурация bucket для больших файлов:

```sql
-- Создание bucket если ещё не создан
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'user-files',
  'user-files',
  false,  -- Приватный bucket
  209715200,  -- 200 MB лимит
  ARRAY['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']
);

-- RLS политики
CREATE POLICY "Users can upload their own files"
ON storage.objects FOR INSERT
TO authenticated
WITH CHECK (bucket_id = 'user-files' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Users can read their own files"
ON storage.objects FOR SELECT
TO authenticated
USING (bucket_id = 'user-files' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Users can delete their own files"
ON storage.objects FOR DELETE
TO authenticated
USING (bucket_id = 'user-files' AND auth.uid()::text = (storage.foldername(name))[1]);
```

---

## 📝 Примеры использования

### Пример 1: Простой запрос (маленький файл)

```typescript
const file = document.querySelector('input[type="file"]').files[0];
const query = "Покажи статистику по продажам";

const response = await analyzeFile(file, query);
await handleAnalysisResult(response);
```

### Пример 2: Большой файл с историей

```typescript
const file = new File([csvData], 'large_data.csv'); // 50 MB
const query = "Удали строки с пустыми значениями";
const history = [
  { query: "Покажи первые 10 строк", response: "..." }
];

const response = await analyzeFile(file, query, history);

if (response.file_delivery_mode === 'url') {
  console.log('📥 Результат доступен по ссылке (действует 1 час):');
  console.log(response.modified_file_url);
}
```

### Пример 3: Автоочистка данных

```typescript
// Для endpoint /api/auto-clean (работает аналогично)
const formData = new FormData();

if (file.size > LARGE_FILE_THRESHOLD) {
  // Signed URL для больших файлов
  formData.append('file_url', signedUrl);
  formData.append('file_name', file.name);
} else {
  // Прямая загрузка для маленьких
  formData.append('file', file);
}

const response = await fetch('https://server.asktab.ru/api/auto-clean', {
  method: 'POST',
  body: formData
});
```

---

## ⚠️ Важные замечания

### 1. Срок действия signed URL

- Signed URL действует **1 час** (3600 секунд)
- После истечения срока файл будет удалён с API сервера
- Если пользователю нужен файл позже - сохраните его в Supabase Storage

### 2. Результаты больших файлов

- Результаты >10 МБ возвращаются через URL
- URL действителен **1 час** с момента создания
- После скачивания сохраните файл в Supabase Storage если нужно

### 3. Очистка временных файлов

- API сервер автоматически удаляет файлы старше 1 часа
- Периодическая очистка каждые 30 минут
- Не полагайтесь на длительное хранение на API сервере

### 4. Обратная совместимость

- Старый код (без signed URL) продолжает работать
- Можно постепенно мигрировать на новый формат
- Оба способа могут использоваться одновременно

---

## 🔄 Миграция с v2.0 на v2.1

### Что нужно изменить:

1. **Добавить обработку `file_delivery_mode`** в ответе
2. **Добавить функцию скачивания по URL** (`downloadFileFromUrl`)
3. **Опционально: добавить автовыбор** режима по размеру файла
4. **Обновить TypeScript интерфейсы** для новых полей

### Что НЕ нужно менять:

- ✅ Существующие запросы с прямой загрузкой работают
- ✅ Структура `text_output`, `plots` не изменилась
- ✅ Endpoint `/api/analyze` остался тем же
- ✅ Формат `chat_history` не изменился

---

## 🐛 Troubleshooting

### Проблема: "file_name обязателен при использовании file_url"

**Решение:** Всегда передавайте `file_name` вместе с `file_url`:

```typescript
formData.append('file_url', signedUrl);
formData.append('file_name', originalFilename); // ✅ Обязательно
```

### Проблема: "Файл не найден или срок действия ссылки истёк"

**Решение:** URL для скачивания действителен 1 час. Скачайте файл сразу или сохраните в Supabase Storage.

### Проблема: "Не удалось скачать файл по URL"

**Решение:** Проверьте CORS заголовки и что URL начинается с `https://`:

```typescript
// URL должен быть вида:
// https://server.asktab.ru/api/download/filename_uuid.csv
```

---

## 📚 Дополнительная информация

- **API Documentation:** [`API_DOCUMENTATION.md`](API_DOCUMENTATION.md)
- **Quick Start:** [`API_QUICKSTART.md`](API_QUICKSTART.md)
- **Changelog v2.1:** [`CHANGELOG_V2.1.md`](CHANGELOG_V2.1.md)
- **Lovable Integration:** [`LOVABLE_INTEGRATION.md`](LOVABLE_INTEGRATION.md)

---

## 💡 Best Practices

1. **Всегда проверяйте `file_delivery_mode`** перед обработкой результата
2. **Используйте автоматический выбор** режима по размеру файла
3. **Сохраняйте результаты** в Supabase Storage если нужно долгое хранение
4. **Обрабатывайте ошибки** скачивания с повторными попытками
5. **Показывайте прогресс** пользователю при работе с большими файлами
6. **Логируйте** какой режим был использован для отладки

---

**Версия документа:** 1.0  
**Дата:** 7 января 2026  
**API Version:** 2.1.0
