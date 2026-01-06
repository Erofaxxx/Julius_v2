# Интеграция с Lovable - CSV Analysis Agent API

Подробная инструкция по интеграции фронтенда Lovable с бэкендом CSV Analysis Agent.

---

## 🏗️ Архитектура

```
┌─────────────────┐         ┌─────────────────┐
│   Lovable       │  HTTP   │   API Server    │
│   Frontend      │ ◄─────► │   (FastAPI)     │
│   (React)       │  JSON   │   + Claude AI   │
└─────────────────┘         └─────────────────┘
```

**API URL:** `https://server.asktab.ru` (замените на реальный домен)

---

## 📁 Управление файлами

### Концепция хранения файла

```typescript
// Храним текущий файл в состоянии
interface FileState {
  file: File | null;           // Оригинальный/текущий файл
  filename: string;            // Имя файла
  lastModifiedCsv: string | null; // Последний изменённый CSV (base64)
}

const [fileState, setFileState] = useState<FileState>({
  file: null,
  filename: '',
  lastModifiedCsv: null
});
```

### Логика замены файла

```typescript
// После каждого ответа от API проверяем was_modified
function handleApiResponse(response: ApiResponse) {
  if (response.was_modified && response.modified_csv) {
    // Заменяем файл на новую версию
    const newFile = base64ToFile(response.modified_csv, fileState.filename);
    
    setFileState(prev => ({
      ...prev,
      file: newFile,
      lastModifiedCsv: response.modified_csv
    }));
    
    console.log('✅ Файл обновлён до новой версии');
  }
  // Если was_modified = false, файл остаётся прежним
}
```

---

## 🔧 Утилиты для работы с файлами

```typescript
// utils/fileUtils.ts

/**
 * Конвертация base64 в File объект
 */
export function base64ToFile(base64: string, filename: string): File {
  // Декодируем base64
  const byteCharacters = atob(base64);
  const byteNumbers = new Array(byteCharacters.length);
  
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  
  const byteArray = new Uint8Array(byteNumbers);
  const blob = new Blob([byteArray], { type: 'text/csv;charset=utf-8;' });
  
  return new File([blob], filename, { type: 'text/csv' });
}

/**
 * Скачивание файла
 */
export function downloadFile(base64: string, filename: string): void {
  const byteCharacters = atob(base64);
  const byteNumbers = new Array(byteCharacters.length);
  
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  
  const byteArray = new Uint8Array(byteNumbers);
  const blob = new Blob([byteArray], { type: 'text/csv;charset=utf-8;' });
  
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Получить размер файла в читаемом формате
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
```

---

## 📡 API сервис

```typescript
// services/csvAgentApi.ts

const API_BASE_URL = 'https://server.asktab.ru'; // Замените на реальный URL

export interface ApiResponse {
  success: boolean;
  query: string;
  text_output: string | null;
  result_data: any;
  plots: string[];
  modified_csv: string | null;
  was_modified: boolean;
  error: string | null;
  file_info: {
    filename: string;
    rows: number;
    columns: number;
  };
  cleaning_steps?: string[];
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  plots?: string[];
  isLoading?: boolean;
  fileModified?: boolean;
  timestamp: Date;
}

/**
 * Отправка файла на анализ
 * @param file - CSV файл (может быть File или base64 строка предыдущей версии)
 * @param query - Запрос пользователя (пустой = автоочистка)
 * @param chatHistory - История чата для контекста
 */
export async function analyzeCSV(
  file: File,
  query: string = '',
  chatHistory?: Array<{ query: string; success: boolean; text_output?: string }>
): Promise<ApiResponse> {
  const formData = new FormData();
  formData.append('file', file);
  
  if (query) {
    formData.append('query', query);
  }
  
  if (chatHistory && chatHistory.length > 0) {
    formData.append('chat_history', JSON.stringify(chatHistory));
  }
  
  const response = await fetch(`${API_BASE_URL}/api/analyze`, {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.message || 'Ошибка API');
  }
  
  return response.json();
}

/**
 * Получить схему CSV файла
 */
export async function getSchema(file: File): Promise<any> {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${API_BASE_URL}/api/schema`, {
    method: 'POST',
    body: formData,
  });
  
  return response.json();
}

/**
 * Проверка здоровья API
 */
export async function healthCheck(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    const data = await response.json();
    return data.status === 'healthy';
  } catch {
    return false;
  }
}
```

---

## 💬 Компонент чата

```tsx
// components/ChatInterface.tsx

import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { analyzeCSV, ApiResponse, ChatMessage } from '../services/csvAgentApi';
import { base64ToFile, downloadFile } from '../utils/fileUtils';

interface FileState {
  file: File | null;
  filename: string;
  lastModifiedCsv: string | null;
}

export function ChatInterface() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [fileState, setFileState] = useState<FileState>({
    file: null,
    filename: '',
    lastModifiedCsv: null
  });
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Автоскролл к последнему сообщению
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  // Загрузка файла
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    // Сохраняем файл
    setFileState({
      file: file,
      filename: file.name,
      lastModifiedCsv: null
    });
    
    // Добавляем сообщение пользователя
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: `📎 Загружен файл: ${file.name}`,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    
    // Отправляем на автоочистку (пустой query)
    await sendRequest(file, '');
  };
  
  // Отправка запроса
  const sendRequest = async (file: File, query: string) => {
    setIsLoading(true);
    
    // Добавляем loading сообщение
    const loadingMessage: ChatMessage = {
      id: `loading-${Date.now()}`,
      role: 'assistant',
      content: '',
      isLoading: true,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, loadingMessage]);
    
    try {
      // Формируем историю для контекста
      const chatHistory = messages
        .filter(m => m.role === 'user' || (m.role === 'assistant' && !m.isLoading))
        .slice(-10)
        .map(m => ({
          query: m.content,
          success: true,
          text_output: m.content
        }));
      
      // Отправляем запрос
      const response = await analyzeCSV(file, query, chatHistory);
      
      // Обрабатываем ответ
      handleApiResponse(response);
      
    } catch (error) {
      // Заменяем loading на ошибку
      setMessages(prev => prev.map(m => 
        m.isLoading 
          ? {
              ...m,
              isLoading: false,
              content: `❌ Ошибка: ${error instanceof Error ? error.message : 'Неизвестная ошибка'}`
            }
          : m
      ));
    } finally {
      setIsLoading(false);
    }
  };
  
  // Обработка ответа API
  const handleApiResponse = (response: ApiResponse) => {
    // Формируем контент сообщения
    let content = '';
    
    if (response.success) {
      // Основной текст ответа
      if (response.text_output) {
        content = response.text_output;
      }
      
      // Результат анализа (если есть)
      if (response.result_data && typeof response.result_data === 'string') {
        content += '\n\n' + response.result_data;
      }
      
      // Информация об изменении файла
      if (response.was_modified) {
        content += '\n\n---\n📝 **Файл был изменён.** Нажмите "Скачать" чтобы сохранить новую версию.';
      }
    } else {
      content = `❌ Ошибка: ${response.error || 'Неизвестная ошибка'}`;
    }
    
    // Заменяем loading сообщение на реальное
    setMessages(prev => prev.map(m => 
      m.isLoading 
        ? {
            id: Date.now().toString(),
            role: 'assistant' as const,
            content: content,
            plots: response.plots,
            fileModified: response.was_modified,
            isLoading: false,
            timestamp: new Date()
          }
        : m
    ));
    
    // ВАЖНО: Обновляем файл если он был изменён
    if (response.was_modified && response.modified_csv) {
      const newFile = base64ToFile(response.modified_csv, fileState.filename);
      
      setFileState(prev => ({
        ...prev,
        file: newFile,
        lastModifiedCsv: response.modified_csv
      }));
    }
  };
  
  // Отправка сообщения пользователя
  const handleSendMessage = async () => {
    if (!input.trim() || !fileState.file || isLoading) return;
    
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    const query = input;
    setInput('');
    
    // Отправляем текущую версию файла
    await sendRequest(fileState.file, query);
  };
  
  // Скачивание текущей версии файла
  const handleDownload = () => {
    if (fileState.lastModifiedCsv) {
      downloadFile(fileState.lastModifiedCsv, `modified_${fileState.filename}`);
    }
  };
  
  return (
    <div className="chat-container">
      {/* Шапка с информацией о файле */}
      {fileState.file && (
        <div className="file-header">
          <span>📄 {fileState.filename}</span>
          {fileState.lastModifiedCsv && (
            <button onClick={handleDownload} className="download-btn">
              📥 Скачать изменённый файл
            </button>
          )}
        </div>
      )}
      
      {/* Область сообщений */}
      <div className="messages-area">
        {messages.length === 0 && (
          <div className="empty-state">
            <p>👋 Загрузите CSV файл для начала работы</p>
            <button onClick={() => fileInputRef.current?.click()}>
              📎 Загрузить CSV
            </button>
          </div>
        )}
        
        {messages.map(message => (
          <div key={message.id} className={`message ${message.role}`}>
            {message.isLoading ? (
              <div className="loading-indicator">
                <span className="dot"></span>
                <span className="dot"></span>
                <span className="dot"></span>
                Анализирую данные...
              </div>
            ) : (
              <>
                <ReactMarkdown>{message.content}</ReactMarkdown>
                
                {/* Графики */}
                {message.plots && message.plots.length > 0 && (
                  <div className="plots-container">
                    {message.plots.map((plot, index) => (
                      <img 
                        key={index} 
                        src={plot} 
                        alt={`График ${index + 1}`}
                        className="plot-image"
                      />
                    ))}
                  </div>
                )}
                
                {/* Кнопка скачивания если файл изменён */}
                {message.fileModified && (
                  <button onClick={handleDownload} className="inline-download-btn">
                    📥 Скачать изменённый файл
                  </button>
                )}
              </>
            )}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      
      {/* Поле ввода */}
      <div className="input-area">
        <input
          type="file"
          ref={fileInputRef}
          accept=".csv"
          onChange={handleFileUpload}
          style={{ display: 'none' }}
        />
        
        <button 
          onClick={() => fileInputRef.current?.click()}
          className="attach-btn"
          disabled={isLoading}
        >
          📎
        </button>
        
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
          placeholder={fileState.file 
            ? "Спросите что-нибудь о данных..." 
            : "Сначала загрузите CSV файл"
          }
          disabled={!fileState.file || isLoading}
        />
        
        <button 
          onClick={handleSendMessage}
          disabled={!input.trim() || !fileState.file || isLoading}
          className="send-btn"
        >
          ➤
        </button>
      </div>
    </div>
  );
}
```

---

## 🎨 Стили (CSS)

```css
/* styles/chat.css */

.chat-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  max-width: 900px;
  margin: 0 auto;
  background: #f5f5f5;
}

.file-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 20px;
  background: #fff;
  border-bottom: 1px solid #e0e0e0;
}

.download-btn, .inline-download-btn {
  background: #4CAF50;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
}

.download-btn:hover, .inline-download-btn:hover {
  background: #45a049;
}

.messages-area {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}

.empty-state {
  text-align: center;
  color: #666;
  margin-top: 100px;
}

.empty-state button {
  margin-top: 20px;
  padding: 12px 24px;
  background: #2196F3;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 16px;
  cursor: pointer;
}

.message {
  margin-bottom: 16px;
  padding: 16px;
  border-radius: 12px;
  max-width: 85%;
}

.message.user {
  background: #2196F3;
  color: white;
  margin-left: auto;
}

.message.assistant {
  background: white;
  border: 1px solid #e0e0e0;
}

.loading-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #666;
}

.dot {
  width: 8px;
  height: 8px;
  background: #2196F3;
  border-radius: 50%;
  animation: bounce 1.4s infinite ease-in-out;
}

.dot:nth-child(1) { animation-delay: -0.32s; }
.dot:nth-child(2) { animation-delay: -0.16s; }

@keyframes bounce {
  0%, 80%, 100% { transform: scale(0); }
  40% { transform: scale(1); }
}

.plots-container {
  margin-top: 16px;
}

.plot-image {
  max-width: 100%;
  border-radius: 8px;
  margin-top: 8px;
}

.input-area {
  display: flex;
  gap: 8px;
  padding: 16px;
  background: white;
  border-top: 1px solid #e0e0e0;
}

.input-area input[type="text"] {
  flex: 1;
  padding: 12px 16px;
  border: 1px solid #e0e0e0;
  border-radius: 24px;
  font-size: 16px;
  outline: none;
}

.input-area input[type="text"]:focus {
  border-color: #2196F3;
}

.attach-btn, .send-btn {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  border: none;
  cursor: pointer;
  font-size: 20px;
}

.attach-btn {
  background: #e0e0e0;
}

.send-btn {
  background: #2196F3;
  color: white;
}

.send-btn:disabled {
  background: #ccc;
  cursor: not-allowed;
}

/* Markdown стили */
.message.assistant h2 {
  font-size: 1.3em;
  margin-top: 16px;
  margin-bottom: 8px;
}

.message.assistant h3 {
  font-size: 1.1em;
  margin-top: 12px;
  margin-bottom: 6px;
}

.message.assistant table {
  width: 100%;
  border-collapse: collapse;
  margin: 12px 0;
  font-size: 14px;
}

.message.assistant th, 
.message.assistant td {
  border: 1px solid #e0e0e0;
  padding: 8px 12px;
  text-align: left;
}

.message.assistant th {
  background: #f5f5f5;
  font-weight: 600;
}

.message.assistant ul, 
.message.assistant ol {
  margin: 8px 0;
  padding-left: 24px;
}

.message.assistant code {
  background: #f0f0f0;
  padding: 2px 6px;
  border-radius: 4px;
  font-family: monospace;
}
```

---

## 📋 Примеры запросов

| Действие | Запрос пользователя |
|----------|---------------------|
| Автоочистка | *(пустой запрос при загрузке)* |
| Анализ | "Покажи статистику по колонке Sales" |
| График | "Построй график продаж по месяцам" |
| Редактирование | "Удали строки где Price = 0" |
| Добавление | "Добавь колонку Profit = Revenue - Cost" |
| Фильтрация | "Оставь только данные за 2024 год" |

---

## ⚠️ Важные моменты

### 1. Всегда используй актуальный файл
```typescript
// При каждом запросе отправляй fileState.file
// Это может быть оригинал или уже изменённая версия
await sendRequest(fileState.file, query);
```

### 2. Проверяй was_modified
```typescript
if (response.was_modified && response.modified_csv) {
  // ЗАМЕНЯЕМ файл на новую версию
  const newFile = base64ToFile(response.modified_csv, filename);
  setFileState(prev => ({ ...prev, file: newFile }));
}
// Если was_modified = false, файл остаётся прежним
```

### 3. Передавай историю чата
```typescript
// Для контекста диалога передавай последние 5-10 сообщений
const chatHistory = messages.slice(-10).map(m => ({
  query: m.content,
  success: true,
  text_output: m.content
}));
```

### 4. Обрабатывай ошибки
```typescript
try {
  const response = await analyzeCSV(file, query);
  if (!response.success) {
    showError(response.error);
  }
} catch (error) {
  showError('Сервер недоступен');
}
```

---

## 🚀 Быстрый старт для Lovable

Скопируй в Lovable промпт:

```
Создай чат-интерфейс для анализа CSV файлов:

1. Пользователь загружает CSV файл
2. При загрузке автоматически отправляется на /api/analyze с пустым query
3. API возвращает очищенный файл и описание
4. Пользователь может задавать вопросы о данных
5. Если API возвращает was_modified=true, заменяй файл на modified_csv
6. Показывай графики из plots[] как картинки
7. Рендери text_output как Markdown

API: POST /api/analyze
FormData: file (File), query (string), chat_history (JSON string)

Ответ содержит:
- text_output: строка Markdown
- plots: массив base64 картинок  
- modified_csv: base64 CSV (если файл изменён)
- was_modified: boolean
```
