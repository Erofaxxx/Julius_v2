"""
API-версия CSV Analysis Agent для интеграции с внешними сервисами
Julius.ai style - многоэтапный анализ с красивым выводом результатов
Поддерживает историю диалога, редактирование данных и возврат изменённого CSV
"""

import os
import io
import json
import traceback
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import contextlib
import base64
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI

# Проверка поддержки Excel форматов
try:
    import openpyxl
    EXCEL_SUPPORT = True
    print("✓ Поддержка Excel (.xlsx, .xlsm): Включена")
except ImportError:
    EXCEL_SUPPORT = False
    print("⚠️ openpyxl не установлен. Поддержка Excel отключена. Установите: pip install openpyxl")

try:
    import xlrd
    XLS_SUPPORT = True
    print("✓ Поддержка старых .xls файлов: Включена")
except ImportError:
    XLS_SUPPORT = False
    print("⚠️ xlrd не установлен. Поддержка .xls файлов отключена. Установите: pip install xlrd")


# Единственная модель - Claude Sonnet 4.5
MODEL_ID = "anthropic/claude-sonnet-4.5"
MODEL_NAME = "Claude Sonnet 4.5"


class CSVAnalysisAgentAPI:
    """
    API-версия агента для анализа и редактирования CSV файлов (Julius.ai style)
    Поддерживает историю диалога, редактирование данных и возврат изменённого CSV
    """

    def __init__(self, api_key: str):
        """
        Инициализация агента

        Args:
            api_key: API ключ для OpenRouter
        """
        self.api_key = api_key

        # Инициализация клиента OpenRouter
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        self.model = MODEL_ID
        self.model_name = MODEL_NAME

        self.current_df = None
        self.original_df = None  # Храним оригинал
        self.current_filename = None
        self.max_retries = 3

        # Метаданные о данных
        self.data_metadata = {
            "has_unnamed_columns": False,
            "first_row_is_header": False,
            "columns_cleaned": False,
            "rows_removed": 0,
            "cols_removed": 0,
            "was_edited": False
        }

        # Настройки для графиков
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100

    def _is_first_row_header(self, df: pd.DataFrame) -> bool:
        """
        Определяем является ли первая строка заголовком

        Критерии:
        1. Текущие колонки типа "Unnamed: 0", "Unnamed: 1"...
        2. Первая строка содержит текстовые значения (потенциальные названия)
        3. Вторая строка содержит числовые/смешанные значения (данные)
        """
        # Проверка 1: Много Unnamed колонок?
        unnamed_count = sum(1 for col in df.columns if 'Unnamed' in str(col))
        if unnamed_count < len(df.columns) * 0.3:  # Меньше 30% unnamed
            return False

        # Проверка 2: Первая строка - текст?
        if len(df) < 2:
            return False

        first_row = df.iloc[0]
        second_row = df.iloc[1]

        # Считаем текстовые значения в первой строке
        text_count_row1 = sum(1 for val in first_row if isinstance(val, str) and not str(val).replace('.', '').replace('-', '').isdigit())

        # Считаем числовые значения во второй строке
        numeric_count_row2 = sum(1 for val in second_row if pd.notna(val) and (isinstance(val, (int, float)) or str(val).replace('.', '').replace('-', '').isdigit()))

        # Если первая строка преимущественно текст, а вторая - числа
        return text_count_row1 > len(first_row) * 0.5 and numeric_count_row2 > len(second_row) * 0.3

    def _detect_separator(self, file_bytes: bytes) -> str:
        """
        Определить разделитель CSV файла
        """
        try:
            # Читаем первые строки для анализа
            sample = file_bytes[:8192].decode('utf-8', errors='ignore')
            lines = sample.split('\n')[:5]
            
            separators = [',', ';', '\t', '|']
            sep_counts = {}
            
            for sep in separators:
                counts = [line.count(sep) for line in lines if line.strip()]
                if counts:
                    # Ищем разделитель с одинаковым количеством во всех строках
                    if len(set(counts)) == 1 and counts[0] > 0:
                        sep_counts[sep] = counts[0]
                    elif counts:
                        sep_counts[sep] = max(counts)
            
            if sep_counts:
                return max(sep_counts, key=sep_counts.get)
            return ','
        except:
            return ','

    def _detect_encoding(self, file_bytes: bytes) -> str:
        """
        Определить кодировку файла
        """
        encodings = ['utf-8', 'utf-8-sig', 'cp1251', 'latin-1', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                file_bytes.decode(encoding)
                return encoding
            except:
                continue
        
        return 'utf-8'

    def smart_load_file(self, file_bytes: bytes, filename: str = "data.csv") -> Dict[str, Any]:
        """
        Умная загрузка CSV/Excel с автоматическим анализом структуры
        Работает как Julius.ai - сначала понимает структуру, потом очищает
        
        Поддерживаемые форматы:
        - CSV (.csv) - с автоопределением разделителя и кодировки
        - Excel (.xlsx, .xls, .xlsm) - читает первый лист

        Returns:
            Dict с информацией о загрузке и очистке
        """
        load_info = {
            "filename": filename,
            "steps": [],
            "warnings": [],
            "original_shape": None,
            "final_shape": None,
            "success": True,
            "file_format": "csv"
        }

        self.current_filename = filename
        
        # Определяем тип файла по расширению
        file_ext = os.path.splitext(filename)[1].lower()

        try:
            # Загрузка в зависимости от формата
            if file_ext in ['.xlsx', '.xls', '.xlsm']:
                # Excel файл
                load_info["file_format"] = "excel"
                load_info["steps"].append(f"📊 Определён формат: Excel ({file_ext})")
                
                # Проверяем наличие необходимых библиотек
                if file_ext == '.xls' and not XLS_SUPPORT:
                    raise Exception(
                        f"Формат .xls не поддерживается. "
                        f"Установите библиотеку: pip install xlrd"
                    )
                if file_ext in ['.xlsx', '.xlsm'] and not EXCEL_SUPPORT:
                    raise Exception(
                        f"Формат Excel не поддерживается. "
                        f"Установите библиотеку: pip install openpyxl"
                    )
                
                try:
                    # Читаем первый лист Excel файла
                    df_raw = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0)
                    load_info["steps"].append("📥 Загружен первый лист Excel файла")
                except Exception as e:
                    raise Exception(f"Ошибка чтения Excel файла: {str(e)}. Убедитесь что файл не повреждён.")
            else:
                # CSV файл
                load_info["file_format"] = "csv"
                # Определяем разделитель и кодировку
                sep = self._detect_separator(file_bytes)
                encoding = self._detect_encoding(file_bytes)
                
                load_info["steps"].append(f"🔍 Определён разделитель: '{sep}', кодировка: {encoding}")

                # Загружаем CSV "как есть"
                df_raw = pd.read_csv(io.BytesIO(file_bytes), sep=sep, encoding=encoding, on_bad_lines='skip')
            
            self.original_df = df_raw.copy()
            load_info["original_shape"] = df_raw.shape
            load_info["steps"].append(f"📥 Загружено: {df_raw.shape[0]} строк × {df_raw.shape[1]} колонок")

            # ШАГ 2: Проверяем "Unnamed" колонки
            unnamed_cols = [col for col in df_raw.columns if 'Unnamed' in str(col)]
            if unnamed_cols:
                self.data_metadata["has_unnamed_columns"] = True
                load_info["warnings"].append(
                    f"⚠️ Найдено {len(unnamed_cols)} колонок типа 'Unnamed'. "
                    f"Возможно первая строка - это заголовки."
                )
                load_info["steps"].append(f"🔍 Обнаружено {len(unnamed_cols)} безымянных колонок")

            # ШАГ 3: Проверяем первую строку - может это заголовки?
            if self._is_first_row_header(df_raw):
                self.data_metadata["first_row_is_header"] = True
                load_info["steps"].append("🎯 Обнаружено: первая строка - это заголовки данных")

                # Делаем первую строку заголовком
                new_columns = df_raw.iloc[0].tolist()
                df_raw.columns = new_columns
                df_raw = df_raw.iloc[1:].reset_index(drop=True)

                load_info["steps"].append("✅ Первая строка преобразована в заголовки")

            # ШАГ 4: Очищаем названия колонок от пробелов
            original_cols = list(df_raw.columns)
            df_raw.columns = df_raw.columns.astype(str).str.strip()
            cleaned_cols = list(df_raw.columns)

            if original_cols != cleaned_cols:
                self.data_metadata["columns_cleaned"] = True
                load_info["steps"].append("🧹 Очищены названия колонок от лишних пробелов")

            # ШАГ 5: Удаляем полностью пустые строки
            rows_before = len(df_raw)
            df_raw = df_raw.dropna(how='all')
            rows_after = len(df_raw)
            rows_removed = rows_before - rows_after

            if rows_removed > 0:
                self.data_metadata["rows_removed"] = rows_removed
                load_info["steps"].append(f"🗑️ Удалено {rows_removed} пустых строк")

            # ШАГ 6: Удаляем полностью пустые колонки
            cols_before = len(df_raw.columns)
            df_raw = df_raw.dropna(axis=1, how='all')
            cols_after = len(df_raw.columns)
            cols_removed = cols_before - cols_after

            if cols_removed > 0:
                self.data_metadata["cols_removed"] = cols_removed
                load_info["steps"].append(f"🗑️ Удалено {cols_removed} пустых колонок")

            # ШАГ 7: Удаляем колонки с только NaN/пустыми значениями
            # и колонки типа "Unnamed" если они полностью пустые
            cols_to_drop = []
            for col in df_raw.columns:
                if 'Unnamed' in str(col):
                    if df_raw[col].isna().all() or (df_raw[col].astype(str).str.strip() == '').all():
                        cols_to_drop.append(col)
            
            if cols_to_drop:
                df_raw = df_raw.drop(columns=cols_to_drop)
                load_info["steps"].append(f"🗑️ Удалено {len(cols_to_drop)} пустых Unnamed колонок")

            # Сохраняем результат
            self.current_df = df_raw.reset_index(drop=True)

            load_info["final_shape"] = self.current_df.shape
            load_info["steps"].append(
                f"✅ Итого: {self.current_df.shape[0]} строк × {self.current_df.shape[1]} колонок"
            )

            return load_info

        except Exception as e:
            load_info["success"] = False
            load_info["error"] = str(e)
            raise Exception(f"Ошибка при загрузке CSV файла '{filename}': {str(e)}")

    def load_csv_from_bytes(self, file_bytes: bytes, filename: str = "data.csv") -> pd.DataFrame:
        """
        Загрузить CSV/Excel из байтов (с умной очисткой)

        Args:
            file_bytes: Байты CSV или Excel файла
            filename: Имя файла (важно для определения формата)

        Returns:
            DataFrame с данными
        """
        self.smart_load_file(file_bytes, filename)
        return self.current_df
    
    # Алиас для обратной совместимости
    def smart_load_csv(self, file_bytes: bytes, filename: str = "data.csv") -> Dict[str, Any]:
        """Алиас для smart_load_file (обратная совместимость)"""
        return self.smart_load_file(file_bytes, filename)

    def load_csv_from_file(self, file_path: str) -> pd.DataFrame:
        """
        Загрузить CSV из пути (для больших файлов - без загрузки в память)

        Args:
            file_path: Путь к файлу

        Returns:
            DataFrame
        """
        filename = os.path.basename(file_path)
        self.current_filename = filename

        load_info = {
            "filename": filename,
            "steps": [],
            "warnings": [],
            "original_shape": None,
            "final_shape": None,
            "success": True,
            "file_format": "csv"
        }

        # Определяем тип файла по расширению
        file_ext = os.path.splitext(filename)[1].lower()

        try:
            # Загрузка в зависимости от формата - напрямую с диска!
            if file_ext in ['.xlsx', '.xls', '.xlsm']:
                load_info["file_format"] = "excel"
                load_info["steps"].append(f"📊 Определён формат: Excel ({file_ext})")

                if file_ext == '.xls' and not XLS_SUPPORT:
                    raise Exception(f"Формат .xls не поддерживается. Установите: pip install xlrd")
                if file_ext in ['.xlsx', '.xlsm'] and not EXCEL_SUPPORT:
                    raise Exception(f"Формат Excel не поддерживается. Установите: pip install openpyxl")

                # Читаем напрямую с диска (не через BytesIO!)
                df_raw = pd.read_excel(file_path, sheet_name=0)
                load_info["steps"].append("📥 Загружен первый лист Excel файла с диска")
            else:
                # CSV файл - читаем первые байты для определения параметров
                with open(file_path, 'rb') as f:
                    sample_bytes = f.read(8192)  # Только первые 8 КБ для анализа

                sep = self._detect_separator(sample_bytes)
                encoding = self._detect_encoding(sample_bytes)

                load_info["steps"].append(f"🔍 Определён разделитель: '{sep}', кодировка: {encoding}")

                # Читаем напрямую с диска (не через BytesIO!)
                df_raw = pd.read_csv(file_path, sep=sep, encoding=encoding, on_bad_lines='skip')

            # НЕ ДЕЛАЕМ КОПИЮ для экономии памяти!
            # self.original_df = df_raw.copy()  # УДАЛЕНО - экономия ~77+ МБ
            self.original_df = None  # Для больших файлов не храним копию

            load_info["original_shape"] = df_raw.shape
            load_info["steps"].append(f"📥 Загружено: {df_raw.shape[0]} строк × {df_raw.shape[1]} колонок")

            # Остальная обработка аналогична smart_load_file
            unnamed_cols = [col for col in df_raw.columns if 'Unnamed' in str(col)]
            if unnamed_cols:
                self.data_metadata["has_unnamed_columns"] = True
                load_info["steps"].append(f"🔍 Обнаружено {len(unnamed_cols)} безымянных колонок")

            if self._is_first_row_header(df_raw):
                self.data_metadata["first_row_is_header"] = True
                load_info["steps"].append("🎯 Первая строка - заголовки, преобразуем")
                new_columns = df_raw.iloc[0].tolist()
                df_raw.columns = new_columns
                df_raw = df_raw.iloc[1:].reset_index(drop=True)

            # Очищаем названия колонок
            df_raw.columns = df_raw.columns.astype(str).str.strip()

            # Удаляем пустые строки
            rows_before = len(df_raw)
            df_raw = df_raw.dropna(how='all')
            rows_removed = rows_before - len(df_raw)
            if rows_removed > 0:
                self.data_metadata["rows_removed"] = rows_removed
                load_info["steps"].append(f"🗑️ Удалено {rows_removed} пустых строк")

            # Удаляем пустые колонки
            cols_before = len(df_raw.columns)
            df_raw = df_raw.dropna(axis=1, how='all')
            cols_removed = cols_before - len(df_raw.columns)
            if cols_removed > 0:
                self.data_metadata["cols_removed"] = cols_removed
                load_info["steps"].append(f"🗑️ Удалено {cols_removed} пустых колонок")

            # Удаляем пустые Unnamed колонки
            cols_to_drop = []
            for col in df_raw.columns:
                if 'Unnamed' in str(col):
                    if df_raw[col].isna().all() or (df_raw[col].astype(str).str.strip() == '').all():
                        cols_to_drop.append(col)
            if cols_to_drop:
                df_raw = df_raw.drop(columns=cols_to_drop)
                load_info["steps"].append(f"🗑️ Удалено {len(cols_to_drop)} пустых Unnamed колонок")

            self.current_df = df_raw.reset_index(drop=True)
            load_info["final_shape"] = self.current_df.shape
            load_info["steps"].append(f"✅ Итого: {self.current_df.shape[0]} строк × {self.current_df.shape[1]} колонок")

            return self.current_df

        except Exception as e:
            raise Exception(f"Ошибка при загрузке файла '{filename}': {str(e)}")

    def analyze_csv_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ схемы CSV файла

        Args:
            df: DataFrame для анализа

        Returns:
            Словарь с информацией о схеме
        """
        schema = {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "missing_values": {col: int(count) for col, count in df.isnull().sum().items()},
            "sample_data": df.head(5).to_dict(orient='records'),
            "summary_stats": {},
            "metadata": self.data_metadata
        }

        # Статистика для числовых колонок
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_df = df[numeric_cols].describe()
            schema["summary_stats"] = {
                col: {stat: float(val) for stat, val in stats_df[col].items()}
                for col in numeric_cols
            }

        return schema

    def get_deep_data_profile(self, df: pd.DataFrame) -> str:
        """
        Создаёт глубокий профиль данных для ИИ-агента.
        Агент видит полную картину данных перед принятием решений.
        
        Returns:
            Строка с детальным описанием данных
        """
        profile_lines = []
        profile_lines.append("=" * 60)
        profile_lines.append("📊 ГЛУБОКИЙ ПРОФИЛЬ ДАННЫХ")
        profile_lines.append("=" * 60)
        
        # Базовая информация
        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isna().sum().sum()
        fill_rate = ((total_cells - total_missing) / total_cells * 100) if total_cells > 0 else 0
        
        profile_lines.append(f"\n📐 РАЗМЕР: {df.shape[0]} строк × {df.shape[1]} колонок")
        profile_lines.append(f"📈 ЗАПОЛНЕННОСТЬ: {fill_rate:.1f}% ({total_cells - total_missing}/{total_cells} ячеек)")
        profile_lines.append(f"⚠️ ВСЕГО ПУСТЫХ ЯЧЕЕК: {total_missing}")
        
        # Детальный анализ каждой колонки
        profile_lines.append(f"\n{'─' * 60}")
        profile_lines.append("📋 ДЕТАЛЬНЫЙ АНАЛИЗ КОЛОНОК:")
        profile_lines.append(f"{'─' * 60}")
        
        for col in df.columns:
            col_data = df[col]
            missing_count = col_data.isna().sum()
            missing_pct = (missing_count / len(df) * 100) if len(df) > 0 else 0
            non_null_count = col_data.notna().sum()
            unique_count = col_data.nunique()
            
            # Определяем тип данных более точно
            dtype = str(col_data.dtype)
            if dtype == 'object':
                # Проверяем на даты
                sample_vals = col_data.dropna().head(10)
                is_date = False
                if len(sample_vals) > 0:
                    try:
                        pd.to_datetime(sample_vals, errors='raise')
                        is_date = True
                        dtype = "datetime (текст)"
                    except:
                        pass
                if not is_date:
                    dtype = "text"
            elif 'int' in dtype or 'float' in dtype:
                dtype = "numeric"
            elif 'datetime' in dtype:
                dtype = "datetime"
            
            # Формируем информацию о колонке
            profile_lines.append(f"\n▸ '{col}' [{dtype}]")
            
            if missing_count > 0:
                warning = "⚠️" if missing_pct > 10 else "ℹ️"
                profile_lines.append(f"  {warning} Пустых: {missing_count} ({missing_pct:.1f}%)")
            else:
                profile_lines.append(f"  ✅ Пустых: 0 (полностью заполнена)")
            
            profile_lines.append(f"  📊 Уникальных значений: {unique_count}")
            
            # Примеры значений
            sample_vals = col_data.dropna().head(5).tolist()
            if sample_vals:
                sample_str = ", ".join([str(v)[:30] for v in sample_vals])
                profile_lines.append(f"  📝 Примеры: {sample_str}")
            
            # Для числовых - статистика
            if 'numeric' in dtype and non_null_count > 0:
                try:
                    profile_lines.append(f"  📈 Мин: {col_data.min():.2f}, Макс: {col_data.max():.2f}, Среднее: {col_data.mean():.2f}")
                except:
                    pass
        
        # Поиск потенциальных проблем
        profile_lines.append(f"\n{'─' * 60}")
        profile_lines.append("🔍 ОБНАРУЖЕННЫЕ ОСОБЕННОСТИ ДАННЫХ:")
        profile_lines.append(f"{'─' * 60}")
        
        problems_found = False
        
        # Колонки с большим количеством пустых значений
        for col in df.columns:
            missing_pct = (df[col].isna().sum() / len(df) * 100) if len(df) > 0 else 0
            if missing_pct > 0:
                problems_found = True
                if missing_pct > 50:
                    profile_lines.append(f"⚠️ '{col}' - {missing_pct:.1f}% пустых (большая часть данных отсутствует)")
                elif missing_pct > 10:
                    profile_lines.append(f"ℹ️ '{col}' - {missing_pct:.1f}% пустых")
                else:
                    profile_lines.append(f"📌 '{col}' - {df[col].isna().sum()} пустых значений ({missing_pct:.1f}%)")
        
        # Проверка на дубликаты
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            problems_found = True
            profile_lines.append(f"⚠️ Найдено {dup_count} дубликатов строк")
        
        if not problems_found:
            profile_lines.append("✅ Данные выглядят чистыми, явных проблем не обнаружено")
        
        # Примеры строк с пустыми значениями
        rows_with_na = df[df.isna().any(axis=1)]
        if len(rows_with_na) > 0:
            profile_lines.append(f"\n{'─' * 60}")
            profile_lines.append(f"📋 ПРИМЕРЫ СТРОК С ПУСТЫМИ ЗНАЧЕНИЯМИ ({min(3, len(rows_with_na))} из {len(rows_with_na)}):")
            profile_lines.append(f"{'─' * 60}")
            for idx, row in rows_with_na.head(3).iterrows():
                na_cols = [col for col in df.columns if pd.isna(row[col])]
                profile_lines.append(f"  Строка {idx}: пустые в колонках [{', '.join(na_cols)}]")
                # Показываем несколько значений из этой строки
                non_na_vals = [(col, row[col]) for col in df.columns[:4] if pd.notna(row[col])]
                if non_na_vals:
                    vals_str = ", ".join([f"{col}={val}" for col, val in non_na_vals])
                    profile_lines.append(f"    Данные: {vals_str}...")
        
        # Примеры полных строк (без пустых)
        complete_rows = df.dropna()
        if len(complete_rows) > 0:
            profile_lines.append(f"\n{'─' * 60}")
            profile_lines.append(f"✅ ПРИМЕРЫ ПОЛНЫХ СТРОК (без пустых):")
            profile_lines.append(f"{'─' * 60}")
            for idx, row in complete_rows.head(2).iterrows():
                vals = [(col, row[col]) for col in df.columns[:5]]
                vals_str = ", ".join([f"{col}={val}" for col, val in vals])
                profile_lines.append(f"  Строка {idx}: {vals_str}...")
        
        profile_lines.append(f"\n{'=' * 60}")
        
        return "\n".join(profile_lines)

    def df_to_csv_base64(self, df: pd.DataFrame = None) -> str:
        """
        Конвертировать DataFrame в base64 CSV
        
        Используется разделитель ';' для совместимости с Windows Excel
        (в европейских/русских локалях Excel по умолчанию ожидает ';')

        Args:
            df: DataFrame для конвертации (по умолчанию current_df)

        Returns:
            Base64 строка CSV файла с разделителем ';'
        """
        if df is None:
            df = self.current_df
        
        if df is None:
            return None
        
        csv_buffer = io.StringIO()
        # sep=';' для Windows Excel, encoding='utf-8-sig' добавляет BOM для корректного отображения
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig', sep=';')
        csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')
        return base64.b64encode(csv_bytes).decode('utf-8')

    def execute_python_code(self, code: str, df: pd.DataFrame) -> Tuple[bool, Any, str, List[str], Optional[pd.DataFrame]]:
        """
        Безопасное выполнение Python кода с возвращением изображений в base64
        и изменённого DataFrame

        Args:
            code: Python код для выполнения
            df: DataFrame для работы

        Returns:
            Кортеж (успех, результат, вывод/ошибка, список base64 изображений, изменённый DataFrame)
        """
        local_vars = {
            'df': df.copy(),
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'result': None,
            'modified_df': None  # Для возврата изменённого DataFrame
        }

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        plot_base64_list = []
        modified_df = None

        try:
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):

                # Выполняем код
                exec(code, local_vars)

                # Получаем результат
                result = local_vars.get('result', None)
                output = stdout_capture.getvalue()
                
                # Проверяем, был ли изменён DataFrame
                modified_df = local_vars.get('modified_df', None)
                
                # Если modified_df не установлен явно, но df был изменён
                if modified_df is None and 'df' in local_vars:
                    # Проверяем, изменился ли df
                    current_df = local_vars['df']
                    if not current_df.equals(df):
                        modified_df = current_df

                # Конвертируем результат в JSON-serializable формат
                if isinstance(result, (np.integer, np.floating)):
                    result = float(result)
                elif isinstance(result, np.ndarray):
                    result = result.tolist()
                elif isinstance(result, pd.DataFrame) or isinstance(result, pd.Series):
                    result = str(result)

                # Сохраняем графики в base64
                if plt.get_fignums():
                    for fig_num in plt.get_fignums():
                        fig = plt.figure(fig_num)

                        buffer = io.BytesIO()
                        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
                        buffer.seek(0)

                        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                        plot_base64_list.append(f"data:image/png;base64,{img_base64}")

                        buffer.close()

                    plt.close('all')

                return True, result, output, plot_base64_list, modified_df

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return False, None, error_msg, [], None
        finally:
            plt.close('all')
            plt.clf()
            local_vars.clear()

    def auto_clean_data(self) -> Dict[str, Any]:
        """
        Автоматическая очистка данных без запроса пользователя
        Вызывается когда загружен файл, но нет текстового запроса
        
        Returns:
            Результат очистки с изменённым CSV
        """
        if self.current_df is None:
            return {
                "success": False,
                "error": "CSV файл не загружен"
            }
        
        df = self.current_df.copy()
        cleaning_steps = []
        original_shape = df.shape
        
        # ШАГ 1: Определяем реальные заголовки
        # Проверяем первые строки на предмет заголовков
        rows_to_skip = 0
        
        # Проверяем первые 5 строк
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            # Если строка содержит много пустых значений - возможно это мусор
            nan_count = row.isna().sum()
            if nan_count > len(row) * 0.8:  # Более 80% пустых
                rows_to_skip = i + 1
                cleaning_steps.append(f"🗑️ Обнаружена пустая строка #{i+1}")
        
        if rows_to_skip > 0:
            df = df.iloc[rows_to_skip:].reset_index(drop=True)
            cleaning_steps.append(f"✅ Удалено {rows_to_skip} пустых строк в начале")
        
        # ШАГ 2: Проверяем первую строку после очистки - может это заголовки?
        if len(df) > 1:
            first_row = df.iloc[0]
            # Проверяем, похожа ли первая строка на заголовки
            string_count = sum(1 for val in first_row if isinstance(val, str) and len(str(val).strip()) > 0)
            
            # Если все колонки - Unnamed и первая строка содержит текст
            unnamed_count = sum(1 for col in df.columns if 'Unnamed' in str(col))
            
            if unnamed_count > len(df.columns) * 0.5 and string_count > len(first_row) * 0.3:
                # Делаем первую строку заголовками
                new_columns = [str(val).strip() if pd.notna(val) else f'Column_{i}' 
                              for i, val in enumerate(first_row)]
                df.columns = new_columns
                df = df.iloc[1:].reset_index(drop=True)
                cleaning_steps.append("🎯 Первая строка преобразована в заголовки")
        
        # ШАГ 3: Удаляем полностью пустые колонки
        empty_cols = [col for col in df.columns 
                     if df[col].isna().all() or (df[col].astype(str).str.strip() == '').all()]
        if empty_cols:
            df = df.drop(columns=empty_cols)
            cleaning_steps.append(f"🗑️ Удалено {len(empty_cols)} пустых колонок")
        
        # ШАГ 4: Удаляем Unnamed колонки если они пустые или с индексами
        unnamed_to_drop = []
        for col in df.columns:
            if 'Unnamed' in str(col):
                col_values = df[col].dropna()
                # Если колонка пустая или содержит только числа (возможно индексы)
                if len(col_values) == 0:
                    unnamed_to_drop.append(col)
                else:
                    # Проверяем, это просто числа (индексы)?
                    try:
                        numeric_values = pd.to_numeric(col_values, errors='coerce')
                        if numeric_values.notna().all():
                            # Проверяем последовательность
                            if (numeric_values.diff().dropna() == 1).all():
                                unnamed_to_drop.append(col)
                    except:
                        pass
        
        if unnamed_to_drop:
            df = df.drop(columns=unnamed_to_drop)
            cleaning_steps.append(f"🗑️ Удалено {len(unnamed_to_drop)} служебных Unnamed колонок")
        
        # ШАГ 5: Удаляем полностью пустые строки
        rows_before = len(df)
        df = df.dropna(how='all')
        rows_removed = rows_before - len(df)
        if rows_removed > 0:
            cleaning_steps.append(f"🗑️ Удалено {rows_removed} пустых строк")
        
        # ШАГ 6: Очищаем названия колонок
        df.columns = [str(col).strip() for col in df.columns]
        
        # ШАГ 7: Преобразуем типы данных
        for col in df.columns:
            # Пробуем преобразовать в числа
            try:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                if numeric_col.notna().sum() > len(df) * 0.5:  # Более 50% успешно преобразовано
                    df[col] = numeric_col
            except:
                pass
            
            # Пробуем преобразовать в даты
            if df[col].dtype == 'object':
                try:
                    date_col = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    if date_col.notna().sum() > len(df) * 0.5:
                        df[col] = date_col
                        cleaning_steps.append(f"📅 Колонка '{col}' преобразована в даты")
                except:
                    pass
        
        # Сохраняем очищенный DataFrame
        df = df.reset_index(drop=True)
        self.current_df = df
        self.data_metadata["was_edited"] = True
        
        final_shape = df.shape
        
        # Формируем сообщение для пользователя
        summary = f"""## 🧹 Автоматическая очистка данных

### 📊 Исходные данные
- **Файл:** {self.current_filename}
- **Размер:** {original_shape[0]} строк × {original_shape[1]} колонок

### ✅ Выполненные шаги очистки
"""
        for step in cleaning_steps:
            summary += f"- {step}\n"
        
        if not cleaning_steps:
            summary += "- Данные уже чистые, изменения не требуются\n"
        
        summary += f"""
### 📈 Результат
- **Размер после очистки:** {final_shape[0]} строк × {final_shape[1]} колонок
- **Колонки:** {', '.join(df.columns.tolist())}

### 📋 Первые строки очищенных данных
"""
        # Добавляем preview таблицы
        preview_df = df.head(5)
        summary += self._df_to_markdown(preview_df)
        
        summary += """

Таблица очищена и готова к анализу! 
Вы можете:
- Задать вопрос о данных
- Попросить построить график
- Запросить статистику
- Попросить изменить данные (добавить/удалить строки или столбцы)
"""
        
        return {
            "success": True,
            "query": "[Автоматическая очистка]",
            "code_attempts": [],
            "final_code": "# Автоматическая очистка данных",
            "result_data": None,
            "text_output": summary,
            "plots": [],
            "error": None,
            "attempts_count": 1,
            "timestamp": datetime.utcnow().isoformat(),
            "load_info": self.data_metadata,
            "modified_csv": self.df_to_csv_base64(df),
            "was_modified": True,
            "cleaning_steps": cleaning_steps
        }
    
    def _df_to_markdown(self, df: pd.DataFrame, max_rows: int = 10) -> str:
        """
        Конвертировать DataFrame в Markdown таблицу
        """
        if df is None or len(df) == 0:
            return "*(пустая таблица)*"
        
        display_df = df.head(max_rows)
        
        # Заголовки
        headers = list(display_df.columns)
        md = "| " + " | ".join(str(h) for h in headers) + " |\n"
        md += "|" + "|".join(["---"] * len(headers)) + "|\n"
        
        # Строки
        for _, row in display_df.iterrows():
            values = []
            for val in row:
                if pd.isna(val):
                    values.append("")
                elif isinstance(val, float):
                    values.append(f"{val:,.2f}")
                else:
                    values.append(str(val))
            md += "| " + " | ".join(values) + " |\n"
        
        if len(df) > max_rows:
            md += f"\n*...и ещё {len(df) - max_rows} строк*\n"
        
        return md

    def generate_code_with_retry(self, user_query: str, schema: Dict,
                                 chat_history: List[Dict] = None,
                                 previous_error: Optional[str] = None) -> str:
        """
        Генерация Python кода с помощью AI (Julius.ai style - многоэтапный подход)
        Поддерживает редактирование данных

        Args:
            user_query: Запрос пользователя
            schema: Схема данных CSV
            chat_history: История предыдущих сообщений
            previous_error: Предыдущая ошибка (для повторной попытки)

        Returns:
            Сгенерированный Python код
        """
        system_prompt = """Ты эксперт-аналитик данных, работающий как Julius.ai.

🧠 ГЛАВНОЕ ПРАВИЛО: СНАЧАЛА ДУМАЙ, ПОТОМ ДЕЙСТВУЙ!

Перед выполнением ЛЮБОГО запроса ты ОБЯЗАН:
1. ИЗУЧИТЬ данные (смотри профиль данных внимательно)
2. ПОНЯТЬ что именно хочет пользователь
3. ПРОВЕРИТЬ есть ли нужные данные/колонки
4. ВЫПОЛНИТЬ задачу с учётом реального состояния данных

═══════════════════════════════════════════════════════════
📋 ОБЯЗАТЕЛЬНАЯ СТРУКТУРА КОДА:
═══════════════════════════════════════════════════════════

```python
# === ШАГ 1: ИЗУЧЕНИЕ ДАННЫХ ===
print("🔍 ШАГ 1: Изучаю данные...")
print(f"Размер: {len(df)} строк, {len(df.columns)} колонок")

# ВСЕГДА проверяй пустые значения ПЕРЕД любой операцией!
missing_info = df.isna().sum()
cols_with_missing = missing_info[missing_info > 0]
if len(cols_with_missing) > 0:
    print(f"⚠️ Найдены пустые значения:")
    for col, count in cols_with_missing.items():
        print(f"   • {col}: {count} пустых ({count/len(df)*100:.1f}%)")
else:
    print("✅ Пустых значений нет")

# Строки с любыми пустыми значениями
rows_with_na = df[df.isna().any(axis=1)]
print(f"📊 Строк с пустыми значениями: {len(rows_with_na)} из {len(df)}")

# === ШАГ 2: ПОНИМАНИЕ ЗАПРОСА ===
print("\\n🎯 ШАГ 2: Анализирую запрос пользователя...")
# Объясни что понял из запроса

# === ШАГ 3: ВЫПОЛНЕНИЕ ===
print("\\n⚙️ ШАГ 3: Выполняю...")

# Для РЕДАКТИРОВАНИЯ данных:
# df = df.dropna()  # Удалить ВСЕ строки с любыми пустыми значениями
# df = df.dropna(subset=['col1', 'col2'])  # Удалить строки с пустыми в конкретных колонках
# df = df.drop(columns=['col'])  # Удалить колонку
# df['new'] = ...  # Добавить колонку
# df = df[df['col'] > 100]  # Фильтрация

# ВАЖНО! После изменения данных:
# modified_df = df.copy()

# === ШАГ 4: РЕЗУЛЬТАТ ===
print("\\n✅ ШАГ 4: Готово!")

result = f\"\"\"
## 📊 Результат

Описание что было сделано...

| Показатель | Значение |
|------------|----------|
| До | {before} |
| После | {after} |
\"\"\"

# Если данные изменены - ОБЯЗАТЕЛЬНО:
# modified_df = df.copy()
```

═══════════════════════════════════════════════════════════
🔧 ТИПИЧНЫЕ ОПЕРАЦИИ С ДАННЫМИ:
═══════════════════════════════════════════════════════════

📌 УДАЛЕНИЕ СТРОК С ПУСТЫМИ ЗНАЧЕНИЯМИ:
```python
# Показать текущее состояние
rows_before = len(df)
rows_with_na = df[df.isna().any(axis=1)]
print(f"Строк с пустыми: {len(rows_with_na)}")

# Удалить ВСЕ строки где есть хотя бы одно пустое значение
df = df.dropna()

# ИЛИ удалить только где пустые в конкретных колонках
# df = df.dropna(subset=['Column1', 'Column2'])

rows_after = len(df)
print(f"Удалено строк: {rows_before - rows_after}")
modified_df = df.copy()
```

📌 УДАЛЕНИЕ КОЛОНОК:
```python
df = df.drop(columns=['Column_Name'])
modified_df = df.copy()
```

📌 ФИЛЬТРАЦИЯ:
```python
df = df[df['Column'] > 100]
df = df[df['Column'].notna()]  # Только непустые
modified_df = df.copy()
```

📌 ЗАПОЛНЕНИЕ ПУСТЫХ:
```python
df['Column'] = df['Column'].fillna(0)  # Заполнить нулями
df['Column'] = df['Column'].fillna(df['Column'].mean())  # Средним
modified_df = df.copy()
```

═══════════════════════════════════════════════════════════
⚠️ КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:
═══════════════════════════════════════════════════════════

1. **ВСЕГДА ПРОВЕРЯЙ df.isna().sum()** перед ответом о пустых данных!

2. **ИСПОЛЬЗУЙ ДАННЫЕ ИЗ ПРОФИЛЯ** - там уже есть вся информация о пустых значениях

3. **НЕ ОТВЕЧАЙ "пустых нет"** пока не проверил! Смотри профиль данных!

4. **modified_df = df.copy()** - ОБЯЗАТЕЛЬНО после любого изменения!

5. **ЛОГИРУЙ ВСЁ** через print() - пользователь должен видеть каждый шаг

6. **ФОРМАТИРУЙ ЧИСЛА**: {value:,.0f} или {value:,.2f}

7. **result = строка с Markdown** - заголовки ##, таблицы, эмодзи

8. **ГИБКИЙ ПОИСК КОЛОНОК** - ищи по ключевым словам, не точное совпадение
"""

        # Формируем детальное описание данных с глубоким профилем
        deep_profile = self.get_deep_data_profile(self.current_df)
        
        column_details = []
        for col in schema['columns']:
            dtype = schema['dtypes'][col]
            missing = schema['missing_values'].get(col, 0)

            examples = []
            if len(schema['sample_data']) > 0:
                for row in schema['sample_data'][:3]:
                    val = row.get(col)
                    if pd.notna(val):
                        examples.append(str(val))

            examples_str = ", ".join(examples[:3]) if examples else "нет данных"

            col_info = f"  • '{col}' ({dtype})"
            if missing > 0:
                col_info += f" [⚠️ пустых: {missing}]"
            col_info += f"\n    Примеры: {examples_str}"
            column_details.append(col_info)

        user_message = f"""
{deep_profile}

═══════════════════════════════════════════════════════════
🎯 ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {user_query}
═══════════════════════════════════════════════════════════

⚡ НАПОМИНАНИЕ:
- Внимательно изучи ПРОФИЛЬ ДАННЫХ выше!
- Там указаны ВСЕ пустые значения по каждой колонке!
- СНАЧАЛА проверь данные, ПОТОМ действуй!
- Если редактируешь данные - установи modified_df = df.copy()
"""

        if self.data_metadata.get("first_row_is_header"):
            user_message += "\n\n✅ ПРИМЕЧАНИЕ: Первая строка CSV была автоматически преобразована в заголовки."

        # Добавляем историю если есть
        if chat_history and len(chat_history) > 0:
            history_text = "\n\nИстория предыдущих запросов:\n"
            for i, item in enumerate(chat_history[-5:], 1):
                history_text += f"\n{i}. Запрос: {item.get('query', '')}\n"
                if item.get('success'):
                    history_text += f"   Результат: {item.get('text_output', '')[:200]}\n"
            user_message += history_text

        if previous_error:
            user_message += f"""

ПРЕДЫДУЩАЯ ПОПЫТКА ЗАВЕРШИЛАСЬ ОШИБКОЙ:
{previous_error}

Исправь код, учитывая эту ошибку.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=4000
            )

            code = response.choices[0].message.content.strip()

            # Убираем markdown разметку если есть
            if code.startswith("```python"):
                code = code[9:]
            if code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]

            return code.strip()

        except Exception as e:
            error_msg = str(e)

            if "401" in error_msg or "Unauthorized" in error_msg or "User not found" in error_msg:
                raise Exception(
                    f"Ошибка аутентификации OpenRouter (401): API ключ неверный или истек. "
                    f"Проверьте OPENROUTER_API_KEY в .env файле. "
                    f"Получите новый ключ на https://openrouter.ai/keys. "
                    f"Детали: {error_msg}"
                )
            elif "403" in error_msg:
                raise Exception(
                    f"Доступ запрещен (403): У API ключа нет доступа к модели {self.model} "
                    f"или недостаточно кредитов. Детали: {error_msg}"
                )
            elif "429" in error_msg:
                raise Exception(
                    f"Превышен лимит запросов (429): Слишком много запросов к API. "
                    f"Подождите немного и попробуйте снова. Детали: {error_msg}"
                )
            else:
                raise Exception(f"Ошибка при генерации кода: {error_msg}")

    def analyze(self, user_query: str = None, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Основной метод анализа для API
        Если user_query пустой - выполняет автоматическую очистку

        Args:
            user_query: Запрос пользователя (если пустой - автоочистка)
            chat_history: История предыдущих сообщений

        Returns:
            Словарь с результатами в формате API
        """
        if self.current_df is None:
            return {
                "success": False,
                "error": "CSV файл не загружен",
                "timestamp": datetime.utcnow().isoformat()
            }

        # Если запрос пустой или отсутствует - автоматическая очистка
        if not user_query or user_query.strip() == "":
            return self.auto_clean_data()

        # Получаем схему данных
        schema = self.analyze_csv_schema(self.current_df)

        result = {
            "success": False,
            "query": user_query,
            "code_attempts": [],
            "final_code": None,
            "result_data": None,
            "text_output": None,
            "plots": [],
            "error": None,
            "attempts_count": 0,
            "timestamp": datetime.utcnow().isoformat(),
            "load_info": self.data_metadata,
            "modified_csv": None,
            "was_modified": False
        }

        previous_error = None

        for attempt in range(self.max_retries):
            result["attempts_count"] = attempt + 1

            try:
                code = self.generate_code_with_retry(
                    user_query,
                    schema,
                    chat_history,
                    previous_error
                )

                result["code_attempts"].append({
                    "attempt": attempt + 1,
                    "code": code,
                    "success": False
                })

            except Exception as e:
                result["error"] = f"Ошибка генерации кода: {str(e)}"
                break

            # Выполняем код
            success, exec_result, output, plot_base64_list, modified_df = self.execute_python_code(
                code, self.current_df
            )

            if success:
                result["success"] = True
                result["final_code"] = code
                result["result_data"] = exec_result
                result["text_output"] = output
                result["plots"] = plot_base64_list
                result["code_attempts"][-1]["success"] = True
                
                # Если данные были изменены
                if modified_df is not None:
                    self.current_df = modified_df
                    self.data_metadata["was_edited"] = True
                    result["modified_csv"] = self.df_to_csv_base64(modified_df)
                    result["was_modified"] = True
                
                break
            else:
                previous_error = output
                result["code_attempts"][-1]["error"] = output

                if attempt == self.max_retries - 1:
                    result["error"] = f"Не удалось выполнить код после {self.max_retries} попыток"
                    result["error_details"] = output

        return result

    def get_schema_info(self) -> Dict[str, Any]:
        """
        Получить информацию о текущем CSV файле

        Returns:
            Информация о схеме данных
        """
        if self.current_df is None:
            return {
                "success": False,
                "error": "CSV файл не загружен"
            }

        schema = self.analyze_csv_schema(self.current_df)
        return {
            "success": True,
            "schema": schema,
            "timestamp": datetime.utcnow().isoformat()
        }

    def get_current_csv(self) -> Optional[str]:
        """
        Получить текущий CSV в base64

        Returns:
            Base64 строка CSV или None
        """
        return self.df_to_csv_base64()

    def cleanup(self):
        """
        Очистка памяти после использования агента
        """
        if self.current_df is not None:
            del self.current_df
            self.current_df = None

        if self.original_df is not None:
            del self.original_df
            self.original_df = None

        plt.close('all')
        gc.collect()
