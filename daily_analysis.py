"""
ПОЛНЫЙ АНАЛИЗ АКЦИЙ ДЛЯ GITHUB ACTIONS
Полностью идентичная версия Google Colab
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, date, timedelta
import lightgbm
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
import warnings
import xml.etree.ElementTree as ET
import time
import random

# ========== КОНФИГУРАЦИЯ И ПРЕДУПРЕЖДЕНИЯ ==========
def warn(*args, **kwargs):
    pass

warnings.warn = warn
warnings.filterwarnings('ignore')

# ========== ДИАГНОСТИЧЕСКИЕ ФУНКЦИИ ==========
def debug_print(message):
    """Вывод отладочной информации"""
    print(f"🔍 [DEBUG] {message}")

def save_debug_data(data, filename):
    """Сохранение отладочных данных"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    debug_print(f"Сохранены отладочные данные в {filename}")

# ========== ФУНКЦИИ ДЛЯ ЗАГРУЗКИ ДАННЫХ MOEX (ТОЧНО КАК В GOOGLE COLAB) ==========
def fetch_shares_data_colab(ticker, start_date, end_date, is_index=False, retry_count=3):
    """
    Загружает данные MOEX точно как в Google Colab: два запроса с фиксированной датой 2024-09-25
    Добавлены повторные попытки и User-Agent для обхода блокировок
    """
    debug_print(f"  📥 Загрузка {ticker} (метод Colab)...")
    
    # Определяем базовый URL
    if is_index:
        base_url = "https://iss.moex.com/iss/engines/stock/markets/index/securities"
    else:
        base_url = "https://iss.moex.com/iss/engines/stock/markets/shares/securities"
    
    # Фиксированная дата разбивки как в Google Colab
    split_date = '2024-09-25'
    
    all_data = []
    
    # Добавляем User-Agent и заголовки для обхода блокировок
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
        'Connection': 'keep-alive',
    }
    
    # Первый запрос: от start_date до 2024-09-25
    url1 = f"{base_url}/{ticker}/candles.json?from={start_date}&till={split_date}&interval=24"
    debug_print(f"    🔄 Запрос 1: {start_date} - {split_date}")
    
    for attempt in range(retry_count):
        try:
            response1 = requests.get(url1, timeout=60, headers=headers)  # Увеличиваем таймаут до 60 секунд
            if response1.status_code == 200:
                data1 = response1.json()
                if 'candles' in data1 and 'data' in data1['candles'] and data1['candles']['data']:
                    part_data1 = [{k: r[i] for i, k in enumerate(data1['candles']['columns'])} 
                                for r in data1['candles']['data']]
                    df1 = pd.DataFrame(part_data1)
                    all_data.append(df1)
                    debug_print(f"      ✅ Загружено {len(df1)} строк")
                    break  # Успешно загрузили, выходим из цикла попыток
                else:
                    debug_print(f"      ⚠️ Нет данных в первой части")
                    break
            else:
                debug_print(f"      ⚠️ Ошибка https {response1.status_code} в первой части (попытка {attempt+1}/{retry_count})")
                if attempt < retry_count - 1:
                    time.sleep(random.uniform(2, 5))  # Случайная задержка между попытками
        except Exception as e:
            debug_print(f"      ❌ Ошибка запроса 1 (попытка {attempt+1}/{retry_count}): {e}")
            if attempt < retry_count - 1:
                time.sleep(random.uniform(3, 7))  # Увеличиваем задержку между попытками
    
    # Добавляем задержку между запросами для снижения нагрузки
    time.sleep(random.uniform(1, 3))
    
    # Второй запрос: от 2024-09-25 до end_date
    url2 = f"{base_url}/{ticker}/candles.json?from={split_date}&till={end_date}&interval=24"
    debug_print(f"    🔄 Запрос 2: {split_date} - {end_date}")
    
    for attempt in range(retry_count):
        try:
            response2 = requests.get(url2, timeout=60, headers=headers)
            if response2.status_code == 200:
                data2 = response2.json()
                if 'candles' in data2 and 'data' in data2['candles'] and data2['candles']['data']:
                    part_data2 = [{k: r[i] for i, k in enumerate(data2['candles']['columns'])} 
                                for r in data2['candles']['data']]
                    df2 = pd.DataFrame(part_data2)
                    all_data.append(df2)
                    debug_print(f"      ✅ Загружено {len(df2)} строк")
                    break
                else:
                    debug_print(f"      ⚠️ Нет данных во второй части")
                    break
            else:
                debug_print(f"      ⚠️ Ошибка https {response2.status_code} во второй части (попытка {attempt+1}/{retry_count})")
                if attempt < retry_count - 1:
                    time.sleep(random.uniform(2, 5))
        except Exception as e:
            debug_print(f"      ❌ Ошибка запроса 2 (попытка {attempt+1}/{retry_count}): {e}")
            if attempt < retry_count - 1:
                time.sleep(random.uniform(3, 7))
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result['begin'] = pd.to_datetime(result['begin'])
        result = result.sort_values('begin').drop_duplicates('begin')
        debug_print(f"  📊 Итого по {ticker}: {len(result)} строк, последняя дата: {result['begin'].max().date() if not result.empty else 'нет'}")
        return result
    else:
        debug_print(f"  ❌ Нет данных по {ticker}")
        return pd.DataFrame()

def fetch_futures_data_colab(ticker, start_date, end_date, retry_count=3):
    """
    Загружает данные фьючерсов одним запросом как в Google Colab
    """
    debug_print(f"  📥 Загрузка фьючерса {ticker} (метод Colab)...")
    
    url = f"https://iss.moex.com/iss/engines/futures/markets/forts/securities/{ticker}/candles.json?from={start_date}&till={end_date}&interval=24"
    
    # Добавляем User-Agent и заголовки для обхода блокировок
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
        'Connection': 'keep-alive',
    }
    
    for attempt in range(retry_count):
        try:
            response = requests.get(url, timeout=60, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if 'candles' in data and 'data' in data['candles'] and data['candles']['data']:
                    part_data = [{k: r[i] for i, k in enumerate(data['candles']['columns'])} 
                               for r in data['candles']['data']]
                    df = pd.DataFrame(part_data)
                    df['begin'] = pd.to_datetime(df['begin'])
                    df = df.sort_values('begin').drop_duplicates('begin')
                    debug_print(f"  📊 Итого по {ticker}: {len(df)} строк, от {df['begin'].min().date() if not df.empty else 'нет'} до {df['begin'].max().date() if not df.empty else 'нет'}")
                    return df
                else:
                    debug_print(f"  ⚠️ Нет данных по фьючерсу {ticker}")
                    return pd.DataFrame()
            else:
                debug_print(f"  ⚠️ Ошибка https {response.status_code} для {ticker} (попытка {attempt+1}/{retry_count})")
                if attempt < retry_count - 1:
                    time.sleep(random.uniform(2, 5))
        except Exception as e:
            debug_print(f"  ❌ Ошибка запроса {ticker} (попытка {attempt+1}/{retry_count}): {e}")
            if attempt < retry_count - 1:
                time.sleep(random.uniform(3, 7))
    
    return pd.DataFrame()

# ========== ФУНКЦИЯ ДОБАВЛЕНИЯ КАПИТАЛИЗАЦИЙ ==========
def add_cap_columns(df, cap_dict):
    """Добавляет колонки с капитализацией для расчета индексов"""
    for key, values in cap_dict.items():
        column_name = f'cap{key}'
        df[column_name] = df['begin'].dt.year.map({
            2023: values[0],
            2024: values[1],
            2025: values[2],
            2026: values[3]
        })
    return df

# ========== ФУНКЦИЯ ДЛЯ ЗАГРУЗКИ USD ==========
def get_currency_rate(date, retry_count=3):
    """Получает курс доллара на указанную дату"""
    url = f'httpss://www.cbr.ru/scripts/XML_daily.asp?date_req={date.strftime("%d/%m/%Y")}'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/xml',
        'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
    }
    
    for attempt in range(retry_count):
        try:
            response = requests.get(url, timeout=30, headers=headers)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                for valute in root.findall('Valute'):
                    char_code = valute.find('CharCode').text
                    if char_code == 'USD':
                        value = valute.find('Value').text
                        return float(value.replace(',', '.'))
                debug_print(f"⚠️ Курс USD на {date} не найден в XML")
                return None
        except Exception as e:
            debug_print(f"⚠️ Ошибка при запросе курса на {date.strftime('%d/%m/%Y')} (попытка {attempt+1}): {e}")
            if attempt < retry_count - 1:
                time.sleep(random.uniform(1, 3))
    
    debug_print(f"❌ Не удалось получить курс на {date} после {retry_count} попыток")
    return None

# ========== ФУНКЦИЯ ДЛЯ ЗАГРУЗКИ BRENT (ТОЧНО КАК В GOOGLE COLAB) ==========
def fetch_brent_data_colab(start_date, end_date):
    """Загружает данные Brent точно как в Google Colab"""
    debug_print("🛢️ Загружаю данные по Brent (метод Colab)...")
    
    # Определяем нужные контракты на каждый год с 2023 по 2026
    brent_contracts = ['BRU3', 'BRU4', 'BRU5', 'BRJ6']
    
    bre_all = []
    loaded_contracts = []
    
    for contract in brent_contracts:
        debug_print(f"  🔄 Загружаю контракт {contract}...")
        
        df = fetch_futures_data_colab(contract, start_date, end_date, retry_count=3)
        
        if not df.empty:
            df = df[['begin', 'close']].rename(columns={'close': 'brent'})
            bre_all.append(df)
            loaded_contracts.append(contract)
            debug_print(f"    ✅ Загружено {len(df)} строк")
            
            # Логируем диапазон дат для этого контракта
            if not df.empty:
                debug_print(f"    📅 {contract}: {df['begin'].min().date()} - {df['begin'].max().date()}")
        else:
            debug_print(f"    ⚠️ Нет данных по контракту {contract}")
        
        # Задержка между загрузкой контрактов
        time.sleep(random.uniform(2, 4))
    
    # Объединяем все данные
    if bre_all:
        bre = pd.concat(bre_all, ignore_index=True)
        bre = bre[['begin', 'brent']]
        
        # Сортируем по дате и удаляем дубликаты
        bre = bre.sort_values('begin').drop_duplicates(subset='begin', keep='last')
        
        # Выводим информацию о диапазоне дат
        if not bre.empty:
            debug_print(f"📊 Всего данных Brent: {len(bre)} строк")
            debug_print(f"📅 Диапазон данных Brent: {bre['begin'].min().date()} - {bre['begin'].max().date()}")
            
            # Проверяем наличие пропусков
            date_range = pd.date_range(start=bre['begin'].min(), end=bre['begin'].max())
            missing_dates = [d for d in date_range if d not in bre['begin'].values]
            if missing_dates:
                debug_print(f"⚠️ Обнаружены пропуски в данных Brent: {len(missing_dates)} дней")
        else:
            debug_print("⚠️ Данные Brent пусты после объединения")
            bre = pd.DataFrame(columns=['begin', 'brent'])
        
        debug_print(f"✅ Успешно загружены контракты: {', '.join(loaded_contracts)}")
        return bre
    else:
        debug_print("❌ Не удалось загрузить данные по Brent ни из одного контракта")
        return pd.DataFrame(columns=['begin', 'brent'])

# ========== ОСНОВНОЙ КОД ==========
def main_pipeline(forecast_days=14):
    """
    Основная функция для запуска всего пайплайна
    Точная копия логики Google Colab
    """
    debug_print(f"🚀 Начинаю полный анализ (метод Google Colab)...")
    
    # === КОНФИГУРАЦИЯ КАПИТАЛИЗАЦИЙ ===
    cap = {
        'IMOEX2': {
            'VTBR': [657.8, 505.5, 503.8, 471.3],
            'SBER': [6600, 7000, 6900, 6900],
            'GAZP': [3700, 3600, 3100, 2900],
            'LKOH': [5200, 5100, 4200, 3700],
            'T': [609.3, 975.3, 862.8, 864.6],
            'GMKN': [2400, 1900, 1600, 2400],
            'PIKK': [555.7, 287.6, 322, 299.9]
        },
        'MOEXOG': {
            'GAZP': [3700, 3600, 3100, 2900],
            'NVTK': [4400, 3600, 3500, 3500],
            'ROSN': [6200, 5700, 4500, 4200],
            'TATN': [1700, 1600, 1400, 1200],
            'LKOH': [5200, 5100, 4200, 3700]
        },
        'MOEXMM': {
            'GMKN': [2400, 1900, 1600, 2400],
            'MAGN': [657.1, 450.8, 367.3, 307.2],
            'NLMK': [1400, 793.5, 669, 623.2],
            'PLZL': [1100, 1900, 2600, 3500],
            'MTLR': [174.7, 63.5, 42.4, 40]
        },
        'MOEXIT': {
            'YDEX': [1500, 1800, 1800, 1800],
            'VKCO': [142.3, 70, 100, 168.4],
            'ASTR': [144.9, 90.3, 79.5, 51.7],
            'POSI': [183.2, 92.6, 88.8, 71.8],
            'SOFL': [46.6, 44.1, 40.1, 31.3]
        },
        'MOEXFN': {
            'VTBR': [657.8, 505.5, 503.8, 471.3],
            'SBER': [6600, 7000, 6900, 6900],
            'T': [609.3, 975.3, 862.8, 864.6],
            'SVCB': [392.8, 355.9, 334.5, 277.9],
            'BSPB': [140.9, 189.5, 166.3, 135.6]
        },
        'MOEXEU': {
            'IRAO': [418.9, 392.7, 330.8, 333.9],
            'UPRO': [145.2, 144.9, 95.1, 91.3],
            'FEES': [245.1, 157.3, 141.2, 162.6],
            'MSNG': [67.9, 74.9, 81.1, 86.3],
            'HYDR': [339.7, 226, 200.2, 185.9]
        },
        'MOEXRE': {
            'PIKK': [555.7, 287.6, 322, 299.9],
            'SMLT': [235.5, 79.3, 74.8, 57.3],
            'LSRG': [95.2, 88.9, 76.7, 69],
            'MSTT': [61.9, 37.2, 33, 31]
        },
        'MOEXTN': {
            'AFLT': [154.7, 299.4, 251.5, 222.9],
            'FLOT': [318.2, 221.6, 190.6, 175.7],
            'SVAV': [29.8, 22.2, 18.6, 18.2],
            'FESH': [256.6, 157.7, 156.6, 155.2],
            'KMAZ': [129.8, 79.9, 61.4, 59.3]
        },
        'MOEXCN': {
            'MVID': [38.7, 18.7, 16.2, 13.8],
            'LENT': [87.6, 163.6, 185.2, 231.3],
            'MGNT': [861, 466.7, 370.1, 307.8],
            'PHOR': [855.1, 904.7, 903, 823.4],
            'BELU': [91, 72.8, 53.5, 52]
        }
    }
    
    sectors = {
        'MOEXOG': ['TATN','SNGSP','SNGS','TRNFP','TATNP','SIBN','BANEP','BANE','EUTR','YAKG','NVTK','GAZP','ROSN'],
        'MOEXMM': ['GMKN', 'MAGN','RUAL','ALRS','SELG','RASP','UGLD','BLNG','NLMK','CHMF','MTLR','ENPG'],
        'MOEXIT': ['VKCO', 'ASTR', 'POSI','SOFL','RTKMP','MTSS','DATA','HEAD','DIAS','MGTSP','CNTL','CNTLP','AFKS','RTKM','RBCM'],
        'MOEXFN': ['SBER', 'SVCB','MOEX','SBERP','MBNK','RENI','CARM','LEAS','SFIN','ZAYM','CBOM','MGKL'],
        'MOEXEU': ['IRAO', 'FEES','MSNG','HYDR','MRKV','TGKA','OGKB','ELFV','MSRS','TGKBP','MRKY','TGKN','LSNG','UPRO','MAGE'],
        'MOEXRE': ['LSRG','WTCMP','MSTT','WTCM','SMLT','PRFN'],
        'MOEXTN': ['FLOT','NMTP','ELMT','KMAZ','WUSH','SVAV','AFLT'],
        'MOEXCN': ['MVID','BELU','APTK','MDMG','KLVZ','AQUA','DELI','GEMC','HNFG','X5','LENT','RAGR','NKNCP','NKNC','KZOS','KZOSP','PRMD','VSEH','ABIO','LIFE']
    }
    
    sectorsi = {
        'IMOEX2': ['VTBR', 'SBER', 'LKOH', 'T', 'GAZP','GMKN','PIKK'],
        'MOEXOG': ['GAZP', 'NVTK', 'ROSN', 'TATN','LKOH'],
        'MOEXMM': ['GMKN', 'MAGN', 'NLMK', 'PLZL','MTLR'],
        'MOEXIT': ['YDEX', 'VKCO', 'ASTR', 'POSI','SOFL'],
        'MOEXFN': ['VTBR', 'SBER', 'T', 'SVCB','BSPB'],
        'MOEXEU': ['IRAO', 'UPRO','FEES','MSNG','HYDR'],
        'MOEXRE': ['PIKK', 'SMLT','LSRG','MSTT'],
        'MOEXTN': ['AFLT', 'FLOT','SVAV','FESH','KMAZ'],
        'MOEXCN': ['MVID', 'LENT','MGNT','PHOR','BELU']
    }
    
    startpars = '2023-05-29'
    forecast_days = 14
    today = date.today()
    
    debug_print(f"📅 Сегодня: {today}")
    debug_print(f"📅 Стартовая дата для данных: {startpars}")
    debug_print(f"📊 Количество секторов для индексов: {len(sectorsi)}")
    debug_print(f"📊 Количество секторов для акций: {len(sectors)}")
        # Добавьте в начало main_pipeline():
    def check_moex():
         try:
            requests.get("httpss://iss.moex.com/iss/engines/stock/markets/shares/securities/SBER/candles.json?from=2024-01-01&till=2024-01-02&interval=24", 
                        timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            return True
         except:
            return False
    
     if not check_moex():
        print("❌ MOEX ISS API недоступен. Проверьте соединение или IP-адрес.")
        sys.exit(1)
    # ========== ЗАГРУЗКА ИНДЕКСОВ СЕКТОРОВ (ТОЧНО КАК В GOOGLE COLAB) ==========
    debug_print("🔧 Загружаю и рассчитываю индексы секторов...")
    ind = {}
    
    for sector, tickers in sectorsi.items():
        debug_print(f"  📈 Обрабатываю сектор: {sector}")
        sector_data = []
        
        # Загружаем данные по всем акциям сектора (точно как в Colab)
        for ticker in tickers:
            df = fetch_shares_data_colab(ticker, startpars, today.strftime('%Y-%m-%d'), is_index=False, retry_count=3)
            if not df.empty:
                df = df[['begin', 'close', 'volume', 'value']].rename(
                    columns={'close': ticker, 'volume': f'vol{ticker}', 'value': f'val{ticker}'}
                )
                sector_data.append(df)
            
            # Задержка между загрузкой разных тикеров
            time.sleep(random.uniform(2, 4))
        
        if not sector_data:
            debug_print(f"  ❌ Нет данных по сектору {sector}")
            # Создаем пустой индекс
            sec_pred = pd.DataFrame(columns=['begin', 'sector'])
            ind[sector] = sec_pred
            continue
        
        # Объединяем все акции сектора (точно как в Colab)
        combined_frame = sector_data[0]
        for df in sector_data[1:]:
            combined_frame = pd.merge(combined_frame, df, on='begin', how='inner')
        
        if combined_frame.empty:
            debug_print(f"  ❌ Нет общих данных в секторе {sector}")
            # Создаем пустой индекс
            sec_pred = pd.DataFrame(columns=['begin', 'sector'])
            ind[sector] = sec_pred
            continue
        
        combined_frame.dropna(inplace=True)
        
        # Добавляем капитализации
        if sector in cap:
            combined_frame = add_cap_columns(combined_frame, cap[sector])
        
        # Сохраняем копию без реального индекса для предсказания
        ccomb = combined_frame.copy()
        
        # Загружаем реальный индекс сектора (точно как в Colab)
        time.sleep(random.uniform(3, 6))  # Задержка перед загрузкой индекса
        sec_real = fetch_shares_data_colab(sector, startpars, today.strftime('%Y-%m-%d'), is_index=True, retry_count=3)
        
        if sec_real.empty:
            debug_print(f"  ⚠️ Нет данных по реальному индексу {sector}")
            # Создаем пустой индекс
            sec_pred = pd.DataFrame({
                'begin': combined_frame['begin'],
                'sector': 0
            })
            ind[sector] = sec_pred.drop_duplicates(subset='begin', keep='first')
            continue
        
        sec_real = sec_real[['begin', 'close']].rename(columns={'close': 'sector'})
        
        # Объединяем с реальным индексом
        combined_frame = pd.merge(combined_frame, sec_real, on='begin', how='inner')
        
        if combined_frame.empty:
            debug_print(f"  ⚠️ Нет общих данных с реальным индексом {sector}")
            # Создаем пустой индекс
            sec_pred = pd.DataFrame({
                'begin': ccomb['begin'],
                'sector': 0
            })
            ind[sector] = sec_pred.drop_duplicates(subset='begin', keep='first')
            continue
        
        # Подготавливаем данные для обучения
        combined_frame.set_index('begin', inplace=True)
        combined_frame.dropna(inplace=True)
        
        if len(combined_frame) < 30:
            debug_print(f"  ⚠️ Мало данных для обучения ({len(combined_frame)} строк)")
            # Создаем пустой индекс
            sec_pred = pd.DataFrame({
                'begin': ccomb['begin'],
                'sector': 0
            })
            ind[sector] = sec_pred.drop_duplicates(subset='begin', keep='first')
            continue
        
        X = combined_frame.drop(columns=['sector'])
        y = combined_frame['sector']
        
        # Обучаем модель
        model = LGBMRegressor(random_state=1, verbose=-1)
        model.fit(X, y)
        
        # Предсказываем на всех данных
        ccomb.set_index('begin', inplace=True)
        pred = model.predict(ccomb)
        
        # Сохраняем предсказания
        sec_pred = pd.DataFrame({
            'begin': ccomb.index,
            'sector': pred
        })
        
        ind[sector] = sec_pred.drop_duplicates(subset='begin', keep='first')
        debug_print(f"  ✅ Сектор {sector}: {len(ind[sector])} предсказаний, последняя дата: {ind[sector]['begin'].iloc[-1].date() if not ind[sector].empty else 'нет'}")
    
    # ========== ЗАГРУЗКА ДАННЫХ BRENT (ТОЧНО КАК В GOOGLE COLAB) ==========
    time.sleep(random.uniform(5, 10))  # Большая задержка перед загрузкой Brent
    bre = fetch_brent_data_colab(startpars, today.strftime('%Y-%m-%d'))
    
    if bre.empty:
        debug_print("⚠️ Не удалось загрузить данные Brent, создаю пустой DataFrame")
        bre = pd.DataFrame(columns=['begin', 'brent'])
    else:
        debug_print(f"✅ Данные Brent загружены: {len(bre)} строк, последняя дата: {bre['begin'].max().date() if not bre.empty else 'нет'}")
    
    # ========== ЗАГРУЗКА И ОБНОВЛЕНИЕ USD ДАННЫХ (ТОЧНО КАК В GOOGLE COLAB) ==========
    debug_print("💵 Загружаю и обновляю данные по USD...")
    file_pa = 'usd_data.json'
    
    try:
        with open(file_pa, 'r') as f:
            usdtt = pd.read_json(json.load(f))
        debug_print(f"✅ Файл usd_data.json загружен: {len(usdtt)} строк")
    except Exception as e:
        debug_print(f"⚠️ Ошибка загрузки usd_data.json: {e}")
        debug_print("📋 Создаю новый DataFrame...")
        usdtt = pd.DataFrame(columns=['begin', 'usdt'])
    
    # Определяем последнюю дату в файле
    if usdtt.empty:
        startusd = (today - timedelta(days=30)).strftime('%Y-%m-%d')
        debug_print("📅 usd_data.json пуст, начинаю загрузку с 30 дней назад")
    else:
        startusd = usdtt['begin'].iloc[-1] if 'begin' in usdtt.columns and not usdtt.empty else (today - timedelta(days=30)).strftime('%Y-%m-%d')
        debug_print(f"📅 Последняя дата в файле USD: {startusd}")
    
    # Преобразуем startusd в дату
    try:
        if isinstance(startusd, str):
            start_date = datetime.strptime(startusd, '%Y-%m-%dT%H:%M:%S.%f').date()
        else:
            start_date = startusd.date() if hasattr(startusd, 'date') else (today - timedelta(days=30))
    except:
        start_date = today - timedelta(days=30)
    
    end_date = today
    
    usd_rates = {}
    current_date = start_date + timedelta(days=1)
    max_retries = 3
    
    debug_print(f"📅 Загружаю данные USD с {current_date} по {end_date}")
    
    # Задержка перед началом загрузки USD
    time.sleep(random.uniform(2, 4))
    
    dates_to_load = []
    while current_date <= end_date:
        dates_to_load.append(current_date)
        current_date += timedelta(days=1)
    
    for load_date in dates_to_load:
        rate = None
        for attempt in range(max_retries):
            try:
                rate = get_currency_rate(load_date, retry_count=2)
                if rate is not None:
                    usd_rates[load_date.strftime("%d/%m/%Y")] = rate
                    debug_print(f"✅ Курс на {load_date}: {rate}")
                    break
                else:
                    debug_print(f"⚠️ Попытка {attempt+1}: курс на {load_date} не получен")
            except Exception as e:
                debug_print(f"⚠️ Ошибка при запросе {load_date} (попытка {attempt+1}): {e}")
            
            # Задержка между попытками
            if attempt < max_retries - 1:
                time.sleep(random.uniform(1, 3))
        
        if rate is None:
            debug_print(f"❌ Не удалось получить курс на {load_date} после {max_retries} попыток")
        
        # Задержка между запросами разных дат
        if load_date != dates_to_load[-1]:
            time.sleep(random.uniform(1, 2))
    
    # Всегда обновляем файл
    if usd_rates:
        usd_new = pd.DataFrame(list(usd_rates.items()), columns=['begin', 'usdt'])
        usd_new['begin'] = pd.to_datetime(usd_new['begin'], format='%d/%m/%Y')
        
        if not usdtt.empty and 'begin' in usdtt.columns:
            usdtt['begin'] = pd.to_datetime(usdtt['begin'])
            usd = pd.concat([usdtt, usd_new], ignore_index=True).drop_duplicates(subset='begin', keep='last')
        else:
            usd = usd_new
    else:
        usd = usdtt.copy()
        if not usd.empty:
            new_row = pd.DataFrame({'begin': [pd.Timestamp(end_date)], 'usdt': [np.nan]})
            usd = pd.concat([usd, new_row], ignore_index=True).drop_duplicates(subset='begin', keep='last')
    
    # Сохраняем обновленные данные
    with open(file_pa, 'w') as f:
        json.dump(usd.to_json(date_format='iso', orient='records'), f)
    
    debug_print(f"💾 Файл USD сохранен. Теперь строк: {len(usd)}, последняя дата: {usd['begin'].max() if not usd.empty else 'нет данных'}")
    
    if 'begin' in usd.columns:
        usd['begin'] = pd.to_datetime(usd['begin'])
    else:
        usd = pd.DataFrame(columns=['begin', 'usdt'])
    
    # ========== ОБРАБОТКА ДАННЫХ АКЦИЙ (ТОЧНО КАК В GOOGLE COLAB) ==========
    debug_print("📊 Обрабатываю данные по акциям...")
    
    # Если не удалось загрузить индексы, используем пустые данные
    if not ind or all(df.empty for df in ind.values()):
        debug_print("❌ Не удалось загрузить индексы, создаю пустые индексы")
        for sector in sectorsi:
            ind[sector] = pd.DataFrame(columns=['begin', 'sector'])
    
    data_framest = {}
    total_tickers = sum(len(tickers) for tickers in sectors.values())
    processed_tickers = 0
    
    for sector, tickers in sectors.items():
        sector_data = {}
        for ticker in tickers:
            processed_tickers += 1
            debug_print(f"  📊 Загрузка акции {ticker} ({processed_tickers}/{total_tickers})...")
            
            df = fetch_shares_data_colab(ticker, startpars, today.strftime('%Y-%m-%d'), is_index=False, retry_count=2)
            if df.empty:
                debug_print(f"  ⚠️ Пропускаю {ticker} - нет данных")
                # Задержка даже если нет данных
                time.sleep(random.uniform(1, 2))
                continue
            
            df = df[['begin', 'close', 'volume']].rename(columns={'close': 'ticker'})
            
            # Проверяем, что есть достаточно данных
            if df.shape[0] < 201:
                debug_print(f"  ⚠️ {ticker}: мало данных ({df.shape[0]} строк)")
                # Задержка даже если мало данных
                time.sleep(random.uniform(1, 2))
                continue
            
            # Объединяем с индексами
            if sector in ind and not ind[sector].empty:
                df = pd.merge(df, ind[sector], on='begin', how='inner')
            else:
                debug_print(f"  ⚠️ Нет предсказаний индекса для {sector}, пропускаю {ticker}")
                time.sleep(random.uniform(1, 2))
                continue
            
            if 'IMOEX2' in ind and not ind['IMOEX2'].empty:
                df = pd.merge(df, ind['IMOEX2'], on='begin', how='inner')
            else:
                debug_print(f"  ⚠️ Нет предсказаний IMOEX2, пропускаю {ticker}")
                time.sleep(random.uniform(1, 2))
                continue
            
            # Объединяем с USD и Brent
            if not usd.empty:
                df = pd.merge(df, usd, on='begin', how='inner')
            else:
                debug_print(f"  ⚠️ Нет данных USD, пропускаю {ticker}")
                time.sleep(random.uniform(1, 2))
                continue
            
            if not bre.empty:
                df = pd.merge(df, bre, on='begin', how='inner')
            else:
                debug_print(f"  ⚠️ Нет данных Brent, пропускаю {ticker}")
                time.sleep(random.uniform(1, 2))
                continue
            
            df.set_index('begin', inplace=True)
            df.dropna(inplace=True)
            
            # Добавляем скользящие средние
            for column in df.columns:
                if 'lag' not in column and 'rolling' not in column and 'sector' not in column and 'ticker' not in column:
                    df[f'{column}_rolling5'] = df[column].rolling(window=5).mean()
            
            df.dropna(inplace=True)
            
            # Добавляем лаги
            for column in df.columns:
                for lag in [7, 14, 28, 35, 31]:
                    if 'lag' not in column and 'rolling' not in column and 'sector_x' not in column:
                        df[f'{column}_lag_{lag}'] = df[column].shift(lag)
            
            df.dropna(inplace=True)
            sector_data[ticker] = df
            
            # Задержка между загрузкой разных акций
            if ticker != tickers[-1]:
                time.sleep(random.uniform(3, 6))
        
        data_framest[sector] = sector_data
    
    debug_print(f"✅ Обработаны данные по {sum(len(v) for v in data_framest.values())} акциям")
    
    # ========== СОЗДАНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ ==========
    debug_print("🎯 Создаю целевую переменную...")
    sppa = []  # Список акций с большим MAE
    spv = {}   # Словарь важности фичей
    smpe = 0   # Среднее MAE по всем акциям
    data_frames = {}
    
    for sector, tickers in sectors.items():
        sector_data = {}
        for ticker in tickers:
            if ticker in data_framest.get(sector, {}):
                result = data_framest[sector][ticker].copy()
                result['target'] = 0  # Инициализация колонки с макс % роста
                
                for i in range(len(result)):
                    current_close = result.iloc[i]['ticker']
                    # Проверяем следующие forecast_days дней
                    if i + forecast_days < len(result):
                        future_prices = result.iloc[i + 1:i + 1 + forecast_days]['ticker']
                        result.at[result.index[i], 'target'] = (max(future_prices) - current_close) * 100 / current_close
                
                result.dropna(inplace=True)
                sector_data[ticker] = result
        
        data_frames[sector] = sector_data
    
    # ========== ОБУЧЕНИЕ МОДЕЛЕЙ (ТОЧНО КАК В GOOGLE COLAB) ==========
    debug_print("🤖 Обучаю модели...")
    
    # Проверяем, есть ли данные для обучения
    total_models_expected = sum(len(tickers) for tickers in sectors.values())
    total_models_available = sum(len(secd) for secd in data_frames.values())
    
    if total_models_available == 0:
        debug_print("❌ Нет данных для обучения моделей!")
        debug_print("📋 Создаю пустые результаты...")
        
        # Создаем пустые результаты
        results_data = {
            "date": str(date.today()),
            "timestamp": datetime.now().isoformat(),
            "predictions": {},
            "medium_potential": {},
            "high_potential": {},
            "poor_accuracy": [],
            "avg_mae": 100,
            "model_accuracy": 0,
            "summary": {
                "high_count": 0,
                "medium_count": 0,
                "is_bad_day": True,
                "is_fifty_fifty": False,
                "is_good_day": False
            },
            "model_stats": {
                "models_loaded": 0,
                "models_learned_today": 0,
                "total_models": 0,
                "expected_models": total_models_expected
            }
        }
        
        # Сохраняем результаты
        with open('latest_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n📊 РЕЗУЛЬТАТЫ АНАЛИЗА")
        print(f"📅 Дата анализа: {date.today()}")
        print("❌ НЕТ ДАННЫХ ДЛЯ АНАЛИЗА")
        print("⚠️ Не удалось загрузить данные с MOEX API")
        print('\nBAD DAY')
        
        return results_data
    
    param_grid = {
        'num_leaves': [31, 50, 21],
        'max_depth': [-1, 5, 10],
        'learning_rate': [0.3, 0.1, 0.2],
        'n_estimators': [250, 200]
    }
    
    # Создаем папку для сохранения параметров моделей
    model_params_dir = 'model_params'
    if not os.path.exists(model_params_dir):
        os.makedirs(model_params_dir)
        debug_print(f"📁 Создана папка для параметров моделей: {model_params_dir}")
    
    models = {}
    models_learned_today = 0
    models_loaded = 0
    
    # СОХРАНЯЕМ ВСЕ ПАРАМЕТРЫ МОДЕЛЕЙ ДЛЯ ВСЕХ АКЦИЙ
    for sector, secd in data_frames.items():
        models[sector] = {}
        for ticker in sectors[sector]:
            if ticker in secd:
                X = secd[ticker].drop(columns=['target']).iloc[:-(1 + forecast_days)]
                y = secd[ticker]['target'].iloc[:-(1 + forecast_days)]
                
                # Проверяем, есть ли сохраненные параметры для этой модели
                param_file = f'{model_params_dir}/{sector}_{ticker}_params.json'
                
                if os.path.exists(param_file):
                    # Загружаем сохраненные параметры
                    try:
                        with open(param_file, 'r') as f:
                            loaded_params = json.load(f)
                        model = LGBMRegressor(random_state=1, **loaded_params)
                        models_loaded += 1
                        debug_print(f"  ✅ Загружены параметры для {ticker}")
                    except Exception as e:
                        debug_print(f"  ❌ Ошибка загрузки параметров для {ticker}: {e}")
                        # Если не удалось загрузить, делаем GridSearch
                        model = LGBMRegressor(random_state=1)
                        grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=4, verbose=0)
                        grid_search.fit(X, y)
                        best_model = grid_search.best_estimator_
                        model = best_model
                        best_params = grid_search.best_params_
                        
                        # Сохраняем параметры модели
                        with open(param_file, 'w') as f:
                            json.dump(best_params, f, indent=2)
                        
                        models_learned_today += 1
                        debug_print(f"  💾 Сохранены параметры для {ticker} (после ошибки загрузки)")
                else:
                    # Делаем GridSearch для новых моделей
                    debug_print(f"  🔍 Подбираю параметры для {ticker} с помощью GridSearch...")
                    model = LGBMRegressor(random_state=1)
                    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=4, verbose=0)
                    grid_search.fit(X, y)
                    best_model = grid_search.best_estimator_
                    model = best_model
                    best_params = grid_search.best_params_
                    
                    # Сохраняем параметры модели
                    with open(param_file, 'w') as f:
                        json.dump(best_params, f, indent=2)
                    
                    models_learned_today += 1
                    debug_print(f"  💾 Сохранены параметры для {ticker}")
                
                # Веса для важных примеров
                threshold = 12
                weights = np.where(y > threshold, 2, 1)
                model.fit(X, y, sample_weight=weights)
                
                # Кросс-валидация
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
                mse_scores = -cv_scores
                
                # Запоминаем акции с большим MAE
                if mse_scores.mean() > 9:  # Изменено с 7.5 на 9 как в Colab
                    sppa.append(ticker)
                    sppa.append(mse_scores.mean())
                
                smpe += mse_scores.mean()
                
                # Собираем важность фичей
                for i in range(len(model.feature_importances_)):
                    if X.columns[i] not in spv.keys():
                        spv[X.columns[i]] = 0
                    spv[X.columns[i]] += model.feature_importances_[i]
                
                models[sector][ticker] = model
            else:
                debug_print(f"  ⚠️ {ticker} нет в данных сектора {sector}")
    
    debug_print(f"📊 Загружено моделей: {models_loaded}, обучено новых: {models_learned_today}")
    
    # ========== ПРЕДСКАЗАНИЯ ==========
    debug_print("🔮 Делаю предсказания...")
    predictions = {}
    
    for sector, secd in data_frames.items():
        for ticker in sectors[sector]:
            if ticker in secd and ticker in models.get(sector, {}):
                try:
                    last_data = secd[ticker].drop(columns=['target']).iloc[-1].copy()
                    prediction = models[sector][ticker].predict([last_data])
                    predictions[ticker] = float(prediction[0])
                except Exception as e:
                    debug_print(f"  ❌ Ошибка предсказания для {ticker}: {e}")
                    predictions[ticker] = 0.0
    
    # ========== КЛАССИФИКАЦИЯ РЕЗУЛЬТАТОВ (ТОЧНО КАК В GOOGLE COLAB) ==========
    sr, mn = {}, {}
    for ticker, pr in predictions.items():
        if pr > 5.5 and pr < 9.5:  # Изменено как в Colab
            sr[ticker] = pr
        elif pr >= 9.5:  # Изменено как в Colab
            mn[ticker] = pr
    
    # Рассчитываем точность модели (обратная величина от средней ошибки)
    avg_mae = smpe / len(predictions) if predictions else 100
    accuracy = max(0, 100 - avg_mae)  # Чем меньше MAE, тем выше точность
    
    # ========== ФОРМИРОВАНИЕ РЕЗУЛЬТАТОВ ==========
    from datetime import datetime, timezone
    
    results_data = {
        "date": str(date.today()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "predictions": predictions,
        "medium_potential": dict(sorted(sr.items(), key=lambda item: item[1], reverse=True)),
        "high_potential": dict(sorted(mn.items(), key=lambda item: item[1], reverse=True)),
        "poor_accuracy": sppa,
        "avg_mae": avg_mae,
        "model_accuracy": round(accuracy, 1),
        "summary": {
            "high_count": len(mn),
            "medium_count": len(sr),
            "is_bad_day": len(mn) < 6,
            "is_fifty_fifty": 6 <= len(mn) < 9,
            "is_good_day": len(mn) >= 9
        },
        "model_stats": {
            "models_loaded": models_loaded,
            "models_learned_today": models_learned_today,
            "total_models": len(predictions),
            "expected_models": total_models_expected
        }
    }
    
    # Сохраняем результаты для следующего шага
    with open('latest_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    debug_print(f"✅ Анализ завершен: {len(predictions)} акций")
    debug_print(f"🎯 Лучших акций: {len(mn)}")
    debug_print(f"📊 Средних акций: {len(sr)}")
    debug_print(f"📈 Точность модели: {accuracy:.1f}%")
    debug_print(f"🤖 Моделей: загружено {models_loaded}, обучено новых {models_learned_today}")
    debug_print(f"📋 Ожидалось моделей: {results_data['model_stats']['expected_models']}")
    
    # Выводим результаты как в Google Colab
    print(f"\n📊 РЕЗУЛЬТАТЫ АНАЛИЗА")
    print(f"📅 Дата анализа: {date.today()}")
    print(f"🎯 Высокий потенциал (>9.5%): {len(mn)} акций")
    print(f"📊 Средний потенциал (5.5-9.5%): {len(sr)} акций")
    print(f"📈 Точность модели: {accuracy:.1f}%")
    
    if len(mn) < 6:
        print('\nBAD DAY')
    elif len(mn) < 9:
        print('\n50/50 DAY')
    else:
        print('\n100% GOOD DAY')
    
    if sr:
        print(f"\n📊 СРЕДНИЙ ПОТЕНЦИАЛ (отсортировано по убыванию):")
        for i, (ticker, value) in enumerate(results_data['medium_potential'].items()):
            if i < 30:  # Ограничим вывод
                print(f"  {ticker}: +{value:.1f}%")
    
    if mn:
        print(f"\n🎯 ВЫСОКИЙ ПОТЕНЦИАЛ (отсортировано по убыванию):")
        for ticker, value in results_data['high_potential'].items():
            print(f"  {ticker}: +{value:.1f}%")
    
    # Сохраняем отладочные данные
    debug_data = {
        'analysis_date': str(today),
        'predictions_count': len(predictions),
        'usd_data_last_date': str(usd['begin'].max()) if not usd.empty else "нет данных",
        'brent_data_last_date': str(bre['begin'].max()) if not bre.empty else "нет данных",
        'results_summary': results_data['summary'],
        'model_stats': results_data['model_stats'],
        'avg_mae': avg_mae,
        'model_accuracy': accuracy
    }
    save_debug_data(debug_data, f"debug_{today}.json")
    
    return results_data

def main():
    """Основная функция для GitHub Actions"""
    print("=" * 60)
    print("🚀 ЗАПУСК ПОЛНОГО АНАЛИЗА В GITHUB ACTIONS (ТОЧНАЯ КОПИЯ GOOGLE COLAB)")
    print("=" * 60)
    
    # Проверяем наличие файла usd_data.json
    if not os.path.exists('usd_data.json'):
        print("❌ Файл usd_data.json не найден!")
        print("📋 Создаю новый файл с начальными данными...")
        
        initial_data = pd.DataFrame(columns=['begin', 'usdt'])
        with open('usd_data.json', 'w') as f:
            json.dump(initial_data.to_json(date_format='iso', orient='records'), f)
        print("✅ Файл usd_data.json создан")
    else:
        print("✅ Файл usd_data.json найден")
    
    # Проверяем папку для параметров моделей
    if not os.path.exists('model_params'):
        print("📁 Создаю папку для параметров моделей...")
        os.makedirs('model_params')
        print("✅ Папка model_params создана")
    else:
        print(f"✅ Папка model_params найдена, файлов: {len(os.listdir('model_params'))}")
    
    try:
        print("🔧 Запускаю полный анализ (точная копия Google Colab)...")
        results = main_pipeline(forecast_days=14)
        
        print(f"\n✅ Анализ завершен: {len(results.get('predictions', {}))} акций")
        print(f"🎯 Лучших акций: {results.get('summary', {}).get('high_count', 0)}")
        print(f"📊 Средних акций: {results.get('summary', {}).get('medium_count', 0)}")
        print(f"📈 Точность модели: {results.get('model_accuracy', 0)}%")
        print(f"🤖 Моделей: загружено {results.get('model_stats', {}).get('models_loaded', 0)}, обучено новых {results.get('model_stats', {}).get('models_learned_today', 0)}")
        print(f"📋 Ожидалось моделей: {results.get('model_stats', {}).get('expected_models', 0)}")
        
        return results
        
    except Exception as e:
        print(f"❌ Ошибка при анализе: {e}")
        import traceback
        traceback.print_exc()
        
        # Создаем файл с информацией об ошибке
        error_data = {
            "date": str(date.today()),
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        with open(f"error_{date.today()}.json", 'w') as f:
            json.dump(error_data, f, indent=2)
        
        sys.exit(1)

if __name__ == "__main__":
    main()
