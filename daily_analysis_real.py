"""
ПОЛНЫЙ АНАЛИЗ АКЦИЙ ДЛЯ GITHUB ACTIONS
Адаптированная версия из Google Colab с пагинацией и исправленной загрузкой Brent
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
import joblib
import warnings
import xml.etree.ElementTree as ET
import time

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

# ========== ФУНКЦИИ ДЛЯ ЗАГРУЗКИ ДАННЫХ MOEX ==========
def fetch_moex_data_paginated(ticker, start_date, end_date, is_index=False, max_days=400):
    """
    Загружает данные MOEX с разбивкой на части
    MOEX API ограничивает 500 строк за запрос
    """
    all_data = []
    current_start = pd.to_datetime(start_date)
    current_end = pd.to_datetime(end_date)
    
    # Определяем тип запроса
    if is_index:
        url_template = "http://iss.moex.com/iss/engines/stock/markets/index/securities/{ticker}/candles.json?from={start}&till={end}&interval=24"
    else:
        url_template = "http://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json?from={start}&till={end}&interval=24"
    
    attempts = 0
    max_attempts = 3
    
    while current_start < current_end:
        part_end = min(current_start + timedelta(days=max_days), current_end)
        
        url = url_template.format(
            ticker=ticker,
            start=current_start.strftime('%Y-%m-%d'),
            end=part_end.strftime('%Y-%m-%d')
        )
        
        debug_print(f"  📥 Загрузка {ticker}: {current_start.date()} - {part_end.date()}")
        
        success = False
        for attempt in range(max_attempts):
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    if 'candles' in data and 'data' in data['candles'] and data['candles']['data']:
                        part_data = [{k: r[i] for i, k in enumerate(data['candles']['columns'])} 
                                   for r in data['candles']['data']]
                        df_part = pd.DataFrame(part_data)
                        df_part['begin'] = pd.to_datetime(df_part['begin'])
                        all_data.append(df_part)
                        debug_print(f"    ✅ Загружено {len(df_part)} строк")
                        success = True
                        break
                    else:
                        debug_print(f"    ⚠️ Нет данных (попытка {attempt+1})")
                else:
                    debug_print(f"    ⚠️ Ошибка HTTP {response.status_code} (попытка {attempt+1})")
            except Exception as e:
                debug_print(f"    ⚠️ Ошибка запроса: {e} (попытка {attempt+1})")
            
            if attempt < max_attempts - 1:
                time.sleep(1)
        
        if not success:
            debug_print(f"    ❌ Не удалось загрузить данные за период {current_start.date()} - {part_end.date()}")
        
        current_start = part_end + timedelta(days=1)
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values('begin').drop_duplicates('begin')
        debug_print(f"  📊 Итого по {ticker}: {len(result)} строк, последняя дата: {result['begin'].max().date() if not result.empty else 'нет'}")
        return result
    else:
        debug_print(f"  ❌ Нет данных по {ticker}")
        return pd.DataFrame()

def fetch_futures_data_paginated(ticker, start_date, end_date, max_days=400):
    """Загружает данные фьючерсов с разбивкой на части"""
    all_data = []
    current_start = pd.to_datetime(start_date)
    current_end = pd.to_datetime(end_date)
    
    attempts = 0
    max_attempts = 3
    
    while current_start < current_end:
        part_end = min(current_start + timedelta(days=max_days), current_end)
        
        url = f"http://iss.moex.com/iss/engines/futures/markets/forts/securities/{ticker}/candles.json?from={current_start.strftime('%Y-%m-%d')}&till={part_end.strftime('%Y-%m-%d')}&interval=24"
        
        debug_print(f"  📥 Загрузка {ticker}: {current_start.date()} - {part_end.date()}")
        
        success = False
        for attempt in range(max_attempts):
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    if 'candles' in data and 'data' in data['candles'] and data['candles']['data']:
                        part_data = [{k: r[i] for i, k in enumerate(data['candles']['columns'])} 
                                   for r in data['candles']['data']]
                        df_part = pd.DataFrame(part_data)
                        df_part['begin'] = pd.to_datetime(df_part['begin'])
                        all_data.append(df_part)
                        debug_print(f"    ✅ Загружено {len(df_part)} строк")
                        success = True
                        break
                    else:
                        debug_print(f"    ⚠️ Нет данных (попытка {attempt+1})")
                else:
                    debug_print(f"    ⚠️ Ошибка HTTP {response.status_code} (попытка {attempt+1})")
            except Exception as e:
                debug_print(f"    ⚠️ Ошибка запроса: {e} (попытка {attempt+1})")
            
            if attempt < max_attempts - 1:
                time.sleep(1)
        
        if not success:
            debug_print(f"    ❌ Не удалось загрузить данные за период {current_start.date()} - {part_end.date()}")
        
        current_start = part_end + timedelta(days=1)
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values('begin').drop_duplicates('begin')
        debug_print(f"  📊 Итого по {ticker}: {len(result)} строк, от {result['begin'].min().date() if not result.empty else 'нет'} до {result['begin'].max().date() if not result.empty else 'нет'}")
        return result
    else:
        debug_print(f"  ❌ Нет данных по {ticker}")
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

# ========== ФУНКЦИЯ ДЛЯ ЗАГРУЗКИ ДАННЫХ BRENT ИЗ НЕСКОЛЬКИХ КОНТРАКТОВ ==========
def fetch_brent_data_multiple_contracts(start_date, end_date):
    """Загружает данные Brent из нескольких контрактов для покрытия всего периода"""
    debug_print("🛢️ Загружаю данные по Brent из нескольких контрактов...")
    
    # Преобразуем даты для работы с ними
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    
    # Определяем годы, которые нужно покрыть (2023-2026)
    start_year = start_date_dt.year
    end_year = end_date_dt.year
    
    # Для каждого года берем 1-2 ключевых контракта
    # Поскольку контракты действуют примерно с сентября до сентября,
    # берем сентябрьский (U) и декабрьский (Z) контракты для каждого года
    brent_contracts = []
    
    for year in range(start_year, end_year + 1):
        # Определяем месяц для кода контракта
        # Сентябрьский контракт для каждого года (наиболее ликвидный)
        if year <= 2026:  # Проверяем, что год не слишком далеко в будущем
            month_code = 'U'  # Сентябрь
            year_code = str(year)[-1]  # Последняя цифра года
            contract = f'BR{month_code}{year_code}'
            brent_contracts.append(contract)
            
            # Для текущего года также берем ближайший квартальный контракт
            if year == end_year:
                # Определяем текущий месяц
                current_month = end_date_dt.month
                # Месяца и их коды: F=янв, G=фев, H=мар, J=апр, K=май, M=июн, 
                # N=июл, Q=авг, U=сен, V=окт, X=ноя, Z=дек
                month_codes = {
                    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
                    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
                }
                
                # Берем следующий квартальный контракт от текущего месяца
                if current_month <= 3:
                    extra_contract = f'BRM{year_code}'  # Июнь
                elif current_month <= 6:
                    extra_contract = f'BRU{year_code}'  # Сентябрь
                elif current_month <= 9:
                    extra_contract = f'BRZ{year_code}'  # Декабрь
                else:
                    # Если уже декабрь, берем март следующего года
                    next_year_code = str(year + 1)[-1]
                    extra_contract = f'BRH{next_year_code}'  # Март следующего года
                
                if extra_contract not in brent_contracts:
                    brent_contracts.append(extra_contract)
    
    # Оставляем только уникальные контракты
    brent_contracts = list(set(brent_contracts))
    
    debug_print(f"📋 Загружаемые контракты: {brent_contracts}")
    
    all_brent_data = []
    
    for contract in brent_contracts:
        debug_print(f"  🔄 Загружаю контракт {contract}...")
        df = fetch_futures_data_paginated(contract, start_date, end_date)
        
        if not df.empty:
            df = df[['begin', 'close']].rename(columns={'close': 'brent'})
            all_brent_data.append(df)
            debug_print(f"    ✅ Загружено {len(df)} строк из {contract}")
            
            # Логируем диапазон дат для этого контракта
            if not df.empty:
                debug_print(f"    📅 Диапазон {contract}: {df['begin'].min().date()} - {df['begin'].max().date()}")
        else:
            debug_print(f"    ⚠️ Нет данных по контракту {contract}")
    
    if all_brent_data:
        # Объединяем все данные
        bre = pd.concat(all_brent_data, ignore_index=True)
        
        # Сортируем по дате
        bre = bre.sort_values('begin')
        
        # Удаляем дубликаты по дате, оставляя последнее значение 
        # (предполагаем, что более новые контракты имеют более актуальные цены)
        bre = bre.drop_duplicates(subset='begin', keep='last')
        
        # Проверяем наличие пропусков
        bre_dates = sorted(bre['begin'].unique())
        if len(bre_dates) > 1:
            date_range = pd.date_range(start=bre_dates[0], end=bre_dates[-1])
            missing_dates = date_range.difference(bre_dates)
            if len(missing_dates) > 0:
                debug_print(f"⚠️ Обнаружены пропуски в данных Brent: {len(missing_dates)} дней")
        
        debug_print(f"📊 Всего данных Brent: {len(bre)} строк")
        debug_print(f"📅 Диапазон данных Brent: {bre['begin'].min().date()} - {bre['begin'].max().date()}")
        
        return bre
    else:
        debug_print("❌ Не удалось загрузить данные по Brent ни из одного контракта")
        return pd.DataFrame(columns=['begin', 'brent'])

# ========== ОСНОВНОЙ КОД ==========
def main_pipeline(forecast_days=14, output_mode="json"):
    """
    Основная функция для запуска всего пайплайна
    forecast_days: количество дней для предсказания
    output_mode: формат вывода ('tg' для Telegram, 'json' для API)
    """
    debug_print(f"🚀 Начинаю полный анализ...")
    
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
        'MOEXIT': ['VKCO', 'ASTR', 'POSI','SOFL','RTKMP','MTSS','DATA','SVET','MGTSP','NSVZ','CNTL','CNTLP','AFKS','RTKM','RBCM'],
        'MOEXFN': ['SBER', 'SVCB','BSPB','MOEX','SBERP','MBNK','RENI','CARM','LEAS','SFIN','ZAYM','CBOM','MGKL'],
        'MOEXEU': ['IRAO', 'FEES','MSNG','HYDR','MRKV','TGKA','OGKB','ELFV','MSRS','TGKBP','MRKY','TGKN','LSNG','UPRO','MAGE'],
        'MOEXRE': ['LSRG','WTCMP','MSTT','WTCM','SMLT','PRFN'],
        'MOEXTN': ['FLOT','NMTP','ELMT','UWGN','SVAV','UNAC','AFLT'],
        'MOEXCN': ['MVID', 'LENT','BELU','APTK','MDMG','KLVZ','AQUA','DELI','GEMC','HNFG','KROT','NKNCP','NKNC','KZOS','KZOSP','PRMD','VSEH','ABIO','LIFE']
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
    
    # ========== ЗАГРУЗКА ИНДЕКСОВ СЕКТОРОВ ==========
    debug_print("🔧 Загружаю и рассчитываю индексы секторов...")
    ind = {}
    
    for sector, tickers in sectorsi.items():
        debug_print(f"  📈 Обрабатываю сектор: {sector}")
        sector_data = []
        
        # Загружаем данные по всем акциям сектора
        for ticker in tickers:
            df = fetch_moex_data_paginated(ticker, startpars, today.strftime('%Y-%m-%d'), is_index=False)
            if not df.empty:
                df = df[['begin', 'close', 'volume', 'value']].rename(
                    columns={'close': ticker, 'volume': f'vol{ticker}', 'value': f'val{ticker}'}
                )
                sector_data.append(df)
        
        if not sector_data:
            debug_print(f"  ❌ Нет данных по сектору {sector}")
            continue
        
        # Объединяем все акции сектора
        combined_frame = sector_data[0]
        for df in sector_data[1:]:
            combined_frame = pd.merge(combined_frame, df, on='begin', how='inner')
        
        if combined_frame.empty:
            debug_print(f"  ❌ Нет общих данных в секторе {sector}")
            continue
        
        combined_frame.dropna(inplace=True)
        
        # Добавляем капитализации
        if sector in cap:
            combined_frame = add_cap_columns(combined_frame, cap[sector])
        
        # Сохраняем копию без реального индекса для предсказания
        ccomb = combined_frame.copy()
        
        # Загружаем реальный индекс сектора
        sec_real = fetch_moex_data_paginated(sector, startpars, today.strftime('%Y-%m-%d'), is_index=True)
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
        debug_print(f"  ✅ Сектор {sector}: {len(ind[sector])} предсказаний, последняя дата: {ind[sector]['begin'].iloc[-1].date()}")
    
    # ========== ЗАГРУЗКА ДАННЫХ BRENT ==========
    bre = fetch_brent_data_multiple_contracts(startpars, today.strftime('%Y-%m-%d'))
    
    if bre.empty:
        debug_print("⚠️ Не удалось загрузить данные Brent, создаю пустой DataFrame")
        bre = pd.DataFrame(columns=['begin', 'brent'])
    else:
        debug_print(f"✅ Данные Brent загружены: {len(bre)} строк, последняя дата: {bre['begin'].max().date()}")
    
    # ========== ЗАГРУЗКА И ОБНОВЛЕНИЕ USD ДАННЫХ ==========
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
    else:
        startusd = usdtt['begin'].iloc[-1] if 'begin' in usdtt.columns and not usdtt.empty else (today - timedelta(days=30)).strftime('%Y-%m-%d')
    
    debug_print(f"📅 Последняя дата в файле USD: {startusd}")
    
    def get_currency_rate(date):
        url = f'https://www.cbr.ru/scripts/XML_daily.asp?date_req={date.strftime("%d/%m/%Y")}'
        response = requests.get(url)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            for valute in root.findall('Valute'):
                char_code = valute.find('CharCode').text
                if char_code == 'USD':
                    value = valute.find('Value').text
                    return float(value.replace(',', '.'))
        else:
            debug_print(f'⚠️ Ошибка при запросе данных за {date.strftime("%d/%m/%Y")}: {response.status_code}')
        return None
    
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
    
    while current_date <= end_date:
        rate = None
        for attempt in range(max_retries):
            try:
                rate = get_currency_rate(current_date)
                if rate is not None:
                    usd_rates[current_date.strftime("%d/%m/%Y")] = rate
                    debug_print(f"✅ Курс на {current_date}: {rate}")
                    break
                else:
                    debug_print(f"⚠️ Попытка {attempt+1}: курс на {current_date} не получен")
            except Exception as e:
                debug_print(f"⚠️ Ошибка при запросе {current_date} (попытка {attempt+1}): {e}")
            time.sleep(0.5)
        
        if rate is None:
            debug_print(f"❌ Не удалось получить курс на {current_date} после {max_retries} попыток")
        
        current_date += timedelta(days=1)
    
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
    
    # ========== ОБРАБОТКА ДАННЫХ АКЦИЙ ==========
    debug_print("📊 Обрабатываю данные по акциям...")
    data_framest = {}
    
    for sector, tickers in sectors.items():
        sector_data = {}
        for ticker in tickers:
            df = fetch_moex_data_paginated(ticker, startpars, today.strftime('%Y-%m-%d'), is_index=False)
            if df.empty:
                debug_print(f"  ⚠️ Пропускаю {ticker} - нет данных")
                continue
            
            df = df[['begin', 'close', 'volume']].rename(columns={'close': 'ticker'})
            
            # Проверяем, что есть достаточно данных
            if df.shape[0] < 201:
                debug_print(f"  ⚠️ {ticker}: мало данных ({df.shape[0]} строк)")
                continue
            
            # Объединяем с индексами
            if sector in ind and not ind[sector].empty:
                df = pd.merge(df, ind[sector], on='begin', how='inner')
            else:
                debug_print(f"  ⚠️ Нет предсказаний индекса для {sector}, пропускаю {ticker}")
                continue
            
            if 'IMOEX2' in ind and not ind['IMOEX2'].empty:
                df = pd.merge(df, ind['IMOEX2'], on='begin', how='inner')
            else:
                debug_print(f"  ⚠️ Нет предсказаний IMOEX2, пропускаю {ticker}")
                continue
            
            # Объединяем с USD и Brent
            if not usd.empty:
                df = pd.merge(df, usd, on='begin', how='inner')
            else:
                debug_print(f"  ⚠️ Нет данных USD, пропускаю {ticker}")
                continue
            
            if not bre.empty:
                df = pd.merge(df, bre, on='begin', how='inner')
            else:
                debug_print(f"  ⚠️ Нет данных Brent, пропускаю {ticker}")
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
    
    # ========== ОБУЧЕНИЕ МОДЕЛЕЙ ==========
    debug_print("🤖 Обучаю модели...")
    param_grid = {
        'num_leaves': [31, 50, 21],
        'max_depth': [-1, 5, 10],
        'learning_rate': [0.3, 0.1, 0.2],
        'n_estimators': [250, 200]
    }
    
    models = {}
    
    for sector, secd in data_frames.items():
        models[sector] = {}
        for ticker in sectors[sector]:
            if ticker in secd:
                X = secd[ticker].drop(columns=['target']).iloc[:-(1 + forecast_days)]
                y = secd[ticker]['target'].iloc[:-(1 + forecast_days)]
                
                # Обучение модели с GridSearch
                model = LGBMRegressor(random_state=1)
                grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=4)
                grid_search.fit(X, y)
                best_model = grid_search.best_estimator_
                model = best_model
                
                # Веса для важных примеров
                threshold = 12
                weights = np.where(y > threshold, 2, 1)
                model.fit(X, y, sample_weight=weights)
                
                # Кросс-валидация
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
                mse_scores = -cv_scores
                
                # Запоминаем акции с большим MAE
                if mse_scores.mean() > 7.5:
                    sppa.append(ticker)
                    sppa.append(mse_scores.mean())
                
                smpe += mse_scores.mean()
                
                # Собираем важность фичей
                for i in range(len(model.feature_importances_)):
                    if X.columns[i] not in spv.keys():
                        spv[X.columns[i]] = 0
                    spv[X.columns[i]] += model.feature_importances_[i]
                
                models[sector][ticker] = model
    
    # ========== ПРЕДСКАЗАНИЯ ==========
    debug_print("🔮 Делаю предсказания...")
    predictions = {}
    
    for sector, secd in data_frames.items():
        for ticker in sectors[sector]:
            if ticker in secd and ticker in models.get(sector, {}):
                last_data = secd[ticker].drop(columns=['target']).iloc[-1].copy()
                prediction = models[sector][ticker].predict([last_data])
                predictions[ticker] = float(prediction[0])
    
    # ========== КЛАССИФИКАЦИЯ РЕЗУЛЬТАТОВ ==========
    sr, mn = {}, {}
    for ticker, pr in predictions.items():
        if pr > 3.5 and pr < 7.5:
            sr[ticker] = pr
        elif pr >= 7.5:
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
        "medium_potential": dict(sr),
        "high_potential": dict(mn),
        "poor_accuracy": sppa,
        "avg_mae": avg_mae,
        "model_accuracy": round(accuracy, 1),
        "summary": {
            "high_count": len(mn),
            "medium_count": len(sr),
            "is_bad_day": len(mn) < 6,
            "is_fifty_fifty": 6 <= len(mn) < 9,
            "is_good_day": len(mn) >= 9
        }
    }
    
    # Сохраняем результаты для следующего шага
    with open('latest_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    debug_print(f"✅ Анализ завершен: {len(predictions)} акций")
    debug_print(f"🎯 Лучших акций: {len(mn)}")
    debug_print(f"📊 Средних акций: {len(sr)}")
    debug_print(f"📈 Точность модели: {accuracy:.1f}%")
    
    # Сохраняем отладочные данные
    debug_data = {
        'analysis_date': str(today),
        'predictions_count': len(predictions),
        'usd_data_last_date': str(usd['begin'].max()) if not usd.empty else "нет данных",
        'brent_data_last_date': str(bre['begin'].max()) if not bre.empty else "нет данных",
        'brent_data_range': f"{bre['begin'].min().date() if not bre.empty else 'нет'} - {bre['begin'].max().date() if not bre.empty else 'нет'}",
        'results_summary': results_data['summary'],
        'avg_mae': avg_mae,
        'model_accuracy': accuracy
    }
    save_debug_data(debug_data, f"debug_{today}.json")
    
    # Выводим важность фичей для отладки
    if spv:
        for i in spv.keys():
            spv[i] = spv[i] / 100
        sorted_dict = sorted(spv.items(), key=lambda item: item[1], reverse=True)[:10]  # Топ-10 фичей
        debug_print("🏆 Топ-10 самых важных фичей:")
        for key, value in sorted_dict:
            debug_print(f"  {key}: {value:.4f}")
    
    return results_data

def main():
    """Основная функция для GitHub Actions"""
    print("=" * 60)
    print("🚀 ЗАПУСК ПОЛНОГО АНАЛИЗА В GITHUB ACTIONS")
    print("=" * 60)
    
    if not os.path.exists('usd_data.json'):
        print("❌ Файл usd_data.json не найден!")
        print("📋 Создаю новый файл с начальными данными...")
        
        initial_data = pd.DataFrame(columns=['begin', 'usdt'])
        with open('usd_data.json', 'w') as f:
            json.dump(initial_data.to_json(date_format='iso', orient='records'), f)
    
    print("✅ Файл usd_data.json найден или создан")
    
    try:
        print("🔧 Запускаю полный анализ...")
        results = main_pipeline(forecast_days=14, output_mode="json")
        
        print(f"✅ Анализ завершен: {len(results.get('predictions', {}))} акций")
        print(f"🎯 Лучших акций: {results.get('summary', {}).get('high_count', 0)}")
        print(f"📊 Средних акций: {results.get('summary', {}).get('medium_count', 0)}")
        print(f"📈 Точность модели: {results.get('model_accuracy', 0)}%")
        
        # Определяем тип дня
        if results['summary']['is_bad_day']:
            day_type = "📉 ПЛОХОЙ ДЕНЬ"
        elif results['summary']['is_fifty_fifty']:
            day_type = "🤝 СРЕДНИЙ ДЕНЬ"
        else:
            day_type = "📈 ХОРОШИЙ ДЕНЬ"
        
        print(f"\n{day_type}")
        
        with open('latest_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
        
    except Exception as e:
        print(f"❌ Ошибка при анализе: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
