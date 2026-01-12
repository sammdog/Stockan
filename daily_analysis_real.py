"""
ПОЛНЫЙ АНАЛИЗ АКЦИЙ ДЛЯ GITHUB ACTIONS
Исправленная версия с разбивкой запросов на части
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
from math import comb
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
        return result
    else:
        return pd.DataFrame()

# ========== ОСНОВНОЙ КОД ==========
def main_pipeline(forecast_days=14, output_mode="json"):
    """
    Основная функция для запуска всего пайплайна
    forecast_days: количество дней для предсказания
    output_mode: формат вывода ('tg' для Telegram, 'json' для API)
    """
    import numpy as np
    import pandas as pd
    import requests
    from datetime import date, timedelta
    import lightgbm
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
    from sklearn.metrics import mean_absolute_error, r2_score, precision_score, mean_squared_error, mean_absolute_percentage_error
    from sklearn.ensemble import GradientBoostingRegressor
    import joblib
    import os
    import json
    
    debug_print(f"🚀 Начинаю полный анализ...")
    
    # === КОНФИГУРАЦИЯ ===
    cap=dict()
    cap['IMOEX2']={}
    cap['MOEXOG']={}
    cap['MOEXMM']={}
    cap['MOEXIT']={}
    cap['MOEXFN']={}
    cap['MOEXEU']={}
    cap['MOEXRE']={}
    cap['MOEXTN']={}
    cap['MOEXCN']={}
    cap['IMOEX2']['VTBR']=[657.8,505.5,523.8]
    cap['IMOEX2']['SBER']=[6600,7000,7200]
    cap['IMOEX2']['GAZP']=[3700,3600,2900]
    cap['IMOEX2']['LKOH']=[5200,5100,4200]
    cap['IMOEX2']['T']=[609.3,975.3,862.8]
    cap['IMOEX2']['GMKN']=[2400,1900,1600]
    cap['IMOEX2']['PIKK']=[555.7,287.6,412]
    cap['MOEXOG']['GAZP']=[3700,3600,2900]
    cap['MOEXOG']['NVTK']=[4400,3600,3100]
    cap['MOEXOG']['ROSN']=[6200,5700,4500]
    cap['MOEXOG']['TATN']=[1700,1600,1400]
    cap['MOEXOG']['LKOH']=[5200,5100,4200]
    cap['MOEXMM']['GMKN']=[2400,1900,1600]
    cap['MOEXMM']['MAGN']=[657.1,450.8,367.3]
    cap['MOEXMM']['NLMK']=[1400,793.5,669,]
    cap['MOEXMM']['PLZL']=[1100,1900,2600]
    cap['MOEXMM']['MTLR']=[174.7,174.7,48.4]
    cap['MOEXIT']['YDEX']=[1500,1800,1600]
    cap['MOEXIT']['VKCO']=[142.3,70,65.1]
    cap['MOEXIT']['ASTR']=[144.9,90.3,79.5]
    cap['MOEXIT']['POSI']=[183.2,92.6,90.8]
    cap['MOEXIT']['SOFL']=[46.6,44.1,40.1]
    cap['MOEXFN']['VTBR']=[657.8,505.5,523.8]
    cap['MOEXFN']['SBER']=[6600,7000,7200]
    cap['MOEXFN']['T']=[609.3,975.3,862.8]
    cap['MOEXFN']['SVCB']=[392.8,355.9,334.5]
    cap['MOEXFN']['BSPB']=[140.9,189.5,166.3]
    cap['MOEXEU']['IRAO']=[418.9,392.7,330.8]
    cap['MOEXEU']['UPRO']=[145.2,144.9,95.1]
    cap['MOEXEU']['FEES']=[245.1,157.3,141.2]
    cap['MOEXEU']['MSNG']=[87.1,67.9,83.1]
    cap['MOEXEU']['HYDR']=[339.7,226,200.2]
    cap['MOEXRE']['PIKK']=[555.7,287.6,412]
    cap['MOEXRE']['SMLT']=[235.5,79.3,74.8]
    cap['MOEXRE']['LSRG']=[95.2,88.9,76.7]
    cap['MOEXRE']['MSTT']=[89.1,61.9,35.7]
    cap['MOEXTN']['AFLT']=[154.7,299.4,251.5]
    cap['MOEXTN']['FLOT']=[318.2,221.6,190.6]
    cap['MOEXTN']['SVAV']=[29.8,22.2,18.6]
    cap['MOEXTN']['FESH']=[142,256.6,167.2]
    cap['MOEXTN']['KMAZ']=[129.8,79.9,61.4]
    cap['MOEXCN']['MVID']=[38.7,18.7,16.2]
    cap['MOEXCN']['LENT']=[87.6,163.6,165.2]
    cap['MOEXCN']['MGNT']=[861,466.7,370.1]
    cap['MOEXCN']['PHOR']=[855.1,904.7,913]
    cap['MOEXCN']['BELU']=[54.7,91,53.2]
    
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
    
    startpars='2023-05-29'
    forecast_days = 14
    today = date.today()
    debug_print(f"📅 Сегодня: {today}")
    debug_print(f"📅 Стартовая дата для данных: {startpars}")
    debug_print(f"📅 Дата анализа: {today}")
    debug_print(f"📊 Количество секторов: {len(sectorsi)}")
    
    from math import comb
    
    def add_cap_columns(df, cap_dict):
        for key, values in cap_dict.items():
            column_name = f'cap{key}'
            df[column_name] = df['begin'].dt.year.map({
                2023: values[0],
                2024: values[1],
                2025: values[2]
            })
        return df
    
    # ========== ИСПРАВЛЕННАЯ ЗАГРУЗКА ИНДЕКСОВ СЕКТОРОВ ==========
    ind = {}
    for sector, tickers in sectorsi.items():
        debug_print(f"🔧 Обрабатываю сектор: {sector}")
        sector_data = []
        
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
        combined_frame = add_cap_columns(combined_frame, cap[sector])
        
        # Загружаем реальный индекс сектора
        sec_real = fetch_moex_data_paginated(sector, startpars, today.strftime('%Y-%m-%d'), is_index=True)
        if sec_real.empty:
            debug_print(f"  ⚠️ Нет данных по реальному индексу {sector}")
            # Используем простую альтернативу
            ind[sector] = combined_frame[['begin']].copy()
            ind[sector]['sector'] = 0
            continue
        
        sec_real = sec_real[['begin', 'close']].rename(columns={'close': 'sector'})
        
        # Объединяем с реальным индексом
        combined_with_real = pd.merge(combined_frame, sec_real, on='begin', how='inner')
        
        if combined_with_real.empty:
            debug_print(f"  ⚠️ Нет общих данных с реальным индексом {sector}")
            ind[sector] = combined_frame[['begin']].copy()
            ind[sector]['sector'] = 0
            continue
        
        # Обучаем модель
        combined_with_real.set_index('begin', inplace=True)
        combined_with_real.dropna(inplace=True)
        
        if len(combined_with_real) < 30:
            debug_print(f"  ⚠️ Мало данных для обучения ({len(combined_with_real)} строк)")
            ind[sector] = combined_frame[['begin']].copy()
            ind[sector]['sector'] = 0
            continue
        
        X = combined_with_real.drop(columns=['sector'])
        y = combined_with_real['sector']
        
        model = LGBMRegressor(random_state=1, verbose=-1)
        model.fit(X, y)
        
        # Предсказываем на тех же данных
        pred = model.predict(X)
        
        # Сохраняем предсказания
        sec_pred = pd.DataFrame({
            'begin': combined_with_real.index,
            'sector': pred
        })
        
        ind[sector] = sec_pred.drop_duplicates(subset='begin', keep='first')
        debug_print(f"  ✅ Сектор {sector}: {len(ind[sector])} предсказаний, последняя дата: {ind[sector]['begin'].iloc[-1].date()}")
    
    # ========== ЗАГРУЗКА ДАННЫХ БРЕНТ ==========
    debug_print("🛢️ Загружаю данные по Brent...")
    bre = fetch_futures_data_paginated('BRU5', startpars, today.strftime('%Y-%m-%d'))
    if bre.empty:
        debug_print("⚠️ Не удалось загрузить данные Brent, создаю пустой DataFrame")
        bre = pd.DataFrame(columns=['begin', 'close'])
    else:
        bre = bre[['begin', 'close']].rename(columns={'close': 'brent'})
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

for sector in data_framest:
    for ticker in data_framest[sector]:
        if not data_framest[sector][ticker].empty:
            last_date = data_framest[sector][ticker].index[-1].date()
            debug_print(f"  📅 {ticker}: последняя дата данных: {last_date}")
            break  # Чтобы не выводить все
    break
# ========== СОЗДАНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ ==========
debug_print("🎯 Создаю целевую переменную...")
sppa = []
spv = {}
smpe = 0
data_frames = {}

for sector, tickers in sectors.items():
    sector_data = {}
    for ticker in tickers:
        if ticker in data_framest.get(sector, {}):
            result = data_framest[sector][ticker].copy()
            result['target'] = 0.0
            
            # ИСПРАВЛЕНИЕ: Правильно рассчитываем цель для каждой строки
            for i in range(len(result)):
                if i + forecast_days < len(result):
                    current_close = result.iloc[i]['ticker']
                    # Ищем максимум в ближайшие forecast_days дней
                    future_start = i + 1
                    future_end = min(i + 1 + forecast_days, len(result))
                    
                    if future_start < future_end:
                        future_prices = result.iloc[future_start:future_end]['ticker']
                        if len(future_prices) > 0:
                            max_future = future_prices.max()
                            result.at[result.index[i], 'target'] = (max_future - current_close) * 100 / current_close
            
            # Сохраняем ВСЕ данные, включая последние дни (даже с target=0)
            # Они понадобятся для предсказания
            sector_data[ticker] = result
            
            # Отладочная информация
            last_date = result.index[-1].date() if len(result) > 0 else "нет данных"
            debug_print(f"  📊 {ticker}: {len(result)} строк, последняя дата: {last_date}")
    
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
                
                model = LGBMRegressor(random_state=1)
                grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=4)
                grid_search.fit(X, y)
                best_model = grid_search.best_estimator_
                model = best_model
                
                threshold = 12
                weights = np.where(y > threshold, 2, 1)
                model.fit(X, y, sample_weight=weights)
                
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
                mse_scores = -cv_scores
                
                if mse_scores.mean() > 7.5:
                    sppa.append(ticker)
                    sppa.append(mse_scores.mean())
                
                smpe += mse_scores.mean()
                
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
            # ИСПРАВЛЕНИЕ: Берем последние данные из data_framest, а не из secd
            if sector in data_framest and ticker in data_framest[sector]:
                # Берем последнюю строку из исходных данных (самые свежие)
                last_data = data_framest[sector][ticker].iloc[-1:].copy()
                
                # Убедимся, что у нас те же признаки, что при обучении
                # Добавляем временные фичи для последней строки
                X_cols = models[sector][ticker].feature_names_in_
                
                # Если в last_data нет всех нужных колонок, создаем их
                for col in X_cols:
                    if col not in last_data.columns and col != 'target':
                        # Пытаемся создать недостающие фичи
                        if '_lag_' in col:
                            base_col = col.split('_lag_')[0]
                            lag = int(col.split('_lag_')[1])
                            # Для лагов нужно взять значения из предыдущих строк
                            if len(data_framest[sector][ticker]) > lag:
                                last_data[col] = data_framest[sector][ticker][base_col].iloc[-lag-1]
                            else:
                                last_data[col] = np.nan
                        elif '_rolling5' in col:
                            base_col = col.replace('_rolling5', '')
                            if len(data_framest[sector][ticker]) >= 5:
                                last_data[col] = data_framest[sector][ticker][base_col].iloc[-5:].mean()
                            else:
                                last_data[col] = np.nan
                
                # Заполняем NaN средними значениями
                last_data = last_data.fillna(method='ffill').fillna(method='bfill')
                
                # Отладочная информация
                debug_print(f"  📈 {ticker}: предсказание для {last_data.index[-1].date()}")
                
                try:
                    prediction = models[sector][ticker].predict(last_data[X_cols])
                    predictions[ticker] = float(prediction[0])
                except Exception as e:
                    debug_print(f"  ⚠️ Ошибка предсказания {ticker}: {e}")
                    # Пробуем альтернативный способ
                    try:
                        last_data_simple = secd[ticker].drop(columns=['target']).iloc[-1:].copy()
                        prediction = models[sector][ticker].predict(last_data_simple)
                        predictions[ticker] = float(prediction[0])
                    except:
                        predictions[ticker] = 0.0
            else:
                # Запасной вариант
                try:
                    last_data = secd[ticker].drop(columns=['target']).iloc[-1:].copy()
                    prediction = models[sector][ticker].predict(last_data)
                    predictions[ticker] = float(prediction[0])
                except:
                    predictions[ticker] = 0.0
    
    # ========== КЛАССИФИКАЦИЯ РЕЗУЛЬТАТОВ ==========
    sr, mn = {}, {}
    for ticker, pr in predictions.items():
        if pr > 3.5 and pr < 7.5:
            sr[ticker] = pr
        elif pr >= 7.5:
            mn[ticker] = pr
    
    # ========== ФОРМИРОВАНИЕ РЕЗУЛЬТАТОВ ==========
    from datetime import datetime, timezone
    
    results_data = {
        "date": str(date.today()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "predictions": predictions,
        "medium_potential": dict(sr),
        "high_potential": dict(mn),
        "poor_accuracy": sppa,
        "avg_mae": smpe / 100,
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
    
    # Сохраняем отладочные данные
    debug_data = {
        'analysis_date': str(today),
        'predictions_count': len(predictions),
        'usd_data_last_date': str(usd['begin'].max()) if not usd.empty else "нет данных",
        'results_summary': results_data['summary']
    }
    save_debug_data(debug_data, f"debug_{today}.json")
    
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
