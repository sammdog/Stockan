"""
ПОЛНЫЙ АНАЛИЗ АКЦИЙ ДЛЯ GITHUB ACTIONS
Максимально приближен к коду Google Colab
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

# ========== КОНФИГУРАЦИЯ И ПРЕДУПРЕЖДЕНИЯ ==========
def warn(*args, **kwargs):
    pass

warnings.warn = warn
warnings.filterwarnings('ignore')

# ========== ДИАГНОСТИЧЕСКИЕ ФУНКЦИИ ==========
def debug_print(message):
    """Вывод отладочной информации"""
    print(f"🔍 [DEBUG] {message}")

# ========== ФУНКЦИИ ДЛЯ ЗАГРУЗКИ ДАННЫХ ==========
def fetch_data_two_parts(ticker, start_date, end_date, split_date='2024-09-25', is_index=False):
    """Загружает данные двумя запросами как в Google Colab"""
    try:
        # Первый запрос
        if is_index:
            url1 = f"http://iss.moex.com/iss/engines/stock/markets/index/securities/{ticker}/candles.json?from={start_date}&till={split_date}&interval=24"
        else:
            url1 = f"http://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json?from={start_date}&till={split_date}&interval=24"
        
        # Второй запрос
        if is_index:
            url2 = f"http://iss.moex.com/iss/engines/stock/markets/index/securities/{ticker}/candles.json?from={split_date}&till={end_date}&interval=24"
        else:
            url2 = f"http://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json?from={split_date}&till={end_date}&interval=24"
        
        debug_print(f"  📥 Загрузка {ticker}: {start_date} - {split_date}")
        response1 = requests.get(url1, timeout=30)
        
        debug_print(f"  📥 Загрузка {ticker}: {split_date} - {end_date}")
        response2 = requests.get(url2, timeout=30)
        
        all_data = []
        
        for response in [response1, response2]:
            if response.status_code == 200:
                data = response.json()
                if 'candles' in data and 'data' in data['candles'] and data['candles']['data']:
                    part_data = [{k: r[i] for i, k in enumerate(data['candles']['columns'])} 
                               for r in data['candles']['data']]
                    df_part = pd.DataFrame(part_data)
                    all_data.append(df_part)
                    debug_print(f"    ✅ Загружено {len(df_part)} строк")
                else:
                    debug_print(f"    ⚠️ Нет данных в части запроса")
            else:
                debug_print(f"    ⚠️ Ошибка HTTP {response.status_code}")
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result['begin'] = pd.to_datetime(result['begin'])
            result = result.sort_values('begin').drop_duplicates('begin')
            debug_print(f"  📊 Итого по {ticker}: {len(result)} строк")
            return result
        else:
            debug_print(f"  ❌ Нет данных по {ticker}")
            return pd.DataFrame()
            
    except Exception as e:
        debug_print(f"  ❌ Ошибка при загрузке {ticker}: {e}")
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
def get_currency_rate(date):
    """Получает курс доллара на указанную дату"""
    url = f'https://www.cbr.ru/scripts/XML_daily.asp?date_req={date.strftime("%d/%m/%Y")}'
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            for valute in root.findall('Valute'):
                char_code = valute.find('CharCode').text
                if char_code == 'USD':
                    value = valute.find('Value').text
                    return float(value.replace(',', '.'))
    except Exception as e:
        debug_print(f"Ошибка при запросе курса на {date.strftime('%d/%m/%Y')}: {e}")
    return None

# ========== ОСНОВНОЙ КОД ==========
def main_pipeline():
    """
    Основная функция для запуска всего пайплайна
    Максимально приближена к Google Colab версии
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
    debug_print(f"📅 Стартовая дата: {startpars}")
    
    # ========== ЗАГРУЗКА ИНДЕКСОВ СЕКТОРОВ ==========
    debug_print("🔧 Загружаю и рассчитываю индексы секторов...")
    ind = {}
    
    for sector, tickers in sectorsi.items():
        debug_print(f"  📈 Обрабатываю сектор: {sector}")
        sector_data = []
        
        for ticker in tickers:
            df = fetch_data_two_parts(ticker, startpars, today.strftime('%Y-%m-%d'), is_index=False)
            if not df.empty:
                df = df[['begin', 'close', 'volume', 'value']].rename(
                    columns={'close': ticker, 'volume': f'vol{ticker}', 'value': f'val{ticker}'}
                )
                sector_data.append(df)
        
        if not sector_data:
            debug_print(f"  ❌ Нет данных по сектору {sector}")
            continue
        
        combined_frame = sector_data[0]
        for df in sector_data[1:]:
            combined_frame = combined_frame.merge(df, on='begin', how='inner')
        
        if combined_frame.empty:
            debug_print(f"  ❌ Нет общих данных в секторе {sector}")
            continue
        
        combined_frame.dropna(inplace=True)
        combined_frame = add_cap_columns(combined_frame, cap[sector])
        
        r = combined_frame['begin']
        combined_frame.set_index('begin', inplace=True)
        combined_frame.dropna(inplace=True)
        ccomb = combined_frame.copy()
        
        # Загружаем реальный индекс
        sec_real = fetch_data_two_parts(sector, startpars, today.strftime('%Y-%m-%d'), is_index=True)
        if sec_real.empty:
            debug_print(f"  ⚠️ Нет данных по реальному индексу {sector}")
            sec_pred = pd.DataFrame({'begin': r, 'sector': 0})
            ind[sector] = sec_pred.drop_duplicates(subset='begin', keep='first')
            continue
        
        sec_real = sec_real[['begin', 'close']].rename(columns={'close': 'sector'})
        combined_frame = pd.merge(combined_frame.reset_index(), sec_real, on='begin', how='inner')
        combined_frame.set_index('begin', inplace=True)
        combined_frame.dropna(inplace=True)
        
        if len(combined_frame) < 30:
            debug_print(f"  ⚠️ Мало данных для обучения ({len(combined_frame)} строк)")
            sec_pred = pd.DataFrame({'begin': r, 'sector': 0})
            ind[sector] = sec_pred.drop_duplicates(subset='begin', keep='first')
            continue
        
        X = combined_frame.drop(columns=['sector'])
        y = combined_frame['sector']
        
        model = LGBMRegressor(random_state=1, verbose=-1)
        model.fit(X, y)
        
        pred = model.predict(ccomb)
        sec_pred = pd.DataFrame({'begin': ccomb.index, 'sector': pred})
        ind[sector] = sec_pred.drop_duplicates(subset='begin', keep='first')
        debug_print(f"  ✅ Сектор {sector}: {len(ind[sector])} предсказаний")
    
    # ========== ЗАГРУЗКА ДАННЫХ BRENT ==========
    debug_print("🛢️ Загружаю данные по Brent...")
    brent_contracts = ['BRU3', 'BRU4', 'BRU5', 'BRJ6']
    bre_all = []
    
    for contract in brent_contracts:
        debug_print(f"  🔄 Загружаю контракт {contract}...")
        try:
            j = requests.get(f'http://iss.moex.com/iss/engines/futures/markets/forts/securities/{contract}/candles.json?from={startpars}&till={today}&interval=24').json()
            if 'candles' in j and 'data' in j['candles'] and j['candles']['data']:
                dataj = [{k: r[i] for i, k in enumerate(j['candles']['columns'])} for r in j['candles']['data']]
                framej = pd.DataFrame(dataj)
                framej['begin'] = pd.to_datetime(framej['begin'])
                bre_all.append(framej)
                debug_print(f"    ✅ Загружено {len(framej)} строк")
            else:
                debug_print(f"    ⚠️ Нет данных по контракту {contract}")
        except Exception as e:
            debug_print(f"    ❌ Ошибка загрузки {contract}: {e}")
    
    if bre_all:
        bre = pd.concat(bre_all, ignore_index=True)
        bre = bre[['begin', 'close']].rename(columns={'close': 'brent'})
        bre = bre.sort_values('begin').drop_duplicates(subset='begin', keep='last')
        debug_print(f"📊 Данных Brent: {len(bre)} строк")
    else:
        debug_print("❌ Не удалось загрузить данные по Brent")
        bre = pd.DataFrame(columns=['begin', 'brent'])
    
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
    
    if usdtt.empty:
        startusd = (today - timedelta(days=30)).strftime('%Y-%m-%d')
    else:
        startusd = usdtt['begin'].iloc[-1]
    
    try:
        if isinstance(startusd, str):
            start_date = datetime.strptime(startusd, '%Y-%m-%dT%H:%M:%S.%f').date()
        else:
            start_date = startusd.date() if hasattr(startusd, 'date') else (today - timedelta(days=30))
    except:
        start_date = today - timedelta(days=30)
    
    end_date = today
    usd_rates = {}
    current_date = start_date + timedelta(days=1) if start_date < end_date else start_date
    
    debug_print(f"📅 Загружаю данные USD с {current_date} по {end_date}")
    
    while current_date <= end_date:
        rate = get_currency_rate(current_date)
        if rate is not None:
            usd_rates[current_date.strftime("%d/%m/%Y")] = rate
            debug_print(f"✅ Курс на {current_date}: {rate}")
        else:
            debug_print(f"⚠️ Не получен курс на {current_date}")
        current_date += timedelta(days=1)
    
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
    
    with open(file_pa, 'w') as f:
        json.dump(usd.to_json(date_format='iso'), f)
    
    debug_print(f"💾 Файл USD сохранен. Теперь строк: {len(usd)}")
    
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
            df = fetch_data_two_parts(ticker, startpars, today.strftime('%Y-%m-%d'), is_index=False)
            if df.empty:
                debug_print(f"  ⚠️ Пропускаю {ticker} - нет данных")
                continue
            
            df = df[['begin', 'close', 'volume']].rename(columns={'close': 'ticker'})
            
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
            
            # Добавляем лаги (ТОЧНО как в Google Colab)
            for column in df.columns:
                for lag in [7, 14, 28, 35, 31]:
                    if 'lag' not in column and 'rolling' not in column and 'sector_x' not in column:
                        df[f'{column}_lag_{lag}'] = df[column].shift(lag)
            
            df.dropna(inplace=True)
            sector_data[ticker] = df
        
        data_framest[sector] = sector_data
    
    debug_print(f"✅ Обработаны данные по {sum(len(v) for v in data_framest.values())} акциям")
    
    # ========== СОЗДАНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ (ТОЧНО как в Google Colab) ==========
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
                result['target'] = 0
                
                # ТОЧНО как в Google Colab: if i + 30 < len(result)
                for i in range(len(result)):
                    current_close = result.iloc[i]['ticker']
                    if i + 30 < len(result):  # Проверяем следующие 30 дней
                        future_prices = result.iloc[i + 1:i + 1 + forecast_days]['ticker']
                        result.at[result.index[i], 'target'] = (max(future_prices) - current_close) * 100 / current_close
                
                result.dropna(inplace=True)
                sector_data[ticker] = result
        
        data_frames[sector] = sector_data
    
    # ========== ОБУЧЕНИЕ МОДЕЛЕЙ (ТОЧНО как в Google Colab) ==========
    debug_print("🤖 Обучаю модели...")
    param_grid = {
        'num_leaves': [31, 50, 21],
        'max_depth': [-1, 5, 10],
        'learning_rate': [0.3, 0.1, 0.2],
        'n_estimators': [250, 200]
    }
    
    model_params_dir = 'model_params'
    if not os.path.exists(model_params_dir):
        os.makedirs(model_params_dir)
    
    models = {}
    models_learned_today = 0
    models_loaded = 0
    
    for sector, secd in data_frames.items():
        models[sector] = {}
        for ticker in sectors[sector]:
            if ticker in secd:
                # ТОЧНО как в Google Colab: iloc[:-(1+forecast_days)]
                X = secd[ticker].drop(columns=['target']).iloc[:-(1 + forecast_days)]
                y = secd[ticker]['target'].iloc[:-(1 + forecast_days)]
                
                param_file = f'{model_params_dir}/{sector}_{ticker}_params.json'
                
                if os.path.exists(param_file):
                    with open(param_file, 'r') as f:
                        loaded_params = json.load(f)
                    model = LGBMRegressor(random_state=1, **loaded_params)
                    models_loaded += 1
                else:
                    debug_print(f"  🔍 Подбираю параметры для {ticker}...")
                    model = LGBMRegressor(random_state=1)
                    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=4, verbose=0)
                    grid_search.fit(X, y)
                    best_model = grid_search.best_estimator_
                    model = best_model
                    best_params = grid_search.best_params_
                    
                    with open(param_file, 'w') as f:
                        json.dump(best_params, f, indent=2)
                    
                    models_learned_today += 1
                
                # Веса для важных примеров (порог 12)
                threshold = 12
                weights = np.where(y > threshold, 2, 1)
                model.fit(X, y, sample_weight=weights)
                
                # Кросс-валидация
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
                mse_scores = -cv_scores
                
                # ТОЧНО как в Google Colab: порог >9.5
                if mse_scores.mean() > 9.5:
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
    
    # Рассчитываем точность как в Google Colab
    avg_mae = smpe / 100  # ТОЧНО как в Google Colab
    accuracy = max(0, 100 - avg_mae)
    
    # ========== ФОРМИРОВАНИЕ РЕЗУЛЬТАТОВ ==========
    results_data = {
        "date": str(date.today()),
        "timestamp": datetime.now().isoformat(),
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
    
    # Сохраняем результаты
    with open('latest_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Выводим результаты как в Google Colab
    print(f"\n{smpe/100}")
    print(sppa)
    
    if len(mn) < 6:
        print('BAD DAY')
    elif len(mn) < 9:
        print('50/50 DAY')
    else:
        print('100% GOOD DAY')
    
    print(len(mn))
    print(sr)
    print(mn)
    
    debug_print(f"✅ Анализ завершен: {len(predictions)} акций")
    debug_print(f"🎯 Лучших акций: {len(mn)}")
    debug_print(f"📊 Средних акций: {len(sr)}")
    debug_print(f"📈 Точность модели: {accuracy:.1f}%")
    
    return results_data

def main():
    """Основная функция"""
    print("=" * 60)
    print("🚀 ЗАПУСК ПОЛНОГО АНАЛИЗА")
    print("=" * 60)
    
    # Проверяем необходимые файлы
    if not os.path.exists('usd_data.json'):
        print("📋 Создаю usd_data.json...")
        initial_data = pd.DataFrame(columns=['begin', 'usdt'])
        with open('usd_data.json', 'w') as f:
            json.dump(initial_data.to_json(date_format='iso'), f)
    
    if not os.path.exists('model_params'):
        print("📁 Создаю папку model_params...")
        os.makedirs('model_params')
    
    try:
        results = main_pipeline()
        return results
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
