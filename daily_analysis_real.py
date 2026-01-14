def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd
import requests
from datetime import date, timedelta, datetime
import matplotlib.pyplot as plt
import lightgbm
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, precision_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os
import json
import xml.etree.ElementTree as ET
import time

# ========== ФУНКЦИИ ДЛЯ ЗАГРУЗКИ ДАННЫХ (ИДЕНТИЧНЫЕ GOOGLE COLAB) ==========
def fetch_moex_data_colab(ticker, start_date, end_date, is_index=False):
    """
    Загружает данные MOEX ТОЧНО как в Google Colab:
    2 запроса с фиксированной датой 2024-09-25
    """
    # Определяем базовый URL
    if is_index:
        base_url = "http://iss.moex.com/iss/engines/stock/markets/index/securities"
    else:
        base_url = "http://iss.moex.com/iss/engines/stock/markets/shares/securities"
    
    # ФИКСИРОВАННАЯ ДАТА РАЗБИВКИ КАК В GOOGLE COLAB
    split_date = '2024-09-25'
    
    all_data = []
    
    # Первый запрос: от start_date до 2024-09-25
    url1 = f"{base_url}/{ticker}/candles.json?from={start_date}&till={split_date}&interval=24"
    try:
        response1 = requests.get(url1, timeout=30)
        if response1.status_code == 200:
            data1 = response1.json()
            if 'candles' in data1 and 'data' in data1['candles'] and data1['candles']['data']:
                part_data1 = [{k: r[i] for i, k in enumerate(data1['candles']['columns'])} 
                            for r in data1['candles']['data']]
                df1 = pd.DataFrame(part_data1)
                all_data.append(df1)
    except Exception as e:
        print(f"Ошибка запроса 1 для {ticker}: {e}")
    
    # Второй запрос: от 2024-09-25 до end_date
    url2 = f"{base_url}/{ticker}/candles.json?from={split_date}&till={end_date}&interval=24"
    try:
        response2 = requests.get(url2, timeout=30)
        if response2.status_code == 200:
            data2 = response2.json()
            if 'candles' in data2 and 'data' in data2['candles'] and data2['candles']['data']:
                part_data2 = [{k: r[i] for i, k in enumerate(data2['candles']['columns'])} 
                            for r in data2['candles']['data']]
                df2 = pd.DataFrame(part_data2)
                all_data.append(df2)
    except Exception as e:
        print(f"Ошибка запроса 2 для {ticker}: {e}")
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result['begin'] = pd.to_datetime(result['begin'])
        result = result.sort_values('begin').drop_duplicates('begin')
        return result
    else:
        return pd.DataFrame()

# ========== КОНФИГУРАЦИЯ КАПИТАЛИЗАЦИЙ ==========
cap = {
    'IMOEX2': {'VTBR': [657.8, 505.5, 503.8, 471.3],
               'SBER': [6600, 7000, 6900, 6900],
               'GAZP': [3700, 3600, 3100, 2900],
               'LKOH': [5200, 5100, 4200, 3700],
               'T': [609.3, 975.3, 862.8, 864.6],
               'GMKN': [2400, 1900, 1600, 2400],
               'PIKK': [555.7, 287.6, 322, 299.9]},
    'MOEXOG': {'GAZP': [3700, 3600, 3100, 2900],
               'NVTK': [4400, 3600, 3500, 3500],
               'ROSN': [6200, 5700, 4500, 4200],
               'TATN': [1700, 1600, 1400, 1200],
               'LKOH': [5200, 5100, 4200, 3700]},
    'MOEXMM': {'GMKN': [2400, 1900, 1600, 2400],
               'MAGN': [657.1, 450.8, 367.3, 307.2],
               'NLMK': [1400, 793.5, 669, 623.2],
               'PLZL': [1100, 1900, 2600, 3500],
               'MTLR': [174.7, 63.5, 42.4, 40]},
    'MOEXIT': {'YDEX': [1500, 1800, 1800, 1800],
               'VKCO': [142.3, 70, 100, 168.4],
               'ASTR': [144.9, 90.3, 79.5, 51.7],
               'POSI': [183.2, 92.6, 88.8, 71.8],
               'SOFL': [46.6, 44.1, 40.1, 31.3]},
    'MOEXFN': {'VTBR': [657.8, 505.5, 503.8, 471.3],
               'SBER': [6600, 7000, 6900, 6900],
               'T': [609.3, 975.3, 862.8, 864.6],
               'SVCB': [392.8, 355.9, 334.5, 277.9],
               'BSPB': [140.9, 189.5, 166.3, 135.6]},
    'MOEXEU': {'IRAO': [418.9, 392.7, 330.8, 333.9],
               'UPRO': [145.2, 144.9, 95.1, 91.3],
               'FEES': [245.1, 157.3, 141.2, 162.6],
               'MSNG': [67.9, 74.9, 81.1, 86.3],
               'HYDR': [339.7, 226, 200.2, 185.9]},
    'MOEXRE': {'PIKK': [555.7, 287.6, 322, 299.9],
               'SMLT': [235.5, 79.3, 74.8, 57.3],
               'LSRG': [95.2, 88.9, 76.7, 69],
               'MSTT': [61.9, 37.2, 33, 31]},
    'MOEXTN': {'AFLT': [154.7, 299.4, 251.5, 222.9],
               'FLOT': [318.2, 221.6, 190.6, 175.7],
               'SVAV': [29.8, 22.2, 18.6, 18.2],
               'FESH': [256.6, 157.7, 156.6, 155.2],
               'KMAZ': [129.8, 79.9, 61.4, 59.3]},
    'MOEXCN': {'MVID': [38.7, 18.7, 16.2, 13.8],
               'LENT': [87.6, 163.6, 185.2, 231.3],
               'MGNT': [861, 466.7, 370.1, 307.8],
               'PHOR': [855.1, 904.7, 903, 823.4],
               'BELU': [91, 72.8, 53.5, 52]}
}

# Сектора для индексов
sectorsi = {
    'IMOEX2': ['VTBR', 'SBER', 'LKOH', 'T', 'GAZP', 'GMKN', 'PIKK'],
    'MOEXOG': ['GAZP', 'NVTK', 'ROSN', 'TATN', 'LKOH'],
    'MOEXMM': ['GMKN', 'MAGN', 'NLMK', 'PLZL', 'MTLR'],
    'MOEXIT': ['YDEX', 'VKCO', 'ASTR', 'POSI', 'SOFL'],
    'MOEXFN': ['VTBR', 'SBER', 'T', 'SVCB', 'BSPB'],
    'MOEXEU': ['IRAO', 'UPRO', 'FEES', 'MSNG', 'HYDR'],
    'MOEXRE': ['PIKK', 'SMLT', 'LSRG', 'MSTT'],
    'MOEXTN': ['AFLT', 'FLOT', 'SVAV', 'FESH', 'KMAZ'],
    'MOEXCN': ['MVID', 'LENT', 'MGNT', 'PHOR', 'BELU']
}

# Сектора для акций
sectors = {
    'MOEXOG': ['TATN', 'SNGSP', 'SNGS', 'TRNFP', 'TATNP', 'SIBN', 'BANEP', 'BANE', 'EUTR', 'YAKG', 'NVTK', 'GAZP', 'ROSN'],
    'MOEXMM': ['GMKN', 'MAGN', 'RUAL', 'ALRS', 'SELG', 'RASP', 'UGLD', 'BLNG', 'NLMK', 'CHMF', 'MTLR', 'ENPG'],
    'MOEXIT': ['VKCO', 'ASTR', 'POSI', 'SOFL', 'RTKMP', 'MTSS', 'DATA', 'HEAD', 'DIAS', 'MGTSP', 'CNTL', 'CNTLP', 'AFKS', 'RTKM', 'RBCM'],
    'MOEXFN': ['SBER', 'SVCB', 'MOEX', 'SBERP', 'MBNK', 'RENI', 'CARM', 'LEAS', 'SFIN', 'ZAYM', 'CBOM', 'MGKL'],
    'MOEXEU': ['IRAO', 'FEES', 'MSNG', 'HYDR', 'MRKV', 'TGKA', 'OGKB', 'ELFV', 'MSRS', 'TGKBP', 'MRKY', 'TGKN', 'LSNG', 'UPRO', 'MAGE'],
    'MOEXRE': ['LSRG', 'WTCMP', 'MSTT', 'WTCM', 'SMLT', 'PRFN'],
    'MOEXTN': ['FLOT', 'NMTP', 'ELMT', 'KMAZ', 'WUSH', 'SVAV', 'AFLT'],
    'MOEXCN': ['MVID', 'BELU', 'APTK', 'MDMG', 'KLVZ', 'AQUA', 'DELI', 'GEMC', 'HNFG', 'X5', 'LENT', 'RAGR', 'NKNCP', 'NKNC', 'KZOS', 'KZOSP', 'PRMD', 'VSEH', 'ABIO', 'LIFE']
}

startpars = '2023-05-29'
forecast_days = 14
today = date.today()

# ========== ФУНКЦИЯ ДОБАВЛЕНИЯ КАПИТАЛИЗАЦИЙ ==========
def add_cap_columns(df, cap_dict):
    for key, values in cap_dict.items():
        column_name = f'cap{key}'
        df[column_name] = df['begin'].dt.year.map({
            2023: values[0],
            2024: values[1],
            2025: values[2],
            2026: values[3]
        })
    return df

# ========== ЗАГРУЗКА ИНДЕКСОВ СЕКТОРОВ (ИДЕНТИЧНО GOOGLE COLAB) ==========
print("🔧 Загружаю и рассчитываю индексы секторов...")
ind = {}

for sector, tickers in sectorsi.items():
    print(f"  📈 Обрабатываю сектор: {sector}")
    sector_data = []
    
    # Загружаем данные по всем акциям сектора (ТОЧНО как в Colab)
    for ticker in tickers:
        df = fetch_moex_data_colab(ticker, startpars, today.strftime('%Y-%m-%d'), is_index=False)
        if not df.empty:
            df = df[['begin', 'close', 'volume', 'value']].rename(
                columns={'close': ticker, 'volume': f'vol{ticker}', 'value': f'val{ticker}'}
            )
            sector_data.append(df)
    
    if not sector_data:
        continue
    
    # Объединяем все акции сектора
    combined_frame = sector_data[0]
    for df in sector_data[1:]:
        combined_frame = pd.merge(combined_frame, df, on='begin', how='inner')
    
    combined_frame.dropna(inplace=True)
    
    # Добавляем капитализации
    if sector in cap:
        combined_frame = add_cap_columns(combined_frame, cap[sector])
    
    # Сохраняем копию без реального индекса для предсказания
    ccomb = combined_frame.copy()
    
    # Загружаем реальный индекс сектора (ТОЧНО как в Colab)
    sec_real = fetch_moex_data_colab(sector, startpars, today.strftime('%Y-%m-%d'), is_index=True)
    if not sec_real.empty:
        sec_real = sec_real[['begin', 'close']].rename(columns={'close': 'sector'})
        combined_frame = pd.merge(combined_frame, sec_real, on='begin', how='inner')
        
        combined_frame.set_index('begin', inplace=True)
        combined_frame.dropna(inplace=True)
        
        # Обучаем модель
        X = combined_frame.drop(columns=['sector'])
        y = combined_frame['sector']
        
        model = LGBMRegressor(random_state=1, verbose=-1)
        model.fit(X, y)
        
        # Предсказываем
        ccomb.set_index('begin', inplace=True)
        pred = model.predict(ccomb)
        
        sec_pred = pd.DataFrame({
            'begin': ccomb.index,
            'sector': pred
        })
    else:
        # Если нет реального индекса, создаем пустой
        sec_pred = pd.DataFrame({
            'begin': combined_frame['begin'],
            'sector': 0
        })
    
    ind[sector] = sec_pred.drop_duplicates(subset='begin', keep='first')
    print(f"  ✅ {sector}: {len(ind[sector])} предсказаний")

# ========== ЗАГРУЗКА BRENT (ИДЕНТИЧНО GOOGLE COLAB) ==========
print("🛢️ Загружаю данные по Brent (метод Colab)...")
brent_contracts = ['BRU3', 'BRU4', 'BRU5', 'BRJ6']
bre_all = []

for contract in brent_contracts:
    print(f"  🔄 Загружаю контракт {contract}...")
    df = fetch_moex_data_colab(contract, startpars, today.strftime('%Y-%m-%d'), is_index=False)
    if not df.empty:
        df = df[['begin', 'close']].rename(columns={'close': 'brent'})
        bre_all.append(df)

if bre_all:
    bre = pd.concat(bre_all, ignore_index=True)
    bre = bre[['begin', 'brent']]
    bre = bre.sort_values('begin').drop_duplicates(subset='begin', keep='last')
    print(f"✅ Данные Brent загружены: {len(bre)} строк")
else:
    bre = pd.DataFrame(columns=['begin', 'brent'])
    print("⚠️ Не удалось загрузить данные Brent")

# ========== ЗАГРУЗКА USD (ИДЕНТИЧНО GOOGLE COLAB) ==========
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
        print(f"Ошибка при запросе курса на {date.strftime('%d/%m/%Y')}: {e}")
    return None

print("💵 Загружаю и обновляю данные по USD...")
file_pa = 'usd_data.json'

# Загружаем сохраненные данные USD
try:
    with open(file_pa, 'r') as f:
        usdtt = pd.read_json(json.load(f))
    print(f"✅ Файл usd_data.json загружен: {len(usdtt)} строк")
except Exception as e:
    print(f"⚠️ Ошибка загрузки usd_data.json: {e}")
    usdtt = pd.DataFrame(columns=['begin', 'usdt'])

# Определяем последнюю дату в файле
if usdtt.empty:
    startusd = '2023-05-29'
    print("📅 Начинаю загрузку USD с 2023-05-29")
else:
    startusd = usdtt['begin'].iloc[-1] if not usdtt.empty else '2023-05-29'
    if isinstance(startusd, str):
        start_date = datetime.strptime(startusd, '%Y-%m-%dT%H:%M:%S.%f').date()
    else:
        start_date = startusd.date()
    print(f"📅 Последняя дата в файле USD: {start_date}")

end_date = today

# Загружаем новые данные USD
usd_rates = {}
current_date = start_date + timedelta(days=1) if not usdtt.empty else datetime.strptime('2023-05-29', '%Y-%m-%d').date()

print(f"📅 Загружаю данные USD с {current_date} по {end_date}")

while current_date <= end_date:
    rate = get_currency_rate(current_date)
    if rate is not None:
        usd_rates[current_date.strftime("%d/%m/%Y")] = rate
    current_date += timedelta(days=1)

# Объединяем старые и новые данные
if usd_rates:
    usd_new = pd.DataFrame(list(usd_rates.items()), columns=['begin', 'usdt'])
    usd_new['begin'] = pd.to_datetime(usd_new['begin'], format='%d/%m/%Y')
    
    if not usdtt.empty:
        usdtt['begin'] = pd.to_datetime(usdtt['begin'])
        usd = pd.concat([usdtt, usd_new], ignore_index=True).drop_duplicates(subset='begin', keep='last')
    else:
        usd = usd_new
else:
    usd = usdtt

# Сохраняем обновленные данные
with open(file_pa, 'w') as f:
    json.dump(usd.to_json(date_format='iso', orient='records'), f)

print(f"💾 Файл USD сохранен. Строк: {len(usd)}")

if 'begin' in usd.columns:
    usd['begin'] = pd.to_datetime(usd['begin'])
else:
    usd = pd.DataFrame(columns=['begin', 'usdt'])

# ========== ОБРАБОТКА ДАННЫХ АКЦИЙ (ИДЕНТИЧНО GOOGLE COLAB) ==========
print("📊 Обрабатываю данные по акциям...")
data_framest = {}

for sector, tickers in sectors.items():
    sector_data = {}
    for ticker in tickers:
        # Загружаем данные акции (ТОЧНО как в Colab)
        df = fetch_moex_data_colab(ticker, startpars, today.strftime('%Y-%m-%d'), is_index=False)
        if df.empty:
            continue
        
        df = df[['begin', 'close', 'volume']].rename(columns={'close': 'ticker'})
        
        # Объединяем с индексами
        if sector in ind and not ind[sector].empty:
            df = pd.merge(df, ind[sector], on='begin', how='inner')
        else:
            continue
        
        if 'IMOEX2' in ind and not ind['IMOEX2'].empty:
            df = pd.merge(df, ind['IMOEX2'], on='begin', how='inner')
        else:
            continue
        
        # Объединяем с USD и Brent
        if not usd.empty:
            df = pd.merge(df, usd, on='begin', how='inner')
        else:
            continue
        
        if not bre.empty:
            df = pd.merge(df, bre, on='begin', how='inner')
        else:
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

print(f"✅ Обработаны данные по {sum(len(v) for v in data_framest.values())} акциям")

# ========== СОЗДАНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ ==========
print("🎯 Создаю целевую переменную...")
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
            
            for i in range(len(result)):
                current_close = result.iloc[i]['ticker']
                if i + forecast_days < len(result):
                    future_prices = result.iloc[i + 1:i + 1 + forecast_days]['ticker']
                    result.at[result.index[i], 'target'] = (max(future_prices) - current_close) * 100 / current_close
            
            result.dropna(inplace=True)
            sector_data[ticker] = result
    
    data_frames[sector] = sector_data

# ========== ОБУЧЕНИЕ МОДЕЛЕЙ ==========
print("🤖 Обучаю модели...")
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

models = {}

for sector, secd in data_frames.items():
    models[sector] = {}
    for ticker in sectors[sector]:
        if ticker in secd:
            X = secd[ticker].drop(columns=['target']).iloc[:-(1 + forecast_days)]
            y = secd[ticker]['target'].iloc[:-(1 + forecast_days)]
            
            # Проверяем, есть ли сохраненные параметры
            param_file = f'{model_params_dir}/{sector}_{ticker}_params.json'
            
            if os.path.exists(param_file):
                try:
                    with open(param_file, 'r') as f:
                        loaded_params = json.load(f)
                    model = LGBMRegressor(random_state=1, **loaded_params)
                except:
                    model = LGBMRegressor(random_state=1)
            else:
                model = LGBMRegressor(random_state=1)
            
            # Веса для важных примеров
            threshold = 12
            weights = np.where(y > threshold, 2, 1)
            model.fit(X, y, sample_weight=weights)
            
            # Кросс-валидация
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
            mse_scores = -cv_scores
            
            if mse_scores.mean() > 9:
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
print("🔮 Делаю предсказания...")
predictions = {}

for sector, secd in data_frames.items():
    for ticker in sectors[sector]:
        if ticker in secd and ticker in models.get(sector, {}):
            try:
                last_data = secd[ticker].drop(columns=['target']).iloc[-1].copy()
                prediction = models[sector][ticker].predict([last_data])
                predictions[ticker] = float(prediction[0])
            except:
                predictions[ticker] = 0.0

# ========== ВЫВОД РЕЗУЛЬТАТОВ ==========
sr, mn = {}, {}
for ticker, pr in predictions.items():
    if pr > 5.5 and pr < 9.5:
        sr[ticker] = pr
    elif pr >= 9.5:
        mn[ticker] = pr

# Выводим результаты как в Google Colab
print(f"\n🎯 ВЫСОКИЙ ПОТЕНЦИАЛ (отсортировано по убыванию):")
for ticker, value in sorted(mn.items(), key=lambda x: x[1], reverse=True):
    print(f"  {ticker}: +{value:.1f}%")

# Сохраняем отладочные данные
debug_data = {
    'predictions': predictions,
    'high_potential': mn,
    'medium_potential': sr
}

debug_filename = f"debug_{today}.json"
with open(debug_filename, 'w') as f:
    json.dump(debug_data, f, indent=2, default=str)

print(f"🔍 [DEBUG] Сохранены отладочные данные в {debug_filename}")
print(f"\nСловарь предсказаний: {mn}")

# Дополнительная информация
print(f"\n📊 Статистика:")
print(f"Всего предсказаний: {len(predictions)}")
print(f"Высокий потенциал: {len(mn)} акций")
print(f"Средний потенциал: {len(sr)} акций")

if len(mn) < 6:
    print('BAD DAY')
elif len(mn) < 9:
    print('50/50 DAY')
else:
    print('100% GOOD DAY')
