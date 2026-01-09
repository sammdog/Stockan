"""
ПОЛНЫЙ АНАЛИЗ АКЦИЙ ДЛЯ GITHUB ACTIONS
Точная копия кода из Colab с минимальными изменениями
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

warnings.filterwarnings('ignore')

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
    # УБРАНО: import matplotlib.pyplot as plt
    import lightgbm
    from lightgbm import LGBMRegressor
    import sklearn
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
    from sklearn.metrics import mean_absolute_error, r2_score, precision_score, mean_squared_error, mean_absolute_percentage_error
    from sklearn.ensemble import GradientBoostingRegressor
    import joblib
    import os
    import json
    
    # === ВАШ ТОЧНЫЙ КОД НАЧИНАЕТСЯ ЗДЕСЬ ===
    
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
    #cap['MOEXRE']['CNRU']=[67.6,44.7,43.7]
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
    #cap['MOEXTL']['AFKS']=[231.6,142.4,146.7]
    #cap['MOEXTL']['MTSS']=[583.3,499.8,436.1]
    
    # Группировка акций по секторам
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
    
    ind={}
    for sector, tickers in sectorsi.items():
        sector_data = []
        for ticker in tickers:
            j = requests.get(f'http://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json?from={startpars}&till=2024-09-25&interval=24').json()
            jo = requests.get(f'http://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json?from=2024-09-25&till={today}&interval=24').json()
            dataj = [{k: r[i] for i, k in enumerate(j['candles']['columns'])} for r in j['candles']['data']]
            datajo = [{k: r[i] for i, k in enumerate(jo['candles']['columns'])} for r in jo['candles']['data']]
            framej = pd.DataFrame(dataj)
            framejo = pd.DataFrame(datajo)
            result = pd.concat([framej, framejo], ignore_index=True)
            result['begin'] = pd.to_datetime(result['begin'])
            result=result[['begin', 'close','volume','value']].rename(columns={'close': ticker,'volume':f'vol{ticker}','value': f'val{ticker}'})
            sector_data.append(result)
        
        # Объединяем все акции сектора
        combined_frame = sector_data[0]
        for df in sector_data[1:]:
            combined_frame = combined_frame.merge(df, on='begin', how='inner')
        
        combined_frame.dropna(inplace=True)
        combined_frame = add_cap_columns(combined_frame, cap[sector])
        
        # КРИТИЧЕСКИ ИСПРАВЛЕНО: Сохраняем ВСЕ даты
        r = combined_frame['begin'].copy()
        
        combined_frame.set_index('begin', inplace=True)
        combined_frame.dropna(inplace=True)
        
        # Загружаем реальные данные индекса сектора
        j = requests.get(f'http://iss.moex.com/iss/engines/stock/markets/index/securities/{sector}/candles.json?from={startpars}&till=2024-09-25&interval=24').json()
        jo = requests.get(f'http://iss.moex.com/iss/engines/stock/markets/index/securities/{sector}/candles.json?from=2024-09-25&till={today}&interval=24').json()
        dataj = [{k: r[i] for i, k in enumerate(j['candles']['columns'])} for r in j['candles']['data']]
        datajo = [{k: r[i] for i, k in enumerate(jo['candles']['columns'])} for r in jo['candles']['data']]
        framej = pd.DataFrame(dataj)
        framejo = pd.DataFrame(datajo)
        sec_real = pd.concat([framej, framejo], ignore_index=True)
        sec_real['begin'] = pd.to_datetime(sec_real['begin'])
        sec_real = sec_real[['begin', 'close']].rename(columns={'close': 'sector'})
        
        # Сбрасываем индекс combined_frame для слияния
        combined_frame_for_merge = combined_frame.reset_index()
        
        # Сливаем с реальным индексом сектора - ОБЯЗАТЕЛЬНО ВНУТРЕННЕЕ СЛИЯНИЕ
        combined_frame_with_real_sec = pd.merge(combined_frame_for_merge, sec_real, on='begin', how='inner')
        combined_frame_with_real_sec.set_index('begin', inplace=True)
        combined_frame_with_real_sec.dropna(inplace=True)
        
        # Проверяем, есть ли достаточное количество данных
        if len(combined_frame_with_real_sec) < 30:
            print(f"⚠️ Мало данных для сектора {sector}: {len(combined_frame_with_real_sec)} строк")
            # Используем альтернативный подход или пропускаем
            ind[sector] = pd.DataFrame({'begin': r, 'sector': [0]*len(r)})
            continue
        
        # Обучаем модель на реальных данных индекса
        X = combined_frame_with_real_sec.drop(columns=['sector'])
        y = combined_frame_with_real_sec['sector']
        
        # Проверяем, что X и y имеют одинаковую длину
        if len(X) != len(y):
            print(f"⚠️ Несовпадение длин X ({len(X)}) и y ({len(y)}) для сектора {sector}")
            # Используем последние доступные данные
            min_len = min(len(X), len(y))
            X = X.iloc[:min_len]
            y = y.iloc[:min_len]
        
        model = LGBMRegressor(random_state=1, verbose=-1)
        model.fit(X, y)
        
        # Теперь ключевой момент: предсказываем на ВСЕХ датах из combined_frame
        # а не только на тех, что были в обучающей выборке
        
        # Получаем последние данные для предсказания
        # Используем ВСЕ данные из combined_frame (которые включают последние даты)
        data_for_prediction = combined_frame.copy()
        
        # Убедимся, что у нас те же колонки, что и в X
        # Добавляем недостающие колонки если нужно
        for col in X.columns:
            if col not in data_for_prediction.columns:
                # Если колонки нет, добавляем с нулевыми значениями или средним
                data_for_prediction[col] = 0
        
        # Удаляем лишние колонки, которых нет в X
        cols_to_drop = [col for col in data_for_prediction.columns if col not in X.columns]
        if cols_to_drop:
            data_for_prediction = data_for_prediction.drop(columns=cols_to_drop)
        
        # Упорядочиваем колонки как в X
        data_for_prediction = data_for_prediction[X.columns]
        
        # Предсказываем
        pred = model.predict(data_for_prediction)
        
        # Создаем DataFrame с предсказаниями на ВСЕ даты
        sec_pred = pd.DataFrame()
        sec_pred['begin'] = data_for_prediction.index
        sec_pred['sector'] = pred
        
        ind[sector] = sec_pred
        ind[sector] = ind[sector].drop_duplicates(subset='begin', keep='first')
        print(f"✅ Сектор {sector}: {len(ind[sector])} предсказаний, последняя дата: {ind[sector]['begin'].iloc[-1]}")
    
    j = requests.get(f'http://iss.moex.com/iss/engines/futures/markets/forts/securities/BRU5/candles.json?from={startpars}&till=2024-09-25&interval=24').json()
    jo = requests.get(f'http://iss.moex.com/iss/engines/futures/markets/forts/securities/BRU5/candles.json?from=2024-09-25&till={today}&interval=24').json()
    dataj = [{k: r[i] for i, k in enumerate(j['candles']['columns'])} for r in j['candles']['data']]
    datajo = [{k: r[i] for i, k in enumerate(jo['candles']['columns'])} for r in jo['candles']['data']]
    framej = pd.DataFrame(dataj)
    framejo = pd.DataFrame(datajo)
    bre = pd.concat([framej, framejo], ignore_index=True)
    bre['begin'] = pd.to_datetime(bre['begin'])
    bre=bre[['begin', 'close']].rename(columns={'close': 'brent'})
    
    # === ИСПРАВЛЕНИЕ: ЗАГРУЗКА USD ДАННЫХ ===
    file_pa = 'usd_data.json'
    
    try:
        with open(file_pa, 'r') as f:
            usdtt = pd.read_json(json.load(f))
    except Exception as e:
        print(f"Ошибка загрузки usd_data.json: {e}")
        # Создаем пустой DataFrame если файл не найден
        usdtt = pd.DataFrame(columns=['begin', 'usdt'])
    
    # Если файл пустой, берем последнюю доступную дату
    if usdtt.empty:
        startusd = (today - timedelta(days=30)).strftime('%Y-%m-%d')  # 30 дней назад
    else:
        startusd = usdtt['begin'].iloc[-1] if 'begin' in usdtt.columns and not usdtt.empty else (today - timedelta(days=30)).strftime('%Y-%m-%d')
    
    import xml.etree.ElementTree as ET
    from datetime import datetime
    
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
            print(f'Ошибка при запросе данных за {date.strftime("%d/%m/%Y")}: {response.status_code}')
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
    current_date = start_date + timedelta(days=1)  # начинаем со следующего дня после последней известной даты
    while current_date <= end_date:
        rate = get_currency_rate(current_date)
        if rate is not None:
            usd_rates[current_date.strftime("%d/%m/%Y")] = rate
        current_date += timedelta(days=1)
    
    # Если получили новые курсы, добавляем их
    if usd_rates:
        usd_new = pd.DataFrame(list(usd_rates.items()), columns=['begin', 'usdt'])
        usd_new['begin'] = pd.to_datetime(usd_new['begin'], format='%d/%m/%Y')
        
        # Объединяем старые и новые данные
        if not usdtt.empty and 'begin' in usdtt.columns:
            usdtt['begin'] = pd.to_datetime(usdtt['begin'])
            usd = pd.concat([usdtt, usd_new], ignore_index=True).drop_duplicates(subset='begin', keep='last')
        else:
            usd = usd_new
        
        # Сохраняем обновленные данные
        with open(file_pa, 'w') as f:
            json.dump(usd.to_json(date_format='iso', orient='records'), f)
    else:
        usd = usdtt
    
    # Убедимся, что begin в datetime
    if 'begin' in usd.columns:
        usd['begin'] = pd.to_datetime(usd['begin'])
    else:
        # Если что-то пошло не так, создаем пустой DataFrame с колонкой begin
        usd = pd.DataFrame(columns=['begin', 'usdt'])
    
    data_framest = {}
    
    for sector, tickers in sectors.items():
        sector_data = {}
        for ticker in tickers:
            j = requests.get(f'http://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json?from={startpars}&till=2024-09-25&interval=24').json()
            jo = requests.get(f'http://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json?from=2024-09-25&till={today}&interval=24').json()
            dataj = [{k: r[i] for i, k in enumerate(j['candles']['columns'])} for r in j['candles']['data']]
            datajo = [{k: r[i] for i, k in enumerate(jo['candles']['columns'])} for r in jo['candles']['data']]
            framej = pd.DataFrame(dataj)
            framejo = pd.DataFrame(datajo)
            result = pd.concat([framej, framejo], ignore_index=True)
            result['begin'] = pd.to_datetime(result['begin'])
            result = result[['begin', 'close','volume']].rename(columns={'close': 'ticker'})
            result = pd.merge(result, ind[sector], on='begin', how='inner')
            result = pd.merge(result, ind['IMOEX2'], on='begin', how='inner')
            usd['begin'] = pd.to_datetime(usd['begin'])
            result = pd.merge(result, usd, on='begin', how='inner')
            result = pd.merge(result, bre, on='begin', how='inner')
            result.set_index('begin', inplace=True)
            result.dropna(inplace=True)
            
            for column in result.columns:
                if 'lag' not in column and 'rolling' not in column and 'sector' not in column and 'ticker' not in column:
                    result[f'{column}_rolling5'] = result[column].rolling(window=5).mean()
            
            result.dropna(inplace=True)
            
            for column in result.columns:
                for lag in [7, 14, 28, 35,31]:
                    if 'lag' not in column and 'rolling' not in column and 'sector_x' not in column:
                        result[f'{column}_lag_{lag}'] = result[column].shift(lag)
            
            result.dropna(inplace=True)
            sector_data[ticker] = result
        data_framest[sector] = sector_data
    
    sppa=[]
    spv={}
    bpredictions={}
    smpe=0
    data_frames = {}
    
    for sector, tickers in sectors.items():
        sector_data = {}
        for ticker in tickers:
            result=data_framest[sector][ticker]
            result['target'] = 0
            for i in range(len(result)):
                current_close = result.iloc[i]['ticker']
                if i + 30 < len(result):
                    future_prices = result.iloc[i + 1:i + 1+forecast_days]['ticker']
                    result.at[result.index[i], 'target'] = (max(future_prices)-current_close)*100/current_close
            result.dropna(inplace=True)
            sector_data[ticker] = result
        data_frames[sector] = sector_data
    
    param_grid = {
        'num_leaves': [31, 50,21],
        'max_depth': [-1, 5, 10],
        'learning_rate': [0.3, 0.1, 0.2],
        'n_estimators': [250, 200]
    }
    
    models = {}
    
    for sector, secd in data_frames.items():
        models[sector] = {}
        for ticker in sectors[sector]:
            models[sector][ticker] = {}
            X = secd[ticker].drop(columns=['target']).iloc[:-(1+forecast_days)]
            y = secd[ticker]['target'].iloc[:-(1+forecast_days)]
            
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
            
            if mse_scores.mean()>7.5:
                sppa.append(ticker)
                sppa.append(mse_scores.mean())
            
            smpe+=mse_scores.mean()
            
            for i in range(len(model.feature_importances_)):
                if X.columns[i] not in spv.keys():
                    spv[X.columns[i]]=0
                spv[X.columns[i]]+=model.feature_importances_[i]
            
            models[sector][ticker] = model
    
    dpor={}
    predictions = {}
    
    for sector, secd in data_frames.items():
        for ticker in sectors[sector]:
            last_data = secd[ticker].drop(columns=['target']).iloc[-1].copy()
            prediction = models[sector][ticker].predict([last_data])
            predictions[ticker]=float(prediction[0])
    
    sr,mn={},{}
    for ticker,pr in predictions.items():
        if pr>3.5 and pr<7.5:
            sr[ticker]=pr
        elif pr>=7.5:
            mn[ticker]=pr
    
    import json
    from datetime import datetime,timezone
    
    results_data = {
        "date": str(date.today()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "predictions": predictions,
        "medium_potential": dict(sr),
        "high_potential": dict(mn),
        "poor_accuracy": sppa,
        "avg_mae": smpe/100,
        "summary": {
            "high_count": len(mn),
            "medium_count": len(sr),
            "is_bad_day": len(mn) < 6,
            "is_fifty_fifty": 6 <= len(mn) < 9,
            "is_good_day": len(mn) >= 9
        }
    }
    
    print("✅ Анализ завершен")
    
    return results_data

def main():
    """Основная функция для GitHub Actions"""
    print("=" * 60)
    print("🚀 ЗАПУСК ПОЛНОГО АНАЛИЗА В GITHUB ACTIONS")
    print("=" * 60)
    
    # Проверяем наличие файла usd_data.json
    if not os.path.exists('usd_data.json'):
        print("❌ Файл usd_data.json не найден!")
        print("📋 Создаю новый файл с начальными данными...")
        
        # Создаем пустой файл с заголовками
        initial_data = pd.DataFrame(columns=['begin', 'usdt'])
        with open('usd_data.json', 'w') as f:
            json.dump(initial_data.to_json(date_format='iso', orient='records'), f)
    
    print("✅ Файл usd_data.json найден или создан")
    
    try:
        print("🔧 Запускаю полный анализ...")
        results = main_pipeline(forecast_days=14, output_mode="json")
        
        # Сохраняем результаты для следующего шага (upload_to_drive.py)
        with open('latest_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Анализ завершен: {len(results.get('predictions', {}))} акций")
        print(f"🎯 Лучших акций: {results.get('summary', {}).get('high_count', 0)}")
        
        return results
        
    except Exception as e:
        print(f"❌ Ошибка при анализе: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
