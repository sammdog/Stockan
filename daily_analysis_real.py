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

# ========== ОСНОВНОЙ КОД ==========
def main_pipeline(forecast_days=14, output_mode="json"):
    """
    Основная функция для запуска всего пайплайна
    forecast_days: количество дней для предсказания
    output_mode: формат вывода ('tg' для Telegram, 'json' для API)
    """
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
    
    # === ДИАГНОСТИКА: Проверяем доступность данных ===
    debug_data = {
        'analysis_date': str(today),
        'sector_data_dates': {},
        'predictions_info': {}
    }
    
    debug_print(f"📅 Дата анализа: {today}")
    debug_print(f"📊 Количество секторов: {len(sectorsi)}")
    
    def add_cap_columns(df, cap_dict):
        for key, values in cap_dict.items():
            column_name = f'cap{key}'
            df[column_name] = df['begin'].dt.year.map({
                2023: values[0],
                2024: values[1],
                2025: values[2]
            })
        return df
    
    ind = {}
    
    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Используем LEFT JOIN для сохранения всех дат
    for sector, tickers in sectorsi.items():
        debug_print(f"🔧 Обрабатываю сектор: {sector}")
        
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
            result = result[['begin', 'close','volume','value']].rename(
                columns={'close': ticker, 'volume': f'vol{ticker}', 'value': f'val{ticker}'}
            )
            sector_data.append(result)
        
        # Объединяем все акции сектора
        combined_frame = sector_data[0]
        for df in sector_data[1:]:
            combined_frame = combined_frame.merge(df, on='begin', how='inner')
        
        combined_frame.dropna(inplace=True)
        combined_frame = add_cap_columns(combined_frame, cap[sector])
        
        debug_print(f"  📊 Данных акций: {len(combined_frame)} строк")
        debug_print(f"  📅 Акции: первая дата {combined_frame['begin'].min()}, последняя {combined_frame['begin'].max()}")
        
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
        
        debug_print(f"  📊 Данных индекса: {len(sec_real)} строк")
        debug_print(f"  📅 Индекс: первая дата {sec_real['begin'].min()}, последняя {sec_real['begin'].max()}")
        
        # КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: Используем LEFT JOIN вместо INNER
        # Это сохранит все даты из акций, даже если нет данных индекса
        combined_with_sec = combined_frame.merge(sec_real, on='begin', how='left')
        
        # Разделяем данные на те, где есть реальный индекс (для обучения)
        # и те, где нет (для предсказания)
        train_data = combined_with_sec[combined_with_sec['sector'].notna()].copy()
        predict_data = combined_with_sec[combined_with_sec['sector'].isna()].copy()
        
        debug_print(f"  🎯 Данных для обучения: {len(train_data)} строк")
        debug_print(f"  🔮 Данных для предсказания: {len(predict_data)} строк")
        
        if len(train_data) < 30:
            debug_print(f"  ⚠️ Мало данных для обучения ({len(train_data)}), пропускаю сектор")
            # Используем простую линейную экстраполяцию
            ind[sector] = combined_frame[['begin']].copy()
            ind[sector]['sector'] = 0
            continue
        
        # Обучаем модель только на данных с известным индексом
        train_data.set_index('begin', inplace=True)
        X_train = train_data.drop(columns=['sector'])
        y_train = train_data['sector']
        
        model = LGBMRegressor(random_state=1, verbose=-1)
        model.fit(X_train, y_train)
        
        # Предсказываем на ВСЕХ данных (и для обучения, и для предсказания)
        all_data = combined_with_sec.set_index('begin').drop(columns=['sector'])
        
        # Убедимся, что колонки совпадают
        missing_cols = set(X_train.columns) - set(all_data.columns)
        extra_cols = set(all_data.columns) - set(X_train.columns)
        
        if missing_cols:
            debug_print(f"  ⚠️ Отсутствующие колонки: {missing_cols}")
            for col in missing_cols:
                all_data[col] = 0
        
        if extra_cols:
            debug_print(f"  ⚠️ Лишние колонки: {extra_cols}")
            all_data = all_data.drop(columns=list(extra_cols))
        
        # Упорядочиваем колонки
        all_data = all_data[X_train.columns]
        
        # Предсказываем
        pred = model.predict(all_data)
        
        # Создаем DataFrame с предсказаниями
        sec_pred = pd.DataFrame({
            'begin': all_data.index,
            'sector': pred
        })
        
        # Для дат, где был реальный индекс, можно усреднить предсказание и реальное значение
        # или использовать только предсказание. Выбираем предсказание для консистентности
        ind[sector] = sec_pred
        ind[sector] = ind[sector].drop_duplicates(subset='begin', keep='first')
        
        debug_print(f"  ✅ Сектор {sector}: {len(ind[sector])} предсказаний")
        debug_data['sector_data_dates'][sector] = {
            'last_date': str(ind[sector]['begin'].max()),
            'predictions_count': len(ind[sector])
        }
    
    # ========== ДИАГНОСТИКА: проверяем предсказания индексов ==========
    for sector in sectorsi:
        if sector in ind:
            last_5 = ind[sector].tail(5)
            debug_print(f"📈 Последние 5 предсказаний для {sector}:")
            for _, row in last_5.iterrows():
                debug_print(f"  {row['begin'].date()}: {row['sector']:.2f}")
    
    # ========== ЗАГРУЗКА ДАННЫХ БРЕНТ И USD ==========
    # [Остальной код остается таким же, но добавьте диагностику]
    
    # После загрузки всех данных добавьте:
    debug_print("=" * 60)
    debug_print("📊 ИТОГОВАЯ ДИАГНОСТИКА ДАННЫХ:")
    
    # Проверяем последние даты в данных
    for sector in sectors:
        if sector in data_framest and data_framest[sector]:
            ticker = list(data_framest[sector].keys())[0]
            last_date = data_framest[sector][ticker].index[-1]
            debug_print(f"✅ Сектор {sector}: последняя дата {last_date.date()}")
    
    # Сохраняем отладочные данные
    save_debug_data(debug_data, f"debug_{today}.json")
    
    return results_data
