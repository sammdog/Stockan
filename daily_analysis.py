"""
Адаптированная версия вашего анализа для GitHub Actions
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

# ========== КОНФИГ ==========
GOOGLE_DRIVE_FILE_ID = os.environ.get('GOOGLE_DRIVE_FILE_ID', 'ваш_ид_файла')
FORECAST_DAYS = 14
# ============================

print("🚀 Запуск ежедневного анализа...")
print(f"Дата: {date.today()}")
print(f"Прогноз на дней: {FORECAST_DAYS}")

def main_pipeline(forecast_days=14, output_mode="tg"):
    """
    Основная функция для запуска всего пайплайна
    forecast_days: количество дней для предсказания
    output_mode: формат вывода ('tg' для Telegram, 'json' для API)
    """
    import numpy as np
    import pandas as pd
    import requests
    from datetime import date, timedelta
    import matplotlib.pyplot as plt
    import lightgbm
    from lightgbm import LGBMClassifier
    from lightgbm import LGBMRegressor
    import sklearn
    from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
    from sklearn.metrics import mean_absolute_error, r2_score,precision_score,mean_squared_error,mean_absolute_percentage_error
    from sklearn.ensemble import GradientBoostingRegressor
    import joblib
    import os
    import json
    #cap словарь с капитализациями тех компаний, которые используются для расчета индекса, в ходе проверок капитализация улучшала результат поэтому я вручную писал её
    #лгбмрегрессор берет инфу о цене обьеме и всем таком акций и по этому предиктит индекс отрасли к которой принадлежат эти акции, точность у метода где-то 90%
    #из апи я не стал брать потому что не рассчитываются индексы в выходные + некоторые индексы обновляются вообще раз в пару дней (айти сектор), моя модель предиктит индексы на каждыф день
    #индекс мосбиржи тоже есть, в нем используются больше акций. список используемых тикеров - sectorsi
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
    #начало извлечения информации, где-то 75% акций берутся прям с этой даты, но такие как head data имеют только +- 200 строк поэтому для баланса ограничил старт парсинга так чтобы было максимум +-550
    forecast_days = 14  # Количество дней для предсказания
    today = date.today() #-timedelta(days=forecast_days-13) #форкаст ставится когда проверяешь модель, + чето там для сдвига дня который проверяешь
    from math import comb
    #функция добавления колонки с капитализацией к данным для индексов
    def add_cap_columns(df, cap_dict):
        for key, values in cap_dict.items():
            # Создаем новую колонку с именем cap<ключ> капитализация
            column_name = f'cap{key}'

            # Определяем значения для новой колонки в зависимости от года
            df[column_name] = df['begin'].dt.year.map({
                2023: values[0],
                2024: values[1],
                2025: values[2]
                })
        return df
    #два реквеста потому что длина страницы которую парсим ограничена и если в один реквест делать то обрежет, в будущем придется или третью страницу брать или сдвигать стартпарс
    #считаем индексы
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
        combined_frame = sector_data[0]
        for df in sector_data[1:]:
            combined_frame = combined_frame.merge(df, on='begin', how='inner')
        combined_frame.dropna(inplace=True)
        combined_frame = add_cap_columns(combined_frame, cap[sector])
        r=combined_frame['begin']
        combined_frame.set_index('begin', inplace=True)
        combined_frame.dropna(inplace=True)
        ccomb=combined_frame.copy()
        j = requests.get(f'http://iss.moex.com/iss/engines/stock/markets/index/securities/{sector}/candles.json?from={startpars}&till=2024-09-25&interval=24').json()
        jo = requests.get(f'http://iss.moex.com/iss/engines/stock/markets/index/securities/{sector}/candles.json?from=2024-09-25&till={today}&interval=24').json()
        dataj = [{k: r[i] for i, k in enumerate(j['candles']['columns'])} for r in j['candles']['data']]
        datajo = [{k: r[i] for i, k in enumerate(jo['candles']['columns'])} for r in jo['candles']['data']]
        framej = pd.DataFrame(dataj)
        framejo = pd.DataFrame(datajo)
        sec = pd.concat([framej, framejo], ignore_index=True)
        sec['begin'] = pd.to_datetime(sec['begin'])
        sec=sec[['begin', 'close']].rename(columns={'close': 'sector'})
        combined_frame = pd.merge(combined_frame, sec, on='begin', how='inner')
        combined_frame.dropna(inplace=True)
        X=combined_frame.drop(columns=['sector','begin'])
        y=combined_frame['sector']
        model=LGBMRegressor(random_state=1,verbose=-1)
        model.fit(X,y)
        pred=model.predict(ccomb)
        sec=pd.DataFrame()
        sec['begin']=r
        sec['sector']=pred
        ind[sector]=sec
        ind[sector] = ind[sector].drop_duplicates(subset='begin', keep='first')
        #print(ind[sector])
    j = requests.get(f'http://iss.moex.com/iss/engines/futures/markets/forts/securities/BRU5/candles.json?from={startpars}&till=2024-09-25&interval=24').json()
    jo = requests.get(f'http://iss.moex.com/iss/engines/futures/markets/forts/securities/BRU5/candles.json?from=2024-09-25&till={today}&interval=24').json()
    dataj = [{k: r[i] for i, k in enumerate(j['candles']['columns'])} for r in j['candles']['data']]
    datajo = [{k: r[i] for i, k in enumerate(jo['candles']['columns'])} for r in jo['candles']['data']]
    framej = pd.DataFrame(dataj)
    framejo = pd.DataFrame(datajo)
    bre = pd.concat([framej, framejo], ignore_index=True)
    bre['begin'] = pd.to_datetime(bre['begin'])
    bre=bre[['begin', 'close']].rename(columns={'close': 'brent'})
    #парсинг цены нефти в датафрейм bre
    #ставим диск в папку контент, забираем оттуда курс доллара за предыдущие дни, фиксируем ласт известный день в стартюсд
    '''from google.colab import drive #сейчас у меня автозагрузка диска, во вкладке файлов нажал подключить диск и не надо региться в нем
    drive.mount('/content/drive')'''
    file_pa='/content/drive/MyDrive/usd'
    with open(file_pa, 'r') as f:
            usdtt = pd.read_json(json.load(f))
    startusd=usdtt['begin'].iloc[-1]
    import xml.etree.ElementTree as ET
    from datetime import datetime


    def get_currency_rate(date):
        url = f'https://www.cbr.ru/scripts/XML_daily.asp?date_req={date.strftime("%d/%m/%Y")}'
        response = requests.get(url)

        if response.status_code == 200:
            root = ET.fromstring(response.content)
            for valute in root.findall('Valute'):
                char_code = valute.find('CharCode').text
                if char_code == 'USD':  # Ищем курс доллара
                    value = valute.find('Value').text
                    return float(value.replace(',', '.'))  # Конвертируем строку в число
        else:
            print(f'Ошибка при запросе данных за {date.strftime("%d/%m/%Y")}: {response.status_code}')
        return None

    # Задаем диапазон дат
    start_date = datetime.strptime(startusd, '%Y-%m-%dT%H:%M:%S.%f').date()#datetime.strptime('2023-05-29','%Y-%m-%d').date() #если надо взять с начала парсинга то ставим второй варик
    end_date = today
    #приходится каждый день отдельно реквестить из-за особенностей апи, поэтому это долго и я экономлю много времени сохраняя старую инфу на диске
    # Словарь для хранения курсов
    usd_rates = {}

    # Получаем курс доллара за каждый день в диапазоне
    current_date = start_date
    while current_date <= end_date:
        rate = get_currency_rate(current_date)
        if rate is not None:
            usd_rates[current_date.strftime("%d/%m/%Y")] = rate
        current_date += timedelta(days=1)

    # Выводим результаты
    usd = pd.DataFrame(list(usd_rates.items()), columns=['begin', 'usdt'])
    usd['begin'] = pd.to_datetime(usd['begin'],format='%d/%m/%Y')
    usdtt['begin'] = pd.to_datetime(usdtt['begin'],format='%Y-%m-%dT%H:%M:%S.%f')
    usd = pd.merge(usd, usdtt, on='begin', how='outer')
    usd['usdt'] = usd['usdt_x'].combine_first(usd['usdt_y'])

    # Удаление старых колонок, в процессе слияния появляются две колонки х у которые не нужны и они заменены на юсдт
    usd = usd.drop(columns=['usdt_x', 'usdt_y'])
    with open(file_pa, 'w') as f:
      json.dump(usd.to_json(date_format='iso'), f)
      #сохраняем новый файл на диск
    data_framest = {}
    #словарь в котором все данные, внутри словари по отраслям в каждом датафреймы тикеров
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
                #if result.shape[0]<201:
                    #print(ticker,result.shape[0])
                result['begin'] = pd.to_datetime(result['begin'])
                result = result[['begin', 'close','volume']].rename(columns={'close': 'ticker'}) #берем цену, обьем торгов, день(бегин)
                # Объединение с индексами. сначала сектор потом моекс потом доллар и брент, обьединяем по столбцу бегин формата дейттайм
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
                # Добавление лагов, кроме некоторых колонок которые опытным путем удалены + без лагов лагов и ролинга
                for column in result.columns:
                    for lag in [7, 14, 28, 35,31]:
                      if 'lag' not in column and 'rolling' not in column and 'sector_x' not in column:
                          result[f'{column}_lag_{lag}'] = result[column].shift(lag)
                result.dropna(inplace=True)
                sector_data[ticker] = result

            data_framest[sector] = sector_data

    sppa=[] #список акций у которых большое мае на крос валидации
    spv={} #словарь важности для колонок
    # Сбор данных
    bpredictions={}
    smpe=0 #среднее мае по всем
    data_frames = {}
    for sector, tickers in sectors.items():
        sector_data = {}
        for ticker in tickers:
            result=data_framest[sector][ticker]
            result['target'] = 0  # Инициализация колонки с макс % роста
            for i in range(len(result)):
                current_close = result.iloc[i]['ticker']
                    # Проверяем следующие 30 дней
                if i + 30 < len(result):  # Убедимся, что индекс не выходит за пределы
                    future_prices = result.iloc[i + 1:i + 1+forecast_days]['ticker']
                    result.at[result.index[i], 'target'] = (max(future_prices)-current_close)*100/current_close
            result.dropna(inplace=True)
            sector_data[ticker] = result
        data_frames[sector] = sector_data
        # Предсказания для каждой акции в секторе
        # параметры которые перебираем при грид серче
    param_grid = {
        'num_leaves': [31, 50,21],
        'max_depth': [-1, 5, 10],
        'learning_rate': [0.3, 0.1, 0.2],
        'n_estimators': [250, 200]
    }
    #проверяем есть ли в контенте папка модели, а диск мы билдим до этого
    if not os.path.exists('models'):
      os.makedirs('models')
    models = {}

    for sector, secd in data_frames.items():
        models[sector] = {}
        for ticker in sectors[sector]:
              models[sector][ticker] = {}
              X = secd[ticker].drop(columns=['target']).iloc[:-(1+forecast_days)]  #нам неизвестна информация о последних 14 днях, мы не знаем для них целевую
              y = secd[ticker]['target'].iloc[:-(1+forecast_days)]  #
              # Смонтировать Google Диск
              #from google.colab import drive
              #drive.mount('/content/drive')
              file_path = f'/content/drive/MyDrive/models/{sector}_{ticker}_params.json'
              if os.path.exists(file_path)!=False:
                with open(file_path, 'r') as f:
                  loaded_params = json.load(f)
                model = LGBMRegressor(random_state=1,**loaded_params)#если для этого тикера уже считали парамтеры то они есть в папке и загружаем их иначе считаем ниже
              else:
                # Обучение модели
                model = LGBMRegressor(random_state=1)
                grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=4)
                grid_search.fit(X, y)

            # Сохранение лучшей модели
                best_model = grid_search.best_estimator_
            # Сохранение модели
                best_params = grid_search.best_params_

                # Сохранение гиперпараметров в файл
                if not os.path.exists(os.path.dirname(file_path)):
                  os.makedirs(os.path.dirname(file_path))
                with open(file_path, 'w') as f:
                  json.dump(best_params, f)
                model=best_model
              threshold = 12 #граница целевой после которой строчки получают дополнительный вес при оьучении, опытным путем получено 12
              weights = np.where(y > threshold, 2, 1)
              model.fit(X, y,sample_weight=weights)
              cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
                # Преобразование отрицательных значений MАE в положительные
              mse_scores = -cv_scores
              #print(ticker,mse_scores.mean())
              if mse_scores.mean()>7.5:
                sppa.append(ticker)
                sppa.append(mse_scores.mean())
              smpe+=mse_scores.mean()
              for i in range(len(model.feature_importances_)):
                if X.columns[i] not in spv.keys():
                  spv[X.columns[i]]=0
                spv[X.columns[i]]+=model.feature_importances_[i]
              models[sector][ticker] = model  # Сохраняем модель для каждого тикера и дня
    #for i in spv.keys():
      #spv[i]=spv[i]/100
    #sorted_dict = sorted(spv.items(), key=lambda item: item[1])
        # Вывод отсортированных значений с ключами
    #for key, value in sorted_dict:
        #print(f' {key}, {value}')
        # Подготовка данных для предсказания
    dpor={} #если хочется посмотреть предикт по какой-то акции не относительно сейчас а относительно другой цены надо вписать её в словарь
    predictions = {}
    for sector, secd in data_frames.items():
        for ticker in sectors[sector]:
            last_data = secd[ticker].drop(columns=['target']).iloc[-1].copy()  # Последние известные данные
                # Прогнозируем
            prediction = models[sector][ticker].predict([last_data])
            #if ticker in dpor.keys():
              #print(((last_data['ticker']*((prediction[0]/100)+1)-dpor[ticker])/dpor[ticker])*100,ticker)
            predictions[ticker]=float(prediction[0])
    sr,mn={},{}
    for ticker,pr in predictions.items():
      if pr>3.5 and pr<7.5:
        sr[ticker]=pr
      elif pr>=7.5:
        mn[ticker]=pr
    import json
    from datetime import datetime,timezone

    # 1. Сохраняем JSON с полными данными
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

    # Сохраняем в Google Drive
    results_path = '/content/drive/MyDrive/stock_analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    # 2. Сохраняем текстовую версию для быстрого чтения
    text_result = format_for_telegram(sr, mn, len(mn))
    text_path = '/content/drive/MyDrive/latest_analysis.txt'
    with open(text_path, 'w') as f:
        f.write(text_result)

    print(f"✅ Результаты сохранены в Google Drive")
    print(f"📁 JSON: {results_path}")
    print(f"📄 Текст: {text_path}")

    # Возвращаем результат
    if output_mode == "tg":
        return text_result
    elif output_mode == "json":
        return results_data
    else:
        return {"predictions": predictions, "sr": sr, "mn": mn}

# Только уберите интерактивные элементы:
# 1. Удалите !pip install
# 2. Удалите код Telegram бота из Colab
# 3. Удалите @bot.message_handler и т.д.
# 4. Оставьте только логику анализа и сохранения

def main():
    """Основная функция"""
    try:
        print("🔧 Запускаю пайплайн анализа...")
        
        # Ваша функция main_pipeline должна вернуть результат
        result = main_pipeline(forecast_days=FORECAST_DAYS, output_mode="json")
        
        # Сохраняем локально
        with open('latest_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print("✅ Анализ завершен")
        print(f"📊 Результаты: {len(result.get('predictions', {}))} акций")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
