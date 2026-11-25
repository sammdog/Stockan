"""
ПОЛНЫЙ АНАЛИЗ АКЦИЙ ДЛЯ GITHUB ACTIONS
Полностью идентичная версия Google Colab
"""
import sys
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from lightgbm import LGBMRegressor
import warnings
import xml.etree.ElementTree as ET
from io import StringIO
from tools.trade_plan import MAX_CANDIDATES, MIN_EDGE, board_details, expected_returns, median_forecasts, recommendations

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

import socket

MOEX_CA_BUNDLE = Path(__file__).resolve().parent / "certs" / "moex-ca.pem"


def moex_get(url, **kwargs):
    return requests.get(url, verify=str(MOEX_CA_BUNDLE), **kwargs)


def check_moex_connectivity():
    """Проверяет, доступен ли MOEX ISS API."""
    test_url = "https://iss.moex.com/iss/engines/stock/markets/shares/securities/SBER/candles.json?from=2024-01-01&till=2024-01-04&interval=24"
    try:
        r = moex_get(test_url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        if r.status_code == 200:
            return True
    except Exception as e:
        print(f"❌ MOEX API недоступен: {e}")
        return False
    return False



# ========== ФУНКЦИИ ДЛЯ ЗАГРУЗКИ ДАННЫХ MOEX новая от 02.11.2026 ==========
def fetch_shares_data_colab(ticker, start_date, end_date, is_index=False):
    """
    Загружает данные MOEX с помощью постраничной загрузки (пагинации),
    чтобы обойти ограничения API на размер ответа.
    """
    debug_print(f"  📥 Загрузка {ticker} (метод Colab с пагинацией)...")
    
    if is_index:
        base_url = "https://iss.moex.com/iss/engines/stock/markets/index/securities"
    else:
        base_url = "https://iss.moex.com/iss/engines/stock/markets/shares/securities"
    
    all_data = []
    start_row = 0
    limit_rows = 500  # Оптимальный размер страницы, чтобы не превышать лимиты
    
    while True:
        # Формируем URL с параметрами начала и лимита строк
        url = f"{base_url}/{ticker}/candles.json?from={start_date}&till={end_date}&interval=24&start={start_row}&limit={limit_rows}"
        debug_print(f"    🔄 Запрос строк {start_row} - {start_row + limit_rows}...")
        
        try:
            response = moex_get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                
                # Проверяем наличие данных в ответе
                if 'candles' in data and 'data' in data['candles'] and data['candles']['data']:
                    part_data = [{k: r[i] for i, k in enumerate(data['candles']['columns'])} 
                                 for r in data['candles']['data']]
                    df_part = pd.DataFrame(part_data)
                    all_data.append(df_part)
                    debug_print(f"      ✅ Получено {len(df_part)} строк")
                    
                    # Если вернулось строк меньше лимита, значит, это была последняя страница
                    if len(df_part) < limit_rows:
                        debug_print(f"    🏁 Достигнут конец данных для {ticker}.")
                        break
                    
                    # Увеличиваем счетчик для следующей страницы
                    start_row += limit_rows
                else:
                    debug_print(f"    🏁 Нет данных на странице {start_row}. Завершаю.")
                    break
            else:
                debug_print(f"    ❌ Ошибка HTTP {response.status_code} на странице {start_row}")
                break
                
        except Exception as e:
            debug_print(f"    ❌ Ошибка запроса на странице {start_row}: {e}")
            break
    
    # Объединяем все загруженные части
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        # Удаляем дубликаты по дате (на случай стыковки страниц)
        result['begin'] = pd.to_datetime(result['begin'])
        result = result.sort_values('begin').drop_duplicates('begin')
        debug_print(f"  📊 Итого по {ticker}: {len(result)} строк, последняя дата: {result['begin'].max().date() if not result.empty else 'нет'}")
        return result
    else:
        debug_print(f"  ❌ Нет данных по {ticker}")
        return pd.DataFrame()

def fetch_futures_data_colab(ticker, start_date, end_date):
    """
    Загружает данные фьючерсов одним запросом как в Google Colab
    """
    debug_print(f"  📥 Загрузка фьючерса {ticker} (метод Colab)...")
    
    url = f"https://iss.moex.com/iss/engines/futures/markets/forts/securities/{ticker}/candles.json?from={start_date}&till={end_date}&interval=24"
    
    try:
        response = moex_get(url, timeout=30)
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
        else:
            debug_print(f"  ⚠️ Ошибка https {response.status_code} для {ticker}")
    except Exception as e:
        debug_print(f"  ❌ Ошибка запроса {ticker}: {e}")
    
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
def fetch_usd_history(start, end):
    """Load official USD rates by effective date."""
    response = requests.get(
        "https://www.cbr.ru/scripts/XML_dynamic.asp",
        params={
            "date_req1": (start - timedelta(days=31)).strftime("%d/%m/%Y"),
            "date_req2": end.strftime("%d/%m/%Y"),
            "VAL_NM_RQ": "R01235",
        },
        timeout=30,
    )
    response.raise_for_status()
    rates = pd.Series({
        pd.to_datetime(record.attrib["Date"], format="%d.%m.%Y"):
        float(record.findtext("Value").replace(",", ".")) / int(record.findtext("Nominal"))
        for record in ET.fromstring(response.content).findall("Record")
    }, dtype=float).sort_index()
    if (rates.empty or rates.index.min().date() > start
            or rates.index.max().date() < end - timedelta(days=MAX_INPUT_AGE_DAYS)
            or not (np.isfinite(rates) & (rates > 0)).all()):
        raise ValueError("CBR returned incomplete or invalid USD history")

    daily = rates.reindex(pd.date_range(start, end), method="ffill")
    return daily.rename("usdt").rename_axis("begin").reset_index()


def load_usd_history(start, end, cache_path=Path("usd_data.json")):
    try:
        usd = fetch_usd_history(start, end)
    except (requests.RequestException, ET.ParseError, ValueError, KeyError, TypeError,
            AttributeError) as exc:
        debug_print(f"CBR USD history unavailable: {exc}")
        try:
            # Accept both legacy pandas JSON layouts, wrapped in a JSON string.
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            cached = pd.read_json(StringIO(payload))
            cached["begin"] = pd.to_datetime(cached["begin"], format="ISO8601")
            cached = cached.set_index("begin")["usdt"].sort_index()
            cached = cached.loc[pd.Timestamp(start):pd.Timestamp(end)]
            expected = pd.date_range(start, cached.index.max()) if not cached.empty else None
            if (expected is None or not cached.index.equals(expected)
                    or cached.index.max().date() < end - timedelta(days=MAX_INPUT_AGE_DAYS)
                    or not (np.isfinite(cached) & (cached > 0)).all()):
                raise ValueError("cache is stale or has missing USD dates")
        except (OSError, ValueError, KeyError, TypeError) as cache_error:
            raise RuntimeError(f"No usable USD history: {cache_error}") from exc
        debug_print(f"Using complete USD cache through {cached.index.max().date()}")
        return cached.rename_axis("begin").reset_index()

    cache_path.write_text(json.dumps(usd.to_json(date_format="iso", orient="records")),
                          encoding="utf-8")
    debug_print(f"USD history (CBR): {len(usd)} dates from {start} to {end}")
    return usd


# Preserve the historical series through the final trading date of BRJ6.
BRENT_ROLL_START = date(2026, 4, 2)
BRENT_LEGACY_CONTRACTS = ("BRU3", "BRU4", "BRU5", "BRJ6")
BRENT_MONTH_CODES = "FGHJKMNQUVXZ"
MAX_INPUT_AGE_DAYS = 10


def fetch_brent_data_colab(start_date, end_date):
    """Append next-calendar-month MOEX BR closes to the legacy history."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    windows = []
    legacy_end = min(end, BRENT_ROLL_START - timedelta(days=1))
    if start <= legacy_end:
        windows.extend((ticker, start, legacy_end) for ticker in BRENT_LEGACY_CONTRACTS)

    window_start = max(start, BRENT_ROLL_START)
    while window_start <= end:
        next_month = (window_start.replace(day=28) + timedelta(days=4)).replace(day=1)
        ticker = f"BR{BRENT_MONTH_CODES[next_month.month - 1]}{next_month.year % 10}"
        window_end = min(end, next_month - timedelta(days=1))
        windows.append((ticker, window_start, window_end))
        window_start = next_month

    parts = []
    for ticker, lower, upper in windows:
        frame = fetch_futures_data_colab(ticker, lower.isoformat(), upper.isoformat())
        if not frame.empty:
            dates = frame["begin"].dt.date
            frame = frame.loc[dates.between(lower, upper), ["begin", "close"]]
            if not frame.empty:
                parts.append(frame.rename(columns={"close": "brent"}))
                debug_print(f"Brent {ticker}: {len(frame)} candles for {lower} to {upper}")
                continue
        if lower >= BRENT_ROLL_START and (end - upper).days > MAX_INPUT_AGE_DAYS:
            raise RuntimeError(f"No Brent candles for {ticker} in {lower} to {upper}")

    if not parts:
        return pd.DataFrame(columns=["begin", "brent"])
    # Legacy contracts overlap. Prefer the later contract in the original list.
    brent = pd.concat(parts, ignore_index=True).drop_duplicates("begin", keep="last")
    brent = brent.sort_values("begin").reset_index(drop=True)
    legacy = brent.loc[brent["begin"].dt.date < BRENT_ROLL_START]
    debug_print(f"Legacy Brent coverage: {len(legacy)} candles, last {legacy['begin'].max()}")
    debug_print(f"Extended Brent coverage: {len(brent)} candles, last {brent['begin'].max()}")
    return brent

# ========== ОСНОВНОЙ КОД ==========
def main_pipeline(forecast_days=14):
    """
    Основная функция для запуска всего пайплайна
    Точная копия логики Google Colab
    """
    debug_print(f"🚀 Начинаю полный анализ (метод Google Colab)...")
    # В самом начале main() или main_pipeline() добавить:
    if not check_moex_connectivity():
        raise SystemExit("Критическая ошибка: не удаётся подключиться к MOEX ISS API. Работа остановлена.")
    # === КОНФИГУРАЦИЯ КАПИТАЛИЗАЦИЙ ===
    cap = {
        'IMOEX2': {
            'VTBR': [657.8, 505.5, 503.8, 471.3],
            'SBER': [6600, 7000, 6900, 6900],
            'GAZP': [3700, 3600, 3100, 2900],
            'LKOH': [5200, 5100, 4200, 3700],
            'T': [609.3, 975.3, 862.8, 864.6],
            'GMKN': [2400, 1900, 1600, 2400],
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
        'IMOEX2': ['VTBR', 'SBER', 'LKOH', 'T', 'GAZP','GMKN'],
        'MOEXOG': ['GAZP', 'NVTK', 'ROSN', 'TATN','LKOH'],
        'MOEXMM': ['GMKN', 'MAGN', 'NLMK', 'PLZL','MTLR'],
        'MOEXIT': ['YDEX', 'VKCO', 'ASTR', 'POSI','SOFL'],
        'MOEXFN': ['VTBR', 'SBER', 'T', 'SVCB','BSPB'],
        'MOEXEU': ['IRAO', 'UPRO','FEES','MSNG','HYDR'],
        'MOEXRE': ['SMLT','LSRG','MSTT'],
        'MOEXTN': ['AFLT', 'FLOT','SVAV','FESH','KMAZ'],
        'MOEXCN': ['MVID', 'LENT','MGNT','PHOR','BELU']
    }
    
    startpars = '2023-05-29'
    today = datetime.now(timezone(timedelta(hours=3))).date()
    completed_through = today - timedelta(days=1)
    
    debug_print(f"📅 Сегодня: {today}")
    debug_print(f"📅 Стартовая дата для данных: {startpars}")
    debug_print(f"📊 Количество секторов для индексов: {len(sectorsi)}")
    debug_print(f"📊 Количество секторов для акций: {len(sectors)}")
    
    # ========== ЗАГРУЗКА ИНДЕКСОВ СЕКТОРОВ (ТОЧНО КАК В GOOGLE COLAB) ==========
    debug_print("🔧 Загружаю и рассчитываю индексы секторов...")
    ind = {}
    
    for sector, tickers in sectorsi.items():
        debug_print(f"  📈 Обрабатываю сектор: {sector}")
        sector_data = []
        
        # Загружаем данные по всем акциям сектора (точно как в Colab)
        for ticker in tickers:
            df = fetch_shares_data_colab(ticker, startpars, completed_through.isoformat(), is_index=False)
            if not df.empty:
                df = df[['begin', 'close', 'volume', 'value']].rename(
                    columns={'close': ticker, 'volume': f'vol{ticker}', 'value': f'val{ticker}'}
                )
                sector_data.append(df)
        
        if not sector_data:
            debug_print(f"  ❌ Нет данных по сектору {sector}")
            continue
        
        # Объединяем все акции сектора (точно как в Colab)
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
        
        # Загружаем реальный индекс сектора (точно как в Colab)
        sec_real = fetch_shares_data_colab(sector, startpars, completed_through.isoformat(), is_index=True)
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
    
    # ========== ЗАГРУЗКА ДАННЫХ BRENT (ТОЧНО КАК В GOOGLE COLAB) ==========
    bre = fetch_brent_data_colab(startpars, completed_through.isoformat())
    
    earliest_input_date = today - timedelta(days=MAX_INPUT_AGE_DAYS)
    if bre.empty or bre["begin"].max().date() < earliest_input_date:
        last_brent = bre["begin"].max()
        raise RuntimeError(f"Brent data is stale: last candle {last_brent}, expected {earliest_input_date} or later")

    # ========== ЗАГРУЗКА И ОБНОВЛЕНИЕ USD ДАННЫХ (ТОЧНО КАК В GOOGLE COLAB) ==========
    debug_print("💵 Загружаю и обновляю данные по USD...")
    usd = load_usd_history(date.fromisoformat(startpars), today)

    # ========== ОБРАБОТКА ДАННЫХ АКЦИЙ (ТОЧНО КАК В GOOGLE COLAB) ==========
    debug_print("📊 Обрабатываю данные по акциям...")
    data_framest = {}
    order_candles = {}
    
    for sector, tickers in sectors.items():
        sector_data = {}
        for ticker in tickers:
            df = fetch_shares_data_colab(ticker, startpars, completed_through.isoformat(), is_index=False)
            if df.empty:
                debug_print(f"  ⚠️ Пропускаю {ticker} - нет данных")
                continue
            
            order_candles[ticker] = df.set_index("begin")[["low", "high", "close"]]
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
            if df.empty or df.index[-1].date() < earliest_input_date:
                debug_print(f"Skipping {ticker}: last usable candle {df.index.max()}")
                continue
            sector_data[ticker] = df
        
        data_framest[sector] = sector_data
    
    debug_print(f"✅ Обработаны данные по {sum(len(v) for v in data_framest.values())} акциям")
    
    # Use a common completed source session and preserve the historical target.
    if not order_candles:
        raise RuntimeError("No completed stock candles available")
    signal_date = max(frame.index.max() for frame in order_candles.values())
    if signal_date.date() < earliest_input_date:
        raise RuntimeError(f"Stock candles are stale: {signal_date.date()}")
    frames = {ticker: frame for group in data_framest.values() for ticker, frame in group.items()}
    ranking = sorted(median_forecasts(frames, signal_date, forecast_days), key=lambda row: (-row[1], row[0]))
    if not ranking:
        raise RuntimeError(f"No median forecasts for the latest completed session {signal_date.date()}")
    edge = expected_returns(order_candles, signal_date)
    if not edge:
        raise RuntimeError("No historical order return estimates")
    eligible = {ticker for ticker, _, _ in ranking[:MAX_CANDIDATES] if edge.get(ticker, -np.inf) >= MIN_EDGE}
    metadata = board_details(moex_get, eligible)
    if eligible - metadata.keys():
        debug_print(f"Missing TQBR lot metadata: {', '.join(sorted(eligible - metadata.keys()))}")
    candidates = recommendations(ranking, edge, metadata)
    predictions = {ticker: score for ticker, score, _ in ranking}
    results_data = {
        "date": today.isoformat(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy": "median_ev_025_tp3",
        "signal_date": signal_date.date().isoformat(),
        "predictions": predictions,
        "prediction_inputs": {
            ticker: {"data_date": signal_date.date().isoformat(), "reference_close": reference}
            for ticker, _, reference in ranking
        },
        "forecast_horizon_rows": forecast_days,
        "medium_potential": {ticker: score for ticker, score in predictions.items() if 5.5 < score < 9.5},
        "high_potential": {ticker: score for ticker, score in predictions.items() if score >= 9.5},
        "trade_candidates": candidates,
        "missing_trade_metadata": sorted(eligible - metadata.keys()),
        "trade_fee_bps": 5,
        "trade_notional_cap_rub": 1000,
    }
    with open("latest_results.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2, allow_nan=False)
    debug_print(f"Median forecasts: {len(predictions)}, session {signal_date.date()}, trade candidates: {len(candidates)}")
    return results_data


def main():
    try:
        main_pipeline()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
