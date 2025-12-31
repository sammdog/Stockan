"""
ПОЛНЫЙ АНАЛИЗ ДЛЯ GITHUB ACTIONS
Точная копия вашего кода из Colab, с одним изменением:
- путь к файлу usd изменен на 'usd_data.json'
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

warnings.filterwarnings('ignore')

def main_pipeline_github(forecast_days=14, output_mode="json"):
    """
    ТОЧНАЯ КОПИЯ вашей функции main_pipeline из Colab
    """
    print("=" * 60)
    print("🚀 ЗАПУСК ПОЛНОГО АНАЛИЗА В GITHUB ACTIONS")
    print("=" * 60)
    
    # === ВСТАВЬТЕ СЮДА ВЕСЬ ВАШ КОД ИЗ COLAB ===
    # НАЧИНАЯ С:
    import numpy as np
    import pandas as pd
    import requests
    from datetime import date, timedelta
    import lightgbm
    from lightgbm import LGBMRegressor
    import sklearn
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
    from sklearn.metrics import mean_absolute_error, r2_score, precision_score, mean_squared_error, mean_absolute_percentage_error
    from sklearn.ensemble import GradientBoostingRegressor
    import joblib
    import os
    import json
    
    # cap словарь...
    cap = dict()
    cap['IMOEX2'] = {}
    # ... ВСТАВЬТЕ ВЕСЬ ВАШ КОД ...
    
    # === ЕДИНСТВЕННОЕ ИЗМЕНЕНИЕ ===
    # ВМЕСТО:
    # file_pa = '/content/drive/MyDrive/usd'
    # ИСПОЛЬЗУЙТЕ:
    file_pa = 'usd_data.json'  # Файл из репозитория
    
    # ... ПРОДОЛЖАЙТЕ ВАШ КОД ...
    
    # В конце функции сохраните обновленный usd обратно
    '''
    with open(file_pa, 'w') as f:
        json.dump(usd.to_json(date_format='iso'), f)
    '''
    # Это сохранит файл в репозиторий для следующего запуска
    
    return results_data  # Ваш результат

def main():
    """Основная функция"""
    try:
        print("🔍 Проверяю наличие файла usd_data.json...")
        if not os.path.exists('usd_data.json'):
            print("❌ Файл usd_data.json не найден!")
            print("📋 Решение:")
            print("   1. Запустите анализ в Colab")
            print("   2. Скачайте usd_github_backup.json")
            print("   3. Загрузите в репозиторий как usd_data.json")
            sys.exit(1)
        
        print("✅ Файл usd_data.json найден")
        
        # Запускаем анализ
        results = main_pipeline_github(forecast_days=14, output_mode="json")
        
        # Сохраняем результаты
        with open('latest_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("✅ Анализ завершен")
        return results
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
