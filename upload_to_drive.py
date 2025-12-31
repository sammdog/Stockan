"""
Загружает результаты на Google Drive через API
"""
import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import time

def upload_to_google_drive():
    """Загружает файл на Google Drive"""
    print("📤 Загружаю результаты на Google Drive...")
    
    try:
        # 1. Получаем credentials из переменных окружения
        credentials_json = os.environ.get('GOOGLE_DRIVE_CREDENTIALS')
        if not credentials_json:
            raise ValueError("GOOGLE_DRIVE_CREDENTIALS не установлен")
        
        # 2. Создаем credentials
        creds_dict = json.loads(credentials_json)
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        
        # 3. Создаем сервис
        service = build('drive', 'v3', credentials=creds)
        
        # 4. Загружаем файл
        file_metadata = {
            'name': 'stock_analysis_results.json',
            'description': f'Анализ акций от {time.strftime("%Y-%m-%d %H:%M")}'
        }
        
        media = MediaFileUpload(
            'latest_results.json',
            mimetype='application/json',
            resumable=True
        )
        
        # Обновляем существующий файл
        file_id = os.environ.get('GOOGLE_DRIVE_FILE_ID')
        
        if file_id:
            # Обновляем существующий файл
            file = service.files().update(
                fileId=file_id,
                body=file_metadata,
                media_body=media
            ).execute()
            print(f"✅ Файл обновлен: {file.get('id')}")
        else:
            # Создаем новый файл
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            print(f"✅ Новый файл создан: {file.get('id')}")
            print(f"📝 Сохраните этот ID в секретах: {file.get('id')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка загрузки на Google Drive: {e}")
        return False

if __name__ == "__main__":
    upload_to_google_drive()
