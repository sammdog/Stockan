"""
Скрипт для загрузки файла usd в репозиторий GitHub
"""
import os
import json
import requests

def upload_to_github():
    """Загружает файл usd в репозиторий GitHub"""
    
    # 1. Читаем сохраненный файл
    with open('/content/usd_github_backup.json', 'r') as f:
        usd_content = f.read()
    
    # 2. Настройки GitHub
    GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')  # Получите из Colab Secrets
    REPO_OWNER = "sammdog"
    REPO_NAME = "Stockan"
    FILE_PATH = "usd_data.json"  # Путь в репозитории
    
    # 3. Создаем/обновляем файл через GitHub API
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Сначала получаем текущий файл (если есть)
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        # Файл существует, обновляем
        sha = response.json()['sha']
        data = {
            "message": "Обновление данных курса USD",
            "content": usd_content.encode('utf-8').decode('latin-1'),
            "sha": sha
        }
    else:
        # Файл не существует, создаем новый
        data = {
            "message": "Добавление данных курса USD",
            "content": usd_content.encode('utf-8').decode('latin-1')
        }
    
    # Отправляем запрос
    response = requests.put(url, headers=headers, json=data)
    
    if response.status_code in [200, 201]:
        print(f"✅ Файл usd загружен в GitHub: {response.json()['content']['sha']}")
        return True
    else:
        print(f"❌ Ошибка загрузки: {response.status_code}")
        print(response.json())
        return False

if __name__ == "__main__":
    upload_to_github()
