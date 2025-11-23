# src/load_test.py

import requests
import random
import time
import argparse
import logging

URL = "http://localhost:8081/recommendations"

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Параметры через аргументы командной строки
parser = argparse.ArgumentParser(description="Simulate load on Recommendation service")
parser.add_argument("--duration", type=int, default=60, help="Duration of test in seconds (default: 60s)")
parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests in seconds (default: 0.5s)")
args = parser.parse_args()

# Симуляция user_id (используем реальные диапазоны из данных)
user_ids = list(range(0, 25000))  # user_id_enc от 0 до ~25000

start = time.time()
end_time = start + args.duration

request_count = 0
success_count = 0
error_count = 0
personal_count = 0
default_count = 0

logging.info(f"Запуск нагрузочного теста на {args.duration} секунд...")

while time.time() < end_time:
    user_id = random.choice(user_ids)
    k = random.choice([10, 20, 50, 100])
    
    payload = {
        "user_id": user_id,
        "k": k
    }
    
    try:
        r = requests.post(URL, json=payload)
        request_count += 1
        
        if r.status_code == 200:
            success_count += 1
            response = r.json()
            recs_count = len(response.get("recommendations", []))
            logging.info(f"user_id={user_id}, k={k} -> {recs_count} рекомендаций")
        else:
            error_count += 1
            logging.warning(f"user_id={user_id} -> status {r.status_code}")
            
    except Exception as e:
        error_count += 1
        logging.error(f"Ошибка: {e}")
    
    time.sleep(args.delay)

# Итоги
duration = time.time() - start
logging.info("=" * 50)
logging.info("ИТОГИ НАГРУЗОЧНОГО ТЕСТА")
logging.info("=" * 50)
logging.info(f"Длительность: {duration:.1f} сек")
logging.info(f"Всего запросов: {request_count}")
logging.info(f"Успешных: {success_count}")
logging.info(f"Ошибок: {error_count}")
logging.info(f"RPS: {request_count / duration:.2f}")
logging.info("=" * 50)
