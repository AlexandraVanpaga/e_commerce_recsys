# РЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА ДЛЯ ЭЛЕКТРОННОЙ КОММЕРЦИИ

**Цель проекта:** Предсказать, какие товары предложить пользователю интернет-магазина.

**Основные задачи:** 
- Анализ событий и характеристик товаров
- Выбор метрик качества
- Построение модели рекомендаций (ALS + CatBoost Ranker)
- Внедрение модели в виде веб-сервиса
- Мониторинг и автоматическое обновление модели через Airflow

---


### Активация виртуальной среды
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Скачивание сырых данных
```bash
python -m src.get_raw_data
```

### Предобработка данных
```bash
python -m src.process_data
```

**Результат предобработки:**
- Удалены дубликаты событий
- Переименованы колонки для удобства
- Извлечена информация о категориях из родительского дерева
- Версионирование чистых данных через DVC

---

## Обучение модели

### Обучение ALS и ранжирование CatBoost
```bash
python -m src.als_and_ranking
```

## Оценка модели

### Оценка работы ALS и ранжирования CatBoost
```bash
python -m src.eval
```

### Подготовка фичей (similar_items, top_products, item_features)
```bash
python -m src.prepare_features
```

---

## Запуск микросервиса

### Поднятие сервисов через Docker Compose
```bash
cd ~/mle_projects/e_commerce_recsys
docker compose up --build
```

### Проверка работы сервисов
```bash
# API
curl http://localhost:8081/health

# Prometheus
curl http://localhost:9090/-/healthy

# Grafana
curl http://localhost:3000/api/health
```

### Настройка мониторинга
**Источник данных в Grafana:** `http://prometheus:9090`

**Дашборды:**
- JSON: `rec_sys_dashboard.json`
- JPG: `rec_sys_dashboard.jpg`

---

## Использование API

### Получение рекомендаций
```bash
curl -X POST http://localhost:8081/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_id": 123, "k": 20}'
```

### Нагрузочное тестирование
```bash
python -m src.test_script
```

---

## Стратегия рекомендательной системы

**Как работает:**
1. Сервис получает `user_id`
2. Генерирует кандидатов через ALS (200 товаров)
3. Ранжирует их CatBoost с фичами:
   - Популярность товара
   - Категории и родительские категории
   - Активность пользователя
   - Веса по типам действий (просмотр=1, корзина=5, покупка=10)
   - Совпадение категорий с предпочтениями пользователя
4. Возвращает топ-K рекомендаций

**Fallback стратегия:** Если для пользователя нет ALS-кандидатов (cold-start), возвращаем топ-100 популярных товаров.

---

## Метрики качества

### Результаты на тестовых данных
**ALS**

Показывает стабильный рост Recall и ожидаемое падение Precision по мере увеличения размера рекомендованного списка.

Top-N	Recall	Precision

@10	0.0065	0.0014

@20	0.0087	0.0010

@50	0.0131	0.0006

@100	0.0175	0.0004

**CatBoost**

Top-N	Recall	Precision

@10	0.0055	0.0011

@20	0.0073	0.0008

@50	0.0122	0.0006

@100	0.0179	0.0004


CatBoost Ranker показывает схожий уровень качества с ALS.


### Выводы
Обе модели пригодны как базы рекомендаций; CatBoost даёт преимущество в возможностях расширения (учёт дополнительных фичей), а ALS представляет собой простой и быстрый базовый бейзлайн. При дообучении на всех данных результаты лучше, так как данных больше. Но эти улучшения - меньше процента, сходны с результатами данными валидации (бз прорывов).

## Дообучение модели

### Стратегия дообучения
Для симуляции реальной ситуации зарезервированы **10% данных** (события после 2015-09-01). В production эти данные объединяются с историческими, и модель переобучается через Airflow.

### Запуск Airflow
```bash
cd airflow

# Инициализация
docker compose up airflow-init

# Очистка (если нужно)
docker compose down --volumes --remove-orphans

# Запуск
docker compose up --build
```

**Название DAG:** `recsys_retrain`

**Триггер:** Запускается вручную через UI Airflow (`http://localhost:8080`)

**Шаги пайплайна:**
1. `retrain_als` — переобучение ALS на новых данных
2. `prepare_features` — пересчёт фичей
3. `generate_recommendations` — генерация ALS кандидатов
4. `train_ranker` — обучение CatBoost ранкера с оценкой качества

---

## Структура проекта
```
e_commerce_recsys/
├── data/
│   ├── archive/              # Сырые данные
│   └── processed/        # Обработанные данные
├── models/               # Сохранённые модели
├── src/
│   ├── get_raw_data.py
│   ├── process_data.py
│   ├── als_and_ranking.py
│   ├── prepare_features.py        # FastAPI сервис
│   └── test_script.py
├── airflow/
│   └── dags/
│       └── recsys_retrain.py
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🛠 Технологии

- **ML:** `implicit` (ALS), `CatBoost` (ранжирование), `scikit-learn`
- **Data:** `pandas`, `numpy`, `scipy`
- **MLOps:** `MLflow`, `DVC`, `Airflow`
- **API:** `FastAPI`, `uvicorn`
- **Monitoring:** `Prometheus`, `Grafana`
- **Infrastructure:** `Docker`, `Docker Compose`, `PostgreSQL`, `S3`