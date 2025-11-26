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

| Метрика | Значение | Интерпретация |
|---------|----------|---------------|
| **Recall@10** | 1.27% | Из всех товаров, которые пользователь купит, модель угадывает ~1.3% в первых 10 рекомендациях |
| **Recall@20** | 1.84% | В топ-20 угадываем ~1.9% покупок |
| **Recall@50** | 2.65% | В топ-50 угадываем ~2.5% покупок |
| **Recall@100** | 3.32% | В топ-100 угадываем ~3.2% покупок |
| **Precision@10-100** | 0.09-0.29% | Из 100 рекомендаций 1-3 реально купят (ожидаемо для e-commerce) |

### Выводы
- Модель работает и показывает персонализацию
- ALS + CatBoost улучшает baseline (топ-продукты) на **11-19%**
- Для e-commerce с миллионами товаров и непредсказуемым поведением это нормальный результат
- Модель успешно поднимает релевантные товары выше в списке

---

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