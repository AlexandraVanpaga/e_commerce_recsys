# src/model_train.py

import os
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from implicit.als import AlternatingLeastSquares
from catboost import CatBoostRanker
from tqdm import tqdm
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

RANDOM_SEED = 42

# S3 настройки
endpoint_url = os.getenv("S3_ENDPOINT_URL")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
bucket_name = os.getenv("S3_BUCKET_NAME")

storage_options = {
    'key': aws_access_key_id,
    'secret': aws_secret_access_key,
    'client_kwargs': {'endpoint_url': endpoint_url}
}


def validate_data(df_events, df_items):
    """Валидация загруженных данных"""
    print("\nВалидация данных...")
    
    # Проверка на пустоту
    assert not df_events.empty, "❌ df_events пустой!"
    assert not df_items.empty, "❌ df_items пустой!"
    
    # Проверка обязательных колонок
    required_events_cols = ['timestamp', 'visitor_id', 'event', 'item_id']
    required_items_cols = ['item_id', 'category_id', 'parent_id']
    
    for col in required_events_cols:
        assert col in df_events.columns, f"❌ Колонка '{col}' отсутствует в events!"
    
    for col in required_items_cols:
        assert col in df_items.columns, f"❌ Колонка '{col}' отсутствует в items!"
    
    # Проверка типов событий
    valid_events = {'view', 'addtocart', 'transaction'}
    actual_events = set(df_events['event'].unique())
    assert actual_events.issubset(valid_events), f"❌ Неизвестные события: {actual_events - valid_events}"
    
    # Проверка на пропуски
    assert df_events['visitor_id'].notna().all(), "❌ Пропуски в visitor_id!"
    assert df_events['item_id'].notna().all(), "❌ Пропуски в item_id!"
    assert df_events['timestamp'].notna().all(), "❌ Пропуски в timestamp!"
    
    # Проверка диапазонов
    assert df_events['visitor_id'].min() > 0, "❌ visitor_id должны быть положительными!"
    assert df_events['item_id'].min() > 0, "❌ item_id должны быть положительными!"
    
    # Статистика
    print(f"✓ События: {len(df_events):,} строк")
    print(f"✓ Товары: {len(df_items):,} строк")
    print(f"✓ Уникальных пользователей: {df_events['visitor_id'].nunique():,}")
    print(f"✓ Уникальных товаров: {df_events['item_id'].nunique():,}")
    print(f"✓ Распределение событий:")
    for event, count in df_events['event'].value_counts().items():
        print(f"  - {event}: {count:,} ({count/len(df_events)*100:.1f}%)")
    
    # Проверка дат
    df_events['datetime'] = pd.to_datetime(df_events['timestamp'])
    date_range = df_events['datetime'].max() - df_events['datetime'].min()
    print(f"✓ Период данных: {df_events['datetime'].min().date()} - {df_events['datetime'].max().date()} ({date_range.days} дней)")
    
    # Предупреждения
    if df_events['visitor_id'].nunique() < 1000:
        print("⚠️ WARNING: Мало уникальных пользователей (<1000)")
    
    if df_events['item_id'].nunique() < 100:
        print("⚠️ WARNING: Мало уникальных товаров (<100)")
    
    transactions = df_events[df_events['event'] == 'transaction']
    if len(transactions) / len(df_events) < 0.001:
        print(f"⚠️ WARNING: Очень мало покупок ({len(transactions)/len(df_events)*100:.2f}%)")
    
    print("✓ Валидация пройдена!\n")


def load_data():
    """Загрузка данных"""
    print("Загрузка данных...")
    df_events = pd.read_csv('data/processed/events.csv')
    df_items = pd.read_csv('data/processed/item_properties.csv')
    
    # Валидация
    validate_data(df_events, df_items)
    
    df_events['datetime'] = pd.to_datetime(df_events['timestamp'])
    print(f"Events: {len(df_events):,}, Items: {len(df_items):,}")
    return df_events, df_items


def split_data(df_events):
    """Разделение данных по датам"""
    train_end = '2015-08-01'
    retrain_end = '2015-09-01'
    
    train_fit = df_events[df_events['datetime'] < train_end].copy()
    train_val = df_events[(df_events['datetime'] >= train_end) & (df_events['datetime'] < retrain_end)].copy()
    new_events = df_events[df_events['datetime'] >= retrain_end].copy()
    
    # Валидация разбиения
    assert len(train_fit) > 0, "❌ Train set пустой!"
    assert len(train_val) > 0, "❌ Val set пустой!"
    assert len(new_events) > 0, "❌ Test set пустой!"
    
    total = len(train_fit) + len(train_val) + len(new_events)
    assert total == len(df_events), "❌ Потеря данных при разбиении!"
    
    print(f"Train fit: {len(train_fit):,} (до {train_end}) - {len(train_fit)/total*100:.1f}%")
    print(f"Train val: {len(train_val):,} ({train_end} - {retrain_end}) - {len(train_val)/total*100:.1f}%")
    print(f"New events: {len(new_events):,} (с {retrain_end}) - {len(new_events)/total*100:.1f}%")
    
    return train_fit, train_val, new_events


def create_encoders(df_events, train_fit, train_val, new_events):
    """Создание и сохранение энкодеров"""
    user_encoder = LabelEncoder()
    user_encoder.fit(df_events['visitor_id'])
    
    item_encoder = LabelEncoder()
    item_encoder.fit(df_events['item_id'])
    
    # Применяем ко всем датасетам
    for df in [train_fit, train_val, new_events]:
        df['user_id_enc'] = user_encoder.transform(df['visitor_id'])
        df['item_id_enc'] = item_encoder.transform(df['item_id'])
    
    print(f"Уникальных пользователей: {len(user_encoder.classes_):,}")
    print(f"Уникальных товаров: {len(item_encoder.classes_):,}")
    
    # Сохранить энкодеры
    os.makedirs('models', exist_ok=True)
    with open('models/user_encoder.pkl', 'wb') as f:
        pickle.dump(user_encoder, f)
    with open('models/item_encoder.pkl', 'wb') as f:
        pickle.dump(item_encoder, f)
    
    return user_encoder, item_encoder


def train_als(train_fit, user_encoder, item_encoder):
    """Обучение ALS модели"""
    train_fit['weight'] = train_fit['event'].map({'view': 1, 'addtocart': 5, 'transaction': 10})
    
    # Валидация весов
    assert train_fit['weight'].notna().all(), "❌ Неизвестные типы событий!"
    
    user_item_matrix = csr_matrix(
        (train_fit['weight'], (train_fit['user_id_enc'], train_fit['item_id_enc'])),
        shape=(len(user_encoder.classes_), len(item_encoder.classes_))
    )
    
    # Валидация матрицы
    assert user_item_matrix.nnz > 0, "❌ Матрица взаимодействий пустая!"
    
    print(f"Матрица: {user_item_matrix.shape}")
    print(f"Заполненность: {user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]) * 100:.4f}%")
    
    model = AlternatingLeastSquares(
        factors=64,
        iterations=15,
        regularization=0.01,
        random_state=RANDOM_SEED,
        use_gpu=False
    )
    
    print("Обучение ALS...")
    model.fit(user_item_matrix.astype('float32'))
    print("ALS обучена")
    
    # Сохранить ALS модель
    with open('models/als_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model, user_item_matrix


def generate_als_recommendations(model, user_item_matrix, train_fit, train_val, user_encoder, item_encoder, n_candidates=200):
    """Генерация ALS рекомендаций"""
    users_with_history = set(train_fit['user_id_enc'].unique())
    val_users = set(train_val['user_id_enc'].unique())
    target_users = users_with_history & val_users
    
    max_user_id = model.user_factors.shape[0]
    target_users = [u for u in target_users if u < max_user_id]
    
    # Валидация
    assert len(target_users) > 0, "❌ Нет целевых пользователей для рекомендаций!"
    
    print(f"Пользователей для рекомендаций: {len(target_users):,}")
    
    personal_recs = []
    errors = 0
    
    for user_id in tqdm(target_users, desc="ALS Recommendations"):
        try:
            item_ids, scores = model.recommend(
                user_id, 
                user_item_matrix[user_id], 
                N=n_candidates, 
                filter_already_liked_items=True
            )
            
            for rank, (item_id, score) in enumerate(zip(item_ids, scores), 1):
                if item_id < len(item_encoder.classes_):
                    personal_recs.append({
                        'user_id_enc': user_id, 
                        'item_id_enc': item_id, 
                        'score': float(score), 
                        'rank': rank
                    })
        except Exception:
            errors += 1
            continue
    
    # Валидация результата
    assert len(personal_recs) > 0, "❌ Не удалось сгенерировать рекомендации!"
    
    personal_als = pd.DataFrame(personal_recs)
    personal_als['visitor_id'] = user_encoder.inverse_transform(personal_als['user_id_enc'])
    personal_als['item_id'] = item_encoder.inverse_transform(personal_als['item_id_enc'])
    
    print(f"ALS рекомендации: {len(personal_als):,}")
    print(f"Пользователей: {personal_als['user_id_enc'].nunique():,}")
    if errors > 0:
        print(f"⚠️ Ошибок при генерации: {errors}")
    
    return personal_als


def prepare_features(train_fit, df_items, item_encoder):
    """Подготовка фичей для ранкера"""
    # Категорийные фичи
    known_items = set(item_encoder.classes_)
    df_items_filtered = df_items[df_items['item_id'].isin(known_items)].copy()
    df_items_filtered['item_id_enc'] = item_encoder.transform(df_items_filtered['item_id'])
    df_items_filtered['parent_id'] = pd.to_numeric(df_items_filtered['parent_id'], errors='coerce').fillna(-1).astype(int)
    
    # Словари
    item_to_cat = df_items_filtered.set_index('item_id_enc')['category_id'].to_dict()
    item_to_parent = df_items_filtered.set_index('item_id_enc')['parent_id'].to_dict()
    
    # Добавляем категории к train_fit
    train_fit['category_id'] = train_fit['item_id_enc'].map(item_to_cat).fillna(-1).astype(int)
    train_fit['parent_id'] = train_fit['item_id_enc'].map(item_to_parent).fillna(-1).astype(int)
    
    # Популярность
    item_popularity = train_fit.groupby('item_id_enc').size().to_dict()
    user_activity = train_fit.groupby('user_id_enc').size().to_dict()
    cat_pop = train_fit.groupby('category_id').size().to_dict()
    parent_pop = train_fit.groupby('parent_id').size().to_dict()
    
    # Веса
    item_weight_sum = train_fit.groupby('item_id_enc')['weight'].sum().to_dict()
    user_weight_sum = train_fit.groupby('user_id_enc')['weight'].sum().to_dict()
    ui_weight = train_fit.groupby(['user_id_enc', 'item_id_enc'])['weight'].sum().reset_index(name='ui_weight')
    
    # Топ категории пользователя
    user_cat_counts = train_fit.groupby(['user_id_enc', 'category_id']).size().reset_index(name='count')
    user_top_cat = user_cat_counts.loc[user_cat_counts.groupby('user_id_enc')['count'].idxmax()].set_index('user_id_enc')['category_id'].to_dict()
    
    user_parent_counts = train_fit.groupby(['user_id_enc', 'parent_id']).size().reset_index(name='count')
    user_top_parent = user_parent_counts.loc[user_parent_counts.groupby('user_id_enc')['count'].idxmax()].set_index('user_id_enc')['parent_id'].to_dict()
    
    features_dict = {
        'item_to_cat': item_to_cat,
        'item_to_parent': item_to_parent,
        'item_popularity': item_popularity,
        'user_activity': user_activity,
        'item_weight_sum': item_weight_sum,
        'user_weight_sum': user_weight_sum,
        'cat_pop': cat_pop,
        'parent_pop': parent_pop,
        'user_top_cat': user_top_cat,
        'user_top_parent': user_top_parent
    }
    
    return features_dict, ui_weight


def prepare_ranking_data(personal_als, train_val, features_dict, ui_weight):
    """Подготовка данных для ранкера"""
    train_for_ranking = personal_als[['user_id_enc', 'item_id_enc', 'score', 'rank']].copy()
    
    # Базовые фичи
    train_for_ranking['als_score'] = train_for_ranking['score']
    train_for_ranking['item_popularity'] = train_for_ranking['item_id_enc'].map(features_dict['item_popularity']).fillna(0)
    train_for_ranking['user_activity'] = train_for_ranking['user_id_enc'].map(features_dict['user_activity']).fillna(0)
    
    # Вес-фичи
    train_for_ranking['item_weight_sum'] = train_for_ranking['item_id_enc'].map(features_dict['item_weight_sum']).fillna(0)
    train_for_ranking['user_weight_sum'] = train_for_ranking['user_id_enc'].map(features_dict['user_weight_sum']).fillna(0)
    train_for_ranking = train_for_ranking.merge(ui_weight, on=['user_id_enc', 'item_id_enc'], how='left')
    train_for_ranking['ui_weight'] = train_for_ranking['ui_weight'].fillna(0)
    
    # Категорийные фичи
    train_for_ranking['category_id'] = train_for_ranking['item_id_enc'].map(features_dict['item_to_cat']).fillna(-1).astype(int)
    train_for_ranking['parent_id'] = train_for_ranking['item_id_enc'].map(features_dict['item_to_parent']).fillna(-1).astype(int)
    train_for_ranking['category_popularity'] = train_for_ranking['category_id'].map(features_dict['cat_pop']).fillna(0)
    train_for_ranking['parent_popularity'] = train_for_ranking['parent_id'].map(features_dict['parent_pop']).fillna(0)
    train_for_ranking['user_top_category'] = train_for_ranking['user_id_enc'].map(features_dict['user_top_cat']).fillna(-1).astype(int)
    train_for_ranking['user_top_parent'] = train_for_ranking['user_id_enc'].map(features_dict['user_top_parent']).fillna(-1).astype(int)
    train_for_ranking['category_match'] = (train_for_ranking['category_id'] == train_for_ranking['user_top_category']).astype(int)
    train_for_ranking['parent_match'] = (train_for_ranking['parent_id'] == train_for_ranking['user_top_parent']).astype(int)
    
    # Таргет
    val_interactions = train_val.groupby(['user_id_enc', 'item_id_enc']).size().reset_index(name='interactions')
    train_for_ranking = train_for_ranking.merge(val_interactions, on=['user_id_enc', 'item_id_enc'], how='left')
    train_for_ranking['target'] = train_for_ranking['interactions'].fillna(0).astype(int)
    
    print(f"Кандидатов: {len(train_for_ranking):,}")
    print(f"Позитивных: {train_for_ranking['target'].sum():,} ({train_for_ranking['target'].mean()*100:.2f}%)")
    
    return train_for_ranking


def train_ranker(train_for_ranking):
    """Обучение CatBoost ранкера"""
    features = [
        'als_score', 
        'item_popularity', 
        'user_activity',
        'item_weight_sum',
        'user_weight_sum',
        'ui_weight',
        'category_popularity', 
        'parent_popularity', 
        'category_match', 
        'parent_match'
    ]
    cat_features = ['category_id', 'parent_id', 'user_top_category', 'user_top_parent']
    
    # Балансировка
    positives = train_for_ranking[train_for_ranking['target'] == 1]
    negatives = train_for_ranking[train_for_ranking['target'] == 0]
    negatives_sampled = negatives.sample(n=min(len(positives) * 20, len(negatives)), random_state=RANDOM_SEED)
    
    train_balanced = pd.concat([positives, negatives_sampled]).sort_values('user_id_enc').reset_index(drop=True)
    
    print(f"Balanced: {len(train_balanced):,} ({train_balanced['target'].mean()*100:.1f}% pos)")
    
    ranker = CatBoostRanker(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_SEED,
        verbose=20
    )
    
    ranker.fit(
        train_balanced[features + cat_features],
        train_balanced['target'],
        group_id=train_balanced['user_id_enc'],
        cat_features=cat_features
    )
    
    # Сохранить модель
    ranker.save_model('models/prod_model_catboost_ranker.cbm')
    print("CatBoost модель сохранена")
    
    return ranker, features, cat_features


def generate_final_recommendations(train_for_ranking, ranker, features, cat_features):
    """Генерация финальных рекомендаций"""
    train_for_ranking['final_score'] = ranker.predict(train_for_ranking[features + cat_features])
    
    recommendations = train_for_ranking.sort_values(['user_id_enc', 'final_score'], ascending=[True, False])
    recommendations = recommendations.groupby('user_id_enc').head(100).reset_index(drop=True)
    
    print(f"Финальные рекомендации: {len(recommendations):,}")
    print(f"Пользователей: {recommendations['user_id_enc'].nunique():,}")
    
    return recommendations


def save_to_s3(personal_als, recommendations, features_dict, ui_weight):
    """Сохранение в S3"""
    # ALS рекомендации
    personal_als.to_parquet(
        f's3://{bucket_name}/recsys/recommendations/personal_als.parquet',
        index=False,
        storage_options=storage_options
    )
    
    # Финальные рекомендации
    recommendations.to_parquet(
        f's3://{bucket_name}/recsys/recommendations/final_recommendations.parquet',
        index=False,
        storage_options=storage_options
    )
    
    # Фичи
    with open('models/features_dict.pkl', 'wb') as f:
        pickle.dump(features_dict, f)
    
    # UI веса
    ui_weight.to_pickle('models/ui_weight.pkl')
    
    print("Данные сохранены в S3")


def main():
    print("=" * 50)
    print("ОБУЧЕНИЕ РЕКОМЕНДАТЕЛЬНОЙ СИСТЕМЫ")
    print("=" * 50)
    
    # 1. Загрузка данных
    df_events, df_items = load_data()
    
    # 2. Разделение данных
    train_fit, train_val, new_events = split_data(df_events)
    
    # 3. Энкодеры
    user_encoder, item_encoder = create_encoders(df_events, train_fit, train_val, new_events)
    
    # 4. Обучение ALS
    als_model, user_item_matrix = train_als(train_fit, user_encoder, item_encoder)
    
    # 5. Генерация ALS рекомендаций
    personal_als = generate_als_recommendations(
        als_model, user_item_matrix, train_fit, train_val, user_encoder, item_encoder, n_candidates=200
    )
    
    # 6. Подготовка фичей
    features_dict, ui_weight = prepare_features(train_fit, df_items, item_encoder)
    
    # 7. Подготовка данных для ранкера
    train_for_ranking = prepare_ranking_data(personal_als, train_val, features_dict, ui_weight)
    
    # 8. Обучение ранкера
    ranker, features, cat_features = train_ranker(train_for_ranking)
    
    # 9. Генерация финальных рекомендаций
    recommendations = generate_final_recommendations(train_for_ranking, ranker, features, cat_features)
    
    # 10. Сохранение в S3
    save_to_s3(personal_als, recommendations, features_dict, ui_weight)
    
    print("=" * 50)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 50)


if __name__ == "__main__":
    main()