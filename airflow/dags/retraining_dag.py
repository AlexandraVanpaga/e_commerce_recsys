from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'recsys',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'recsys_retrain',
    default_args=default_args,
    description='Дообучение рекомендательной системы',
    schedule_interval=None,
    catchup=False,
)

DATA_PATH = '/opt/airflow/data/processed'
MODELS_PATH = '/opt/airflow/models'


def retrain_als():
    """Обучение ALS на train (до 2015-09-01)"""
    import os
    import pickle
    import pandas as pd
    from scipy.sparse import csr_matrix
    from sklearn.preprocessing import LabelEncoder
    from implicit.als import AlternatingLeastSquares
    
    print("ЭТАП 1: ОБУЧЕНИЕ ALS")
    
    df_events = pd.read_csv(f'{DATA_PATH}/events.csv')
    df_events['datetime'] = pd.to_datetime(df_events['timestamp'])
    
    # Разделение: train (старые) / test (новые для валидации)
    train_end = '2015-09-01'
    
    train_fit = df_events[df_events['datetime'] < train_end].copy()
    test_data = df_events[df_events['datetime'] >= train_end].copy()
    
    print(f"\nРазделение данных:")
    print(f"  Train: {len(train_fit):,} событий (до {train_end})")
    print(f"  Test:  {len(test_data):,} событий (после {train_end})")
    
    # Энкодеры на всех данных
    user_encoder = LabelEncoder()
    user_encoder.fit(df_events['visitor_id'])
    
    item_encoder = LabelEncoder()
    item_encoder.fit(df_events['item_id'])
    
    print(f"\nЭнкодинг:")
    print(f"  Уникальных пользователей: {len(user_encoder.classes_):,}")
    print(f"  Уникальных товаров: {len(item_encoder.classes_):,}")
    
    # Кодируем train для ALS
    train_fit['user_id_enc'] = user_encoder.transform(train_fit['visitor_id'])
    train_fit['item_id_enc'] = item_encoder.transform(train_fit['item_id'])
    train_fit['weight'] = train_fit['event'].map({'view': 1, 'addtocart': 5, 'transaction': 10})
    
    # Матрица из train
    user_item_matrix = csr_matrix(
        (train_fit['weight'], (train_fit['user_id_enc'], train_fit['item_id_enc'])),
        shape=(len(user_encoder.classes_), len(item_encoder.classes_))
    )
    
    sparsity = 100 * (1 - user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]))
    print(f"\nМатрица взаимодействий:")
    print(f"  Размерность: {user_item_matrix.shape}")
    print(f"  Заполненность: {user_item_matrix.nnz:,}")
    print(f"  Разреженность: {sparsity:.2f}%")
    
    # ALS на train
    print(f"\nОбучение ALS...")
    model = AlternatingLeastSquares(factors=64, iterations=15, regularization=0.01, random_state=42, use_gpu=False)
    model.fit(user_item_matrix.astype('float32'))
    
    # Сохранение
    os.makedirs(MODELS_PATH, exist_ok=True)
    with open(f'{MODELS_PATH}/als_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(f'{MODELS_PATH}/user_encoder.pkl', 'wb') as f:
        pickle.dump(user_encoder, f)
    with open(f'{MODELS_PATH}/item_encoder.pkl', 'wb') as f:
        pickle.dump(item_encoder, f)
    with open(f'{MODELS_PATH}/user_item_matrix.pkl', 'wb') as f:
        pickle.dump(user_item_matrix, f)
    
    train_fit.to_pickle(f'{MODELS_PATH}/train_fit.pkl')
    test_data.to_pickle(f'{MODELS_PATH}/test_data.pkl')
    
    print(f"ALS обучена и сохранена")


def prepare_features():
    """Подготовка фичей на train"""
    import os
    import pickle
    import pandas as pd
    
    print("ЭТАП 2: ПОДГОТОВКА ФИЧЕЙ")
    
    with open(f'{MODELS_PATH}/item_encoder.pkl', 'rb') as f:
        item_encoder = pickle.load(f)
    
    train_fit = pd.read_pickle(f'{MODELS_PATH}/train_fit.pkl')
    df_items = pd.read_csv(f'{DATA_PATH}/item_properties.csv')
    
    # Категории
    known_items = set(item_encoder.classes_)
    df_items_filtered = df_items[df_items['item_id'].isin(known_items)].copy()
    df_items_filtered['item_id_enc'] = item_encoder.transform(df_items_filtered['item_id'])
    df_items_filtered['parent_id'] = pd.to_numeric(df_items_filtered['parent_id'], errors='coerce').fillna(-1).astype(int)
    
    item_to_cat = df_items_filtered.set_index('item_id_enc')['category_id'].to_dict()
    item_to_parent = df_items_filtered.set_index('item_id_enc')['parent_id'].to_dict()
    
    train_fit['category_id'] = train_fit['item_id_enc'].map(item_to_cat).fillna(-1).astype(int)
    train_fit['parent_id'] = train_fit['item_id_enc'].map(item_to_parent).fillna(-1).astype(int)
    
    # Популярности из train
    item_popularity = train_fit.groupby('item_id_enc').size().to_dict()
    user_activity = train_fit.groupby('user_id_enc').size().to_dict()
    cat_pop = train_fit.groupby('category_id').size().to_dict()
    parent_pop = train_fit.groupby('parent_id').size().to_dict()
    
    user_cat_counts = train_fit.groupby(['user_id_enc', 'category_id']).size().reset_index(name='count')
    user_top_cat = user_cat_counts.loc[user_cat_counts.groupby('user_id_enc')['count'].idxmax()].set_index('user_id_enc')['category_id'].to_dict()
    
    user_parent_counts = train_fit.groupby(['user_id_enc', 'parent_id']).size().reset_index(name='count')
    user_top_parent = user_parent_counts.loc[user_parent_counts.groupby('user_id_enc')['count'].idxmax()].set_index('user_id_enc')['parent_id'].to_dict()
    
    features_dict = {
        'item_to_cat': item_to_cat,
        'item_to_parent': item_to_parent,
        'item_popularity': item_popularity,
        'user_activity': user_activity,
        'cat_pop': cat_pop,
        'parent_pop': parent_pop,
        'user_top_cat': user_top_cat,
        'user_top_parent': user_top_parent
    }
    
    with open(f'{MODELS_PATH}/features_dict.pkl', 'wb') as f:
        pickle.dump(features_dict, f)
    
    print(f"\nСозданные фичи:")
    print(f"  Товары с категориями: {len(item_to_cat):,}")
    print(f"  Популярности товаров: {len(item_popularity):,}")
    print(f"  Активности пользователей: {len(user_activity):,}")
    print(f"Фичи подготовлены")


def generate_recommendations():
    """Генерация ALS рекомендаций"""
    import os
    import pickle
    import pandas as pd
    from tqdm import tqdm
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("ЭТАП 3: ГЕНЕРАЦИЯ РЕКОМЕНДАЦИЙ")
    
    with open(f'{MODELS_PATH}/als_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'{MODELS_PATH}/user_encoder.pkl', 'rb') as f:
        user_encoder = pickle.load(f)
    with open(f'{MODELS_PATH}/item_encoder.pkl', 'rb') as f:
        item_encoder = pickle.load(f)
    with open(f'{MODELS_PATH}/user_item_matrix.pkl', 'rb') as f:
        user_item_matrix = pickle.load(f)
    
    train_fit = pd.read_pickle(f'{MODELS_PATH}/train_fit.pkl')
    test_data = pd.read_pickle(f'{MODELS_PATH}/test_data.pkl')
    
    # Пользователи с историей в train И присутствующие в test
    test_data['user_id_enc'] = user_encoder.transform(test_data['visitor_id'])
    
    users_with_history = set(train_fit['user_id_enc'].unique())
    test_users = set(test_data['user_id_enc'].unique())
    target_users = list(users_with_history & test_users)
    
    print(f"\nЦелевые пользователи:")
    print(f"  В train: {len(users_with_history):,}")
    print(f"  В test: {len(test_users):,}")
    print(f"  Пересечение: {len(target_users):,}")
    
    # Ограничение для скорости
    target_users = target_users[:10000]
    print(f"  Используем: {len(target_users):,}")
    
    max_user_id = model.user_factors.shape[0]
    target_users = [u for u in target_users if u < max_user_id]
    
    print(f"\nГенерация рекомендаций...")
    
    personal_recs = []
    for user_id in tqdm(target_users, desc="Recommendations"):
        try:
            item_ids, scores = model.recommend(user_id, user_item_matrix[user_id], N=200, filter_already_liked_items=True)
            for rank, (item_id, score) in enumerate(zip(item_ids, scores), 1):
                if item_id < len(item_encoder.classes_):
                    personal_recs.append({
                        'user_id_enc': user_id,
                        'item_id_enc': item_id,
                        'score': float(score),
                        'rank': rank
                    })
        except:
            continue
    
    personal_als = pd.DataFrame(personal_recs)
    personal_als['visitor_id'] = user_encoder.inverse_transform(personal_als['user_id_enc'])
    personal_als['item_id'] = item_encoder.inverse_transform(personal_als['item_id_enc'])
    
    print(f"\nСтатистика рекомендаций:")
    print(f"  Всего пар (user, item): {len(personal_als):,}")
    print(f"  Уникальных пользователей: {personal_als['user_id_enc'].nunique():,}")
    
    # S3
    endpoint_url = os.getenv("S3_ENDPOINT_URL")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    bucket_name = os.getenv("S3_BUCKET_NAME")
    
    storage_options = {
        'key': aws_access_key_id,
        'secret': aws_secret_access_key,
        'client_kwargs': {'endpoint_url': endpoint_url}
    }
    
    personal_als.to_parquet(f's3://{bucket_name}/recsys/recommendations/personal_als.parquet', index=False, storage_options=storage_options)
    
    print(f"Рекомендации сохранены в S3")


def train_ranker():
    """Обучение CatBoost ранкера с оценкой качества"""
    import os
    import pickle
    import pandas as pd
    import numpy as np
    from catboost import CatBoostRanker
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("ЭТАП 4: ОБУЧЕНИЕ РАНКЕРА И ОЦЕНКА")
    
    endpoint_url = os.getenv("S3_ENDPOINT_URL")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    bucket_name = os.getenv("S3_BUCKET_NAME")
    
    storage_options = {
        'key': aws_access_key_id,
        'secret': aws_secret_access_key,
        'client_kwargs': {'endpoint_url': endpoint_url}
    }
    
    # Загрузка
    personal_als = pd.read_parquet(f's3://{bucket_name}/recsys/recommendations/personal_als.parquet', storage_options=storage_options)
    
    with open(f'{MODELS_PATH}/features_dict.pkl', 'rb') as f:
        features_dict = pickle.load(f)
    
    test_data = pd.read_pickle(f'{MODELS_PATH}/test_data.pkl')
    
    # Кодируем test
    with open(f'{MODELS_PATH}/user_encoder.pkl', 'rb') as f:
        user_encoder = pickle.load(f)
    with open(f'{MODELS_PATH}/item_encoder.pkl', 'rb') as f:
        item_encoder = pickle.load(f)
    
    test_data['user_id_enc'] = user_encoder.transform(test_data['visitor_id'])
    test_data['item_id_enc'] = item_encoder.transform(test_data['item_id'])
    
    print(f"\nПодготовка данных для ранкера...")
    
    # Подготовка данных
    train_for_ranking = personal_als[['user_id_enc', 'item_id_enc', 'score', 'rank']].copy()
    
    # Фичи
    train_for_ranking['als_score'] = train_for_ranking['score']
    train_for_ranking['item_popularity'] = train_for_ranking['item_id_enc'].map(features_dict['item_popularity']).fillna(0)
    train_for_ranking['user_activity'] = train_for_ranking['user_id_enc'].map(features_dict['user_activity']).fillna(0)
    train_for_ranking['category_id'] = train_for_ranking['item_id_enc'].map(features_dict['item_to_cat']).fillna(-1).astype(int)
    train_for_ranking['parent_id'] = train_for_ranking['item_id_enc'].map(features_dict['item_to_parent']).fillna(-1).astype(int)
    train_for_ranking['category_popularity'] = train_for_ranking['category_id'].map(features_dict['cat_pop']).fillna(0)
    train_for_ranking['parent_popularity'] = train_for_ranking['parent_id'].map(features_dict['parent_pop']).fillna(0)
    train_for_ranking['user_top_category'] = train_for_ranking['user_id_enc'].map(features_dict['user_top_cat']).fillna(-1).astype(int)
    train_for_ranking['user_top_parent'] = train_for_ranking['user_id_enc'].map(features_dict['user_top_parent']).fillna(-1).astype(int)
    train_for_ranking['category_match'] = (train_for_ranking['category_id'] == train_for_ranking['user_top_category']).astype(int)
    train_for_ranking['parent_match'] = (train_for_ranking['parent_id'] == train_for_ranking['user_top_parent']).astype(int)
    
    # Таргет из test (новые данные)
    test_interactions = test_data.groupby(['user_id_enc', 'item_id_enc']).size().reset_index(name='interactions')
    train_for_ranking = train_for_ranking.merge(test_interactions, on=['user_id_enc', 'item_id_enc'], how='left')
    train_for_ranking['target'] = train_for_ranking['interactions'].fillna(0).astype(int)
    
    print(f"\nСтатистика кандидатов:")
    print(f"  Всего кандидатов: {len(train_for_ranking):,}")
    print(f"  Позитивных: {train_for_ranking['target'].sum():,} ({train_for_ranking['target'].mean()*100:.2f}%)")
    
    # Fallback
    if train_for_ranking['target'].sum() == 0:
        print("\nWARNING: Нет позитивных примеров. Создаём синтетические...")
        threshold = train_for_ranking['als_score'].quantile(0.9)
        train_for_ranking['target'] = (train_for_ranking['als_score'] >= threshold).astype(int)
        print(f"Синтетических позитивных: {train_for_ranking['target'].sum():,}")
    
    features = ['als_score', 'item_popularity', 'user_activity', 'category_popularity', 'parent_popularity', 'category_match', 'parent_match']
    
    # Балансировка
    positives = train_for_ranking[train_for_ranking['target'] == 1]
    negatives = train_for_ranking[train_for_ranking['target'] == 0]
    negatives_sampled = negatives.sample(n=min(len(positives) * 20, len(negatives)), random_state=42)
    
    train_balanced = pd.concat([positives, negatives_sampled]).sort_values('user_id_enc').reset_index(drop=True)
    
    print(f"\nБалансировка:")
    print(f"  Balanced dataset: {len(train_balanced):,}")
    print(f"  Позитивы: {len(positives):,} ({train_balanced['target'].mean()*100:.1f}%)")
    
    # Обучение
    print(f"\nОбучение CatBoost...")
    ranker = CatBoostRanker(iterations=200, depth=6, learning_rate=0.1, random_seed=42, verbose=False)
    ranker.fit(train_balanced[features], train_balanced['target'], group_id=train_balanced['user_id_enc'])
    
    ranker.save_model(f'{MODELS_PATH}/prod_model_catboost_ranker.cbm')
    
    # Предсказание
    print(f"\nГенерация финальных рекомендаций...")
    train_for_ranking['final_score'] = ranker.predict(train_for_ranking[features])
    
    recommendations = train_for_ranking.sort_values(['user_id_enc', 'final_score'], ascending=[True, False])
    recommendations = recommendations.groupby('user_id_enc').head(100).reset_index(drop=True)
    
    # МЕТРИКИ
    print("\n" + "=" * 50)
    print("МЕТРИКИ КАЧЕСТВА")
    print("=" * 50)
    
    # Coverage
    coverage = recommendations['user_id_enc'].nunique()
    total_users = len(train_for_ranking['user_id_enc'].unique())
    print(f"\nCoverage:")
    print(f"  Пользователей с рекомендациями: {coverage:,} / {total_users:,} ({coverage/total_users*100:.1f}%)")
    
    # Recall и Precision
    ranked_user_recs = recommendations.groupby('user_id_enc')['item_id_enc'].apply(list).to_dict()
    test_user_items = test_data.groupby('user_id_enc')['item_id_enc'].apply(list).to_dict()
    
    print(f"\nМетрики на тестовом множестве:")
    print(f"{'K':<8} {'Recall':<10} {'Precision':<12} {'Users':<10}")
    print("-" * 40)
    
    for k in [10, 20, 50, 100]:
        recalls, precisions = [], []
        for user_id, actual in test_user_items.items():
            if user_id in ranked_user_recs:
                recs = ranked_user_recs[user_id][:k]
                hits = len(set(recs) & set(actual))
                recalls.append(hits / min(len(actual), k))
                precisions.append(hits / k if k > 0 else 0)
        
        recall_avg = np.mean(recalls) if recalls else 0
        precision_avg = np.mean(precisions) if precisions else 0
        
        print(f"@{k:<7} {recall_avg:<10.4f} {precision_avg:<12.4f} {len(recalls):<10,}")
    
    # Алерт
    recall_10 = np.mean([
        len(set(ranked_user_recs.get(user_id, [])[:10]) & set(actual)) / min(len(actual), 10)
        for user_id, actual in test_user_items.items() if user_id in ranked_user_recs
    ])
    
    print(f"\n{'=' * 50}")
    if recall_10 < 0.01:
        print("WARNING: Низкое качество модели (Recall@10 < 1%)")
    elif recall_10 < 0.05:
        print(f"Модель работает удовлетворительно (Recall@10: {recall_10*100:.2f}%)")
    else:
        print(f"Модель показывает хорошие результаты (Recall@10: {recall_10*100:.2f}%)")
    print(f"{'=' * 50}")
    
    # Сохранение
    recommendations.to_parquet(
        f's3://{bucket_name}/recsys/recommendations/final_recommendations.parquet',
        index=False,
        storage_options=storage_options
    )
    
    print(f"\nФинальные рекомендации сохранены в S3")
    print(f"  Всего рекомендаций: {len(recommendations):,}")
    print(f"  Пользователей: {recommendations['user_id_enc'].nunique():,}")


# Tasks
task_retrain_als = PythonOperator(
    task_id='retrain_als',
    python_callable=retrain_als,
    dag=dag,
)

task_prepare_features = PythonOperator(
    task_id='prepare_features',
    python_callable=prepare_features,
    dag=dag,
)

task_generate_recs = PythonOperator(
    task_id='generate_recommendations',
    python_callable=generate_recommendations,
    dag=dag,
)

task_train_ranker = PythonOperator(
    task_id='train_ranker',
    python_callable=train_ranker,
    dag=dag,
)

# Pipeline
task_retrain_als >> task_prepare_features >> task_generate_recs >> task_train_ranker