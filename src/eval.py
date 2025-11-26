# src/evaluate_model.py

import pickle
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from tqdm import tqdm

load_dotenv()

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


def recall_at_k(recommended, actual, k=100):
    recommended_k = set(recommended[:k])
    actual_set = set(actual)
    if len(actual_set) == 0:
        return 0
    return len(recommended_k & actual_set) / len(actual_set)


def precision_at_k(recommended, actual, k=100):
    recommended_k = set(recommended[:k])
    actual_set = set(actual)
    if len(recommended_k) == 0:
        return 0
    return len(recommended_k & actual_set) / len(recommended_k)


def evaluate_model_at_k(user_recs_dict, val_user_items, name, k_values=[10, 20, 50, 100], is_personalized=True):
    print(f"\n{name}")
    
    for k in k_values:
        recalls, precisions = [], []
        
        for user_id, actual in val_user_items.items():
            if is_personalized:
                if user_id not in user_recs_dict:
                    continue
                recs = user_recs_dict[user_id]
            else:
                recs = user_recs_dict
            
            recalls.append(recall_at_k(recs, actual, k))
            precisions.append(precision_at_k(recs, actual, k))
        
        print(f"  @{k}: Recall={np.mean(recalls):.4f}, Precision={np.mean(precisions):.4f}")
    
    print(f"  Users: {len(recalls):,}")


def main():
    print("=" * 70)
    print("ФИНАЛЬНАЯ ОЦЕНКА НА TEST SET (HOLDOUT)")
    print("=" * 70)
    
    # Загрузка данных
    df_events = pd.read_csv('data/processed/events.csv')
    df_events['datetime'] = pd.to_datetime(df_events['timestamp'])
    
    with open('models/user_encoder.pkl', 'rb') as f:
        user_encoder = pickle.load(f)
    with open('models/item_encoder.pkl', 'rb') as f:
        item_encoder = pickle.load(f)
    with open('models/als_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    from catboost import CatBoostRanker
    ranker = CatBoostRanker()
    ranker.load_model('models/catboost_ranker_final.cbm')
    
    # Train и Test split
    train_end = '2015-08-01'
    test_start = '2015-09-01'
    
    train_fit = df_events[df_events['datetime'] < train_end].copy()
    test = df_events[df_events['datetime'] >= test_start].copy()
    
    # Энкодинг
    train_fit['user_id_enc'] = user_encoder.transform(train_fit['visitor_id'])
    train_fit['item_id_enc'] = item_encoder.transform(train_fit['item_id'])
    train_fit['weight'] = train_fit['event'].map({'view': 1, 'addtocart': 5, 'transaction': 10})
    
    test['user_id_enc'] = user_encoder.transform(test['visitor_id'])
    test['item_id_enc'] = item_encoder.transform(test['item_id'])
    
    print(f"\nTrain: {len(train_fit):,} событий")
    print(f"Test: {len(test):,} событий")
    print(f"Test пользователей: {test['user_id_enc'].nunique():,}")
    
    # Создание матрицы для ALS
    from scipy.sparse import csr_matrix
    user_item_matrix = csr_matrix(
        (train_fit['weight'], (train_fit['user_id_enc'], train_fit['item_id_enc'])),
        shape=(len(user_encoder.classes_), len(item_encoder.classes_))
    )
    
    # Загрузка фичей
    with open('models/features_dict.pkl', 'rb') as f:
        features_dict = pickle.load(f)
    
    ui_weight = pd.read_pickle('models/ui_weight.pkl')
    
    # Подготовка для оценки
    top_popular_list = train_fit['item_id_enc'].value_counts().head(100).index.tolist()
    
    # ГЕНЕРАЦИЯ ALS РЕКОМЕНДАЦИЙ ДЛЯ TEST
    users_with_history = set(train_fit['user_id_enc'].unique())
    test_users = set(test['user_id_enc'].unique())
    target_test_users = list(users_with_history & test_users)
    
    max_user_id = model.user_factors.shape[0]
    target_test_users = [u for u in target_test_users if u < max_user_id]
    
    print(f"\nГенерация ALS рекомендаций для {len(target_test_users):,} test пользователей...")
    
    test_als_recs = []
    for user_id in tqdm(target_test_users, desc="ALS Test"):
        try:
            item_ids, scores = model.recommend(
                user_id, 
                user_item_matrix[user_id], 
                N=200, 
                filter_already_liked_items=True
            )
            
            for rank, (item_id, score) in enumerate(zip(item_ids, scores), 1):
                if item_id < len(item_encoder.classes_):
                    test_als_recs.append({
                        'user_id_enc': user_id, 
                        'item_id_enc': item_id, 
                        'score': float(score), 
                        'rank': rank
                    })
        except:
            continue
    
    test_als = pd.DataFrame(test_als_recs)
    print(f"ALS рекомендаций: {len(test_als):,}")
    
    # ПРИМЕНЕНИЕ РАНКЕРА
    test_for_ranking = test_als[['user_id_enc', 'item_id_enc', 'score', 'rank']].copy()
    
    # Фичи
    test_for_ranking['als_score'] = test_for_ranking['score']
    test_for_ranking['item_popularity'] = test_for_ranking['item_id_enc'].map(features_dict['item_popularity']).fillna(0)
    test_for_ranking['user_activity'] = test_for_ranking['user_id_enc'].map(features_dict['user_activity']).fillna(0)
    test_for_ranking['item_weight_sum'] = test_for_ranking['item_id_enc'].map(features_dict['item_weight_sum']).fillna(0)
    test_for_ranking['user_weight_sum'] = test_for_ranking['user_id_enc'].map(features_dict['user_weight_sum']).fillna(0)
    test_for_ranking = test_for_ranking.merge(ui_weight, on=['user_id_enc', 'item_id_enc'], how='left')
    test_for_ranking['ui_weight'] = test_for_ranking['ui_weight'].fillna(0)
    test_for_ranking['category_id'] = test_for_ranking['item_id_enc'].map(features_dict['item_to_cat']).fillna(-1).astype(int)
    test_for_ranking['parent_id'] = test_for_ranking['item_id_enc'].map(features_dict['item_to_parent']).fillna(-1).astype(int)
    test_for_ranking['category_popularity'] = test_for_ranking['category_id'].map(features_dict['cat_pop']).fillna(0)
    test_for_ranking['parent_popularity'] = test_for_ranking['parent_id'].map(features_dict['parent_pop']).fillna(0)
    test_for_ranking['user_top_category'] = test_for_ranking['user_id_enc'].map(features_dict['user_top_cat']).fillna(-1).astype(int)
    test_for_ranking['user_top_parent'] = test_for_ranking['user_id_enc'].map(features_dict['user_top_parent']).fillna(-1).astype(int)
    test_for_ranking['category_match'] = (test_for_ranking['category_id'] == test_for_ranking['user_top_category']).astype(int)
    test_for_ranking['parent_match'] = (test_for_ranking['parent_id'] == test_for_ranking['user_top_parent']).astype(int)
    
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
    
    print("Применение CatBoost ранкера...")
    test_for_ranking['final_score'] = ranker.predict(test_for_ranking[features + cat_features])
    
    test_recommendations = test_for_ranking.sort_values(['user_id_enc', 'final_score'], ascending=[True, False])
    test_recommendations = test_recommendations.groupby('user_id_enc').head(100).reset_index(drop=True)
    
    print(f"Финальных рекомендаций: {len(test_recommendations):,}")
    
    # ОЦЕНКА
    test_user_items = test.groupby('user_id_enc')['item_id_enc'].apply(list).to_dict()
    
    test_als_user_recs = test_als.groupby('user_id_enc')['item_id_enc'].apply(list).to_dict()
    test_ranked_user_recs = test_recommendations.groupby('user_id_enc')['item_id_enc'].apply(list).to_dict()
    
    print("\n" + "=" * 70)
    evaluate_model_at_k(top_popular_list, test_user_items, "BASELINE (Top Popular) - TEST", is_personalized=False)
    evaluate_model_at_k(test_als_user_recs, test_user_items, "ALS - TEST", is_personalized=True)
    evaluate_model_at_k(test_ranked_user_recs, test_user_items, "CATBOOST RANKER - TEST", is_personalized=True)
    
    # Coverage
    users_with_recs = len([u for u in test_user_items if u in test_ranked_user_recs])
    print(f"\nCoverage: {users_with_recs:,} / {len(test_user_items):,} пользователей ({users_with_recs/len(test_user_items)*100:.1f}%)")
    print("=" * 70)
    
    # Сохранение результатов
    results = {
        'baseline_recall@10': np.mean([recall_at_k(top_popular_list, actual, 10) for actual in test_user_items.values()]),
        'als_recall@10': np.mean([recall_at_k(test_als_user_recs.get(u, []), actual, 10) for u, actual in test_user_items.items()]),
        'ranker_recall@10': np.mean([recall_at_k(test_ranked_user_recs.get(u, []), actual, 10) for u, actual in test_user_items.items()]),
    }
    
    pd.DataFrame([results]).to_csv('models/test_metrics.csv', index=False)
    print(f"\nМетрики сохранены в models/test_metrics.csv")


if __name__ == "__main__":
    main()