def prepare_features():
    """Подготовка категорийных фичей"""
    import os
    import pickle
    import pandas as pd
    from dotenv import load_dotenv
    
    load_dotenv()
    
    endpoint_url = os.getenv("S3_ENDPOINT_URL")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    bucket_name = os.getenv("S3_BUCKET_NAME")
    
    storage_options = {
        'key': aws_access_key_id,
        'secret': aws_secret_access_key,
        'client_kwargs': {'endpoint_url': endpoint_url}
    }
    
    print("Подготовка фичей...")
    
    # Загрузка
    with open('/opt/airflow/models/item_encoder.pkl', 'rb') as f:
        item_encoder = pickle.load(f)
    
    train_all = pd.read_pickle('/opt/airflow/models/train_all.pkl')
    df_items = pd.read_csv('/opt/airflow/data/processed/item_properties.csv')
    
    # Категории товаров
    known_items = set(item_encoder.classes_)
    df_items_filtered = df_items[df_items['item_id'].isin(known_items)].copy()
    df_items_filtered['item_id_enc'] = item_encoder.transform(df_items_filtered['item_id'])
    df_items_filtered['parent_id'] = pd.to_numeric(df_items_filtered['parent_id'], errors='coerce').fillna(-1).astype(int)
    
    # Словари
    item_to_cat = df_items_filtered.set_index('item_id_enc')['category_id'].to_dict()
    item_to_parent = df_items_filtered.set_index('item_id_enc')['parent_id'].to_dict()
    
    # Добавляем к train_all
    train_all['category_id'] = train_all['item_id_enc'].map(item_to_cat).fillna(-1).astype(int)
    train_all['parent_id'] = train_all['item_id_enc'].map(item_to_parent).fillna(-1).astype(int)
    
    # Популярность
    item_popularity = train_all.groupby('item_id_enc').size().to_dict()
    user_activity = train_all.groupby('user_id_enc').size().to_dict()
    cat_pop = train_all.groupby('category_id').size().to_dict()
    parent_pop = train_all.groupby('parent_id').size().to_dict()
    
    # Топ категории пользователя
    user_cat_counts = train_all.groupby(['user_id_enc', 'category_id']).size().reset_index(name='count')
    user_top_cat = user_cat_counts.loc[user_cat_counts.groupby('user_id_enc')['count'].idxmax()].set_index('user_id_enc')['category_id'].to_dict()
    
    user_parent_counts = train_all.groupby(['user_id_enc', 'parent_id']).size().reset_index(name='count')
    user_top_parent = user_parent_counts.loc[user_parent_counts.groupby('user_id_enc')['count'].idxmax()].set_index('user_id_enc')['parent_id'].to_dict()
    
    # Сохранение features_dict
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
    
    with open('/opt/airflow/models/features_dict.pkl', 'wb') as f:
        pickle.dump(features_dict, f)
    
    # === СОЗДАНИЕ ITEM/USER FEATURES ДЛЯ СЕРВИСА ===
    
    # Item features
    item_features = pd.DataFrame({
        'item_id_enc': list(item_to_cat.keys()),
        'category_id': list(item_to_cat.values()),
        'parent_id': [item_to_parent.get(i, -1) for i in item_to_cat.keys()],
        'item_popularity': [item_popularity.get(i, 0) for i in item_to_cat.keys()],
        'category_popularity': [cat_pop.get(item_to_cat.get(i, -1), 0) for i in item_to_cat.keys()],
        'parent_popularity': [parent_pop.get(item_to_parent.get(i, -1), 0) for i in item_to_cat.keys()]
    })
    
    # User features
    user_features = pd.DataFrame({
        'user_id_enc': list(user_top_cat.keys()),
        'user_activity': [user_activity.get(u, 0) for u in user_top_cat.keys()],
        'user_top_category': list(user_top_cat.values()),
        'user_top_parent': [user_top_parent.get(u, -1) for u in user_top_cat.keys()]
    })
    
    # Сохранение в S3
    item_features.to_parquet(
        f's3://{bucket_name}/recsys/features/item_features.parquet',
        index=False,
        storage_options=storage_options
    )
    
    user_features.to_parquet(
        f's3://{bucket_name}/recsys/features/user_features.parquet',
        index=False,
        storage_options=storage_options
    )
    
    # Локально тоже
    item_features.to_csv('/opt/airflow/models/item_features.csv', index=False)
    user_features.to_csv('/opt/airflow/models/user_features.csv', index=False)
    
    print(f"Item features: {len(item_features):,}")
    print(f"User features: {len(user_features):,}")
    print("Фичи подготовлены и сохранены")