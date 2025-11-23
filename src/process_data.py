# process_data.py
import pandas as pd
import os
from config import PATHS

def process_data():
    # Загрузка
    df_item_properties_part1 = pd.read_csv(f"{PATHS['extracted_data']}/item_properties_part1.csv")
    df_item_properties_part2 = pd.read_csv(f"{PATHS['extracted_data']}/item_properties_part2.csv")
    df_events = pd.read_csv(f"{PATHS['extracted_data']}/events.csv")
    df_category_tree = pd.read_csv(f"{PATHS['extracted_data']}/category_tree.csv")

    df_category_tree.rename(columns = {'categoryid' : 'category_id', 'parentid' : 'parent_id'}, inplace = True)
    
    # Объединение item_properties
    df_item_properties = pd.concat([df_item_properties_part1, df_item_properties_part2], axis=0, ignore_index=True)

    # Переименование колонок
    df_category_tree.rename(columns = {'categoryid' : 'category_id', 'parentid' : 'parent_id'}, inplace = True)
    df_events.rename(columns = {'visitorid' : 'visitor_id', 'itemid' : 'item_id', 'transactionid' : 'transaction_id'}, inplace = True)
    df_item_properties.rename(columns = {'itemid' : 'item_id'}, inplace = True)
    
    # Удаление дубликатов events
    df_events = df_events.drop_duplicates()
    
    # Извлечение visitor_id из value
    visitor_ids = set(df_events['visitor_id'].astype(str))
    df_item_properties['interaction_visitor'] = df_item_properties['value'].astype(str).apply(
        lambda x: [v for v in x.split() if v in visitor_ids]
    )
    
    # Добавление категорий

    df_item_properties['property'] = df_item_properties['property'].astype(str).str.strip()
    df_category_tree['category_id'] = df_category_tree['category_id'].astype(str).str.strip()
    df_item_properties = df_item_properties.merge(
        df_category_tree,
        left_on='property',
        right_on='category_id',
        how='left'
    )

    # приводим время в понятный вид
    df_item_properties['timestamp'] = pd.to_datetime(df_item_properties['timestamp'], unit='ms')
    df_events['timestamp'] = pd.to_datetime(df_events['timestamp'], unit='ms')
    
    # Сохранение
    processed_path = PATHS['processed_data']
    os.makedirs(processed_path, exist_ok=True)
    df_item_properties.to_csv(f"{processed_path}/item_properties.csv", index=False)
    df_category_tree.to_csv(f"{processed_path}/category_tree.csv", index=False)
    df_events.to_csv(f"{processed_path}/events.csv", index=False)
    print(f"Данные обработаны и сохранены в: {processed_path}")

if __name__ == "__main__":
    process_data()