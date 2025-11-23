import os

BASE_DIR = '/home/mle-user/mle_projects/e_commerce_recsys/'


PATHS = {
    # Сырые данные
    'raw_data': os.path.join(BASE_DIR, 'data', 'archive.zip'),
    
    # Разархивированные данные (корневая папка)
    'extracted_data': os.path.join(BASE_DIR, 'data', 'archive'),
       
    # CSV файлы
    'category_tree_csv': os.path.join(BASE_DIR, 'category_tree.csv'),
    'events_csv': os.path.join(BASE_DIR, 'events.csv'),
    'item_properties_part1_csv': os.path.join(BASE_DIR, 'item_properties_part1.csv'),
    'item_properties_part2_csv': os.path.join(BASE_DIR, 'item_properties_part1.csv'),
    
    # Обработанные данные (корневая папка)
    'processed_data': os.path.join(BASE_DIR, 'data', 'processed'),


    # Модели
    'models_dir': os.path.join(BASE_DIR, 'models'),
    'best_model': os.path.join(BASE_DIR, 'models', 'best_calorie_model.pth'),
    'checkpoint': os.path.join(BASE_DIR, 'models', 'checkpoint.pth'),
    
    # Логи
    'logs_dir': os.path.join(BASE_DIR, 'logs'),
    'training_log': os.path.join(BASE_DIR, 'logs', 'training_history.csv'),
    'training_viz': os.path.join(BASE_DIR, 'logs', 'training_visualization.png'),

}