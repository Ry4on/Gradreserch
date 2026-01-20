import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_dataset_from_folders(root_dir):
    X = []
    y = []
    label_map = {
        'Normal': 0,
        'Inner Race Fault': 1,
        'Outer Race Fault': 2
    }

    for class_name in label_map:
        folder = os.path.join(root_dir, class_name)
        files = sorted(os.listdir(folder))  # сортируем для стабильности
        for file in tqdm(files, desc=class_name):
            file_path = os.path.join(folder, file)
            try:
                series = pd.read_csv(file_path, header=None).values.squeeze()
                X.append(series)
                y.append(label_map[class_name])
            except Exception as e:
                print(f"Ошибка с файлом {file_path}: {e}")
    return X, y
