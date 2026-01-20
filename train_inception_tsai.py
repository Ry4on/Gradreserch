
from tsai.all import *
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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
        files = sorted(os.listdir(folder))
        for file in tqdm(files, desc=class_name):
            file_path = os.path.join(folder, file)
            try:
                series = pd.read_csv(file_path, header=None).values.squeeze()
                X.append(series)
                y.append(label_map[class_name])
            except Exception as e:
                print(f"Ошибка с файлом {file_path}: {e}")
    return X, y

def pad_or_trim(series, target_length=2048):
    if len(series) > target_length:
        return series[:target_length]
    elif len(series) < target_length:
        return np.pad(series, (0, target_length - len(series)))
    return series

if __name__ == "__main__":
    root_dir = "C:\\Users\\lasve\\Downloads\\archive\\CSV Files"
    X_raw, y_raw = load_dataset_from_folders(root_dir)
    X = np.array([pad_or_trim(s).reshape(-1) for s in X_raw]) 
    y = np.array(y_raw)

    # tsai требует 3D: (samples, seq_len, channels)
    X = X[..., None]

    splits = get_splits(y, valid_size=0.2, stratify=True)
    tfms = [None, TSClassification()]
    batch_tfms = TSStandardize()

    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)

    learn = ts_learner(dls, InceptionTime, metrics=accuracy)
    learn.fit_one_cycle(20, 1e-3)

    learn.show_results()
