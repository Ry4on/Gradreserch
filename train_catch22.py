import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sktime.classification.feature_based import Catch22Classifier
from sktime.datatypes._panel._convert import from_2d_array_to_nested

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

    root_dir = r"C:\\Users\\lasve\\Downloads\\archive\\CSV Files"

    
    X_raw, y_raw = load_dataset_from_folders(root_dir)
    X = np.array([pad_or_trim(s).reshape(-1) for s in X_raw])
    y = np.array(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

   
    X_train_df = from_2d_array_to_nested(X_train)
    X_test_df = from_2d_array_to_nested(X_test)


    model = Catch22Classifier()
    model.fit(X_train_df, y_train)
    y_pred = model.predict(X_test_df)
    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Inner Fault", "Outer Fault"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix — Catch22Classifier")
    plt.tight_layout()
    plt.show()

    # F1 по классам
    f1_per_class = f1_score(y_test, y_pred, average=None)
    plt.bar(["Normal", "Inner", "Outer"], f1_per_class, color="skyblue")
    plt.ylim(0.8, 1.05)
    plt.title("F1-score по классам — Catch22Classifier")
    plt.ylabel("F1-score")
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    print("Catch22Classifier Results:")
    print(classification_report(y_test, y_pred))
