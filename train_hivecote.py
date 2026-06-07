import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt

from sktime.datatypes._panel._convert import from_2d_array_to_nested

try:
    from sktime.classification.hybrid import HIVECOTEV2
except ImportError:
    # На некоторых версиях название/расположение класса отличается.
    from sktime.classification.hybrid import HIVECOTE as HIVECOTEV2


LABEL_MAP = {
    "Normal": 0,
    "Inner Race Fault": 1,
    "Outer Race Fault": 2,
}

DISPLAY_LABELS = ["Normal", "Inner Fault", "Outer Fault"]


def load_dataset_from_folders(root_dir: str):
    """
    Загружает датасет из папок:
    root_dir/
        Normal/
        Inner Race Fault/
        Outer Race Fault/

    Каждый файл должен быть CSV-файлом с одним временным рядом.
    """
    X = []
    y = []

    for class_name, label in LABEL_MAP.items():
        folder = os.path.join(root_dir, class_name)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Не найдена папка класса: {folder}")

        files = sorted(os.listdir(folder))
        for file in tqdm(files, desc=class_name):
            file_path = os.path.join(folder, file)
            if os.path.isdir(file_path):
                continue

            try:
                series = pd.read_csv(file_path, header=None).values.squeeze()
                series = np.asarray(series, dtype=np.float32).reshape(-1)
                X.append(series)
                y.append(label)
            except Exception as e:
                print(f"Ошибка с файлом {file_path}: {e}")

    return X, y


def pad_or_trim(series, target_length: int = 2048):
    """Обрезает или дополняет временной ряд нулями до target_length."""
    series = np.asarray(series, dtype=np.float32).reshape(-1)

    if len(series) > target_length:
        return series[:target_length]
    if len(series) < target_length:
        return np.pad(series, (0, target_length - len(series)))
    return series


def limit_samples_per_class(X, y, max_samples_per_class, random_state=42):
    """
    Необязательное ограничение количества объектов каждого класса.
    Полезно для пробного запуска HIVE-COTE, потому что он может обучаться очень долго.
    Если max_samples_per_class <= 0, ограничение не применяется.
    """
    if max_samples_per_class is None or max_samples_per_class <= 0:
        return X, y

    rng = np.random.default_rng(random_state)
    selected_indices = []

    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        if len(cls_indices) > max_samples_per_class:
            cls_indices = rng.choice(cls_indices, size=max_samples_per_class, replace=False)
        selected_indices.extend(cls_indices.tolist())

    selected_indices = np.array(sorted(selected_indices))
    return X[selected_indices], y[selected_indices]


def build_hivecote(random_state=42, n_jobs=-1):
    """
    Создаёт HIVE-COTE с учётом различий между версиями sktime.
    """
    try:
        return HIVECOTEV2(random_state=random_state, n_jobs=n_jobs)
    except TypeError:
        try:
            return HIVECOTEV2(random_state=random_state)
        except TypeError:
            return HIVECOTEV2()


def save_confusion_matrix(y_test, y_pred, save_path: Path):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DISPLAY_LABELS)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix — HIVE-COTE")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_f1_plot(y_test, y_pred, save_path: Path):
    f1_per_class = f1_score(y_test, y_pred, average=None)
    plt.bar(DISPLAY_LABELS, f1_per_class)
    plt.ylim(0.0, 1.05)
    plt.title("F1-score по классам — HIVE-COTE")
    plt.ylabel("F1-score")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Обучение HIVE-COTE на датасете SUBF/CSV Files")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"C:\Users\lasve\Downloads\archive\CSV Files",
        help="Путь к папке CSV Files, внутри которой лежат папки классов",
    )
    parser.add_argument("--target-length", type=int, default=2048, help="Длина временного ряда после pad/trim")
    parser.add_argument("--test-size", type=float, default=0.2, help="Доля тестовой выборки")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Количество потоков CPU; -1 = все доступные")
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=0,
        help="Ограничить количество объектов каждого класса для пробного запуска. 0 = без ограничения",
    )
    parser.add_argument("--save-dir", type=str, default="results_hivecote", help="Папка для сохранения результатов")
    parser.add_argument("--no-plots", action="store_true", help="Не сохранять графики")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Загрузка данных...")
    X_raw, y_raw = load_dataset_from_folders(args.data_dir)

    X = np.array([pad_or_trim(s, args.target_length) for s in X_raw], dtype=np.float32)
    y = np.array(y_raw, dtype=np.int64)

    X, y = limit_samples_per_class(
        X,
        y,
        max_samples_per_class=args.max_samples_per_class,
        random_state=args.random_state,
    )

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Классы и количество: {dict(zip(*np.unique(y, return_counts=True)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    # sktime-классификаторы обычно принимают nested DataFrame для панельных временных рядов.
    X_train_df = from_2d_array_to_nested(X_train)
    X_test_df = from_2d_array_to_nested(X_test)

    model = build_hivecote(random_state=args.random_state, n_jobs=args.n_jobs)

    print("Обучение HIVE-COTE...")
    print("Важно: HIVE-COTE обычно обучается значительно дольше ROCKET/Catch22/InceptionTime.")
    train_start = time.perf_counter()
    model.fit(X_train_df, y_train)
    train_time = time.perf_counter() - train_start

    print("Предсказание HIVE-COTE...")
    predict_start = time.perf_counter()
    y_pred = model.predict(X_test_df)
    predict_time = time.perf_counter() - predict_start

    metrics = {
        "model": "HIVE-COTE",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "train_time_seconds": float(train_time),
        "predict_time_seconds": float(predict_time),
        "target_length": args.target_length,
        "max_samples_per_class": args.max_samples_per_class,
    }

    report = classification_report(y_test, y_pred, target_names=DISPLAY_LABELS, zero_division=0)

    print("\nHIVE-COTE Results:")
    print(json.dumps(metrics, ensure_ascii=False, indent=4))
    print("\nClassification report:")
    print(report)

    with open(save_dir / "hivecote_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    with open(save_dir / "hivecote_classification_report.txt", "w", encoding="utf-8") as f:
        f.write("HIVE-COTE Results\n")
        f.write(json.dumps(metrics, ensure_ascii=False, indent=4))
        f.write("\n\nClassification report:\n")
        f.write(report)

    if not args.no_plots:
        save_confusion_matrix(y_test, y_pred, save_dir / "hivecote_confusion_matrix.png")
        save_f1_plot(y_test, y_pred, save_dir / "hivecote_f1_by_class.png")

    print(f"\nГотово. Результаты сохранены в папку: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
