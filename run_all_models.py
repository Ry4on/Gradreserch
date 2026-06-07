import argparse
import json
import os
import time
import warnings

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
)

warnings.filterwarnings("ignore")


LABEL_MAP = {
    "Normal": 0,
    "Inner Race Fault": 1,
    "Outer Race Fault": 2,
}

CLASS_NAMES = {
    0: "Normal",
    1: "Inner Race Fault",
    2: "Outer Race Fault",
}


def load_dataset_from_folders(root_dir: str):
    """
    Загружает датасет из папок:
        root_dir/
            Normal/
            Inner Race Fault/
            Outer Race Fault/

    Каждый CSV-файл должен содержать один временной ряд.
    """
    X = []
    y = []

    for class_name, label in LABEL_MAP.items():
        folder = os.path.join(root_dir, class_name)

        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Не найдена папка класса: {folder}\n"
                f"Проверь, что структура датасета такая: {list(LABEL_MAP.keys())}"
            )

        files = sorted(os.listdir(folder))

        for file in tqdm(files, desc=f"Loading {class_name}"):
            file_path = os.path.join(folder, file)

            if not file.lower().endswith((".csv", ".txt")):
                continue

            try:
                series = pd.read_csv(file_path, header=None).values.squeeze()
                series = np.asarray(series, dtype=np.float32).reshape(-1)
                X.append(series)
                y.append(label)
            except Exception as e:
                print(f"Ошибка с файлом {file_path}: {e}")

    if len(X) == 0:
        raise ValueError("Не удалось загрузить ни одного временного ряда.")

    return X, np.asarray(y, dtype=np.int64)


def pad_or_trim(series, target_length=2048):
    """
    Приводит все временные ряды к одной длине:
    - если ряд длиннее target_length, обрезает;
    - если короче, дополняет нулями.
    """
    series = np.asarray(series, dtype=np.float32).reshape(-1)

    if len(series) > target_length:
        return series[:target_length]

    if len(series) < target_length:
        return np.pad(series, (0, target_length - len(series)), mode="constant")

    return series


def limit_samples_per_class(X, y, max_samples_per_class=None, random_state=42):
    """
    Ограничивает число объектов каждого класса.
    Полезно для HIVE-COTE, который может обучаться очень долго.
    """
    if max_samples_per_class is None:
        return X, y

    rng = np.random.default_rng(random_state)
    selected_indices = []

    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        if len(label_indices) > max_samples_per_class:
            label_indices = rng.choice(label_indices, size=max_samples_per_class, replace=False)
        selected_indices.extend(label_indices.tolist())

    selected_indices = np.asarray(sorted(selected_indices))
    return X[selected_indices], y[selected_indices]


def compute_metrics(model_name, y_true, y_pred, fit_time, predict_time):
    """
    Считает основные метрики качества классификации.
    """
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "fit_time_sec": fit_time,
        "predict_time_sec": predict_time,
    }


def save_detailed_report(output_dir, model_name, y_true, y_pred):
    """
    Сохраняет classification_report и confusion_matrix в отдельные файлы.
    """
    safe_name = model_name.replace(" ", "_").replace("/", "_")
    report_path = os.path.join(output_dir, f"{safe_name}_classification_report.txt")
    cm_path = os.path.join(output_dir, f"{safe_name}_confusion_matrix.csv")

    labels = sorted(CLASS_NAMES.keys())
    target_names = [CLASS_NAMES[i] for i in labels]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{CLASS_NAMES[i]}" for i in labels],
        columns=[f"pred_{CLASS_NAMES[i]}" for i in labels],
    )
    cm_df.to_csv(cm_path, index=True, encoding="utf-8-sig")


def run_catch22(X_train, X_test, y_train, y_test, output_dir):
    from sktime.classification.feature_based import Catch22Classifier
    from sktime.datatypes._panel._convert import from_2d_array_to_nested

    X_train_df = from_2d_array_to_nested(X_train)
    X_test_df = from_2d_array_to_nested(X_test)

    model = Catch22Classifier()

    start = time.time()
    model.fit(X_train_df, y_train)
    fit_time = time.time() - start

    start = time.time()
    y_pred = model.predict(X_test_df)
    predict_time = time.time() - start

    save_detailed_report(output_dir, "Catch22Classifier", y_test, y_pred)
    return compute_metrics("Catch22Classifier", y_test, y_pred, fit_time, predict_time)


def run_rocket(X_train, X_test, y_train, y_test, output_dir, num_kernels=10000, random_state=42):
    from sktime.classification.kernel_based import RocketClassifier
    from sktime.datatypes._panel._convert import from_2d_array_to_nested

    X_train_df = from_2d_array_to_nested(X_train)
    X_test_df = from_2d_array_to_nested(X_test)

    model = RocketClassifier(
        num_kernels=num_kernels,
        random_state=random_state,
    )

    start = time.time()
    model.fit(X_train_df, y_train)
    fit_time = time.time() - start

    start = time.time()
    y_pred = model.predict(X_test_df)
    predict_time = time.time() - start

    save_detailed_report(output_dir, "ROCKET", y_test, y_pred)
    return compute_metrics("ROCKET", y_test, y_pred, fit_time, predict_time)


def run_hivecote(X_train, X_test, y_train, y_test, output_dir, random_state=42):
    from sktime.classification.hybrid import HIVECOTEV2
    from sktime.datatypes._panel._convert import from_2d_array_to_nested

    X_train_df = from_2d_array_to_nested(X_train)
    X_test_df = from_2d_array_to_nested(X_test)

    model = HIVECOTEV2(random_state=random_state)

    start = time.time()
    model.fit(X_train_df, y_train)
    fit_time = time.time() - start

    start = time.time()
    y_pred = model.predict(X_test_df)
    predict_time = time.time() - start

    save_detailed_report(output_dir, "HIVE-COTE", y_test, y_pred)
    return compute_metrics("HIVE-COTE", y_test, y_pred, fit_time, predict_time)


def run_inceptiontime(
    X,
    y,
    train_idx,
    test_idx,
    output_dir,
    epochs=20,
    lr=1e-3,
    batch_size=64,
    random_state=42,
):
    from tsai.all import (
        InceptionTime,
        TSClassification,
        TSStandardize,
        accuracy,
        get_ts_dls,
        set_seed,
        ts_learner,
    )

    set_seed(random_state, reproducible=True)

    # В твоём исходном файле использовалась форма X[..., None].
    # Оставляем этот формат, чтобы поведение было максимально близко к старому коду.
    X_3d = X[..., None]

    splits = (list(train_idx), list(test_idx))
    tfms = [None, TSClassification()]
    batch_tfms = TSStandardize()

    dls = get_ts_dls(
        X_3d,
        y,
        splits=splits,
        tfms=tfms,
        batch_tfms=batch_tfms,
        bs=batch_size,
    )

    learn = ts_learner(dls, InceptionTime, metrics=accuracy)

    start = time.time()
    learn.fit_one_cycle(epochs, lr)
    fit_time = time.time() - start

    start = time.time()
    preds, targs = learn.get_preds(ds_idx=1)
    predict_time = time.time() - start

    y_pred = preds.argmax(dim=1).cpu().numpy()
    y_true = targs.cpu().numpy()

    save_detailed_report(output_dir, "InceptionTime", y_true, y_pred)
    return compute_metrics("InceptionTime", y_true, y_pred, fit_time, predict_time)


def main():
    parser = argparse.ArgumentParser(
        description="Запуск нескольких моделей классификации временных рядов на одном датасете."
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Путь к папке CSV Files, внутри которой лежат папки Normal, Inner Race Fault, Outer Race Fault.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model_results",
        help="Папка для сохранения CSV-таблицы, отчётов и матриц ошибок.",
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=2048,
        help="Длина временного ряда после pad_or_trim.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Доля тестовой выборки.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Фиксация случайности для воспроизводимости.",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Ограничить число объектов каждого класса. Полезно для быстрых пробных запусков.",
    )

    parser.add_argument("--skip-catch22", action="store_true", help="Не запускать Catch22Classifier.")
    parser.add_argument("--skip-rocket", action="store_true", help="Не запускать ROCKET.")
    parser.add_argument("--skip-hivecote", action="store_true", help="Не запускать HIVE-COTE.")
    parser.add_argument("--skip-inception", action="store_true", help="Не запускать InceptionTime.")

    parser.add_argument(
        "--rocket-kernels",
        type=int,
        default=10000,
        help="Количество случайных ядер для ROCKET.",
    )
    parser.add_argument(
        "--inception-epochs",
        type=int,
        default=20,
        help="Количество эпох обучения InceptionTime.",
    )
    parser.add_argument(
        "--inception-lr",
        type=float,
        default=1e-3,
        help="Learning rate для InceptionTime.",
    )
    parser.add_argument(
        "--inception-batch-size",
        type=int,
        default=64,
        help="Batch size для InceptionTime.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n=== Загрузка данных ===")
    X_raw, y = load_dataset_from_folders(args.data_dir)
    X = np.asarray([pad_or_trim(s, args.target_length) for s in X_raw], dtype=np.float32)

    X, y = limit_samples_per_class(
        X,
        y,
        max_samples_per_class=args.max_samples_per_class,
        random_state=args.random_state,
    )

    print(f"Всего объектов: {len(X)}")
    print(f"Форма X: {X.shape}")
    print("Распределение классов:")
    for label in sorted(np.unique(y)):
        print(f"  {label} ({CLASS_NAMES[label]}): {(y == label).sum()}")

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    results = []

    print("\n=== Запуск моделей ===")

    if not args.skip_catch22:
        print("\n--- Catch22Classifier ---")
        try:
            results.append(run_catch22(X_train, X_test, y_train, y_test, args.output_dir))
        except Exception as e:
            print(f"Ошибка Catch22Classifier: {e}")

    if not args.skip_rocket:
        print("\n--- ROCKET ---")
        try:
            results.append(
                run_rocket(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    args.output_dir,
                    num_kernels=args.rocket_kernels,
                    random_state=args.random_state,
                )
            )
        except Exception as e:
            print(f"Ошибка ROCKET: {e}")

    if not args.skip_hivecote:
        print("\n--- HIVE-COTE ---")
        try:
            results.append(run_hivecote(X_train, X_test, y_train, y_test, args.output_dir, args.random_state))
        except Exception as e:
            print(f"Ошибка HIVE-COTE: {e}")

    if not args.skip_inception:
        print("\n--- InceptionTime ---")
        try:
            results.append(
                run_inceptiontime(
                    X,
                    y,
                    train_idx,
                    test_idx,
                    args.output_dir,
                    epochs=args.inception_epochs,
                    lr=args.inception_lr,
                    batch_size=args.inception_batch_size,
                    random_state=args.random_state,
                )
            )
        except Exception as e:
            print(f"Ошибка InceptionTime: {e}")

    if not results:
        print("\nНе удалось получить результаты ни одной модели.")
        return

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1_weighted", ascending=False)

    csv_path = os.path.join(args.output_dir, "model_comparison_results.csv")
    json_path = os.path.join(args.output_dir, "model_comparison_results.json")

    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("\n=== Итоговая таблица ===")
    print(results_df.to_string(index=False))

    print(f"\nCSV сохранён: {csv_path}")
    print(f"JSON сохранён: {json_path}")
    print(f"Подробные classification_report и confusion_matrix сохранены в папке: {args.output_dir}")


if __name__ == "__main__":
    main()
