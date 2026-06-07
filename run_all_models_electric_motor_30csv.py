import argparse
import json
import os
import re
import time
import warnings
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
)

warnings.filterwarnings("ignore")


# ============================================================
# Скрипт для Electric Motor Vibrations Dataset,
# где датасет лежит НЕ в 30 папках, а в виде 30 CSV-файлов.
#
# Поддерживаемые режимы разметки:
# 1) scenario:
#    каждый CSV-файл = отдельный класс, всего до 30 классов.
#
# 2) fault_type:
#    укрупнённые классы:
#    normal, mechanical_fault, electrical_fault, combined_fault.
#
# 3) binary_fault:
#    два класса:
#    normal, fault.
# ============================================================


def clean_file_stem(file_name: str) -> str:
    """
    Убирает расширение и номер в начале имени файла.

    Пример:
    '01 - m1_half_shaft_speed_no_mechanical_load.csv'
    -> 'm1_half_shaft_speed_no_mechanical_load'
    """
    stem = Path(file_name).stem
    stem = re.sub(r"^\s*\d+\s*-\s*", "", stem)
    return stem.strip()


def get_label_from_file(file_name: str, label_mode: str) -> str:
    """
    Формирует метку класса по имени CSV-файла.
    """
    name = clean_file_stem(file_name).lower()

    if label_mode == "scenario":
        return clean_file_stem(file_name)

    has_mech = (
        "mechanically" in name
        or "mechanicaly" in name
        or "imbalanced" in name
        or "umbalanced" in name
    )
    has_elec = (
        "electrically" in name
        or "ohm_fault" in name
        or "electrical" in name
    )

    if label_mode == "fault_type":
        if has_mech and has_elec:
            return "combined_fault"
        if has_mech:
            return "mechanical_fault"
        if has_elec:
            return "electrical_fault"
        return "normal"

    if label_mode == "binary_fault":
        if has_mech or has_elec:
            return "fault"
        return "normal"

    raise ValueError(f"Неизвестный label_mode: {label_mode}")


def read_numeric_signal_from_csv(file_path: Path, signal_column=None):
    """
    Читает CSV-файл и возвращает одномерный числовой сигнал.

    По умолчанию:
    - берётся первый числовой столбец.

    Если signal_column указан:
    - int или строка с числом: номер столбца, начиная с 0;
    - str: имя столбца.
    """
    read_attempts = [
        {"sep": None, "engine": "python"},
        {"sep": None, "engine": "python", "header": None},
        {"sep": ",", "engine": "python"},
        {"sep": ";", "engine": "python"},
        {"sep": "\t", "engine": "python"},
    ]

    last_error = None

    for kwargs in read_attempts:
        try:
            df = pd.read_csv(file_path, **kwargs)
            numeric_df = df.apply(pd.to_numeric, errors="coerce")
            numeric_df = numeric_df.dropna(axis=1, how="all")

            if numeric_df.shape[1] == 0:
                continue

            if signal_column is None:
                series = numeric_df.iloc[:, 0].dropna().values
            else:
                col = signal_column

                if isinstance(col, str) and col.isdigit():
                    col = int(col)

                if isinstance(col, int):
                    series = numeric_df.iloc[:, col].dropna().values
                else:
                    if col not in numeric_df.columns:
                        raise ValueError(
                            f"Столбец {col} не найден. "
                            f"Доступные столбцы: {list(numeric_df.columns)}"
                        )
                    series = numeric_df[col].dropna().values

            series = np.asarray(series, dtype=np.float32).reshape(-1)

            if len(series) == 0:
                continue

            return series

        except Exception as e:
            last_error = e

    raise ValueError(f"Не удалось прочитать числовой сигнал из {file_path}. Последняя ошибка: {last_error}")


def make_windows(series, window_length=2048, step=2048, min_length_ratio=0.8):
    """
    Разбивает длинный сигнал на окна.

    window_length — длина одного окна.
    step — шаг окна.
           step = window_length означает без перекрытия.
           step = window_length // 2 означает перекрытие 50%.

    Если сигнал короче window_length, он дополняется нулями.
    """
    series = np.asarray(series, dtype=np.float32).reshape(-1)

    if len(series) == 0:
        return []

    if len(series) < window_length:
        return [np.pad(series, (0, window_length - len(series)), mode="constant")]

    windows = []

    for start in range(0, len(series) - window_length + 1, step):
        windows.append(series[start:start + window_length])

    # Хвост файла.
    last_full_start = ((len(series) - window_length) // step) * step
    tail_start = last_full_start + step

    if tail_start < len(series):
        tail = series[tail_start:]
        if len(tail) >= int(window_length * min_length_ratio):
            windows.append(np.pad(tail, (0, window_length - len(tail)), mode="constant"))

    return windows


def load_dataset_from_30_csv_files(
    data_dir,
    label_mode="fault_type",
    window_length=2048,
    step=2048,
    signal_column=None,
    max_files=None,
    max_windows_per_file=None,
):
    """
    Загружает датасет из одной папки, где лежат 30 CSV-файлов.

    Пример:
    data_dir/
        01 - m1_half_shaft_speed_no_mechanical_load.csv
        02 - m1_load_0.5Nm_half_speed.csv
        ...
        30 - ...
    """
    root = Path(data_dir)

    if not root.exists():
        raise FileNotFoundError(f"Папка не найдена: {root}")

    files = sorted([
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in [".csv", ".txt"]
    ])

    if max_files is not None:
        files = files[:max_files]

    if len(files) == 0:
        raise ValueError(
            f"В папке {root} не найдено CSV/TXT файлов. "
            f"Проверь, что ты указываешь именно папку с 30 CSV-файлами."
        )

    X = []
    y_str = []
    meta = []

    for file_path in tqdm(files, desc="Loading CSV files"):
        try:
            label = get_label_from_file(file_path.name, label_mode)
            series = read_numeric_signal_from_csv(file_path, signal_column=signal_column)
            windows = make_windows(series, window_length=window_length, step=step)

            if max_windows_per_file is not None:
                windows = windows[:max_windows_per_file]

            for w_idx, window in enumerate(windows):
                X.append(window)
                y_str.append(label)
                meta.append({
                    "source_file": file_path.name,
                    "window_index": w_idx,
                    "label": label,
                    "series_length": len(series),
                    "window_length": window_length,
                    "step": step,
                })

        except Exception as e:
            print(f"Ошибка файла {file_path}: {e}")

    if len(X) == 0:
        raise ValueError("Не удалось сформировать ни одного окна данных.")

    label_names = sorted(set(y_str))
    label_map = {name: idx for idx, name in enumerate(label_names)}
    y = np.asarray([label_map[label] for label in y_str], dtype=np.int64)
    X = np.asarray(X, dtype=np.float32)

    meta_df = pd.DataFrame(meta)
    label_df = pd.DataFrame({
        "label_id": list(label_map.values()),
        "label_name": list(label_map.keys()),
    }).sort_values("label_id")

    return X, y, label_map, meta_df, label_df


def limit_samples_per_class(X, y, max_samples_per_class=None, random_state=42):
    """
    Ограничивает число окон каждого класса.
    Полезно для быстрого теста и для HIVE-COTE.
    """
    if max_samples_per_class is None:
        return X, y

    rng = np.random.default_rng(random_state)
    selected = []

    for label in np.unique(y):
        idx = np.where(y == label)[0]

        if len(idx) > max_samples_per_class:
            idx = rng.choice(idx, size=max_samples_per_class, replace=False)

        selected.extend(idx.tolist())

    selected = np.asarray(sorted(selected))
    return X[selected], y[selected]


def compute_metrics(model_name, y_true, y_pred, fit_time, predict_time):
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "fit_time_sec": fit_time,
        "predict_time_sec": predict_time,
    }


def save_detailed_report(output_dir, model_name, y_true, y_pred, label_map):
    safe_name = model_name.replace(" ", "_").replace("/", "_").replace("-", "_")

    report_path = os.path.join(output_dir, f"{safe_name}_classification_report.txt")
    cm_path = os.path.join(output_dir, f"{safe_name}_confusion_matrix.csv")

    id_to_label = {idx: name for name, idx in label_map.items()}
    labels = sorted(id_to_label.keys())
    target_names = [id_to_label[i] for i in labels]

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
        index=[f"true_{id_to_label[i]}" for i in labels],
        columns=[f"pred_{id_to_label[i]}" for i in labels],
    )
    cm_df.to_csv(cm_path, index=True, encoding="utf-8-sig")


def run_catch22(X_train, X_test, y_train, y_test, output_dir, label_map):
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

    save_detailed_report(output_dir, "Catch22Classifier", y_test, y_pred, label_map)
    return compute_metrics("Catch22Classifier", y_test, y_pred, fit_time, predict_time)


def run_rocket(X_train, X_test, y_train, y_test, output_dir, label_map, num_kernels=10000, random_state=42):
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

    save_detailed_report(output_dir, "ROCKET", y_test, y_pred, label_map)
    return compute_metrics("ROCKET", y_test, y_pred, fit_time, predict_time)


def run_hivecote(X_train, X_test, y_train, y_test, output_dir, label_map, random_state=42):
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

    save_detailed_report(output_dir, "HIVE-COTE", y_test, y_pred, label_map)
    return compute_metrics("HIVE-COTE", y_test, y_pred, fit_time, predict_time)


def run_inceptiontime(
    X,
    y,
    train_idx,
    test_idx,
    output_dir,
    label_map,
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

    # tsai в твоём исходном коде использовался в формате:
    # X = X[..., None]
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

    save_detailed_report(output_dir, "InceptionTime", y_true, y_pred, label_map)
    return compute_metrics("InceptionTime", y_true, y_pred, fit_time, predict_time)


def main():
    parser = argparse.ArgumentParser(
        description="Запуск Catch22, ROCKET, HIVE-COTE и InceptionTime на датасете из 30 CSV-файлов."
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Папка, в которой лежат 30 CSV-файлов Electric Motor Vibrations Dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="electric_motor_30csv_results",
        help="Папка для сохранения результатов.",
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        default="fault_type",
        choices=["scenario", "fault_type", "binary_fault"],
        help=(
            "scenario = каждый CSV отдельный класс; "
            "fault_type = normal/mechanical/electrical/combined; "
            "binary_fault = normal/fault."
        ),
    )
    parser.add_argument("--window-length", type=int, default=2048)
    parser.add_argument("--step", type=int, default=2048)
    parser.add_argument(
        "--signal-column",
        type=str,
        default=None,
        help="Номер или имя столбца с сигналом. По умолчанию первый числовой столбец.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-windows-per-file", type=int, default=None)
    parser.add_argument("--max-samples-per-class", type=int, default=None)

    parser.add_argument("--skip-catch22", action="store_true")
    parser.add_argument("--skip-rocket", action="store_true")
    parser.add_argument("--skip-hivecote", action="store_true")
    parser.add_argument("--skip-inception", action="store_true")

    parser.add_argument("--rocket-kernels", type=int, default=10000)
    parser.add_argument("--inception-epochs", type=int, default=20)
    parser.add_argument("--inception-lr", type=float, default=1e-3)
    parser.add_argument("--inception-batch-size", type=int, default=64)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n=== Загрузка датасета из 30 CSV-файлов ===")
    print(f"data-dir: {args.data_dir}")
    print(f"label-mode: {args.label_mode}")
    print(f"window-length: {args.window_length}")
    print(f"step: {args.step}")

    X, y, label_map, meta_df, label_df = load_dataset_from_30_csv_files(
        data_dir=args.data_dir,
        label_mode=args.label_mode,
        window_length=args.window_length,
        step=args.step,
        signal_column=args.signal_column,
        max_files=args.max_files,
        max_windows_per_file=args.max_windows_per_file,
    )

    X, y = limit_samples_per_class(
        X,
        y,
        max_samples_per_class=args.max_samples_per_class,
        random_state=args.random_state,
    )

    label_df.to_csv(os.path.join(args.output_dir, "label_map.csv"), index=False, encoding="utf-8-sig")
    meta_df.to_csv(os.path.join(args.output_dir, "loaded_windows_metadata.csv"), index=False, encoding="utf-8-sig")

    print(f"\nВсего окон: {len(X)}")
    print(f"Форма X: {X.shape}")

    print("\nКарта классов:")
    print(label_df.to_string(index=False))

    print("\nРаспределение классов:")
    id_to_label = {idx: label for label, idx in label_map.items()}
    for label_id in sorted(np.unique(y)):
        print(f"  {label_id} ({id_to_label[label_id]}): {(y == label_id).sum()}")

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
            results.append(run_catch22(X_train, X_test, y_train, y_test, args.output_dir, label_map))
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
                    label_map,
                    num_kernels=args.rocket_kernels,
                    random_state=args.random_state,
                )
            )
        except Exception as e:
            print(f"Ошибка ROCKET: {e}")

    if not args.skip_hivecote:
        print("\n--- HIVE-COTE ---")
        try:
            results.append(
                run_hivecote(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    args.output_dir,
                    label_map,
                    random_state=args.random_state,
                )
            )
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
                    label_map,
                    epochs=args.inception_epochs,
                    lr=args.inception_lr,
                    batch_size=args.inception_batch_size,
                    random_state=args.random_state,
                )
            )
        except Exception as e:
            print(f"Ошибка InceptionTime: {e}")

    if len(results) == 0:
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
    print(f"Карта классов сохранена: {os.path.join(args.output_dir, 'label_map.csv')}")
    print(f"Метаданные окон сохранены: {os.path.join(args.output_dir, 'loaded_windows_metadata.csv')}")


if __name__ == "__main__":
    main()
