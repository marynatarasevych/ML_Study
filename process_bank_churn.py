 """
Модуль предобработки табличных данных для задач классификации/регрессии.

Пайплайн:
1. Разбиение на train/val с сохранением баланса классов (stratify)
2. Отделение признаков (inputs) от целевой переменной (target)
3. Добавление категориального признака Age_Group (если есть колонка Age)
4. Импутация пропусков в категориальных и числовых признаках
5. One-Hot кодирование категориальных признаков
6. (опционально) Масштабирование числовых признаков MinMaxScaler'ом
"""

from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def split_train_val(
    raw_df: pd.DataFrame,
    target: str,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разбивает датафрейм на обучающую и валидационную выборки со стратификацией по target.

    Args:
        raw_df: исходный датафрейм.
        target: название целевой колонки, по которой делается stratify.
        test_size: доля валидационной выборки.
        random_state: seed для воспроизводимости.

    Returns:
        Кортеж (train_df, val_df).
    """
    train_df, val_df = train_test_split(
        raw_df,
        test_size=test_size,
        random_state=random_state,
        stratify=raw_df[target],
    )
    return train_df, val_df


def get_input_cols(
    train_df: pd.DataFrame,
    target: str,
    cols_to_drop_name: List[str],
) -> List[str]:
    """
    Формирует список колонок-признаков, исключая целевую переменную и служебные колонки.

    Args:
        train_df: обучающий датафрейм (используется только для списка колонок).
        target: название целевой колонки.
        cols_to_drop_name: список дополнительных колонок, которые не нужны как признаки.

    Returns:
        Список названий колонок-признаков.
    """
    cols_to_drop = cols_to_drop_name + [target]
    return train_df.drop(columns=cols_to_drop).columns.tolist()


def split_inputs_targets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    input_cols: List[str],
    target: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Отделяет признаки от целевой переменной для train и val выборок.

    Args:
        train_df: обучающий датафрейм.
        val_df: валидационный датафрейм.
        input_cols: список колонок-признаков.
        target: название целевой колонки.

    Returns:
        Кортеж (train_inputs, train_targets, val_inputs, val_targets).
    """
    train_inputs = train_df[input_cols].copy()
    train_targets = train_df[target].copy()
    val_inputs = val_df[input_cols].copy()
    val_targets = val_df[target].copy()
    return train_inputs, train_targets, val_inputs, val_targets


def add_age_group(
    df: pd.DataFrame,
    age_col: str = "Age",
    bins: Optional[List[int]] = None,
    labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Добавляет категориальный признак Age_Group на основе бинирования возраста.
    Если колонки age_col в датафрейме нет — возвращает df без изменений.

    Args:
        df: датафрейм с признаками.
        age_col: название колонки с возрастом.
        bins: границы бинов для pd.cut.
        labels: подписи бинов.

    Returns:
        Датафрейм с добавленной колонкой 'Age_Group' (если age_col присутствует).
    """
    if age_col not in df.columns:
        return df

    bins = bins or [18, 25, 35, 45, 55, 65, 100]
    labels = labels or ["18-25", "25-35", "35-45", "45-55", "55-65", "65+"]

    df = df.copy()
    df["Age_Group"] = pd.cut(df[age_col], bins=bins, labels=labels)
    return df


def get_categorical_cols(df: pd.DataFrame) -> List[str]:
    """
    Возвращает список категориальных (object-type) колонок датафрейма.

    Args:
        df: датафрейм с признаками.

    Returns:
        Список названий категориальных колонок.
    """
    return df.select_dtypes("object").columns.tolist()


def get_numeric_cols(df: pd.DataFrame) -> List[str]:
    """
    Возвращает список числовых колонок датафрейма.

    Args:
        df: датафрейм с признаками.

    Returns:
        Список названий числовых колонок.
    """
    return df.select_dtypes(include=np.number).columns.tolist()


def impute_categorical(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, SimpleImputer]:
    """
    Заполняет пропуски в категориальных колонках наиболее частым значением.
    Импутер обучается только на train и применяется к train и val.

    Args:
        train_inputs: обучающие признаки.
        val_inputs: валидационные признаки.
        categorical_cols: список категориальных колонок.

    Returns:
        Кортеж (train_inputs, val_inputs, imputer_cat).
    """
    imputer_cat = SimpleImputer(strategy="most_frequent")
    imputer_cat.fit(train_inputs[categorical_cols])

    train_inputs = train_inputs.copy()
    val_inputs = val_inputs.copy()
    train_inputs[categorical_cols] = imputer_cat.transform(train_inputs[categorical_cols])
    val_inputs[categorical_cols] = imputer_cat.transform(val_inputs[categorical_cols])

    return train_inputs, val_inputs, imputer_cat


def impute_numeric(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    numeric_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, SimpleImputer]:
    """
    Заполняет пропуски в числовых колонках медианой.
    Импутер обучается только на train и применяется к train и val.

    Args:
        train_inputs: обучающие признаки.
        val_inputs: валидационные признаки.
        numeric_cols: список числовых колонок.

    Returns:
        Кортеж (train_inputs, val_inputs, imputer_num).
    """
    imputer_num = SimpleImputer(strategy="median")
    imputer_num.fit(train_inputs[numeric_cols])

    train_inputs = train_inputs.copy()
    val_inputs = val_inputs.copy()
    train_inputs[numeric_cols] = imputer_num.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = imputer_num.transform(val_inputs[numeric_cols])

    return train_inputs, val_inputs, imputer_num


def encode_categorical(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str]]:
    """
    Кодирует категориальные колонки через One-Hot Encoding.
    Энкодер обучается только на train и применяется к train и val,
    закодированные колонки добавляются к исходным датафреймам.

    Args:
        train_inputs: обучающие признаки.
        val_inputs: валидационные признаки.
        categorical_cols: список категориальных колонок для кодирования.

    Returns:
        Кортеж (train_inputs, val_inputs, encoder, encoded_cols).
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='if_binary')
    encoder.fit(train_inputs[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    train_inputs = train_inputs.copy()
    val_inputs = val_inputs.copy()
    train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
    val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])

    return train_inputs, val_inputs, encoder, encoded_cols


def scale_numeric(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    numeric_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Масштабирует числовые колонки в диапазон [0, 1] с помощью MinMaxScaler.
    Скейлер обучается только на train и применяется к train и val.

    Args:
        train_inputs: обучающие признаки.
        val_inputs: валидационные признаки.
        numeric_cols: список числовых колонок для масштабирования.

    Returns:
        Кортеж (train_inputs, val_inputs, scaler).
    """
    scaler = MinMaxScaler()
    scaler.fit(train_inputs[numeric_cols])

    train_inputs = train_inputs.copy()
    val_inputs = val_inputs.copy()
    train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])

    return train_inputs, val_inputs, scaler


def preprocess_data(
    raw_df: pd.DataFrame,
    target: str,
    cols_to_drop_name: List[str],
    scaler_numeric: bool = True,
) -> dict:
    """
    Полный пайплайн предобработки данных: split -> add Age_Group -> impute ->
    encode -> (опционально) scale.

    Args:
        raw_df: исходный датафрейм с признаками и целевой переменной.
        target: название целевой колонки.
        cols_to_drop_name: список колонок, которые нужно исключить из признаков
            (помимо target).
        scaler_numeric: если True — числовые признаки масштабируются MinMaxScaler'ом.

    Returns:
        Словарь со следующими ключами:
            'X_train': обработанные обучающие признаки (pd.DataFrame),
            'train_targets': обучающие таргеты (pd.Series),
            'X_val': обработанные валидационные признаки (pd.DataFrame),
            'val_targets': валидационные таргеты (pd.Series),
            'input_cols': исходный список колонок-признаков (List[str]),
            'imputer_cat': обученный SimpleImputer для категориальных признаков,
            'imputer_num': обученный SimpleImputer для числовых признаков,
            'encoder': обученный OneHotEncoder (или None),
            'scaler': обученный MinMaxScaler (или None, если scaler_numeric=False),
    """
    # 1. Разбиение на train / val
    train_df, val_df = split_train_val(raw_df, target)

    # 2. Отделение признаков от таргета
    input_cols = get_input_cols(train_df, target, cols_to_drop_name)
    train_inputs, train_targets, val_inputs, val_targets = split_inputs_targets(
        train_df, val_df, input_cols, target
    )

    # 3. Добавление признака Age_Group (если есть колонка Age)
    train_inputs = add_age_group(train_inputs)
    val_inputs = add_age_group(val_inputs)

    # 4. Список категориальных колонок
    categorical_cols = get_categorical_cols(train_inputs)

    encoder = None
    encoded_cols: List[str] = []
    imputer_cat = None

    if categorical_cols:
        # 5. Импутация категориальных пропусков
        train_inputs, val_inputs, imputer_cat = impute_categorical(
            train_inputs, val_inputs, categorical_cols
        )
        # 6. One-Hot кодирование
        train_inputs, val_inputs, encoder, encoded_cols = encode_categorical(
            train_inputs, val_inputs, categorical_cols
        )

    # 7. Список числовых колонок
    numeric_cols = get_numeric_cols(train_inputs)

    # 8. Импутация числовых пропусков
    train_inputs, val_inputs, imputer_num = impute_numeric(
        train_inputs, val_inputs, numeric_cols
    )

    # 9. Масштабирование числовых признаков (опционально)
    scaler = None
    if scaler_numeric:
        train_inputs, val_inputs, scaler = scale_numeric(
            train_inputs, val_inputs, numeric_cols
        )

    # 10. Финальный набор признаков: числовые + закодированные категориальные
    final_cols = numeric_cols + encoded_cols
    X_train = train_inputs[final_cols]
    X_val = val_inputs[final_cols]

    return {
        "X_train": X_train,
        "train_targets": train_targets,
        "X_val": X_val,
        "val_targets": val_targets,
        "input_cols": input_cols,
        "imputer_cat": imputer_cat,
        "imputer_num": imputer_num,
        "encoder": encoder,
        "scaler": scaler,
    }

def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    scaler: Optional[MinMaxScaler] = None,
    encoder: Optional[OneHotEncoder] = None,
    imputer_cat: Optional[SimpleImputer] = None,
    imputer_num: Optional[SimpleImputer] = None,
) -> pd.DataFrame:
    """
    Применяет уже обученные на train трансформеры (imputer/encoder/scaler)
    к новым данным (например, test.csv) для получения предсказаний.
 
    В отличие от preprocess_data, эта функция НЕ вызывает train_test_split
    и НЕ обучает (.fit) никакие трансформеры — только .transform() с уже
    готовыми объектами, чтобы избежать data leakage и рассинхронизации
    набора признаков между train и новыми данными.
 
    Args:
        new_df: новый "сырой" датафрейм (без таргета), например test.csv.
        input_cols: список колонок-признаков, использованных при обучении
            (значение 'input_cols' из preprocess_data).
        scaler: обученный на train MinMaxScaler. Если None — масштабирование
            не применяется (соответствует scaler_numeric=False при обучении).
        encoder: обученный на train OneHotEncoder. Если None — категориальные
            признаки не кодируются (используется, если их не было при обучении).
        imputer_cat: обученный на train SimpleImputer для категориальных
            признаков. Если None — импутация категориальных пропусков
            пропускается.
        imputer_num: обученный на train SimpleImputer для числовых
            признаков. Если None — импутация числовых пропусков пропускается.
 
    Returns:
        X_new: обработанный датафрейм с признаками, готовый для подачи
            в обученную модель (тот же набор колонок, что и X_train/X_val).
    """
    # 1. Отбираем только те колонки, что использовались при обучении
    new_inputs = new_df[input_cols].copy()
 
    # 2. Добавляем Age_Group так же, как при обучении
    new_inputs = add_age_group(new_inputs)
 
    # 3. Категориальные признаки: импутация + one-hot тем же encoder'ом
    categorical_cols = get_categorical_cols(new_inputs)
    encoded_cols: List[str] = []
 
    if categorical_cols and imputer_cat is not None:
        new_inputs[categorical_cols] = imputer_cat.transform(new_inputs[categorical_cols])
 
    if categorical_cols and encoder is not None:
        encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
        new_inputs[encoded_cols] = encoder.transform(new_inputs[categorical_cols])
 
    # 4. Числовые признаки: импутация тем же imputer'ом
    numeric_cols = get_numeric_cols(new_inputs)
 
    if imputer_num is not None:
        new_inputs[numeric_cols] = imputer_num.transform(new_inputs[numeric_cols])
 
    # 5. Масштабирование тем же scaler'ом (если использовалось при обучении)
    if scaler is not None:
        new_inputs[numeric_cols] = scaler.transform(new_inputs[numeric_cols])
 
    # 6. Тот же финальный набор колонок, что и в X_train/X_val
    final_cols = numeric_cols + encoded_cols
    X_new = new_inputs[final_cols]
 
    return X_new