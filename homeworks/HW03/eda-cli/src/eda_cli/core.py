from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import pandas as pd
from pandas.api import types as ptypes


@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    missing: int
    missing_share: float
    unique: int
    example_values: List[Any]
    is_numeric: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    columns: List[ColumnSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
        }


def summarize_dataset(
    df: pd.DataFrame,
    example_values_per_column: int = 3,
) -> DatasetSummary:
    """
    Полный обзор датасета по колонкам:
    - количество строк/столбцов;
    - типы;
    - пропуски;
    - количество уникальных;
    - несколько примерных значений;
    - базовые числовые статистики (для numeric).
    """
    n_rows, n_cols = df.shape
    columns: List[ColumnSummary] = []

    for name in df.columns:
        s = df[name]
        dtype_str = str(s.dtype)

        non_null = int(s.notna().sum())
        missing = n_rows - non_null
        missing_share = float(missing / n_rows) if n_rows > 0 else 0.0
        unique = int(s.nunique(dropna=True))

        # Примерные значения выводим как строки.
        examples = (
            s.dropna().astype(str).unique()[:example_values_per_column].tolist()
            if non_null > 0
            else []
        )

        is_numeric = bool(ptypes.is_numeric_dtype(s))
        min_val: Optional[float] = None
        max_val: Optional[float] = None
        mean_val: Optional[float] = None
        std_val: Optional[float] = None

        if is_numeric and non_null > 0:
            min_val = float(s.min())
            max_val = float(s.max())
            mean_val = float(s.mean())
            std_val = float(s.std())

        columns.append(
            ColumnSummary(
                name=name,
                dtype=dtype_str,
                non_null=non_null,
                missing=missing,
                missing_share=missing_share,
                unique=unique,
                example_values=examples,
                is_numeric=is_numeric,
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
            )
        )

    return DatasetSummary(n_rows=n_rows, n_cols=n_cols, columns=columns)


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Таблица пропусков по колонкам: count/share.
    """
    if df.empty:
        return pd.DataFrame(columns=["missing_count", "missing_share"])

    total = df.isna().sum()
    share = total / len(df)
    result = pd.DataFrame(
        {
            "missing_count": total,
            "missing_share": share,
        }
    ).sort_values("missing_share", ascending=False)
    return result


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Корреляция Пирсона для числовых колонок.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)


def top_categories(
    df: pd.DataFrame,
    max_columns: int = 5,
    top_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Для категориальных/строковых колонок считает top-k значений.
    Возвращает словарь: колонка -> DataFrame со столбцами value/count/share.
    """
    result: Dict[str, pd.DataFrame] = {}
    candidate_cols: List[str] = []

    for name in df.columns:
        s = df[name]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            candidate_cols.append(name)

    for name in candidate_cols[:max_columns]:
        s = df[name]
        vc = s.value_counts(dropna=True).head(top_k)
        if vc.empty:
            continue
        share = vc / vc.sum()
        table = pd.DataFrame(
            {
                "value": vc.index.astype(str),
                "count": vc.values,
                "share": share.values,
            }
        )
        result[name] = table

    return result


def compute_quality_flags(
    summary: DatasetSummary,
    missing_df: pd.DataFrame,
    min_missing_share: float = 0.5,
    high_cardinality_threshold: int = 100,
    zero_share_threshold: float = 0.8,
) -> Dict[str, Any]:
    """
    Простейшие эвристики «качества» данных:
    - слишком много пропусков;
    - подозрительно мало строк;
    - постоянные колонки (все значения одинаковые);
    - высокая кардинальность категориальных признаков;
    - много нулевых значений в числовых колонках.
    """
    flags: Dict[str, Any] = {}
    flags["too_few_rows"] = summary.n_rows < 100
    flags["too_many_columns"] = summary.n_cols > 100

    max_missing_share = (
        float(missing_df["missing_share"].max()) if not missing_df.empty else 0.0
    )
    flags["max_missing_share"] = max_missing_share
    flags["too_many_missing"] = max_missing_share > min_missing_share
    flags["min_missing_share_threshold"] = min_missing_share

    # Новые эвристики
    constant_columns = []
    high_cardinality_categoricals = []
    high_zero_share_columns = []

    for col in summary.columns:
        # 1. Проверка на постоянные колонки.
        if col.unique == 1 and col.non_null > 0:
            constant_columns.append(col.name)

        # 2. Проверка на высокую кардинальность категориальных признаков
        if not col.is_numeric and col.unique > high_cardinality_threshold:
            high_cardinality_categoricals.append(
                {
                    "column": col.name,
                    "unique_count": col.unique,
                    "threshold": high_cardinality_threshold,
                }
            )

        # 3. Проверка на много нулевых значений в числовых колонках.
        if col.is_numeric and col.non_null > 0:
            pass

    flags["has_constant_columns"] = len(constant_columns) > 0
    flags["constant_columns"] = constant_columns
    flags["has_high_cardinality_categoricals"] = len(high_cardinality_categoricals) > 0
    flags["high_cardinality_categoricals"] = high_cardinality_categoricals
    flags["high_cardinality_threshold"] = high_cardinality_threshold
    flags["has_high_zero_share_columns"] = len(high_zero_share_columns) > 0
    flags["high_zero_share_columns"] = high_zero_share_columns
    flags["zero_share_threshold"] = zero_share_threshold

    # Простейший «скор» качества с учетом новых факторов.
    score = 1.0
    score -= max_missing_share

    if summary.n_rows < 100:
        score -= 0.2
    if summary.n_cols > 100:
        score -= 0.1
    if flags["has_constant_columns"]:
        score -= 0.15
    if flags["has_high_cardinality_categoricals"]:
        score -= 0.1 * min(1, len(high_cardinality_categoricals) / 5)
    if flags["has_high_zero_share_columns"]:
        score -= 0.1 * min(1, len(high_zero_share_columns) / 5)

    score = max(0.0, min(1.0, score))
    flags["quality_score"] = score

    return flags


def get_high_zero_share_columns(
    df: pd.DataFrame, zero_share_threshold: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Находит числовые колонки с высокой долей нулевых значений.
    """
    high_zero_columns = []

    for col in df.select_dtypes(include="number").columns:
        s = df[col]
        zero_count = (s == 0).sum()
        total_non_null = s.notna().sum()

        if total_non_null > 0:
            zero_share = zero_count / total_non_null
            if zero_share > zero_share_threshold:
                high_zero_columns.append(
                    {
                        "column": col,
                        "zero_count": int(zero_count),
                        "zero_share": float(zero_share),
                        "threshold": zero_share_threshold,
                    }
                )

    return high_zero_columns


def check_id_duplicates(df: pd.DataFrame, id_column: str = "user_id") -> Dict[str, Any]:
    """
    Проверяет уникальность идентификационной колонки.
    """
    result = {
        "has_id_duplicates": False,
        "id_column": id_column,
        "total_rows": len(df),
        "unique_ids": 0,
        "duplicate_count": 0,
        "duplicate_examples": [],
    }

    if id_column in df.columns:
        id_series = df[id_column]
        unique_count = id_series.nunique(dropna=True)
        duplicate_count = len(id_series) - unique_count

        result["unique_ids"] = int(unique_count)
        result["duplicate_count"] = int(duplicate_count)
        result["has_id_duplicates"] = duplicate_count > 0

        if duplicate_count > 0:
            # Находим примеры дубликатов.
            duplicates = id_series[id_series.duplicated(keep=False)]
            if not duplicates.empty:
                result["duplicate_examples"] = duplicates.head(5).tolist()

    return result


def flatten_summary_for_print(summary: DatasetSummary) -> pd.DataFrame:
    """
    Превращает DatasetSummary в табличку для более удобного вывода.
    """
    rows: List[Dict[str, Any]] = []
    for col in summary.columns:
        rows.append(
            {
                "name": col.name,
                "dtype": col.dtype,
                "non_null": col.non_null,
                "missing": col.missing,
                "missing_share": col.missing_share,
                "unique": col.unique,
                "is_numeric": col.is_numeric,
                "min": col.min,
                "max": col.max,
                "mean": col.mean,
                "std": col.std,
            }
        )
    return pd.DataFrame(rows)
