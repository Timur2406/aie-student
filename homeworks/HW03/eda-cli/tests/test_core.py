from __future__ import annotations

import pandas as pd
import numpy as np

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    get_high_zero_share_columns,
    check_id_duplicates,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует.
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_constant_columns_heuristic():
    """Тест для проверки эвристики постоянных колонок"""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "constant_col": [
                "same",
                "same",
                "same",
                "same",
                "same",
            ],
            "varying_col": ["A", "B", "A", "C", "B"],
            "numeric_constant": [0, 0, 0, 0, 0],
        }
    )

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, high_cardinality_threshold=10)

    # Проверяем, что флаг наличия постоянных колонок установлен в True.
    assert flags["has_constant_columns"] == True

    # Проверяем, что обе постоянные колонки найдены.
    assert "constant_col" in flags["constant_columns"]
    assert "numeric_constant" in flags["constant_columns"]
    assert len(flags["constant_columns"]) == 2

    # Проверяем, что оценка качества снижена из-за постоянных колонок.
    assert flags["quality_score"] < 1.0


def test_high_cardinality_categorical_heuristic():
    """Тест для проверки эвристики высокой кардинальности категориальных признаков"""
    # Создаем DataFrame с категориальным признаком с высокой кардинальностью.
    df = pd.DataFrame(
        {
            "id": range(150),
            "low_cardinality": ["A", "B", "C"] * 50,
            "high_cardinality": [
                f"value_{i}" for i in range(150)
            ],  # 150 уникальных значений
            "numeric_col": np.random.randn(150),
        }
    )

    summary = summarize_dataset(df)
    missing_df = missing_table(df)

    # Тест с порогом 100 (должен обнаружить высокую кардинальность).
    flags = compute_quality_flags(summary, missing_df, high_cardinality_threshold=100)
    assert flags["has_high_cardinality_categoricals"] == True
    assert any(
        cat["column"] == "high_cardinality"
        for cat in flags["high_cardinality_categoricals"]
    )

    # Тест с очень высоким порогом (не должен обнаружить).
    flags_high_threshold = compute_quality_flags(
        summary, missing_df, high_cardinality_threshold=200
    )
    assert flags_high_threshold["has_high_cardinality_categoricals"] == False

    # Тест с низким порогом (должен обнаружить даже low_cardinality как высокую).
    flags_low_threshold = compute_quality_flags(
        summary, missing_df, high_cardinality_threshold=2
    )
    assert flags_low_threshold["has_high_cardinality_categoricals"] == True
    assert len(flags_low_threshold["high_cardinality_categoricals"]) == 2


def test_high_zero_share_heuristic():
    """Тест для проверки эвристики высокой доли нулевых значений"""
    df = pd.DataFrame(
        {
            "id": range(10),
            "no_zeros": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "many_zeros": [0, 0, 0, 0, 0, 0, 0, 1, 2, 3],
            "all_zeros": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "some_zeros": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    # Проверяем функцию get_high_zero_share_columns.
    high_zero_cols = get_high_zero_share_columns(df, zero_share_threshold=0.6)

    # Должны найти колонки с долей нулей > 60%.
    assert len(high_zero_cols) == 2
    assert any(col["column"] == "many_zeros" for col in high_zero_cols)
    assert any(col["column"] == "all_zeros" for col in high_zero_cols)

    # Проверяем конкретные значения для many_zeros.
    for col in high_zero_cols:
        if col["column"] == "many_zeros":
            assert col["zero_share"] == 0.7
            assert col["zero_count"] == 7

    # Проверяем с более высоким порогом.
    high_zero_cols_strict = get_high_zero_share_columns(df, zero_share_threshold=0.9)
    assert len(high_zero_cols_strict) == 1
    assert high_zero_cols_strict[0]["column"] == "all_zeros"


def test_id_duplicates_heuristic():
    """Тест для проверки эвристики дубликатов ID"""
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 1, 2, 4, 5, 3],
            "unique_id": [100, 101, 102, 103, 104, 105, 106, 107],
            "name": [
                "Alice",
                "Bob",
                "Charlie",
                "David",
                "Eve",
                "Frank",
                "Grace",
                "Henry",
            ],
            "value": [10, 20, 30, 40, 50, 60, 70, 80],
        }
    )

    # Проверяем функцию check_id_duplicates.
    result_with_duplicates = check_id_duplicates(df, id_column="user_id")

    assert result_with_duplicates["has_id_duplicates"] == True
    assert result_with_duplicates["id_column"] == "user_id"
    assert result_with_duplicates["total_rows"] == 8
    assert result_with_duplicates["unique_ids"] == 5
    assert result_with_duplicates["duplicate_count"] == 3

    # Проверяем, что найдены примеры дубликатов.
    assert len(result_with_duplicates["duplicate_examples"]) > 0
    assert set(result_with_duplicates["duplicate_examples"]).issubset({1, 2, 3})

    # Проверяем с уникальной колонкой.
    result_unique = check_id_duplicates(df, id_column="unique_id")
    assert result_unique["has_id_duplicates"] == False
    assert result_unique["unique_ids"] == 8
    assert result_unique["duplicate_count"] == 0

    # Проверяем с несуществующей колонкой.
    result_nonexistent = check_id_duplicates(df, id_column="nonexistent_column")
    assert result_nonexistent["has_id_duplicates"] == False
    assert result_nonexistent["unique_ids"] == 0


def test_quality_score_integration():
    """Интеграционный тест проверки влияния новых эвристик на общую оценку качества"""
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3],
            "constant": ["A", "A", "A", "A", "A"],
            "high_card": [f"val_{i}" for i in range(5)],
            "many_zeros": [0, 0, 0, 0, 1],
            "missing": [1, 2, None, None, None],
        }
    )

    summary = summarize_dataset(df)
    missing_df = missing_table(df)

    # Строгие настройки для выявления всех проблем.
    flags_strict = compute_quality_flags(
        summary,
        missing_df,
        min_missing_share=0.5,
        high_cardinality_threshold=3,
        zero_share_threshold=0.7,
    )

    # Проверяем, что все флаги проблем установлены.
    assert flags_strict["has_constant_columns"] == True
    assert flags_strict["too_many_missing"] == True

    # Проверяем, что оценка качества снижена из-за множества проблем.
    assert flags_strict["quality_score"] < 0.5

    # Сравниваем с менее строгими настройками
    flags_lenient = compute_quality_flags(
        summary,
        missing_df,
        min_missing_share=0.7,
        high_cardinality_threshold=10,
        zero_share_threshold=0.9,
    )

    assert flags_lenient["too_many_missing"] == False
    assert flags_lenient["quality_score"] > flags_strict["quality_score"]


def test_compute_quality_flags_with_custom_thresholds():
    """Тест проверки работы функции compute_quality_flags с пользовательскими порогами"""
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [
                "A",
                "B",
                "C",
                "D",
                "E",
            ],
            "col3": [0, 0, 0, 1, 2],
        }
    )

    summary = summarize_dataset(df)
    missing_df = missing_table(df)

    # Тестируем с разными пользовательскими порогами.
    flags_custom = compute_quality_flags(
        summary,
        missing_df,
        min_missing_share=0.3,
        high_cardinality_threshold=4,
        zero_share_threshold=0.5,
    )

    # Проверяем, что пороги сохраняются в результатах.
    assert flags_custom["min_missing_share_threshold"] == 0.3
    assert flags_custom["high_cardinality_threshold"] == 4
    assert flags_custom["zero_share_threshold"] == 0.5

    # Проверяем, что высокоя кардинальность обнаружена.
    assert flags_custom["has_high_cardinality_categoricals"] == True
