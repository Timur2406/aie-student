from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    check_id_duplicates,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    get_high_zero_share_columns,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    plot_problematic_columns,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(
        6, help="Максимум числовых колонок для гистограмм."
    ),
    top_k_categories: int = typer.Option(
        5, help="Сколько top-значений выводить для категориальных признаков."
    ),
    title: str = typer.Option("EDA-отчёт", help="Заголовок отчёта в Markdown."),
    min_missing_share: float = typer.Option(
        0.5, help="Порог доли пропусков для выделения проблемных колонок."
    ),
    high_cardinality_threshold: int = typer.Option(
        100, help="Порог уникальных значений для категориальных признаков."
    ),
    zero_share_threshold: float = typer.Option(
        0.8, help="Порог доли нулевых значений для числовых колонок."
    ),
    id_column: str = typer.Option(
        None, help="Идентификационная колонка для проверки уникальности."
    ),
) -> None:
    """
    Сгенерировать полный EDA-отчёт:
    - текстовый overview и summary по колонкам (CSV/Markdown);
    - статистика пропусков;
    - корреляционная матрица;
    - top-k категорий по категориальным признакам;
    - картинки: гистограммы, матрица пропусков, heatmap корреляции.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df, top_k=top_k_categories)

    high_zero_columns = get_high_zero_share_columns(df, zero_share_threshold)

    id_check_result = {}
    if id_column:
        id_check_result = check_id_duplicates(df, id_column)

    quality_flags = compute_quality_flags(
        summary,
        missing_df,
        min_missing_share=min_missing_share,
        high_cardinality_threshold=high_cardinality_threshold,
        zero_share_threshold=zero_share_threshold,
    )

    quality_flags["has_high_zero_share_columns"] = len(high_zero_columns) > 0
    quality_flags["high_zero_share_columns"] = high_zero_columns

    if quality_flags["has_high_zero_share_columns"]:
        zero_penalty = 0.1 * min(1, len(high_zero_columns) / 5)
        quality_flags["quality_score"] = max(
            0.0, quality_flags["quality_score"] - zero_penalty
        )

    if id_column:
        quality_flags.update(id_check_result)

    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"## Параметры отчёта\n\n")
        f.write(f"- Top-k категорий: `{top_k_categories}`\n")
        f.write(f"- Порог пропусков: `{min_missing_share:.0%}`\n")
        f.write(f"- Порог высокой кардинальности: `{high_cardinality_threshold}`\n")
        f.write(f"- Порог нулевых значений: `{zero_share_threshold:.0%}`\n")
        if id_column:
            f.write(f"- Проверяемая ID колонка: `{id_column}`\n")
        f.write(f"\n## Основная информация\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Оценка качества: **{quality_flags['quality_score']:.2f}**\n")
        f.write(
            f"- Макс. доля пропусков по колонке: **{quality_flags['max_missing_share']:.2%}**\n"
        )
        f.write(
            f"- Порог пропусков: **{quality_flags['min_missing_share_threshold']:.0%}**\n"
        )
        f.write(
            f"- Есть колонки с пропусками > порога: **{quality_flags['too_many_missing']}**\n"
        )
        f.write(f"- Слишком мало строк (<100): **{quality_flags['too_few_rows']}**\n")
        f.write(
            f"- Слишком много колонок (>100): **{quality_flags['too_many_columns']}**\n"
        )

        f.write(f"\n### Новые эвристики качества\n\n")

        if quality_flags["has_constant_columns"]:
            f.write(f"- Есть постоянные колонки (все значения одинаковые):\n")
            for col in quality_flags["constant_columns"]:
                f.write(f"  - `{col}`\n")
        else:
            f.write(f"- Нет постоянных колонок\n")

        if quality_flags["has_high_cardinality_categoricals"]:
            f.write(
                f"\n- Есть категориальные признаки с высокой кардинальностью (> {quality_flags['high_cardinality_threshold']}):\n"
            )
            for cat in quality_flags["high_cardinality_categoricals"]:
                f.write(
                    f"  - `{cat['column']}`: {cat['unique_count']} уникальных значений\n"
                )
        else:
            f.write(f"\n- Нет категориальных признаков с высокой кардинальности\n")

        if quality_flags["has_high_zero_share_columns"]:
            f.write(
                f"\n- Есть числовые колонки с высокой долей нулей (> {zero_share_threshold:.0%}):\n"
            )
            for col_info in quality_flags["high_zero_share_columns"]:
                f.write(
                    f"  - `{col_info['column']}`: {col_info['zero_share']:.1%} нулей ({col_info['zero_count']} из {int(col_info['zero_count'] / col_info['zero_share'])})\n"
                )
        else:
            f.write(f"\n- Нет числовых колонок с высокой долей нулей\n")

        if id_column and quality_flags.get("has_id_duplicates", False):
            f.write(f"\n- Есть дубликаты в ID колонке `{id_column}`:\n")
            f.write(f"  - Уникальных значений: {quality_flags['unique_ids']}\n")
            f.write(f"  - Всего строк: {quality_flags['total_rows']}\n")
            f.write(f"  - Дубликатов: {quality_flags['duplicate_count']}\n")
            if quality_flags["duplicate_examples"]:
                f.write(
                    f"  - Примеры дубликатов: {', '.join(map(str, quality_flags['duplicate_examples'][:3]))}\n"
                )
        elif id_column:
            f.write(f"\n- ID колонка `{id_column}` уникальна\n")

        f.write("\n## Колонки\n\n")
        f.write("См. файл `summary.csv`.\n\n")

        f.write("## Пропуски\n\n")
        if missing_df.empty:
            f.write("Пропусков нет или датасет пуст.\n\n")
        else:
            f.write(f"Порог для проблемных колонок: **{min_missing_share:.0%}**\n\n")
            problematic_cols = missing_df[
                missing_df["missing_share"] > min_missing_share
            ]
            if not problematic_cols.empty:
                f.write("### Проблемные колонки (пропусков > порога):\n\n")
                for idx, row in problematic_cols.iterrows():
                    f.write(f"- `{idx}`: {row['missing_share']:.1%} пропусков\n")
                f.write("\n")
            f.write("См. файлы `missing.csv` и `missing_matrix.png`.\n\n")

        f.write("## Корреляция числовых признаков\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок для корреляции.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

        f.write("## Категориальные признаки\n\n")
        f.write(f"Top-{top_k_categories} значений для категориальных признаков:\n\n")
        if not top_cats:
            f.write("Категориальные/строковые признаки не найдены.\n\n")
        else:
            for col_name, table in top_cats.items():
                f.write(f"### `{col_name}`\n\n")
                f.write(table.to_markdown(index=False))
                f.write("\n\n")
            f.write("\nПолные таблицы см. в папке `top_categories/`.\n\n")

        f.write("## Гистограммы числовых колонок\n\n")
        f.write(
            f"Сгенерировано гистограмм для первых **{max_hist_columns}** числовых колонок.\n"
        )
        f.write("См. файлы `hist_*.png`.\n")

    # 6. Картинки
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")
    plot_problematic_columns(
        missing_df, min_missing_share, out_root / "problematic_columns.png"
    )

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"- Основной markdown: {md_path}")
    typer.echo(
        "- Табличные файлы: summary.csv, missing.csv, correlation.csv, top_categories/*.csv"
    )
    typer.echo(
        "- Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png, problematic_columns.png"
    )
    typer.echo(f"\nПараметры отчёта:")
    typer.echo(f"  - Top-k категорий: {top_k_categories}")
    typer.echo(f"  - Порог пропусков: {min_missing_share:.0%}")
    typer.echo(f"  - Порог кардинальности: {high_cardinality_threshold}")
    typer.echo(f"  - Порог нулей: {zero_share_threshold:.0%}")
    if id_column:
        typer.echo(f"  - Проверка ID: {id_column}")


if __name__ == "__main__":
    app()
