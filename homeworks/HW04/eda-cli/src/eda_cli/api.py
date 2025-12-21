from __future__ import annotations

from time import perf_counter

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .core import (
    compute_quality_flags,
    missing_table,
    summarize_dataset,
    get_high_zero_share_columns,
    check_id_duplicates,
)

app = FastAPI(
    title="AIE Dataset Quality API",
    version="0.2.0",
    description=(
        "HTTP-сервис-заглушка для оценки готовности датасета к обучению модели. "
        "Использует простые эвристики качества данных вместо настоящей ML-модели."
    ),
    docs_url="/docs",
    redoc_url=None,
)


# ---------- Модели запросов/ответов ----------


class QualityRequest(BaseModel):
    """Агрегированные признаки датасета – 'фичи' для заглушки модели."""

    n_rows: int = Field(..., ge=0, description="Число строк в датасете")
    n_cols: int = Field(..., ge=0, description="Число колонок")
    max_missing_share: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Максимальная доля пропусков среди всех колонок (0..1)",
    )
    numeric_cols: int = Field(
        ...,
        ge=0,
        description="Количество числовых колонок",
    )
    categorical_cols: int = Field(
        ...,
        ge=0,
        description="Количество категориальных колонок",
    )


class QualityResponse(BaseModel):
    """Ответ заглушки модели качества датасета."""

    ok_for_model: bool = Field(
        ...,
        description="True, если датасет считается достаточно качественным для обучения модели",
    )
    quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Интегральная оценка качества данных (0..1)",
    )
    message: str = Field(
        ...,
        description="Человекочитаемое пояснение решения",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Время обработки запроса на сервере, миллисекунды",
    )
    flags: dict[str, bool] | None = Field(
        default=None,
        description="Булевы флаги с подробностями (например, too_few_rows, too_many_missing)",
    )
    dataset_shape: dict[str, int] | None = Field(
        default=None,
        description="Размеры датасета: {'n_rows': ..., 'n_cols': ...}, если известны",
    )


# ---------- Системный эндпоинт ----------


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Простейший health-check сервиса."""
    return {
        "status": "ok",
        "service": "dataset-quality",
        "version": "0.2.0",
    }


# ---------- Заглушка /quality по агрегированным признакам ----------


@app.post("/quality", response_model=QualityResponse, tags=["quality"])
def quality(req: QualityRequest) -> QualityResponse:
    """
    Эндпоинт-заглушка, который принимает агрегированные признаки датасета
    и возвращает эвристическую оценку качества.
    """

    start = perf_counter()

    # Базовый скор от 0 до 1
    score = 1.0

    # Чем больше пропусков, тем хуже
    score -= req.max_missing_share

    # Штраф за слишком маленький датасет
    if req.n_rows < 1000:
        score -= 0.2

    # Штраф за слишком широкий датасет
    if req.n_cols > 100:
        score -= 0.1

    # Штрафы за перекос по типам признаков (если есть числовые и категориальные)
    if req.numeric_cols == 0 and req.categorical_cols > 0:
        score -= 0.1
    if req.categorical_cols == 0 and req.numeric_cols > 0:
        score -= 0.05

    # Нормируем скор в диапазон [0, 1]
    score = max(0.0, min(1.0, score))

    # Простое решение "ок / не ок"
    ok_for_model = score >= 0.7
    if ok_for_model:
        message = "Данных достаточно, модель можно обучать (по текущим эвристикам)."
    else:
        message = (
            "Качество данных недостаточно, требуется доработка (по текущим эвристикам)."
        )

    latency_ms = (perf_counter() - start) * 1000.0

    # Флаги, которые могут быть полезны для последующего логирования/аналитики
    flags = {
        "too_few_rows": req.n_rows < 1000,
        "too_many_columns": req.n_cols > 100,
        "too_many_missing": req.max_missing_share > 0.5,
        "no_numeric_columns": req.numeric_cols == 0,
        "no_categorical_columns": req.categorical_cols == 0,
    }

    # Примитивный лог — на семинаре можно обсудить, как это превратить в нормальный logger
    print(
        f"[quality] n_rows={req.n_rows} n_cols={req.n_cols} "
        f"max_missing_share={req.max_missing_share:.3f} "
        f"score={score:.3f} latency_ms={latency_ms:.1f} ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
    )


# ---------- /quality-from-csv: реальный CSV через нашу EDA-логику ----------


@app.post(
    "/quality-from-csv",
    response_model=QualityResponse,
    tags=["quality"],
    summary="Оценка качества по CSV-файлу с использованием EDA-ядра",
)
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    """
    Эндпоинт, который принимает CSV-файл, запускает EDA-ядро
    (summarize_dataset + missing_table + compute_quality_flags)
    и возвращает оценку качества данных.

    Именно это по сути связывает S03 (CLI EDA) и S04 (HTTP-сервис).
    """

    start = perf_counter()

    if file.content_type not in (
        "text/csv",
        "application/vnd.ms-excel",
        "application/octet-stream",
    ):
        # content_type от браузера может быть разным, поэтому проверка мягкая
        # но для демонстрации оставим простую ветку 400
        raise HTTPException(
            status_code=400, detail="Ожидается CSV-файл (content-type text/csv)."
        )

    try:
        # FastAPI даёт file.file как file-like объект, который можно читать pandas'ом
        df = pd.read_csv(file.file)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(
            status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame)."
        )

    # Используем EDA-ядро из S03
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(summary, missing_df)

    # Ожидаем, что compute_quality_flags вернёт quality_score в [0,1]
    score = float(flags_all.get("quality_score", 0.0))
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7

    if ok_for_model:
        message = "CSV выглядит достаточно качественным для обучения модели (по текущим эвристикам)."
    else:
        message = (
            "CSV требует доработки перед обучением модели (по текущим эвристикам)."
        )

    latency_ms = (perf_counter() - start) * 1000.0

    # Оставляем только булевы флаги для компактности
    flags_bool: dict[str, bool] = {
        key: bool(value) for key, value in flags_all.items() if isinstance(value, bool)
    }

    # Размеры датасета берём из summary (если там есть поля n_rows/n_cols),
    # иначе — напрямую из DataFrame.
    try:
        n_rows = int(getattr(summary, "n_rows"))
        n_cols = int(getattr(summary, "n_cols"))
    except AttributeError:
        n_rows = int(df.shape[0])
        n_cols = int(df.shape[1])

    print(
        f"[quality-from-csv] filename={file.filename!r} "
        f"n_rows={n_rows} n_cols={n_cols} score={score:.3f} "
        f"latency_ms={latency_ms:.1f} ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags_bool,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
    )


# ---------- Новый эндпоинт для получения полных флагов качества ----------


@app.post(
    "/quality-flags-from-csv",
    tags=["quality"],
    summary="Получение полного набора флагов качества из CSV-файла",
    response_model=dict,
)
async def quality_flags_from_csv(
    file: UploadFile = File(...),
    id_column: str = "user_id",
    zero_share_threshold: float = 0.8,
    high_cardinality_threshold: int = 100,
) -> dict:
    """
    Эндпоинт принимает CSV-файл и возвращает полный набор флагов качества,
    включая все эвристики из HW03.

    Параметры запроса:
    - file: CSV-файл (обязательный)
    - id_column: имя колонки для проверки дубликатов (по умолчанию 'user_id')
    - zero_share_threshold: порог для определения "много нулевых значений" (0.0-1.0)
    - high_cardinality_threshold: порог для высокой кардинальности
    """

    start = perf_counter()

    # Проверка типа файла.
    if file.content_type not in (
        "text/csv",
        "application/vnd.ms-excel",
        "application/octet-stream",
    ):
        raise HTTPException(
            status_code=400, detail="Ожидается CSV-файл (content-type text/csv)."
        )

    try:
        # Чтение CSV в DataFrame.
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(
            status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame)."
        )

    # Используем EDA-ядро.
    summary = summarize_dataset(df)
    missing_df = missing_table(df)

    # Получаем все флаги качества с параметрами из HW03.
    flags_all = compute_quality_flags(
        summary,
        missing_df,
        high_cardinality_threshold=high_cardinality_threshold,
        zero_share_threshold=zero_share_threshold,
    )

    # Добавляем проверку нулевых значений (функция из HW03).
    try:
        high_zero_columns = get_high_zero_share_columns(df, zero_share_threshold)
        flags_all["has_high_zero_share_columns"] = len(high_zero_columns) > 0
        flags_all["high_zero_share_columns"] = high_zero_columns
    except Exception as e:
        flags_all["has_high_zero_share_columns"] = False
        flags_all["high_zero_share_error"] = str(e)

    # Добавляем проверку дубликатов ID (функция из HW03).
    try:
        duplicate_check = check_id_duplicates(df, id_column)
        flags_all["has_id_duplicates"] = duplicate_check["has_id_duplicates"]
        flags_all["id_duplicates_details"] = {
            "id_column": duplicate_check["id_column"],
            "total_rows": duplicate_check["total_rows"],
            "unique_ids": duplicate_check["unique_ids"],
            "duplicate_count": duplicate_check["duplicate_count"],
        }
        if duplicate_check["duplicate_examples"]:
            flags_all["id_duplicates_details"]["duplicate_examples"] = duplicate_check[
                "duplicate_examples"
            ]
    except Exception as e:
        flags_all["has_id_duplicates"] = False
        flags_all["id_duplicates_error"] = str(e)

    # Отделяем булевы флаги для чистого вывода.
    bool_flags = {
        "too_few_rows": flags_all.get("too_few_rows", False),
        "too_many_columns": flags_all.get("too_many_columns", False),
        "too_many_missing": flags_all.get("too_many_missing", False),
        "has_constant_columns": flags_all.get("has_constant_columns", False),
        "has_high_cardinality_categoricals": flags_all.get(
            "has_high_cardinality_categoricals", False
        ),
        "has_high_zero_share_columns": flags_all.get(
            "has_high_zero_share_columns", False
        ),
        "has_id_duplicates": flags_all.get("has_id_duplicates", False),
    }

    # Собираем детальную информацию.
    detailed_info = {}

    if flags_all.get("constant_columns"):
        detailed_info["constant_columns"] = flags_all["constant_columns"]

    if flags_all.get("high_cardinality_categoricals"):
        detailed_info["high_cardinality_categoricals"] = flags_all[
            "high_cardinality_categoricals"
        ]

    if flags_all.get("high_zero_share_columns"):
        detailed_info["high_zero_share_columns"] = flags_all["high_zero_share_columns"]

    if flags_all.get("id_duplicates_details"):
        detailed_info["id_duplicates_details"] = flags_all["id_duplicates_details"]

    # Логирование.
    latency_ms = (perf_counter() - start) * 1000.0
    print(
        f"[quality-flags-from-csv] filename={file.filename!r} "
        f"shape={df.shape} bool_flags={sum(bool_flags.values())} "
        f"latency_ms={latency_ms:.1f} ms"
    )

    # Создаём ответ.
    response = {
        "flags": bool_flags,
        "dataset_info": {
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "file_name": file.filename,
        },
        "parameters_used": {
            "id_column": id_column,
            "zero_share_threshold": zero_share_threshold,
            "high_cardinality_threshold": high_cardinality_threshold,
        },
        "latency_ms": round(latency_ms, 2),
    }

    # Добавляем детальную информацию если есть.
    if detailed_info:
        response["details"] = detailed_info

    # Добавляем quality_score если он есть.
    if "quality_score" in flags_all:
        response["quality_score"] = round(float(flags_all["quality_score"]), 3)

    return response
