"""
services/dataset_cleaner.py

Business logic for DataPilot AI's deterministic dataset cleaning
feature.

Responsibilities:
- Read a CSV file into a pandas DataFrame, preserving its original
  text encoding whenever possible.
- Apply a fixed, deterministic, 13-stage cleaning pipeline: column
  name standardization, duplicate/empty-row removal, empty-column
  removal, whitespace trimming, missing-value normalization and
  filling, safe numeric/datetime conversion, categorical text
  standardization, outlier detection (reporting only), constant-column
  removal, and internal whitespace collapsing.
- Save the cleaned dataset to a dedicated output directory under a
  collision-resistant filename, using the same encoding as the input.
- Return a structured summary of every action taken.

This module performs ONLY deterministic, rule-based cleaning. It has
NO knowledge of Slack, NO knowledge of any AI provider, performs NO
networking, and touches NO database. It is a pure, reusable
transformation service: a local CSV path goes in, a local CSV path
comes out, plus a summary of what changed. Any future caller (Slack,
CLI, REST API, desktop app) can use this class without modification.

This is NOT the same responsibility as services.dataset_profiler
.DatasetProfiler, which only analyzes and reports on a dataset without
modifying it. DatasetCleaner physically transforms the data.

Public interface (unchanged from prior versions):
    class DatasetCleaner:
        def __init__(self, output_dir: str | None = None) -> None
        def clean_dataset(self, input_csv_path: str) -> CleaningResult
    class CleaningResult (all original fields and properties preserved,
        with additional fields appended for richer reporting).
    class DatasetCleanerError(Exception)
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR: Final[str] = "data/cleaned"
_OUTPUT_SUFFIX: Final[str] = "_cleaned.csv"
_DATETIME_SUCCESS_THRESHOLD: Final[float] = 0.9
_NUMERIC_SUCCESS_THRESHOLD: Final[float] = 0.9
_MIN_NON_NULL_SAMPLE: Final[int] = 1
_IQR_MULTIPLIER: Final[float] = 1.5
_CANDIDATE_ENCODINGS: Final[tuple[str, ...]] = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
_DEFAULT_ENCODING: Final[str] = "utf-8"

# Literal tokens treated as "missing" once trimmed of surrounding
# whitespace, regardless of surrounding case variations already
# covered by the explicit entries below.
_MISSING_TOKENS: Final[frozenset[str]] = frozenset(
    {"", " ", "NA", "N/A", "NULL", "null", "None", "-"}
)

# Known categorical value sets eligible for case standardization.
# Each entry maps a set of case-insensitive input tokens to their
# canonical output form. A column is only standardized if EVERY one
# of its non-null values (case-insensitively) belongs to a single one
# of these sets, to avoid corrupting unrelated free-text columns.
_CATEGORICAL_CANONICAL_SETS: Final[tuple[dict[str, str], ...]] = (
    {"male": "Male", "female": "Female"},
    {"yes": "Yes", "no": "No"},
    {"true": "True", "false": "False"},
)


class DatasetCleanerError(Exception):
    """
    Raised whenever DatasetCleaner cannot successfully read, clean, or
    save a dataset.

    Callers only need to catch this single exception type; they do
    not need to know about pandas' or the filesystem's internal
    exception hierarchies.
    """


@dataclass(frozen=True)
class CleaningResult:
    """
    Structured, immutable summary of a single dataset cleaning
    operation.

    Exposes every piece of information computed internally by
    DatasetCleaner, so callers (e.g. Slack Block Kit builders) can
    present a rich, structured Dataset Cleaning Report without parsing
    logs or recomputing anything.

    Original attributes (preserved for backward compatibility with
    existing integrations):
        input_file: The original local CSV path that was cleaned.
        output_file: The local path where the cleaned CSV was saved.
        original_rows: Number of rows before cleaning.
        cleaned_rows: Number of rows after cleaning.
        original_columns: Number of columns before cleaning.
        cleaned_columns: Number of columns after cleaning.
        duplicates_removed: Number of duplicate rows removed.
        empty_rows_removed: Number of completely empty rows removed.
        columns_removed: Names of all columns removed (empty-only
            columns and constant-value columns combined).
        columns_converted: Names of columns converted to numeric or
            datetime type (post-rename column names).
        renamed_columns: Mapping of original column name to its
            normalized snake_case name, for every renamed column.
        cleaning_actions: Ordered, human-readable log of every
            cleaning action performed.

    Additional attributes (new):
        rows_before: Alias of original_rows.
        rows_after: Alias of cleaned_rows.
        columns_before: Alias of original_columns.
        columns_after: Alias of cleaned_columns.
        empty_columns_removed: Count of columns removed for containing
            only null values.
        missing_values_filled: Mapping of column name to the number of
            missing values filled in that column.
        removed_columns: Same as columns_removed; exposed under the
            requested field name for reporting convenience.
        outlier_summary: Mapping of numeric column name to a dict with
            "count", "lower_bound", and "upper_bound" describing
            IQR-based outliers detected (not removed).
    """

    input_file: str
    output_file: str
    original_rows: int
    cleaned_rows: int
    original_columns: int
    cleaned_columns: int
    duplicates_removed: int
    empty_rows_removed: int
    columns_removed: list[str] = field(default_factory=list)
    columns_converted: list[str] = field(default_factory=list)
    renamed_columns: dict[str, str] = field(default_factory=dict)
    cleaning_actions: list[str] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0
    columns_before: int = 0
    columns_after: int = 0
    empty_columns_removed: int = 0
    missing_values_filled: dict[str, int] = field(default_factory=dict)
    removed_columns: list[str] = field(default_factory=list)
    outlier_summary: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def rows_removed(self) -> int:
        """
        Returns:
            The total number of rows removed during cleaning
            (original_rows minus cleaned_rows).
        """
        return self.original_rows - self.cleaned_rows

    @property
    def has_removed_duplicates(self) -> bool:
        """
        Returns:
            True if at least one duplicate row was removed.
        """
        return self.duplicates_removed > 0

    @property
    def has_removed_empty_rows(self) -> bool:
        """
        Returns:
            True if at least one completely empty row was removed.
        """
        return self.empty_rows_removed > 0

    @property
    def has_removed_columns(self) -> bool:
        """
        Returns:
            True if at least one column was removed.
        """
        return len(self.columns_removed) > 0

    @property
    def has_converted_columns(self) -> bool:
        """
        Returns:
            True if at least one column was converted to a numeric or
            datetime type.
        """
        return len(self.columns_converted) > 0

    @property
    def total_changes(self) -> int:
        """
        Returns:
            The total count of distinct changes applied to the
            dataset: duplicate rows removed, empty rows removed,
            columns removed, columns converted, and columns renamed.
        """
        return (
            self.duplicates_removed
            + self.empty_rows_removed
            + len(self.columns_removed)
            + len(self.columns_converted)
            + len(self.renamed_columns)
        )

    @property
    def cleaning_successful(self) -> bool:
        """
        Returns:
            True if the cleaned output file path was recorded and at
            least one row remains in the cleaned dataset.
        """
        return bool(self.output_file) and self.cleaned_rows > 0


class DatasetCleaner:
    """
    Applies a fixed, deterministic 13-stage cleaning pipeline to a CSV
    dataset and saves the result to disk.

    Pipeline order:
        1. Standardize column names to snake_case.
        2. Remove duplicate rows.
        3. Remove rows that are completely empty.
        4. Remove columns that are completely empty.
        5. Trim leading/trailing whitespace from string columns.
        6. Normalize known "missing" tokens (e.g. "NA", "N/A", "-") to
           proper nulls.
        7. Fill missing values: numeric columns with the column
           median, categorical columns with the column mode (or
           "Unknown" if no mode exists).
        8. Convert numeric-looking string columns (including
           thousands-separated values like "1,000") to numeric type,
           where safe.
        9. Convert datetime-looking string columns to datetime type,
           where safe, exported as YYYY-MM-DD strings in the output.
        10. Standardize known categorical text values (e.g.
            male/Male/MALE -> Male).
        11. Detect numeric outliers via the IQR method (reported only,
            never removed).
        12. Drop columns with only one unique value.
        13. Collapse repeated internal whitespace within string
            values.

    Row order is preserved throughout (aside from rows explicitly
    removed), and the input file's text encoding is detected and
    reused when saving the cleaned output. No transformation is
    applied unless it is considered safe; failures on any single
    column are logged as warnings and skipped rather than aborting
    the entire cleaning run.

    This class has no external dependencies beyond pandas and the
    filesystem. It performs no networking, no AI calls, and no
    database access, making it trivially reusable from any interface.
    """

    def __init__(self, output_dir: str | None = None) -> None:
        """
        Initialize the dataset cleaner.

        Args:
            output_dir: Directory in which cleaned CSV files are
                saved. Falls back to "data/cleaned" if not provided.
                Created automatically if it does not exist.

        Raises:
            DatasetCleanerError: If the output directory cannot be
                created.
        """
        self._output_dir: Path = Path(output_dir or _DEFAULT_OUTPUT_DIR)
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.exception(
                "Failed to create cleaned output directory: %s", self._output_dir
            )
            raise DatasetCleanerError(
                f"Could not create output directory '{self._output_dir}'."
            ) from exc

        logger.debug("DatasetCleaner initialized. output_dir=%s", self._output_dir)

    def clean_dataset(self, input_csv_path: str) -> CleaningResult:
        """
        Clean a CSV dataset and save the result to disk.

        Args:
            input_csv_path: Filesystem path to the CSV file to clean.

        Returns:
            A CleaningResult containing the cleaned file path alongside
            every piece of information computed during cleaning.

        Raises:
            DatasetCleanerError: If the input path is empty, the file
                does not exist, the file cannot be parsed as CSV, the
                resulting dataset is empty, or the cleaned file cannot
                be saved.
        """
        cleaned_path_str = input_csv_path.strip()
        if not cleaned_path_str:
            logger.warning("Dataset cleaning requested with an empty file path.")
            raise DatasetCleanerError("No CSV file path was provided to clean.")

        df, encoding = self._load_csv(cleaned_path_str)
        original_rows = int(df.shape[0])
        original_columns = int(df.shape[1])

        actions: list[str] = []

        # Stage 1: standardize column names to snake_case.
        df, renamed_columns = self._standardize_column_names(df)
        if renamed_columns:
            actions.append(f"Standardized {len(renamed_columns)} column name(s) to snake_case.")
        logger.info("Standardized column names. renamed=%d", len(renamed_columns))

        # Stage 2: remove duplicate rows.
        df, duplicates_removed = self._remove_duplicate_rows(df)
        actions.append(f"Removed {duplicates_removed} duplicate row(s).")
        logger.info("Removed duplicates. count=%d", duplicates_removed)

        # Stage 3: remove completely empty rows.
        df, empty_rows_removed = self._remove_empty_rows(df)
        actions.append(f"Removed {empty_rows_removed} completely empty row(s).")
        logger.info("Removed empty rows. count=%d", empty_rows_removed)

        # Stage 4: remove completely empty columns.
        df, empty_columns_removed_list = self._remove_null_only_columns(df)
        if empty_columns_removed_list:
            actions.append(
                f"Removed {len(empty_columns_removed_list)} empty column(s): "
                f"{', '.join(empty_columns_removed_list)}."
            )
        logger.info("Removed empty columns. count=%d", len(empty_columns_removed_list))

        # Stage 5: trim leading/trailing whitespace from string columns.
        df = self._trim_string_columns(df)
        actions.append("Trimmed leading/trailing whitespace from string columns.")

        # Stage 6: normalize known "missing" tokens to proper nulls.
        df, missing_tokens_normalized = self._normalize_missing_tokens(df)
        if missing_tokens_normalized:
            actions.append(
                f"Normalized missing-value tokens in {missing_tokens_normalized} cell(s)."
            )

        # Stage 7: fill missing values (numeric -> median, categorical -> mode/"Unknown").
        df, missing_values_filled = self._fill_missing_values(df)
        if missing_values_filled:
            actions.append(
                "Filled missing values in column(s): "
                + ", ".join(f"{col} ({count})" for col, count in missing_values_filled.items())
                + "."
            )
        logger.info("Filled missing values. columns=%d", len(missing_values_filled))

        # Stage 8: convert numeric-looking string columns.
        df, numeric_converted = self._convert_numeric_columns(df)
        if numeric_converted:
            actions.append(
                f"Converted numeric-looking column(s) to numeric type: "
                f"{', '.join(numeric_converted)}."
            )
        logger.info("Converted numeric columns. columns=%s", numeric_converted)

        # Stage 9: convert datetime-looking string columns.
        df, datetime_converted = self._convert_datetime_columns(df)
        if datetime_converted:
            actions.append(
                f"Converted datetime-looking column(s) to datetime type: "
                f"{', '.join(datetime_converted)}."
            )
        logger.info("Normalized date columns. columns=%s", datetime_converted)

        columns_converted = numeric_converted + datetime_converted

        # Stage 10: standardize known categorical text values.
        df, standardized_columns = self._standardize_categorical_values(df)
        if standardized_columns:
            actions.append(
                f"Standardized categorical text in column(s): {', '.join(standardized_columns)}."
            )

        # Stage 11: detect numeric outliers (reporting only).
        outlier_summary = self._detect_outliers(df)
        if outlier_summary:
            actions.append(
                f"Detected outliers in {len(outlier_summary)} numeric column(s) (not removed)."
            )

        # Stage 12: drop constant (single-unique-value) columns.
        df, constant_columns_removed = self._remove_constant_columns(df)
        if constant_columns_removed:
            actions.append(
                f"Removed {len(constant_columns_removed)} constant column(s): "
                f"{', '.join(constant_columns_removed)}."
            )

        # Stage 13: collapse repeated internal whitespace in string values.
        df = self._collapse_internal_whitespace(df)
        actions.append("Collapsed repeated internal whitespace in string columns.")

        cleaned_rows = int(df.shape[0])
        cleaned_columns = int(df.shape[1])

        all_removed_columns = empty_columns_removed_list + constant_columns_removed

        output_path = self._build_output_path(cleaned_path_str)
        self._save_csv(df, output_path, encoding, datetime_converted)
        logger.info("Saved cleaned dataset. output=%s, encoding=%s", output_path, encoding)

        result = CleaningResult(
            input_file=cleaned_path_str,
            output_file=str(output_path),
            original_rows=original_rows,
            cleaned_rows=cleaned_rows,
            original_columns=original_columns,
            cleaned_columns=cleaned_columns,
            duplicates_removed=duplicates_removed,
            empty_rows_removed=empty_rows_removed,
            columns_removed=all_removed_columns,
            columns_converted=columns_converted,
            renamed_columns=renamed_columns,
            cleaning_actions=actions,
            rows_before=original_rows,
            rows_after=cleaned_rows,
            columns_before=original_columns,
            columns_after=cleaned_columns,
            empty_columns_removed=len(empty_columns_removed_list),
            missing_values_filled=missing_values_filled,
            removed_columns=all_removed_columns,
            outlier_summary=outlier_summary,
        )

        logger.info(
            "Cleaning completed. input=%s, output=%s, original_rows=%d, "
            "cleaned_rows=%d, duplicates_removed=%d, empty_rows_removed=%d, "
            "columns_removed=%d",
            result.input_file,
            result.output_file,
            result.original_rows,
            result.cleaned_rows,
            result.duplicates_removed,
            result.empty_rows_removed,
            len(result.columns_removed),
        )

        return result

    def _load_csv(self, file_path: str) -> tuple[pd.DataFrame, str]:
        """
        Load a CSV file into a pandas DataFrame, detecting its text
        encoding so the same encoding can be reused when saving.

        Args:
            file_path: Filesystem path to the CSV file.

        Returns:
            A tuple of (parsed DataFrame, detected encoding name).

        Raises:
            DatasetCleanerError: If the file does not exist, is not a
                file, cannot be parsed as CSV under any candidate
                encoding, or contains zero rows.
        """
        path = Path(file_path)

        if not path.exists() or not path.is_file():
            logger.warning("Dataset cleaning requested for missing file: %s", path)
            raise DatasetCleanerError(f"CSV file not found: '{file_path}'.")

        df: pd.DataFrame | None = None
        used_encoding = _DEFAULT_ENCODING
        last_error: Exception | None = None

        for encoding in _CANDIDATE_ENCODINGS:
            try:
                df = pd.read_csv(path, encoding=encoding)
                used_encoding = encoding
                break
            except UnicodeDecodeError as exc:
                last_error = exc
                continue
            except pd.errors.EmptyDataError as exc:
                logger.warning("CSV file is empty: %s", path)
                raise DatasetCleanerError(f"The CSV file '{file_path}' is empty.") from exc
            except pd.errors.ParserError as exc:
                logger.error("Failed to parse CSV file %s: %s", path, exc)
                raise DatasetCleanerError(
                    f"Could not parse '{file_path}' as a valid CSV file."
                ) from exc

        if df is None:
            logger.exception("Failed to read CSV file with any known encoding: %s", path)
            raise DatasetCleanerError(
                f"Could not read the file '{file_path}' with any supported encoding."
            ) from last_error

        if df.shape[0] == 0:
            logger.warning("CSV file contains no data rows: %s", path)
            raise DatasetCleanerError(f"The CSV file '{file_path}' contains no data rows.")

        logger.info(
            "Loaded dataset. input=%s, rows=%d, columns=%d, encoding=%s",
            path, *df.shape, used_encoding,
        )
        return df, used_encoding

    @staticmethod
    def _to_snake_case(name: object) -> str:
        """
        Convert a single column name into snake_case.

        Strips surrounding whitespace, removes special symbols (e.g.
        "$", "(", ")"), collapses internal whitespace into single
        underscores, collapses duplicate underscores, and lowercases
        the result.

        Args:
            name: The original column name (any type; converted to
                str first).

        Returns:
            A normalized snake_case column name.
        """
        text = str(name).strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", "_", text.strip())
        text = re.sub(r"_+", "_", text)
        return text.lower().strip("_") or "column"

    def _standardize_column_names(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        """
        Standardize every column name to snake_case, resolving any
        collisions that result from normalization by appending a
        numeric suffix.

        Args:
            df: The dataset whose columns should be renamed.

        Returns:
            A tuple of (DataFrame with standardized column names,
            mapping of original column name to its new name, for
            every column whose name actually changed).
        """
        rename_map: dict[str, str] = {}
        seen_names: dict[str, int] = {}

        for column in df.columns:
            normalized = self._to_snake_case(column)
            if normalized in seen_names:
                seen_names[normalized] += 1
                normalized = f"{normalized}_{seen_names[normalized]}"
            else:
                seen_names[normalized] = 1

            if normalized != column:
                rename_map[column] = normalized

        cleaned = df.rename(columns=rename_map)
        return cleaned, rename_map

    @staticmethod
    def _remove_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Remove exact duplicate rows, keeping the first occurrence and
        preserving row order.

        Args:
            df: The dataset to deduplicate.

        Returns:
            A tuple of (deduplicated DataFrame, number of rows removed).
        """
        duplicate_count = int(df.duplicated().sum())
        cleaned = df.drop_duplicates(keep="first").reset_index(drop=True)
        return cleaned, duplicate_count

    @staticmethod
    def _remove_empty_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Remove rows in which every value is null, preserving row
        order.

        Args:
            df: The dataset to clean.

        Returns:
            A tuple of (DataFrame with empty rows removed, number of
            rows removed).
        """
        empty_mask = df.isnull().all(axis=1)
        empty_count = int(empty_mask.sum())
        cleaned = df.loc[~empty_mask].reset_index(drop=True)
        return cleaned, empty_count

    @staticmethod
    def _remove_null_only_columns(
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Remove columns in which every value is null.

        Args:
            df: The dataset to clean.

        Returns:
            A tuple of (DataFrame with fully-null columns removed,
            list of removed column names).
        """
        null_only_columns = [
            column for column in df.columns if df[column].isnull().all()
        ]
        cleaned = df.drop(columns=null_only_columns)
        return cleaned, null_only_columns

    @staticmethod
    def _trim_string_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Trim leading and trailing whitespace from every string
        (object-dtype) column.

        Args:
            df: The dataset to clean.

        Returns:
            The DataFrame with string columns trimmed. Non-string
            values within object columns (e.g. NaN) are left
            unchanged.
        """
        string_columns = df.select_dtypes(include="object").columns

        for column in string_columns:
            try:
                df[column] = df[column].apply(
                    lambda value: value.strip() if isinstance(value, str) else value
                )
            except Exception:  # noqa: BLE001 - never let one bad column abort cleaning
                logger.warning("Failed to trim whitespace for column '%s'; skipping.", column)

        return df

    @staticmethod
    def _normalize_missing_tokens(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Replace known "missing value" tokens (e.g. "NA", "N/A",
        "NULL", "-") with proper pandas nulls across all object
        columns.

        Args:
            df: The dataset to clean.

        Returns:
            A tuple of (DataFrame with missing tokens normalized to
            NA, total number of cells that were converted).
        """
        string_columns = df.select_dtypes(include="object").columns
        total_normalized = 0

        for column in string_columns:
            try:
                mask = df[column].apply(
                    lambda value: isinstance(value, str) and value.strip() in _MISSING_TOKENS
                )
                count = int(mask.sum())
                if count:
                    df.loc[mask, column] = pd.NA
                    total_normalized += count
            except Exception:  # noqa: BLE001 - never let one bad column abort cleaning
                logger.warning(
                    "Failed to normalize missing tokens for column '%s'; skipping.", column
                )

        return df, total_normalized

    @staticmethod
    def _fill_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        """
        Fill missing values: numeric columns with the column median,
        categorical (object) columns with the column mode, or
        "Unknown" if no mode can be determined.

        Args:
            df: The dataset to clean.

        Returns:
            A tuple of (DataFrame with missing values filled, mapping
            of column name to the number of values filled in that
            column).
        """
        filled: dict[str, int] = {}

        for column in df.columns:
            missing_count = int(df[column].isnull().sum())
            if missing_count == 0:
                continue

            try:
                if pd.api.types.is_numeric_dtype(df[column]):
                    median_value = df[column].median()
                    if pd.isna(median_value):
                        continue
                    df[column] = df[column].fillna(median_value)
                    filled[column] = missing_count
                else:
                    mode_values = df[column].mode(dropna=True)
                    fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
                    df[column] = df[column].fillna(fill_value)
                    filled[column] = missing_count
            except Exception:  # noqa: BLE001 - never let one bad column abort cleaning
                logger.warning("Failed to fill missing values for column '%s'; skipping.", column)

        return df, filled

    def _convert_numeric_columns(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Convert string columns that predominantly contain numeric
        values (including thousands-separated values like "1,000")
        into a numeric dtype.

        A column is only converted if at least
        _NUMERIC_SUCCESS_THRESHOLD of its non-null values can be
        parsed as numbers, ensuring the conversion is safe and does
        not silently discard meaningful non-numeric data.

        Args:
            df: The dataset to clean.

        Returns:
            A tuple of (DataFrame with eligible columns converted,
            list of column names that were converted).
        """
        converted_columns: list[str] = []
        string_columns = df.select_dtypes(include="object").columns

        for column in string_columns:
            try:
                non_null_values = df[column].dropna()
                if len(non_null_values) < _MIN_NON_NULL_SAMPLE:
                    continue

                cleaned_values = non_null_values.apply(
                    lambda value: value.replace(",", "").strip()
                    if isinstance(value, str)
                    else value
                )
                parsed = pd.to_numeric(cleaned_values, errors="coerce")
                success_ratio = parsed.notna().mean()

                if success_ratio >= _NUMERIC_SUCCESS_THRESHOLD:
                    full_cleaned = df[column].apply(
                        lambda value: value.replace(",", "").strip()
                        if isinstance(value, str)
                        else value
                    )
                    df[column] = pd.to_numeric(full_cleaned, errors="coerce")
                    converted_columns.append(column)
            except Exception:  # noqa: BLE001 - never let one bad column abort cleaning
                logger.warning(
                    "Failed to convert column '%s' to numeric; skipping.", column
                )

        return df, converted_columns

    def _convert_datetime_columns(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Convert string columns that predominantly contain datetime
        values into a datetime dtype.

        A column is only converted if at least
        _DATETIME_SUCCESS_THRESHOLD of its non-null values can be
        parsed as dates/timestamps.

        Args:
            df: The dataset to clean.

        Returns:
            A tuple of (DataFrame with eligible columns converted,
            list of column names that were converted).
        """
        converted_columns: list[str] = []
        string_columns = df.select_dtypes(include="object").columns

        for column in string_columns:
            try:
                non_null_values = df[column].dropna()
                if len(non_null_values) < _MIN_NON_NULL_SAMPLE:
                    continue

                parsed = pd.to_datetime(non_null_values, errors="coerce")
                success_ratio = parsed.notna().mean()

                if success_ratio >= _DATETIME_SUCCESS_THRESHOLD:
                    df[column] = pd.to_datetime(df[column], errors="coerce")
                    converted_columns.append(column)
            except Exception:  # noqa: BLE001 - never let one bad column abort cleaning
                logger.warning(
                    "Failed to convert column '%s' to datetime; skipping.", column
                )

        return df, converted_columns

    @staticmethod
    def _standardize_categorical_values(
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Standardize the casing of known categorical text values (e.g.
        "male"/"Male"/"MALE" -> "Male", "yes"/"YES" -> "Yes").

        A column is only standardized if every one of its non-null
        values matches (case-insensitively) a single known canonical
        set, to avoid corrupting unrelated free-text columns.

        Args:
            df: The dataset to clean.

        Returns:
            A tuple of (DataFrame with eligible columns standardized,
            list of column names that were standardized).
        """
        standardized_columns: list[str] = []
        string_columns = df.select_dtypes(include="object").columns

        for column in string_columns:
            try:
                non_null_values = df[column].dropna()
                if non_null_values.empty:
                    continue

                lowered_unique = {
                    value.strip().lower()
                    for value in non_null_values
                    if isinstance(value, str)
                }
                if not lowered_unique:
                    continue

                for canonical_set in _CATEGORICAL_CANONICAL_SETS:
                    if lowered_unique.issubset(canonical_set.keys()):
                        df[column] = df[column].apply(
                            lambda value: canonical_set.get(value.strip().lower(), value)
                            if isinstance(value, str)
                            else value
                        )
                        standardized_columns.append(column)
                        break
            except Exception:  # noqa: BLE001 - never let one bad column abort cleaning
                logger.warning(
                    "Failed to standardize categorical values for column '%s'; skipping.",
                    column,
                )

        return df, standardized_columns

    @staticmethod
    def _detect_outliers(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """
        Detect numeric outliers using the IQR method. Outliers are
        reported only; they are never removed or modified.

        Args:
            df: The dataset to analyze.

        Returns:
            A mapping of numeric column name to a dict with "count",
            "lower_bound", and "upper_bound", for every numeric column
            with a non-zero interquartile range and at least one
            detected outlier.
        """
        outlier_summary: dict[str, dict[str, Any]] = {}
        numeric_columns = df.select_dtypes(include="number").columns

        for column in numeric_columns:
            try:
                series = df[column].dropna()
                if series.empty:
                    continue

                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1

                if iqr == 0 or pd.isna(iqr):
                    continue

                lower_bound = q1 - _IQR_MULTIPLIER * iqr
                upper_bound = q3 + _IQR_MULTIPLIER * iqr
                outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())

                if outlier_count > 0:
                    outlier_summary[column] = {
                        "count": outlier_count,
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound),
                    }
            except Exception:  # noqa: BLE001 - never let one bad column abort cleaning
                logger.warning("Failed to detect outliers for column '%s'; skipping.", column)

        return outlier_summary

    @staticmethod
    def _remove_constant_columns(
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Remove columns that contain only one unique non-null value.

        Args:
            df: The dataset to clean.

        Returns:
            A tuple of (DataFrame with constant columns removed, list
            of removed column names).
        """
        constant_columns = [
            column for column in df.columns if df[column].nunique(dropna=True) <= 1
        ]
        cleaned = df.drop(columns=constant_columns)
        return cleaned, constant_columns

    @staticmethod
    def _collapse_internal_whitespace(df: pd.DataFrame) -> pd.DataFrame:
        """
        Collapse repeated internal whitespace within string values
        into a single space (e.g. "John     Smith" -> "John Smith").

        Args:
            df: The dataset to clean.

        Returns:
            The DataFrame with internal whitespace collapsed in every
            string (object-dtype) column. Non-string values are left
            unchanged.
        """
        string_columns = df.select_dtypes(include="object").columns

        for column in string_columns:
            try:
                df[column] = df[column].apply(
                    lambda value: re.sub(r"\s+", " ", value) if isinstance(value, str) else value
                )
            except Exception:  # noqa: BLE001 - never let one bad column abort cleaning
                logger.warning(
                    "Failed to collapse internal whitespace for column '%s'; skipping.", column
                )

        return df

    def _build_output_path(self, input_csv_path: str) -> Path:
        """
        Build a collision-resistant output path for the cleaned CSV.

        Args:
            input_csv_path: The original input CSV path, used to
                derive the base output filename.

        Returns:
            A Path within the output directory, e.g.
            "data/cleaned/sales_cleaned.csv", or
            "data/cleaned/sales_cleaned_20260708_104532.csv" if the
            base filename already exists.
        """
        stem = Path(input_csv_path).stem
        candidate = self._output_dir / f"{stem}{_OUTPUT_SUFFIX}"

        if not candidate.exists():
            return candidate

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self._output_dir / f"{stem}_cleaned_{timestamp}.csv"

    @staticmethod
    def _save_csv(
        df: pd.DataFrame,
        output_path: Path,
        encoding: str,
        datetime_columns: list[str],
    ) -> None:
        """
        Save a cleaned DataFrame to disk as CSV, using the same text
        encoding detected from the input file and formatting any
        datetime columns as YYYY-MM-DD strings.

        Args:
            df: The cleaned dataset to save.
            output_path: The destination path.
            encoding: The text encoding to write with (matching the
                input file's detected encoding).
            datetime_columns: Names of columns that were converted to
                datetime dtype, to be formatted as YYYY-MM-DD strings
                on export.

        Raises:
            DatasetCleanerError: If the file cannot be written.
        """
        try:
            export_df = df.copy()
            for column in datetime_columns:
                if column in export_df.columns:
                    export_df[column] = export_df[column].dt.strftime("%Y-%m-%d")
                    export_df[column] = export_df[column].where(
                        export_df[column].notna(), ""
                    )

            export_df.to_csv(output_path, index=False, encoding=encoding)
        except OSError as exc:
            logger.exception("Failed to save cleaned CSV to %s", output_path)
            raise DatasetCleanerError(
                f"Could not save the cleaned CSV to '{output_path}'."
            ) from exc