"""
services/dataset_cleaner.py

Business logic for DataPilot AI's deterministic dataset cleaning
feature.

Responsibilities:
- Read a CSV file into a pandas DataFrame.
- Apply a fixed, deterministic sequence of safe cleaning operations:
  duplicate removal, empty-row removal, whitespace trimming, safe
  numeric/datetime conversion, column name normalization, and removal
  of fully-null columns.
- Save the cleaned dataset to a dedicated output directory under a
  collision-resistant filename.
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
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR: Final[str] = "data/cleaned"
_OUTPUT_SUFFIX: Final[str] = "_cleaned.csv"
_DATETIME_SUCCESS_THRESHOLD: Final[float] = 0.9
_NUMERIC_SUCCESS_THRESHOLD: Final[float] = 0.9
_MIN_NON_NULL_SAMPLE: Final[int] = 1


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

    Exposes every piece of information already computed internally by
    DatasetCleaner, so callers (e.g. Slack Block Kit builders) can
    present a rich, structured Dataset Cleaning Report without parsing
    logs or recomputing anything.

    Attributes:
        input_file: The original local CSV path that was cleaned.
        output_file: The local path where the cleaned CSV was saved.
        original_rows: Number of rows in the input dataset before
            cleaning.
        cleaned_rows: Number of rows in the output dataset after
            cleaning.
        original_columns: Number of columns in the input dataset
            before cleaning.
        cleaned_columns: Number of columns in the output dataset after
            cleaning.
        duplicates_removed: Number of duplicate rows removed.
        empty_rows_removed: Number of completely empty rows removed.
        columns_removed: Names of columns removed because they
            contained only null values (original column names, before
            renaming).
        columns_converted: Names of columns whose values were
            converted to a numeric or datetime type (post-rename
            column names).
        renamed_columns: Mapping of original column name to its
            normalized name, for every column whose name changed.
        cleaning_actions: A human-readable, ordered log of every
            cleaning action performed.
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
            True if at least one null-only column was removed.
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
    Applies a fixed sequence of safe, deterministic cleaning
    operations to a CSV dataset and saves the result to disk.

    Cleaning is performed in the following order:
        1. Remove duplicate rows.
        2. Remove rows that are completely empty.
        3. Trim leading/trailing whitespace from all string columns.
        4. Convert numeric-looking string columns to numeric types,
           where safe.
        5. Convert datetime-looking string columns to datetime types,
           where safe.
        6. Normalize column names (trim, collapse whitespace, replace
           spaces with underscores, lowercase).
        7. Remove columns that contain only null values.

    Column order is preserved throughout (aside from columns removed
    in step 7), and no value is transformed unless the transformation
    is considered safe (i.e. it does not risk silently corrupting or
    losing information).

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
            every piece of information computed during cleaning:
            original/cleaned row and column counts, duplicates removed,
            empty rows removed, columns removed/converted/renamed, and
            the ordered list of cleaning actions performed.

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

        df = self._load_csv(cleaned_path_str)
        original_rows = int(df.shape[0])
        original_columns = int(df.shape[1])

        actions: list[str] = []

        df, duplicates_removed = self._remove_duplicate_rows(df)
        actions.append(f"Removed {duplicates_removed} duplicate row(s).")

        df, empty_rows_removed = self._remove_empty_rows(df)
        actions.append(f"Removed {empty_rows_removed} completely empty row(s).")

        df = self._trim_string_columns(df)
        actions.append("Trimmed leading/trailing whitespace from string columns.")

        df, numeric_converted = self._convert_numeric_columns(df)
        if numeric_converted:
            actions.append(
                f"Converted numeric-looking column(s) to numeric type: "
                f"{', '.join(numeric_converted)}."
            )

        df, datetime_converted = self._convert_datetime_columns(df)
        if datetime_converted:
            actions.append(
                f"Converted datetime-looking column(s) to datetime type: "
                f"{', '.join(datetime_converted)}."
            )

        df, renamed_columns = self._normalize_column_names(df)
        if renamed_columns:
            actions.append(
                f"Normalized {len(renamed_columns)} column name(s)."
            )

        columns_converted = [
            renamed_columns.get(column, column)
            for column in (numeric_converted + datetime_converted)
        ]

        df, columns_removed = self._remove_null_only_columns(df)
        if columns_removed:
            actions.append(
                f"Removed {len(columns_removed)} column(s) containing only "
                f"null values: {', '.join(columns_removed)}."
            )

        cleaned_rows = int(df.shape[0])
        cleaned_columns = int(df.shape[1])

        output_path = self._build_output_path(cleaned_path_str)
        self._save_csv(df, output_path)

        result = CleaningResult(
            input_file=cleaned_path_str,
            output_file=str(output_path),
            original_rows=original_rows,
            cleaned_rows=cleaned_rows,
            original_columns=original_columns,
            cleaned_columns=cleaned_columns,
            duplicates_removed=duplicates_removed,
            empty_rows_removed=empty_rows_removed,
            columns_removed=columns_removed,
            columns_converted=columns_converted,
            renamed_columns=renamed_columns,
            cleaning_actions=actions,
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

    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame.

        Args:
            file_path: Filesystem path to the CSV file.

        Returns:
            A pandas DataFrame containing the parsed CSV data.

        Raises:
            DatasetCleanerError: If the file does not exist, is not a
                file, cannot be parsed as CSV, or contains zero rows.
        """
        path = Path(file_path)

        if not path.exists() or not path.is_file():
            logger.warning("Dataset cleaning requested for missing file: %s", path)
            raise DatasetCleanerError(f"CSV file not found: '{file_path}'.")

        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError as exc:
            logger.warning("CSV file is empty: %s", path)
            raise DatasetCleanerError(f"The CSV file '{file_path}' is empty.") from exc
        except pd.errors.ParserError as exc:
            logger.error("Failed to parse CSV file %s: %s", path, exc)
            raise DatasetCleanerError(
                f"Could not parse '{file_path}' as a valid CSV file."
            ) from exc
        except (OSError, UnicodeDecodeError) as exc:
            logger.exception("Failed to read CSV file: %s", path)
            raise DatasetCleanerError(f"Could not read the file '{file_path}'.") from exc

        if df.shape[0] == 0:
            logger.warning("CSV file contains no data rows: %s", path)
            raise DatasetCleanerError(f"The CSV file '{file_path}' contains no data rows.")

        logger.info("Cleaning started. input=%s, rows=%d, columns=%d", path, *df.shape)
        return df

    @staticmethod
    def _remove_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Remove exact duplicate rows, keeping the first occurrence.

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
        Remove rows in which every value is null.

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
        cleaned = df.copy()
        string_columns = cleaned.select_dtypes(include="object").columns

        for column in string_columns:
            cleaned[column] = cleaned[column].apply(
                lambda value: value.strip() if isinstance(value, str) else value
            )

        return cleaned

    def _convert_numeric_columns(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Convert string columns that predominantly contain numeric
        values into a numeric dtype.

        A column is only converted if at least
        _NUMERIC_SUCCESS_THRESHOLD of its non-null values can be
        parsed as numbers, ensuring the conversion is safe and does
        not silently discard meaningful non-numeric data.

        Args:
            df: The dataset to clean.

        Returns:
            A tuple of (DataFrame with eligible columns converted,
            list of original column names that were converted).
        """
        cleaned = df.copy()
        converted_columns: list[str] = []
        string_columns = cleaned.select_dtypes(include="object").columns

        for column in string_columns:
            non_null_values = cleaned[column].dropna()
            if len(non_null_values) < _MIN_NON_NULL_SAMPLE:
                continue

            parsed = pd.to_numeric(non_null_values, errors="coerce")
            success_ratio = parsed.notna().mean()

            if success_ratio >= _NUMERIC_SUCCESS_THRESHOLD:
                cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
                converted_columns.append(column)

        return cleaned, converted_columns

    def _convert_datetime_columns(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Convert string columns that predominantly contain datetime
        values into a datetime dtype.

        A column is only converted if at least
        _DATETIME_SUCCESS_THRESHOLD of its non-null values can be
        parsed as dates/timestamps. Columns already converted to
        numeric by _convert_numeric_columns are skipped, since a
        column cannot simultaneously be a safe numeric and a safe
        datetime conversion.

        Args:
            df: The dataset to clean.

        Returns:
            A tuple of (DataFrame with eligible columns converted,
            list of original column names that were converted).
        """
        cleaned = df.copy()
        converted_columns: list[str] = []
        string_columns = cleaned.select_dtypes(include="object").columns

        for column in string_columns:
            non_null_values = cleaned[column].dropna()
            if len(non_null_values) < _MIN_NON_NULL_SAMPLE:
                continue

            parsed = pd.to_datetime(non_null_values, errors="coerce")
            success_ratio = parsed.notna().mean()

            if success_ratio >= _DATETIME_SUCCESS_THRESHOLD:
                cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")
                converted_columns.append(column)

        return cleaned, converted_columns

    @staticmethod
    def _normalize_column_names(
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        """
        Normalize column names: trim surrounding whitespace, collapse
        internal whitespace, replace spaces with underscores, and
        lowercase the result.

        Example: "Customer Name" becomes "customer_name".

        Args:
            df: The dataset whose columns should be renamed.

        Returns:
            A tuple of (DataFrame with normalized column names,
            mapping of original column name to its new name, for every
            column whose name actually changed).
        """
        rename_map: dict[str, str] = {}

        for column in df.columns:
            normalized = " ".join(str(column).strip().split())
            normalized = normalized.replace(" ", "_").lower()
            if normalized != column:
                rename_map[column] = normalized

        cleaned = df.rename(columns=rename_map)
        return cleaned, rename_map

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
    def _save_csv(df: pd.DataFrame, output_path: Path) -> None:
        """
        Save a cleaned DataFrame to disk as CSV.

        Args:
            df: The cleaned dataset to save.
            output_path: The destination path.

        Raises:
            DatasetCleanerError: If the file cannot be written.
        """
        try:
            df.to_csv(output_path, index=False)
        except OSError as exc:
            logger.exception("Failed to save cleaned CSV to %s", output_path)
            raise DatasetCleanerError(
                f"Could not save the cleaned CSV to '{output_path}'."
            ) from exc