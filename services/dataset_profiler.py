"""
services/dataset_profiler.py

Business logic for DataPilot AI's Dataset Intelligence feature.

Responsibilities:
- Read a CSV file into a pandas DataFrame.
- Compute structural and statistical metadata about the dataset
  (shape, column types, missing values, duplicates, memory usage,
  numeric and categorical statistics).
- Calculate a simple, rule-based Data Quality Score (0-10).
- Generate rule-based cleaning recommendations.
- Send a compact summary of all of the above to an injected AIProvider,
  which acts as a senior data engineer and produces a structured,
  plain-text intelligence report.
- Return all of the above — structured metadata plus the AI report —
  as a single, rich result object, so callers (e.g. Slack UI builders)
  never need to parse or re-derive information that has already been
  computed.

This module has NO knowledge of Slack and NO knowledge of any concrete
AI vendor (Gemini, Groq, OpenAI, Claude, etc.). It depends only on the
AIProvider abstraction, supplied via constructor injection, satisfying
the Dependency Inversion Principle exactly like SQLAnalyzer,
SQLCleaner, SQLGenerator, SQLOptimizer, and SQLValidator.

This is NOT a SQL feature. It is a dataset-level analysis feature: it
profiles CSV files rather than SQL queries.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pandas as pd

from services.ai import AIProvider, AIProviderError

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT: Final[str] = (
    "You are a senior data engineer performing a dataset intelligence "
    "review. You will be given a compact structural and statistical "
    "summary of a CSV dataset — not the raw data itself. Based only "
    "on this summary, produce a clear, actionable report for a data "
    "team. Return plain text only: no markdown, no JSON, no code "
    "fences, no greetings, no commentary about being an AI. Structure "
    "your entire response using exactly these section headers, in "
    "this order:\n\n"
    "Dataset Overview\n"
    "<a concise description of what this dataset appears to represent, "
    "its size, and its overall shape>\n\n"
    "Key Insights\n"
    "<the most important observations about the data, patterns, or "
    "notable statistics>\n\n"
    "Data Quality Issues\n"
    "<specific problems found: missing values, duplicates, "
    "inconsistent types, and anything else the summary reveals>\n\n"
    "Cleaning Recommendations\n"
    "<concrete, prioritized steps to clean and prepare this dataset>\n\n"
    "Suggested Visualizations\n"
    "<specific chart types and which columns they should use, and why "
    "they would be useful for this dataset>\n\n"
    "Suggested SQL Analyses\n"
    "<specific SQL queries or types of analysis that would extract "
    "value from this dataset once it is loaded into a database>"
)

_MAX_TOP_CATEGORICAL_VALUES: Final[int] = 5
_QUALITY_SCORE_MAX: Final[float] = 10.0
_QUALITY_SCORE_MIN: Final[float] = 0.0


class DatasetProfilerError(Exception):
    """
    Raised whenever the dataset profiling service cannot produce a
    result.

    Callers (e.g. Slack event handlers in app.py) only need to catch
    this single exception type; they do not need to know about
    pandas' parsing exceptions, filesystem errors, or AIProviderError.
    """


@dataclass
class DatasetProfileResult:
    """
    Structured result of a single dataset profiling operation.

    Exposes every piece of metadata already computed internally by
    DatasetProfiler, alongside the final AI-generated report, so
    callers (e.g. Slack Block Kit builders) can present rich,
    structured UI without parsing or re-deriving values from the
    report text.

    Attributes:
        filename: The original CSV filename that was profiled.
        rows: Number of rows in the dataset.
        columns: Number of columns in the dataset.
        column_names: Ordered list of column names.
        data_types: Mapping of column name to its pandas dtype, as a
            string (e.g. "int64", "object").
        missing_values_by_column: Mapping of column name to its
            missing value count, for columns with at least one
            missing value.
        total_missing_values: Total number of missing values across
            the entire dataset.
        duplicate_rows: Number of duplicate rows detected.
        memory_usage_bytes: Total memory usage of the DataFrame, in
            bytes.
        numeric_statistics: Mapping of numeric column name to its
            descriptive statistics (count, mean, std, min, quartiles,
            max).
        categorical_statistics: Mapping of categorical column name to
            its unique value count and top value frequencies.
        quality_score: The computed Data Quality Score, from 0.0 to
            10.0.
        cleaning_recommendations: Locally generated, rule-based
            cleaning recommendation strings.
        ai_report: The full plain-text report generated by the AI
            provider (Dataset Overview, Key Insights, Data Quality
            Issues, Cleaning Recommendations, Suggested
            Visualizations, Suggested SQL Analyses).
    """

    filename: str
    rows: int
    columns: int
    column_names: list[str] = field(default_factory=list)
    data_types: dict[str, str] = field(default_factory=dict)
    missing_values_by_column: dict[str, int] = field(default_factory=dict)
    total_missing_values: int = 0
    duplicate_rows: int = 0
    memory_usage_bytes: int = 0
    numeric_statistics: dict[str, dict[str, float]] = field(default_factory=dict)
    categorical_statistics: dict[str, dict[str, Any]] = field(default_factory=dict)
    quality_score: float = 0.0
    cleaning_recommendations: list[str] = field(default_factory=list)
    ai_report: str = ""


class DatasetProfiler:
    """
    Encapsulates the business logic needed to profile a CSV dataset
    and produce an AI-generated data intelligence report.

    The actual text generation is delegated to an injected AIProvider
    implementation, so this class is completely decoupled from any
    specific AI vendor. It can be unit-tested with a mock/fake
    AIProvider, and reused across interfaces (Slack, CLI, HTTP API)
    without modification.

    Structural metadata extraction, quality scoring, and cleaning
    recommendations are all computed locally via pandas, deterministic
    rules, and never sent to the AI as raw row data — only a compact
    summary is sent, keeping prompts small and predictable regardless
    of dataset size.
    """

    def __init__(self, provider: AIProvider) -> None:
        """
        Initialize the profiler with an injected AI provider.

        Args:
            provider: A concrete AIProvider implementation (e.g.
                GeminiProvider, GroqProvider) responsible for actual
                text generation. This profiler never constructs or
                imports a concrete provider itself.
        """
        self._provider: AIProvider = provider

        logger.debug(
            "DatasetProfiler initialized with provider=%s (vendor=%s, model=%s)",
            getattr(provider, "name", provider.__class__.__name__),
            getattr(provider, "vendor", "unknown"),
            getattr(provider, "model", "unknown"),
        )

    def profile_dataset(self, file_path: str) -> DatasetProfileResult:
        """
        Profile a CSV dataset and generate an AI-powered intelligence
        report.

        Args:
            file_path: Filesystem path to the CSV file to profile.

        Returns:
            A DatasetProfileResult containing every computed metadata
            field (rows, columns, data types, missing values,
            duplicates, memory usage, numeric/categorical statistics,
            quality score, cleaning recommendations) alongside the
            full plain-text AI report (Dataset Overview, Key Insights,
            Data Quality Issues, Cleaning Recommendations, Suggested
            Visualizations, Suggested SQL Analyses).

        Raises:
            DatasetProfilerError: If the file path is empty, the file
                does not exist, the file cannot be parsed as CSV, the
                resulting dataset is empty, or the underlying AI
                provider fails to generate a valid report.
        """
        cleaned_path = file_path.strip()
        if not cleaned_path:
            logger.warning("Dataset profiling requested with an empty file path.")
            raise DatasetProfilerError("No CSV file path was provided to profile.")

        dataframe = self._load_csv(cleaned_path)
        filename = Path(cleaned_path).name

        metadata = self._extract_metadata(dataframe, filename)
        quality_score = self._calculate_quality_score(metadata)
        recommendations = self._generate_recommendations(dataframe, metadata)

        logger.info(
            "Profiled dataset via provider=%s. filename=%s, rows=%d, columns=%d, "
            "quality_score=%.1f",
            getattr(self._provider, "name", self._provider.__class__.__name__),
            filename,
            metadata["rows"],
            metadata["columns"],
            quality_score,
        )

        user_prompt = self._build_user_prompt(metadata, quality_score, recommendations)

        try:
            report = self._provider.generate(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except AIProviderError as exc:
            logger.error("AI provider failed to generate dataset report: %s", exc)
            raise DatasetProfilerError(
                "The AI service was unable to analyze this dataset. "
                "Please try again shortly."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - final safety net, always logged
            logger.exception(
                "Unexpected error while generating dataset report via provider."
            )
            raise DatasetProfilerError(
                "An unexpected error occurred while analyzing the dataset."
            ) from exc

        cleaned_report = report.strip()
        if not cleaned_report:
            logger.error("AI provider returned an empty dataset report.")
            raise DatasetProfilerError(
                "The AI service returned an empty dataset report."
            )

        logger.info(
            "Dataset profiling completed successfully. report_length=%d chars",
            len(cleaned_report),
        )

        return DatasetProfileResult(
            filename=metadata["filename"],
            rows=metadata["rows"],
            columns=metadata["columns"],
            column_names=metadata["column_names"],
            data_types=metadata["data_types"],
            missing_values_by_column=metadata["missing_values_by_column"],
            total_missing_values=metadata["total_missing_values"],
            duplicate_rows=metadata["duplicate_rows"],
            memory_usage_bytes=metadata["memory_usage_bytes"],
            numeric_statistics=metadata["numeric_statistics"],
            categorical_statistics=metadata["categorical_statistics"],
            quality_score=quality_score,
            cleaning_recommendations=recommendations,
            ai_report=cleaned_report,
        )

    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame.

        Args:
            file_path: Filesystem path to the CSV file.

        Returns:
            A pandas DataFrame containing the parsed CSV data.

        Raises:
            DatasetProfilerError: If the file does not exist, is not a
                file, cannot be parsed as CSV, or contains zero rows.
        """
        path = Path(file_path)

        if not path.exists() or not path.is_file():
            logger.warning("Dataset profiling requested for missing file: %s", path)
            raise DatasetProfilerError(f"CSV file not found: '{file_path}'.")

        try:
            dataframe = pd.read_csv(path)
        except pd.errors.EmptyDataError as exc:
            logger.warning("CSV file is empty: %s", path)
            raise DatasetProfilerError(
                f"The CSV file '{file_path}' is empty."
            ) from exc
        except pd.errors.ParserError as exc:
            logger.error("Failed to parse CSV file %s: %s", path, exc)
            raise DatasetProfilerError(
                f"Could not parse '{file_path}' as a valid CSV file."
            ) from exc
        except (OSError, UnicodeDecodeError) as exc:
            logger.exception("Failed to read CSV file: %s", path)
            raise DatasetProfilerError(
                f"Could not read the file '{file_path}'."
            ) from exc

        if dataframe.shape[0] == 0:
            logger.warning("CSV file contains no data rows: %s", path)
            raise DatasetProfilerError(
                f"The CSV file '{file_path}' contains no data rows."
            )

        logger.debug(
            "Loaded CSV file %s. rows=%d, columns=%d", path, *dataframe.shape
        )
        return dataframe

    def _extract_metadata(self, df: pd.DataFrame, filename: str) -> dict[str, Any]:
        """
        Extract structural and statistical metadata from a DataFrame.

        Args:
            df: The dataset to analyze.
            filename: The original filename, included for reporting.

        Returns:
            A dictionary containing filename, rows, columns, column
            names, data types, missing value counts, duplicate row
            count, memory usage in bytes, numeric statistics, and
            categorical statistics.
        """
        missing_by_column = df.isnull().sum()
        total_missing = int(missing_by_column.sum())
        duplicate_rows = int(df.duplicated().sum())
        memory_usage_bytes = int(df.memory_usage(deep=True).sum())

        numeric_df = df.select_dtypes(include="number")
        numeric_statistics: dict[str, dict[str, float]] = {}
        if not numeric_df.empty:
            described = numeric_df.describe().to_dict()
            numeric_statistics = {
                column: {
                    stat_name: float(stat_value)
                    for stat_name, stat_value in stats.items()
                }
                for column, stats in described.items()
            }

        categorical_df = df.select_dtypes(include=["object", "category"])
        categorical_statistics: dict[str, dict[str, Any]] = {}
        for column in categorical_df.columns:
            value_counts = categorical_df[column].value_counts().head(
                _MAX_TOP_CATEGORICAL_VALUES
            )
            categorical_statistics[column] = {
                "unique_values": int(categorical_df[column].nunique()),
                "top_values": {
                    str(value): int(count)
                    for value, count in value_counts.items()
                },
            }

        metadata: dict[str, Any] = {
            "filename": filename,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns),
            "data_types": {column: str(dtype) for column, dtype in df.dtypes.items()},
            "missing_values_by_column": {
                column: int(count)
                for column, count in missing_by_column.items()
                if count > 0
            },
            "total_missing_values": total_missing,
            "duplicate_rows": duplicate_rows,
            "memory_usage_bytes": memory_usage_bytes,
            "numeric_statistics": numeric_statistics,
            "categorical_statistics": categorical_statistics,
        }

        logger.debug(
            "Extracted metadata for %s: rows=%d, columns=%d, missing=%d, duplicates=%d",
            filename,
            metadata["rows"],
            metadata["columns"],
            total_missing,
            duplicate_rows,
        )
        return metadata

    def _calculate_quality_score(self, metadata: dict[str, Any]) -> float:
        """
        Calculate a simple, rule-based Data Quality Score from 0 to 10.

        The score starts at a perfect 10.0 and is penalized based on:
        - The proportion of missing values across the entire dataset.
        - The proportion of duplicate rows.
        - The presence of columns with inconsistent/mixed types
          (object columns containing a mix of numeric-looking and
          non-numeric values).

        Args:
            metadata: The metadata dictionary produced by
                _extract_metadata.

        Returns:
            A quality score between 0.0 and 10.0, rounded to one
            decimal place. Higher is better.
        """
        total_cells = metadata["rows"] * metadata["columns"]
        missing_ratio = (
            metadata["total_missing_values"] / total_cells if total_cells > 0 else 0.0
        )
        duplicate_ratio = (
            metadata["duplicate_rows"] / metadata["rows"]
            if metadata["rows"] > 0
            else 0.0
        )

        inconsistent_type_columns = sum(
            1
            for stats in metadata["categorical_statistics"].values()
            if stats["unique_values"] > 0
            and self._looks_type_inconsistent(stats)
        )
        inconsistent_type_ratio = (
            inconsistent_type_columns / metadata["columns"]
            if metadata["columns"] > 0
            else 0.0
        )

        score = _QUALITY_SCORE_MAX
        score -= missing_ratio * 4.0
        score -= duplicate_ratio * 3.0
        score -= inconsistent_type_ratio * 3.0

        score = max(_QUALITY_SCORE_MIN, min(_QUALITY_SCORE_MAX, score))
        return round(score, 1)

    @staticmethod
    def _looks_type_inconsistent(categorical_stats: dict[str, Any]) -> bool:
        """
        Heuristically determine whether a categorical column's top
        values suggest mixed/inconsistent data types (e.g. a column
        containing both numeric strings and free text).

        Args:
            categorical_stats: The per-column statistics dictionary
                produced by _extract_metadata for a single column.

        Returns:
            True if the column's sampled top values contain a mix of
            numeric-looking and non-numeric-looking strings.
        """
        top_values = list(categorical_stats["top_values"].keys())
        if len(top_values) < 2:
            return False

        numeric_like_count = sum(
            1 for value in top_values if _is_numeric_like(value)
        )
        return 0 < numeric_like_count < len(top_values)

    def _generate_recommendations(
        self, df: pd.DataFrame, metadata: dict[str, Any]
    ) -> list[str]:
        """
        Generate rule-based cleaning recommendations for a dataset.

        Args:
            df: The dataset being analyzed.
            metadata: The metadata dictionary produced by
                _extract_metadata.

        Returns:
            A list of human-readable cleaning recommendation strings.
            May be empty if no issues were detected.
        """
        recommendations: list[str] = []

        if metadata["duplicate_rows"] > 0:
            recommendations.append(
                f"Remove {metadata['duplicate_rows']} duplicate row(s)."
            )

        for column, missing_count in metadata["missing_values_by_column"].items():
            recommendations.append(
                f"Fill or impute {missing_count} missing value(s) in column "
                f"'{column}'."
            )

        object_columns = df.select_dtypes(include="object").columns
        for column in object_columns:
            sample = df[column].dropna().astype(str).head(50)

            if sample.empty:
                continue

            if self._column_looks_datetime(sample):
                recommendations.append(
                    f"Convert column '{column}' to a proper datetime type."
                )

            if self._column_has_whitespace_issues(sample):
                recommendations.append(
                    f"Trim leading/trailing whitespace in column '{column}'."
                )

            if self._column_looks_numeric_string(sample):
                recommendations.append(
                    f"Convert column '{column}' from text to a numeric type."
                )

        logger.debug(
            "Generated %d cleaning recommendation(s) for dataset.",
            len(recommendations),
        )
        return recommendations

    @staticmethod
    def _column_looks_datetime(sample: pd.Series) -> bool:
        """
        Heuristically determine whether a sample of string values
        appears to represent dates or timestamps.

        Args:
            sample: A non-null sample of string values from a column.

        Returns:
            True if the sample can be parsed as datetimes with a low
            failure rate.
        """
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
        except (ValueError, TypeError):
            return False

        success_ratio = parsed.notna().mean()
        return bool(success_ratio >= 0.9)

    @staticmethod
    def _column_has_whitespace_issues(sample: pd.Series) -> bool:
        """
        Determine whether any values in a sample have leading or
        trailing whitespace.

        Args:
            sample: A non-null sample of string values from a column.

        Returns:
            True if at least one value differs from its stripped form.
        """
        return bool((sample != sample.str.strip()).any())

    @staticmethod
    def _column_looks_numeric_string(sample: pd.Series) -> bool:
        """
        Heuristically determine whether a sample of string values
        actually represents numbers stored as text.

        Args:
            sample: A non-null sample of string values from a column.

        Returns:
            True if the sample can be converted to numeric values with
            a low failure rate.
        """
        try:
            parsed = pd.to_numeric(sample, errors="coerce")
        except (ValueError, TypeError):
            return False

        success_ratio = parsed.notna().mean()
        return bool(success_ratio >= 0.9)

    def _build_user_prompt(
        self,
        metadata: dict[str, Any],
        quality_score: float,
        recommendations: list[str],
    ) -> str:
        """
        Build a compact, human-readable prompt summarizing dataset
        metadata for the AI provider.

        Args:
            metadata: The metadata dictionary produced by
                _extract_metadata.
            quality_score: The computed Data Quality Score (0-10).
            recommendations: The rule-based cleaning recommendations
                generated locally.

        Returns:
            A plain-text prompt string summarizing the dataset,
            suitable for sending to an AIProvider.
        """
        lines: list[str] = [
            f"Filename: {metadata['filename']}",
            f"Rows: {metadata['rows']}",
            f"Columns: {metadata['columns']}",
            f"Column names: {', '.join(metadata['column_names'])}",
            f"Data types: {metadata['data_types']}",
            f"Total missing values: {metadata['total_missing_values']}",
            f"Missing values by column: {metadata['missing_values_by_column']}",
            f"Duplicate rows: {metadata['duplicate_rows']}",
            f"Memory usage (bytes): {metadata['memory_usage_bytes']}",
            f"Numeric statistics: {metadata['numeric_statistics']}",
            f"Categorical statistics: {metadata['categorical_statistics']}",
            f"Computed Data Quality Score (0-10): {quality_score}",
            f"Locally generated cleaning recommendations: {recommendations}",
        ]
        return "Analyze this dataset summary:\n\n" + "\n".join(lines)


def _is_numeric_like(value: str) -> bool:
    """
    Determine whether a single string value looks like a number.

    Args:
        value: The string value to check.

    Returns:
        True if the value can be parsed as a float.
    """
    try:
        float(value)
        return True
    except ValueError:
        return False