"""
services/file_handler.py

Business service responsible for downloading files uploaded to Slack
and safely persisting them to local disk as validated CSV files.

Responsibilities:
- Authenticate with Slack's file download mechanism using the bot
  token.
- Download the uploaded file to a temporary location.
- Validate that the download is a well-formed, non-empty CSV file
  within a configurable size limit.
- Atomically move the validated file into a dedicated uploads
  directory under a safe, collision-resistant filename.
- Return the local filesystem path of the saved file.

This module has EXACTLY ONE responsibility: getting an uploaded file
from Slack onto local disk safely. It does NOT parse CSV content, does
NOT use pandas, does NOT call any AI provider, and has NO knowledge of
services.dataset_profiler.DatasetProfiler or any other downstream
consumer. Those concerns belong to other services that accept the
local file path this module returns.
"""

import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Final

import requests

logger = logging.getLogger(__name__)

_DEFAULT_UPLOAD_DIR: Final[str] = "data/uploads"
_DEFAULT_MAX_UPLOAD_MB: Final[int] = 20
_ALLOWED_EXTENSION: Final[str] = ".csv"
_ALLOWED_MIME_TYPES: Final[frozenset[str]] = frozenset(
    {"text/csv", "application/csv", "application/vnd.ms-excel", "text/plain"}
)
_DOWNLOAD_CHUNK_SIZE_BYTES: Final[int] = 64 * 1024
_DOWNLOAD_TIMEOUT_SECONDS: Final[float] = 30.0
_CONTENT_SNIFF_BYTES: Final[int] = 4096
_SAFE_FILENAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_.-]+")


class SlackFileHandlerError(Exception):
    """
    Raised whenever SlackFileHandler cannot successfully download,
    validate, or save an uploaded file.

    Callers only need to catch this single exception type; they do
    not need to know about requests' or the filesystem's internal
    exception hierarchies.
    """


class SlackFileHandler:
    """
    Downloads files uploaded to Slack and saves them locally as
    validated CSV files.

    This class is intentionally narrow in scope (Single Responsibility
    Principle): it authenticates with Slack, downloads bytes, validates
    them as a well-formed CSV within size limits, and writes them to a
    dedicated uploads directory under a safe filename. It has no
    knowledge of what happens to the file afterward — it does not
    import pandas, does not call any AIProvider, and does not know
    about DatasetProfiler or any other consumer.

    Configuration is read from environment variables:
        SLACK_BOT_TOKEN: Required. Used to authenticate the download
            request against Slack's private file URLs.
        MAX_UPLOAD_MB: Optional. Maximum allowed file size in
            megabytes. Defaults to 20.
    """

    def __init__(
        self,
        bot_token: str | None = None,
        upload_dir: str | None = None,
        max_upload_mb: int | None = None,
    ) -> None:
        """
        Initialize the file handler.

        Args:
            bot_token: Slack bot token used to authenticate file
                downloads. Falls back to the SLACK_BOT_TOKEN
                environment variable if not provided.
            upload_dir: Directory in which downloaded files are saved.
                Falls back to "data/uploads" if not provided. Created
                automatically if it does not exist.
            max_upload_mb: Maximum allowed upload size in megabytes.
                Falls back to the MAX_UPLOAD_MB environment variable,
                then to 20 MB.

        Raises:
            ValueError: If no bot token is available from either the
                argument or the environment.
            SlackFileHandlerError: If the upload directory cannot be
                created.
        """
        resolved_token = bot_token or os.getenv("SLACK_BOT_TOKEN")
        if not resolved_token:
            raise ValueError(
                "Slack bot token not found. Set SLACK_BOT_TOKEN in your "
                "environment or pass bot_token explicitly."
            )
        self._bot_token: str = resolved_token

        resolved_max_mb = (
            max_upload_mb
            if max_upload_mb is not None
            else int(os.getenv("MAX_UPLOAD_MB", _DEFAULT_MAX_UPLOAD_MB))
        )
        self._max_upload_bytes: int = resolved_max_mb * 1024 * 1024

        self._upload_dir: Path = Path(upload_dir or _DEFAULT_UPLOAD_DIR)
        try:
            self._upload_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.exception(
                "Failed to create upload directory: %s", self._upload_dir
            )
            raise SlackFileHandlerError(
                f"Could not create upload directory '{self._upload_dir}'."
            ) from exc

        logger.debug(
            "SlackFileHandler initialized. upload_dir=%s, max_upload_mb=%d",
            self._upload_dir,
            resolved_max_mb,
        )

    def download_csv(self, download_url: str, filename: str) -> str:
        """
        Download a CSV file from a private Slack file URL and save it
        locally.

        Args:
            download_url: Slack's private file download URL (e.g. a
                file's `url_private_download` field from the Slack
                API).
            filename: The original filename as reported by Slack,
                used only to derive the file extension and build a
                safe local filename. Never used as-is for the final
                path.

        Returns:
            The local filesystem path (as a string) of the saved,
            validated CSV file.

        Raises:
            SlackFileHandlerError: If the filename or URL is invalid,
                the file is not a CSV, the download fails for any
                network reason (authentication, timeout, connection
                error, non-success status code), the file exceeds the
                configured size limit, or the downloaded file is empty
                or does not appear to be valid text/CSV content.
        """
        cleaned_url = download_url.strip()
        cleaned_filename = filename.strip()

        if not cleaned_url:
            raise SlackFileHandlerError("No download URL was provided.")
        if not cleaned_filename:
            raise SlackFileHandlerError("No filename was provided.")

        self._validate_extension(cleaned_filename)

        safe_filename = self._build_safe_filename(cleaned_filename)
        final_path = self._upload_dir / safe_filename
        temp_path = self._upload_dir / f"{safe_filename}.part"

        logger.info("Download started. filename=%s", cleaned_filename)

        try:
            self._stream_download(cleaned_url, temp_path)
            self._validate_downloaded_file(temp_path)
            temp_path.replace(final_path)
        except SlackFileHandlerError:
            self._cleanup_partial_file(temp_path)
            raise
        except Exception as exc:  # noqa: BLE001 - final safety net, always logged
            logger.exception(
                "Unexpected error while downloading file. filename=%s",
                cleaned_filename,
            )
            self._cleanup_partial_file(temp_path)
            raise SlackFileHandlerError(
                "An unexpected error occurred while downloading the file."
            ) from exc

        logger.info(
            "Download completed. filename=%s, saved_to=%s",
            cleaned_filename,
            final_path,
        )
        return str(final_path)

    def _validate_extension(self, filename: str) -> None:
        """
        Validate that a filename has an allowed CSV extension.

        Args:
            filename: The original filename reported by Slack.

        Raises:
            SlackFileHandlerError: If the filename's extension is not
                ".csv" (case-insensitive).
        """
        extension = Path(filename).suffix.lower()
        if extension != _ALLOWED_EXTENSION:
            logger.warning(
                "Rejected upload with disallowed extension. filename=%s, extension=%s",
                filename,
                extension,
            )
            raise SlackFileHandlerError(
                f"Only .csv files are supported. Received a file with "
                f"extension '{extension or '(none)'}'."
            )

    def _build_safe_filename(self, filename: str) -> str:
        """
        Build a collision-resistant, filesystem-safe local filename
        from an untrusted, user-supplied filename.

        Uses only the base name (via pathlib), strips any directory
        components to prevent path traversal, replaces disallowed
        characters, and prefixes the result with a timestamp so
        repeated uploads of the same filename do not collide. If a
        timestamp collision does occur, a short random suffix is
        appended to guarantee uniqueness.

        Args:
            filename: The original filename reported by Slack (e.g.
                "sales.csv").

        Returns:
            A safe filename such as "20260708_104532_sales.csv", or
            "20260708_104532_sales_a1b2c3d4.csv" if a collision was
            detected.
        """
        base_name = Path(filename).name
        sanitized_stem = _SAFE_FILENAME_PATTERN.sub("_", Path(base_name).stem).strip("_")
        if not sanitized_stem:
            sanitized_stem = "upload"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{sanitized_stem}{_ALLOWED_EXTENSION}"

        if (self._upload_dir / safe_filename).exists():
            unique_suffix = uuid.uuid4().hex[:8]
            safe_filename = (
                f"{timestamp}_{sanitized_stem}_{unique_suffix}{_ALLOWED_EXTENSION}"
            )

        return safe_filename

    def _stream_download(self, download_url: str, temp_path: Path) -> None:
        """
        Stream-download a file from Slack to a temporary path,
        authenticating with the bot token and enforcing the
        configured maximum upload size while writing.

        Args:
            download_url: Slack's private file download URL.
            temp_path: Local temporary path to write the download to.

        Raises:
            SlackFileHandlerError: If authentication fails (401/403),
                the file is not found (404), any other non-success
                status code is returned, the request times out, a
                connection error occurs, or the downloaded content
                exceeds the configured maximum size.
        """
        headers = {"Authorization": f"Bearer {self._bot_token}"}

        try:
            with requests.get(
                download_url,
                headers=headers,
                stream=True,
                timeout=_DOWNLOAD_TIMEOUT_SECONDS,
            ) as response:
                self._handle_response_status(response.status_code)
                self._check_declared_content_length(response.headers.get("Content-Length"))

                bytes_written = 0
                with open(temp_path, "wb") as temp_file:
                    for chunk in response.iter_content(
                        chunk_size=_DOWNLOAD_CHUNK_SIZE_BYTES
                    ):
                        if not chunk:
                            continue
                        bytes_written += len(chunk)
                        if bytes_written > self._max_upload_bytes:
                            logger.warning(
                                "Upload exceeded maximum size while streaming. "
                                "max_bytes=%d",
                                self._max_upload_bytes,
                            )
                            raise SlackFileHandlerError(
                                "The uploaded file exceeds the maximum allowed "
                                f"size of {self._max_upload_bytes // (1024 * 1024)} MB."
                            )
                        temp_file.write(chunk)
        except requests.exceptions.Timeout as exc:
            logger.error("Download timed out. url=%s", self._redact_url(download_url))
            raise SlackFileHandlerError(
                "The file download timed out. Please try again."
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            logger.error(
                "Connection error during download. url=%s",
                self._redact_url(download_url),
            )
            raise SlackFileHandlerError(
                "Could not connect to Slack to download the file."
            ) from exc
        except requests.exceptions.RequestException as exc:
            logger.exception(
                "Unexpected network error during download. url=%s",
                self._redact_url(download_url),
            )
            raise SlackFileHandlerError(
                "An unexpected network error occurred while downloading the file."
            ) from exc
        except OSError as exc:
            logger.exception("Failed to write downloaded file to %s", temp_path)
            raise SlackFileHandlerError(
                "Could not write the downloaded file to disk."
            ) from exc

    @staticmethod
    def _handle_response_status(status_code: int) -> None:
        """
        Translate an HTTP response status code from Slack into a
        SlackFileHandlerError where appropriate.

        Args:
            status_code: The HTTP status code returned by Slack.

        Raises:
            SlackFileHandlerError: If the status code indicates
                authentication failure, missing access, a missing
                file, or any other non-success response.
        """
        if status_code == 200:
            return
        if status_code == 401:
            raise SlackFileHandlerError(
                "Authentication with Slack failed while downloading the file. "
                "Please check the bot token configuration."
            )
        if status_code == 403:
            raise SlackFileHandlerError(
                "Access to this file was denied by Slack."
            )
        if status_code == 404:
            raise SlackFileHandlerError(
                "The requested file could not be found on Slack."
            )
        raise SlackFileHandlerError(
            f"Slack returned an unexpected status code ({status_code}) "
            "while downloading the file."
        )

    def _check_declared_content_length(self, content_length_header: str | None) -> None:
        """
        Validate the declared Content-Length header, if present,
        against the configured maximum upload size before streaming
        begins.

        Args:
            content_length_header: The raw Content-Length header value,
                or None if not provided by the server.

        Raises:
            SlackFileHandlerError: If the declared content length
                exceeds the configured maximum upload size.
        """
        if content_length_header is None:
            return

        try:
            declared_length = int(content_length_header)
        except ValueError:
            return

        if declared_length > self._max_upload_bytes:
            logger.warning(
                "Rejected upload based on declared Content-Length. "
                "declared_bytes=%d, max_bytes=%d",
                declared_length,
                self._max_upload_bytes,
            )
            raise SlackFileHandlerError(
                "The uploaded file exceeds the maximum allowed size of "
                f"{self._max_upload_bytes // (1024 * 1024)} MB."
            )

    def _validate_downloaded_file(self, temp_path: Path) -> None:
        """
        Validate a fully-downloaded temporary file before it is moved
        into the uploads directory.

        Checks that the file is non-empty and that its content appears
        to be decodable text (a lightweight sniff test), without
        performing any CSV parsing. Full structural CSV validation is
        intentionally out of scope for this service.

        Args:
            temp_path: Path to the downloaded temporary file.

        Raises:
            SlackFileHandlerError: If the file is empty, exceeds the
                configured maximum size on disk, or does not appear to
                be valid text content.
        """
        try:
            file_size = temp_path.stat().st_size
        except OSError as exc:
            logger.exception("Failed to stat downloaded file: %s", temp_path)
            raise SlackFileHandlerError(
                "Could not verify the downloaded file on disk."
            ) from exc

        if file_size == 0:
            logger.warning("Downloaded file is empty: %s", temp_path)
            raise SlackFileHandlerError("The downloaded file is empty.")

        if file_size > self._max_upload_bytes:
            logger.warning(
                "Downloaded file exceeds maximum size on disk. size_bytes=%d, max_bytes=%d",
                file_size,
                self._max_upload_bytes,
            )
            raise SlackFileHandlerError(
                "The uploaded file exceeds the maximum allowed size of "
                f"{self._max_upload_bytes // (1024 * 1024)} MB."
            )

        try:
            with open(temp_path, "rb") as downloaded_file:
                sample = downloaded_file.read(_CONTENT_SNIFF_BYTES)
        except OSError as exc:
            logger.exception("Failed to read downloaded file for validation: %s", temp_path)
            raise SlackFileHandlerError(
                "Could not read the downloaded file for validation."
            ) from exc

        try:
            sample.decode("utf-8")
        except UnicodeDecodeError as exc:
            logger.warning(
                "Downloaded file does not appear to be valid text/CSV content: %s",
                temp_path,
            )
            raise SlackFileHandlerError(
                "The uploaded file does not appear to be a valid CSV file."
            ) from exc

    @staticmethod
    def _cleanup_partial_file(temp_path: Path) -> None:
        """
        Remove a partially-downloaded or invalid temporary file, if it
        exists, to ensure no partial downloads are ever left on disk.

        Args:
            temp_path: Path to the temporary file to remove.
        """
        try:
            if temp_path.exists():
                temp_path.unlink()
                logger.debug("Cleaned up partial file: %s", temp_path)
        except OSError:
            logger.exception("Failed to clean up partial file: %s", temp_path)

    @staticmethod
    def _redact_url(url: str) -> str:
        """
        Produce a log-safe representation of a Slack file URL,
        truncating it to avoid logging overly long or sensitive query
        strings while still providing useful debugging context.

        Args:
            url: The full download URL.

        Returns:
            A truncated version of the URL safe for logging.
        """
        max_length = 100
        return url if len(url) <= max_length else f"{url[:max_length]}..."