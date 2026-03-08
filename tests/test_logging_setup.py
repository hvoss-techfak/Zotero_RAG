"""Tests for logging setup."""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zoterorag.logging_setup import setup_logging, _NOISY_LOGGERS, parse_log_level


class TestLoggingSetup:
    """Tests for logging setup module."""

    def test_noisy_loggers_tuple_defined(self):
        assert isinstance(_NOISY_LOGGERS, tuple)
        assert "httpx" in _NOISY_LOGGERS
        assert "requests" in _NOISY_LOGGERS
        assert "ollama" in _NOISY_LOGGERS

    def test_parse_log_level_handles_strings(self):
        assert parse_log_level("warning") == logging.WARNING
        assert parse_log_level("ERROR") == logging.ERROR

    def test_setup_logging_quiet_http_default(self):
        setup_logging(level="WARNING")
        for name in _NOISY_LOGGERS:
            assert logging.getLogger(name).level == logging.WARNING

    def test_setup_logging_quiet_http_false(self):
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.DEBUG)

        setup_logging(level="WARNING", quiet_http=False)

        for name in _NOISY_LOGGERS:
            assert logging.getLogger(name).level == logging.DEBUG

    def test_setup_logging_custom_level(self):
        setup_logging(level="ERROR", quiet_http=True, quiet_http_level="ERROR")
        assert logging.getLogger().level == logging.ERROR
        for name in _NOISY_LOGGERS:
            assert logging.getLogger(name).level == logging.ERROR

    def test_setup_logging_sets_framework_loggers(self):
        setup_logging(level="WARNING")
        assert logging.getLogger("mcp").level == logging.WARNING
        assert logging.getLogger("anyio").level == logging.WARNING


class TestLoggingSetupEdgeCases:
    def test_setup_logging_idempotent(self):
        setup_logging(level="WARNING")
        setup_logging(level="WARNING")
        setup_logging(level="WARNING")

    def test_setup_logging_configures_root_logger(self):
        setup_logging(level="ERROR")
        assert logging.getLogger().level == logging.ERROR

    def test_noisy_loggers_includes_common_http_clients(self):
        expected = ["httpx", "httpcore", "urllib3", "requests"]
        for name in expected:
            assert name in _NOISY_LOGGERS
