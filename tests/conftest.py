"""
tests/conftest.py

Shared pytest configuration and fixtures available to all test files.
pytest automatically loads this file before running any tests.
"""

import pytest


def pytest_configure(config):
    """Register custom markers so pytest doesn't warn about unknown markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m not slow')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests that require the full pipeline"
    )