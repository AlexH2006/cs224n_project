"""
TLDR: Pytest configuration for kl_sanity_tests — registers markers.
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: integration tests requiring bitsandbytes and GPU")
