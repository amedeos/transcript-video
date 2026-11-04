"""Shared test fixtures."""

from __future__ import annotations

import os

import pytest

_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")


@pytest.fixture(autouse=True)
def _no_thread_env():
    """OMP_NUM_THREADS / MKL_NUM_THREADS count as an explicit --threads.

    Clear them so the thread-policy tests don't depend on the environment
    running them (CI images often set OMP_NUM_THREADS). Tests that need a
    value set it with monkeypatch.setenv.

    Deliberately not built on ``monkeypatch``: requesting it from an autouse
    conftest fixture changes the teardown order of per-module autouse
    fixtures (test_asr's cache clearing would run while its patches are
    still applied).
    """
    saved = {var: os.environ.pop(var) for var in _THREAD_ENV_VARS if var in os.environ}
    yield
    for var in _THREAD_ENV_VARS:
        os.environ.pop(var, None)
    os.environ.update(saved)
