"""Test the core pytsg API"""

import importlib
from importlib.metadata import PackageNotFoundError

import pytsg


def test_pytsg_version():
    assert pytsg.__version__ is not None


def test_pytsg_version_when_package_is_not_installed(monkeypatch):
    def missing_version(_package_name):
        raise PackageNotFoundError

    monkeypatch.setattr("importlib.metadata.version", missing_version)

    reloaded_pytsg = importlib.reload(pytsg)

    assert reloaded_pytsg.__version__ == "0.0.0"
