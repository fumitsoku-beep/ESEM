import psysem


def test_package_exposes_version() -> None:
    assert isinstance(psysem.__version__, str)
    assert psysem.__version__
