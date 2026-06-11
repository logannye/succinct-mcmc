def test_import_rewind_and_version():
    import rewind
    assert isinstance(rewind.__version__, str)
    assert rewind.__version__
