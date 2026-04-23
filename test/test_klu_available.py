import pydiffsol as ds


def test_is_klu_available_is_boolean():
    assert isinstance(ds.is_klu_available(), bool)
