from pytest import raises


def test_hello_world():
    from fortuna import get_hello

    assert isinstance(get_hello(), str)

    with raises(TypeError):
        get_hello("Angela", "Augustus")
