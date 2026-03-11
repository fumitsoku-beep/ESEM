import pytest

from psysem import build_parameter_index_map


def test_build_parameter_index_map_orders_by_parameter_index() -> None:
    table = (
        {"is_free": True, "parameter": "b2", "parameter_index": 2},
        {"is_free": False, "parameter": None, "parameter_index": None},
        {"is_free": True, "parameter": "b1", "parameter_index": 1},
    )
    mapping = build_parameter_index_map(table)
    assert mapping.n_free == 2
    assert [entry.parameter_index for entry in mapping.entries] == [1, 2]
    assert [entry.vector_position for entry in mapping.entries] == [0, 1]
    assert mapping.index_to_position() == {1: 0, 2: 1}


def test_build_parameter_index_map_rejects_inconsistent_name_for_same_index() -> None:
    table = (
        {"is_free": True, "parameter": "a", "parameter_index": 1},
        {"is_free": True, "parameter": "b", "parameter_index": 1},
    )
    with pytest.raises(ValueError, match="Inconsistent parameter name"):
        build_parameter_index_map(table)

