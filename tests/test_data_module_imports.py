from psysem import esem_spec_from_dict as top_level_factory
from psysem.data import esem_spec_from_dict as data_factory
from psysem.esem_spec import esem_spec_from_dict as legacy_factory


def test_data_and_legacy_import_paths_are_compatible() -> None:
    assert top_level_factory is data_factory
    assert top_level_factory is legacy_factory
