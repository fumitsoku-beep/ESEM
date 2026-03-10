import pandas as pd
import pytest

from psysem.data import SpecValidationError, esem_spec_from_dict, validate_esem_spec


def _minimal_payload() -> dict:
    return {
        "blocks": [
            {
                "name": "internalizing",
                "items": ["i1", "i2", "i3", "i4"],
                "n_factors": 2,
            }
        ],
        "estimator": "WLSMV",
        "rotation": {"method": "geomin", "oblique": True},
        "variable_types": {
            "i1": "ordinal",
            "i2": "ordinal",
            "i3": "ordinal",
            "i4": "ordinal",
            "wellbeing": "continuous",
        },
        "structural": ["wellbeing ~ internalizing_f1 + internalizing_f2"],
    }


def _minimal_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "i1": [1, 2, 3],
            "i2": [2, 2, 3],
            "i3": [1, 3, 4],
            "i4": [2, 2, 3],
            "wellbeing": [45.0, 50.0, 62.0],
        }
    )


def test_esem_spec_from_dict_success() -> None:
    payload = _minimal_payload()
    spec = esem_spec_from_dict(payload)
    validate_esem_spec(spec, _minimal_data())
    assert spec.estimator == "WLSMV"
    assert len(spec.blocks) == 1
    assert spec.blocks[0].name == "internalizing"


def test_esem_spec_rejects_missing_variable_type_for_item() -> None:
    payload = _minimal_payload()
    del payload["variable_types"]["i4"]

    with pytest.raises(SpecValidationError, match="Missing variable type for item `i4`"):
        esem_spec_from_dict(payload)


def test_esem_spec_rejects_missing_item_column_in_data() -> None:
    payload = _minimal_payload()
    spec = esem_spec_from_dict(payload)
    data = _minimal_data().drop(columns=["i3"])

    with pytest.raises(SpecValidationError, match="Item column `i3` is missing from data"):
        validate_esem_spec(spec, data)


def test_esem_spec_rejects_duplicate_items_across_blocks() -> None:
    payload = _minimal_payload()
    payload["blocks"].append(
        {
            "name": "externalizing",
            "items": ["i1", "e2", "e3", "e4"],
            "n_factors": 2,
        }
    )
    payload["variable_types"].update({"e2": "ordinal", "e3": "ordinal", "e4": "ordinal"})

    with pytest.raises(SpecValidationError, match="appears in multiple blocks"):
        esem_spec_from_dict(payload)


def test_esem_spec_allows_overlap_when_enabled() -> None:
    payload = _minimal_payload()
    payload["allow_item_overlap"] = True
    payload["blocks"].append(
        {
            "name": "externalizing",
            "items": ["i1", "e2", "e3", "e4"],
            "n_factors": 2,
        }
    )
    payload["variable_types"].update({"e2": "ordinal", "e3": "ordinal", "e4": "ordinal"})
    data = _minimal_data().assign(e2=[1, 2, 2], e3=[2, 2, 3], e4=[1, 2, 3])

    spec = esem_spec_from_dict(payload)
    validate_esem_spec(spec, data)


def test_esem_spec_rejects_non_integer_ordinal_values() -> None:
    payload = _minimal_payload()
    spec = esem_spec_from_dict(payload)
    data = _minimal_data().assign(i2=[1.0, 2.5, 3.0])

    with pytest.raises(
        SpecValidationError,
        match="marked ordinal but contains non-ordinal values",
    ):
        validate_esem_spec(spec, data)


def test_esem_spec_rejects_invalid_factor_count() -> None:
    payload = _minimal_payload()
    payload["blocks"][0]["n_factors"] = 4

    with pytest.raises(SpecValidationError, match="must be smaller than item count"):
        esem_spec_from_dict(payload)


def test_esem_spec_rejects_missing_structural_column_in_data() -> None:
    payload = _minimal_payload()
    spec = esem_spec_from_dict(payload)
    data = _minimal_data().drop(columns=["wellbeing"])

    with pytest.raises(SpecValidationError, match="Structural variable `wellbeing` is missing from data"):
        validate_esem_spec(spec, data)
