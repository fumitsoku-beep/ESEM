from __future__ import annotations

from typing import Any, Mapping

from .contracts import ESEMBlockSpec, ESEMSpec, RotationSpec, SpecValidationError
from .utils import optional_str, parse_str_list, require_str
from .validator import validate_esem_spec


def esem_spec_from_dict(payload: Mapping[str, Any]) -> ESEMSpec:
    """Build an :class:`ESEMSpec` from a dict-like payload."""
    if not isinstance(payload, Mapping):
        raise SpecValidationError("Spec payload must be a mapping.")

    blocks_raw = payload.get("blocks")
    if not isinstance(blocks_raw, list):
        raise SpecValidationError("`blocks` must be a list of block objects.")

    blocks = tuple(_parse_block(block_payload) for block_payload in blocks_raw)
    estimator = require_str(payload.get("estimator"), "estimator").upper()
    rotation = _parse_rotation(payload.get("rotation")) if "rotation" in payload else None
    structural = tuple(parse_str_list(payload.get("structural", []), "structural"))
    group = optional_str(payload.get("group"), "group")
    weight = optional_str(payload.get("weight"), "weight")
    cluster = optional_str(payload.get("cluster"), "cluster")
    participant_id = optional_str(payload.get("id"), "id")
    allow_item_overlap = bool(payload.get("allow_item_overlap", False))
    variable_types = _parse_variable_types(payload.get("variable_types"))

    spec = ESEMSpec(
        blocks=blocks,
        estimator=estimator,
        variable_types=variable_types,
        rotation=rotation,
        structural=structural,
        group=group,
        weight=weight,
        cluster=cluster,
        id=participant_id,
        allow_item_overlap=allow_item_overlap,
    )
    validate_esem_spec(spec)
    return spec


def _parse_block(payload: Any) -> ESEMBlockSpec:
    if not isinstance(payload, Mapping):
        raise SpecValidationError("Each block must be a mapping.")

    name = require_str(payload.get("name"), "blocks[].name")
    items = tuple(parse_str_list(payload.get("items"), f"blocks[{name}].items"))
    n_factors = payload.get("n_factors")
    if not isinstance(n_factors, int):
        raise SpecValidationError(f"`blocks[{name}].n_factors` must be an integer.")

    rotation = _parse_rotation(payload.get("rotation")) if "rotation" in payload else None
    return ESEMBlockSpec(name=name, items=items, n_factors=n_factors, rotation=rotation)


def _parse_rotation(payload: Any) -> RotationSpec:
    if not isinstance(payload, Mapping):
        raise SpecValidationError("`rotation` must be an object with method/oblique fields.")

    method = require_str(payload.get("method"), "rotation.method").lower()
    oblique = payload.get("oblique", True)
    if not isinstance(oblique, bool):
        raise SpecValidationError("`rotation.oblique` must be a boolean.")
    return RotationSpec(method=method, oblique=oblique)


def _parse_variable_types(payload: Any) -> dict[str, str]:
    if not isinstance(payload, Mapping):
        raise SpecValidationError("`variable_types` must be a mapping from name -> type.")

    parsed: dict[str, str] = {}
    for key, value in payload.items():
        name = require_str(key, "variable_types key")
        var_type = require_str(value, f"variable_types[{name}]").lower()
        parsed[name] = var_type
    return parsed
