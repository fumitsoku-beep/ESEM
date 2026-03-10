from __future__ import annotations

from ..contracts import ESEMSpec


def validate_block_rules(spec: ESEMSpec, errors: list[str]) -> set[str]:
    """Validate block-level constraints and return the deduplicated item set."""
    block_names_seen: set[str] = set()
    item_set: set[str] = set()

    for block in spec.blocks:
        if block.name in block_names_seen:
            errors.append(f"Duplicate block name `{block.name}`.")
        block_names_seen.add(block.name)

        if not block.items:
            errors.append(f"Block `{block.name}` must include at least one item.")

        if block.n_factors <= 0:
            errors.append(f"Block `{block.name}` must have `n_factors` > 0.")

        if block.items and block.n_factors >= len(block.items):
            errors.append(
                f"Block `{block.name}` has invalid `n_factors={block.n_factors}` "
                f"for {len(block.items)} items; it must be smaller than item count."
            )

        block_item_seen: set[str] = set()
        for item in block.items:
            if item in block_item_seen:
                errors.append(f"Block `{block.name}` contains duplicate item `{item}`.")
            block_item_seen.add(item)

            if not spec.allow_item_overlap and item in item_set:
                errors.append(
                    f"Item `{item}` appears in multiple blocks. "
                    "Set `allow_item_overlap=True` to allow this."
                )
            item_set.add(item)

    return item_set
