from __future__ import annotations

import re
from typing import Iterable

from .constants import LATENT_FACTOR_TOKEN


def extract_tokens(expressions: Iterable[str]) -> set[str]:
    tokens: set[str] = set()
    for expr in expressions:
        tokens.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr))
    return tokens


def observed_structural_variables(structural: Iterable[str]) -> set[str]:
    observed: set[str] = set()
    for token in extract_tokens(structural):
        # Convention: tokens ending with "_f<number>" are latent references.
        if LATENT_FACTOR_TOKEN.match(token):
            continue
        observed.add(token)
    return observed
