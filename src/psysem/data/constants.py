from __future__ import annotations

import re

SUPPORTED_ESTIMATORS = frozenset({"ML", "MLR", "WLSMV"})
SUPPORTED_VARIABLE_TYPES = frozenset({"continuous", "ordinal"})
LATENT_FACTOR_TOKEN = re.compile(r".+_f\d+$")
