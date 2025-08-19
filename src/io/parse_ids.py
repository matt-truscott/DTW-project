import re
from dataclasses import dataclass

# uXXXXsYYYY_sgZZZZ (case-insensitive; underscore before s is allowed by import step already)
_RX = re.compile(r"u(\d{4})s(\d{4})_sg(\d{4})", re.IGNORECASE)

GENUINE = {1, 2, 6, 7}
SKILLED = {3, 4, 5}

@dataclass(frozen=True)
class SigId:
    user: int
    session: int
    attempt: int

def parse_sig(name: str) -> SigId:
    m = _RX.search(name)
    if not m:
        raise ValueError(f"Unrecognized file: {name}")
    return SigId(*(int(g) for g in m.groups()))

def attempt_label(attempt: int) -> str:
    if attempt in GENUINE:
        return "genuine"
    if attempt in SKILLED:
        return "skilled"
    raise ValueError(f"Unknown attempt number: {attempt}")
