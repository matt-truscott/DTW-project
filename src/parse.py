import re

ID_RE = re.compile(r"u(\d{4})_s(\d{4})_sg(\d{4})")


def parse_id(fname: str) -> tuple[int, int, int, str]:
    """Return ``(user, session, sample, label)`` from a signature filename."""
    m = ID_RE.search(str(fname))
    if not m:
        raise ValueError(
            f"Filename {fname!r} does not match 'u####_s####_sg####' pattern"
        )
    u, s, g = map(int, m.groups())
    label = "genuine" if g in (1, 2, 6, 7) else "skilled"
    return u, s, g, label
