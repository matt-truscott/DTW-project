import re
ID_RE = re.compile(r"u(\d{4})_s(\d{4})_sg(\d{4})")

def parse_id(fname: str):
    """
    Return (user, session, sample, label) from 'u1001_s0001_sg0003.mat'
    label ∈ {'genuine','skilled'}
    """
    u, s, g = map(int, ID_RE.match(fname).groups())
    label = "genuine" if g in (1,2,6,7) else "skilled"
    return u, s, g, label