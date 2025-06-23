from pathlib import Path
import pandas as pd

from . import main as bio
from . import parse as ps


def build_df(raw_root=Path(r"C:\Users\mattt\Skripsie\Projects\DTW-project\data\raw")):
    rows = []
    for path in (raw_root / "GlobalFeatures").glob("*.mat"):
        sig_id = path.stem
        user, ses, samp, label = ps.parse_id(sig_id)
        g_vec = bio.load_global(sig_id)
        loc = bio.load_local(sig_id)
        rows.append(
            dict(
                user=user,
                session=ses,
                sample=samp,
                label=label,
                T=loc.shape[0],
                global_vec=g_vec,
            )
        )
    return (
        pd.DataFrame(rows)
        .set_index(["user", "session", "sample"])
        .sort_index()
    )


meta = build_df()
meta.to_feather("data/interim/metadata_v1.feather")
