from pathlib import Path
import pandas as pd
from . import main as bio
from . import parse as ps

RAW_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"
PROC_ROOT = Path(__file__).resolve().parents[1] / "data" / "processed"
PROC_ROOT.mkdir(parents=True, exist_ok=True)


def build_df(raw_root: Path = RAW_ROOT) -> pd.DataFrame:
    rows = []
    for path in (raw_root / "GlobalFeatures").glob("*.mat"):
        sig_id = path.stem
        user, ses, samp, label = ps.parse_id(sig_id)
        g_vec = bio.load_global(sig_id)
        loc = bio.load_local(sig_id)
        rows.append(dict(
            user=user,
            session=ses,
            sample=samp,
            label=label,
            T=loc.shape[0],
            global_vec=g_vec
        ))
    df = pd.DataFrame(rows)
    df.set_index(["user", "session", "sample"], inplace=True)
    df.sort_index(inplace=True)
    return df


def main() -> None:
    df = build_df()
    out = PROC_ROOT / "metadata_v1.feather"
    df.to_feather(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
