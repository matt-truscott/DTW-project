from pathlib import Path
import shutil

from . import parse as ps

RAW_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"
PROC_ROOT = Path(__file__).resolve().parents[1] / "data" / "processed"


def group_sessions(raw_root: Path = RAW_ROOT, proc_root: Path = PROC_ROOT) -> None:
    """Copy raw feature files into per-user/session directories."""
    g_dir = raw_root / "GlobalFeatures"
    l_dir = raw_root / "LocalFunctions"
    for g_path in g_dir.glob("*.mat"):
        sig_id = g_path.stem
        user, session, _, _ = ps.parse_id(sig_id)
        dest_dir = proc_root / f"u{user:04d}" / f"s{session:04d}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(g_path, dest_dir / g_path.name)
        l_path = l_dir / f"{sig_id}.mat"
        if l_path.exists():
            shutil.copy2(l_path, dest_dir / l_path.name)


def main() -> None:
    group_sessions()


if __name__ == "__main__":
    main()
