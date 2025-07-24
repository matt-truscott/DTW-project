from __future__ import annotations

from pathlib import Path
import re
import shutil
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROC_ROOT = PROJECT_ROOT / "data" / "processed"


def load_paths(config_file: Path | None = None) -> tuple[Path, Path]:
    """Return RAW_ROOT and PROC_ROOT using ``config.yaml`` if available."""
    if config_file is None:
        config_file = PROJECT_ROOT / "config.yaml"

    raw_root = DEFAULT_RAW_ROOT
    proc_root = DEFAULT_PROC_ROOT
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        raw_root = Path(cfg.get("raw_root", raw_root)).expanduser()
        proc_root = Path(cfg.get("processed_root", proc_root)).expanduser()
    return raw_root, proc_root


ID_RE = re.compile(
    r"u(?P<user>\d{4})_?s(?P<session>\d{4})_sg(?P<sample>\d{4})\.mat$",
    re.I,
)


def copy_subset(
    subfolder: str,
    *,
    raw_root: Path | None = None,
    processed_root: Path | None = None,
    config_file: Path | None = None,
) -> None:
    """Copy one subset of BiosecurID files into a user-organised tree."""
    if raw_root is None or processed_root is None:
        raw_root, processed_root = load_paths(config_file)

    src_dir = raw_root / subfolder
    for mat_path in src_dir.glob("*.mat"):
        m = ID_RE.search(mat_path.name)
        if m is None:
            print(f"[SKIP] Unrecognised filename: {mat_path.name}")
            continue

        user = f"u{m.group('user')}"
        dest_dir = processed_root / user / subfolder
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_file = dest_dir / mat_path.name
        if dest_file.exists():
            continue

        shutil.copy2(mat_path, dest_file)


def main() -> None:
    raw_root, proc_root = load_paths()
    for subset in ("GlobalFeatures", "LocalFunctions"):
        copy_subset(subset, raw_root=raw_root, processed_root=proc_root)


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
