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


RAW_ROOT, PROC_ROOT = load_paths()

# Compile one regex to parse   uXXXX_sYYYY_sgZZZZ.mat
ID_RE = re.compile(
    r"u(?P<user>\d{4})_?s(?P<session>\d{4})_sg(?P<sample>\d{4})\.mat$",  # underscore optional
    re.I,
)

def copy_subset(subfolder: str) -> None:
    src_dir = RAW_ROOT / subfolder
    for mat_path in src_dir.glob("*.mat"):
        m = ID_RE.search(mat_path.name)
        if m is None:
            print(f"[SKIP] Unrecognised filename: {mat_path.name}")
            continue

        user     = f"u{m.group('user')}"
        dest_dir = PROC_ROOT / user / subfolder        # <── only the user level
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_file = dest_dir / mat_path.name
        if dest_file.exists():
            continue                                  # already copied

        shutil.copy2(mat_path, dest_file)
        # progress (optional)
        # print(f"→ {dest_file.relative_to(PROC_ROOT)}")

def main() -> None:
    for subset in ("GlobalFeatures", "LocalFunctions"):
        copy_subset(subset)

if __name__ == "__main__":
    main()
