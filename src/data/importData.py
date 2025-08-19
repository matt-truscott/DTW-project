"""
Utilities to copy BiosecurID .mat feature files from a raw dump into a
user-centred processed layout under data/processed/uXXXX/{GlobalFeatures,LocalFunctions}/.
"""

from __future__ import annotations
import re
import shutil
from pathlib import Path
from typing import Optional

# Automatically derive the project root from this file’s location
_PROJECT_ROOT      = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT   = _PROJECT_ROOT / "data" / "raw"
DEFAULT_PROC_ROOT  = _PROJECT_ROOT / "data" / "processed"

# Match filenames like u1001s0001_sg0001.mat (underscore before session is optional)
ID_RE = re.compile(
    r"^u(?P<user>\d{4})_?s(?P<session>\d{4})_sg(?P<sample>\d{4})\.mat$",
    re.IGNORECASE
)

def copy_subset(
    subset: str,
    raw_root: Optional[Path] = None,
    proc_root: Optional[Path] = None,
) -> None:
    """
    Copy all .mat files from raw_root/{subset}/ into
    proc_root/uXXXX/{subset}/, preserving filenames.
    """
    raw_root  = Path(raw_root)  if raw_root  else DEFAULT_RAW_ROOT
    proc_root = Path(proc_root) if proc_root else DEFAULT_PROC_ROOT
    src_dir   = raw_root / subset

    if not src_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {src_dir}")

    copied = 0
    skipped = 0
    unknown = 0

    for mat_path in sorted(src_dir.glob("*.mat")):
        match = ID_RE.match(mat_path.name)
        if not match:
            print(f"[SKIP] Unrecognised filename: {mat_path.name}")
            unknown += 1
            continue

        user = f"u{match.group('user')}"
        dest_dir = proc_root / user / subset
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_file = dest_dir / mat_path.name
        if dest_file.exists():
            skipped += 1
            continue  # already copied

        shutil.copy2(mat_path, dest_file)
        copied += 1

    print(f"[{subset}] copied={copied}, skipped={skipped}, unknown={unknown}")

def main(
    raw_root: Optional[Path] = None,
    proc_root: Optional[Path] = None,
    subsets: Optional[list[str]] = None
) -> None:
    """
    Copy all subsets in one go. Run this once to populate data/processed.
    """
    if subsets is None:
        subsets = ["GlobalFeatures", "LocalFunctions"]

    for subset in subsets:
        copy_subset(subset, raw_root=raw_root, proc_root=proc_root)

if __name__ == "__main__":
    print("Copying BiosecurID raw .mat files into per-user processed layout...")
    main()
    print("Done. Processed data now under:", DEFAULT_PROC_ROOT)
