from pathlib import Path
import re
import shutil

# ----- change these two if your layout moves -----
RAW_ROOT = Path(r"C:\Users\mattt\Skripsie\Projects\DTW-project\data\raw")
PROC_ROOT = Path(r"C:\Users\mattt\Skripsie\Projects\DTW-project\data\processed")
# -------------------------------------------------

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