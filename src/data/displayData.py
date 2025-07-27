import scipy.io as sio
from pathlib import Path

def inspect_mat_file(mat_path):
    """
    Load a MATLAB .mat file and inspect its contents.

    Prints, for each top‑level variable:
      - its Python type
      - its shape tuple
      - explicit row & column counts
      - its dtype
      - any struct‑field or dtype.names (headings)
      - the full array contents

    Returns
    -------
    features : dict[str, numpy.ndarray]
        Mapping from variable name → array
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        print(f"⚠️  File not found: {mat_path}")
        return {}

    try:
        mat = sio.loadmat(str(mat_path),
                          squeeze_me=True,
                          struct_as_record=False)
    except NotImplementedError:
        print(f"⚠️  Cannot read (maybe v7.3 .mat?) {mat_path}")
        return {}

    features = {}
    print(f"Loaded `{mat_path.name}` with scipy.io.loadmat:\n")

    for name, data in mat.items():
        if name.startswith("__"):
            continue  # skip MATLAB metadata

        # store for programmatic access
        features[name] = data

        # basic info
        dtype = getattr(data, "dtype", None)
        shape = getattr(data, "shape", None)
        tname = type(data).__name__
        print(f"Variable `{name}`:")
        print(f"  • type = {tname}")
        print(f"  • shape = {shape}")

        # explicit rows x columns
        if shape is None:
            print("  • rows = ?, columns = ?")
        elif len(shape) == 1:
            print(f"  • rows = {shape[0]}, columns = 1")
        elif len(shape) == 2:
            r, c = shape
            print(f"  • rows = {r}, columns = {c}")
        else:
            print(f"  • rows/columns = {shape}")

        print(f"  • dtype = {dtype}")

        # any stored headings?
        headings = None
        if hasattr(data, "_fieldnames"):
            headings = data._fieldnames
        elif hasattr(data, "dtype") and data.dtype.names:
            headings = data.dtype.names

        if headings:
            print("  • Headings:")
            for h in headings:
                print(f"      – {h}")
        else:
            print("  • No headings stored (raw numeric array)")

        # full contents
        print("  • Data:")
        print(data)
        print("-" * 60)

    return features
