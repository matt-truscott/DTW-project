import os
import re
import scipy.io as sio

mat_file = r'data\raw\LocalFunctions\u1001s0001_sg0001.mat'

# 1) filename → genuine vs forgery
fname = os.path.basename(mat_file)
m = re.match(r'u(\d{4})s(\d{4})_sg(\d{4})\.mat', fname)
if m:
    user, session, sample = m.groups()
    sample = int(sample)
    genuines = {1, 2, 6, 7}
    label = "genuine" if sample in genuines else "skilled forgery"
    print(f"User {user}, session {session}, sample {sample:04d} → {label}\n")
else:
    print("⚠️ Filename didn’t match expected pattern\n")

try:
    mat = sio.loadmat(mat_file, struct_as_record=False, squeeze_me=True)
    print("Loaded with scipy.io.loadmat\n")

    for name, data in mat.items():
        if name.startswith("__"):
            continue

        # --- info printout ---
        shape = getattr(data, "shape", None)
        dtype = getattr(data, "dtype", None)
        info = [f"type={type(data).__name__}"]
        if shape is not None:
            info.append(f"shape={shape}")
        if dtype is not None:
            info.append(f"dtype={dtype}")
        print(f"Variable `{name}`: " + ", ".join(info))

        # --- look for any stored headings ---
        headings = None
        if hasattr(data, "_fieldnames"):
            headings = data._fieldnames
        elif hasattr(data, "dtype") and data.dtype.names:
            headings = data.dtype.names

        if headings:
            print("Headings:")
            for h in headings:
                print(f"  • {h}")
        else:
            print("No heading names stored (just raw numeric array).")

        # --- print the actual values ---
        print("Data:")
        print(data[1])
        print("-" * 40)

except FileNotFoundError:
    print("File not found—please check the path.")
