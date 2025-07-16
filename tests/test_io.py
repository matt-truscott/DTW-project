import scipy.io as sio

mat_file_path = r'C:\Users\mattt\Skripsie\Projects\DTW-project\data\processed\u1001\GlobalFeatures\u1001s0001_sg0001.mat'

try:
    mat = sio.loadmat(mat_file_path)
    print("Loaded .mat file successfully with scipy.io.", list(mat.keys()))
except NotImplementedError:
    # Handle MATLAB v7.3 files (HDF5-based)
    import h5py
    try:
        with h5py.File(mat_file_path, 'r') as f:
            print("variables in the file:", list(f.keys()))
            # You can access the data like this:
            # data = f['variable_name'][:]
            # Replace 'variable_name' with the actual variable name in your .mat file
    except Exception as e:
        print(f"An error occurred while reading the file with h5py: {e}")
except FileNotFoundError:
    print("File not found. Please check the file path.")