import numpy as np
import os

BASE_DIR = os.path.expanduser("~/scratch/knn_datasets/dense_datasets/gist")

# ---------- fvecs / ivecs readers ----------
def read_fvecs(fname):
    with open(fname, "rb") as f:
        vectors = []
        while True:
            dim = np.fromfile(f, dtype=np.int32, count=1)
            if not dim.size:
                break
            vec = np.fromfile(f, dtype=np.float32, count=dim[0])
            vectors.append(vec)
    return np.vstack(vectors)

def read_ivecs(fname):
    with open(fname, "rb") as f:
        vectors = []
        while True:
            dim = np.fromfile(f, dtype=np.int32, count=1)
            if not dim.size:
                break
            vec = np.fromfile(f, dtype=np.int32, count=dim[0])
            vectors.append(vec)
    return np.vstack(vectors)

# ---------- load raw data ----------
print("Loading GIST base vectors...")
base = read_fvecs(os.path.join(BASE_DIR, "gist_base.fvecs"))

print("Loading GIST query vectors...")
queries = read_fvecs(os.path.join(BASE_DIR, "gist_query.fvecs"))

print("Loading GIST ground truth...")
gt = read_ivecs(os.path.join(BASE_DIR, "gist_groundtruth.ivecs"))

# ---------- sanity checks ----------
print("Base shape:", base.shape)        # (1_000_000, 960)
print("Query shape:", queries.shape)    # (1_000, 960)
print("GT shape:", gt.shape)            # (1_000, 100)

assert base.dtype == np.float32
assert queries.dtype == np.float32
assert np.issubdtype(gt.dtype, np.integer)

# ---------- save in kANNolo / LoRANN format ----------
np.save(os.path.join(BASE_DIR, "dataset.npy"), base)
np.save(os.path.join(BASE_DIR, "queries.npy"), queries)
np.save(os.path.join(BASE_DIR, "groundtruth.npy"), gt)

print("Saved:")
print(" - dataset.npy")
print(" - queries.npy")
print(" - groundtruth.npy")
