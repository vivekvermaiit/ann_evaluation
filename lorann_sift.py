import time
import numpy as np
import lorann
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path.home() / "knn_datasets/dense_datasets/sift1M"
DATA_PATH = BASE_DIR / "dataset.npy"
QUERY_PATH = BASE_DIR / "queries.npy"
GT_PATH = BASE_DIR / "groundtruth.npy"
OUT_PATH = "lorann_sift_results_2048.tsv"

# -----------------------------
# Parameters (match Kannolo)
# -----------------------------
NUM_QUERIES = 10_000
K = 10

# LoRANN index hyperparameters
N_CLUSTERS = 2048
GLOBAL_DIM = None          # low dimensional data
QUANT_BITS = 8
DISTANCE = lorann.L2

# LoRANN query hyperparameters
POINTS_TO_RERANK = 800    # fixed, not swept
CLUSTERS_TO_SEARCH_SWEEP = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

# -----------------------------
# Load data
# -----------------------------
print("Loading data...")
data = np.load(DATA_PATH).astype(np.float32)
queries = np.load(QUERY_PATH).astype(np.float32)[:NUM_QUERIES]
groundtruth = np.load(GT_PATH)[:NUM_QUERIES, :K]

# Ensure contiguous memory (important)
data = np.ascontiguousarray(data)
queries = np.ascontiguousarray(queries)

# -----------------------------
# Build or load index
# -----------------------------
INDEX_PATH = "sift1m_lorann_2048.idx" #for 2048 clusters

start = time.perf_counter()
if Path(INDEX_PATH).exists():
    print("Loading LoRANN index...")
    index = lorann.LorannIndex.load(INDEX_PATH)
else:
    print("Building LoRANN index...")
    index = lorann.LorannIndex(
        data=data,
        n_clusters=N_CLUSTERS,
        global_dim=GLOBAL_DIM,
        quantization_bits=QUANT_BITS,
        distance=DISTANCE,
    )
    index.build(verbose=True)
    index.save(INDEX_PATH)
end = time.perf_counter()
index_time = (end - start)
print(f"  Index time to build SIFT: {index_time:.2f} s")

# -----------------------------
# Run sweep
# -----------------------------
results_rows = []

print("\nRunning sweep...")
for cts in CLUSTERS_TO_SEARCH_SWEEP:
    print(f"clusters_to_search = {cts}")

    start = time.perf_counter()
    results = index.search(
        queries,
        K,
        clusters_to_search=cts,
        points_to_rerank=POINTS_TO_RERANK,
    )
    end = time.perf_counter()

    # Average query time (microseconds)
    avg_time_us = (end - start) / NUM_QUERIES * 1e6

    # Accuracy / Recall@10
    accuracy = lorann.compute_recall(results, groundtruth)

    results_rows.append((cts, avg_time_us, accuracy))

    print(f"  Avg time: {avg_time_us:.2f} µs | Accuracy@10: {accuracy:.4f}")

# -----------------------------
# Write TSV
# -----------------------------
print(f"\nWriting results to {OUT_PATH}")
with open(OUT_PATH, "w") as f:
    f.write("clusters_to_search\tQuery Time (microsecs)\tAccuracy\n")
    for cts, t, acc in results_rows:
        f.write(f"{cts}\t{t:.3f}\t{acc:.6f}\n")

print("Done.")
