import numpy as np
import hnswlib
import time
import os

# Paths
BASE_DIR = os.path.expanduser("~/scratch/knn_datasets/dense_datasets/mnist")
dataset_path = os.path.join(BASE_DIR, "dataset.npy")
query_path   = os.path.join(BASE_DIR, "queries.npy")
gt_path      = os.path.join(BASE_DIR, "groundtruth.npy")
output_tsv   = "mnist_hnswlib_report.tsv"

# Parameters
K = 10
NUM_QUERIES = 10000
ef_search_values = [10, 20, 40, 80, 120, 200, 400, 600, 800]

print("Loading data...")
data     = np.load(dataset_path)
queries  = np.load(query_path)[:NUM_QUERIES]
groundtruth = np.load(gt_path)[:NUM_QUERIES, :K]

dim = data.shape[1]
num_elements = data.shape[0]

print("Initializing hnswlib index...")
p = hnswlib.Index(space="l2", dim=dim)

# Recommend larger M for high-d data — try 32 or 24
M = 8
efc = 80

print("Building index...")

start_build = time.perf_counter()

p.init_index(
    max_elements=num_elements,
    ef_construction=efc,
    M=M
)
p.add_items(data)

end_build = time.perf_counter()

index_build_time = end_build - start_build
print(f"HNSWLIB Index build time: {index_build_time:.2f} s")


# Set ef (query time parameter)
p.set_ef(10)  # will be overridden in sweep

results = []

# Sweep ef_search
print("Running sweep...")
for ef in ef_search_values:
    p.set_ef(ef)
    print(f"ef_search = {ef}")

    start = time.time()
    labels, distances = p.knn_query(queries, k=K)
    end = time.time()

    avg_time_us = (end - start) / NUM_QUERIES * 1e6

    # Compute recall
    # groundtruth and labels are both shape (NUM_QUERIES, K)
    # hnswlib returns integer IDs
    correct = 0
    for i in range(NUM_QUERIES):
        # compare sets of neighbors
        gt_set = set(groundtruth[i])
        res_set = set(labels[i])
        correct += len(gt_set & res_set)
    recall_at_10 = correct / (NUM_QUERIES * K)

    print(f"  Avg time: {avg_time_us:.2f} µs | Recall@10: {recall_at_10:.4f}")
    results.append((ef, avg_time_us, recall_at_10))

# Write results to TSV
print(f"\nWriting results to {output_tsv}")
with open(output_tsv, "w") as f:
    f.write("ef_search\tQuery Time (microsecs)\tAccuracy\n")
    for ef, t, acc in results:
        f.write(f"{ef}\t{t:.3f}\t{acc:.6f}\n")

print("Done.")
