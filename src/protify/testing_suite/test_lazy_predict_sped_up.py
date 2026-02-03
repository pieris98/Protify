# Example usage:
# python -m src.protify.testing_suite.test_lazy_predict_sped_up

import time
import numpy as np
from src.protify.probes.lazy_predict import LazyClassifier, LazyRegressor

# Larger synthetic data for more reliable benchmarking
X = np.random.randn(1000, 256)  # 1000 samples, 256 features
y_cls = np.random.randint(0, 2, 1000)
y_reg = np.random.randn(1000)

NUM_RUNS = 3
times = []

for run in range(NUM_RUNS):
    print(f"\n=== Run {run + 1}/{NUM_RUNS} ===")
    run_start = time.time()
    
    # Classifier (n_jobs=-1 uses all cores)
    clf = LazyClassifier(classifiers="all", verbose=0, n_jobs=-1)
    clf_scores = clf.fit(X[:800], X[800:], y_cls[:800], y_cls[800:])
    
    # Regressor (n_jobs=-1 uses all cores)
    reg = LazyRegressor(regressors="all", verbose=0, n_jobs=-1)
    reg_scores = reg.fit(X[:800], X[800:], y_reg[:800], y_reg[800:])
    
    run_time = time.time() - run_start
    times.append(run_time)
    print(f"Run {run + 1} time: {run_time:.2f}s")

print(f"\n=== Results ===")
print(f"Times: {[f'{t:.2f}s' for t in times]}")
print(f"Average: {np.mean(times):.2f}s")
print(f"Std: {np.std(times):.2f}s")

# BASELINE:
# Times: ['26.06s', '26.31s', '26.94s']
# Average: 26.43s
# Std: 0.37s
# PARALLELIZED VERSION:
# === Results ===
# Times: ['23.54s', '11.55s', '8.83s']
# Average: 14.64s
# Std: 6.39s
# Times: ['14.22s', '14.22s', '10.64s']
# Average: 13.03s
# Std: 1.69s