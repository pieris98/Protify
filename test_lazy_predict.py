from src.protify.probes.lazy_predict import LazyClassifier 
from src.protify.probes.lazy_predict import LazyRegressor
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", type=int, default=0, help="0=summary, 1=full table")
args = parser.parse_args()

# Small synthetic data
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)

clf = LazyClassifier(classifiers="all", verbose=args.verbose)
clf_scores = clf.fit(X[:80], X[80:], y[:80], y[80:])

# Test regressor with continuous target
y_reg = np.random.rand(100)
rg = LazyRegressor(regressors="all", verbose=args.verbose)
rg_scores = rg.fit(X[:80], X[80:], y_reg[:80], y_reg[80:])