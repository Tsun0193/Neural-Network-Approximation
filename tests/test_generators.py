import numpy as np
from data.generators import generate_1d_convex

def test_generate_1d_convex_sizes():
    X_train, X_val, y_train, y_val = generate_1d_convex(n_samples=100)
    assert X_train.shape == y_train.shape
    assert X_val.shape == y_val.shape
    assert X_train.shape[1] == 1
