import pytest
import numpy as np
import torch

from model.simple_model import ReLU_Network
from data.generators import generate_1d_convex
from utils.maths import get_derivative, compute_approximation_error
from utils.algo import optimal_approx

# -------------------------------
# Test for model/simple_model.py
# -------------------------------
def test_relu_network_forward_shape():
    model = ReLU_Network(input_dim=1, hidden_layers=1, hidden_units=8)
    x = torch.randn(10, 1)
    y = model(x)
    assert y.shape == (10, 1), "Output shape mismatch"

# -------------------------------
# Test for data/generators.py
# -------------------------------
def test_generate_1d_convex_shapes():
    X_train, X_val, y_train, y_val = generate_1d_convex(n_samples=500)
    assert X_train.shape[1] == 1
    assert X_train.shape == y_train.shape
    assert X_val.shape == y_val.shape

# -------------------------------
# Test for utils/maths.py
# -------------------------------
def test_get_derivative_approx():
    f = lambda x: x**2
    df = get_derivative(f)
    approx = df(2.0)
    assert np.isclose(approx, 4.0, atol=1e-2), "Derivative approximation off"

def test_compute_approximation_error_keys():
    f = lambda x: x**2
    result = compute_approximation_error(f, (0.0, 1.0))
    assert "error" in result and "c" in result and "d" in result

# -------------------------------
# Test for utils/algo.py
# -------------------------------
def test_optimal_approx_output_format():
    f = lambda x: x**2
    intervals, errors, rounds = optimal_approx(n=5, f=f, a=0, b=1, stepsize=0.01)
    assert isinstance(intervals, list)
    assert isinstance(errors, list)
    assert isinstance(rounds, int)
    assert len(intervals) == len(errors)