import numpy as np
from utils.maths import get_derivative, compute_approximation_error

def test_get_derivative_square_function():
    f = lambda x: x ** 2
    df = get_derivative(f)
    approx = df(3.0)
    assert np.isclose(approx, 6.0, atol=1e-1)

def test_compute_approximation_error_output_keys():
    f = lambda x: x ** 2
    result = compute_approximation_error(f, (0, 1))
    assert "error" in result and "c" in result and "d" in result
