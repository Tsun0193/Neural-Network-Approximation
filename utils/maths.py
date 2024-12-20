import numpy as np
import inspect
from typing import Callable, Tuple

def get_derivative(f: Callable, epsilon: float = 1e-6) -> Callable:
    """
    Compute the derivative of a function.

    Parameters:
    - f: The function to differentiate.
    - epsilon: Small value for numerical stability.

    Returns:
    - The derivative of the function.
    """
    
    def df(x: float) -> float:
        return (f(x + epsilon) - f(x)) / epsilon

    return df

def trace_back(f: Callable, df_c: float, interval: Tuple, epsilon: float = 1e-6) -> float:
    """
    Find the point c in the interval where df(c) equals df_c.

    Parameters:
    - f: The function to differentiate.
    - df_c: The target value of the derivative.
    - interval: Tuple (a, b), the interval over which to find c.
    - epsilon: Tolerance for convergence.

    Returns:
    - The point c where df(c) = df_c.
    """
    a, b = interval
    df = get_derivative(f, epsilon)
    
    # Check if the value lies within the range of df(a) to df(b)
    if not (df(a) <= df_c <= df(b) or df(b) <= df_c <= df(a)):
        raise ValueError("Target derivative df_c is not within the range of the derivative on the interval.")

    # Use bisection method to find c
    while b - a > epsilon:
        c = (a + b) / 2
        current_df = df(c)
        
        if np.isclose(current_df, df_c, atol=epsilon):
            return c
        
        if current_df < df_c:
            a = c
        else:
            b = c

    return (a + b) / 2

def compute_approximation_error(f: Callable,
                                interval: Tuple) -> float:
    """
    Compute the optimal approximation error for a convex function.

    Parameters:
    - f: The convex function to approximate.
    - interval: Tuple (a, b), the interval over which to approximate.
    
    Returns:
    - Optimal approximation error.
    """

    # Get the derivative of the function
    df = get_derivative(f)

    a, b = interval

    df_c = (f(b) - f(a)) / (b - a)
    c = trace_back(f, df_c, interval)

    df_d = (f(c) - f(a)) / (c - a)
    d = trace_back(f, df_d, interval)

    try:
        equation = inspect.getsource(f).strip()
    except Exception:
        equation = "<unable to retrieve function>"

    res = {
        "error": (df_c - df_d) * (c - a) / 2,
        "c": c,
        "d": d,
        "df_c": df_c,
        "df_d": df_d,
        "extra_info": {
            "function": equation,
            "interval": interval
        }
    }

    return res