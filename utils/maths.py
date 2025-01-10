import numpy as np
import inspect
from typing import Callable, Tuple, Literal, List

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

def constructor(f: Callable,
                interval: Tuple,
                type: Literal["upper", "lower"]) -> Callable:
    """
    Construct a function which has a constant second-order derivative:
    f''(x) = max(f''(a), f''(b)) for x in [a, b] (upper bound) or
    f''(x) = min(f''(a), f''(b)) for x in [a, b] (lower bound).

    Parameters:
    - f: The function to approximate.
    - interval: Tuple (a, b), the interval over which to approximate.
    - type: The type of approximation to perform.

    Note: The function is just for experimentation purposes and may not be useful in practice.
    """

    a, b = interval

    def second_df(func: Callable,
                  x: float,
                  epsilon: float = 1e-6) -> float:
        return (func(x + epsilon) - 2 * func(x) + func(x - epsilon)) / epsilon ** 2
    
    fpp_a = second_df(f, a)
    fpp_b = second_df(f, b)

    if type == "upper":
        fpp = max(fpp_a, fpp_b)
    elif type == "lower":
        fpp = min(fpp_a, fpp_b)
    else:
        raise ValueError("Invalid type. Must be 'upper' or 'lower'.")
    
    return lambda x: fpp * (x ** 2) / 2


def get_slope_by_2_points(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Compute the slope of a line passing through two points.

    Parameters:
    - p1: The first point (x, y).
    - p2: The second point (x, y).

    Returns:
    - The slope of the line.
    """

    x1, y1 = p1
    x2, y2 = p2

    return (y2 - y1) / (x2 - x1)

def optimal_approx(
    n: int,
    f: Callable,
    a: float,
    b: float,
    stepsize: float
) -> Tuple[List[Tuple[float, float]], List[float]]:
    """
    Compute the optimal approximation of a convex function.

    Parameters:
    - n: The number of points to use.
    - f: The function to approximate.
    - a: The start of the interval.
    - b: The end of the interval.
    - stepsize: The stepsize to use for the approximation.

    Returns:
    - Tuple of points and the optimal approximation error.
    """

    # Initialize intervals equally distributed
    intervals = [(a + i * (b - a) / n, a + (i + 1) * (b - a) / n) for i in range(n)]

    def compute_segment_error(segment: Tuple[float, float]) -> float:
        """Compute error for a single segment using the custom utility."""
        result = compute_approximation_error(f, segment)
        return result["error"]

    # Compute initial errors
    errors = [compute_segment_error(interval) for interval in intervals]

    while True:
        max_error = max(errors)
        min_error = min(errors)
        prev_error_diff = max_error - min_error

        for i in range(1, n):
            left_error = errors[i - 1]
            right_error = errors[i]
            x_common = intervals[i][0]  # common endpoint

            if left_error > right_error:
                while left_error > right_error:
                    x_common -= stepsize
                    intervals[i - 1] = (intervals[i - 1][0], x_common)
                    intervals[i] = (x_common, intervals[i][1])
                    left_error = compute_segment_error(intervals[i - 1])
                    right_error = compute_segment_error(intervals[i])
            elif left_error < right_error:
                while left_error < right_error:
                    x_common += stepsize
                    intervals[i - 1] = (intervals[i - 1][0], x_common)
                    intervals[i] = (x_common, intervals[i][1])
                    left_error = compute_segment_error(intervals[i - 1])
                    right_error = compute_segment_error(intervals[i])

        # Recompute errors
        errors = [compute_segment_error(interval) for interval in intervals]
        max_error = max(errors)
        min_error = min(errors)
        current_error_diff = max_error - min_error

        # Check convergence
        if prev_error_diff <= current_error_diff:
            break

    return intervals, errors
