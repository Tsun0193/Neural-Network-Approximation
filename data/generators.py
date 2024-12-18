import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42

def is_perfect_square(n):
    """
    Check if a number is a perfect square.
    """
    return int(np.sqrt(n)) ** 2 == n

def generate_1d_convex(n_samples=1000, seed=SEED,
                       func=lambda x: np.square(x)):
    """
    Generate datasets for a 1D convex function approximation task.
    
    :param int n_samples: Total number of samples.
    :param int seed: Random seed for reproducibility.
    :param callable func: Function to apply on X.
    
    :return: Training and validation splits.
    """
    X = np.linspace(0, 1, n_samples).reshape(-1, 1)
    y = func(X)  # Example of a convex function
    return train_test_split(X, y, test_size=0.2, random_state=seed)

def generate_1d_non_convex(n_samples=1000, seed=SEED,
                           func=lambda x: np.sin(2 * np.pi * x)):
    """
    Generate datasets for a 1D non-convex function approximation task.
    
    :param int n_samples: Total number of samples.
    :param int seed: Random seed for reproducibility.
    :param callable func: Function to apply on X.
    
    :return: Training and validation splits.
    """
    X = np.linspace(0, 1, n_samples).reshape(-1, 1)
    y = func(X)  # Example of a non-convex function
    return train_test_split(X, y, test_size=0.2, random_state=seed)

def generate_2d_convex(n_samples=10000, seed=SEED,
                       func=lambda x: np.square(x[0]) + np.square(x[1])):
    """
    Generate datasets for a 2D convex function approximation task.
    
    :param int n_samples: Total number of samples (must be a perfect square).
    :param int seed: Random seed for reproducibility.
    :param callable func: Function to apply on X.
    
    :return: Training and validation splits.
    """
    if not is_perfect_square(n_samples):
        raise ValueError(f"n_samples={n_samples} is not a perfect square.")
    
    grid_size = int(np.sqrt(n_samples))
    X1 = np.linspace(0, 1, grid_size)
    X2 = np.linspace(0, 1, grid_size)
    X1, X2 = np.meshgrid(X1, X2)
    X = np.vstack([X1.ravel(), X2.ravel()]).T
    y = func(X.T)
    return train_test_split(X, y, test_size=0.2, random_state=seed)

def generate_2d_non_convex(n_samples=10000, seed=SEED,
                           func=lambda x: np.sin(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1])):
    """
    Generate datasets for a 2D non-convex function approximation task.
    
    :param int n_samples: Total number of samples (must be a perfect square).
    :param int seed: Random seed for reproducibility.
    :param callable func: Function to apply on X.
    
    :return: Training and validation splits.
    """
    if not is_perfect_square(n_samples):
        raise ValueError(f"n_samples={n_samples} is not a perfect square.")
    
    grid_size = int(np.sqrt(n_samples))
    X1 = np.linspace(0, 1, grid_size)
    X2 = np.linspace(0, 1, grid_size)
    X1, X2 = np.meshgrid(X1, X2)
    X = np.vstack([X1.ravel(), X2.ravel()]).T
    y = func(X.T)
    return train_test_split(X, y, test_size=0.2, random_state=seed)

if __name__ == "__main__":
    # Test 1D Convex
    convex_1d = generate_1d_convex()
    assert len(convex_1d) == 4, f"Error in generate_1d_convex(). Expected 4, but got {len(convex_1d)}"
    
    # Test 1D Non-Convex
    non_convex_1d = generate_1d_non_convex()
    assert len(non_convex_1d) == 4, f"Error in generate_1d_non_convex(). Expected 4, but got {len(non_convex_1d)}"
    
    # Test 2D Convex
    convex_2d = generate_2d_convex()
    assert len(convex_2d) == 4, f"Error in generate_2d_convex(). Expected 4, but got {len(convex_2d)}"
    
    # Test 2D Non-Convex
    non_convex_2d = generate_2d_non_convex()
    assert len(non_convex_2d) == 4, f"Error in generate_2d_non_convex(). Expected 4, but got {len(non_convex_2d)}"
    
    print("All tests passed!")
