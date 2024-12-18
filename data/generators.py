# data/generate_data.py
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42

def generate_1d_convex(n_samples=1000, seed=SEED):
    """
    Generate datasets for a 1D convex function approximation task.

    This function creates input-output pairs where the output is a convex function
    of the input. Specifically, it uses the square function as an example of a convex function.

    :param int n_samples: 
        The total number of samples to generate. Default is 1000.
    
    :return: 
        A tuple containing the training and validation splits:
        (X_train, X_val, y_train, y_val)
    
    :rtype: tuple of numpy.ndarray
    """
    X = np.linspace(0, 1, n_samples).reshape(-1, 1)
    y = np.square(X)  # Example of a convex function
    return train_test_split(X, y, test_size=0.2, random_state=seed)

def generate_1d_non_convex(n_samples=1000, seed=SEED):
    """
    Generate datasets for a 1D non-convex function approximation task.

    This function creates input-output pairs where the output is a non-convex function
    of the input. Specifically, it uses the sine function as an example of a non-convex function.

    :param int n_samples: 
        The total number of samples to generate. Default is 1000.
    
    :return: 
        A tuple containing the training and validation splits:
        (X_train, X_val, y_train, y_val)
    
    :rtype: tuple of numpy.ndarray
    """
    X = np.linspace(0, 1, n_samples).reshape(-1, 1)
    y = np.sin(2 * np.pi * X)  # Example of a non-convex function
    return train_test_split(X, y, test_size=0.2, random_state=seed)

def generate_2d_convex(n_samples=1000, seed=SEED):
    """
    Generate datasets for a 2D convex function approximation task.

    This function creates input-output pairs where the output is a convex function
    of two input variables. Specifically, it uses the sum of squares of the inputs
    as an example of a convex function.

    :param int n_samples: 
        The total number of samples to generate. Default is 1000.
    
    :return: 
        A tuple containing the training and validation splits:
        (X_train, X_val, y_train, y_val)
    
    :rtype: tuple of numpy.ndarray
    """
    grid_size = int(np.sqrt(n_samples))
    X1 = np.linspace(0, 1, grid_size)
    X2 = np.linspace(0, 1, grid_size)
    X1, X2 = np.meshgrid(X1, X2)
    X = np.vstack([X1.ravel(), X2.ravel()]).T
    y = np.square(X[:, 0]) + np.square(X[:, 1])  # Convex function
    return train_test_split(X, y, test_size=0.2, random_state=seed)

def generate_2d_non_convex(n_samples=1000, seed=SEED):
    """
    Generate datasets for a 2D non-convex function approximation task.

    This function creates input-output pairs where the output is a non-convex function
    of two input variables. Specifically, it uses the product of sine and cosine functions
    as an example of a non-convex function.

    :param int n_samples: 
        The total number of samples to generate. Default is 1000.
    
    :return: 
        A tuple containing the training and validation splits:
        (X_train, X_val, y_train, y_val)
    
    :rtype: tuple of numpy.ndarray
    """
    grid_size = int(np.sqrt(n_samples))
    X1 = np.linspace(0, 1, grid_size)
    X2 = np.linspace(0, 1, grid_size)
    X1, X2 = np.meshgrid(X1, X2)
    X = np.vstack([X1.ravel(), X2.ravel()]).T
    y = np.sin(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1])  # Non-convex function
    return train_test_split(X, y, test_size=0.2, random_state=seed)

if __name__ == "__main__":
    assert len(generate_1d_convex()) == 4, f"Error in generate_1d_convex(). Expected 4, but got {len(generate_1d_convex())}"
    assert len(generate_1d_non_convex()) == 4, f"Error in generate_1d_non_convex(). Expected 4, but got {len(generate_1d_non_convex())}"
    assert len(generate_2d_convex()) == 4, f"Error in generate_2d_convex(). Expected 4, but got {len(generate_2d_convex())}"
    assert len(generate_2d_non_convex()) == 4, f"Error in generate_2d_non_convex(). Expected 4, but got {len(generate_2d_non_convex())}"
    
    print("Ready!")