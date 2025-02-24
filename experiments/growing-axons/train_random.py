"""Experiments with random initialization.

Train Axon model with random initialization and save errors to a pickle file.
The considered functions are the same as in experiments.ipynb.
"""
import argparse
import os
import pickle

import numpy as np
import torch
from axon_model import train_random_model

# Set up argument parser
parser = argparse.ArgumentParser(description="Train Axon network with random initialization.")
parser.add_argument("function", type=str, default="x2", 
                    help="Possible values: x2, sqrt, exp, sin, 2d, diff")
parser.add_argument("--K", type=int, default=10, 
                    help="The maximal number of basis functions to add.")
parser.add_argument("--num_epochs", type=int, default=1000, 
                    help="Number of training epochs.")
parser.add_argument("--eps", type=float, 
                    help="Epsilon for the equation -eps^2 u'' + u = 1, u(0) = u(1) = 0.")


def u(x, eps: float = 0.05):
    """Solution for the equation -eps^2 u'' + u = 1, u(0) = u(1) = 0."""
    if eps is None:
        raise ValueError("Epsilon (eps) is not defined.")
    
    a = (1 - np.exp(1 / eps)) / (np.exp(2 / eps) - 1)
    b = (np.exp(1 / eps) - np.exp(2 / eps)) / (np.exp(2 / eps) - 1)
    return a * np.exp(x / eps) + b * np.exp(-x / eps) + 1


def f_2d(x):
    """2D function: sqrt(x1^2 + x2^2)."""
    return np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2).astype(np.float64)


# Mapping of 1D functions
function_mapping_1d = {
    "x2": lambda x: x ** 2,
    "sqrt": lambda x: np.sqrt(x),
    "exp": lambda x: np.exp(-x),
    "sin": lambda x: np.sin(20 * x),
}


if __name__ == "__main__":
    # Set device (GPU if available, otherwise CPU)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Parse command-line arguments
    args = parser.parse_args()
    K = args.K
    num_epochs = args.num_epochs
    fname = f"error_{args.function}.pkl"  # File to save errors

    if args.function == "diff":
        # Solve the differential equation
        xs = np.linspace(0, 1, 1000).reshape(-1, 1)
        eps = args.eps
        errors = {eps: []}

        for k in range(1, K + 1):
            err_k = train_random_model(xs, lambda x: u(x, eps), k, num_epochs, device=device)
            errors[eps].append(err_k)

        # Save errors to file
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                errors1 = pickle.load(f)
            with open(fname, "wb") as f:
                errors1["error"].update(errors)
                pickle.dump(errors1, f)
        else:
            with open(fname, "wb") as f:
                pickle.dump({"error": errors}, f)

    elif args.function == "2d":
        # 2D function approximation
        x = np.linspace(-1, 1, 100)
        xx, yy = np.meshgrid(x, x)
        xs = np.hstack([xx.flatten()[:, np.newaxis], yy.flatten()[:, np.newaxis]])
        errors = []

        for k in range(1, K + 1):
            err_k = train_random_model(xs, f_2d, k, num_epochs, device=device)
            errors.append(err_k)

        # Save errors to file
        with open(fname, "wb") as f:
            pickle.dump({"error": errors}, f)

    elif args.function in function_mapping_1d.keys():
        # 1D function approximation
        xs = np.linspace(0, 1, 1000).reshape(-1, 1)
        f = function_mapping_1d[args.function]
        errors = []

        for k in range(1, K + 1):
            err_k = train_random_model(xs, f, k, num_epochs, device=device)
            errors.append(err_k)

        # Save errors to file
        with open(fname, "wb") as f:
            pickle.dump({"error": errors}, f)

    else:
        raise NotImplementedError(f"Function '{args.function}' is not supported.")