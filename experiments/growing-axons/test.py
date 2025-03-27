import numpy as np
import torch
import nevergrad as ng
from axon_approximation import axon_algorithm
from axon_model import AxonNetwork, train_random_model


def get_opt_oneplus_one(n, **kwargs):
    """
    Returns a OnePlusOne optimizer for a given variable shape.

    Args:
        n: Number of variables.
        **kwargs: Additional arguments for the optimizer.

    Returns:
        nevergrad.optimizers.OnePlusOne: Configured optimizer.
    """
    return ng.optimizers.OnePlusOne(instrumentation=n, **kwargs)


if __name__ == "__main__":
    # Generate input data
    xs = np.linspace(0, 1, 1000)[:, None]  # 1000 points in [0, 1]
    ys = np.sin(20 * xs).flatten()  # Target function: sin(20 * x)

    # Run Axon algorithm to approximate the function
    basis_matrix, basis_coefs, qr_inverse, ortho_coefs, ortho_norms, errors = axon_algorithm(
        xs, ys, num_basis=10, get_optimizer=lambda n: get_opt_oneplus_one(n, budget=1200)
	)
    
    # Initialize and evaluate the PyTorch Axon model
    model = AxonNetwork(
        torch.from_numpy(xs.astype(np.float32)),
        torch.from_numpy(ys.astype(np.float32)),
        basis_coefs, qr_inverse, ortho_coefs, ortho_norms, basis_matrix
    )
    
    # Compute the approximation error
    predictions = model(torch.from_numpy(xs.astype(np.float32))).data.numpy().squeeze()
    approximation_error = np.linalg.norm(predictions - ys) / np.linalg.norm(ys)
    print(f"Approximation Error: {approximation_error:.6f}")

    # Train a random model for comparison
    random_model_errors = train_random_model(xs, lambda x: np.sin(20 * x), num_basis=10, num_epochs=1)
    print("Random Model Errors:", random_model_errors)