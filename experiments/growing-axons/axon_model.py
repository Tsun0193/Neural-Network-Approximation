import numpy as np
import nevergrad as ng


def relu(x):
    """Rectified Linear Unit (ReLU) activation function."""
    return np.maximum(0, x)


def repu(x, q):
    """Rectified Power Unit (RePU) activation function with exponent q."""
    return np.where(x > 0, np.power(x, q), 0)


def objective_function(w, x, residuals, nonlinearity):
    """
    Objective function for the Axon algorithm.

    Args:
        w: Weight vector.
        x: Input data matrix.
        residuals: Residuals from the current approximation.
        nonlinearity: Activation function to apply.

    Returns:
        float: Objective value to minimize.
    """
    if np.dot(w, w) < 1e-7:
        return 100  # Penalize small weights to avoid instability
    new_basis = nonlinearity(x @ w)
    numerator = -(new_basis @ residuals) ** 2
    denominator = w.T @ x.T @ x @ w
    regularization = 1e-8 * (np.dot(w, w) - 1) ** 2
    return numerator / denominator + regularization


def modified_objective_function(w, x, residuals, nonlinearity):
    """
    Modified objective function for the Axon algorithm with orthogonalization.

    Args:
        w: Weight vector.
        x: Input data matrix.
        residuals: Residuals from the current approximation.
        nonlinearity: Activation function to apply.

    Returns:
        float: Objective value to minimize.
    """
    new_basis = nonlinearity(x @ w)
    # Orthogonalize against existing basis
    new_basis = new_basis - x @ (x.T @ new_basis)
    if np.dot(new_basis.flatten(), new_basis.flatten()) < 1e-7:
        return 100  # Penalize small basis vectors
    numerator = -(new_basis.flatten() @ residuals.flatten()) ** 2
    denominator = new_basis.flatten() @ new_basis.flatten()
    regularization = 1e-8 * (np.dot(new_basis.flatten(), new_basis.flatten()) - 1) ** 2
    return numerator / denominator + regularization


class AxonModel:
    """
    Axon model for function approximation using orthogonal basis functions.

    Attributes:
        input_dim (int): Input dimension.
        output_dim (int): Output dimension.
        nonlinearity (callable): Activation function.
        basis_coefficients (list): Coefficients for constructing basis functions.
        orthogonal_coefficients (list): Coefficients for orthogonalization.
        orthogonal_norms (list): Norms for normalization.
        qr_inverse (np.ndarray): Inverse of R from QR decomposition of [1, x].
    """

    def __init__(self, input_dim=1, output_dim=1, nonlinearity=relu):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nonlinearity = nonlinearity
        self.basis_coefficients = None
        self.orthogonal_coefficients = None
        self.orthogonal_norms = None
        self.qr_inverse = None
        self.output_coefficients = None
        self.basis_matrix = None

    def train(self, xs, ys, num_basis, get_optimizer=None, use_modified_objective=True):
        """
        Train the Axon model.

        Args:
            xs: Input data (numpy array).
            ys: Target values (numpy array).
            num_basis: Number of basis functions to construct.
            get_optimizer: Function to create an optimizer for a given number of variables.
            use_modified_objective: Whether to use the modified objective function.

        Returns:
            list: Relative errors after adding each basis function.
        """
        num_samples = xs.shape[0]
        design_matrix = np.hstack([np.ones((num_samples, 1)), xs])
        q, r = np.linalg.qr(design_matrix)
        self.qr_inverse = np.linalg.inv(r)

        self.basis_coefficients = []
        self.orthogonal_coefficients = []
        self.orthogonal_norms = []
        errors = []

        residuals = ys - q @ q.T @ ys
        objective = modified_objective_function if use_modified_objective else objective_function

        for _ in range(num_basis):
            # Solve optimization problem
            if get_optimizer is None:
                optimizer = ng.optimizers.OnePlusOne(parametrization=q.shape[1], budget=1200)
            else:
                optimizer = get_optimizer(q.shape[1])

            result = optimizer.minimize(lambda w: objective(w, q, residuals, self.nonlinearity))
            weights = result.args[0]
            weights /= np.linalg.norm(weights)

            # Construct new basis function
            new_basis = self.nonlinearity(q @ weights)

            # Orthogonalize and normalize
            orthogonal_coeffs = []
            norms = [np.linalg.norm(new_basis)]
            new_basis /= norms[0]

            for _ in range(2):  # Re-orthogonalization
                orthogonal_coeffs.append(q.T @ new_basis)
                new_basis -= q @ orthogonal_coeffs[-1]
                norms.append(np.linalg.norm(new_basis))
                new_basis /= norms[-1]

            # Store coefficients and norms
            self.orthogonal_norms.append(norms)
            self.orthogonal_coefficients.append(orthogonal_coeffs)
            self.basis_coefficients.append(weights)

            # Update basis matrix and residuals
            q = np.hstack([q, new_basis.reshape(-1, 1)])
            residuals = ys - q @ q.T @ ys
            errors.append(np.linalg.norm(residuals) / np.linalg.norm(ys))

        self.output_coefficients = q.T @ ys
        self.basis_matrix = q
        return errors

    def predict(self, xs):
        """
        Predict using the trained Axon model.

        Args:
            xs: Input data (numpy array).

        Returns:
            np.ndarray: Predicted values.
        """
        if None in [self.basis_coefficients, self.qr_inverse, self.orthogonal_coefficients, self.orthogonal_norms]:
            raise ValueError("Model is not trained.")

        design_matrix = np.hstack([np.ones((xs.shape[0], 1)), xs])
        basis = design_matrix @ self.qr_inverse

        for i, weights in enumerate(self.basis_coefficients):
            new_basis = self.nonlinearity(basis @ weights)
            new_basis /= self.orthogonal_norms[i][0]

            for coeff, norm in zip(self.orthogonal_coefficients[i], self.orthogonal_norms[i][1:]):
                new_basis -= basis @ coeff
                new_basis /= norm

            basis = np.hstack([basis, new_basis.reshape(-1, 1)])

        return basis @ self.output_coefficients


def axon_algorithm(xs, ys, num_basis, get_optimizer=None, use_modified_objective=True, nonlinearity=relu):
    """
    Greedy algorithm for function approximation using the Axon method.

    Args:
        xs: Input data (numpy array).
        ys: Target values (numpy array).
        num_basis: Number of basis functions to compute.
        get_optimizer: Function to create an optimizer for a given number of variables.
        use_modified_objective: Whether to use the modified objective function.
        nonlinearity: Activation function to use.

    Returns:
        tuple: Basis matrix, basis coefficients, QR inverse, orthogonal coefficients, orthogonal norms, and errors.
    """
    model = AxonModel(xs.shape[1], ys.shape[1], nonlinearity)
    errors = model.train(xs, ys, num_basis, get_optimizer, use_modified_objective)
    return (
        model.basis_matrix,
        model.basis_coefficients,
        model.qr_inverse,
        model.orthogonal_coefficients,
        model.orthogonal_norms,
        errors,
    )