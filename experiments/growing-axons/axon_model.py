import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from tqdm import tqdm

class AxonNetwork(nn.Module):
    """Neural network for function approximation using orthogonal basis functions.
    
    Attributes:
        qr_inverse (torch.Tensor): Inverse R matrix from QR decomposition
        ortho_coeffs (list): Coefficients for basis orthogonalization
        norm_factors (list): Normalization factors for basis functions
        coeff (torch.Tensor): Final projection coefficients
        device (torch.device): Computation device
    """

    def __init__(self, x, y, basis_coef_init=None, r=None, orth_coefs=None, 
                 norms=None, bas_np=None, num_basis=3, device='cpu'):
        super().__init__()
        self.device = device
        self.ortho_coeffs = []
        self.norm_factors = []
        self.layers = nn.ModuleList()

        if None in [basis_coef_init, r, orth_coefs, norms, bas_np]:
            self._initialize_from_scratch(x, y, num_basis)
        else:
            self._initialize_precomputed(x, y, basis_coef_init, r, 
                                       orth_coefs, norms, bas_np)

    def forward(self, x):
        basis = self._compute_basis(x)
        return basis @ self.coeff

    def _initialize_from_scratch(self, x, y, num_basis):
        """Initialize network with QR decomposition and random basis expansion."""
        # QR decomposition of [1|x]
        ones = torch.ones((x.shape[0], 1), device=x.device)
        design = torch.cat([ones, x], 1)
        q, r = torch.linalg.qr(design)
        self.qr_inverse = torch.inverse(r).to(self.device)
        
        # Create basis expansion layers
        input_dim = x.shape[1] + 1
        for i in range(num_basis - input_dim):
            self.layers.append(nn.Linear(input_dim + i, 1, bias=False))
            
        # Build orthogonal basis
        basis = self._build_initial_basis(q.to(self.device))
        self.coeff = (basis.T @ y.to(self.device)).detach()

    def _initialize_precomputed(self, x, y, weights, r, ortho, norms, basis_np):
        """Initialize with precomputed parameters."""
        self.qr_inverse = torch.inverse(torch.tensor(r, dtype=torch.float32))
        self.ortho_coeffs = [
            [torch.tensor(c, dtype=torch.float32) for c in layer_coeff]
            for layer_coeff in ortho
        ]
        self.norm_factors = [
            [torch.tensor(n, dtype=torch.float32) for n in layer_norms]
            for layer_norms in norms
        ]
        
        # Initialize layers with pretrained weights
        for w in weights:
            layer = nn.Linear(w.shape[1], 1, bias=False)
            layer.weight.data = torch.tensor(w, dtype=torch.float32)
            self.layers.append(layer)
            
        basis = torch.tensor(basis_np, dtype=torch.float32)
        self.coeff = (basis.T @ y.to(self.device)).detach()

    def _build_initial_basis(self, initial_basis):
        """Construct orthogonal basis during initialization."""
        current_basis = initial_basis
        for layer in self.layers:
            # Generate new basis component
            new_comp = F.relu(layer(current_basis))
            
            # Orthogonalize against existing basis
            proj_coeff = (current_basis.T @ new_comp).flatten()
            self.ortho_coeffs.append([proj_coeff.detach().to(self.device)])
            new_comp -= current_basis @ proj_coeff
            
            # Normalize and store
            norm = torch.norm(new_comp)
            self.norm_factors.append([norm.detach().to(self.device)])
            current_basis = torch.cat([current_basis, new_comp/norm], 1)
        return current_basis

    def _compute_basis(self, x):
        """Compute orthogonal basis during forward pass."""
        # Initial basis through QR inverse
        ones = torch.ones((x.shape[0], 1), device=self.device)
        design = torch.cat([ones, x], 1)
        basis = design @ self.qr_inverse

        for idx, layer in enumerate(self.layers):
            # Generate new component
            new_comp = F.relu(layer(basis))
            
            # Apply orthogonalization and normalization
            proj_coeff = self.ortho_coeffs[idx][0].to(self.device)
            norm = self.norm_factors[idx][0].to(self.device)
            new_comp -= (basis @ proj_coeff).unsqueeze(-1)
            basis = torch.cat([basis, new_comp/norm], 1)
            
        return basis

def init_weights(module):
    """Xavier initialization for linear layers."""
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)

def train_random_model(xs, func, num_basis, epochs, device='cpu'):
    """Train multiple models with random initializations.
    
    Returns:
        list: Relative errors from multiple training runs
    """
    targets = func(xs).flatten()
    xs_tensor = torch.tensor(xs, dtype=torch.float32, device=device)
    targets_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
    
    errors = []
    for _ in tqdm(range(20), desc="Training iterations"):
        model = AxonNetwork(
            xs_tensor.cpu(), targets_tensor.cpu(),
            num_basis=num_basis + xs.shape[1] + 1,
            device=device
        ).to(device)
        model.apply(init_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(epochs):
            pred = model(xs_tensor)
            loss = F.mse_loss(pred, targets_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_pred = model(xs_tensor)
            err = torch.norm(final_pred - targets_tensor) / torch.norm(targets_tensor)
            errors.append(err.item())
            
    return errors