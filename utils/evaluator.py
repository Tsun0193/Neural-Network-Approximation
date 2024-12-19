import torch
import numpy as np
from tqdm.auto import tqdm
from typing import Literal

def evaluate_model(model, 
                   test_loader: torch.utils.data.DataLoader,
                   show_progress: bool=True,
                   device: Literal['cpu', 'cuda']='cpu'):
    """
    Evaluate a PyTorch model.

    :param model: PyTorch model to evaluate
    :param test_loader: DataLoader object for test data
    :param device: device to use for evaluation, default='cpu'

    :return: predictions and actual values
    """
    print("Evaluation Configurations:")
    print(f"Device: {device}")
    print()

    model.eval()
    predictions = []
    actuals = []

    if show_progress:
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc='Evaluating', unit='batch'):
                inputs = inputs.to(device).float()
                outputs = model(inputs)
                predictions.append(outputs.cpu().numpy())
                actuals.append(targets.numpy())
    else:
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device).float()
                outputs = model(inputs)
                predictions.append(outputs.cpu().numpy())
                actuals.append(targets.numpy())

    return np.vstack(predictions), np.vstack(actuals)