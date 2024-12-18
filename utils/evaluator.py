import torch
import numpy as np
from tqdm.auto import tqdm

def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate a PyTorch model.

    :param model: PyTorch model to evaluate
    :param test_loader: DataLoader object for test data
    :param device: device to use for evaluation, default='cpu'

    :return: predictions and actual values
    """
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating', unit='batch'):
            inputs = inputs.to(device).float()
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.numpy())

    return np.vstack(predictions), np.vstack(actuals)