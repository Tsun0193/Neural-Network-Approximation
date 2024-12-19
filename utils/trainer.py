import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from typing import Literal

def train_model(model, train_loader, val_loader, 
                epochs: int=100,
                lr: float=0.001, 
                show_progress: bool=True,
                device: Literal['cpu', 'cuda']='cpu'):
    """
    Train a PyTorch model.

    :param model: PyTorch model to train
    :param train_loader: DataLoader object for training data
    :param val_loader: DataLoader object for validation data
    :param epochs: number of epochs to train, default=100
    :param lr: learning rate, default=0.001
    :param device: device to use for training, default='cpu'

    :return: history of training and validation loss
    """
    print("Training Configurations:")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {lr}")
    print()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    history = {'train_loss': [], 'val_loss': []}
    
    if show_progress:
        for epoch in tqdm(range(epochs), desc='Training', unit='epoch'):
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device).float(), targets.to(device).float()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            train_loss /= len(train_loader.dataset)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device).float(), targets.to(device).float()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
            val_loss /= len(val_loader.dataset)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
    else:
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device).float(), targets.to(device).float()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            train_loss /= len(train_loader.dataset)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device).float(), targets.to(device).float()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
            val_loss /= len(val_loader.dataset)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if (epoch+1) % 10 == 0 or epoch == 0:
                print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')
    
    return history