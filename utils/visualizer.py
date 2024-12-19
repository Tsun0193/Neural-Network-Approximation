import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython import get_ipython
from scipy.stats import pearsonr

def plot_results_1d(X_train, y_train, X_test, y_test, y_pred, 
                    history, title, plot_size=(12, 5),
                    save_path: str = None):
    """
    Plot training history and function approximation results for 1D data.
    
    :param X_train: training data input
    :param y_train: training data output
    :param X_test: test data input
    :param y_test: test data output
    :param y_pred: predicted output
    :param history: training history
    :param title: plot title
    :param plot_size: size of the plot, default=(12, 5)
    :param save_path: path to save the plot, default=None
    
    :return: None
    """

    plt.figure(figsize=plot_size)
    
    # Plot training history
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    
    # Plot function approximation
    plt.subplot(1, 2, 2)
    plt.scatter(X_train, y_train, label='Train Data', color='blue', alpha=0.5)
    plt.scatter(X_test, y_test, label='Test Data', color='green', alpha=0.5)
    plt.scatter(X_test, y_pred, label='Predictions', color='red', alpha=0.5)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('f(X)')
    plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        if get_ipython():
            plt.show()
        else:
            print('Plot displayed in non-interactive mode. Saving to file instead. Please provide a save_path.')

    plt.close()

def plot_results_2d(X_test, y_test, y_pred, 
                    title,
                    history, 
                    plotsize=(15, 5), 
                    save_path: str = None):
    """
    Plot training history, actual vs predicted values with correlation, 
    and residuals histogram for 2D data.
    
    Parameters
    ----------
    X_test : np.ndarray of shape (n_samples, 2)
        The test input points.
    y_test : np.ndarray of shape (n_samples,)
        The actual output function values.
    y_pred : np.ndarray of shape (n_samples,)
        The predicted output function values.
    title : str
        Title for the entire figure.
    history : dict
        Dictionary containing 'train_loss' and 'val_loss'.
    plotsize : tuple (default=(15,5))
        Figure size.
    save_path : str or None
        Path to save the figure. If None, shows the plot.
    """
    
    plt.figure(figsize=plotsize)

    # Plot Training and Validation Loss
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()

    # Scatter Plot of Actual vs Predicted with Correlation
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    
    # Calculate Pearson Correlation
    corr_coef, p_value = pearsonr(y_test, y_pred)
    plt.text(0.05, 0.95, f'Pearson r = {corr_coef:.3f}', 
             transform=plt.gca().transAxes, 
             fontsize=12, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    # Residuals Histogram
    plt.subplot(1, 3, 3)
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, bins=30, color='skyblue')
    plt.title('Residuals Histogram')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

    plt.close()