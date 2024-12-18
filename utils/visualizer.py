import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython import get_ipython

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

def plot_results_2d(X_test, y_test, y_pred, title, 
                    plotsize=(12, 5), 
                    save_path: str = None):
    """
    Plot actual vs predicted values and error heatmap for 2D data.
    
    :param X_test: test data input
    :param y_test: test data output
    :param y_pred: predicted output
    :param title: plot title
    :param plotsize: size of the plot, default=(12, 5)
    :param save_path: path to save the plot, default=None
    
    :return: None
    """
    
    plt.figure(figsize=plotsize)
    
    # Plot actual vs predicted
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_test.flatten(), y=y_pred.flatten(), alpha=0.5)
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    
    # Plot error heatmap
    plt.subplot(1, 2, 2)
    error = np.abs(y_test - y_pred)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=error.flatten(), cmap='viridis', alpha=0.5)
    plt.title('Error Heatmap')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.colorbar(label='Absolute Error')
    
    plt.tight_layout()
    plt.suptitle(title)
    if save_path:
        plt.savefig(save_path)
    else:
        if get_ipython():
            plt.show()
        else:
            print('Plot displayed in non-interactive mode. Saving to file instead. Please provide a save_path.')

    plt.close()