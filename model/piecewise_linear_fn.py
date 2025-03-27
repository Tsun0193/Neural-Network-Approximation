import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
from utils.maths import *
from typing import List

class ReluSegmentNetwork(nn.Module):
    def __init__(self, x_points, y_points):
        """
        Initialize the ReLU segment network.

        Parameters:
            x_points (list): List of x breakpoints [x1, x2, ..., xn].
            y_points (list): List of y values at the breakpoints [y1, y2, ..., yn].
        """
        super(ReluSegmentNetwork, self).__init__()
        self.x_points = torch.tensor(x_points, dtype=torch.float32)
        self.y_points = torch.tensor(y_points, dtype=torch.float32)
        self.slopes = self.calculate_slopes()

    def calculate_slopes(self):
        """
        Calculate the slopes for each segment based on x and y points.

        Returns:
            Tensor: Slopes for each segment.
        """
        slopes = [
            get_slope_by_2_points((self.x_points[i].item(), self.y_points[i].item()),
                                  (self.x_points[i + 1].item(), self.y_points[i + 1].item()))
            for i in range(len(self.x_points) - 1)
        ]
        return torch.tensor(slopes, dtype=torch.float32)

    def forward(self, x):
        """
        Forward pass to compute the piecewise linear function and individual segment outputs.

        Parameters:
            x (Tensor): Input x values.

        Returns:
            Tuple[Tensor, List[Tensor]]: The base value at x1 and a list of outputs for each segment.
        """
        base_value = self.y_points[0]  # f*n(x1)
        segment_outputs = []
        for i in range(len(self.slopes)):
            sign_k = torch.sign(self.slopes[i])
            abs_k = torch.abs(self.slopes[i])
            x_i, x_next = self.x_points[i], self.x_points[i + 1]
            O_i = sign_k * F.relu(abs_k * (F.relu(x - x_i) - F.relu(x - x_next)))
            segment_outputs.append(O_i)
        return base_value, segment_outputs
    

class FixedWidthReluNetwork(nn.Module):
    def __init__(self, x_points, y_points):
        """
        Initialize the fixed-width ReLU network.

        Parameters:
            x_points (list): List of x breakpoints [x1, x2, ..., xn].
            y_points (list): List of y values at the breakpoints [y1, y2, ..., yn].
        """
        super(FixedWidthReluNetwork, self).__init__()
        self.x_points = torch.tensor(x_points, dtype=torch.float32)
        self.y_points = torch.tensor(y_points, dtype=torch.float32)
        self.slopes = self.calculate_slopes()

    def calculate_slopes(self):
        """
        Calculate the slopes for each segment based on x and y points.

        Returns:
            Tensor: Slopes for each segment.
        """
        slopes = []
        for i in range(len(self.x_points) - 1):
            dx = self.x_points[i + 1] - self.x_points[i]
            dy = self.y_points[i + 1] - self.y_points[i]
            slopes.append(dy / dx)
        return torch.tensor(slopes, dtype=torch.float32)

    def forward(self, x):
        """
        Forward pass for the fixed-width architecture.

        Parameters:
            x (Tensor): Input x values.

        Returns:
            Tensor: Final output of the function.
        """
        f_x = torch.full_like(x, self.y_points[0])  # Initialize with f*n(x1)
        for i in range(len(self.slopes)):
            x_i, x_next = self.x_points[i], self.x_points[i + 1]
            sigma_x_minus_xi = F.relu(x - x_i)
            sigma_x_minus_xnext = F.relu(x - x_next)
            O_i = self.slopes[i] * (sigma_x_minus_xi - sigma_x_minus_xnext)
            f_x += O_i
        return f_x
