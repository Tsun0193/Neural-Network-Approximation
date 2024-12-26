#!/usr/bin/env python
# coding: utf-8

# In[1]:


cd ..


# In[2]:


import torch
import matplotlib.pyplot as plt
from model.piecewise_linear_fn import ReluSegmentNetwork, FixedWidthReluNetwork
from utils.maths import *


# In[3]:


x_points = [0, 1, 2, 3, 4]
y_points = [0, 1, 0.5, 2.5, 1.5]

# plot
plt.plot(x_points, y_points, 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data points')
plt.show()


# In[4]:


model = ReluSegmentNetwork(
    x_points=x_points,
    y_points=y_points
)


# In[5]:


x = torch.linspace(-1, 10, 500)

with torch.no_grad():
    base_value, segment_outputs = model(x)

f_x = base_value + sum(segment_outputs)

original_f_x = torch.zeros_like(x)
for i in range(len(x_points) - 1):
    mask = (x >= x_points[i]) & (x < x_points[i + 1])
    original_f_x[mask] = (
        y_points[i]
        + (x[mask] - x_points[i])
        * (y_points[i + 1] - y_points[i])
        / (x_points[i + 1] - x_points[i])
    )
original_f_x[x >= x_points[-1]] = y_points[-1]  # Handle the last point


# In[6]:


fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

# Plot the original piecewise linear function by connecting the given x,y coordinates
axs[0].plot(x_points, y_points, label="Original Piecewise Linear Function", color="green", marker="o")
axs[0].set_title("Original Piecewise Linear Function")
axs[0].set_xlabel("x")
axs[0].set_ylabel("f(x)")
axs[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
axs[0].axvline(0, color="black", linewidth=0.5, linestyle="--")
axs[0].legend(loc="upper left")
axs[0].grid()

# Plot the reconstructed piecewise linear function
axs[1].plot(x.numpy(), f_x.numpy(), label="Reconstructed Function", color="blue")
axs[1].set_title("Reconstructed Piecewise Linear Function")
axs[1].set_xlabel("x")
axs[1].set_ylabel("f(x)")
axs[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
axs[1].axvline(0, color="black", linewidth=0.5, linestyle="--")
axs[1].legend(loc="upper left")
axs[1].grid()

plt.tight_layout()
plt.show()


# In[7]:


x_points = [1.2, 2.5, 3.7, 4.8, 6.0]
y_points = [1.5, 0.7, 2.8, 3.2, 9.0]

# plot
plt.plot(x_points, y_points, 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data points')
plt.show()


# In[8]:


fixed_model = FixedWidthReluNetwork(
    x_points=x_points,
    y_points=y_points
)


# In[9]:


x = torch.linspace(-1, 10, 500)

# Compute the function output
with torch.no_grad():
    f_x = fixed_model(x)

# Visualize the results
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

# Plot the original piecewise linear function by connecting the given x, y coordinates
axs[0].plot(x_points, y_points, label="Original Piecewise Linear Function", color="green", marker="o")
axs[0].set_title("Original Piecewise Linear Function")
axs[0].set_xlabel("x")
axs[0].set_ylabel("f(x)")
axs[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
axs[0].axvline(0, color="black", linewidth=0.5, linestyle="--")
axs[0].legend(loc="upper left")
axs[0].grid()

# Plot the reconstructed piecewise linear function
axs[1].plot(x.numpy(), f_x.numpy(), label="Reconstructed Function", color="blue")
axs[1].set_title("Reconstructed Piecewise Linear Function")
axs[1].set_xlabel("x")
axs[1].set_ylabel("f(x)")
axs[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
axs[1].axvline(0, color="black", linewidth=0.5, linestyle="--")
axs[1].legend(loc="upper left")
axs[1].grid()

plt.tight_layout()
plt.show()


# In[ ]:




