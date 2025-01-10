# Function Approximation with ReLU Neural Networks

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Setup Instructions](#setup-instructions)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Create a Virtual Environment](#2-create-a-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
- [Running Experiments](#running-experiments)
  - [Experiment 1: Reproduce [1] - 1D Convex Function Approximation](#experiment-1-reproduce-1d-convex-function-approximation)
  - [Experiment 2: Extend to 1D Non-Convex Function Approximation](#experiment-2-extend-to-1d-non-convex-function-approximation)
  - [Experiment 3A: 2D Convex Function Approximation](#experiment-3a-2d-convex-function-approximation)
  - [Experiment 3B: 2D Non-Convex Function Approximation](#experiment-3b-2d-non-convex-function-approximation)
  - [Compare Results from [1] and [2]](#compare-results-from-1-and-2)
- [Results and Plots](#results-and-plots)
- [Extending the Project](#extending-the-project)
- [References](#references)
- [License](#license)
- [Additional Notes](#additional-notes)

## Project Overview

This project aims to approximate functions \( f : [0, 1] \rightarrow \mathbb{R} \) and \( f : [0, 1]^2 \rightarrow \mathbb{R} \) using deep neural networks (DNNs) with ReLU activation functions. Leveraging the universal approximation capabilities of ReLU-based DNNs, this project explores their effectiveness in approximating both convex and non-convex functions in one and two dimensions.

The project builds upon the theoretical foundations and experimental results from the following papers:

1. **[1]** Bo Liu, Yi Liang (2021). Optimal function approximation with ReLU neural networks, *Neurocomputing*, Volume 435.
2. **[2]** Fokina, Daria and Oseledets, Ivan. (2023). Growing axons: greedy learning of neural networks with application to function approximation. *Russian Journal of Numerical Analysis and Mathematical Modelling*, 38. 1-12.
3. **[3]** [Axon Algorithm Implementation](https://github.com/dashafok/axon-approximation)

## Directory Structure

## Setup Instructions

### 1. Clone the Repository

Begin by cloning the repository to your local machine:

```bash
git clone https://github.com/Tsun0193/Neural-Network-Approximation.git
cd function_approximation
```