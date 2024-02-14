
# NetRate Algorithm Implementation in Python

## Overview
This GitHub repository contains a Python implementation of the NetRate algorithm, as detailed in the research paper "Uncovering the Temporal Dynamics of Diffusion Networks". NetRate is a pivotal tool for analyzing the dynamics of information, influence, and disease spread in networks, focusing particularly on the temporal aspects of diffusion processes.

## Contents
1. **NetRate.ipynb**: A Jupyter Notebook with the Python implementation of the NetRate algorithm. It includes:
   - Importing essential libraries like CVXPY for convex optimization and NumPy for numerical operations.
   - Data processing functions for managing network cascades and adjacency matrices.
   - Network estimation functions based on different diffusion models, including log-survival functions and constraints for convex optimization.
   - Core implementation of the NetRate algorithm, covering initialization, cascade processing, convex optimization, and estimation of the network structure.

2. **NetRate.pdf**: The original research paper explaining the theory and mathematical concepts behind the NetRate algorithm.

## Usage
1. Clone the repository.
2. Install Python and required libraries (CVXPY, NumPy, etc.).
3. Open `NetRate.ipynb` in a Jupyter environment.
4. Execute the cells sequentially to explore the implementation and its workflow.

## Implementation Details
The notebook guides through the process of estimating a network's structure using various diffusion models. It begins with library imports, followed by defining functions for data processing and network estimation. Each function is documented with its purpose and methodology. The final section executes the NetRate algorithm, applying convex optimization to deduce network connections from observed data.

## Research Paper
The paper "Uncovering the Temporal Dynamics of Diffusion Networks" introduces the NetRate algorithm, designed to infer network edges and estimate node transmission rates from observed node infection times in cascade data. The algorithm models diffusion as discrete networks of continuous temporal processes at varying rates, aiming to enable forecasting, influencing, and mitigating infections in a wide context.

**Note**: Expand this README with installation instructions, dependencies, usage examples, and other relevant details for users and contributors.
