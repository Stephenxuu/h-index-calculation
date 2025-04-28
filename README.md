# Fast H-Index Computation

This repository contains an optimized implementation of the h-index computation algorithm from the paper "Search in the Dark: The Case with Recall and Gaussian Learning" by Manel Baucells and Sasa Zorc.

## License

This code is provided under the following license terms from the original paper:

Copyright (c) 2024 Manel Baucells and Sasa Zorc

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
2. Any use of this software must cite the original paper:
   ```
   Baucells, M., & Zorc, S. (2024). Search in the Dark: The Case with Recall and Gaussian Learning.
   ```

## Description

This implementation provides a fast and optimized version of the h-index computation algorithm using Numba for parallel processing. The code includes:

1. Statistical functions for t and normal distributions
2. Parameter update functions for the sequential search process
3. H-index computation with and without recall
4. A sampling mechanism for sequential decision making

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Numba
- Pandas

## Installation

```bash
pip install numpy scipy numba pandas
```

## Usage

The main functionality is implemented in `fast_h_index.py`. Here's a basic example:

```python
from fast_h_index import h_index_full, h_index_value, compute_initial_parameters, update_parameters

# Initialize parameters
alpha0 = 0.5
nu0 = 1.0
beta0 = 1.0
mu0 = 0.0
c = 0.1  # cost parameter
n = 30   # maximum iterations
G = 200  # grid size

# Solve the full h-index table
h_matrix, z_grid = h_index_full(recall=1, mu_flag=0, sigma_flag=0, 
                               alpha0=alpha0, nu0=nu0, n=n, G=G)

# Get initial observations
x_values = np.array([0.1, -0.4, 0.2])

# Compute initial parameters
z_k, mu_k, sigma_k = compute_initial_parameters(x_values, alpha0, nu0, beta0, mu0)

# Sampling mechanism
k = 3  # Start from k_0
stop = False
all_observations = list(x_values)

while not stop and k < n:
    # Compute standardized z and cost
    z_val = (z_k - mu_k) / sigma_k
    c_value = c / sigma_k
    
    # Get h-value
    h_val = h_index_value(h_matrix, z_grid, n=n, k=k, z_val=z_val)
    
    # Decision: continue or stop
    if h_val > c_value:
        # Continue: draw new sample and update parameters
        new_sample = np.random.normal(mu_k, sigma_k)
        all_observations.append(new_sample)
        
        # Update parameters
        z_k, mu_k, sigma_k = update_parameters(z_k, mu_k, sigma_k, new_sample, 
                                             k, alpha0, nu0, beta0, mu0)
        k += 1
    else:
        stop = True
```

## Functions

### Statistical Functions
- `_t_pdf(x, df)`: t-distribution PDF
- `_norm_pdf(x)`: normal distribution PDF
- `_t_cdf(x, df)`: t-distribution CDF
- `_norm_cdf(x)`: normal distribution CDF

### Parameter Update Functions
- `update_parameters(z_k, mu_k, sigma_k, y, k, alpha0, nu0, beta0, mu0)`: Updates parameters based on new observation
- `compute_initial_parameters(x_values, alpha0, nu0, beta0, mu0)`: Computes initial parameters

### H-Index Computation Functions
- `H_myopic_jit(recall, sigma_flag, z, k, alpha0)`: Computes myopic H-function
- `build_grids(G, rho=0.85, Z=30, ita=0.75)`: Builds c and z grids
- `h_index_recall1(mu_flag, sigma_flag, alpha0, nu0, n, G, c, z)`: Computes h-index for recall=1
- `h_index_full(recall, mu_flag, sigma_flag, alpha0, nu0, n, G)`: Solves full h-index table
- `h_index_value(h_matrix, z_grid, n, k, z_val)`: Retrieves interpolated h-value

## Citation

If you use this code in your research, please cite:

```
Baucells, M., & Zorc, S. (2024). Search in the Dark: The Case with Recall and Gaussian Learning.
```

## Author

This implementation is based on the work of Manel Baucells and Sasa Zorc.

## Acknowledgments

Thanks to the authors for their original work and for making their research available to the community.
