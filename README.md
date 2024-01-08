# README for h_index.py

## Overview
`h_index` contains the implementation of a mathematical computation algorithm. The main function in this script is `compute_index`, which performs a series of computations and returns a pandas DataFrame. `data_generating` is the function for us to generate the database. `H_myopic` is the function for myopic normalized gain, which would be called in the other two functions. And `data_generating` is the file we generated the database for table 4 and 5. `h_index` is user-friendly. Please use `h_index` unless you need to perform a large times of calculations.

## Dependencies
To run this script, the following Python libraries are required:
- `math`: Standard Python library for basic mathematical operations.
- `numpy`: Essential library for numerical computations in Python.
- `pandas`: Library providing high-performance, easy-to-use data structures and data analysis tools.
- `scipy`: Library used for scientific and technical computing.


## Functions
The script includes several functions, of which the most notable are:
- `H_myopic`: A function for myopic normalized gain.(Equation (7))
- `compute_index`: A function for h-index and this version is for users.
- `data_generating`: The same function as `compute_index` and this function is for generating the database.

##Parameters
The `compute_index` function takes the following parameters:

- `recall`: whether the function includes a recall mechanism in its operation.
  - `0`: Without Recall.
  - `1`: With Recall.
- `mu`: whether the mean is known
  - `0`: unknown mean.
  - `1`: known mean.
- `sigma`: whether the variance is known
  - `0`: unknown variance.
  - `1`: known variance.
- `alpha0`: $alpha_0$. 2$alpha_0$ (the degrees of freedom) is the pseudo-count of independent data implicit in our knowledge of the variance
- `nu0`: $nu_0$ is pseudo-count of data implicit in our knowledge of the mean. The value must be non-negative(>=0).
- `n`: specifies the sample size. The value must be a positive integer
- `G`: specifies the size of the grid used in the function. The value must be a positive integer

The `compute_index` function generates the following parameters during the execution:
- `z`: 'offer' list: range from [-30,30], increasing in index
- `c`: 'serach cost' list: range from (0,0.85*G], decreasing in index
- `H`: 3-array gain of searching: $H(k, j_z, j_c)$
- `h`: internal cost function: $h(k, j_z)$
- `H_u`: $H_{n,k+1}\left(\frac{z_u-\mu_u}{\sigma_u},0,1;\frac{c}{\sigma_u}\right)$
- `s`: $sigma_u$
- `mu_u`: $\mu_u$
- `L`: $\Lambda_{k+1}$
- `theta_z`: $\theta_z$ for bilinear interpolation
- `theta_c`: $\theta_c$ for bilinear interpolation

## Usage
To use the functions in this script, import the script into your Python environment and call the functions with appropriate parameters. Example:
```python
import h_index

# Example of using 'compute_index' function from the script
df = compute_index()
#input: recall, mu, sigma, alpha0, nu0, n, G
print(df)
df.to_csv('test1.csv', header=True, index=True)

# Example of using 'data_generating' function from the script
# Assign: recall, mu, sigma, alpha0, nu0, n, G
recall = 1
mu = 0
sigma = 0
alpha0 = -0.5  
nu0 = 0
n = 10
G = 40

df2=data_generating(recall, mu, sigma, alpha0, nu0, n, G)
df2.to_csv('test2.csv', header=True, index=True)
```

Replace the arguments with relevant values as per your data and analysis needs.


## Note on Algorithm Performance and G Value
- **Performance Scaling:** A larger G value will improve the accuracy of the calculations. But choosing a larger G value will significantly increase the running time. The running time of the algorithm approximately increases by the third power of the growth multiple of G.
- **Recommended G Value:** Combining runtime and accuracy, the recommended G value is 57. The G value we used for table 4 and 5 is 285.
- **Estimated Time:** With a G value of 57, the estimated running time of the algorithm is approximately 4-5 minutes.


This README is based on a preliminary review of the script. For detailed information and specific use cases, refer to the inline comments and documentation within the script itself.
