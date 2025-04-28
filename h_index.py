import numpy as np
from scipy.stats import t, norm
from numba import njit
import pandas as pd
import math

@njit
def _t_pdf(x, df):
    """
    Compute the probability density function (PDF) of the t-distribution.
    
    Parameters:
    -----------
    x : float
        The value at which to evaluate the PDF
    df : float
        Degrees of freedom
    
    Returns:
    --------
    float
        The probability density at x
    """
    return (1 + x**2/df)**(-(df+1)/2) * math.gamma((df+1)/2) / (math.sqrt(df*math.pi) * math.gamma(df/2))

@njit
def _norm_pdf(x):
    """
    Compute the probability density function (PDF) of the normal distribution.
    
    Parameters:
    -----------
    x : float
        The value at which to evaluate the PDF
    
    Returns:
    --------
    float
        The probability density at x
    """
    return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

@njit
def _t_cdf(x, df):
    """
    Compute the cumulative distribution function (CDF) of the t-distribution.
    
    Parameters:
    -----------
    x : float
        The value at which to evaluate the CDF
    df : float
        Degrees of freedom
    
    Returns:
    --------
    float
        The cumulative probability at x
    """
    if x == 0:
        return 0.5
    elif x > 0:
        return 1 - 0.5 * (1 - math.erf(x/math.sqrt(2)))
    else:
        return 0.5 * (1 - math.erf(-x/math.sqrt(2)))

@njit
def _norm_cdf(x):
    """
    Compute the cumulative distribution function (CDF) of the normal distribution.
    
    Parameters:
    -----------
    x : float
        The value at which to evaluate the CDF
    
    Returns:
    --------
    float
        The cumulative probability at x
    """
    return 0.5 * (1 + math.erf(x/math.sqrt(2)))

@njit
def update_parameters(z_k, mu_k, sigma_k, y, k, alpha0, nu0, beta0, mu0):
    """
    Update the parameters z_k, mu_k, and sigma_k based on the new observation y.
    
    Parameters:
    -----------
    z_k : float
        Current best offer
    mu_k : float
        Current mean
    sigma_k : float
        Current standard deviation
    y : float
        New observation
    k : int
        Current iteration number
    alpha0 : float
        Initial alpha parameter
    nu0 : float
        Initial nu parameter
    beta0 : float
        Initial beta parameter
    mu0 : float
        Initial mean
    
    Returns:
    --------
    tuple
        (z_k_plus, mu_k_plus, sigma_k_plus) where:
        - z_k_plus is the updated best offer
        - mu_k_plus is the updated mean
        - sigma_k_plus is the updated standard deviation
    """
    # Update z_k
    z_k_plus = max(z_k, y)
    
    # Update mu_k
    mu_k_plus = mu_k + (y - mu_k) / (nu0 + k + 1)
    
    # Update sigma_k
    if k < 3:  # k_0 = 3
        # For k < k_0, use the given formula
        nu_k = nu0 + k
        alpha_k = alpha0 + k/2
        beta_k = beta0 + (k * nu0 / (nu0 + k)) * ((mu_k - mu0)**2)
        sigma_k_plus = np.sqrt((1 + 1/nu_k) * (2*beta_k / (2*alpha_k)))
    else:
        # For k >= k_0, use the update formula
        L_k_plus = np.sqrt((1 - (1/(nu0 + k + 1))**2) / (2*alpha0 + k + 1))
        sigma_k_plus = L_k_plus * np.sqrt((2*alpha0 + k) * sigma_k**2 + (y - mu_k)**2)
    
    return z_k_plus, mu_k_plus, sigma_k_plus

@njit
def compute_initial_parameters(x_values, alpha0, nu0, beta0, mu0):
    """
    Compute initial parameters based on the first k_0 observations.
    
    Parameters:
    -----------
    x_values : np.ndarray
        Array of observations (should have at least 3 values)
    alpha0 : float
        Initial alpha parameter
    nu0 : float
        Initial nu parameter
    beta0 : float
        Initial beta parameter
    mu0 : float
        Initial mean
    
    Returns:
    --------
    tuple
        (z_k, mu_k, sigma_k) where:
        - z_k is the best offer so far
        - mu_k is the computed mean
        - sigma_k is the computed standard deviation
    """
    k = len(x_values)
    if k < 3:
        raise ValueError("Need at least 3 observations to compute initial parameters")
    
    # Compute mean
    x_bar = np.mean(x_values)
    
    # Compute mu_k
    mu_k = (nu0 * mu0 + k * x_bar) / (nu0 + k)
    
    # Compute beta_k
    sum_squared_diff = np.sum((x_values - x_bar)**2)
    beta_k = beta0 + sum_squared_diff + (k * nu0 / (nu0 + k)) * ((x_bar - mu0)**2)
    
    # Compute alpha_k
    alpha_k = alpha0 + k/2
    
    # Compute nu_k
    nu_k = nu0 + k
    
    # Compute sigma_k
    sigma_k = np.sqrt((1 + 1/nu_k) * (2*beta_k / (2*alpha_k)))
    
    # Compute z_k (best offer so far)
    z_k = np.max(x_values)
    
    return z_k, mu_k, sigma_k

@njit
def H_myopic_jit(recall, sigma_flag, z, k, alpha0):
    """
    Compute the myopic H-function value.
    
    Parameters:
    -----------
    recall : int
        1 for with recall, 0 for without recall
    sigma_flag : int
        0 for unknown variance, 1 for known variance
    z : float
        The standardized value
    k : int
        Current iteration number
    alpha0 : float
        Initial alpha parameter
    
    Returns:
    --------
    float
        The myopic H-function value
    """
    if recall == 1:
        if sigma_flag == 0:
            df = 2 * alpha0 + k
            return ((df + z**2) / (df - 1)) * _t_pdf(z, df) - z * (1 - _t_cdf(z, df))
        else:  # sigma known
            return _norm_pdf(z) - z * (1 - _norm_cdf(z))
    else:  # recall = 0
        return -z

def build_grids(G, rho=0.85, Z=30, ita=0.75):
    """
    Build the grids for c and z values.
    
    Parameters:
    -----------
    G : int
        Grid size
    rho : float, optional
        Decay parameter for c grid, default is 0.85
    Z : float, optional
        Scale parameter for z grid, default is 30
    ita : float, optional
        Shape parameter for z grid, default is 0.75
    
    Returns:
    --------
    tuple
        (c, z) where:
        - c is the grid of cost values
        - z is the grid of standardized values
    """
    c = np.zeros(G+2)
    z = np.zeros(G+2)
    for j in range(G+2):
        c[j] = G * rho**j
        z[j] = Z * ((1-ita)*(2*j - G - 1)/(G-1) + ita*((2*j - G - 1)/(G-1))**3)
    c[G+1] = 0
    return c, z

@njit
def h_index_recall1(mu_flag, sigma_flag, alpha0, nu0, n, G, c, z):
    """
    Compute the h-index values for the recall=1 case.
    
    Parameters:
    -----------
    mu_flag : int
        0 for unknown mean, 1 for known mean
    sigma_flag : int
        0 for unknown variance, 1 for known variance
    alpha0 : float
        Initial alpha parameter
    nu0 : float
        Initial nu parameter
    n : int
        Number of iterations
    G : int
        Grid size
    c : np.ndarray
        Grid of cost values
    z : np.ndarray
        Grid of standardized values
    
    Returns:
    --------
    np.ndarray
        Matrix of h-index values
    """
    H = np.zeros((n+1, G+2, G+2))
    h = np.zeros((n+1, G+2))
    
    if sigma_flag == 0:
        k_min = max(np.floor(2 - 2*alpha0), 1)
    else:
        k_min = 1
    k_min = int(k_min)
    
    for k in range(n-1, k_min-1, -1):
        for j_z in range(G, 0, -1):
            for j_c in range(G, 0, -1):
                H[k, j_z, j_c] = H_myopic_jit(1, sigma_flag, z[j_z], k, alpha0)
                if k < n-1:
                    for j_u in range(G, 0, -1):
                        if mu_flag == 0:
                            mu_u = 1 / (nu0 + k + 1)
                        else:
                            mu_u = 0
                        if sigma_flag == 0:
                            L = np.sqrt((1 - mu_u**2) / (2*alpha0 + k + 1))
                            s = L * np.sqrt(2*alpha0 + k + z[j_u]**2)
                        else:
                            s = np.sqrt(1 - mu_u**2)
                        z_new = (max(z[j_z], z[j_u]) - z[j_u]*mu_u) / s
                        if k == n-2:
                            H_u = H_myopic_jit(1, sigma_flag, z_new, k+1, alpha0)
                        else:
                            j_1 = G
                            while j_1 > 1 and z_new < z[j_1]:
                                j_1 -= 1
                            if j_1 == G:
                                j_1 = G-1
                            j_2 = G
                            while j_2 > 1 and c[j_c]/s > c[j_2]:
                                j_2 -= 1
                            if j_2 == G:
                                j_2 = G-1
                            theta_z = (z_new - z[j_1]) / (z[j_1+1] - z[j_1])
                            theta_c = (c[j_c]/s - c[j_2]) / (c[j_2+1] - c[j_2])
                            H_u = (1-theta_c)*((1-theta_z)*H[k+1,j_1,j_2] + theta_z*H[k+1,j_1+1,j_2]) + theta_c*((1-theta_z)*H[k+1,j_1,j_2+1] + theta_z*H[k+1,j_1+1,j_2+1])
                        if sigma_flag == 0:
                            density = _t_pdf(z[j_u], df=2*alpha0+k)
                        else:
                            density = _norm_pdf(z[j_u])
                        dz = (z[j_u+1] - z[j_u-1]) / 2
                        H[k, j_z, j_c] += s * max(0, H_u - c[j_c]/s) * density * dz

            j = G
            while j > 1 and c[j] < H[k, j_z, j]:
                j -= 1
            if j == G:
                j = G-1
            if (H[k, j_z, j+1] - H[k, j_z, j] + c[j] - c[j+1]) == 0:
                h[k, j_z] = c[j]
            else:
                h[k, j_z] = c[j] + (c[j+1] - c[j]) * (c[j] - H[k,j_z,j]) / (H[k,j_z,j+1] - H[k,j_z,j] + c[j] - c[j+1])
    
    return h

def h_index_full(recall, mu_flag, sigma_flag, alpha0, nu0, n, G):
    """
    Solve the full h-index table.
    
    Parameters:
    -----------
    recall : int
        1 for with recall, 0 for without recall
    mu_flag : int
        0 for unknown mean, 1 for known mean
    sigma_flag : int
        0 for unknown variance, 1 for known variance
    alpha0 : float
        Initial alpha parameter
    nu0 : float
        Initial nu parameter
    n : int
        Number of iterations
    G : int
        Grid size
    
    Returns:
    --------
    tuple
        (h_matrix, z_grid) where:
        - h_matrix is the matrix of h-index values
        - z_grid is the grid of standardized values
    """
    c, z_grid = build_grids(G)
    if recall == 1:
        h_matrix = h_index_recall1(mu_flag, sigma_flag, alpha0, nu0, n, G, c, z_grid)
    else:
        raise NotImplementedError("recall=0 case not yet optimized in Numba version.")
    return h_matrix, z_grid

def h_index_value(h_matrix, z_grid, n, k, z_val):
    """
    Retrieve the interpolated h(k, z) value from h_matrix and z_grid.
    
    Parameters:
    -----------
    h_matrix : np.ndarray
        Matrix of h-index values
    z_grid : np.ndarray
        Grid of standardized values
    n : int
        Number of iterations
    k : int
        Current iteration number
    z_val : float
        The standardized value to evaluate
    
    Returns:
    --------
    float
        The interpolated h-index value
    """
    G = len(z_grid) - 2
    if k > n or k < 0:
        raise ValueError(f"k must be between 0 and {n}")

    j = G
    while j > 0 and z_val < z_grid[j]:
        j -= 1
    if j == G:
        j = G-1

    if (z_grid[j+1] - z_grid[j]) == 0:
        theta = 0
    else:
        theta = (z_val - z_grid[j]) / (z_grid[j+1] - z_grid[j])
    
    h_interp = (1-theta) * h_matrix[k, j] + theta * h_matrix[k, j+1]
    return h_interp

if __name__ == "__main__":
    # Initial parameters
    alpha0 = 0.5
    nu0 = 1.0
    beta0 = 1.0
    mu0 = 0.0
    c = 0.1  # adjust the cost as you want
    max_iterations = 30
    
    # Step 1: Solve the full h-index table (only need to do once)
    h_matrix, z_grid = h_index_full(recall=1, mu_flag=0, sigma_flag=0, alpha0=-0.5, nu0=0, n=30, G=200)
    
    # Step 2: Get initial observations (k_0 = 3)
    x_values = np.array([0.1, -0.4, 0.2])  # Replace with your actual initial observations
    
    # Step 3: Compute initial parameters
    z_k, mu_k, sigma_k = compute_initial_parameters(x_values, alpha0, nu0, beta0, mu0)
    print(f"Initial parameters (k=3):")
    print(f"z_k = {z_k:.4f}")
    print(f"mu_k = {mu_k:.4f}")
    print(f"sigma_k = {sigma_k:.4f}")
    
    # Step 4: Sampling mechanism
    k = 3  # Start from k_0
    stop = False
    all_observations = list(x_values)  # Store all observations
    
    while not stop and k < max_iterations:
        # Compute standardized z and cost
        z_val = (z_k - mu_k) / sigma_k
        c_value = c / sigma_k
        
        # Get h-value
        h_val = h_index_value(h_matrix, z_grid, n=30, k=k, z_val=z_val)
        print(f"\nIteration k={k}:")
        print(f"Standardized z = {z_val:.4f}")
        print(f"Cost value = {c_value:.4f}")
        print(f"h-value = {h_val:.4f}")
        
        # Decision: continue or stop
        if h_val > c_value:
            # Continue: draw new sample and update parameters
            # Here you would typically draw a new sample from your distribution
            # For demonstration, we'll generate a random sample
            new_sample = np.random.normal(mu_k, sigma_k)
            all_observations.append(new_sample)
            
            # Update parameters
            z_k, mu_k, sigma_k = update_parameters(z_k, mu_k, sigma_k, new_sample, k, alpha0, nu0, beta0, mu0)
            print(f"New sample = {new_sample:.4f}")
            print(f"Updated parameters:")
            print(f"z_k = {z_k:.4f}")
            print(f"mu_k = {mu_k:.4f}")
            print(f"sigma_k = {sigma_k:.4f}")
            
            k += 1
        else:
            # Stop: sampling ends
            stop = True
            print("\nSampling stopped:")
            print(f"Final k = {k}")
            print(f"Final z_k = {z_k:.4f}")
            print(f"Final mu_k = {mu_k:.4f}")
            print(f"Final sigma_k = {sigma_k:.4f}")
            print(f"Total samples collected: {len(all_observations)}")
            print(f"All observations: {np.array(all_observations)}")
