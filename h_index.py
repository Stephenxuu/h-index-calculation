import math
import numpy as np
import pandas as pd
from scipy.stats import t , norm

# Function to compute the myopic H-index
def H_myopic(recall, sigma, z, k, alpha0):
    if recall == 1:
        if sigma == 0:
            return ((2 * alpha0 + k + z**2) / (2 * alpha0 + k - 1)) * t.pdf(z, df=2 * alpha0 + k) - z * (1 - t.cdf(z, df=2 * alpha0 + k))
        if sigma == 1:
            return norm.pdf(z) - z * (1 - norm.cdf(z))
    elif recall ==0:
        return -z

# Main Function to compute the h-index (For users)
def compute_index():    
    """
    This function performs a series of computations and returns a pandas DataFrame of h[k,j_z].

    It prompts the user to input the parameters for the computation.

    Returns:
    pandas.DataFrame: The result of the computations.
    """
     # Get user inputs for recall, mu, sigma, alpha0, nu0, n, G
    recall = int(input("Enter 0 or 1: 0 for without recall, 1 for with recall "))
    while recall != 0 and recall != 1:
        print("Please enter 0 or 1")
        recall = int(input("Enter 0 or 1: 0 for without recall, 1 for with recall "))
    mu = int(input("Enter 0 or 1: 0 for unknown mean, 1 for known mean"))
    while mu != 0 and mu != 1:
        print("Please enter 0 or 1")
        mu = int(input("Enter 0 or 1: 0 for unknown mean, 1 for known mean"))
    sigma = int(input("Enter 0 or 1: 0 for unknown variance, 1 for known variance "))
    while sigma != 0 and sigma != 1:
        print("Please enter 0 or 1")
        sigma = int(input("Enter 0 or 1: 0 for unknown variance, 1 for known variance "))
    alpha0 = float(input("Enter the alpha0 parameter (float): "))
    nu0 = float(input("Enter the nu0 parameter (>=0, float): "))
    while nu0 < 0:
        print("nu0 must be greater than or equal to 0")
        nu0 = float(input("Enter the nu0 parameter (>=0, float): "))
    if sigma==0:
        k_min= max(math.floor(2 - 2 * alpha0),1)
    elif sigma==1:
        k_min= 1
    n = int(input(f"Enter the number of iterations to perform (>= {k_min}, integer): "))
    while n < k_min:
        print(f"Number of iterations must be greater than or equal to {k_min}")
        n = int(input("Enter the number of iterations to perform (integer): "))
    G = int(input("Enter the grid size (>0, integer): "))
    while G <= 0:
        print("Grid size must positive integer")
        G = int(input("Enter the grid size (>0, integer): "))

    # Initialize arrays for H, h, c, z
    H = np.zeros((n+1, G+2, G+2))
    h = np.zeros((n+1, G+2))
    c = np.zeros(G+2)
    z = np.zeros(G+2)
    
    # Set initial values for k, delta, Z, ita
    k=n-1
    delta=0.85
    Z=30
    ita=0.75
    
    # Compute c and z values
    for j in range(G+2):
        c[j] = G*delta**j
        z[j] = Z * ((1-ita)* (2 * j - G - 1) / (G - 1) + ita * ((2 * j - G - 1) / (G - 1))**3) 
    
    # Computation loop 1: k
    while k >= k_min:
        j_z=G
        # Computation loop 2: j_z
        while j_z > 0:
            j_c=G
            # Computation loop 3: j_c
            while j_c > 0:
                H[k,j_z, j_c] = H_myopic(recall, sigma, z[j_z], k, alpha0)
                if k < n - 1:
                    j_u=G
                    # Computation loop 2: j_u
                    while j_u > 0:
                        if mu == 0:
                            mu_u = 1/ (nu0 + k + 1)
                        elif mu==1:
                            mu_u = 0
                        if sigma == 0:
                            L = np.sqrt((1 - mu_u**2) / (2 * alpha0 + k + 1))
                            s = L * np.sqrt(2 * alpha0 + k + z[j_u]**2)
                        elif sigma == 1:
                            s = np.sqrt(1 - mu_u**2)
                        if recall == 1:
                            z_new = (max(z[j_z], z[j_u]) - z[j_u] * mu_u) / s
                        elif recall == 0:
                            z_new = (z[j_u] - z[j_u] * mu_u) / s
                        if k == n - 2:
                            H_u = H_myopic(recall, sigma, z_new, k+1, alpha0)
                        else:
                            j_1, j_2 = G , G 
                            while j_1 >1 and z_new < z[j_1]: #when j_z=57, j_u=1, z_new>z[57]
                                j_1 -= 1
                            if j_1==G:
                                j_1 = G-1
                            while j_2 >1 and c[j_c] / s > c[j_2]: # what if j_c==1
                                j_2 -= 1
                            if j_2==G:
                                j_2 = G-1
                            theta_z = (z_new - z[j_1]) / (z[j_1 + 1] - z[j_1])
                            theta_c = (c[j_c] / s - c[j_2]) / (c[j_2 + 1] - c[j_2])
                            H_u = (1 - theta_c) * ((1 - theta_z) * H[k+1, j_1, j_2] + theta_z * H[k+1, j_1 + 1, j_2]) + theta_c * ((1 - theta_z) * H[k+1, j_1, j_2 + 1] + theta_z * H[k+1, j_1 + 1, j_2 + 1])
                        if sigma == 0:
                            H[k, j_z, j_c] += s * max(0, H_u - c[j_c] / s) * t.pdf(z[j_u], df=2 * alpha0 + k) * (z[j_u + 1] - z[j_u - 1])/2
                        elif sigma == 1:
                            H[k, j_z, j_c] += s * max(0, H_u - c[j_c] / s) * norm.pdf(z[j_u]) * (z[j_u + 1] - z[j_u - 1])/2
                        j_u -= 1
                j_c -= 1
            j = G
            while j > 1 and c[j] < H[k, j_z, j]:
                j -= 1
            if j == G:
                j = G-1
            h[k, j_z] = c[j] + (c[j + 1] - c[j]) * (c[j] - H[k, j_z, j]) / (H[k, j_z, j + 1] - H[k, j_z, j] + c[j] - c[j + 1])    
            j_z -= 1
        k -= 1
    
    # Create DataFrame from h array
    df=pd.DataFrame(h)

    # Filter the rows
    df_filtered = df[(df.index >= k_min) & (df.index < n)]

    # Filter the columns
    df_filtered = df_filtered.loc[:, (df_filtered.columns >= 1) & (df_filtered.columns <= G)]

    # Set the column and row labels
    column_labels = ['z={:.1f}'.format(z[j]) for j in range(1,G+1)]
    row_labels = ['n={}, k_min={}, k={}'.format(n, k_min, i) for i in range(k_min,n)]
    df_filtered.columns = column_labels
    df_filtered.index = row_labels
    return df_filtered

# Function to compute the h-index (For generating data)
def data_generating(recall, mu, sigma, alpha0, nu0, n, G):    
    """
    This function performs a series of computations and returns a pandas DataFrame of h[k,j_z].

    It prompts the user to input the parameters for the computation.

    Returns:
    pandas.DataFrame: The result of the computations.
    """
    # Initialize arrays for H, h, c, z
    H = np.zeros((n+1, G+2, G+2))
    h = np.zeros((n+1, G+2))
    c = np.zeros(G+2)
    z = np.zeros(G+2)
    
    # Set initial values for k, delta, Z, ita
    k=n-1
    delta=0.85
    Z=30
    ita=0.75
    
    # calculate k_min
    if sigma==0:
        k_min= max(math.floor(2 - 2 * alpha0),1)
    elif sigma==1:
        k_min= 1
    
    # Compute c and z values
    for j in range(G+2):
        c[j] = G*delta**j
        z[j] = Z * ((1-ita)* (2 * j - G - 1) / (G - 1) + ita * ((2 * j - G - 1) / (G - 1))**3) 
    
    # Computation loop 1: k
    while k >= k_min:
        j_z=G
        # Computation loop 2: j_z
        while j_z > 0:
            j_c=G
            # Computation loop 3: j_c
            while j_c > 0:
                H[k,j_z, j_c] = H_myopic(recall, sigma, z[j_z], k, alpha0)
                if k < n - 1:
                    j_u=G
                    # Computation loop 2: j_u
                    while j_u > 0:
                        if mu == 0:
                            mu_u = 1/ (nu0 + k + 1)
                        elif mu==1:
                            mu_u = 0
                        if sigma == 0:
                            L = np.sqrt((1 - mu_u**2) / (2 * alpha0 + k + 1))
                            s = L * np.sqrt(2 * alpha0 + k + z[j_u]**2)
                        elif sigma == 1:
                            s = np.sqrt(1 - mu_u**2)
                        if recall == 1:
                            z_new = (max(z[j_z], z[j_u]) - z[j_u] * mu_u) / s
                        elif recall == 0:
                            z_new = (z[j_u] - z[j_u] * mu_u) / s
                        if k == n - 2:
                            H_u = H_myopic(recall, sigma, z_new, k+1, alpha0)
                        else:
                            j_1, j_2 = G , G 
                            while j_1 >1 and z_new < z[j_1]: #when j_z=57, j_u=1, z_new>z[57]
                                j_1 -= 1
                            if j_1==G:
                                j_1 = G-1
                            while j_2 >1 and c[j_c] / s > c[j_2]: # what if j_c==1
                                j_2 -= 1
                            if j_2==G:
                                j_2 = G-1
                            theta_z = (z_new - z[j_1]) / (z[j_1 + 1] - z[j_1])
                            theta_c = (c[j_c] / s - c[j_2]) / (c[j_2 + 1] - c[j_2])
                            H_u = (1 - theta_c) * ((1 - theta_z) * H[k+1, j_1, j_2] + theta_z * H[k+1, j_1 + 1, j_2]) + theta_c * ((1 - theta_z) * H[k+1, j_1, j_2 + 1] + theta_z * H[k+1, j_1 + 1, j_2 + 1])
                        if sigma == 0:
                            H[k, j_z, j_c] += s * max(0, H_u - c[j_c] / s) * t.pdf(z[j_u], df=2 * alpha0 + k) * (z[j_u + 1] - z[j_u - 1])/2
                        elif sigma == 1:
                            H[k, j_z, j_c] += s * max(0, H_u - c[j_c] / s) * norm.pdf(z[j_u]) * (z[j_u + 1] - z[j_u - 1])/2
                        j_u -= 1
                j_c -= 1
            j = G
            while j > 1 and c[j] < H[k, j_z, j]:
                j -= 1
            if j == G:
                j = G-1
            h[k, j_z] = c[j] + (c[j + 1] - c[j]) * (c[j] - H[k, j_z, j]) / (H[k, j_z, j + 1] - H[k, j_z, j] + c[j] - c[j + 1])    
            j_z -= 1
        k -= 1
    
    # Create DataFrame from h array
    df=pd.DataFrame(h)

    # Filter the rows
    df_filtered = df[(df.index >= k_min) & (df.index < n)]

    # Filter the columns
    df_filtered = df_filtered.loc[:, (df_filtered.columns >= 1) & (df_filtered.columns <= G)]

    # Set the column and row labels
    column_labels = ['z={:.1f}'.format(z[j]) for j in range(1,G+1)]
    row_labels = ['n={}, k_min={}, k={}'.format(n, k_min, i) for i in range(k_min,n)]
    df_filtered.columns = column_labels
    df_filtered.index = row_labels
    return df_filtered