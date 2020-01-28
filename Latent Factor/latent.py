import csv
import numpy as np
import math

# This function does Matrix Factorization with missing values using gradient descent and returns a latent factor model U & V

def matrix_factorization(user_movie_matrix, dimensions, ita=0.01, iterations=50, lambda_=10):
    """ ...
    ->user_movie_matrix: input matrix to be factorized, learn latent factor model from this matrix
    ->dimensions: the dimension for latent factor
    ->ita: the learning rate
    ->iterations: the maximum number of iterations to perform gradient descent
    ->lambda_: the regularization parameter
    ->Output:  U--latent factor model of dimension M*dimensions
    		   V--latent factor model of dimension dimensions*N 
    """
    m, n = user_movie_matrix.shape
    u = np.random.rand(m, dimensions)
    v = np.random.rand(dimensions, n)
    for iteration in range(iterations):
        ita_n = ita/math.sqrt(iteration+1)
        for i_user in range(m):
            for j_movie in range(n):
                # Only calculate non-missing values
                if user_movie_matrix[i_user][j_movie] > 0:
                    eij = user_movie_matrix[i_user][j_movie] - np.dot(u[i_user, :], v[:, j_movie])
                    # Gradient descent
                    for dimension in range(dimensions):
                        u[i_user][dimension] -= ita_n * (lambda_ * u[i_user][dimension] - 2 * v[dimension][j_movie] * eij)
                        v[dimension][j_movie] -= ita_n * (lambda_ * v[dimension][j_movie] - 2 * u[i_user][dimension] * eij)
        u_dot_v = np.dot(u, v)
        error = 0
        for i_user in range(m):
            for j_movie in range(n):
                if user_movie_matrix[i_user][j_movie] > 0:
                    error += (user_movie_matrix[i_user][j_movie] - u_dot_v[i_user, j_movie]) ** 2
                    # Frobenius norm
                    for dimension in range(dimensions):
                        error += lambda_ / 2 * (u[i_user][dimension] * 2 + v[dimension][j_movie] * 2)
        if error < 0.01:
            break
    return u, v