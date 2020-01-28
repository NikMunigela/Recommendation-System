import numpy as np


def rmse(M, M_p):
    """
    This function is used to compute the Root Mean Square Error.

    """
    x_len = M.shape[0]
    y_len = M.shape[1]
    error = 0
    N = x_len * y_len
    for x in range(x_len):
        for y in range(y_len):
            error += ((M[x][y] - M_p[x][y]) ** 2) / N
    error = error ** 0.5
    return error

def mae(M, M_p):
	x_len = M.shape[0]
	y_len = M.shape[1]
	error = 0
	N = x_len * y_len
	for x in range(x_len):
	    for y in range(y_len):
                error += abs(M_p[x][y] - M[x][y]) / N	
		
	return error
