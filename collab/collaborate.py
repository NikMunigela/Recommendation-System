import time
import numpy as np
from collaborative_filtering.similarities import pearson_sim as sim
from collaborative_filtering.constants import INT_MIN

class Collaborate:
    """
    This class holds function to perform collaboratie filtering with and without baseline approach
    """
    def __init__(self, M):
        """
        Initialize user-movie(ratings) matrix
        Note: Matrix needs to have items as rows and users as columns

        """
        self.M = M

    def estimate(self, user, item, k=2, baseline=False):
        """
        This function estimates the rating for a given input user and respective item.

        """
        # Ratings matrix
        r = self.M
        # Mean baseline deviation
        mu = 0
        # User baseline deviation
        b_user = 0
        # Item baseline deviation
        b_item = 0
        # With baseline deviation considered
        if baseline is True:
            mu = sum(sum(r))/np.count_nonzero(r)
            b_user = sum(r[:, user])/np.count_nonzero(r[:, user]) - mu
            b_item = sum(r[item])/np.count_nonzero(r[item]) - mu
        # Overall baseline deviation
        b = mu + b_user + b_item
        ### Estimating the Rating ###
        # Calculating similarities
        S = np.zeros(r.shape[0])
        for i in range(r.shape[0]):
            S[i] = sim(r, item, i)
            if np.isnan(S[i]):
                S[i] = 0
        # Rating estimation
        numerator = 0
        s_list = list(S)
        max = sorted(S, reverse=True)[1:3]

        max_idx = list()
        for i in max:
            if i != 1:
                if len(max_idx) == k:
                    break
                else:
                    max_idx.append(s_list.index(i))

        denominator = np.sum(max)

        for i in max_idx:
            if baseline:
                b_ui = b_user + sum(r[i])/np.count_nonzero(r[i])
            else:
                b_ui = 0
            numerator += (r[i, user] - b_ui)*S[i]

        rating = b + (numerator/denominator)
        if np.isnan(rating):
            rating = 0

        return rating

    def fill(self, k=2, baseline=False):
        """
        This function, based on collaborative filtering estimates fills the gaps in the input matrix 

        """
        # Filling in the matrix
        filled = np.zeros(self.M.shape)
        for i in range(self.M.shape[0]):
            for j in range(self.M.shape[1]):
                if self.M[i, j] == 0:
                    filled[i, j] = self.estimate(j, i, k=k, baseline=baseline)
                else:
                    filled[i, j] = self.M[i, j]
        return filled
