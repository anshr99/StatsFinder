###
## Here we compute the estimated beta vector, predicted Y values, residual vector and residual variance for a Simple Linear Regression Model
###

## Steps to achieve target
# 1. Load CSV files and construst Design Matrix
# 2. Estimate beta coefficient values
# 3. Calculate predicted values of the model
# 4. Calcuate residual values

import numpy as np

def load_data(filename):
    # Load the CSV file
    data = np.loadtxt(filename, delimiter=",", skiprows=1, ndmin=2)
    # Extraction of x and y values
    x = data[:, 0]
    y = data[:, 1]
    # Construct design matrix
    Xm = np.column_stack((np.ones(len(x)), x)) # design matrix is a matrix with column 1 having values (1, 1, ..., 1) and column 2 is vector xi
    # Construct response column vector
    Ym = y.reshape(-1, 1)
    return Xm, Ym

def beta_estimate(Xm, Ym):
    # the formula to estimate beta using matrix notation is B = (X^t * X)^-1 * (X^t * Y) using marrix notation
        Xt = np.transpose(Xm)
        beta = np.linalg.inv(Xt @ Xm) @ (Xt @ Ym) 
        return beta

def predict_y(Xm, beta):
    # the formula to calcuate predicted Y (mu) for a given X goes by MUi = X * B using matrix notation
        mu = Xm @ beta
        return mu

def residual_estimate(Ym, mu):
    # the residual vector denoted by 'e' in Linear Regression is the difference between Y-actual and Y-predicted
    e = Ym - mu
    return e

def res_variance(Xm, e):
      # residual variance/mean square error denoted by 'Sigma^2' is calculated by the formula 1/(n-2) * Sum(e^2)
      n = Xm.shape[0]
      sigma_sq = np.sum(e**2) / (n - 2)
      return sigma_sq

