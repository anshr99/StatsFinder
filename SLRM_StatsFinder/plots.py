###
## Here we plot the results of our Simple Linear Regression Model
###

## The plot should display:
# 1. original observed values
# 2. predicted Y values
# 3. line of best fit

import numpy as np
import matplotlib.pyplot as plt

def plot_regression(Xm, Ym, mu, xlabel, ylabel):
    x = Xm[:, 1] # extracting x values
    y = Ym.flatten() # extracting y values
    y_predicted = mu.flatten() # extracting Y-predicted values
    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, label = "observed data", color = "red") # plot observed data
    plt.scatter(x, y_predicted, label = "predicted data", color = "blue") # plot predicted data
    plt.plot(x, y_predicted, label = "line of best fit", color = "green") # plot line of best fit
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Simple Linear Regression")
    plt.legend()
    plt.grid(True)
    plt.show()
