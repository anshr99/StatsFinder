###
## Here we test our modules and demonstrate Simple Linear Regression using 100 different values for Area of house in square feet and their respective prices
###

from pathlib import Path
from SLRM_StatsFinder import (
    load_data,
    beta_estimate,
    predict_y,
    residual_estimate,
    res_variance,
    plot_regression
)

def main():
        filename = Path(__file__).resolve().parent.parent / "data" / "house_price.csv" # locate CSV file
        Xm, Ym = load_data(filename) # construction of matrices
        beta = beta_estimate(Xm, Ym) # estimated beta vector
        mu = predict_y(Xm, beta) # calculate predicted values
        e = residual_estimate(Ym, mu) # calculate residuals
        sigma_sq = res_variance(Xm, e) # estimated residual variance
        print("Design Matrix:")
        print(Xm)
        print("Observed Y Values:")
        print(Ym)
        print("Estimated Beta Vector:")
        print(beta)
        print("Predicted Y Values:")
        print(mu)
        print("Residual Vector:")
        print(e)
        print("Estimated Residual Variance:")
        print(sigma_sq)
        plot_regression(Xm, Ym, mu, "House Area (sq ft)", "House Price")

if __name__ == "__main__":
    main()