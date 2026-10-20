###
## Here we test our modules and demonstrate Simple Linear Regression using 25 values of student study hours and their respective examination scores
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
        filename = Path(__file__).resolve().parent.parent / "data" / "exam_score.csv" # locate CSV file
        Xm, Ym = load_data(filename) # construction of matrices
        beta = beta_estimate(Xm, Ym) # estimated beta vector
        mu = predict_y(Xm, beta) # calculate predicted values
        e = residual_estimate(Ym, mu) # calculate residuals
        sigma_sq = res_variance(Xm, e) # estimated residual variance
        print("Design Matrix X:")
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
        plot_regression(Xm, Ym, mu, "Study Hours", "Exam Score")

if __name__ == "__main__":
    main()