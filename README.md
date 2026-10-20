# StatsFinder

`StatsFinder` is a small Python package that performs Simple Linear Regression using Ordinary Least Squares (OLS) and matrix operations.

The package provides functions to:

- Load CSV datasets and construct the design matrix.
- Estimate regression coefficients using the OLS method.
- Calculate predicted Y values.
- Calculate the residual vector.
- Estimate the residual variance ($\hat{\sigma}^2$).
- Visualise the observed values, predicted values and line of best fit.

NumPy is used for matrix operations and regression calculations. Matplotlib is used to visualise the regression results.

## Installation

From inside the `StatsFinder` directory, install the package in editable mode:

```sh
python -m pip install -e .
```

The required dependencies are specified in `pyproject.toml` and will be installed automatically.

## Examples of use

A Simple Linear Regression model can be fitted by importing the functions from `SLRM_StatsFinder`:

```python
from SLRM_StatsFinder import (
    load_data,
    beta_estimate,
    predict_y,
    residual_estimate,
    sigma_square_estimate
)

Xm, Ym = load_data("data/exam_score.csv")

beta = beta_estimate(Xm, Ym)
mu = predict_y(Xm, beta)
e = residual_estimate(Ym, mu)
sigma_square = sigma_square_estimate(Xm, e)

print("Estimated beta vector:")
print(beta)

print("Predicted Y values:")
print(mu)

print("Residual vector:")
print(e)

print("Estimated residual variance:")
print(sigma_square)
```

The observed and predicted values can be visualised as follows:

```python
from SLRM_StatsFinder import plot_regression

plot_regression(Xm, Ym, mu, "Study Hours", "Exam Score")
```

The plot displays the original observations, predicted Y values and line of best fit.

## Demonstration

Two example datasets are provided:

- `exam_score.csv` contains study hours and examination scores.
- `house_price.csv` contains house areas and corresponding prices.

Both datasets contain synthetic numerical observations for demonstrating Simple Linear Regression.

The exam score demonstration can be run from the root directory using:

```sh
python demo/exam_demo.py
```

The house price demonstration can be run using:

```sh
python demo/house_price_demo.py
```

Each demonstration estimates the regression coefficients, calculates predicted values and residuals, estimates the residual variance, and displays a regression plot.