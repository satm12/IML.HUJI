from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def question1_model(x):
    return (x+3)*(x+2)*(x+1)*(x-1)*(x-2)


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.sort(np.random.uniform(-1.2, 2, n_samples))
    y = question1_model(X) + np.random.normal(0, np.sqrt(noise), n_samples)
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), 2.0 / 3.0)
    train_X, train_y, test_X, test_y = train_X[0].to_numpy(), train_y.to_numpy(), test_X[0].to_numpy(), test_y.to_numpy()

    fig = go.Figure(data=[go.Scatter(x=X,
                                     y=question1_model(X),
                                     name='noiseless',
                                     mode="lines",
                                     line=dict(
                                         color='black'
                                     )),
                          go.Scatter(x=train_X,
                                     y=train_y,
                                     name='train',
                                     mode="markers"
                                     ),
                          go.Scatter(x=test_X,
                                     y=test_y,
                                     name='train',
                                     mode="markers"
                                     )
                          ])
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_scores = []
    validation_scores = []
    for d in range(10):
        model = PolynomialFitting(d)
        train_score, validation_score = cross_validate(model, train_X, train_y, mean_square_error)
        train_scores.append(train_score)
        validation_scores.append(validation_score)

    fig = go.Figure(data=[go.Scatter(x=np.arange(10),
                                     y=train_scores,
                                     name='train scores',
                                     mode="markers"),
                          go.Scatter(x=np.arange(10),
                                     y=validation_scores,
                                     name='validation scores',
                                     mode="markers"),
                          ])
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    opt_d = np.argmin(validation_scores)
    model = PolynomialFitting(opt_d)
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    test_error = mean_square_error(y_pred, test_y)
    print(f'Optimal Degree: {opt_d}\nTest Error: {test_error:.2}')


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(1500, 10)
