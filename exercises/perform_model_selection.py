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
    data = datasets.load_diabetes(as_frame=True).frame
    train, test = data.iloc[:n_samples], data.iloc[n_samples:]
    train_X, train_y = train.drop(columns='target'), train['target']
    test_X, test_y = test.drop(columns='target'), test['target']

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    results = []
    for lam in np.linspace(0, 10, n_evaluations + 1)[1:]:
        ridge_model = RidgeRegression(lam=lam, include_intercept=True)
        lasso_model = Lasso(alpha=lam)
        ridge_train_score, ridge_val_score = cross_validate(ridge_model, train_X, train_y, mean_square_error)
        lasso_train_score, lasso_vali_score = cross_validate(lasso_model, train_X.to_numpy(), train_y.to_numpy(), mean_square_error)
        results.append([lam, ridge_train_score, ridge_val_score, lasso_train_score, lasso_vali_score])
    result_data = pd.DataFrame(results, columns=['lam', 'ridge_train', 'ridge_val', 'lasso_train', 'lasso_val'])

    fig = go.Figure(data=[go.Scatter(x=result_data['lam'],
                                     y=result_data['ridge_train'],
                                     name='Ridge Train',
                                     mode="lines",
                                     line=dict(
                                         color='darkblue'
                                     )),
                          go.Scatter(x=result_data['lam'],
                                     y=result_data['ridge_val'],
                                     name='Ridge Validation',
                                     mode="lines",
                                     line=dict(
                                         color='blue'
                                     )),
                          go.Scatter(x=result_data['lam'],
                                     y=result_data['lasso_train'],
                                     name='Lasso Train',
                                     mode="lines",
                                     line=dict(
                                         color='darkred'
                                     )),
                          go.Scatter(x=result_data['lam'],
                                     y=result_data['lasso_val'],
                                     name='Lasso Validation',
                                     mode="lines",
                                     line=dict(
                                         color='red'
                                     ))
                          ],
                    layout=dict(
                        title=rf"$\textbf{{Ridge/Lasso Corss Validation Scores}}$",
                        xaxis={"title": 'Lambda'},
                        yaxis={"title": 'Loss'},
                    ))
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    opt_lam_ridge = result_data.loc[result_data['ridge_val'].idxmin()]['lam']
    opt_lam_lasso = result_data.loc[result_data['lasso_val'].idxmin()]['lam']
    print(f'Optimal Ridge Parameter: {opt_lam_ridge}\nOptimal Lasso Parameter: {opt_lam_lasso}')

    lin_model = LinearRegression(include_intercept=True)
    lin_model.fit(train_X.to_numpy(), train_y.to_numpy())
    lin_error = mean_square_error(lin_model.predict(test_X.to_numpy()), test_y.to_numpy())

    ridge_model = RidgeRegression(lam=opt_lam_ridge, include_intercept=True)
    ridge_model.fit(train_X.to_numpy(), train_y.to_numpy())
    ridge_error = mean_square_error(ridge_model.predict(test_X.to_numpy()), test_y.to_numpy())

    lasso_model = Lasso(alpha=opt_lam_lasso)
    lasso_model.fit(train_X.to_numpy(), train_y.to_numpy())
    lasso_error = mean_square_error(lasso_model.predict(test_X.to_numpy()), test_y.to_numpy())

    print(f'Linear Error Error: {lin_error}\nRidge Error: {ridge_error}\nLasso Error: {lasso_error:}')


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(100, 0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()
