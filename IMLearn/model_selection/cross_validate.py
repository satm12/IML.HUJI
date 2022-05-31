from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    X_split = np.array_split(X, cv)
    y_split = np.array_split(y, cv)
    train_scores = []
    validation_scores = []
    for i in range(cv):
        train_X = np.concatenate(np.delete(X_split, i, axis=0))
        train_y = np.concatenate(np.delete(y_split, i, axis=0))
        validation_X, validation_y = X_split[i], y_split[i]
        estimator.fit(train_X, train_y)
        y_train_pred = estimator.predict(train_X)
        y_validation_pred = estimator.predict(validation_X)
        train_scores.append(scoring(y_train_pred, train_y))
        validation_scores.append(scoring(y_validation_pred, validation_y))
    return np.average(train_scores), np.average(validation_scores)



