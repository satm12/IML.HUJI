from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from scipy.stats import multivariate_normal
from ...metrics import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m, n = X.shape if X.ndim > 1 else (X.size, 1)
        self.classes_ = np.unique(y)
        nk = np.array([np.sum(y == k) for k in self.classes_])
        self.pi_ = nk / m
        self.mu_ = np.array([sum(X[y == k]) for k in self.classes_]) / nk[:, np.newaxis]
        self.cov_ = np.zeros((n, n))
        for k in self.classes_:
            X_k = X[y == k]
            X_k_mean = np.mean(X_k, axis=0)
            self.cov_ += (X_k - X_k_mean).T.dot(X_k - X_k_mean)
        self.cov_ = (1.0 / (m - self.classes_.size)) * self.cov_
        self._cov_inv = inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # a = np.einsum('ij,kj->ki', self._cov_inv, self.mu_)

        a = self._cov_inv.dot(self.mu_.T)
        b = np.log(self.pi_) - 0.5 * np.einsum("ci,ij,cj->c", self.mu_, self._cov_inv, self.mu_)
        return self.classes_[np.argmax(a.T.dot(X.T).T + b, axis=1)]

        # return self.classes_[np.argmax(self.likelihood(X), axis=0)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        return np.array([multivariate_normal.pdf(X, mean=self.mu_[k], cov=self.cov_) * self.pi_[k]
                         for k in self.classes_])

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self.predict(X)
        return misclassification_error(y, y_pred)
