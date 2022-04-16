from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import IMLearn.metrics.loss_functions as metrics


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f'../datasets/{f}')

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def track_loss(perceptron, _, __):
            losses.append(perceptron.loss(X, y))

        model = Perceptron(callback=track_loss)
        model.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        x = np.arange(1, len(losses) + 1)
        y = np.array(losses)
        fig = px.line(x=x, y=y, title=f'{n} - Model Loss', labels={'x': 'iteration number', 'y': 'model loss'})
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """

    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f'../datasets/{f}')

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)
        lda_pred = lda.predict(X)
        lda_acc = metrics.accuracy(y, lda_pred)

        naive_bayes = GaussianNaiveBayes()
        naive_bayes.fit(X, y)
        naive_pred = naive_bayes.predict(X)
        naive_acc = metrics.accuracy(naive_pred, y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        make_subplots(1, 2, subplot_titles=(f'LDA Predictions - Accuracy: {lda_acc}',
                                            f'Naive Bayes Predictions - Accuracy: {naive_acc}')) \
            .add_traces([go.Scatter(x=X[:, 0], y=X[:, 1],
                                    mode="markers", showlegend=False,
                                    marker=dict(color=lda_pred, symbol=y,
                                                size=15,
                                                line=dict(color="black", width=2),
                                                colorscale=[custom[0], custom[-1]]
                                                ))
                         ], rows=1, cols=1) \
            .add_traces([get_ellipse(lda.mu_[i], lda.cov_)
                         for i in range(lda.classes_.size)],
                        rows=1, cols=1) \
            .add_traces([go.Scatter(x=[lda.mu_[i][0]], y=[lda.mu_[i][1]],
                                    mode="markers", showlegend=False,
                                    marker=dict(color='black',
                                                symbol='x',
                                                size=10
                                                )) for i in range(lda.classes_.size)],
                        rows=1, cols=1) \
            .add_traces([go.Scatter(x=X[:, 0], y=X[:, 1],
                                    mode="markers", showlegend=False,
                                    marker=dict(color=naive_pred, symbol=y,
                                                size=15,
                                                line=dict(color="black", width=2),
                                                colorscale=[custom[0], custom[-1]]
                                                ))
                         ], rows=1, cols=2) \
            .add_traces([get_ellipse(naive_bayes.mu_[i], np.diag(naive_bayes.vars_[i]))
                         for i in range(naive_bayes.classes_.size)],
                        rows=1, cols=2) \
            .add_traces([go.Scatter(x=[naive_bayes.mu_[i][0]], y=[naive_bayes.mu_[i][1]],
                                    mode="markers", showlegend=False,
                                    marker=dict(color='black',
                                                symbol='x',
                                                size=10
                                                )) for i in range(lda.classes_.size)],
                        rows=1, cols=2) \
            .update_traces(showlegend=False) \
            .update_layout(title=f'Classification Predictions - {f}', margin=dict(t=100)).show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
