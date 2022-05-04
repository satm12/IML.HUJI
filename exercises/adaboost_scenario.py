import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics.loss_functions import accuracy

def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    model = AdaBoost(DecisionStump, n_learners)
    model.fit(train_X, train_y)
    x = np.arange(n_learners) + 1
    trian_loss = [model.partial_loss(train_X, train_y, i) for i in range(1, n_learners)]
    test_loss = [model.partial_loss(test_X, test_y, i) for i in range(1, n_learners)]

    fig = go.Figure(data=[
        go.Scatter(x=x,
                   y=trian_loss,
                   name='train',
                   mode="lines"),
        go.Scatter(x=x,
                   y=test_loss,
                   name='test',
                   mode="lines")
    ], layout=go.Layout(title="Plot Title",
                        xaxis=dict(title="Learners Count"),
                        yaxis=dict(title="Loss")))
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.05, .05])

    def partial_predict(ada, learner_count):
        return lambda X: ada.partial_predict(X, learner_count)

    model_predictions = [partial_predict(model, t) for t in T]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{t} Learners}}$" for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.05)
    for i, m in enumerate(model_predictions):
        fig.add_traces([decision_surface(m, lims[0], lims[1], showscale=False)],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.update_layout(title=rf"$\textbf{{Decision Boundaries Of Model}}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    best_test = np.array(test_loss).argmin()
    acc = accuracy(test_y, model.partial_predict(test_X, best_test))

    fig = go.Figure(data=[
        decision_surface(partial_predict(model, best_test), lims[0], lims[1], showscale=False)])
    fig.update_layout(title=rf"$\textbf{{Model Boundaries With {best_test + 1} Learners. Accuracy: {acc}}}$",
                      margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 4: Decision surface with weighted samples
    fig = px.scatter(x=train_X[:, 0], y=train_X[:, 1], size=5 * model.D_ / np.max(model.D_), color=train_y.astype(str))
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)


