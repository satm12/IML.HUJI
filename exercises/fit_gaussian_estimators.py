from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    sigma = 1
    sample_size = 1000
    samples = np.random.normal(mu, sigma, sample_size)
    model = UnivariateGaussian()
    model.fit(samples)
    print(f'({model.mu_}, {model.var_})')

    # Question 2 - Empirically showing sample mean is consistent
    def estimated_expected_value(samples):
        model = UnivariateGaussian()
        model.fit(samples)
        return model.mu_
    
    sample_counts = np.arange(10, sample_size+1, 10)
    expected_values = np.array([estimated_expected_value(samples[:i]) for i in sample_counts])
    distance = abs(expected_values - mu)
    
    x = sample_counts
    y = distance
    fig = px.line(x=x, y=y, 
                  title='Estimation Distance', 
                  labels={'x':'sample size', 'y':'estimation error'})
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    x = samples
    y = model.pdf(samples)
    fig = px.scatter(x=x, y=y, title='Sample PDFs', labels={'x':'sample', 'y':'pdf'})
    fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1.0, 0.2, 0, 0.5],
                      [0.2, 2.0, 0,   0],
                      [0,   0,   1,   0],
                      [0.5, 0,   0,   1]])
    sample_size = 1000
    samples = np.random.multivariate_normal(mu, sigma, sample_size)
    
    model = MultivariateGaussian()
    model.fit(samples)
    print(model.mu_)
    print(model.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    # very slow try to vectorize
    data = np.zeros((200, 200))
    for i, v1 in enumerate(f1):
        for j, v3 in enumerate(f3):
            data[i, j] = MultivariateGaussian.log_likelihood(np.array([v1, 0, v3, 0]), sigma, samples)
    
    fig = px.imshow(data, labels=dict(x="f3", y="f1", color="Log-likelihood"), 
                    x=f1, y=f3)
    fig.show()

    # Question 6 - Maximum likelihood
    max_index = np.unravel_index(data.argmax(), data.shape)
    max_param = (f1[max_index[0]], f3[max_index[1]])
    print(f'({max_param[0]:.3}, {max_param[1]:.3})')


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
