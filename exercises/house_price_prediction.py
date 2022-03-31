from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from typing import NoReturn
from utils import *
import sys
sys.path.append("../")
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)

    # remove invalid prices
    df = df.drop(df[df.price <= 0].index)
    df = df.drop(df[df['price'].isnull()].index)

    # remove inconsistent data
    df = df[df['bedrooms'] * 50 < df['sqft_living']]

    df['zipcode'] = df.zipcode.astype('int')

    # convert to datetime
    df = df.dropna(subset=['date'])
    df['date_str'] = df['date'].astype('str')
    df = df.drop(df[df.date_str.str.len() < 5].index)
    df = df.drop(columns=['date_str'])
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%dT%H%M%S')

    # calculate house age
    df['age'] = df.date.dt.year - df.yr_built

    # calculate is renovated
    df['renovated'] = (df['yr_renovated'] > 5).astype(int)

    # calculate renovation age
    df['renovation_age'] = pd.DataFrame([df["date"].dt.year - df["yr_renovated"], df["age"]]).min()

    # ecnode zipcodes
    df = df.join(pd.get_dummies(df.zipcode, prefix='zipcode'))

    # choose fetures
    zipcodes = [name for name in df.columns if name.startswith('zipcode_')]
    fetures = ['bedrooms', 'bathrooms', 'sqft_living',
               'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
               'sqft_above', 'sqft_basement', 'age', 'renovated', 'renovation_age',
               'sqft_living15', 'sqft_lot15'] + zipcodes

    y_df = df['price']
    x_df = df[fetures]

    y_df = y_df.reset_index(drop=True)
    x_df = x_df.reset_index(drop=True)

    return x_df, y_df


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    def pearson_corr(x, y):
        return x.cov(y) / (x.std() * y.std())

    for feature in X.columns:
        figure = px.scatter(x=X[feature], y=y,
                            title=f'Feature: "{feature}" Pearson Correlation: {pearson_corr(X[feature], y):.2f}',
                            labels={'x': 'value', 'y': 'price'},
                            opacity=0.4)
        figure.write_image(f'{output_path}/{feature}-plot.png')


def measure_run(train_X, train_y, test_X, test_y):
    model = LinearRegression()
    model.fit(train_X, train_y)
    return model.loss(test_X, test_y)


def generate_sample(percent, train_X, train_y):
    reset_X = train_X.reset_index(drop=True)
    reset_y = train_y.reset_index(drop=True)

    indices = reset_X.sample(frac=percent / 100.0).index
    sample_X = reset_X.iloc[indices]
    sample_y = reset_y.iloc[indices]
    return sample_X, sample_y


def measure_loss(percent, train_X, train_y, test_X, test_y):
    measures = np.array([measure_run(*generate_sample(percent, train_X, train_y), test_X, test_y)
                         for _ in range(10)])
    return [measures.mean(), measures.std()]


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data('../datasets/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    df = pd.DataFrame([[p, *measure_loss(p, train_X, train_y, test_X, test_y)] for p in range(10, 101)],
                      columns=['percent', 'mean', 'std'])
    x = df['percent']
    mean_pred = df['mean']
    std_pred = df['std']

    fig = go.Figure(data=[
        go.Scatter(x=x, y=mean_pred, mode="markers+lines", name="Mean Prediction Loss", line=dict(dash="dash"),
                   marker=dict(color="green", opacity=.7)),
        go.Scatter(x=x, y=mean_pred - 2 * std_pred, fill=None, mode="lines", line=dict(color="lightgrey"),
                   showlegend=False),
        go.Scatter(x=x, y=mean_pred + 2 * std_pred, fill='tonexty', mode="lines", line=dict(color="lightgrey"),
                   showlegend=False)],
                    layout=dict(
                        xaxis={"title": 'Training Sample Percent'},
                        yaxis={"title": 'Loss'},
                    ))
    fig.show()
