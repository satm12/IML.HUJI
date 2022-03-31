import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from utils import *
pio.templates.default = "simple_white"

import sys
sys.path.append("../")


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    df = df.dropna(subset=['Temp'])
    df = df[df['Temp'] > -70]
    df['DayOfYear'] = df.Date.dt.dayofyear
    df["DYear"] = df["Year"].astype(str)

    return df


def measure_poly_loss(k, train_X, train_y, test_X, test_y):
    model = PolynomialFitting(k)
    model.fit(train_X, train_y)
    return round(model.loss(test_X, test_y), 2)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('../datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    idf = df[df['Country'] == 'Israel']

    fig = px.scatter(idf, x='DayOfYear', y="Temp", color="DYear")
    fig.update_traces(marker=dict(size=6), selector=dict(mode='markers'))
    fig.show()

    month_std = idf.groupby(['Month'])['Temp'].std()
    fig = px.bar(df, x=month_std.index, y=month_std, labels={'x': 'Month', 'y': 'Tempeture std'})
    fig.show()

    # Question 3 - Exploring differences between countries
    agg_df = df.groupby(['Country', 'Month'])['Temp'].agg(['mean', 'std'])

    figures = [go.Scatter(x=agg_df.loc[country].index,
                          y=agg_df.loc[country]['mean'],
                          name=country,
                          mode="lines",
                          error_y=dict(
                              type='data',  # value of error bar given in data coordinates
                              array=agg_df.loc[country]['std'],
                              visible=True
                          )
                          ) for country in agg_df.index.unique(level='Country')]

    fig = go.Figure(data=figures)
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    X = df[df['Country'] == 'Israel']['DayOfYear']
    y = df[df['Country'] == 'Israel']['Temp']
    train_X, train_y, test_X, test_y = split_train_test(X, y, 0.75)

    losses = pd.DataFrame([[k, measure_poly_loss(k, train_X, train_y, test_X, test_y)] for k in range(1, 11)],
                          columns=['k', 'loss'])

    fig = px.bar(df, x=losses['k'], y=losses['loss'], text_auto=True,
                 labels={'x': 'Polynomial Degree', 'y': 'Model loss'})
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    train = df[df['Country'] == 'Israel']
    train_X = train['DayOfYear']
    train_y = train['Temp']

    model = PolynomialFitting(5)
    model.fit(train_X, train_y)

    wow = pd.DataFrame([[country, model.loss(df[df['Country'] == country]['DayOfYear'], df[df['Country'] == country]['Temp'])]
                        for country in df[df['Country'] != 'Israel'].Country.unique()]
        , columns=['country', 'loss'])
    fig = px.bar(wow, x='country', y='loss', text_auto=True, labels={'x': 'Month', 'y': 'Tempeture std'})
    fig.show()