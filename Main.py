import pandas as pd
import numpy as np


def read_file(filename, tbond=False):
    # Will use the column on index 5 - AdjClose - the Adjusted Closing Price.
    data = pd.read_csv(filename, sep=',', usecols=[0, 5], names=['Date', 'Price'], header=0)
    if not tbond:
        returns = np.array(data["Price"][:-1], np.float) / np.array(data["Price"][1:], np.float) - 1
        data["Returns"] = np.append(returns, np.nan)
    if tbond:
        data["Returns"] = data["Price"] / 100
    data.index = data["Date"]
    data = data["Returns"][0:-1]
    return data


# Loading pre-processing training data
googleData = read_file("resources/GOOG.csv")
nasdaqData = read_file("resources/NASDAQ.csv")
tbondData = read_file("resources/tbond5yr.csv", tbond=True)


# Initialize Stochastic Gradient Descent object for Linear Regression
from sklearn.linear_model import SGDRegressor
sgd_regressor = SGDRegressor(eta0=0.1, max_iter=100000, fit_intercept=False, verbose=0)

# Training Phase
sgd_regressor.fit((nasdaqData - tbondData).values.reshape(-1, 1), (googleData - tbondData))

#  Testing Phase
print("The Beta for Google computed with Stochastic Gradient Descent Regression is " + str(sgd_regressor.coef_[0]))


# Initialize Ordinary least squares Linear Regression object
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression(fit_intercept=False)

# Training Phase
linear_regressor.fit((nasdaqData - tbondData).values.reshape(-1, 1), (googleData - tbondData))

# Testing Phase
print("The Beta for Google computed with Ordinary least squares Linear Regression is " + str(linear_regressor.coef_[0]))
print("While Reuters declares a Beta for GOOG of 0.91 and Yahoo Finance a Beta(3Y Monthly) of 1.3")
