import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, r2_score


def norm_column(column):
    min = column.min()
    max = column.max()
    ret = column.apply(lambda x: (x - min)/(max - min))

    return ret


class LinearRegressionSGD(BaseEstimator):
    def __init__(self, l_r = 0.1, epochs = 1000, start_coef = 'zeros'):
        self.coef = []

        self.l_r = l_r
        self.epochs = epochs
        self.start_coef = start_coef

    def fit(self, x, y):
        if self.start_coef == 'ones':
            coef = np.ones(len(x[0]))
        elif self.start_coef == 'rand':
            coef = np.random.uniform(1.0, 10.0, size=len(x[0]))
        else:
            coef = np.zeros(len(x[0]))

        for _ in range(self.epochs):
            grad = np.mean(x.transpose() * (np.dot(x, coef) - y), axis=1)
            coef = np.subtract(coef, self.l_r * grad)

        self.coef = coef
        # print(self.get_params())
        return self

    def predict(self, x):
        return x.dot(self.coef)

    def get_params(self, deep=True):
        return {
            'l_r': self.l_r,
            'epochs': self.epochs,
            'start_coef': self.start_coef
        }

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self


class LinearRegressionADAM(BaseEstimator):
    def __init__(self, l_r = 0.1, epochs = 1000, start_coef = 'zeros'):
        self.coef = []
        self.b1 = 0.9
        self.b2 = 0.99

        self.l_r = l_r
        self.epochs = epochs
        self.start_coef = start_coef

    def fit(self, x, y):
        if self.start_coef == 'ones':
            coef = np.ones(len(x[0]))
        elif self.start_coef == 'rand':
            coef = np.random.uniform(1.0, 10.0, size=len(x[0]))
        else:
            coef = np.zeros(len(x[0]))

        m_coef = np.zeros(coef.shape)
        v_coef = np.zeros(coef.shape)
        moment_m_coef = np.zeros(coef.shape)
        moment_v_coef = np.zeros(coef.shape)
        t=0
        for _ in range(self.epochs):
            grad = self.gradients(coef, x, y)
            t+=1
            m_coef = self.b1 * m_coef + (1 - self.b1) * grad
            v_coef = self.b2 * v_coef + (1 - self.b2) * grad ** 2
            moment_m_coef = m_coef / (1 - self.b1 ** t)
            moment_v_coef = v_coef / (1 - self.b2 ** t)

            delta = ((self.l_r / moment_v_coef ** 0.5 + 1e-8) *
                     (self.b1 * moment_m_coef + (1 - self.b1) * grad / (1 - self.b1 ** t)))

            coef = np.subtract(coef, delta)

        self.coef = coef
        # print(self.get_params())
        return self

    def predict(self, x):
        return x.dot(self.coef)

    def gradients(self,coef, x, y):
        return np.mean(x.transpose() * (np.dot(x, coef) - y), axis=1)

    def get_params(self, deep=True):
        return {
            'l_r': self.l_r,
            'epochs': self.epochs,
            'start_coef': self.start_coef
        }

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self






########################### PRYGOTOWANIE DANYCH #################################

column_names = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"]

data = pd.read_csv('auto-mpg.data',
                  na_values='?',
                  comment='\t',
                  sep=' ',
                  skipinitialspace=True,
                  names=column_names)

data['horsepower'].fillna(data['horsepower'].mean(), inplace=True)


# avg_mpg_by_year = data.groupby('origin')['mpg'].mean().to_numpy()
# print(avg_mpg_by_year)
# years = data['origin'].unique()
# print(years)
#
# plt.scatter(years, avg_mpg_by_year)
# plt.xlabel("pochodzenie")
# plt.ylabel("średnie mpg")
# plt.show()


mpg_min = data['mpg'].min()
mpg_max = data['mpg'].max()

data = pd.get_dummies(data, columns=["cylinders", "model_year", "origin"], drop_first=False)

data = data.sample(frac=1, random_state=43)
# data = data.sample(frac=1)
x = data.drop('mpg', axis=1)
y = data['mpg'].copy()
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, shuffle=False)


###################### WLASNE FUNKCJE ##################################

X_TRAIN = X_train.copy()
X_TEST = X_test.copy()
Y_TRAIN = Y_train.copy()
Y_TEST = Y_test.copy().to_numpy()

# Normalizacja danych
for col in ["displacement", "horsepower", "weight", "acceleration"]:
    X_TRAIN[col] = norm_column(X_TRAIN[col])
    X_TEST[col] = norm_column(X_TEST[col])

Y_TRAIN = norm_column(Y_TRAIN).to_numpy()


X_TRAIN = X_TRAIN.to_numpy()
X_TRAIN = np.insert(X_TRAIN, 0, 1, axis=1)

X_TEST = X_TEST.to_numpy()
X_TEST = np.insert(X_TEST, 0, 1, axis=1)

param_grid = {
    'l_r': [0.1, 0.01, 0.001, 0.0001],
    'epochs': [1000, 3000, 5000],
    'start_coef': ['zeros', 'ones', 'rand']
}

# grid = GridSearchCV(estimator=LinearRegressionADAM(),
#                     param_grid=param_grid,
#                     cv=5,
#                     scoring='neg_mean_squared_error')
# grid.fit(X_TRAIN, Y_TRAIN)
#
# print(grid.best_params_)


a = LinearRegressionADAM(0.01, 3000, 'zeros')
a.fit(X_TRAIN, Y_TRAIN)
y_pred_ADAM = a.predict(X_TEST)

for i in range(len(y_pred_ADAM)):
    y_pred_ADAM[i] = y_pred_ADAM[i]*(mpg_max - mpg_min) + mpg_min

print("ADAM--->   MSE: ", mean_squared_error(Y_test, y_pred_ADAM), "   R^2: ", r2_score(Y_test, y_pred_ADAM), "\n")



# grid2 = GridSearchCV(estimator=LinearRegressionSGD(),
#                     param_grid=param_grid,
#                     cv=5,
#                     scoring='neg_mean_squared_error')
# grid2.fit(X_TRAIN, Y_TRAIN)
#
# print(grid2.best_params_)


# s = LinearRegressionSGD(0.1, 5000, 'zeros')
# s.fit(X_TRAIN, Y_TRAIN)
# y_pred_SGD = s.predict(X_TEST)
#
#
# for i in range(len(y_pred_SGD)):
#     y_pred_SGD[i] = y_pred_SGD[i]*(mpg_max - mpg_min) + mpg_min
#
# print("SGD--->   MSE: ", mean_squared_error(Y_TEST, y_pred_SGD), "   R^2: ", r2_score(Y_TEST, y_pred_SGD))







################### BIBLIOTEKA ##############################

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred_LIB = model.predict(X_test)

print("LIB--->   MSE: ", mean_squared_error(Y_test, y_pred_LIB), "   R^2: ", r2_score(Y_test, y_pred_LIB), "\n")









############### WYKRESY ###############

print('---------------------------')
for t, p1, p2 in zip(Y_test, y_pred_ADAM, y_pred_LIB):
    print(f"{t}    {p1}    {p2}")



x = range(len(Y_test))


plt.figure(figsize=(15, 6))
plt.plot(x, y_pred_ADAM, color='red', label='wartości przewidziane z ADAM')
plt.plot(x, y_pred_LIB, color='green', label='wartości przewidziane z biblioteki')
plt.plot(x, Y_test, color='blue', label='Wartości rzeczywiste')
plt.legend()
plt.xlabel("nr punktu")
plt.ylabel("mpg")
plt.show()




