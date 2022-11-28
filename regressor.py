import joblib
from Score import readTestData, scoreModel
from Training import createModel
from WindowMaker import getWindows

# import tensorflow as tf
# import tensorflow.keras as keras
import numpy as np
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import SGD, Adam
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR

import pandas as pd


df = pd.read_csv('windows.csv')
split = int(np.floor(df.shape[0] * 0.8))

train_X, train_y = df.drop('time_to_failure', axis=1)[:split], df['time_to_failure'][:split]
test_X, test_y = df.drop('time_to_failure', axis=1)[split:], df['time_to_failure'][split:]
print('loaded in')


model = LGBMRegressor(n_estimators=27, max_depth=4).fit(train_X, train_y)

# model = LGBMRegressor()
# params = { 'n_estimators': range(1, 30), 'max_depth': range(3, 10) }
# # n_est = 27, max_depth = 4
# model = GridSearchCV(model, params, scoring='neg_mean_absolute_error', return_train_score=True) 
# model.fit(train_X, train_y)
# print(model.best_params_)

# sums = []
# for i, x in enumerate(model.predict(test_X)):
#     print(x, test_y.iloc[i])
#     sums.append(abs(x - test_y.iloc[i]))
# print(np.mean(sums))
print(model.score(test_X, test_y))
print(mean_absolute_error(model.predict(test_X), test_y))

print(mean_absolute_error(model.predict(df.drop('time_to_failure', axis=1)), df['time_to_failure']))
pred = model.predict(df.drop('time_to_failure', axis=1))
output = pd.DataFrame(columns=['predicted', 'actual'])
output['predicted'] = pred
output['actual'] = df['time_to_failure'].tolist()
output.to_csv('pred_all.csv', index=False)


# SVR: -60.5644328880081
# LGBM -29.580216670260004

# LGBM: improved = 1.95 MAE
# original: best was 2.7 MAE
