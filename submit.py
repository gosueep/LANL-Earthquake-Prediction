import os
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error

from featurize import make_features


df = pd.read_csv('windows.csv')
train_X, train_y = df.drop('time_to_failure', axis=1), df['time_to_failure']

model = LGBMRegressor(n_estimators=27, max_depth=8)
model.fit(train_X, train_y)

# model = LGBMRegressor()
# params = { 'n_estimators': range(1, 30), 'max_depth': range(3, 10) }
# # n_est = 27, max_depth = 8
# model = GridSearchCV(model, params, scoring='neg_mean_absolute_error', return_train_score=True) 
# model.fit(train_X, train_y)
# print(model.best_params_)


filepath = './data/test'
segments = os.listdir(filepath)
output_rows = []
for i, seg in enumerate(segments):
    
    print(f'Segment {i} out of {len(segments)}')

    s = pd.read_csv(filepath + '/' + seg)
    x = make_features(s, with_time=False, verbose=False)
    y = model.predict(x)

    # print(np.mean(y))
    # print(np.median(y))

    output_rows.append([seg.split('.')[0], np.median(y)])
    # print(output_rows)

output = pd.DataFrame(output_rows, columns=['seg_id', 'time_to_failure'])
output.to_csv('submission.csv', index=False)


