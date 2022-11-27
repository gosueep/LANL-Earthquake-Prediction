import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from common import readData
import joblib
import pandas as pd
# import librosa
from python_speech_features import mfcc

import sys


window_size = 150000


def make_features(df, with_time=True, verbose=True):
    num_windows = int(np.floor(df.shape[0] / window_size))
    X = pd.DataFrame(index=range(num_windows), dtype=np.float64)

    for window in range(num_windows):
        if verbose:
            print(f'Window: {window}')

        w = df.iloc[window*window_size : window*window_size + window_size]

        x = pd.Series(w['acoustic_data'].values)
        
        if with_time:
            y = w['time_to_failure'].values[-1]
            X.loc[window, 'time_to_failure'] = y

        X.loc[window, 'num_peaks'] = len(find_peaks(x)[0])
        X.loc[window, 'autocorr'] = x.autocorr()

        X.loc[window, 'quant_95'] = np.quantile(x, .95)
        X.loc[window, 'quant_99'] = np.quantile(x, .99)

        for roll_size in [10000]:
            X.loc[window, f'roll_std_{roll_size}'] = x.rolling(roll_size).std().dropna().values.std()
        
        X.loc[window, f'mfcc'] = mfcc(x)[1][1]
        # print(mfcc(x))
        # w_mfcc = mfcc(x)
        # for i in range(len(w_mfcc[1])):
        #     X.loc[window, f'mfcc{i}'] = mfcc(x)[1][i]

        w_max = np.max(x)
        X.loc[window, 'max'] = w_max

    return X


if __name__ =='__main__':

    df = pd.read_csv('./data/train.csv')
    X = make_features(df)
    print(X)
    X.to_csv('windows.csv', index=False)

