import matplotlib.pyplot as plt
from numpy import arange
import pandas as pd
from lightgbm import LGBMRegressor


# df = pd.read_csv('pred.csv')
# df['diff'] = df.apply(lambda row : abs(row['predicted'] - row['actual']), axis=1)
# print(min(df['diff']))
# print(df.loc[df['diff'] == min(df['diff'])])
# 3.41665692353661,3.3469998382
WINDOW = 247
WINDOW_SIZE = 150000


win_df = pd.read_csv('windows.csv')
print(win_df.loc[247])
train_X, train_y = win_df.drop('time_to_failure', axis=1), win_df['time_to_failure']
model = LGBMRegressor(n_estimators=27, max_depth=8).fit(train_X, train_y)

raw = pd.read_csv('data/train.csv')
raw_window = raw.iloc[WINDOW*WINDOW_SIZE : WINDOW*WINDOW_SIZE + WINDOW_SIZE]
print(raw_window)


# dataFile = open('data/testsplit0.csv', 'r', buffering=10000)
# dataFile.readline()

# outFile = open('plot.csv', 'w')

# acous = []
# time = []
# r = 10000000
# s = 1
# x = arange(0,r,s)
# for _ in range(r):
#     line = dataFile.readline()
#     outFile.write(line)
#     a,t = line.split(',')

#     acous.append(int(a))
#     time.append(float(t))

plt.subplot(212)
plt.plot(raw_window['time_to_failure'], raw_window['acoustic_data'], 'r')
# plt.subplot(211)
# plt.plot(x, raw_window['time_to_failure'], 'b')

plt.title('Initial Plot of Time vs Acoustic Data')
plt.ylabel('Acoustic Data')
plt.xlabel('Time in seconds')
plt.show()