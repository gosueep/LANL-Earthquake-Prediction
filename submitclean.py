import pandas as pd

df = pd.read_csv('submission.csv')

df['seg_id'] = df.apply(lambda row : row['seg_id'].split('.')[0], axis=1)
print(df)

df.to_csv('submission.csv', index=False)