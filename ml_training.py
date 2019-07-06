import pandas as pd
# import quandl
import math, datetime
import numpy as np

from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import scale

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
# df = quandl.get('WIKI/GOOGL')
# df.to_csv('Quandl_Google_Data.csv', encoding='utf-8')

df = pd.read_csv('Quandl_Google_Data.csv', index_col='Date')

df = df[['Adj. Open', 'Adj. High',	'Adj. Low', 'Adj. Close' , 'Adj. Volume']]
df['HL_PCT'] = ((df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']) * 100
df['PCT_Change'] = ((df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']) * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

forecast_col = 'Adj. Close'

df.fillna (-99999, inplace = True)

forecast_out = int(math.ceil(0.0029*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

x = np.array(df.drop(['label'],1))
x = scale(x)
x_lately = x[-forecast_out:]
x = x[:-forecast_out]


df.dropna(inplace = True)
y = np.array(df['label'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(x_train, y_train)
confidence = clf.score(x_test, y_test)

# print('LinearRegression_accuracy:',confidence*100)

forecast_set = clf.predict(x_lately)
df['Forecast'] = np.nan

last_date = datetime.datetime.strptime(df.iloc[-1].name, '%d-%m-%Y')

last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

print('Success')