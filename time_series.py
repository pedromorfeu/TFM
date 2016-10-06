import numpy as np
import pandas as pd
from util import *
from matplotlib import pyplot as plt
from datetime import datetime
from statsmodels.tsa.stattools import adfuller

raw = pd.read_csv("ip.txt", sep="\s+\t", engine="python", parse_dates=[0], date_parser=parse_dates,
               index_col="Tiempoinicio", skip_blank_lines=True, na_values="")

print(raw.head())
print(raw.dtypes)
print(raw.index)

print(min(raw.index))
print(max(raw.index))

print(raw["2015-10-06 22:01:20"])
print(raw["2015-10-06"])
print(raw["2015-10-06":"2015-11-06"])

timeseries = raw["APHu"]
print(timeseries)

plt.plot(timeseries)
plt.show()

# We can check stationarity using the following:
#     Plotting Rolling Statistics: We can plot the moving average or moving variance and see if it varies with time. By moving average/variance I mean that at any instant ‘t’, we’ll take the average/variance of the last year, i.e. last 12 months. But again this is more of a visual technique.
#     Dickey-Fuller Test: This is one of the statistical tests for checking stationarity. Here the null hypothesis is that the TS is non-stationary. The test results comprise of a Test Statistic and some Critical Values for difference confidence levels. If the ‘Test Statistic’ is less than the ‘Critical Value’, we can reject the null hypothesis and say that the series is stationary.

def test_stationarity(_timeseries):
    # Determing rolling statistics
    # rolmean = pd.rolling_mean(timeseries, window=12)
    rolmean = _timeseries.rolling(min_periods=1, window=2, center=False).mean()
    # rolstd = pd.rolling_std(timeseries, window=12)
    rolstd = _timeseries.rolling(min_periods=1, window=2, center=False).std()

    # Plot rolling statistics:
    orig = plt.plot(_timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(_timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

test_stationarity(timeseries)

### NOT STATIONARY -> must be stabilized


# One of the first tricks to reduce trend can be transformation. For example, in this case we can clearly see that the there is a significant positive trend. So we can apply transformation which penalize higher values more than smaller values. These can be taking a log, square root, cube root, etc. Lets take a log transform here for simplicity:
ts_log = np.log(timeseries)
plt.plot(ts_log)
plt.show()

moving_avg = ts_log.rolling(2).mean()
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
plt.show()

ts_log_moving_avg_diff = ts_log.sub(moving_avg)
ts_log_moving_avg_diff.head(12)

ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)

# This looks like a much better series. The rolling values appear to be varying slightly but there is no specific trend. Also, the test statistic is smaller than the 1% critical values so we can say with 99% confidence that this is a stationary series.

