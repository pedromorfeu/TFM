import numpy as np
import pandas as pd

from util import *
from matplotlib import pyplot as plt
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose


data = pd.read_csv("ip.txt", sep="\s+\t", engine="python", parse_dates=[0], date_parser=parse_dates,
                   index_col="Tiempoinicio", skip_blank_lines=True, na_values="")

print(data.columns)
print(data.head())
print(data.dtypes)
print(data.index)
print(data.shape)

print(min(data.index))
print(max(data.index))

print(data["2015-10-06 22:01:20"])
print(data["2015-10-06"])
print(data["2015-10-06":"2015-11-06"])

timeseries = data["APHu"]
print_timeseries("timeseries", timeseries)

plt.plot(timeseries, "o")
plt.plot(timeseries["2015-10-06"])
plt.plot(timeseries["2015-10-07"], "o")
plt.plot(timeseries["2015-10-08"])
plt.plot(timeseries["2015-10-09"])

plt.plot(timeseries)
plt.plot(timeseries, 'o', markersize=6, markeredgecolor='black', markeredgewidth=1, alpha=0.7)
plt.title("Serie temporal completa para 'APHu'")

plt.ylim([2400, 2410])
plt.plot(timeseries["2015-12"])
plt.plot(timeseries["2015-12"], 'o', markersize=6, markeredgecolor='black', markeredgewidth=1, alpha=0.7)
plt.title("Serie temporal de diciembre de 2015 para 'APHu'")


# Plot all
fig, axes = plt.subplots(nrows=3, ncols=5)
for i in range(0, len(data.columns)):
    col = data.columns[i]
    ax_plt = axes.flatten()[i]
    ax_plt.plot(data[col])
    ax_plt.plot(data[col], 'o', markersize=6, markeredgecolor='black', markeredgewidth=1, alpha=0.7)
    ax_plt.set_title(col)

timeseries.mean()
# group by day and count observations
timeseries.groupby(lambda x: x.date()).count()
# some days and observations:
# 2015-10-06     835
# 2015-10-07    7470
# 2015-10-08    6888
# 2015-10-09    3150
# 2015-10-10      20
# 2015-10-12     520
# 2015-12-14     100

date = "2015-10-07"
timeseries[date].shape
plt.plot(timeseries[date])
plt.plot(timeseries[date], 'o', markersize=6, markeredgecolor='black', markeredgewidth=1, alpha=0.7)

# Specify a date to analyze the timeseries
timeseries_sample = timeseries[date]
print_timeseries("timeseries_sample", timeseries_sample)
timeseries_sample.shape
plt.plot(timeseries_sample)
plt.plot(timeseries_sample, 'o', markersize=6, markeredgecolor='black', markeredgewidth=1, alpha=0.7)

# Frequency is ~9 seconds
# to 5 seconds frequency and forward fill
timeseries_sample = timeseries_sample.asfreq('5s', method='pad')
timeseries_sample.shape
timeseries_sample
plt.plot(timeseries_sample)
plt.plot(timeseries_sample, 'o', markersize=6, markeredgecolor='black', markeredgewidth=1, alpha=0.7)

# Alternative:
# Resample by second ('s') and interpolate
timeseries_sample = timeseries_sample.resample('10s').mean().interpolate()
print_timeseries("timeseries_sample", timeseries_sample)
timeseries_sample.shape
plt.plot(timeseries_sample)
plt.plot(timeseries_sample, 'o', markersize=6, markeredgecolor='black', markeredgewidth=1, alpha=0.7)


# We can check stationarity using the following:
#     Plotting Rolling Statistics: We can plot the moving average or moving variance and see if it varies with time. By moving average/variance I mean that at any instant ‘t’, we’ll take the average/variance of the last year, i.e. last 12 months. But again this is more of a visual technique.
#     Dickey-Fuller Test: This is one of the statistical tests for checking stationarity. Here the null hypothesis is that the TS is non-stationary. The test results comprise of a Test Statistic and some Critical Values for difference confidence levels. If the ‘Test Statistic’ is less than the ‘Critical Value’, we can reject the null hypothesis and say that the series is stationary.

test_stationarity(timeseries_sample, _plot=True)

rolmean = timeseries_sample.rolling(min_periods=1, window=20, center=False).mean()
orig = plt.plot(timeseries_sample, color='blue', label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show()

# ['APHu', 'APVs', 'ACPv', 'ZSx', 'ZUs', 'H7x', 'H1x', 'H2x', 'H6x', 'H3x', 'H4x', 'H5x', 'ACPx', 'Svo']
# test_stationarity(raw["APHu"])
# test_stationarity(raw["APVs"])
# test_stationarity(raw["ACPv"])
# test_stationarity(raw["ZSx"])
# test_stationarity(raw["ZUs"])
# test_stationarity(raw["H7x"])
# test_stationarity(raw["H1x"])
# test_stationarity(raw["H2x"])
# test_stationarity(raw["H6x"])
# test_stationarity(raw["H3x"])
# test_stationarity(raw["H4x"])
# test_stationarity(raw["H5x"])
# test_stationarity(raw["ACPx"])
# test_stationarity(raw["Svo"])


### NOT STATIONARY -> must be stabilized


# One of the first tricks to reduce trend can be transformation. For example, in this case we can clearly see that the there is a significant positive trend. So we can apply transformation which penalize higher values more than smaller values. These can be taking a log, square root, cube root, etc. Let's take a log transform here for simplicity:
ts_log = np.log(timeseries_sample)
print("Missing values:", not np.all(np.isfinite(ts_log)))
ts_log.dropna(inplace=True)
print("Missing values:", not np.all(np.isfinite(ts_log)))
plt.plot(ts_log)
plt.plot(ts_log, 'o', markersize=6, markeredgecolor='black', markeredgewidth=1, alpha=0.7)

# Clean ts_log (NaN and infinites)
ts_log
np.all(np.isfinite(ts_log))

ts_log.index[np.isinf(ts_log)]
ts_log.index[np.isnan(ts_log)]

# this is -inf
ts_log["2015-10-09 09:18:11"] = 0

ts_log
ts_log.shape
# ts_log.drop("2015-10-09 09:18:11")


# Series.rolling(window, min_periods=None, freq=None, center=False, win_type=None, axis=0)
# Provides rolling transformations.
# window : int Size of the moving window. This is the number of observations used for calculating the statistic.
moving_avg = ts_log.rolling(20).mean()
print("Missing values:", not np.all(np.isfinite(moving_avg)))
moving_avg.dropna(inplace=True)
print("Missing values:", not np.all(np.isfinite(moving_avg)))
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
# Markers plot
plt.plot(ts_log, 'o', markersize=6, markeredgewidth=1, alpha=0.7)
plt.plot(moving_avg, '^', markersize=6, markeredgewidth=1, alpha=0.5)


ts_log_moving_avg_diff = ts_log.sub(moving_avg)
print("Missing values:", not np.all(np.isfinite(ts_log_moving_avg_diff)))
ts_log_moving_avg_diff.dropna(inplace=True)
print("Missing values:", not np.all(np.isfinite(ts_log_moving_avg_diff)))
ts_log_moving_avg_diff.head(12)
test_stationarity(ts_log_moving_avg_diff, _plot=True)

# This looks like a much better series. The rolling values appear to be varying slightly but there is no specific trend. Also, the test statistic is smaller than the 1% critical values so we can say with 99% confidence that this is a stationary series.

# There can be many technique for assigning weights. A popular one is exponentially weighted moving average where weights are assigned to all the previous values with a decay factor.
expwighted_avg = ts_log.ewm(halflife=12).mean()
plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')
ts_log_ewma_diff = ts_log.sub(expwighted_avg)
print("Missing values:", not np.all(np.isfinite(ts_log_ewma_diff)))
test_stationarity(ts_log_ewma_diff, _plot=True)


### Differencing
# One of the most common methods of dealing with both trend and seasonality is differencing.
# In this technique, we take the difference of the observation at a particular instant with that at the previous instant.
ts_log_diff = ts_log.sub(ts_log.shift())
print("Missing values:", not np.all(np.isfinite(ts_log_diff)))
ts_log_diff.dropna(inplace=True)
print("Missing values:", not np.all(np.isfinite(ts_log_diff)))
# This appears to have reduced trend considerably. Lets verify using our plots:
test_stationarity(ts_log_diff, _plot=True)
# Markers plot

ts_log_diff["2015-12"]

decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

# # One of the most common methods of dealing with both trend and seasonality is differencing.
# ts_log_diff = ts_log_moving_avg_diff.sub(ts_log_moving_avg_diff.shift())
# print("Missing values:", ts_log_diff.isnull().sum())
# ts_log_diff.dropna(inplace=True)
# print("Missing values:", ts_log_diff.isnull().sum())
# plt.plot(ts_log_diff)
#
# # This appears to have reduced trend considerably. Lets verify using our plots:
# test_stationarity(ts_log_diff)


### FORECASTING
plot_acf_pacf(ts_log_diff)

# An importance concern here is how to determine the value of ‘p’ and ‘q’. We use two plots to determine these numbers. Lets discuss them first.
#     Autocorrelation Function (ACF)
#     Partial Autocorrelation Function (PACF)

# AR Model
# ts_log_diff.isnull().sum()
# ts_log_diff.dropna(inplace=True)
# ts_log_diff.index[ts_log_diff.isnull()]
# model = ARIMA(ts_log, order=(1, 0, 0))
# results_AR = model.fit(disp=-1)
# plt.plot(ts_log_diff)
# plt.plot(results_AR.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))

# MA Model
# model = ARIMA(ts_log_diff, order=(0, 1, 2))
# results_MA = model.fit(disp=-1)
# plt.plot(ts_log_diff)
# plt.plot(results_MA.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))

# Combined Model
model = ARIMA(ts_log, order=(2, 1, 2))
results_ARIMA = model.fit(disp=-1)
predictions = results_ARIMA.fittedvalues

observations = timeseries_sample.shape[0]
predictions1 = results_ARIMA.predict(start=observations, end=observations+50000)
predictions1
predictions1.shape

# results_ARIMA.plot_predict(start=1, end=50000)

# Markers plot
# plt.plot(ts_log_diff, 'o', markersize=6, markeredgewidth=1, alpha=0.7)
# plt.plot(predictions1, '^', markersize=6, markeredgewidth=1, alpha=0.7)
plt.plot(ts_log_diff)
plt.plot(predictions1)
error = predictions1-ts_log_diff
plt.legend(loc='best')
plt.title('RSS: %.4f'% sum((error)**2))

print("Missing values:", not np.all(np.isfinite(predictions)))
np.any(np.isinf(predictions))
np.any(np.isnan(predictions))

# plt.plot(ts_log_diff)
# plt.plot(predictions, color='red')
# error = predictions-ts_log_diff
# plt.title('RSS: %.4f'% sum((error)**2))



predictions_ARIMA_diff = pd.Series(predictions, copy=True)
print(predictions_ARIMA_diff.head())

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())

predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
print(predictions_ARIMA_log.head())
print(ts_log.head())

# plt.plot(ts_log)
# plt.plot(predictions_ARIMA_log)

predictions_ARIMA = np.exp(predictions_ARIMA_log)
print("Missing values:", not np.all(np.isfinite(predictions_ARIMA)))
print(predictions_ARIMA.head())
print(timeseries_sample.head())
predictions_ARIMA.index[np.isinf(predictions_ARIMA)]
predictions_ARIMA.index[np.isnan(predictions_ARIMA)]
print("Missing values:", not np.all(np.isfinite(predictions_ARIMA)))

plt.clf()
# Markers plot
# plt.plot(timeseries_sample, 'o', markersize=6, markeredgewidth=1, alpha=0.7)
# plt.plot(predictions_ARIMA, '^', markersize=6, markeredgewidth=1, alpha=0.7)
plt.plot(timeseries_sample)
plt.plot(predictions_ARIMA)
error = predictions_ARIMA-timeseries_sample
print("Missing values:", not np.all(np.isfinite(error)))
error.index[np.isinf(error)]
error.index[np.isnan(error)]
error.dropna(inplace=True)
plt.title('RMSE: %.4f'% np.sqrt(sum(error**2)/len(timeseries_sample)))
plt.show()
