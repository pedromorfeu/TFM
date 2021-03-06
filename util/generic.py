import numpy as np
import re
import locale
import seaborn as sns
import os
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from datetime import datetime
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import SARIMAX
from sys import platform

print("Util :: Generic")

try:
    from pyspark.mllib.linalg import Vectors, DenseVector
    print ("Successfully imported Spark Modules")
except ImportError as e:
    print ("Can not import Spark Modules", e)


if platform == "win32":
    # Windows locale:
    locale.setlocale(locale.LC_TIME, "spanish")
else:
    # MacOS locale:
    locale.setlocale(locale.LC_TIME, "es_ES")

#print(locale.getdefaultlocale())
print(locale.getlocale(locale.LC_TIME))


def getOS():
    if platform == "linux" or platform == "linux2":
        return "linux"
    elif platform == "darwin":
        return "macos"
    elif platform == "win32":
        return "windows"


def parse_dates(dates):
    print(dates)
    if type(dates) is "str":
        date = dates
        return parse_date(date)
    else:
        parsed_dates = np.empty(len(dates), dtype="datetime64[s]")
        for i in range(len(dates)):
            parsed_dates[i] = parse_date(dates[i])
            #print("parsed", parsed_dates[i])
        return parsed_dates


def parse_date(date_string):
    # 06-oct-2015 21:57:03

    # Windows regex
    regex = "\\1\\2.\\3"
    locale_date_string = re.sub("(.+-)(.+)(-.+)", regex, date_string)

    # MacOS - no need of regex
    #locale_date_string = date_string
    return datetime.strptime(locale_date_string, "%d-%b-%Y %H:%M:%S")


def plot_correlation_heatmaps(correlation_X, correlation_inverse_X, annotation=False, color_map="Reds"):
    # sns.plt.clf()
    f, (ax1, ax2) = sns.plt.subplots(1, 2, figsize=(12, 5))

    # sns.plt.figure()
    ax1.set_title("correlation_X")
    sns.heatmap(correlation_X, annot=annotation, fmt=".2f", robust=True, ax=ax1, cmap=color_map)

    # sns.plt.figure()
    ax2.set_title("correlation_inverse_X")
    sns.heatmap(correlation_inverse_X, annot=annotation, fmt=".2f", robust=True, ax=ax2, cmap=color_map)

    sns.plt.show()

# plt.figure()
# plt.imshow(covariance_X, cmap='Reds', interpolation='nearest')
#
# plt.figure()
# plt.imshow(covariance_inverse_X, cmap='Reds', interpolation='nearest')
#
# plt.show()


def print_matrix(name, matrix):
    print(name, matrix[:5, :], sep="\n")
    print("...")
    print(matrix[-5:, ], sep="\n")
    print(matrix.shape)


def print_timeseries(name, timeseries, n=5):
    print(name, timeseries.head(n), sep="\n")
    print("...")
    print(timeseries.tail(n), sep="\n")
    print(timeseries.shape)


def save_matrix(filename, matrix, columns_names=None, index_ts=None):
    print(str(datetime.now()), "Saving matrix...", filename)

    folder = "generated"
    os.makedirs(folder, exist_ok=True)

    f = open(os.path.join(folder, filename), "w")
    if columns_names is not None:
        for i in range(len(columns_names)):
            f.write(str(columns_names[i]))
            f.write("\t") if i < len(columns_names) - 1 else None
        f.write("\n")

    for i in range(matrix.shape[0]):
        if index_ts is not None:
            f.write(index_ts[i].to_datetime().strftime("%d-%b-%Y %H:%M:%S"))
            f.write("\t")
        for j in range(matrix.shape[1]):
            f. write(str(matrix[i, j]))
            f.write("\t") if j < matrix.shape[1]-1 else None
        f.write("\n")
    f.close()
    print(str(datetime.now()), "Saved")


def test_stationarity(_timeseries, _plot=False, _critical="5%"):
    # critical: 10%, 5%, 1%
    # Determing rolling statistics
    # rolmean = pd.rolling_mean(timeseries, window=12)
    rolmean = _timeseries.rolling(min_periods=1, window=12, center=False).mean()
    # rolstd = pd.rolling_std(timeseries, window=12)
    rolstd = _timeseries.rolling(min_periods=1, window=12, center=False).std()

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(_timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

    test_value = dfoutput[0]
    critical_value = dftest[4][_critical]
    stationary = (test_value < critical_value)
    print("Stationary?", stationary)

    if _plot:
        # Plot rolling statistics:
        orig = plt.plot(_timeseries, color='blue', label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        # std = plt.plot(rolstd, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show(block=True)

    return stationary, test_value


def plot_acf_pacf(_timeseries, spikes_plot=True, lags=10):
    lag_acf = acf(_timeseries, nlags=lags)
    lag_pacf = pacf(_timeseries, nlags=lags, method='ols')

    print("lag_acf", list(range(lags)))
    print("lag_acf", lag_acf)
    print("lag_pacf", list(range(1, lags)))
    print("lag_pacf", lag_pacf)

    plt.clf()

    if spikes_plot:
        ax1 = plt.subplot(211)
        plot_acf(_timeseries, lags=lags, ax=ax1)
        ax2 = plt.subplot(212)
        plot_pacf(_timeseries, lags=lags, ax=ax2)
    else:
        #Plot ACF:
        plt.subplot(121)
        plt.plot(lag_acf)
        plt.axhline(y=0,linestyle='--',color='gray')
        plt.axhline(y=-1.96/np.sqrt(len(_timeseries)),linestyle='--',color='gray')
        plt.axhline(y=1.96/np.sqrt(len(_timeseries)),linestyle='--',color='gray')
        plt.title('Autocorrelation Function')

        #Plot PACF:
        plt.subplot(122)
        plt.plot(lag_pacf)
        plt.axhline(y=0,linestyle='--',color='gray')
        plt.axhline(y=-1.96/np.sqrt(len(_timeseries)),linestyle='--',color='gray')
        plt.axhline(y=1.96/np.sqrt(len(_timeseries)),linestyle='--',color='gray')
        plt.title('Partial Autocorrelation Function')
        plt.tight_layout()

    plt.show(block=True)

    return lag_acf, lag_pacf


def arima_order_select(_timeseries, max_ar=4, max_i=2, max_ma=4, min_i=0):
    min_rmse, min_p, min_d, min_q = np.inf, 0, 0, 0

    for p in range(max_ar):
        for d in range(min_i, max_i):
            for q in range(max_ma):
                try:
                    print("Creating model (p,d,q)=(%i,%i,%i)" % (p, d, q))
                    model = SARIMAX(_timeseries, order=(p, d, q))
                    results_ARIMA = model.fit(disp=-1)
                    predictions_ARIMA = results_ARIMA.predict()
                    error = predictions_ARIMA - _timeseries
                    rmse = np.sqrt(sum((error) ** 2) / len(_timeseries))
                    if rmse < min_rmse:
                        min_rmse = rmse
                        (min_p, min_d, min_q) = (p, d, q)
                    print("RMSE", rmse)
                    # plt.plot(ts_log)
                    # plt.plot(predictions_ARIMA)
                    # plt.show(block=True)
                except (ValueError, np.linalg.linalg.LinAlgError) as err:
                    print("Ignoring model (p,d,q)=(%i,%i,%i)." % (p, d, q), "Error:", err)

    print("Minimum RMSE", min_rmse)
    print("Selected (p,d,q)=(%i,%i,%i)" % (min_p, min_d, min_q))
    return (min_rmse, min_p, min_d, min_q)


def save_plot_per_column(_data, _columns, _suffix="", _folder="figures"):
    print("Saving plots...")
    if not os.path.exists(_folder):
        os.makedirs(_folder)
    plt.ioff()
    plt.figure(figsize=(16, 9))
    for i in range(len(_columns)):
        plt.clf()
        plt.plot(_data[:, i])
        title = _columns[i] + _suffix
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(_folder, title))
    plt.ion()
    print("Plots saved")


def save_plot_per_component_gaussian(_n_components, generated_gaussian, nipals_T, _columns, _folder="figures", _suffix="_gaussian", end=5000):
    print("Saving plots...")
    if not os.path.exists(_folder):
        os.makedirs(_folder)
    plt.ioff()
    plt.figure(figsize=(16, 9))
    for i in range(_n_components):
        plt.clf()
        mean = generated_gaussian[:, i].mean()
        std = generated_gaussian[:, i].std()
        plt.axhline(generated_gaussian[:, i].max(), color="gray", linewidth=1)
        plt.axhline(generated_gaussian[:, i].min(), color="gray", linewidth=1)
        plt.axhline(mean, color="gray", linewidth=1)
        plt.axhline(mean + 2 * std, color="gray", linewidth=1)
        plt.axhline(mean + 3 * std, color="gray", linewidth=1)
        plt.axhline(mean - 2 * std, color="gray", linewidth=1)
        plt.axhline(mean - 3 * std, color="gray", linewidth=1)
        start = len(nipals_T)+1
        plt.plot(range(start, end), generated_gaussian[:end-start, i], label="Gaussian")
        plt.plot(nipals_T[:, i], label=_columns[i])
        title = _columns[i] + _suffix
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(_folder, title))
    plt.ion()
    print("Plots saved")


def save_mixed_plots_per_column(_data, _inverse, _inverse_gaussian, _columns, _suffix="", _folder="figures"):
    print("Saving plots...")
    if not os.path.exists(_folder):
        os.makedirs(_folder)
    plt.ioff()
    plt.figure(figsize=(16, 9))
    for i in range(len(_columns)):
        plt.clf()
        plt.plot(_inverse_gaussian[:, i], label="gaussian")
        plt.plot(_inverse[:, i], label="inverse")
        plt.plot(_data[:, i], label="original")
        title = _columns[i] + _suffix
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(_folder, title))
    plt.ion()
    print("Plots saved")


def save_data_plot(_data, _filename="original", _folder="figures"):
    print("Saving plot...")
    if not os.path.exists(_folder):
        os.makedirs(_folder)
    plt.ioff()
    plt.figure(figsize=(16, 9))
    plt.clf()
    plt.plot(_data)
    title = _filename
    plt.title(title)
    plt.savefig(os.path.join(_folder, title), dpi=100)
    plt.ion()
    print("Plot saved")


def save_plot_per_component(_n_components, _generated_gaussian, _timeseries_samples, _models_iterative):
    print("Saving plots...")
    plt.ioff()
    plt.figure(figsize=(16, 9))
    for i in range(_n_components):
        plt.clf()
        max_gaussian = _models_iterative[i][1].shape[0]
        plt.plot(_generated_gaussian[:max_gaussian, i], label="gaussian")
        plt.plot(_models_iterative[i][1].values, label="prediction")
        plt.plot(_timeseries_samples[i], label="original")
        plt.axhline(_generated_gaussian[:, i].max(), color="gray", linewidth=1)
        plt.axhline(_generated_gaussian[:, i].min(), color="gray", linewidth=1)
        plt.axhline(_generated_gaussian[:, i].mean(), color="gray", linewidth=1)
        plt.axhline(_generated_gaussian[:, i].mean() + 2 * _generated_gaussian[:, i].std(), color="gray", linewidth=1)
        plt.axhline(_generated_gaussian[:, i].mean() + 3 * _generated_gaussian[:, i].std(), color="gray", linewidth=1)
        plt.axhline(_generated_gaussian[:, i].mean() - 2 * _generated_gaussian[:, i].std(), color="gray", linewidth=1)
        plt.axhline(_generated_gaussian[:, i].mean() - 3 * _generated_gaussian[:, i].std(), color="gray", linewidth=1)
        title = "component_" + str(i+1)
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join("figures", title))
    plt.ion()
    print("Plots saved")

# Standardization of datasets is a common requirement for many machine learning estimators implemented in scikit-learn;
# they might behave badly if the individual features do not more or less look like standard normally distributed data: Gaussian with zero mean and unit variance.
# In practice we often ignore the shape of the distribution and just transform the data to center it by removing the mean value of each feature,
# then scale it by dividing non-constant features by their standard deviation.
# (http://scikit-learn.org/stable/modules/preprocessing.html)
# TODO: standardize with mean and std from gaussian
def scale(_X, _mean, _std):
    return (_X - _mean) / _std


def transform_normal(_x, _mus, _sigmas):
    normal_array = _mus + _sigmas * _x
    return tuple(normal_array.tolist())


def dot(_x, _y, _mus):
    normal_array = np.dot(_x, _y) + _mus
    return tuple(normal_array.tolist())


def calculate_min_distance(_points_rdd, _point, _n_points=1):
    # distances = np.sqrt(((scale(generated_gaussian_copy) - scale(preds)) ** 2).sum(axis=1))
    distances_rdd = _points_rdd.map(lambda x: (x, np.sqrt((x - _point) ** 2).sum()))
    # distances_rdd = _v.map(lambda x: (x, Vectors.dense(x).squared_distance(_x1)) )
    print("distances_rdd", distances_rdd)
    # min_distance = distances_rdd.min(lambda x: x[1])
    min_distance = distances_rdd.takeOrdered(_n_points, key=lambda x: x[1])
    # print("min distance", min_distance)
    return min_distance
