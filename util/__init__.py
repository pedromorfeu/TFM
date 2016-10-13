import numpy as np
import re
import locale
import seaborn as sns
import os
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
from statsmodels.tsa.stattools import adfuller, acf, pacf

print(locale.getdefaultlocale())
locale.setlocale(locale.LC_TIME, "spanish")


def parse_dates(dates):
    if type(dates) is "str":
        date = dates
        return parse_date(date)
    else:
        parsed_dates = np.empty(len(dates), dtype="datetime64[s]")
        for i in range(len(dates)):
            parsed_dates[i] = parse_date(dates[i])
        return parsed_dates


def parse_date(date_string):
    # 06-oct-2015 21:57:03
    locale_date_string = re.sub("(.+-)(.+)(-.+)", "\\1\\2.\\3", date_string)
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


def save_matrix(filename, matrix, columns_names=None):
    print(str(datetime.now()), "Saving matrix...")

    folder = "generated"
    os.makedirs(folder, exist_ok=True)

    f = open(os.path.join(folder, filename), "w")
    if columns_names is not None:
        for i in range(len(columns_names)):
            f.write(columns_names[i])
            f.write("\t") if i < len(columns_names) - 1 else None
        f.write("\n")

    for i in range(matrix.shape[0]):
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
    rolmean = _timeseries.rolling(min_periods=1, window=20, center=False).mean()
    # rolstd = pd.rolling_std(timeseries, window=12)
    rolstd = _timeseries.rolling(min_periods=1, window=20, center=False).std()

    if _plot:
        # Plot rolling statistics:
        orig = plt.plot(_timeseries, color='blue', label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        # std = plt.plot(rolstd, color='black', label='Rolling Std')
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

    test_value = dfoutput[0]
    critical_value = dftest[4][_critical]
    if test_value < critical_value:
        return True
    return False


def plot_acf_pacf(_timeseries):
    lag_acf = acf(_timeseries, nlags=20)
    lag_pacf = pacf(_timeseries, nlags=20, method='ols')

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

    return lag_acf, lag_pacf