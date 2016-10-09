# Limitations: does not handle missing data

import numpy as np
import pandas as pd
from util import *
from matplotlib import pylab
from datetime import datetime
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# Download the CSV data file from:
# http://datasets.connectmv.com/info/silicon-wafer-thickness
# raw = np.genfromtxt('silicon-wafer-thickness.csv', delimiter=',', skip_header=1)

data = pd.read_csv("ip.txt", sep="\s+\t", engine="python", parse_dates=[0], date_parser=parse_dates,
               index_col="Tiempoinicio", skip_blank_lines=True, na_values="")

print(data.columns)
print(data.head())
print(data.dtypes)
print(data.index)
print(data.shape)

# raw = data.values[:, 1:]
# raw = raw.astype(float)

raw = data.values
N, K = raw.shape
print(N, K)
print_matrix("raw", raw)


# Preprocessing: mean center and scale the data columns to unit variance
X = raw - raw.mean(axis=0)
print(X.shape)
print(X[:5, :])
# X = X / X.std(axis=0)
print(X.shape)
print(X[:5, :])

# Verify the centering and scaling
X.mean(axis=0)  # array([ -3.92198351e-17,  -1.74980803e-16, ...
X.std(axis=0)  # [ 1.  1.  1.  1.  1.  1.  1.  1.  1.]

# We are going to calculate only 2 principal components
A = 5

# We could of course use SVD ...
print(str(datetime.now()), "Calculating SVD...")
u, d, v = np.linalg.svd(X[:1000, :])
print(str(datetime.now()), "Done")
print(u.shape)
print(d.shape)
print(v.shape)
print("v", v)

# U, S, V = self._fit(X)
# U = U[:, :self.n_components_]
# U *= S[:self.n_components_]
U = u[:, :2]
U *= d[:2]
print_matrix("U", U)

# Transpose the "v" array from SVD, which contains the loadings, but retain
# only the first A columns
svd_P = v.T[:, range(0, A)]
print(svd_P.shape)
print_matrix("svd_P", svd_P)

# Compute the scores from the loadings:
svd_T = np.dot(X, svd_P)
print(svd_T.shape)
print_matrix("svd_T", svd_T)

# invert
XX = np.dot(svd_T, svd_P.T) + np.mean(raw, axis=0)
print_matrix("XX", XX)


# But what if we really only wanted calculate A=2 components (imagine SVD on
# a really big data set where N and K &gt;&gt; 1000). This is why will use the NIPALS,
# nonlinear iterative partial least squares, method.

# scores
nipals_T = np.zeros((N, A))
# loadings
nipals_P = np.zeros((K, A))

tolerance = 1E-10
# for each component
for a in range(A):

    print(str(datetime.now()),"Loop", a)
    t_a_guess = np.random.rand(N, 1) * 2
    t_a = t_a_guess + 1.0
    itern = 0

    # Repeat until the score, t_a, converges, or until a maximum number of
    # iterations has been reached
    while np.linalg.norm(t_a_guess - t_a) > tolerance or itern < 500:

        # 0: starting point for convergence checking on next loop
        t_a_guess = t_a

        # 1: Regress the scores, t_a, onto every column in X; compute the
        #    regression coefficient and store it in the loadings, p_a
        #    i.e. p_a = (X' * t_a)/(t_a' * t_a)
        p_a = np.dot(X.T, t_a) / np.dot(t_a.T, t_a)

        # 2: Normalize loadings p_a to unit length
        p_a = p_a / np.linalg.norm(p_a)

        # 3: Now regress each row in X onto the loading vector; store the
        #    regression coefficients in t_a.
        #    i.e. t_a = X * p_a / (p_a.T * p_a)
        t_a = np.dot(X, p_a) / np.dot(p_a.T, p_a)

        itern += 1

        if itern % 100 == 0:
            print("diff", np.linalg.norm(t_a_guess - t_a))
            # pylab.plot(t_a, 'r-')
            # pylab.plot(t_a_guess, 'g--')
            # pylab.title("Loop" + str(a))
            # pylab.show()

    # We've converged, or reached the limit on the number of iteration

    # Deflate the part of the data in X that we've explained with t_a and p_a
    X = X - np.dot(t_a, p_a.T)

    # Store result before computing the next component
    nipals_T[:, a] = t_a.ravel()
    nipals_P[:, a] = p_a.ravel()

# loadings
print_matrix("nipals_P", nipals_P)
# scores
print_matrix("nipals_T", nipals_T)


### Generate data
mus = np.mean(nipals_T, axis=0)
sigmas = np.std(nipals_T, axis=0)

generated_X = np.zeros((100000, A))
for i in range(A):
    # calculate normal distribution by component and store it in column i
    # generated_X[:, i] = np.random.normal(mus[i], sigmas[i], 100000)
    # alternative:
    # generated_X[:, i] = mus[i] + sigmas[i] * np.random.randn(100000)
    # generate random
    generated_X[:, i] = mus[i] + sigmas[i] * np.random.rand(1, 100000)
    
# invert matrix: dot product between random data and the loadings, nipals_P
XX = np.dot(generated_X, nipals_P.T) + np.mean(raw, axis=0)
# XX = np.dot(nipals_T, nipals_P.T) + np.mean(raw, axis=0)
print_matrix("XX", XX)

save_matrix("inverse_X.csv", XX, data.columns)


### Time series
timeseries = pd.Series(nipals_T[:, 0], index=data.index)
print(timeseries.head())
test_stationarity(timeseries, _plot=False)
# not stationary -> must be stabilized

# ts_log = timeseries
ts_log = np.log(timeseries)
print("Missing values:", not np.all(np.isfinite(ts_log)))
ts_log.dropna(inplace=True)
print("Missing values:", not np.all(np.isfinite(ts_log)))
# plt.plot(ts_log)
# plt.plot(ts_log, 'o', markersize=6, markeredgecolor='black', markeredgewidth=1, alpha=0.7)
ts_log.index[np.isinf(ts_log)]
ts_log.index[np.isnan(ts_log)]
test_stationarity(ts_log, _plot=True)

# Differencing
ts_log_diff = ts_log.sub(ts_log.shift())
print("Missing values:", not np.all(np.isfinite(ts_log_diff)))
ts_log_diff.index[np.isinf(ts_log_diff)]
ts_log_diff.index[np.isnan(ts_log_diff)]
ts_log_diff.dropna(inplace=True)
print("Missing values:", not np.all(np.isfinite(ts_log_diff)))
# This appears to have reduced trend considerably. Lets verify using our plots:
test_stationarity(ts_log_diff, _plot=True)

#Forecasting
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

model = ARIMA(ts_log, order=(1, 0, 1))
results_ARIMA = model.fit(disp=-1)
predictions = results_ARIMA.fittedvalues
print(predictions)
# predictions = results_ARIMA.predict(start=1, end=50000)
# results_ARIMA.plot_predict(start="2015-10-06 21:57:03", end="2016-06-20 12:04:46")
# results_ARIMA.plot_predict(start=0, end=50000)
# results_ARIMA.plot_predict(start=ts_log.index[0], end=ts_log.index[-1])
# Markers plot
print("Missing values:", not np.all(np.isfinite(predictions)))
np.any(np.isinf(predictions))
np.any(np.isnan(predictions))
plt.plot(ts_log_diff)
plt.plot(predictions, color='red')
error = predictions-ts_log_diff
plt.title('RSS: %.4f'% sum((error)**2))

predictions_ARIMA_diff = pd.Series(predictions, copy=True)
print(predictions_ARIMA_diff.head())
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())
# plt.plot(ts_log)
# plt.plot(predictions_ARIMA_diff_cumsum)
predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
print(predictions_ARIMA_log.head())
print(ts_log.head())
# plt.plot(ts_log)
# plt.plot(predictions_ARIMA_log)
predictions_ARIMA = np.exp(predictions_ARIMA_log)
print("Missing values:", not np.all(np.isfinite(predictions_ARIMA)))
print(predictions_ARIMA.head())
print(timeseries.head())
predictions_ARIMA.index[np.isinf(predictions_ARIMA)]
predictions_ARIMA.index[np.isnan(predictions_ARIMA)]
print("Missing values:", not np.all(np.isfinite(predictions_ARIMA)))

plt.clf()
# Markers plot
plt.plot(timeseries, 'o', markersize=6, markeredgewidth=1, alpha=0.7)
plt.plot(predictions_ARIMA, '^', markersize=6, markeredgewidth=1, alpha=0.7)
# plt.plot(timeseries)
# plt.plot(predictions_ARIMA)
error = predictions_ARIMA-timeseries
print("Missing values:", not np.all(np.isfinite(error)))
error.index[np.isinf(error)]
error.index[np.isnan(error)]
error.dropna(inplace=True)
plt.title('RMSE: %.4f'% np.sqrt(sum(error**2)/len(timeseries)))
plt.show()



# PCA also has two very important outputs we should calculate:

# The SPE_X, squared prediction error to the X-space is the residual distance
# from the model to each data point.
SPE_X = np.sum(X ** 2, axis=1)
print("SPE_X", X[:5, :])

# And Hotelling's T2, the directed distance from the model center to
# each data point.
inv_covariance = np.linalg.inv(np.dot(nipals_T.T, nipals_T) / N)
Hot_T2 = np.zeros((N, 1))
for n in range(N):
    Hot_T2[n] = np.dot(np.dot(nipals_T[n, :], inv_covariance), nipals_T[n, :].T)

# You can verify the NIPALS and SVD results are the same:
# (you may find that the signs have flipped, but this is still correct)
nipals_T / svd_T
nipals_P / svd_P

# But since PCA is such a visual tool, we should plot the SPE_X and
# Hotelling's T2 values

# pylab.plot(SPE_X, 'r.-')  # see how observation 154 is inconsistent
# pylab.plot(Hot_T2, 'b.-')  # observations 38, 39,110, and 154 are outliers
#
# # And we should also plot the scores:
# pylab.figure()
# pylab.plot(nipals_T[:, 0], nipals_T[:, 1], '.')
# pylab.grid()
#
# pylab.show()

# Confirm the outliers in the raw data, giving one extra point above and below
raw[37:41, :]
raw[109:112, :]
raw[153:156, :]

# Next step for you: exclude observation 38, 39, 110 and 154 and
# rebuild the PCA model. Can you interpret what the loadings, nipals_P, mean?
