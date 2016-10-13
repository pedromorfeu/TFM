# Limitations: does not handle missing data

import numpy as np
import pandas as pd
from util import *
from matplotlib import pylab
from datetime import datetime
from statsmodels.tsa.stattools import arma_order_select_ic
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# Download the CSV data file from:
# http://datasets.connectmv.com/info/silicon-wafer-thickness
# raw = np.genfromtxt('silicon-wafer-thickness.csv', delimiter=',', skip_header=1)


N_COMPONENTS = 5
NEW_DATA_SIZE = 100000
TS_FREQUENCY = "5s"


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
svd_P = v.T[:, range(0, N_COMPONENTS)]
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
nipals_T = np.zeros((N, N_COMPONENTS))
# loadings
nipals_P = np.zeros((K, N_COMPONENTS))

tolerance = 1E-10
# for each component
for a in range(N_COMPONENTS):

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

generated_X = np.zeros((NEW_DATA_SIZE, N_COMPONENTS))
for i in range(N_COMPONENTS):
    # calculate normal distribution by component and store it in column i
    # generated_X[:, i] = np.random.normal(mus[i], sigmas[i], NEW_DATA_SIZE)
    # alternative normal:
    generated_X[:, i] = mus[i] + sigmas[i] * np.random.randn(NEW_DATA_SIZE)
    # generate random not-normal:
    # generated_X[:, i] = mus[i] + sigmas[i] * np.random.rand(1, NEW_DATA_SIZE)
    
# invert matrix: dot product between random data and the loadings, nipals_P
XX = np.dot(generated_X, nipals_P.T) + np.mean(raw, axis=0)
# XX = np.dot(nipals_T, nipals_P.T) + np.mean(raw, axis=0)
print_matrix("XX", XX)

save_matrix("inverse_X_gaussian.csv", XX, data.columns)

exit()

### Time series
for i in range(N_COMPONENTS):
    print("Time series analysis for component", i+1)
    # time serie for component
    timeseries = pd.Series(nipals_T[:, i], index=data.index)
    # observations per day
    dates_count = timeseries.groupby(lambda x: x.date()).count()
    # day with more observations
    date = dates_count.idxmax().strftime("%Y-%m-%d")
    print("The date with more observations is", date)
    # Specify a date to analyze the timeseries
    timeseries_sample = timeseries[date]
    print(timeseries_sample.head())
    # Resample and interpolate
    print("Resampling time series by", TS_FREQUENCY)
    timeseries_sample = timeseries_sample.resample(TS_FREQUENCY).mean().interpolate()
    print(timeseries_sample.head())
    stationary = test_stationarity(timeseries_sample, _plot=False, _critical="5%")
    print("Stationary?", stationary)
    # not stationary -> must be stabilized

    # Differencing
    print("Differencing the time series...")
    timeseries_sample_diff = timeseries_sample.sub(timeseries_sample.shift())
    print("Missing values:", not np.all(np.isfinite(timeseries_sample_diff)))
    timeseries_sample_diff.index[np.isinf(timeseries_sample_diff)]
    timeseries_sample_diff.index[np.isnan(timeseries_sample_diff)]
    timeseries_sample_diff.dropna(inplace=True)
    print("Missing values:", not np.all(np.isfinite(timeseries_sample_diff)))
    # This appears to have reduced trend considerably. Lets verify using our plots:
    stationary = test_stationarity(timeseries_sample_diff, _plot=False, _critical="5%")
    print("Stationary?", stationary)
    if not stationary:
        # TODO: try other methods to make the timeseries stationary
        raise ValueError("Timeseries is not stationary after differencing.")

    #Forecasting
    # plot_acf_pacf(timeseries_sample_diff)
    print(str(datetime.now()), "Calculating AR and MA orders...")
    res = arma_order_select_ic(timeseries_sample_diff, ic=['aic', 'bic'], trend='nc', max_ar=3, max_ma=3)
    print(str(datetime.now()), "AR and MA orders calculated")
    # , fit_kw={"method" : "css"}
    # AIC and BIC min order (AR, MA) = (p, q)
    aic = res.aic_min_order
    bic = res.bic_min_order
    print("AIC=", aic)
    print("BIC=", bic)

    p = aic[1]
    q = aic[0]
    d = 1

    print("Creating model ARIMA(p,d,q)=", p, d, q)
    model = ARIMA(timeseries_sample, order=(p, d, q))
    print(str(datetime.now()), "Fitting model")
    results_ARIMA = model.fit(disp=-1)
    print(str(datetime.now()), "Model fitted")
    fitted = results_ARIMA.fittedvalues
    print("Predicting...")
    predictions = results_ARIMA.predict(start=1, end=NEW_DATA_SIZE, typ='levels')
    print(fitted)
    print(predictions)

    # plt.clf()
    # Markers plot
    # plt.plot(timeseries_sample, 'o', markersize=6, markeredgewidth=1, alpha=0.7)
    # plt.plot(predictions_ARIMA, '^', markersize=6, markeredgewidth=1, alpha=0.7)
    # plt.plot(timeseries_sample)
    # plt.plot(predictions_ARIMA)
    error = predictions-timeseries_sample
    print("Missing values:", not np.all(np.isfinite(error)))
    error.index[np.isinf(error)]
    error.index[np.isnan(error)]
    error.dropna(inplace=True)
    rmse = np.sqrt(sum(error**2)/len(timeseries_sample))
    print("RMSE", rmse)
    # plt.title('RMSE: %.4f'% rmse)
    # plt.show()

    # add noise
    predictions = predictions + np.random.normal(0, predictions.std(), NEW_DATA_SIZE)

    generated_X[:, i] = predictions

# invert matrix: dot product between random data and the loadings, nipals_P
XX = np.dot(generated_X, nipals_P.T) + np.mean(raw, axis=0)
# XX = np.dot(nipals_T, nipals_P.T) + np.mean(raw, axis=0)
print_matrix("XX", XX)

save_matrix("inverse_X.csv", XX, data.columns)


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
print("Hot_T2", Hot_T2)

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
