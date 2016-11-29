# Limitations: does not handle missing data

import numpy as np
import pandas as pd
from util import *
from warnings import warn
from matplotlib import pylab
from datetime import datetime
from statsmodels.tsa.stattools import arma_order_select_ic
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import SARIMAX
from scipy.spatial.distance import cdist

# Download the CSV data file from:
# http://datasets.connectmv.com/info/silicon-wafer-thickness
# raw = np.genfromtxt('silicon-wafer-thickness.csv', delimiter=',', skip_header=1)


N_COMPONENTS = 5
GAUSSIAN_DATA_SIZE = 500000
NEW_DATA_SIZE = 500
TS_FREQUENCY = "10s"
N_INDEXES = 1
ERROR_FACTOR = np.ones(N_COMPONENTS)
# ERROR_FACTOR = [0.1, 0.6, 1, 0.2, 0.5]
WEIGHT_FACTOR = [  9.36023523e-01,   3.62926651e-02,   1.83666150e-02,    7.15911735e-03,   7.56237144e-04]


# If the frequency is higher than the sample steps, then we have more real data
# If we interpolate, then we are introducing new data, which is induced

start = datetime.now()

data = pd.read_csv("ip.txt", sep="\s+\t", engine="python", parse_dates=[0], date_parser=parse_dates,
               index_col="Tiempoinicio", skip_blank_lines=True, na_values="")

print(type(data))
print(data.columns)
print(data.head())
print(data.dtypes)
print(data.index)
print(data.shape)

# observations per day
dates_count = data.groupby(lambda x: x.date()).count()
# day with more observations
date = dates_count.idxmax()[0].strftime("%Y-%m-%d")
print("The date with more observations is", date)
date = "2015-10-06"
# Specify a date to analyze the timeseries
data = data[date]



# Resampling and Interpolation
data = data.resample(TS_FREQUENCY).mean().interpolate()
save_matrix("data.csv", data.values, data.columns)


save_plots(data.values, data.columns, "_original", "figures")


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


# from sklearn.decomposition import PCA
# pca = PCA(n_components=N_COMPONENTS, whiten=True)
# pca.fit(raw)
# pca.explained_variance_
# pca.explained_variance_ratio_


# We could of course use SVD ...
print(str(datetime.now()), "Calculating SVD...")
u, d, v = np.linalg.svd(X[:, :])
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
save_matrix("nipals_T_ts.csv", nipals_T, columns_names=(["time"] + list(range(N_COMPONENTS))), index_ts=data.index)


### Generate Gaussian data
mus = np.mean(nipals_T, axis=0)
sigmas = np.std(nipals_T, axis=0)

generated_gaussian = np.zeros((GAUSSIAN_DATA_SIZE, N_COMPONENTS))
for i in range(N_COMPONENTS):
    # calculate normal distribution by component and store it in column i
    generated_gaussian[:, i] = np.random.normal(mus[i], sigmas[i], GAUSSIAN_DATA_SIZE)
    # alternative normal:
    # generated_gaussian[:, i] = mus[i] + sigmas[i] * np.random.randn(NEW_DATA_SIZE)
    # generate random not-normal:
    # generated_gaussian[:, i] = mus[i] + sigmas[i] * np.random.rand(1, NEW_DATA_SIZE)
print_matrix("generated_gaussian", generated_gaussian)
# save_matrix("generated_gaussian.csv", generated_gaussian, [x for x in range(N_COMPONENTS)])

# invert matrix: dot product between random data and the loadings, nipals_P
inverse_gaussian = np.dot(generated_gaussian, nipals_P.T) + np.mean(raw, axis=0)
#XX = np.dot(nipals_T, nipals_P.T) + np.mean(raw, axis=0)
print_matrix("inverse_gaussian", inverse_gaussian)
# save_matrix("inverse_X_gaussian.csv", inverse_gaussian, data.columns)


# Distribution is lost?
# generated_gaussian1 = generated_gaussian[[49759,54624,387021,470272,351343,280105,466031,262411,351343,461796,49759,461796,421062,488637,267797,336513,298218,351343,470272,280105,488637,488637,461796,439268,191932,191932,490392,471595,162526,49759,470272,438261,439268,354944,191932,162526,125887,21877,81369,280105,49759,298218,51884,174147,476307,249508,49759,470272,54624,249508,351343,351343,351343,54624,63723,387021,105950,105950,249508,351343,298218,387021,453389,63723,298218,209531,476307,421062,54624,461796,126435,336513,63723,387021,351343,351343,418841,298218,336513,249508,249508,209531,461796,191932,470272,453389,280105,298218,471595,63723,488637,336513,54624,418841,438261,418841,209531,191932,196019,42245,162526,81369,619,463750,45697,321104,296778,321104,252665,36295,339945,321104,55584,252665,66269,252665,252665,371227,337810,446308,107608,259422,179787,448397,378798,85130,448397,274824,259422,219898,145734,498819,97469,235934,219898,107829,147629,145470,249172,345322,345322,204786,322374,98778,428704,345322,67617,145734,145470,145470,147629,147629,84817,274824,266032,137424,259422,219898,235934,235934,266032,291196,235934,157776,298172,17870,151083,151083,284301,444308,304783,489786,284301,304783,317082,98778,317082,317082,489786,446701,322374,258694,489786,98778,468810,271536,283326,58657,43720,283326,127151,284301,345322,341100,151083,298172,204786,145734,97469,67617,313355,17870,145734,322374,266032,380702,157776,147629,313355,298172,291196,204786,147629,17870,17870,298172,313355,145734,284301,97469,145470,313355,291196,235934,151083,67617,235934,151083,271536,291196,284301,266032,271536,298172,157776,345322,254728,291196,266032,284301,235934,145734,266032,145734,249172,157776,204786,151083,254728,17870,341100,322374,444308,43720,58657,258694,271536,341100,151083,271536,127151,317082,345322,468810,304783,258694,98778,304783,58657,284301,17870,322374,291196,291196,145734,298172,291196,266032,284301,67617,271536,254728,157776,254728,157776,151083,249172,254728,254728,291196,341100,17870,98778,284301,43720,468810,254728,67617,423052,157776,291196,345322,291196,423052,298172,298172,157776,254728,423052,313355,17870,304783,254728,423052,345322,313355,369952,341100,444308,317082,468810,485436,478509,304783,16709,485436,468810,468810,254728,446701,258694,258694,283326,271536,341100,283326,283326,284301,341100,151083,254728,345322,444308,43720,284301,322374,345322,317082,254728,43720,283326,345322,127151,283326,98778,485436,489786,489786,489786,444308,227486,271536,345322,468810,317082,283326,16709,283326,485436,275288,331536,80395,399245,399245,489786,495451,283326,283326,283326,478509,446701,16709,258694,258694,485436,304783,444308,446701,43720,98778,485436,446701,258694,468810,43720,317082,284301,98778,468810,98778,485436,345322,271536,304783,468810,489786,43720,485436,43720,102837,446701,468810,446701,317082,304783,485436,444308,444308,322374,485436,227486,304783,102837,16709,80395,317082,127151,304783,322374,322374,151083,341100,284301,291196,249172,284301,254728,291196,151083,151083,317082,446701,468810,43720,254728,254728,17870,298172,322374,235934,345322,151083,271536,468810,291196,304783,345322,271536,345322,298172,271536,291196,249172,266032,291196,145470,67617,235934,345322,341100,468810,291196,317082,284301,258694,317082,304783,151083,43720,254728,266032,145470,291196,284301,423052,17870,423052,43720,322374,127151,127151,17870,317082,345322,446701,98778,283326,283326,446701,98778], :]
# generated_gaussian1 = generated_gaussian[[102600,335105,347656,438732,280881,363283,488650,42246,91596,86114,168615,174505,58652,58652,427284,232152,450102,323120,76409,96776,177159,345244], :]
# The plot shows non-normal distribution
# plt.hist(generated_gaussian1[:, 0])
# Seems that with random numbers the distribution is kept -> T2 Hotelling p-value=1
# random_indexes = np.random.randint(0, GAUSSIAN_DATA_SIZE, size=NEW_DATA_SIZE)
# generated_gaussian1 = generated_gaussian[random_indexes, :]
# generated_gaussian1 = generated_gaussian[[], :]
# inverse_gaussian1 = np.dot(generated_gaussian1, nipals_P.T) + np.mean(raw, axis=0)
# save_matrix("inverse_X_gaussian1.csv", inverse_gaussian1, data.columns)

# from scipy.stats.mstats import normaltest
# normaltest(generated_gaussian[:, 0])
# normaltest(generated_gaussian1[:, 0])
#
# normaltest(np.random.normal(mus[0], sigmas[0], GAUSSIAN_DATA_SIZE))
# normaltest(mus[0] + sigmas[0] * np.random.randn(GAUSSIAN_DATA_SIZE))


save_plots(inverse_gaussian[:NEW_DATA_SIZE, :], data.columns, "_inverse_gaussian", "figures")


### Time series
models = []
timeseries_samples = []
for i in range(N_COMPONENTS):
    print("Time series analysis for component", i)
    # time serie for component
    timeseries = pd.Series(nipals_T[:, i], index=data.index)
    print_timeseries("timeseries", timeseries)

    # # observations per day
    # dates_count = timeseries.groupby(lambda x: x.date()).count()
    # # day with more observations
    # date = dates_count.idxmax().strftime("%Y-%m-%d")
    # print("The date with more observations is", date)
    # date="2015-10-06"
    # # Specify a date to analyze the timeseries
    # timeseries_sample = timeseries[date]

    timeseries_sample = timeseries
    print_timeseries("timeseries_sample", timeseries_sample)
    # Resample and interpolate
    # print("Resampling time series by", TS_FREQUENCY)
    # timeseries_sample = timeseries_sample.resample(TS_FREQUENCY).mean().interpolate()
    timeseries_samples.append(timeseries_sample.values)
    # timeseries_sample = timeseries_sample.asfreq(TS_FREQUENCY, method="ffill")
    print_timeseries("timeseries_sample", timeseries_sample)
    stationary, test_value = test_stationarity(timeseries_sample, _plot=False, _critical="5%")

    # not stationary -> must be stabilized

    # Stabilizing the variance
    subtract_constant = 0
    # For negative data, you can add a suitable constant to make all the data positive
    # before applying the transformation.
    print("any negative?", np.any(timeseries_sample < 0))
    if np.any(timeseries_sample < 0):
        subtract_constant = abs(min(timeseries_sample)) + 1
    timeseries_sample_positive = timeseries_sample + subtract_constant
    print_timeseries("timeseries_sample_positive", timeseries_sample_positive)
    print("any negative?", np.any(timeseries_sample_positive < 0))
    # ts_log = np.log(timeseries_sample_positive)
    ts_log = timeseries_sample
    print_timeseries("ts_log", ts_log)
    print("Missing values?", not np.all(np.isfinite(ts_log)))

    moving_avg = ts_log.rolling(12).mean()
    ts_log_moving_avg_diff = ts_log - moving_avg
    ts_log_moving_avg_diff.head(5)
    ts_log_moving_avg_diff.dropna(inplace=True)
    ts_log_moving_avg_diff.head()
    stationary, test_value_moving_avg_diff = test_stationarity(ts_log_moving_avg_diff, _plot=False, _critical="5%")

    expwighted_avg = ts_log.ewm(halflife=20).mean()
    # plt.plot(ts_log)
    # plt.plot(expwighted_avg, color='red')
    ts_log_ewma_diff = ts_log - expwighted_avg
    stationary, test_value_ewma_diff = test_stationarity(ts_log_ewma_diff, _plot=False, _critical="5%")

    # Differencing
    print("Differencing the time series...")
    ts_log_diff = ts_log - ts_log.shift()
    print("Missing values?", not np.all(np.isfinite(ts_log_diff)))
    ts_log_diff.index[np.isinf(ts_log_diff)]
    ts_log_diff.index[np.isnan(ts_log_diff)]
    ts_log_diff.dropna(inplace=True)
    print("Missing values:", not np.all(np.isfinite(ts_log_diff)))
    print_timeseries("ts_log_diff", ts_log_diff)
    # This appears to have reduced trend considerably. Lets verify using our plots:
    stationary, test_value_diff = test_stationarity(ts_log_diff, _plot=False, _critical="5%")

    if not stationary:
        # TODO: try other methods to make the timeseries stationary
        raise ValueError("Timeseries is not stationary after differencing.")

    # Minimum value for integration part (d):
    # If differenciated series is more stationary, then use i=1
    min_i = 0
    if test_value_diff < test_value:
        min_i = 1

    # Forecasting
    # plot_acf_pacf(ts_log_diff)
    # print(str(datetime.now()), "component", i,"Calculating AR and MA orders...")
    # res = arma_order_select_ic(ts_log_diff, ic=['aic', 'bic'], trend='nc')
    # print(str(datetime.now()), "AR and MA orders calculated", res.aic_min_order, res.bic_min_order)
    # # , fit_kw={"method" : "css"}
    # # AIC and BIC min order (AR, MA) = (p, q)
    # aic = res.aic_min_order
    # bic = res.bic_min_order
    # print("AIC=", aic)
    # print("BIC=", bic)
    #
    # p = aic[0]
    # q = aic[1]
    # d = 0

    # Calculate best order (order with minimum error)
    (min_rmse, p, d, q) = arima_order_select(ts_log, min_i=min_i)

    print("Creating model (p,d,q)=(%i,%i,%i)" % (p, d, q))
    model = SARIMAX(ts_log, order=(p, d, q))
    print(str(datetime.now()), "Fitting model...")
    results_ARIMA = model.fit(disp=-1)
    print(str(datetime.now()), "Model fitted")
    print(str(datetime.now()), "Predicting...")
    # predictions_ARIMA = results_ARIMA.predict(start=ts_log.shape[0], end=ts_log.shape[0]+NEW_DATA_SIZE-1)
    predictions_ARIMA = results_ARIMA.predict(start=ts_log.shape[0], end=ts_log.shape[0])
    print(str(datetime.now()), "Predicted")

    # print(ts_log.tail(5))
    # print(predictions_ARIMA.tail(5))
    # out_of_sample = predictions_ARIMA
    # ts_log_predicted = ts_log.append(predictions_ARIMA)
    # print(predictions_ARIMA.tail(5))
    ts_log_predicted = results_ARIMA.fittedvalues

    models.append((results_ARIMA, ts_log_predicted, min_rmse))

print("Models calculated and stored")


# Calculate best order (order with minimum error) again
# (min_rmse, p, d, q) = arima_order_select(predictions_ARIMA)

print(str(datetime.now()), "Iterative prediction with new data")
models_iterative = models.copy()
generated_gaussian_copy = generated_gaussian.copy()
scaled_generated_gaussian = scale(generated_gaussian.copy())
generated_X = np.zeros((NEW_DATA_SIZE, N_COMPONENTS))
chosen_indexes = np.zeros(NEW_DATA_SIZE)
i = 0
for i in range(NEW_DATA_SIZE):
    print("Iteration", i)
    j = 0
    # Array to store each predicted point
    preds = np.zeros(N_COMPONENTS)
    preds_transformed = np.zeros(N_COMPONENTS)
    for (results_ARIMA, ts_log_predicted, min_rmse) in models_iterative:
        model1 = SARIMAX(ts_log_predicted, order=results_ARIMA.model.order)
        results_ARIMA1 = model1.filter(results_ARIMA.params)
        # results_ARIMA1 = model1.fit(disp=-1)
        predictions_ARIMA1 = results_ARIMA1.predict(start=ts_log_predicted.shape[0], end=ts_log_predicted.shape[0])
        # print("predictions_ARIMA1", predictions_ARIMA1)
        ts_log_predicted1 = ts_log_predicted.append(predictions_ARIMA1)
        models_iterative[j] = (results_ARIMA1, ts_log_predicted1, min_rmse)
        # add random error from series standard deviation and mean 0
        add_error = 0 + min_rmse * np.random.randn()
        # add some error atenuation
        add_error = ERROR_FACTOR[j] * add_error
        # print("Adding error using RMSE", min_rmse, ":", add_error)
        preds[j] = predictions_ARIMA1 + add_error

        # Euclidean distance
        distances = np.sqrt( ( (generated_gaussian_copy[:, j] - preds[j]) ** 2) )
        sorted_indexes = distances.argsort()[:N_INDEXES]
        random_index = np.random.randint(N_INDEXES)
        min_index = sorted_indexes[random_index]
        preds_transformed[j] = generated_gaussian_copy[min_index, j]
        j += 1

    print("preds", preds)

    if False:
        # Euclidean distance
        # distances = np.sqrt(((generated_gaussian_copy - preds) ** 2).sum(axis=1))
        # standardized distances
        # distances = np.sqrt(((scale(generated_gaussian_copy) - scale(preds)) ** 2).sum(axis=1))
        distances = np.sqrt( ( (WEIGHT_FACTOR * (scaled_generated_gaussian - scale(preds))) ** 2 ).sum(axis=1) )

        # distances = cdist([preds], generated_gaussian_copy, 'mahalanobis', VI=None)[0]
        # distances = cdist( ([WEIGHT_FACTOR * scale(preds)]), WEIGHT_FACTOR * (scaled_generated_gaussian), 'mahalanobis', VI=None)[0]

        # take first N_INDEXES nearest indexes
        sorted_indexes = distances.argsort()[:N_INDEXES]
        # select the nearest index
        # min_index = sorted_indexes[0]
        # select random index
        random_index = np.random.randint(N_INDEXES)
        min_index = sorted_indexes[random_index]
        preds_transformed = generated_gaussian_copy[min_index]
        chosen_indexes[i] = min_index
        # preds_transformed = preds
        # hypotesis: repetitions make results worse
        # generated_gaussian_copy = np.delete(generated_gaussian_copy, min_index, 0)

    generated_X[i] = preds_transformed
    print("preds_transformed", preds_transformed)

    # Store the new values for re-feeding the models in the next prediction
    j = 0
    for (results_ARIMA, ts_log_predicted, min_rmse) in models_iterative:
        ts_log_predicted.set_value(ts_log_predicted.last_valid_index(), preds_transformed[j])
        j += 1
print(str(datetime.now()), "Done iterative prediction")


f = open(os.path.join("generated", "chosen_indexes.csv"), "w")
for i in chosen_indexes:
    f.write(str(int(i)) + ",")
f.close()



print("Saving plots...")
plt.ioff()
for i in range(N_COMPONENTS):
    plt.clf()
    max_gaussian = models_iterative[i][1].shape[0]
    plt.plot(generated_gaussian[:max_gaussian, i], label="gaussian")
    plt.plot(models_iterative[i][1].values, label="prediction")
    plt.plot(timeseries_samples[i], label="original")
    title = "component" + str(i+1)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join("figures", title))
plt.ion()
print("Plots saved")


# models_iterative = models.copy()
# generated_X = np.zeros((NEW_DATA_SIZE, N_COMPONENTS))
# for i in range(N_COMPONENTS):
#     (results_ARIMA, ts_log_predicted, min_rmse) = models_iterative[i]
#     generated_X[:, i] = results_ARIMA.predict(start=0, end=NEW_DATA_SIZE-1)
#
# plt.plot(nipals_T[:, 0])
# plt.plot(generated_X[:, 0])


# distribution of generated_X isn't normal; however, generated_gaussian is
print_matrix("generated_X", generated_X)
save_matrix("generated_X.csv", generated_X, [x for x in range(N_COMPONENTS)])

# invert matrix: dot product between random data and the loadings, nipals_P
inverse = np.dot(generated_X[:], nipals_P.T) + np.mean(raw, axis=0)
# XX = np.dot(nipals_T, nipals_P.T) + np.mean(raw, axis=0)
print_matrix("XX", inverse)
save_matrix("inverse_X.csv", inverse, data.columns)


save_plots(inverse, data.columns, "_inverse", "figures")


save_all_plots(data.values, inverse, inverse_gaussian[:NEW_DATA_SIZE, :], data.columns, "_mixed", "figures")


# error = ts_log_predicted - ts_log
# print("Missing values:", not np.all(np.isfinite(error)))
# error.dropna(inplace=True)
# print("Missing values:", not np.all(np.isfinite(error)))
# rmse = np.sqrt(sum((error) ** 2) / len(ts_log))
# print("RMSE", rmse)

# plt.clf()
# plt.plot(timeseries_sample)
# plt.plot(predictions_ARIMA[:max(timeseries_sample.index) + 500])
# plt.title('RMSE: %.4f' % rmse)
# plt.show(block=True)
#
# plt.clf()
# plt.plot(timeseries_sample)
# plt.plot(predictions_ARIMA)
# plt.title('RMSE: %.4f' % rmse)
# plt.show(block=True)

# add noise
# predictions_ARIMA = predictions_ARIMA + np.random.normal(0, rmse, NEW_DATA_SIZE)

# generated_X[:, i] = predictions_ARIMA




# PCA also has two very important outputs we should calculate:

# The SPE_X, squared prediction error to the X-space is the residual distance
# from the model to each data point.
# SPE_X = np.sum(X ** 2, axis=1)
# print("SPE_X", X[:5, :])

# And Hotelling's T2, the directed distance from the model center to
# each data point.
inv_covariance = np.linalg.inv(np.dot(nipals_T.T, nipals_T) / N)
Hot_T2 = np.zeros((N, 1))
for n in range(N):
    Hot_T2[n] = np.dot(np.dot(nipals_T[n, :], inv_covariance), nipals_T[n, :].T)
# print("Hot_T2", Hot_T2)

# You can verify the NIPALS and SVD results are the same:
# (you may find that the signs have flipped, but this is still correct)
# nipals_T / svd_T
# nipals_P / svd_P

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

timedelta = datetime.now() - start
print("Total seconds", timedelta.total_seconds())
print("Total minutes", timedelta.total_seconds()/60)
