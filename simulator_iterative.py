# Limitations: does not handle missing data

import numpy as np
import pandas as pd
from docutils.nodes import generated
from nltk.cluster.gaac import GAAClusterer

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
N_INDEXES = 20
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
save_matrix("inverse_X_gaussian.csv", inverse_gaussian, data.columns)


random_indexes = np.random.randint(0, GAUSSIAN_DATA_SIZE, size=NEW_DATA_SIZE)
# Distribution is lost?
# generated_gaussian1 = generated_gaussian[[102600,335105,347656,438732,280881,363283,488650,42246,91596,86114,168615,174505,58652,58652,427284,232152,450102,323120,76409,96776,177159,345244,31241,475613,395275,209327,115568,395275,489529,86013,395275,321606,11966,6,475613,288130,485036,475613,486113,37599,116920,209327,198688,37599,288130,288130,95297,67171,457076,485036,145922,95297,185395,472992,198688,67171,288130,95297,472992,95297,198688,288130,37599,275051,6,10537,228402,10537,209327,404524,257899,482521,10537,494694,423108,454167,63939,86013,401156,257899,362899,448933,485455,218759,449159,288130,345244,167952,67171,11966,486113,480859,9137,116920,395275,167952,321606,395275,132076,252462,475613,252462,491382,9137,6,489529,9137,9137,86013,475613,489529,489529,489529,491382,132076,209327,9137,209327,9137,491382,395275,321606,191007,279790,384017,217138,20568,418152,196131,490119,279790,90671,88730,399733,88730,418152,76074,55860,425203,425203,20568,147767,34263,350547,123955,350547,62835,88730,402834,402834,139726,217138,139726,63939,196131,384017,20568,63939,139726,285798,399733,418152,76074,475174,399733,352665,218292,490119,343696,38035,399733,194249,15570,218292,41288,300445,399438,433380,284803,370785,191462,360029,438888,160119,360029,191462,469150,20282,275845,208347,296743,433380,303948,208954,327929,384607,160119,384607,257246,420520,160119,48066,94782,181338,113750,159064,7968,181338,308018,21079,7968,222680,308018,308996,308996,113750,21079,21079,21079,308996,71351,1579,222680,406933,438103,34150,311363,354818,135905,311363,198108,198108,311363,53822,110564,198108,102251,102251,198108,34150,481655,368102,487460,26069,21079,359266,438103,26069,359266,21079,136402,160432,135905,34150,263041,311363,263041,263041,1579,416617,21079,308018,181338,304329,413773,151054,271591,125109,190629,413773,189389,323999,125109,403514,22482,472289,189389,403514,442025,167513,472289,403514,125109,125109,472289,472289,413773,442025,472289,189389,420861,319210,420861,473993,442025,442025,167513,362202,473993,357499,387628,199421,385191,229386,492067,482507,199421,295115,199421,442304,492067,492002,97741,293281,492002,190629,420861,125109,323999,245236,236081,17681,302521,420861,17681,188809,176393,472289,420861,413773,319210,472289,189389,100155,22482,245236,442025,318765,245236,271591,236081,357079,413773,245236,319210,271591,188809,319210,318765,125109,472289,323999,17681,57479,236081,420861,403514,323999,188809,472289,106511,420861,357499,17681,106511,57479,403514,403514,388508,17681,420861,357079,119010,1117,482521,38256,229773,38256,423108,14765,75298,1117,471112,36171,423108,46607,40631,14765,75298,324353,40631,116196,295115,295115,324353,263778,40631,36171,361883,40631,324353,14765,293281,419438,324353,36171,221904,101616,263778,167513,380986,295115,295115,125109,357499,293677,397164,52439,52439,233215,357499,190629,397164,472289,17681,106511,106511,290863,472289,373633,125109,472289,17681,236081,413773,413773,420861,323999,188809,190629,188809,442025,472289,450744,106511,100155,17681,125109,189389,17681,245236,125109,271591,450744,106511,149184,373633,472289,321882,106511,373633,152814,106511,450744,472289,106511,17681,357499,106511,152814,373633,290863,309303,155662,373633,309303,123088,406949,63569,454686,63569,291267,127930,16843,291267,258040,139961,340438,89717,297256,247314,139961,183256,205747,85676,34345,367137,389073,219494,120858], :]
generated_gaussian1 = generated_gaussian[random_indexes, :]
# generated_gaussian1 = generated_gaussian[[], :]
inverse_gaussian1 = np.dot(generated_gaussian1, nipals_P.T) + np.mean(raw, axis=0)
save_matrix("inverse_X_gaussian1.csv", inverse_gaussian1, data.columns)



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
error_factor = np.zeros(N_COMPONENTS)
chosen_indexes = np.zeros(NEW_DATA_SIZE)
for i in range(NEW_DATA_SIZE):
    print("Iteration", i)
    j = 0
    # Array to store each predicted point
    preds = np.zeros(N_COMPONENTS)
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
        add_error = error_factor[j] * add_error
        # print("Adding error using RMSE", min_rmse, ":", add_error)
        preds[j] = predictions_ARIMA1 + add_error
        j += 1

    # Euclidean distance
    print("preds", preds)
    # distances = np.sqrt(((generated_gaussian_copy - preds) ** 2).sum(axis=1))
    # standardized distances
    # distances = np.sqrt(((scale(generated_gaussian_copy) - scale(preds)) ** 2).sum(axis=1))
    # distances = np.sqrt(((scaled_generated_gaussian - scale(preds)) ** 2).sum(axis=1))

    distances = cdist([preds], generated_gaussian_copy, 'mahalanobis', VI=None)[0]

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
