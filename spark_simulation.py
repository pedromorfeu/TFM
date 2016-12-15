# # Path for spark source folder
# os.environ['SPARK_HOME']="C:/BigData/TFM/spark-1.6.2-bin-hadoop2.6"
#
# # Append pyspark to Python Path
# sys.path.append("C:/BigData/TFM/spark-1.6.2-bin-hadoop2.6/python")
#
# Or: set PYTHONPATH and SPARK_HOME in Edit configurations
# PYTHONPATH=C:/BigData/TFM/spark-1.6.2-bin-hadoop2.6/python
# SPARK_HOME=C:/BigData/TFM/spark-1.6.2-bin-hadoop2.6

try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.sql import SQLContext, Row
    from pyspark.sql.types import *
    from pyspark.mllib.random import RandomRDDs
    print ("Successfully imported Spark Modules")
except ImportError as e:
    print ("Can not import Spark Modules", e)


# Initialize SparkContext
conf = SparkConf()\
    .setAppName("SparkSimulation")\
    .setMaster("local[4]")\
    .set("spark.mongodb.input.uri", "mongodb://127.0.0.1/simulation.gaussian?readPreference=primaryPreferred")\
    .set("spark.mongodb.output.uri", "mongodb://127.0.0.1/simulation.gaussian")

# Note: set "spark.jars.packages org.mongodb.spark:mongo-spark-connector_2.10:1.1.0" in $SPARK_HOME/conf/spark-defaults.conf
sc = SparkContext(conf=conf)
print(sc)

sqlContext = SQLContext(sc)
print(sqlContext)

### SIMULATOR
# Limitations: does not handle missing data

import numpy as np
import pandas as pd
from util import *
from warnings import warn
from datetime import datetime
from statsmodels.tsa.stattools import arma_order_select_ic
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import SARIMAX
from scipy.spatial.distance import cdist
from pymongo import MongoClient

# Download the CSV data file from:
# http://datasets.connectmv.com/info/silicon-wafer-thickness
# raw = np.genfromtxt('silicon-wafer-thickness.csv', delimiter=',', skip_header=1)


N_COMPONENTS = 5
GAUSSIAN_DATA_SIZE = 10000
NEW_DATA_SIZE = 5
TS_FREQUENCY = "10s"
N_INDEXES = 1
ERROR_FACTOR = np.ones(N_COMPONENTS)
# ERROR_FACTOR = [0.1, 0.6, 1, 0.2, 0.5]
WEIGHT_FACTOR = np.ones(N_COMPONENTS)
# WEIGHT_FACTOR = [  9.36023523e-01,   3.62926651e-02,   1.83666150e-02,    7.15911735e-03,   7.56237144e-04  ]


client = MongoClient("mongodb://localhost:27017")
print(client)
db = client.simulation
print(db)
collection = db.generated
print(collection)

# Clear collection
collection.delete_many({})


# If the frequency is higher than the sample steps, then we have more real data
# If we interpolate, then we are introducing new data, which is induced

start = datetime.now()

data = pd.read_csv("ip.txt", sep="\s+\t", engine="python", parse_dates=[0], date_parser=parse_dates,
               index_col="Tiempoinicio", skip_blank_lines=True, na_values="")

# data = pd.read_csv("ip_gen.txt", index_col="Tiempoinicio", parse_dates=[0])

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


save_data_plot(_data=data, _filename="original")
save_plot_per_column(data.values, data.columns, "_original", "figures")


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
normal_vector_rdd, d, generated_gaussian_rdd = np.linalg.svd(X[:, :])
print(str(datetime.now()), "Done")
print(normal_vector_rdd.shape)
print(d.shape)
print(generated_gaussian_rdd.shape)
print("v", generated_gaussian_rdd)

# U, S, V = self._fit(X)
# U = U[:, :self.n_components_]
# U *= S[:self.n_components_]
U = normal_vector_rdd[:, :2]
U *= d[:2]
print_matrix("U", U)

# Transpose the "v" array from SVD, which contains the loadings, but retain
# only the first A columns
svd_P = generated_gaussian_rdd.T[:, range(0, N_COMPONENTS)]
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

print(str(datetime.now()), "calculating normal vectors")
normal_vector_rdd = RandomRDDs.normalVectorRDD(sc, GAUSSIAN_DATA_SIZE, N_COMPONENTS)
print(str(datetime.now()), "applying normal factors")
generated_gaussian_rdd = normal_vector_rdd.map(lambda x: transform_normal(x, mus, sigmas)).cache()
print(str(datetime.now()), "done")
print(generated_gaussian_rdd.take(5))

schema_component = ["type"] + ["c" + str(i) for i in range(N_COMPONENTS)]
schema_rdd = generated_gaussian_rdd.map(lambda x: ("component",) + x)
generated_gaussian_df = sqlContext.createDataFrame(schema_rdd, schema_component)
generated_gaussian_df.printSchema()
generated_gaussian_df.write.format("com.mongodb.spark.sql.DefaultSource").mode("overwrite").save()

# invert matrix: dot product between random data and the loadings, nipals_P
inverse_gaussian_rdd = generated_gaussian_rdd.map(lambda x: np.dot(np.array(x), nipals_P.T) + np.mean(raw, axis=0))
print("inverse_gaussian", inverse_gaussian_rdd.take(2))
print(inverse_gaussian_rdd.first())
print(type(inverse_gaussian_rdd.first()))

schema_inverse = ["type"] + data.columns.values.tolist()
schema_rdd = inverse_gaussian_rdd.map(lambda x: ["inverse"] + x.tolist())
inverse_gaussian_df = sqlContext.createDataFrame(schema_rdd, schema_inverse)
inverse_gaussian_df.printSchema()
inverse_gaussian_df.write.format("com.mongodb.spark.sql.DefaultSource").mode("append").save()


# df = sqlContext.read.format("com.mongodb.spark.sql.DefaultSource").load()
# df.printSchema()
# print(df.map(lambda x: x["c0"]).take(5))
#
# plt.plot(df.map(lambda x: x["c0"]).collect())
# plt.show(block=True)

### Time series
models = []
timeseries_samples = []
i=0
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
# generated_gaussian_copy = generated_gaussian.copy()
# scaled_generated_gaussian = scale(generated_gaussian.copy(), generated_gaussian.mean(axis=0), generated_gaussian.std(axis=0))
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

        # # Euclidean distance
        # distances = np.sqrt( ( (generated_gaussian_copy[:, j] - preds[j]) ** 2) )
        # sorted_indexes = distances.argsort()[:N_INDEXES]
        # random_index = np.random.randint(N_INDEXES)
        # min_index = sorted_indexes[random_index]
        # preds_transformed[j] = generated_gaussian_copy[min_index, j]
        j += 1

    print("preds", preds)

    # Euclidean distance
    # distances = np.sqrt(((generated_gaussian_copy - preds) ** 2).sum(axis=1))
    distances = calculate_min_distance(generated_gaussian_rdd, preds, _n_points=N_INDEXES)
    # only points
    distances = [x[0] for x in distances]
    # standardized distances
    # distances = np.sqrt(((scale(generated_gaussian_copy) - scale(preds)) ** 2).sum(axis=1))
    # scaled_preds = scale(preds, generated_gaussian.mean(axis=0), generated_gaussian.std(axis=0))
    # distances = np.sqrt( ( (WEIGHT_FACTOR * (scaled_generated_gaussian - scaled_preds)) ** 2 ).sum(axis=1) )

    # distances = cdist([preds], generated_gaussian_copy, 'mahalanobis', VI=None)[0]
    # distances = cdist( ([WEIGHT_FACTOR * scaled_preds]), (WEIGHT_FACTOR * scaled_generated_gaussian), 'mahalanobis', VI=None)[0]

    # take first N_INDEXES nearest indexes
    # sorted_indexes = distances.argsort()[:N_INDEXES]
    # select the nearest index
    # min_index = sorted_indexes[0]
    # select random index
    random_index = np.random.randint(N_INDEXES)
    # min_index = sorted_indexes[random_index]
    preds_transformed = distances[random_index]
    # preds_transformed = generated_gaussian_copy[min_index]
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

    # Store predicted transformed point in MongoDB
    doc_component = dict(zip(schema_component, ("component", ) + preds_transformed))
    collection.insert_one(doc_component)

    # Calculate and store inverse point
    inverse = np.dot(preds_transformed, nipals_P.T) + np.mean(raw, axis=0)
    doc_inverse = dict(zip(schema_inverse, ["inverse"] + inverse.tolist()))
    collection.insert_one(doc_inverse)
print(str(datetime.now()), "Done iterative prediction")
