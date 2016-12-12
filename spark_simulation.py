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

# Download the CSV data file from:
# http://datasets.connectmv.com/info/silicon-wafer-thickness
# raw = np.genfromtxt('silicon-wafer-thickness.csv', delimiter=',', skip_header=1)


N_COMPONENTS = 5
GAUSSIAN_DATA_SIZE = 10000
NEW_DATA_SIZE = 2000
TS_FREQUENCY = "10s"
N_INDEXES = 1
ERROR_FACTOR = np.ones(N_COMPONENTS)
# ERROR_FACTOR = [0.1, 0.6, 1, 0.2, 0.5]
WEIGHT_FACTOR = np.ones(N_COMPONENTS)
# WEIGHT_FACTOR = [  9.36023523e-01,   3.62926651e-02,   1.83666150e-02,    7.15911735e-03,   7.56237144e-04  ]


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

print(str(datetime.now()), "calculating normal vectors")
u = RandomRDDs.normalVectorRDD(sc, GAUSSIAN_DATA_SIZE, N_COMPONENTS)
print(str(datetime.now()), "applying normal factors")
v = u.map(lambda x: transform_normal(x, mus, sigmas)).cache()
print(str(datetime.now()), "done")
print(v.take(5))

# TODO: calculate and store gaussian inverted

columns = ["c"+str(i) for i in range(N_COMPONENTS)]
vs = sqlContext.createDataFrame(v, columns)
vs.printSchema()
vs.write.format("com.mongodb.spark.sql.DefaultSource").mode("overwrite").save()

# df = sqlContext.read.format("com.mongodb.spark.sql.DefaultSource").load()
# df.printSchema()
# print(df.map(lambda x: x["c0"]).take(5))
#
# plt.plot(df.map(lambda x: x["c0"]).collect())
# plt.show(block=True)

