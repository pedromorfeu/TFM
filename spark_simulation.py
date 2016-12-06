import matplotlib.pylab as plt
import os
import sys

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
conf = SparkConf().setAppName("testApp").setMaster("local[4]")
sc = SparkContext(conf=conf)

print(sc)


# Generate a random double RDD that contains 1 million i.i.d. values drawn from the
# standard normal distribution `N(0, 1)`, evenly distributed in 10 partitions.
u = RandomRDDs.normalRDD(sc, 1000000, 4)
# Apply a transform to get a random double RDD following `N(1, 4)`.
v = u.map(lambda x: 1.0 + 2.0 * x)

v_data = v.collect()
print(type(v_data))
print(v_data)
print(v_data[:5])
print(v_data[-5:])

# plt.plot(v_data)
# plt.show(block=True)



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
GAUSSIAN_DATA_SIZE = 500000
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
