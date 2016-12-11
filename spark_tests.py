import matplotlib.pylab as plt
import os
import sys
import numpy as np
from datetime import datetime

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


sigmas = [1, 1, 1, 1, 1]
means = [2, 3, 4, 5, 6]

# rdds = []
# for i in range(5):
#     # Generate a random double RDD that contains 1 million i.i.d. values drawn from the
#     # standard normal distribution `N(0, 1)`, evenly distributed in 4 partitions.
#     u = RandomRDDs.normalRDD(sc, 1000)
#     # To transform the distribution in the generated RDD from standard normal to some other normal N(mean, sigma^2),
#     # use RandomRDDs.normal(sc, n, p, seed) .map(lambda v: mean + sigma * v)
#     # Apply a transform to get a random double RDD following `N(1, 2)`.
#     v = u.map(lambda x: means[i] + sigmas[i] * x).cache()
#     rdds.append(v)

# print(rdds)
# print(type(rdds[0].take(5)))
# print(rdds[0].take(5))
# print(rdds[1].take(5))
# print(rdds[2].take(5))
# print(rdds[3].take(5))
# print(rdds[4].take(5))
#
# plt.plot(rdds[0].collect())
# plt.plot(rdds[1].collect())
# plt.plot(rdds[2].collect())
# plt.plot(rdds[3].collect())
# plt.plot(rdds[4].collect())
# plt.show(block=True)


def f(x, i):
    return means[i] + sigmas[i] * x


print(str(datetime.now()), "calculating normal vectors")
u = RandomRDDs.normalVectorRDD(sc, 10000000, 5)
print(str(datetime.now()), "applying normal factors")
v = u.map(lambda x: ( f(x[0], 0), f(x[1], 1), f(x[2], 2), f(x[3],3), f(x[4], 4) )).cache()
print(str(datetime.now()), "done")


def calculate_min_distance(_v, _x1):
    # distances = np.sqrt(((scale(generated_gaussian_copy) - scale(preds)) ** 2).sum(axis=1))
    distances_rdd = _v.map(lambda x: (x, np.sqrt((np.array(x) - np.array(_x1)) ** 2).sum()) )
    # print("distances_rdd", distances_rdd)
    min_distance = distances_rdd.min(lambda x: x[1])
    # print("min distance", min_distance)
    return min_distance


x1 = [2, 3, 4, 5, 6]
print(str(datetime.now()), "calculating min distance for", x1)
d1 = calculate_min_distance(v, x1)
print(str(datetime.now()), "min distance", d1)

x2 = [3, 4, 5, 6, 7]
print(str(datetime.now()), "calculating min distance for", x2)
d2 = calculate_min_distance(v, x2)
print(str(datetime.now()), "min distance", d2)

# Sanity check - the point exists in the dataset
# print(distances_rdd.filter(lambda x: x[0] == min_point).count())

plt.plot(v.take(10000))
plt.show(block=True)

sc.stop()