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


u = RandomRDDs.normalVectorRDD(sc, 1000, 5)
v = u.map(lambda x: ( f(x[0], 0), f(x[1], 1), f(x[2], 2), f(x[3],3), f(x[4], 4) ))
print(type(v.take(5)))
print(v.take(5))

# plt.plot(v.map(lambda x: x[0]).collect())
# plt.plot(v.map(lambda x: x[1]).collect())
# plt.plot(v.map(lambda x: x[2]).collect())
# plt.plot(v.map(lambda x: x[3]).collect())
# plt.plot(v.map(lambda x: x[4]).collect())
plt.plot(v.collect())
plt.show(block=True)

sc.stop()