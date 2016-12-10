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
# standard normal distribution `N(0, 1)`, evenly distributed in 4 partitions.
u = RandomRDDs.normalRDD(sc, 1000000, 4)
# Apply a transform to get a random double RDD following `N(1, 2)`.
v = u.map(lambda x: 1.0 + 2.0 * x)

v_data = v.collect()
print(type(v_data))
# print(v_data)
print(v_data[:5])
print(v_data[-5:])

# plt.plot(v_data)
# plt.show(block=True)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
print(client)
db = client.simulation
print(db)
collection = db.random
print(collection)

collection.delete_many({})

sc.stop()