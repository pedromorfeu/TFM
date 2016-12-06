import matplotlib.pylab as plt

# # Path for spark source folder
# os.environ['SPARK_HOME']="C:/BigData/Flywire/spark-1.6.2-bin-hadoop2.6"
#
# # Append pyspark to Python Path
# sys.path.append("C:/BigData/Flywire/spark-1.6.2-bin-hadoop2.6/python")
#
# Or: set PYTHONPATH and SPARK_HOME in Edit configurations

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

