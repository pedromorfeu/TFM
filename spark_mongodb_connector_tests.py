import matplotlib.pylab as plt
import os
import sys
import numpy as np

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
    .setAppName("testApp")\
    .setMaster("local[4]")\
    .set("spark.mongodb.input.uri", "mongodb://127.0.0.1/test.myCollection?readPreference=primaryPreferred")\
    .set("spark.mongodb.output.uri", "mongodb://127.0.0.1/test.myCollection")

# Note: set "spark.jars.packages org.mongodb.spark:mongo-spark-connector_2.10:1.1.0" in $SPARK_HOME/conf/spark-defaults.conf

sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

print(sc)
print(sqlContext)

# charactersRdd = sc.parallelize([("Bilbo Baggins",  50), ("Gandalf", 1000), ("Thorin", 195), ("Balin", 178), ("Kili", 77),
#    ("Dwalin", 169), ("Oin", 167), ("Gloin", 158), ("Fili", 82)])
# characters = sqlContext.createDataFrame(charactersRdd, ["name", "age"])
# characters.printSchema()
# characters.write.format("com.mongodb.spark.sql.DefaultSource").mode("overwrite").save()


# Generate a random double RDD that contains 1 million i.i.d. values drawn from the
# standard normal distribution `N(0, 1)`, evenly distributed in 4 partitions.
u = RandomRDDs.normalRDD(sc, 100000)
# Apply a transform to get a random double RDD following `N(1, 2)`.
v = u.map(lambda x: (1.0 + 2.0 * x, ))
print(v.take(5))

vs = sqlContext.createDataFrame(v, ["number"])
vs.printSchema()
vs.write.format("com.mongodb.spark.sql.DefaultSource").mode("overwrite").save()



df = sqlContext.read.format("com.mongodb.spark.sql.DefaultSource").load()
df.printSchema()

rdd = df.map(lambda x: (x["number"]))
print(rdd.take(5))

x1 = 1
distances_rdd = rdd.map(lambda x: (x, np.sqrt(((x - x1) ** 2))) )
print(distances_rdd.take(5))

print(distances_rdd.min(lambda x: x[1]))

sc.stop()