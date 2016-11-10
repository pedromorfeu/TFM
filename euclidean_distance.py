import numpy
from scipy.spatial import distance
from datetime import datetime
import json
from sys import getsizeof


p = (1, 2, 3)
q1 = (3, 4, 1)
q2 = (4, 6, 0)


# SciPy
print(distance.euclidean(p, q1))
print(distance.euclidean(p, q2))


# Numpy
def dist(x,y):
    return numpy.sqrt(numpy.sum((x-y)**2))
print(dist(numpy.array(p), numpy.array(q1)))
print(dist(numpy.array(p), numpy.array(q2)))


# Nearest points
points = numpy.array([[3, 4, 1], [4, 6, 0], [1, 2, 5]])
distances = numpy.sqrt(((points - p)**2).sum(axis=1))
print(distances)
sorted_indexes = distances.argsort()
print(sorted_indexes)
print(points[sorted_indexes][0])


# Millions of points
# sigma * np.random.randn(...) + mu
points = 2 * numpy.random.randn(10000000, 3) + 3
# print(points)
print(str(datetime.now()))
distances = numpy.sqrt(((points - p)**2).sum(axis=1))
# print(distances)
sorted_indexes = distances.argsort()
# print(sorted_indexes)
print(points[sorted_indexes][0])
print(str(datetime.now()))

# with 1 billion points it is too memory intensive
# solutions:
# - mongodb $near
# - map reduce

# 10.000.000 points = 240MB
# 100.000.000 points = 2400MB = 2.4GB
numpy.random.seed(1000)
points = 2 * numpy.random.randn(100000000, 3) + 3
print(points)
print(getsizeof(points)/1000/1000, "MB")

min_distance = numpy.inf
min_point = None

init = 0
step = 10000000
print(str(datetime.now()))
for i in range(10):
    print("Iteration", i, "sample from", init, "to", (init+step))
    points_sample = points[init:init+step]
    init += step
    distances = numpy.sqrt(((points_sample - p)**2).sum(axis=1))
    sorted_indexes = distances.argsort()
    distance = distances[sorted_indexes][0]
    point = points_sample[sorted_indexes][0]
    print("distance", distance)
    print("point", point)
    if distance < min_distance:
        min_distance = distance
        min_point = point

print(str(datetime.now()))
print("min_distance", min_distance)
print("min_point", min_point)
