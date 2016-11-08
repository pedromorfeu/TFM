import numpy
from scipy.spatial import distance
from datetime import datetime
import json


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


points = 2 * numpy.random.randn(10, 3) + 3
docs = []
for pi in points:
    x = {"type": "Point", "coordinates": pi.toarray()}
    docs.append(x)
print(docs)