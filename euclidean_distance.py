p = (1, 2, 3)
q1 = (3, 4, 1)
q2 = (4, 6, 0)

# SciPy
from scipy.spatial import distance
print(distance.euclidean(p, q1))
print(distance.euclidean(p, q2))

# Numpy
import numpy
def dist(x,y):
    return numpy.sqrt(numpy.sum((x-y)**2))
print(dist(numpy.array(p), numpy.array(q1)))
print(dist(numpy.array(p), numpy.array(q2)))



# Nearest points
points = numpy.array([[3,4,1],[4,6,0],[1,2,5]])
distances = numpy.sqrt( ((points - p)**2).sum(axis=1) )
print(distances)
sorted_indexes = distances.argsort()
print(sorted_indexes)
print(points[sorted_indexes][0])


# Millions of points
# sigma * np.random.randn(...) + mu
points = 2 * numpy.random.randn(1000000, 3) + 3
print(points)
distances = numpy.sqrt( ((points - p)**2).sum(axis=1) )
print(distances)
sorted_indexes = distances.argsort()
print(sorted_indexes)
print(points[sorted_indexes][0])
