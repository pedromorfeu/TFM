import numpy as np
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


# np
def dist(x,y):
    return np.sqrt(np.sum((x-y)**2))
print(dist(np.array(p), np.array(q1)))
print(dist(np.array(p), np.array(q2)))

np.array([-4.70324186, -2.05215551,  2.56444493]) - np.array([-3.2683099767221981])
(np.array([-4.70324186, -2.05215551,  2.56444493]) - np.array([-3.2683099767221981])) ** 2
np.sqrt( (np.array([-4.70324186, -2.05215551,  2.56444493]) - np.array([-3.2683099767221981])) ** 2 )


# Nearest points
points = np.array([[3, 4, 1], [4, 6, 0], [1, 2, 5]])
distances = np.sqrt(((points - p)**2).sum(axis=1))
print(distances)
sorted_indexes = distances.argsort()
print(sorted_indexes)
print(points[sorted_indexes][0])


# Millions of points
# sigma * np.random.randn(...) + mu
print(str(datetime.now()), "millions start")

sigmas = [1, 1, 1, 1, 1]
means = [2, 3, 4, 5, 6]

MAX_POINTS = 100*1000*1000
points = np.zeros((MAX_POINTS, 5))
for i in range(5):
    # calculate normal distribution by component and store it in column i
    points[:, i] = np.random.normal(means[i], sigmas[i], MAX_POINTS)

# points = 2 * np.random.randn(10000000, 5) + 3
print(str(datetime.now()), "millions generated")
x1 = [2, 3, 4, 5, 6]
distances = np.sqrt(((points - x1)**2).sum(axis=1))
sorted_indexes = distances.argsort()
print(str(datetime.now()), points[sorted_indexes][0])
x2 = [2, 3, 4, 5, 6]
distances = np.sqrt(((points - x2)**2).sum(axis=1))
sorted_indexes = distances.argsort()
print(str(datetime.now()), points[sorted_indexes][0])
print(str(datetime.now()), "millions end")

exit()

# with 1 billion points it is too memory intensive
# solutions:
# - mongodb $near
# - map reduce

# 10.000.000 points = 240MB = <1 min
# 100.000.000 points = 2400MB = 2.4GB =
np.random.seed(1000)
points = 2 * np.random.randn(100000000, 3) + 3
print(points)
print(getsizeof(points)/1000/1000, "MB")

min_distance = np.inf
min_point = None

init = 0
step = 10000000
print(str(datetime.now()))
for i in range(10):
    print("Iteration", i, "sample from", init, "to", (init+step))
    points_sample = points[init:init+step]
    init += step
    distances = np.sqrt(((points_sample - p)**2).sum(axis=1))
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
