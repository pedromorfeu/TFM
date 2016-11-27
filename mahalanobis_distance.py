import numpy as np
from scipy.spatial.distance import mahalanobis, cdist

x = [1, 2, 3]
q1 = [3, 4, 1]
q2 = [4, 6, 0]
q3 = [[4, 6, 0], [3, 2, 2]]


cdist([x], [q1], 'euclidean')
cdist([x], [q2], 'euclidean')


cdist([1,2,3], [[2,3,4],[1,2,1],[3,4,6],[2,3,7],[6,4,3]], 'mahalanobis', VI=None)
cdist([x], [q2], 'mahalanobis', VI=None)
