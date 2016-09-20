from sklearn.decomposition import PCA
from sklearn import covariance

from pandas import read_csv
from datetime import datetime, time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import locale
import re

from util import parse_dates, parse_date, plot_covariance_heatmaps


start_time = time.time()

data = read_csv("ip.txt", sep="\s+\t", engine="python", parse_dates=[0], date_parser=parse_dates,
                skip_blank_lines=True, na_values="")

print("--- %s seconds ---" % (time.time() - start_time))
print("--- %.2f minutes ---" % ((time.time() - start_time)/60))

# print(data)
print(data.values.shape)
print(data.values[:5, :])


# ignore first field (date) by now
X = data.values[:,1:]

print("X.dtype", X.dtype)
print("X.shape", X.shape)
print("X", X[:5, :], sep="\n")

pca = PCA(n_components=2)
new_X = pca.fit_transform(X)

print("new_X.shape", new_X.shape)
print("new_X", new_X[:5, ], sep="\n")

print("pca.explained_variance_ratio_", pca.explained_variance_ratio_)

inverse_X = pca.inverse_transform(new_X)

print("inverse_X.shape", inverse_X.shape)
print("inverse_X", inverse_X[:5, ], sep="\n")

covariance_X = covariance.empirical_covariance(X)
print("covariance_X.shape", covariance_X.shape)
# print("covariance_X", covariance_X, sep="\n")
covariance_X = covariance_X.astype(float)


covariance_inverse_X = covariance.empirical_covariance(inverse_X)
print("covariance_inverse_X.shape", covariance_inverse_X.shape)
# print("covariance_inverse_X", covariance_inverse_X, sep="\n")
covariance_inverse_X = covariance_inverse_X.astype(float)


plot_covariance_heatmaps(covariance_X, covariance_inverse_X)