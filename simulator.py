from sklearn.decomposition import PCA
from sklearn import covariance

from pandas import read_csv

from datetime import datetime, timedelta

import seaborn as sns

import numpy as np

import time

# import locale
#
#
# print(locale.getdefaultlocale())
# locale.setlocale(locale.LC_TIME, "spanish")
#
# print(datetime(2016, 10, 20).strftime("%a, %d %b %Y %H:%M:%S"))
# print(datetime(2016, 4, 20).strftime("%a, %d %b %Y %H:%M:%S"))
# print(time.strftime("%a, %d %b %Y %H:%M:%S"))
#
# print(datetime.strptime("sep.", "%b"))
#
# exit()
#

def parse_date(date_string):
    # 06-oct-2015 21:57:03
    return datetime.strptime(date_string, "%d-%b-%Y %H:%M:%S")



# data = read_csv("ip_sample.txt", sep="\s+\t", header=1, engine="python", parse_dates=[0], infer_datetime_format=True, date_parser=parse_date)
data = read_csv("ip.txt", sep="\s+\t", header=2, engine="python", na_values="")
print(data.values[5:,:])

# ignore first field (date) by now
X = data.values[:,1:]

print("X.dtype", X.dtype)
print("X.shape", X.shape)
print("X", X[:5,:], sep="\n")

pca = PCA(n_components=5)
new_X = pca.fit_transform(X)

print("new_X.shape", new_X.shape)
print("new_X", new_X[:5,], sep="\n")

print("pca.explained_variance_ratio_", pca.explained_variance_ratio_)

inverse_X = pca.inverse_transform(new_X)

print("inverse_X.shape", inverse_X.shape)
print("inverse_X", inverse_X[:5,], sep="\n")

covariance_X = covariance.empirical_covariance(X)
print("covariance_X.shape", covariance_X.shape)
# print("covariance_X", covariance_X, sep="\n")
covariance_X = covariance_X.astype(float)

covariance_inverse_X = covariance.empirical_covariance(inverse_X)
print("covariance_inverse_X.shape", covariance_inverse_X.shape)
# print("covariance_inverse_X", covariance_inverse_X, sep="\n")
covariance_inverse_X = covariance_inverse_X.astype(float)


f1 = sns.plt.figure()
sns.plt.title("covariance_X")
c1 = sns.heatmap(covariance_X)
# sns.plt.text(0, 0, covariance_X[0,0])

f2 = sns.plt.figure()
sns.plt.title("covariance_inverse_X")
c2 = sns.heatmap(covariance_inverse_X)
# sns.plt.text(0, 0, covariance_inverse_X[0,0])

sns.plt.show()
