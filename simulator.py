from sklearn import covariance
from sklearn.decomposition import PCA
from sklearn.mixture import GMM

from pandas import read_csv
from datetime import datetime, time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import locale
import re

from util import parse_dates, parse_date, plot_covariance_heatmaps

N_COMPONENTS = 5
NEW_DATA_SIZE = 1000000


start_time = time.time()

data = read_csv("ip.txt", sep="\s+\t", engine="python", parse_dates=[0], date_parser=parse_dates,
                skip_blank_lines=True, na_values="")

print("--- %s seconds ---" % (time.time() - start_time))
print("--- %.2f minutes ---" % ((time.time() - start_time)/60))

# print(data)
print(data.values.shape)
print(data.values[:5, :])


# ignore first field (date) by now
X = data.values[:, 1:]

X = X.astype(float)

# plt.plot(X[:2], "o")
# # plt.boxplot(X[:2, :])
# plt.legend(data.columns[1:3])
# plt.ylim(50)
#
# plt.show()
# exit()

print("X.dtype", X.dtype)
print("X.shape", X.shape)
print("X", X[:5, :], sep="\n")

pca = PCA(n_components=N_COMPONENTS)
new_X = pca.fit_transform(X)

print("new_X.shape", new_X.shape)
print("new_X", new_X[:5, ], sep="\n")
print("...")
print(new_X[-5:, ], sep="\n")

# *** Generate data ***

# rs = np.random.RandomState(1)
# gmm = GMM()
# gmm.fit(new_X)
# generated_X = gmm.sample(NEW_DATA_SIZE, random_state=rs)


mus = np.mean(new_X, axis=0)
sigmas = np.std(new_X, axis=0)


generated_X = np.zeros((NEW_DATA_SIZE, N_COMPONENTS))
for i in range(N_COMPONENTS):
    # calculate normal distribution by component and store it in column i
    generated_X[:, i] = np.random.normal(mus[i], sigmas[i], NEW_DATA_SIZE)


print("generated_X.shape", generated_X.shape)
print("generated_X", generated_X[:5, ], sep="\n")
print("...")
print(generated_X[-5:, ], sep="\n")

# new_X =  np.vstack((new_X, generated_X))


inverse_X = pca.inverse_transform(generated_X)

print("inverse_X.shape", inverse_X.shape)
print("inverse_X", inverse_X[:5, ], sep="\n")

correlation_X = np.corrcoef(np.transpose(X))
print("correlation_X.shape", correlation_X.shape)

correlation_inverse_X = np.corrcoef(np.transpose(inverse_X))
print("correlation_inverse_X.shape", correlation_inverse_X.shape)


plot_covariance_heatmaps(correlation_X, correlation_inverse_X, annotation=False)