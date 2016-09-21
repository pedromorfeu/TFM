from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn import covariance

import seaborn as sns
import matplotlib.pyplot as plt

# import some data to play with
iris = datasets.load_iris()
X = iris.data
Y = iris.target


print("X.shape", X.shape)
print("X", X[:5,:])

# print(Y.shape)
# print(Y[:5])

pca = PCA(n_components=2)
new_X = pca.fit_transform(X)

print("new_X.shape", new_X.shape)
print("new_X", new_X[:5,], sep="\n")

print("pca.explained_variance_ratio_", pca.explained_variance_ratio_)
print("pca.components_", pca.components_)
# print("covariance", pca.get_covariance()[:5,])

inverse_X = pca.inverse_transform(new_X)

print("inverse_X.shape", inverse_X.shape)
print("inverse_X", inverse_X[:5,], sep="\n")

covariance_X = covariance.empirical_covariance(X)
print("covariance_X", covariance_X, sep="\n")

covariance_inverse_X = covariance.empirical_covariance(inverse_X)
print("covariance_inverse_X", covariance_inverse_X, sep="\n")


# f1 = sns.plt.figure()
# sns.plt.title("covariance_X")
# c1 = sns.heatmap(covariance_X)
# # sns.plt.text(0, 0, covariance_X[0,0])
#
# f2 = sns.plt.figure()
# sns.plt.title("covariance_inverse_X")
# c2 = sns.heatmap(covariance_inverse_X)
# # sns.plt.text(0, 0, covariance_inverse_X[0,0])
#
# sns.plt.show()
