import numpy
from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

print( "Loading ... " )
mnist = fetch_mldata( 'MNIST original' ) #, data_home='data' )

# print( "Transforming  ... " )

X = mnist.data
Y = mnist.target

print("X")
print(X.shape)
print(X[:5,:])

print("Applying PCA...")
pca = PCA(n_components=10)
new_X = pca.fit_transform( X )

print("new_X")
print(new_X.shape)
print(new_X[:5,:])

# X = pca.inverse_transform(new_X)
# print("X")
# print(X.shape)
# print(X[:5,:])


# print("pca.components_", pca.components_)
print("pca.explained_variance_ratio_", pca.explained_variance_ratio_)
# print("pca.mean_", pca.mean_)
print("pca.n_components_", pca.n_components_)
print("pca.noise_variance_", pca.noise_variance_)

print("covariance", pca.get_covariance())