import numpy as np
rng = np.random.RandomState(42)

n_samples_train, n_features = 40, 10
n_samples_test = 20
# X_train = rng.randn(n_samples_train, n_features)
# X_test = rng.randn(n_samples_test, n_features)
X_train = rng.randint(10, size=(n_samples_train, n_features))
X_test = rng.randint(10, size=(n_samples_test, n_features))

from sklearn.decomposition import PCA
pca = PCA(whiten=True)

pca.fit(X_train)

X_train_mean = X_train.mean(0)
X_train_centered = X_train - X_train_mean
U, S, VT = np.linalg.svd(X_train_centered, full_matrices=False)
components = VT / S[:, np.newaxis] * np.sqrt(n_samples_train)

print(X_train)
print(X_train_mean)
print(X_train_centered)
print(VT)
print(components)
