import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns


a = np.random.random((14, 14))
print(a)

# plt.figure()
# plt.imshow(a, cmap='Reds', interpolation='nearest')
#
# plt.figure()
# plt.imshow(a, cmap='Blues', interpolation='nearest')
#
# plt.show()


sns.plt.figure()
sns.plt.title("covariance_X")
sns.heatmap(a, annot=True, fmt=".2f")

sns.plt.figure()
sns.plt.title("covariance_inverse_X")
sns.heatmap(a, annot=True, fmt=".2f")

sns.plt.show()