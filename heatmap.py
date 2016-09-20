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


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# sns.plt.figure()
ax1.set_title("covariance_X")
sns.heatmap(a, annot=False, fmt=".2f", robust=True, ax=ax1)

# sns.plt.figure()
ax2.set_title("covariance_inverse_X")
sns.heatmap(a, annot=False, fmt=".2f", robust=True, ax=ax2)

sns.plt.show()