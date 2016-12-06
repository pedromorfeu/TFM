import numpy as np
import matplotlib.pylab as plt
import pandas as pd

N_COLUMNS = 8
MAX_POINTS = 1000


def generate_sinoidal(fR=0.05, fPi=10, fun=np.sin, fMultiply=1, fAdd=0, round=2):
    x = np.linspace(-np.pi, fPi*np.pi, MAX_POINTS)
    sin0 = fMultiply * fun(x) + fAdd
    std0 = np.std(sin0)
    noise = np.random.normal(0, std0*fR, MAX_POINTS)
    sin = np.add(sin0, noise)
    return np.round(sin, round)


def generate_gaussian(mean=2, std=3, fR=0.5, round=0):
    return np.round(np.random.normal(mean, std*fR, MAX_POINTS), round)


generated_data = np.zeros((MAX_POINTS, N_COLUMNS))

# generated_data[:, 0] = generate_gaussian(2, 3, 0.4)
# generated_data[:, 1] = generate_gaussian(-5, 1.5, 0.1)
# generated_data[:, 2] = generate_gaussian(5, 1, 0.2)
# generated_data[:, 3] = generate_gaussian(-8, 3, 0.09, round=2)
# generated_data[:, 4] = generate_gaussian(-2, 2.1, 0.5)
# generated_data[:, 5] = generate_gaussian(3, 1, 0.1)
generated_data[:, 0] = generate_sinoidal(fR=0.05, fPi=5, fun=np.sin, round=0)
generated_data[:, 1] = generate_sinoidal(fR=0.01, fPi=5, fun=np.sin, fMultiply=2, fAdd=3)
generated_data[:, 2] = generate_sinoidal(fR=0.02, fun=np.cos)
generated_data[:, 3] = generate_sinoidal(fR=0.09, fPi=15, fun=np.cos, fMultiply=4, fAdd=-1)
generated_data[:, 4] = generate_sinoidal(fR=0.05, fPi=5, fun=np.sin, round=1)
generated_data[:, 5] = generate_sinoidal(fR=0.01, fPi=4, fun=np.cos, fMultiply=5)
generated_data[:, 6] = generate_sinoidal(fR=0.02, fun=np.cos)
generated_data[:, 7] = generate_sinoidal(fR=0.09, fPi=15, fun=np.cos, fAdd=-1)

# plt.plot(generated_data)
# plt.show()

rng = pd.date_range('10/06/2015', periods=MAX_POINTS, freq='10s')
df = pd.DataFrame(generated_data, index=rng, columns=[("col"+str(x+1)) for x in range(N_COLUMNS)])
df.head()

plt.plot(df)
plt.show()

df.to_csv("ip_gen.txt", index_label="Tiempoinicio")
