import numpy as np
import statsmodels.api as sm
nobs = 10000
np.random.seed(987689)
x = np.random.randn(nobs, 3)
x = sm.add_constant(x, prepend=True)
y = x.sum(1) + np.random.randn(nobs)

xf = 0.5 * np.ones((2,4))

model = sm.OLS(y, x)
results = model.fit()

print("results.predict", results.predict(xf))
print("results.model.predict", results.model.predict(results.params, xf))

results._results.model.endog = None
results._results.model.wendog = None
results._results.model.exog = None
results._results.model.wexog = None
results.model.data._orig_endog = None
results.model.data._orig_exog = None
results.model.data.endog = None
results.model.data.exog = None
#results.model._data = None

results._results.model.fittedvalues = None
results._results.model.resid = None
results._results.model.wresid = None
#extra
results._results.model.pinv_wexog = None

# import pickle
# fh = open('try_shrink2.pickle', 'w')
# pickle.dump(results._results, fh)  #pickling wrapper doesn't work
# fh.close()
# fh = open('try_shrink2.pickle', 'r')
# results2 = pickle.load(fh)
# fh.close()

results2 = results._results
print("results2.predict", results2.predict(xf))
print("results2.model.predict", results2.model.predict(results.params, xf))