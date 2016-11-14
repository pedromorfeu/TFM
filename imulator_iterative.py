[1mdiff --git a/simulator_iterative.py b/simulator_iterative.py[m
[1mindex 9a60944..0344c1c 100644[m
[1m--- a/simulator_iterative.py[m
[1m+++ b/simulator_iterative.py[m
[36m@@ -204,7 +204,7 @@[m [mfor i in range(N_COMPONENTS):[m
     timeseries_samples.append(timeseries_sample)[m
     # timeseries_sample = timeseries_sample.asfreq(TS_FREQUENCY, method="ffill")[m
     print_timeseries("timeseries_sample", timeseries_sample)[m
[31m-    stationary = test_stationarity(timeseries_sample, _plot=False, _critical="5%")[m
[32m+[m[32m    stationary, test_value = test_stationarity(timeseries_sample, _plot=False, _critical="5%")[m
 [m
     # not stationary -> must be stabilized[m
 [m
[36m@@ -228,13 +228,13 @@[m [mfor i in range(N_COMPONENTS):[m
     ts_log_moving_avg_diff.head(5)[m
     ts_log_moving_avg_diff.dropna(inplace=True)[m
     ts_log_moving_avg_diff.head()[m
[31m-    stationary = test_stationarity(ts_log_moving_avg_diff, _plot=False, _critical="5%")[m
[32m+[m[32m    stationary, test_value_moving_avg_diff = test_stationarity(ts_log_moving_avg_diff, _plot=False, _critical="5%")[m
 [m
     expwighted_avg = ts_log.ewm(halflife=20).mean()[m
     # plt.plot(ts_log)[m
     # plt.plot(expwighted_avg, color='red')[m
     ts_log_ewma_diff = ts_log - expwighted_avg[m
[31m-    stationary = test_stationarity(ts_log_ewma_diff, _plot=False, _critical="5%")[m
[32m+[m[32m    stationary, test_value_ewma_diff = test_stationarity(ts_log_ewma_diff, _plot=False, _critical="5%")[m
 [m
     # Differencing[m
     print("Differencing the time series...")[m
[36m@@ -246,12 +246,18 @@[m [mfor i in range(N_COMPONENTS):[m
     print("Missing values:", not np.all(np.isfinite(ts_log_diff)))[m
     print_timeseries("ts_log_diff", ts_log_diff)[m
     # This appears to have reduced trend considerably. Lets verify using our plots:[m
[31m-    stationary = test_stationarity(ts_log_diff, _plot=False, _critical="5%")[m
[32m+[m[32m    stationary, test_value_diff = test_stationarity(ts_log_diff, _plot=False, _critical="5%")[m
 [m
     if not stationary:[m
         # TODO: try other methods to make the timeseries stationary[m
         raise ValueError("Timeseries is not stationary after differencing.")[m
 [m
[32m+[m[32m    # Minimum value for integration part (d):[m
[32m+[m[32m    # If differenciated series is more stationary, then use i=1[m
[32m+[m[32m    min_i = 0[m
[32m+[m[32m    if test_value_diff < test_value:[m
[32m+[m[32m        min_i = 1[m
[32m+[m
     # Forecasting[m
     # plot_acf_pacf(ts_log_diff)[m
     # print(str(datetime.now()), "component", i,"Calculating AR and MA orders...")[m
[36m@@ -269,7 +275,7 @@[m [mfor i in range(N_COMPONENTS):[m
     # d = 0[m
 [m
     # Calculate best order (order with minimum error)[m
[31m-    (min_rmse, p, d, q) = arima_order_select(ts_log)[m
[32m+[m[32m    (min_rmse, p, d, q) = arima_order_select(ts_log, min_i=min_i)[m
 [m
     print("Creating model (p,d,q)=(%i,%i,%i)" % (p, d, q))[m
     model = SARIMAX(ts_log, order=(p, d, q))[m
[36m@@ -281,11 +287,12 @@[m [mfor i in range(N_COMPONENTS):[m
     predictions_ARIMA = results_ARIMA.predict(start=ts_log.shape[0], end=ts_log.shape[0])[m
     print(str(datetime.now()), "Predicted")[m
 [m
[31m-    print(ts_log.tail(5))[m
[31m-    print(predictions_ARIMA.tail(5))[m
[31m-    out_of_sample = predictions_ARIMA[m
[31m-    ts_log_predicted = ts_log.append(predictions_ARIMA)[m
[31m-    print(predictions_ARIMA.tail(5))[m
[32m+[m[32m    # print(ts_log.tail(5))[m
[32m+[m[32m    # print(predictions_ARIMA.tail(5))[m
[32m+[m[32m    # out_of_sample = predictions_ARIMA[m
[32m+[m[32m    # ts_log_predicted = ts_log.append(predictions_ARIMA)[m
[32m+[m[32m    # print(predictions_ARIMA.tail(5))[m
[32m+[m[32m    ts_log_predicted = results_ARIMA.fittedvalues[m
 [m
     models.append((results_ARIMA, ts_log_predicted, min_rmse))[m
 [m
[36m@@ -313,7 +320,9 @@[m [mfor i in range(NEW_DATA_SIZE):[m
         ts_log_predicted1 = ts_log_predicted.append(predictions_ARIMA1)[m
         models_iterative[j] = (results_ARIMA1, ts_log_predicted1, min_rmse)[m
         # add random error from series standard deviation and mean 0[m
[31m-        add_error = 0 + min_rmse * np.random.randn()[m
[32m+[m[32m        # add_error = 0 + min_rmse * np.random.randn()[m
[32m+[m[32m        add_error = 0[m
[32m+[m[32m        # print("Adding error using RMSE", min_rmse, ":", add_error)[m
         preds[j] = predictions_ARIMA1 + add_error[m
         j += 1[m
 [m
[36m@@ -322,7 +331,8 @@[m [mfor i in range(NEW_DATA_SIZE):[m
     distances = np.sqrt(((generated_gaussian_copy - preds) ** 2).sum(axis=1))[m
     sorted_indexes = distances.argsort()[m
     min_index = sorted_indexes[0][m
[31m-    preds_transformed = generated_gaussian_copy[min_index][m
[32m+[m[32m    # preds_transformed = generated_gaussian_copy[min_index][m
[32m+[m[32m    preds_transformed = preds[m
     # hypotesis: repetitions make results worse[m
     # generated_gaussian_copy = np.delete(generated_gaussian_copy, min_index, 0)[m
     generated_X[i] = preds_transformed[m
[36m@@ -347,7 +357,7 @@[m [msave_matrix("inverse_X.csv", XX, data.columns)[m
 [m
 [m
 for i in range(N_COMPONENTS):[m
[31m-    gaussian_ts = pd.Series(generated_X[:, i], index=pd.date_range("2015-10-06 23:59:50", periods=NEW_DATA_SIZE, freq=TS_FREQUENCY))[m
[32m+[m[32m    gaussian_ts = pd.Series(generated_gaussian[:NEW_DATA_SIZE, i], index=pd.date_range("2015-10-06 23:59:50", periods=NEW_DATA_SIZE, freq=TS_FREQUENCY))[m
     plt.plot(gaussian_ts, "o", color="gray")[m
     plt.plot(models_iterative[i][1])[m
     plt.plot(timeseries_samples[i])[m
[1mdiff --git a/util/__init__.py b/util/__init__.py[m
[1mindex 7c44ab9..dff309c 100644[m
[1m--- a/util/__init__.py[m
[1m+++ b/util/__init__.py[m
[36m@@ -124,7 +124,7 @@[m [mdef test_stationarity(_timeseries, _plot=False, _critical="5%"):[m
         plt.title('Rolling Mean & Standard Deviation')[m
         plt.show(block=True)[m
 [m
[31m-    return stationary[m
[32m+[m[32m    return stationary, test_value[m
 [m
 [m
 def plot_acf_pacf(_timeseries, spikes_plot=True, lags=10):[m
[36m@@ -166,11 +166,11 @@[m [mdef plot_acf_pacf(_timeseries, spikes_plot=True, lags=10):[m
     return lag_acf, lag_pacf[m
 [m
 [m
[31m-def arima_order_select(_timeseries, max_ar=4, max_i=2, max_ma=4):[m
[32m+[m[32mdef arima_order_select(_timeseries, max_ar=4, max_i=2, max_ma=4, min_i=0):[m
     min_rmse, min_p, min_d, min_q = np.inf, 0, 0, 0[m
 [m
     for p in range(max_ar):[m
[31m-        for d in range(max_i):[m
[32m+[m[32m        for d in range(min_i, max_i):[m
             for q in range(max_ma):[m
                 try:[m
                     print("Creating model (p,d,q)=(%i,%i,%i)" % (p, d, q))[m
