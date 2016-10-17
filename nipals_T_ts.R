#install.packages("zoo")
#install.packages("xts")
#install.packages("tseries")
#install.packages("forecast")
library(zoo)
library(xts)
library(tseries)
library(forecast)
#detach("package:xts", unload = T)
#detach("package:zoo", unload = T)

data <- read.csv2("generated/nipals_T_ts.csv", sep = "\t", header = T, stringsAsFactors = F, strip.white = T, blank.lines.skip=T,
                  colClasses = c("character", rep("numeric", 5)), dec=".")
head(data)
tail(data)
str(data)
summary(data)

# Time series
Sys.setlocale("LC_TIME", "spanish")

data$time <- sub("(\\d+-)(\\w+)(-\\d+\\s\\d+:\\d+:\\d+)", "\\1\\2.\\3", data$time)
data$time <- as.POSIXct(data$time, format="%d-%b-%Y %H:%M:%S")
head(data)
str(data)

dataTs <- xts(data[2:6], order.by = data$time)

head(dataTs)
class(dataTs)
start(dataTs)
end(dataTs)
frequency(dataTs)

summary(dataTs$X0)
str(dataTs$X0)

myts <- dataTs["2015-10-06"]$X0
summary(myts)
str(myts)

plot(myts)
abline(reg = lm(myts~time(myts)))

periodicity(myts)
cycle(myts)
endpoints(myts, on = "seconds", k = 10)
to.period(myts, period = "seconds", k=10)
myts_period <- to.period(myts, period = "seconds", k=10)
head(myts)
head(myts_period)

?approx
?spline

period.apply(myts$X0, INDEX = endpoints(myts$X0, on = "seconds", k = 10), FUN=mean)

# Correlation plots. Following are the ACF plots for the series
# the decay of ACF chart is very slow, which means that the population is not stationary
acf(myts)
pacf(myts)

# fit an ARIMA model of order P, D, Q
fit <- arima(myts, order=c(p, d, q))

#The null-hypothesis for an ADF test is that the data are 
#non-stationary. So large p-values are indicative of 
#non-stationarity, and small p-values suggest stationarity. 
#Using the usual 5% threshold, differencing is required if 
#the p-value is greater than 0.05.
adf.test(myts)

diff <- diff(myts, differences=1)
diff[is.na(diff)] <- 0
head(diff)
tail(diff)
plot(myts)
plot(diff)

adf.test(diff, alternative="stationary", k=0)

acf(diff)
pacf(diff)


#Another popular unit root test is the 
#Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test. 
#This reverses the hypotheses, so the null-hypothesis is 
#that the data are stationary. In this case, small p-values 
#(e.g., less than 0.05) suggest that differencing is required.
kpss.test(diff)

#A useful R function is ndiffs() which uses these tests to 
#determine the appropriate number of first differences 
#required for a non-seasonal time series.
#The following code can be used to find how to make a 
#seasonal series stationary. The resulting series stored as 
#xstar has been differenced appropriately.
#seasonal
ns <- ndiffs(myts)
if(ns > 0) {
  xstar <- diff(myts, lag=frequency(myts), differences=ns)
} else {
  xstar <- myts
}
#non-seasonal
nd <- ndiffs(myts)
if(nd > 0) {
  xstar <- diff(myts, differences=nd)
}
head(xstar)
tail(xstar)

par(mfrow=c(1,2))
#there are three spikes in the ACF and then no significant spikes thereafter
Acf(xstar, main="")
#In the PACF, there are three spikes decreasing with the lag, and then no significant spikes thereafter
Pacf(xstar, main="")

#The following R code was used to automatically select a model.
fit <- auto.arima(myts, seasonal=FALSE)
fit 

#plot 1 by 1
par(mfrow=c(1,1))
#forecast: h - Number of periods for forecasting
#forecast: include - include X observations of the original series in your plot
plot(forecast(fit,h=1000), include=800)

#The ACF plot of the residuals from the ARIMA(3,1,1) model shows 
#all correlations within the threshold limits indicating that 
#the residuals are behaving like white noise. 
#A portmanteau test returns a large p-value, also suggesting 
#the residuals are white noise.
fit1 <- Arima(xstar, order=c(2,0,0))
Acf(residuals(fit1))
Box.test(residuals(fit1), lag=24, fitdf=4, type="Ljung")

plot(forecast(fit1,h=100),include=80)
