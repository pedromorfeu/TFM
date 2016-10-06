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

data <- read.csv2("ip.txt", sep = "\t", header = T, stringsAsFactors = F, strip.white = T, blank.lines.skip=T,
                  colClasses = c("character", rep("numeric", 14)), dec=".")
data <- data[!is.na(data$APHu),]
# head(data)
# str(data)
# summary(data)

data[is.na(data),]
head(data)

# Time series
Sys.setlocale("LC_TIME", "spanish")

data$Tiempoinicio <- sub("(\\d+-)(\\w+)(-\\d+\\s\\d+:\\d+:\\d+)", "\\1\\2.\\3", data$Tiempoinicio)
data$Tiempoinicio <- as.POSIXct(data$Tiempoinicio, format="%d-%b-%Y %H:%M:%S")
head(data)

dataTs <- xts(data[2:15], order.by = data$Tiempoinicio)

head(dataTs)
class(dataTs)
start(dataTs)
end(dataTs)
frequency(dataTs)

summary(dataTs$APHu)
str(dataTs$APHu)

plot(dataTs$APHu)
abline(reg = lm(dataTs$APHu~time(dataTs$APHu)))

#Correlation plots. Following are the ACF plots for the series
#the decay of ACF chart is very slow, which means that the population is not stationary
acf(dataTs$APHu)

cycle(dataTs)
plot(aggregate(dataTs, FUN = mean, by = dataTs$APHu))

boxplot(dataTs$APHu~cycle(dataTs$APHu))

# fit an ARIMA model of order P, D, Q
fit <- arima(myts, order=c(p, d, q))

#The null-hypothesis for an ADF test is that the data are 
#non-stationary. So large p-values are indicative of 
#non-stationarity, and small p-values suggest stationarity. 
#Using the usual 5% threshold, differencing is required if 
#the p-value is greater than 0.05.
adf.test(dataTs$APHu)             

#Another popular unit root test is the 
#Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test. 
#This reverses the hypotheses, so the null-hypothesis is 
#that the data are stationary. In this case, small p-values 
#(e.g., less than 0.05) suggest that differencing is required.
kpss.test(dataTs$APHu)

#A useful R function is ndiffs() which uses these tests to 
#determine the appropriate number of first differences 
#required for a non-seasonal time series.
#The following code can be used to find how to make a 
#seasonal series stationary. The resulting series stored as 
#xstar has been differenced appropriately.
#seasonal
ns <- ndiffs(dataTs$APHu)
if(ns > 0) {
  xstar <- diff(dataTs$APHu, lag=frequency(dataTs$APHu), differences=ns)
} else {
  xstar <- dataTs$APHu
}
#non-seasonal
nd <- ndiffs(dataTs$APHu)
if(nd > 0) {
  xstar <- diff(dataTs$APHu, differences=nd)
}

par(mfrow=c(1,2))
#there are three spikes in the ACF and then no significant spikes thereafter
Acf(xstar, main="")
#In the PACF, there are three spikes decreasing with the lag, and then no significant spikes thereafter
Pacf(xstar, main="")

#The following R code was used to automatically select a model.
fit <- auto.arima(xstar, seasonal=FALSE)
fit 

#plot 1 by 1
par(mfrow=c(1,1))
#forecast: h - Number of periods for forecasting
#forecast: include - include X observations of the original series in your plot
plot(forecast(fit,h=10),include=80)


#The ACF plot of the residuals from the ARIMA(3,1,1) model shows 
#all correlations within the threshold limits indicating that 
#the residuals are behaving like white noise. 
#A portmanteau test returns a large p-value, also suggesting 
#the residuals are white noise.
fit1 <- Arima(xstar, order=c(2,0,0))
Acf(residuals(fit1))
Box.test(residuals(fit1), lag=24, fitdf=4, type="Ljung")



summary(data[, seq(2,15)])
var(data[, seq(2,3)])
var(data[, c("Svo","ACPx")])

summary(data$Svo)
mean(data$Svo)
var(data$Svo)
var(data$ACPx)
sd(data$Svo)
cov(data[, seq(2,5)], data[, seq(2,5)])
diag(data)

plot(data)

#create vectors -- these will be our columns
a <- c(1,2,3,4,5,6)
b <- c(2,3,5,6,1,9)
c <- c(3,5,5,5,10,8)
d <- c(10,20,30,40,50,55)
e <- c(7,8,9,4,6,10)

#create matrix from vectors
M <- cbind(a,b,c,d,e)
M

cov(M)

a <- c(7,4,6,8,8,7,5,9,7,8)
b <- c(4,1,3,6,5,2,3,5,4,2)
c <- c(3,8,5,1,7,9,3,8,5,2)
M <- cbind(a,b,c)
M

cov(M)

plot(data$APHu)
boxplot(data$APHu)
