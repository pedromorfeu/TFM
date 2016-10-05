data(AirPassengers)
class(AirPassengers)
start(AirPassengers)
end(AirPassengers)
frequency(AirPassengers)

summary(AirPassengers)
str(AirPassengers)

plot(AirPassengers)
abline(reg = lm(AirPassengers~time(AirPassengers)))

cycle(AirPassengers)
plot(aggregate(AirPassengers, FUN = mean))

boxplot(AirPassengers~cycle(AirPassengers))



