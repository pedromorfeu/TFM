# install.packages("Hotelling")
library("Hotelling")
#detach("package:Hotelling", unload=TRUE)

# install.packages("biotools")
# library("biotools")
#detach("package:biotools", unload=TRUE)

### GAUSSIAN
data <- read.csv2("ip.txt", sep = "\t", header = T, stringsAsFactors = F, strip.white = T, blank.lines.skip=T,
                  colClasses = c("character", rep("numeric", 14)), dec=".")
data <- data[!is.na(data$APHu),]
# filter by date
data <- (data[startsWith(data$Tiempoinicio, "06-oct-2015"), ])
data <- data[, seq(2,15)]
nrow(data)
ncol(data)
mean(data$APHu)
sd(data$APHu)
plot(data$APHu)

head(data)

# check the variables with zero variance and give them white noise
# otherwise the test will fail due to inability to invert
cov(data)
# ZSx and H7x have zero variance
data$ZSx <- mean(data$ZSx) + 0.001 * rnorm(nrow(data))
data$H7x <- mean(data$H7x) + 0.001 * rnorm(nrow(data))
cov(data)
mean(data$APHu)
sd(data$APHu)

head(data)

inverted_data_gaussian <- read.csv2("generated/inverse_X_gaussian.csv", sep = "\t", header=T, stringsAsFactors = F, dec = ".")
nrow(inverted_data_gaussian)
ncol(inverted_data_gaussian)
mean(inverted_data_gaussian$APHu)
sd(inverted_data_gaussian$APHu)
#plot(inverted_data_gaussian$APHu)

cov(inverted_data_gaussian)
inverted_data_gaussian$ZSx <- mean(inverted_data_gaussian$ZSx) + 0.001 * rnorm(nrow(inverted_data_gaussian))
inverted_data_gaussian$H7x <- mean(inverted_data_gaussian$H7x) + 0.001 * rnorm(nrow(inverted_data_gaussian))
cov(inverted_data_gaussian)

print(hotelling.test(x = data, y = inverted_data_gaussian))


plot(data$APHu)
plot(data[, 1], type="l", col="gray", ylim=c(min(data[, 1]), max(data[, 1])))
plot(inverted_data[, 1], type="l", col="red", ylim=c(min(data[, 1]), max(data[, 1])))
lines(data[, 1], col="gray")


### ARIMA
data_filtered <- read.csv2("ip.txt", sep = "\t", header = T, stringsAsFactors = F, strip.white = T, blank.lines.skip=T,
                  colClasses = c("character", rep("numeric", 14)), dec=".")
data_filtered <- data_filtered[!is.na(data_filtered$APHu),]
# filter by date
data_filtered <- (data_filtered[startsWith(data_filtered$Tiempoinicio, "09-oct-2015"), ])
data_filtered <- data_filtered[, seq(2,15)]
head(data_filtered)
str(data_filtered)
summary(data_filtered)

inverted_data <- read.csv2("generated/inverse_X.csv", sep = "\t", header=T, stringsAsFactors = F, 
                           colClasses = rep("numeric", 14), dec = ".")
head(inverted_data)
str(inverted_data)
summary(inverted_data)

# Hotelling T2
res = hotelling.test(x = data_filtered, y = inverted_data)
print(res)


# COMPONENTS
nipals_T <- read.csv2("generated/nipals_T_ts.csv", sep = "\t", header = T, stringsAsFactors = F, dec=".")
head(nipals_T)
str(nipals_T)

mean <- mean(nipals_T$X0)
std <- sd(nipals_T$X0)
boxplot(nipals_T$X0)
abline(h=mean)
summary(nipals_T$X0)


colMeans(data_filtered)
colMeans(inverted_data)

plot(data$APHu)
plot(inverted_data_gaussian$APHu)
plot(inverted_data$APHu)

plot(inverted_data[, 1], type="l", col="red")
lines(data_filterd[, 1], col="gray")


# Box's M
box_data <- data
box_data$label <- 1
#head(box_data)

box_inverted_data <- inverted_data
box_inverted_data$label <- 2
# head(box_inverted_data)

all <- rbind(box_data, box_inverted_data)
# head(all)
# tail(all)
# str(all)
# summary(all)


# res_box <- boxM(data = all[, -15], grouping = all[, 15])
#print(res_box) 

# cov.Mtest(all[, -15], all[, 15])
# BoxMTest(all[, -15], factor(all[, 15]))
