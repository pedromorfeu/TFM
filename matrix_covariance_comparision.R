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
# head(data)
# tail(data)

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
inverted_data_gaussian <- inverted_data_gaussian[c(1,2,3,4,5,6,7,8,9,10), ]
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


plot(data[, 1], type="l", col="gray", ylim=c(min(data[, 1]), max(data[, 1])))
plot(inverted_data[, 1], type="l", col="red", ylim=c(min(data[, 1]), max(data[, 1])))
lines(data[, 1], col="gray")


### ARIMA
inverted_data <- read.csv2("generated/inverse_X.csv", sep = "\t", header=T, stringsAsFactors = F, 
                           colClasses = rep("numeric", 14), dec = ".")
nrow(inverted_data)
ncol(inverted_data)
mean(inverted_data$APHu)
sd(inverted_data$APHu)
#plot(inverted_data$APHu)
head(inverted_data)
tail(inverted_data)


cov(inverted_data)
inverted_data$ZSx <- mean(inverted_data$ZSx) + 0.001 * rnorm(nrow(inverted_data))
inverted_data$H7x <- mean(inverted_data$H7x) + 0.001 * rnorm(nrow(inverted_data))
cov(inverted_data)

# Hotelling T2
print(hotelling.test(x = data, y = inverted_data))


# COMPONENTS
nipals_T <- read.csv2("generated/nipals_T_ts.csv", sep = "\t", header = T, stringsAsFactors = F, dec=".")
nipals_T <- nipals_T[, seq(2,6)]
head(nipals_T)
str(nipals_T)

cov(nipals_T)


generated_gaussian <- read.csv2("generated/generated_gaussian.csv", sep = "\t", header = T, stringsAsFactors = F, dec=".")
# generated_gaussian <- generated_gaussian[c(1:100000, 1:100000, 1:100000), ]
generated_gaussian <- generated_gaussian[sample(nrow(generated_gaussian), 100, replace = T), ]
head(generated_gaussian)
str(generated_gaussian)

cov(generated_gaussian)

# Hotelling T2
print(hotelling.test(x = nipals_T, y = generated_gaussian))


generated_X <- read.csv2("generated/generated_X.csv", sep = "\t", header = T, stringsAsFactors = F, dec=".")
head(generated_X)
str(generated_X)

cov(generated_X)

# Hotelling T2
print(hotelling.test(x = nipals_T, y = generated_X))



test_X <- read.csv2("generated/test_X.csv", sep = "\t", header = T, stringsAsFactors = F, dec=".")
head(test_X)
str(test_X)

cov(test_X)

# Hotelling T2
print(hotelling.test(x = nipals_T, y = test_X))


colMeans(data_filtered)
colMeans(inverted_data)

plot(data$APHu)
plot(data_filtered$APHu)
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
