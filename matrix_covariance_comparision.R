# install.packages("Hotelling")
library("Hotelling")

# install.packages("biotools")
library("biotools")


### GAUSSIAN
data <- read.csv2("ip.txt", sep = "\t", header = T, stringsAsFactors = F, strip.white = T, blank.lines.skip=T,
                  colClasses = c("character", rep("numeric", 14)), dec=".")
data <- data[!is.na(data$APHu),]
# filter by date
data <- data[, seq(2,15)]
head(data)
str(data)
summary(data)

inverted_data_gaussian <- read.csv2("generated/inverse_X_gaussian.csv", sep = "\t", header=T, stringsAsFactors = F, 
                           colClasses = rep("numeric", 14), dec = ".")
head(inverted_data_gaussian)
str(inverted_data_gaussian)
summary(inverted_data_gaussian)

res = hotelling.test(x = data, y = inverted_data_gaussian)
print(res)


plot(data$APHu)
plot(data[, 1], type="l", col="gray", ylim=c(min(data[, 1]), max(data[, 1])))
plot(inverted_data[, 1], type="l", col="red", ylim=c(min(data[, 1]), max(data[, 1])))
lines(data[, 1], col="gray")


### ARIMA
data_filtered <- read.csv2("ip.txt", sep = "\t", header = T, stringsAsFactors = F, strip.white = T, blank.lines.skip=T,
                  colClasses = c("character", rep("numeric", 14)), dec=".")
data_filtered <- data_filtered[!is.na(data_filtered$APHu),]
# filter by date
data_filtered <- (data_filtered[startsWith(data_filtered$Tiempoinicio, "07-oct-2015"), ])
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
