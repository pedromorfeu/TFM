# install.packages("Hotelling")
library("Hotelling")

# install.packages("biotools")
library("biotools")


data <- read.csv2("ip.txt", sep = "\t", header = T, stringsAsFactors = F, strip.white = T, blank.lines.skip=T,
                  colClasses = c("character", rep("numeric", 14)), dec=".")
data <- data[!is.na(data$APHu),]
# filter by date
data <- (data[startsWith(data$Tiempoinicio, "07-oct-2015"), ])
data <- data[, seq(2,15)]
head(data)
str(data)
summary(data)

sd(data[, 1])
apply(data, 2, sd)
apply(data, 2, mean)


inverted_data_gaussian <- read.csv2("generated/inverse_X_gaussian.csv", sep = "\t", header=T, stringsAsFactors = F, 
                           colClasses = rep("numeric", 14), dec = ".")
head(inverted_data_gaussian)
str(inverted_data_gaussian)
summary(inverted_data_gaussian)


inverted_data <- read.csv2("generated/inverse_X.csv", sep = "\t", header=T, stringsAsFactors = F, 
                           colClasses = rep("numeric", 14), dec = ".")
head(inverted_data)
str(inverted_data)
summary(inverted_data)


plot(inverted_data[, 1], type="l", col="red")
lines(data[, 1], col="gray")


# Hotelling T2
res = hotelling.test(x = data, y = inverted_data)
#res = hotelling.stat(x = data, y = inverted_data)
print(res)


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
