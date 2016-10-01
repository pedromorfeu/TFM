# install.packages("Hotelling")
library("Hotelling")

# install.packages("biotools")
library("biotools")


data <- read.csv2("ip.txt", sep = "\t", header = T, stringsAsFactors = F, strip.white = T, blank.lines.skip=T,
                  colClasses = c("character", rep("numeric", 14)), dec=".")
data <- data[, seq(2,15)]
data <- data[!is.na(data$APHu),]
# head(data)
# str(data)
# summary(data)


inverted_data <- read.csv2("generated/inverse_X.csv", sep = "\t", header=T, stringsAsFactors = F, 
                           colClasses = rep("numeric", 14), dec = ".")
# head(inverted_data)
# str(inverted_data)
# summary(inverted_data)

# Hotelling T2
res = hotelling.test(x = data, y = inverted_data)
res


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

boxM(data = all[, -15], grouping = all[, 15])

