data <- read.csv2("ip.txt", sep = "\t", header = T)

data$APHu <- as.numeric(trimws(data$APHu))
str(data)

