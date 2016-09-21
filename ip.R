data <- read.csv2("ip.txt", sep = "\t", header = F, strip.white = T, blank.lines.skip=T, skip = 2)
head(data)

colnames(data) <- c("Tiempoinicio", "APHu", "APVs", "ACPv", "ZSx", "ZUs", "H7x", "H1x", "H2x", "H6x", "H3x", "H4x", "H5x", "ACPx", "Svo")
head(data)

data$APHu <- as.numeric(data$APHu)
data$APVs <- as.numeric(data$APHu)
data$ACPv <- as.numeric(data$ACPv)
data$ZSx <- as.numeric(data$ZSx)
data$ZUs <- as.numeric(data$ZUs)
data$H7x <- as.numeric(data$H7x)
data$H1x <- as.numeric(data$H1x)
data$H2x <- as.numeric(data$H2x)
data$H6x <- as.numeric(data$H6x)
data$H3x <- as.numeric(data$H3x)
data$H4x <- as.numeric(data$H4x)
data$H5x <- as.numeric(data$H5x)
data$ACPx <- as.numeric(data$ACPx)
data$Svo <- as.numeric(data$Svo)
str(data)

data[is.na(data),]
head(data)

summary(data[, seq(2,15)])
var(data[, seq(2,3)])
var(data[, c("Svo","ACPx")])

summary(data$Svo)
mean(data$Svo)
var(data$Svo)
var(data$ACPx)
sd(data$Svo)
