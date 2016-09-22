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
