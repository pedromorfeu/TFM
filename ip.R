data <- read.csv2("ip.txt", sep = "\t", header = T, stringsAsFactors = F, strip.white = T, blank.lines.skip=T,
                  colClasses = c("character", rep("numeric", 14)), dec=".")
data <- data[!is.na(data$APHu),]
# head(data)
# str(data)
# summary(data)

data[is.na(data),]
head(data)

Sys.setlocale("LC_TIME", "spanish")

data$Tiempoinicio <- sub("(\\d+-)(\\w+)(-\\d+\\s\\d+:\\d+:\\d+)", "\\1\\2.\\3", data$Tiempoinicio)
data$Tiempoinicio <- as.POSIXct(data$Tiempoinicio, format="%d-%b-%Y %H:%M:%S")

head(data)

boxplot(data[1:10, "APHu"]~data[1:10, "Tiempoinicio"])

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
