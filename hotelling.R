ç# install.packages("Hotelling")
library("Hotelling")

data(container.df)
head(container.df)
str(container.df)
summary(container.df)

factor(container.df$gp)

split.data = split(container.df[,-1], container.df$gp)
head(split.data)

x = split.data[[1]]
y = split.data[[2]]
(x)
(y)

res = hotelling.stat(x, y)
res

res = hotelling.stat(x, y, TRUE)
res

res = hotelling.test(x, y)
res

print(res)


# library("biotools")
# boxM(container.df[, 2:10], container.df[, 1])
# 

a = c(1,2,3,4, 3,4,5,2)
#b = c(1,2,3,4, 3,4,5,2)
#b = c(1.1,2.1,3.1,4.1, 3.1,4.1,5.1,2.1)
#b = c(1.2,2.2,3.2,4.2, 3.2,4.2,5.2,2.2,)
b = c(1.2,2.2,3.2,4.2,5.2, 3.2,4.2,5.2,2.2,8.2)
a1 = matrix(a, ncol=2)
a1
b1 = matrix(b, ncol=2)
b1

res = hotelling.test(a1, b1)
res

# p-value > 0.05 (alpha), the matrices variances are equal




install.packages("rrcov")
library("rrcov")

T2.test(x, y)
T2.test(data, inverted_data)


install.packages("ICSNP")
library("ICSNP")

HotellingsT2(X = data, Y = inverted_data)


mean(data[,1]) - mean(inverted_data[,1])

