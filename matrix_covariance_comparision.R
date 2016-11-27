# install.packages("Hotelling")
library("Hotelling")
#detach("package:Hotelling", unload=TRUE)

# install.packages("biotools")
# library("biotools")
#detach("package:biotools", unload=TRUE)

#################
# Original data #
#################
ip_data <- read.csv2("ip.txt", sep = "\t", header = T, stringsAsFactors = F, strip.white = T, blank.lines.skip=T,
                  colClasses = c("character", rep("numeric", 14)), dec=".")
ip_data <- ip_data[!is.na(ip_data$APHu),]
# filter by date
ip_data <- (ip_data[startsWith(ip_data$Tiempoinicio, "06-oct-2015"), ])
ip_data <- ip_data[, seq(2,15)]
nrow(ip_data)
ncol(ip_data)
mean(ip_data$APHu)
sd(ip_data$APHu)
plot(ip_data$APHu)

# check the variables with zero variance and give them white noise
# otherwise the test will fail due to inability to invert
cov(ip_data)
# ZSx and H7x have zero variance
ip_data$ZSx <- mean(ip_data$ZSx) + 0.001 * rnorm(nrow(ip_data))
ip_data$H7x <- mean(ip_data$H7x) + 0.001 * rnorm(nrow(ip_data))
cov(ip_data)
mean(ip_data$APHu)
sd(ip_data$APHu)

head(ip_data)

############################################
# Original data resampled and interpolated #
############################################
data <- read.csv2("generated/data.csv", sep = "\t", header=T, stringsAsFactors = F, dec = ".")
# filter by date
nrow(data)
ncol(data)
mean(data$APHu)
sd(data$APHu)
plot(data$APHu)

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


### GAUSSIAN
inverted_data_gaussian <- read.csv2("generated/inverse_X_gaussian1.csv", sep = "\t", header=T, stringsAsFactors = F, dec = ".")
# sample data
# inverted_data_gaussian <- inverted_data_gaussian[sample(1:nrow(inverted_data_gaussian), 100, replace=T), ]
# sample_list <- c(316289,316289,264139,478894,267345,67631,12284,173798,12284,24780,340131,64462,340131,340131,467538,107701,438926,451920,389340,467538,22443,372411,322347,50292,348993,318677,322347,56108,32391,325721,372411,431056,226810,56108,33184,38791,107701,480253,183959,340131,325721,38791,32391,63568,122734,56108,467538,416203,14856,237454,431056,416203,431056,440109,440109,397288,160867,117909,373965,373965,103265,286954,103265,72137,323755,165150,444543,373965,43158,456010,70990,205352,373965,373965,205352,204737,195913,373965,335150,72137,205352,98022,294916,31278,323755,72137,294916,204970,363644,416617,294916,440109,416617,303589,120464,204970,440109,436558,120464,72137,294916,294916,103265,58870,189018,165150,43158,368716,456010,130017,111555,7314,130017,456010,26533,368716,220589,220589,335966,302442,185015,1977,247428,328588,193472,103970,117458,386729,362442,306305,386729,227368,78134,380708,185015,302442,380708,78134,398092,169579,398092,78134,380708,39439,398092,235503,274351,323314,423506,96528,219360,402722,498521,227362,361684,206201,317226,391582,139875,96528,206201,227362,361684,421371,206542,359006,359006,151485,284512,331984,359006,180377,352238,337302,196036,230342,352238,243862,328588,243862,206648,278216,247428,117458,351539,478560,47298,349901,163221,402714,291980,497709,387113,458831,62742,8585,62742,471683,310892,224422,441487,153047,322284,471683,60278,189978,208894,441487,431910,225911,209485,30,225911,209433,25623,441487,206115,153047,441487,471683,194044,322284,458831,209433,452574,46471,396397,51265,51265,51265,51265,334582,47238,165865,88837,105918,258727,189978,234315,439283,361684,361684,349901,431910,329430,310892,60278,342547,471683,310892,60278,189978,101293,249954,47238,101293,60278,441487,206115,153047,50760,178047,289016,42649,256234,404309,441802,260582,256234,89257,419416,344670,256234,260582,282007,89257,138805,25981,396397,483068,487749,396397,483068,194044,209433,205521,224422,181945,60278,249954,205521,88837,10037,104433,104433,11059,256420,462649,234315,278075,141585,63619,165256,183969,141585,389224,317332,317332,402722,319190,405101,60454,180791,89398,395295,83956,246733,395295,395295,319190,96528,96528,52949,60454,180791,180791,60454,165256,317332,76863,141585,141585,317332,449118,405101,32046,141585,328364,412289,379215,76863,141585,449118,412289,317332,32046,405101,395295,255068,424113,402722,357546,395295,357546,185383,391582,317332,141585,395295,328364,255068,60454,405101,180791,449118,165256,103474,40686,156931,243376,246326,165256,317332,412289,328704,328704,372414,253109,25025,46652,328704,497157,185919,90956,390965,29556,497157,335474,210672,445810,76274,90956,29556,445810,76696,29556,253109,469416,25025,25025,46652,391930,373597,46652,446736,324964,89993,84604,309236,76274,108557,84604,29556,5578,479628,319190,412289,23964,180791,326718,185383,180791,328364,165256,141585,317332,288799,317332,326718,185383,141585,288799,185383,317332,424113,23964,141585,319190,32046,288799,141585,326718,449118,449118,412289,32046,412289,180791,424113,424113,352196,449118,424113,399174,473881,218908,76863,252458,63619,264214,218908,63619,484722,71777,103474,384874,11059,293134,293134,384874,495911,484722,495911,89835,156931,391542,281869,328129,240592,138504,376411,269516,131594,172242,243347,29556,497157,263515,29556,90956,185919,253109,263515,372414,29556,253109,390965,252016,252016,223480)
# inverted_data_gaussian <- inverted_data_gaussian[sample_list, ]
nrow(inverted_data_gaussian)
ncol(inverted_data_gaussian)

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
generated_gaussian_copy <- generated_gaussian
# generated_gaussian_copy <- generated_gaussian_copy[c(1:100000, 1:100000, 1:100000), ]
generated_gaussian_copy <- generated_gaussian_copy[sample(nrow(generated_gaussian_copy), 200, replace = T), ]
head(generated_gaussian_copy)
str(generated_gaussian_copy)

cov(generated_gaussian_copy)

# Hotelling T2
print(hotelling.test(x = nipals_T, y = generated_gaussian_copy))


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
