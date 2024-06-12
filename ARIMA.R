# SKRIPSI
# Karin Nathania Huang 
# 01112200030
# Prediksi dengan model ARIMA

setwd("/Users/jiao/Documents/SKRIPSI")

data = read.csv("WTI_USD Historical Data.csv") 

# Invert Data 
newdata = data[nrow(data):1, ]

summary(newdata)
str(newdata)

# Convert to num n fill the NA
library(zoo)
library(dplyr)
#data$Vol. = gsub("[K]", "0", data$Vol.)
#data$Vol. = gsub("[M]", "0000", data$Vol.)
#data$Vol. = gsub("\\.", "", data$Vol.)
#data$Vol. = as.numeric(data$Vol.)
#data$Vol. = na.locf(data$Vol.)

newdata$Change.. = gsub("\\%", "", newdata$Change..)
newdata$Change.. = as.numeric(newdata$Change..)
newdata$Change.. = (newdata$Change.. / 100)

# Ubah date dari chr ke date
# Install and load the required packages if not installed
if (!requireNamespace("lubridate", quietly = TRUE)) {
  install.packages("lubridate")
}
if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr")
}

library(lubridate)
library(dplyr)

# Ubah kolom 'tanggal' ke dalam format Date
newdata$Date <- as.Date(newdata$Date,format = "%m/%d/%Y" )
# Tambahkan kolom baru untuk tanggal, bulan, dan tahun
newdata <- newdata %>%
    mutate(
      tanggal = day(Date),
      bulan = month(Date, label = TRUE),
      tahun = year(Date)
  )

print(newdata)

# Split data into 3 sets
n_rows <- nrow(newdata)
rows_per_set <- n_rows %/% 3

set1 <- newdata[1:rows_per_set, ]
set2 <- newdata[(rows_per_set + 1):(2 * rows_per_set), ]
set3 <- newdata[(2 * rows_per_set + 1):n_rows, ]

print(set1) # 2 januari 2019 - 30 agustus 2020
print(set2) # 31 agustus 2020 - 1 mei 2022
print(set3) # 2 mei 2022 - 29 desember 2023

# ARIMA
library(ggplot2)
library(tseries)
library(TSA)

# untuk set 1
data1 <-ts(set1$Price)

adf.test(data1)
# Dickey-Fuller = -1.8617, Lag order = 8, p-value = 0.6368
# alternative hypothesis: stationary
# karena p-value = 0.6368 lebih dari 0.05 maka H0 diterima yaitu data ini tidak stasioner
plot(data1)

# dilakukan log differencing agar data stasioner
ddata1 = diff(log(data1))
adf.test(ddata1)
# Dickey-Fuller = -5.6859, Lag order = 8, p-value = 0.01
# alternative hypothesis: stationary
plot(ddata1)

acf(ddata1)
pacf(ddata1)
eacf(ddata1)

# order (1,0,0)
# order (0,0,1)
# order (2,0,0)
# order (3,0,0)
# order (1,0,2)

mod11 = arima(ddata1, order = c(1,0,0))
mod12 = arima(ddata1, order = c(0,0,1))
mod13 = arima(ddata1, order = c(2,0,0))
mod14 = arima(ddata1, order = c(3,0,0))
mod15 = arima(ddata1, order = c(1,0,2))

mod11$aic #-1918.546 terkecil 3
mod12$aic #-1918.34
mod13$aic #-1916.568
mod14$aic #-1920.209 terkecil 1
mod15$aic #-1919.709 terkecil 2

res11 = mod11$residuals
hist(res11)
acf(res11)

res14 = mod14$residuals
hist(res14)
acf(res14)

res15 = mod15$residuals
hist(res15)
acf(res15)

set1_tail = tail(set1,25)
set1_tail_price = data.frame(set1_tail$Price)

# data training = 491 (2 januari 2019 - 31 juli 2020)
# data testing = 24 (2 agustus 2020 - 30 agustus 2020)

returnhat11 = matrix(nrow = 24, ncol = 1)
for(k in 1:24){
  mod = arima(ddata1[k:(490+k)], order = c(1,0,0))
  returnhat11[k,] = predict(mod, n.ahead = 1)$pred
}
ts.plot(returnhat11)
returnhat11 <- c(0, tail(returnhat11, 24))

returndata11 = matrix(nrow = 25, ncol = 1)
for (k in 2:25) {
  returndata11[k,] = set1_tail_price[k - 1,] * exp(returnhat11[k])
}
price11 = returndata11


returnhat15 = matrix(nrow = 24, ncol = 1)
for(k in 1:24){
  mod = arima(ddata1[k:(490+k)], order = c(1,0,2))
  returnhat15[k,] = predict(mod, n.ahead = 1)$pred
}
ts.plot(returnhat15)
returnhat15 <- c(0, tail(returnhat15, 24))

returndata15 = matrix(nrow = 25, ncol = 1)
for (k in 2:25) {
  returndata15[k,] = set1_tail_price[k - 1,] * exp(returnhat15[k])
}
price15 = returndata15

# untuk set 2 
data2 <-ts(set2$Price)

adf.test(data2)
# Dickey-Fuller = -3.061, Lag order = 8, p-value = 0.1292
# alternative hypothesis: stationary
# karena p-value = 0.1292 lebih dari 0.05 maka H0 diterima yaitu data ini tidak stasioner
plot(data2)

# dilakukan log differencing agar data stasioner
ddata2 = diff(log(data2))
adf.test(ddata2)
# Dickey-Fuller = -8.7978, Lag order = 8, p-value = 0.01
# alternative hypothesis: stationary
plot(ddata2)

acf(ddata2)
pacf(ddata2)
eacf(ddata2)

# order (0,0,0)
# order (0,0,1)
# order (1,0,1)
# order (0,0,2)
# order (1,0,2)

mod21 = arima(ddata2, order = c(0,0,0))
mod22 = arima(ddata2, order = c(0,0,1))
mod23 = arima(ddata2, order = c(1,0,1))
mod24 = arima(ddata2, order = c(0,0,2))
mod25 = arima(ddata2, order = c(1,0,2))

mod21$aic #-2171.933 terkecil 1
mod22$aic #-2170.27 terkecil 2
mod23$aic #-2168.337
mod24$aic #-2168.641 terkecil 3
mod25$aic #-2167.824 

res21 = mod21$residuals
hist(res21)
acf(res21)

res22 = mod22$residuals
hist(res22)
acf(res22)

res24 = mod24$residuals
hist(res24)
acf(res24)

set2_tail = tail(set2,25)
set2_tail_price = data.frame(set2_tail$Price)

# data training = 491 (31 agustus 2020 - 31 maret 2022)
# data testing = 24 (1 april 2022 - 1 mei 2022)

returnhat22 = matrix(nrow = 24, ncol = 1)
for(k in 1:24){
  mod = arima(ddata2[k:(490+k)], order = c(0,0,1))
  returnhat22[k,] = predict(mod, n.ahead = 1)$pred
}
ts.plot(returnhat22)
returnhat22 <- c(0, tail(returnhat22, 24))

returndata22 = matrix(nrow = 25, ncol = 1)
for (k in 2:25) {
  returndata22[k,] = set2_tail_price[k - 1,] * exp(returnhat22[k])
}
price22 = returndata22


returnhat24 = matrix(nrow = 24, ncol = 1)
for(k in 1:24){
  mod = arima(ddata2[k:(490+k)], order = c(0,0,0))
  returnhat24[k,] = predict(mod, n.ahead = 1)$pred
}
ts.plot(returnhat24)
returnhat24 <- c(0, tail(returnhat24, 24))

returndata24 = matrix(nrow = 25, ncol = 1)
for (k in 2:25) {
  returndata24[k,] = set2_tail_price[k - 1,] * exp(returnhat24[k])
}
price24 = returndata24

# untuk set 3 
data3 <-ts(set3$Price)

adf.test(data3)
# Dickey-Fuller = -2.0369, Lag order = 8, p-value = 0.5627
# alternative hypothesis: stationary
# karena p-value = 0.5627 lebih dari 0.05 maka H0 diterima yaitu data ini tidak stasioner
plot(data3)

# dilakukan log differencing agar data stasioner
ddata3 = diff(log(data3))
adf.test(ddata3)
# Dickey-Fuller = -7.7559, Lag order = 8, p-value = 0.01
# alternative hypothesis: stationary
plot(ddata3)

acf(ddata3)
pacf(ddata3)
eacf(ddata3)

# order (0,0,0)
# order (0,0,1)
# order (1,0,1)
# order (1,0,2)
# order (0,0,3)

mod31 = arima(ddata3, order = c(0,0,0))
mod32 = arima(ddata3, order = c(0,0,1))
mod33 = arima(ddata3, order = c(1,0,1))
mod34 = arima(ddata3, order = c(1,0,2))
mod35 = arima(ddata3, order = c(0,0,3))

mod31$aic #-2483.581 terkecil 3
mod32$aic #-2483.005
mod33$aic #-2481.691 
mod34$aic #-2485.419 terkecil 2
mod35$aic #-2487.235 terkecil 1

res31 = mod31$residuals
hist(res31)
acf(res31)

res34 = mod34$residuals
hist(res34)
acf(res34)

res35 = mod35$residuals
hist(res35)
acf(res35)

set3_tail = tail(set3,25)
set3_tail_price = data.frame(set3_tail$Price)

# data training = 491 (2 mei 2022 - 30 nov 2022)
# data testing = 24 (1 dec 2022 - 29 dec 2023)

returnhat31 = matrix(nrow = 24, ncol = 1)
for(k in 1:24){
  mod = arima(ddata3[k:(490+k)], order = c(0,0,0))
  returnhat31[k,] = predict(mod, n.ahead = 1)$pred
}
ts.plot(returnhat31)
returnhat31 <- c(0, tail(returnhat31, 24))

returndata31 = matrix(nrow = 25, ncol = 1)
for (k in 2:25) {
  returndata31[k,] = set3_tail_price[k - 1,] * exp(returnhat31[k])
}
price31 = returndata31


returnhat35 = matrix(nrow = 24, ncol = 1)
for(k in 1:24){
  mod = arima(ddata3[k:(490+k)], order = c(0,0,3))
  returnhat35[k,] = predict(mod, n.ahead = 1)$pred
}
ts.plot(returnhat35)
returnhat35 <- c(0, tail(returnhat35, 24))

returndata35 = matrix(nrow = 25, ncol = 1)
for (k in 2:25) {
  returndata35[k,] = set3_tail_price[k - 1,] * exp(returnhat35[k])
}
price35 = returndata35

alldata = data.frame(set1_tail_price, returnhat11, price11, returnhat15, price15,
                     set2_tail_price, returnhat22, price22, returnhat24, price24,
                     set3_tail_price, returnhat31, price31, returnhat35, price35 )

new_alldata = tail(alldata, 24)

# for set 1
# order(1,0,0)   
mse11 = mean(new_alldata$set1_tail.Price - new_alldata$price11)^2
mse11

rmse11 = sqrt((mean(new_alldata$set1_tail.Price - new_alldata$price11)^2))
rmse11

mape11 = (mean((abs(new_alldata$set1_tail.Price - new_alldata$price11))/(new_alldata$set1_tail.Price)))*100 
mape11

# order(1,0,2)
mse15 = mean(new_alldata$set1_tail.Price - new_alldata$price15)^2
mse15

rmse15 = sqrt((mean(new_alldata$set1_tail.Price - new_alldata$price15)^2))
rmse15

mape15 = (mean((abs(new_alldata$set1_tail.Price - new_alldata$price15))/(new_alldata$set1_tail.Price)))*100 
mape15

########################################
# mod15 best for set 1 (1, 0 ,2)
########################################

# for set 2
# order(0,0,1)
mse22 = mean(new_alldata$set2_tail.Price - new_alldata$price22)^2
mse22

rmse22 = sqrt((mean(new_alldata$set2_tail.Price - new_alldata$price22)^2))
rmse22

mape22 = (mean((abs(new_alldata$set2_tail.Price - new_alldata$price22))/(new_alldata$set2_tail.Price)))*100 
mape22

# order(0,0,2)
mse24 = mean(new_alldata$set2_tail.Price - new_alldata$price24)^2
mse24

rmse24 = sqrt((mean(new_alldata$set2_tail.Price - new_alldata$price24)^2))
rmse24

mape24 = (mean((abs(new_alldata$set2_tail.Price - new_alldata$price24))/(new_alldata$set2_tail.Price)))*100 
mape24

########################################
# mod22 best for set 2 (0, 0, 1)
########################################

# for set 3
# order(0,0,0)
mse31 = mean(new_alldata$set3_tail.Price - new_alldata$price31)^2
mse31

rmse31 = sqrt((mean(new_alldata$set3_tail.Price - new_alldata$price31)^2))
rmse31

mape31 = (mean((abs(new_alldata$set3_tail.Price - new_alldata$price31))/(new_alldata$set3_tail.Price)))*100 
mape31

# order(0,0,3)
mse35 = mean(new_alldata$set3_tail.Price - new_alldata$price35)^2
mse35

rmse35 = sqrt((mean(new_alldata$set3_tail.Price - new_alldata$price35)^2))
rmse35

mape35 = (mean((abs(new_alldata$set3_tail.Price - new_alldata$price35))/(new_alldata$set3_tail.Price)))*100 
mape35

########################################
# mod35 best for set 3 (0, 0, 3)
########################################

dataakhir1 = tail(set1,24)

data_set1 <- data.frame(time = set1$Date, price = set1$Price)
predprice1 <- data.frame(time = dataakhir1$Date, price = new_alldata$price15)
combined_plot1 <- ggplot() +
  geom_line(data = data_set1, aes(x = time, y = price, color = "Actual Price")) +
  geom_line(data = predprice1, aes(x = time, y = price, color = "Predicted Price")) +
  scale_color_manual(values = c("Actual Price" = "skyblue", "Predicted Price" = "black")) +
  labs(color = "") +
  theme_minimal()

print(combined_plot1)

########################################

dataakhir2 = tail(set2,24)
data_set2 <- data.frame(time = set2$Date, price = set2$Price)
predprice2 <- data.frame(time = dataakhir2$Date, price = new_alldata$price22)
combined_plot2 <- ggplot() +
  geom_line(data = data_set2, aes(x = time, y = price, color = "Actual Price")) +
  geom_line(data = predprice2, aes(x = time, y = price, color = "Predicted Price")) +
  scale_color_manual(values = c("Actual Price" = "skyblue", "Predicted Price" = "black")) +
  labs(color = "") +
  theme_minimal()

print(combined_plot2)

########################################

dataakhir3 = tail(set3,24)
data_set3 <- data.frame(time = set3$Date, price = set3$Price)
predprice3 <- data.frame(time = dataakhir3$Date, price = new_alldata$price35)
combined_plot3 <- ggplot() +
  geom_line(data = data_set3, aes(x = time, y = price, color = "Actual Price")) +
  geom_line(data = predprice3, aes(x = time, y = price, color = "Predicted Price")) +
  scale_color_manual(values = c("Actual Price" = "skyblue", "Predicted Price" = "black")) +
  labs(color = "") +
  theme_minimal()

print(combined_plot3)
