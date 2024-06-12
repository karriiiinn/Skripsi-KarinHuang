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

library(ggplot2)
library(tseries)
library(TSA)

data1 <-ts(newdata$Price)

adf.test(data1)
# Dickey-Fuller = -1.4881, Lag order = 11, p-value = 0.795
# alternative hypothesis: stationary
# karena p-value = 0.6368 lebih dari 0.05 maka H0 diterima yaitu data ini tidak stasioner
plot(data1)

# dilakukan log differencing agar data stasioner
ddata1 = diff(log(data1))
adf.test(ddata1)
# Dickey-Fuller = -10.605, Lag order = 11, p-value = 0.01
# alternative hypothesis: stationary
plot(ddata1)

acf(ddata1)
pacf(ddata1)
eacf(ddata1)

# order(0,0,1)
# order(0,0,2)
# order(2,0,1)
# order(1,0,2)
# order(0,0,3)

mod11 = arima(ddata1, order = c(0,0,1))
mod12 = arima(ddata1, order = c(0,0,2))
mod13 = arima(ddata1, order = c(2,0,1))
mod14 = arima(ddata1, order = c(1,0,2))
mod15 = arima(ddata1, order = c(0,0,3))

mod11$aic #-6415.459 terkecil 2
mod12$aic #-6414.669 terkecil 3
mod13$aic #-6412.99
mod14$aic #-6415.798 terkecil 1
mod15$aic #-6413.19

res11 = mod11$residuals
hist(res11)
acf(res11)

res12 = mod12$residuals
hist(res12)
acf(res12)

res14 = mod14$residuals
hist(res14)
acf(res14)

data_tail = tail(newdata,517)
data_tail_price = data.frame(data_tail$Price)

# data training = 1031 (2 januari 2019 - 29 april 2022)
# data testing = 516 (1 may 2022 - 29 dec 2023)

returnhat11 = matrix(nrow = 516, ncol = 1)
for(k in 1:516){
  mod = arima(ddata1[k:(1030+k)], order = c(0,0,1))
  returnhat11[k,] = predict(mod, n.ahead = 1)$pred
}
ts.plot(returnhat11)
returnhat11 <- c(0, tail(returnhat11, 517))

returndata11 = matrix(nrow = 517, ncol = 1)
for (k in 2:517) {
  returndata11[k,] = data_tail_price[k - 1,] * exp(returnhat11[k])
}
price11 = returndata11


returnhat12 = matrix(nrow = 516, ncol = 1)
for(k in 1:516){
  mod = arima(ddata1[k:(1030+k)], order = c(0,0,2))
  returnhat12[k,] = predict(mod, n.ahead = 1)$pred
}
ts.plot(returnhat12)
returnhat12 <- c(0, tail(returnhat12, 516))

returndata12 = matrix(nrow = 517, ncol = 1)
for (k in 2:517) {
  returndata12[k,] = data_tail_price[k - 1,] * exp(returnhat12[k])
}
price12 = returndata12


returnhat14 = matrix(nrow = 516, ncol = 1)
for(k in 1:516){
  mod = arima(ddata1[k:(1030+k)], order = c(1,0,2))
  returnhat14[k,] = predict(mod, n.ahead = 1)$pred
}
ts.plot(returnhat14)
returnhat14 <- c(0, tail(returnhat14, 516))

returndata14 = matrix(nrow = 517, ncol = 1)
for (k in 2:517) {
  returndata14[k,] = data_tail_price[k - 1,] * exp(returnhat14[k])
}
price14 = returndata14

alldata = data.frame(data_tail_price, price11, price12, price14)
alldata = tail(alldata, 516)

mse11 = mean(alldata$data_tail.Price - alldata$price11)^2
mse11

rmse11 = sqrt((mean(alldata$data_tail.Price - alldata$price11)^2))
rmse11

mape11 = (mean((abs(alldata$data_tail.Price - alldata$price11))/(alldata$data_tail.Price)))*100 
mape11


mse12 = mean(alldata$data_tail.Price - alldata$price12)^2
mse12

rmse12 = sqrt((mean(alldata$data_tail.Price - alldata$price12)^2))
rmse12

mape12 = (mean((abs(alldata$data_tail.Price - alldata$price12))/(alldata$data_tail.Price)))*100 
mape12

mse14 = mean(alldata$data_tail.Price - alldata$price14)^2
mse14


rmse14 = sqrt((mean(alldata$data_tail.Price - alldata$price14)^2))
rmse14

mape14 = (mean((abs(alldata$data_tail.Price - alldata$price14))/(alldata$data_tail.Price)))*100 
mape14 

#best orde 11 (0,0,1) 

dataakhir1 = tail(newdata,516)

data_set1 <- data.frame(time = newdata$Date, price = newdata$Price)
predprice1 <- data.frame(time = dataakhir1$Date, price = alldata$price11)
combined_plot1 <- ggplot() +
  geom_line(data = data_set1, aes(x = time, y = price, color = "Actual Price")) +
  geom_line(data = predprice1, aes(x = time, y = price, color = "Predicted Price")) +
  scale_color_manual(values = c("Actual Price" = "skyblue", "Predicted Price" = "black")) +
  labs(color = "") +
  theme_minimal()

print(combined_plot1)
