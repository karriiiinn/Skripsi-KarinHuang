# SKRIPSI
# Karin Nathania Huang 
# 01112200030
# Prediksi dengan model LSTM

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

returndata = matrix(nrow = 1548, ncol = 1)
for (k in 2:1548) {
  returndata[k,] = log(newdata$Price[k] / newdata$Price[k-1])
}
returndata

newdata$returndata = c(returndata)

# Split data into 3 sets
n_rows <- nrow(newdata)
rows_per_set <- n_rows %/% 3

set1 <- newdata[1:rows_per_set, ]
set2 <- newdata[(rows_per_set + 1):(2 * rows_per_set), ]
set3 <- newdata[(2 * rows_per_set + 1):n_rows, ]

print(set1) # 2 januari 2019 - 30 agustus 2020
print(set2) # 31 agustus 2020 - 1 mei 2022
print(set3) # 2 mei 2022 - 29 desember 2023

dataset1 = data.frame(set1$tanggal, set1$bulan, set1$tahun, set1$Price, set1$returndata)
dataset2 = data.frame(set2$tanggal, set2$bulan, set2$tahun, set2$Price, set2$returndata)
dataset3 = data.frame(set3$tanggal, set3$bulan, set3$tahun, set3$Price, set3$returndata)

#############################################################################################
#############################################################################################
#############################################################################################
# untuk set 1
dataset1[] <- lapply(dataset1, as.numeric)
str(dataset1)

dataset1 = tail(dataset1, 515)

x1 = data.frame(dataset1$set1.tanggal, dataset1$set1.bulan, dataset1$set1.tahun, 
                dataset1$set1.returndata)

baris1_set2 = head(dataset2$set2.Price,1)
price1 = c(dataset1$set1.Price, baris1_set2)
price1 = tail(price1,515)
y1 = data.frame(price1)


matrix_data1 <- as.matrix(x1)
dv1 <- matrix_data1

# Pembagian data train dan test
n_rows1 <- nrow(dv1)

train_size1 <- 490
# train_index1 <- sample(seq_len(nrow(dataset1)), size = 490, replace = FALSE)
# train1 <- dataset1[train_index1, ]
# train_price1 <- train1$set1.Price

# test1 <- dataset1[-train_index1, ]
# test_price1 <- test1$set1.Price

#train_index1 = sample(seq_len(nrow(x1)), size = 490, replace = FALSE)
#train1 = x1[train_index1, ]
#train_price1 = y1[train_index1,]

#test1 = x1[-train_index1,]
#test_price1 = y1[-train_index1,]

# Membuat indeks sekuensial untuk pemilihan baris
sequential_indices1 <- seq_len(n_rows1)

# Membagi matriks menjadi set pelatihan dan pengujian
train_matrix1 <- dv1[sequential_indices1[1:train_size1], ]
test_matrix1 <- dv1[sequential_indices1[(train_size1 + 1):n_rows1], ]
train_price1 = y1[sequential_indices1[1:train_size1], ]
test_price1 = y1[sequential_indices1[(train_size1 + 1):n_rows1], ]

# Tampilkan dimensi dari set pelatihan dan pengujian
cat("Dimensi Set Pelatihan:", dim(train_matrix1), "\n")
cat("Dimensi Set Pengujian:", dim(test_matrix1), "\n")

# LSTM
# install.packages("remotes")# install.packages("remotes")
# remotes::install_github("rstudio/tensorflow", force = TRUE)
# reticulate::install_python()

# library(tensorflow)
#install_tensorflow(envname = "r-tensorflow")

# install.packages("keras")
# keras::install_keras()

library(keras)
library(tensorflow)

train_reshaped1 = array_reshape(train_matrix1, c(490, 4, 1))
test_reshaped1 = array_reshape(test_matrix1,c(25, 4, 1))
#############################################################################################
#############################################################################################

# LSTM LAYER = 1 
# NEURONS = 20
# BATCH SIZE = 32

model11 <- keras_model_sequential() %>%
  layer_lstm(units = 20, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

# Visualisasi model
# install.packages("graphviz")
# reticulate::py_install(c("graphviz", "pydot"))
# plot(model, show_shapes = TRUE, show_layer_names = TRUE)

model11 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

# Melatih model dengan data latihan
history11 <- model11 %>% fit(
  x = train_reshaped1,  # Data latihan yang telah diubah bentuk
  y = train_price1,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 32,  
  shuffle = FALSE
)

# Visualisasi loss selama pelatihan
plot(history11)

# Menggunakan model untuk prediksi pada data uji 
test_reshaped11 <- array_reshape(test_matrix1, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred11 <- model11 %>% predict(test_reshaped11)

mse11 = mean(test_price1 - pred11)^2
mse11

rmse11 = sqrt((mean(test_price1 - pred11)^2))
rmse11

mape11 = (mean((abs(test_price1 - pred11))/(abs(test_price1))))*100 
mape11

# Menggunakan model untuk prediksi pada data pelatihan 
train_reshaped11 <- array_reshape(train_matrix1, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred11_train <- model11 %>% predict(train_reshaped11)

mse11_train = mean(train_price1 - pred11_train)^2
mse11_train

rmse11_train = sqrt((mean(train_price1 - pred11_train)^2))
rmse11_train

mape11_train = (mean((abs(train_price1 - pred11_train))/(abs(train_price1))))*100 
mape11_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 30
# BATCH SIZE = 32

model12 <- keras_model_sequential() %>%
  layer_lstm(units = 30, activation = 'tanh', input_shape = c(4, 1))%>%
  layer_dense(units = 1, activation = 'leaky_relu')

model12 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history12 <- model12 %>% fit(
  x = train_reshaped1,  # Data latihan yang telah diubah bentuk
  y = train_price1,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 32,  
  shuffle = FALSE
)
plot(history12)

test_reshaped12 <- array_reshape(test_matrix1, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred12 <- model12 %>% predict(test_reshaped12)

mse12 = mean(test_price1 - pred12)^2
mse12

rmse12 = sqrt((mean(test_price1 - pred12)^2))
rmse12

mape12 = (mean((abs(test_price1 - pred12))/(abs(test_price1))))*100 
mape12

train_reshaped12 <- array_reshape(train_matrix1, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred12_train <- model12 %>% predict(train_reshaped12)

mse12_train = mean(train_price1 - pred12_train)^2
mse12_train

rmse12_train = sqrt((mean(train_price1 - pred12_train)^2))
rmse12_train

mape12_train = (mean((abs(train_price1 - pred12_train))/(abs(train_price1))))*100 
mape12_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 40
# BATCH SIZE = 32

model13 <- keras_model_sequential() %>%
  layer_lstm(units = 40, activation = 'tanh', input_shape = c(4, 1))%>%
  layer_dense(units = 1, activation = 'leaky_relu')

model13 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history13 <- model13 %>% fit(
  x = train_reshaped1,  # Data latihan yang telah diubah bentuk
  y = train_price1,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 32,  
  shuffle = FALSE
)
plot(history13)

test_reshaped13 <- array_reshape(test_matrix1, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred13 <- model13 %>% predict(test_reshaped13)

mse13 = mean(test_price1 - pred13)^2
mse13

rmse13 = sqrt((mean(test_price1 - pred13)^2))
rmse13

mape13 = (mean((abs(test_price1 - pred13))/(abs(test_price1))))*100 
mape13

train_reshaped13 <- array_reshape(train_matrix1, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred13_train <- model13 %>% predict(train_reshaped13)

mse13_train = mean(train_price1 - pred13_train)^2
mse13_train

rmse13_train = sqrt((mean(train_price1 - pred13_train)^2))
rmse13_train

mape13_train = (mean((abs(train_price1 - pred13_train))/(abs(train_price1))))*100 
mape13_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 50
# BATCH SIZE = 32

model14 <- keras_model_sequential() %>%
  layer_lstm(units = 50, activation = 'tanh', input_shape = c(4, 1))%>%
  layer_dense(units = 1, activation = 'leaky_relu')

model14 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history14 <- model14 %>% fit(
  x = train_reshaped1,  # Data latihan yang telah diubah bentuk
  y = train_price1,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 32,  
  shuffle = FALSE
)
plot(history14)

test_reshaped14 <- array_reshape(test_matrix1, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred14 <- model14 %>% predict(test_reshaped14)

mse14 = mean(test_price1 - pred14)^2
mse14

rmse14 = sqrt((mean(test_price1 - pred14)^2))
rmse14

mape14 = (mean((abs(test_price1 - pred14))/(abs(test_price1))))*100 
mape14

train_reshaped14 <- array_reshape(train_matrix1, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred14_train <- model14 %>% predict(train_reshaped14)

mse14_train = mean(train_price1 - pred14_train)^2
mse14_train

rmse14_train = sqrt((mean(train_price1 - pred14_train)^2))
rmse14_train

mape14_train = (mean((abs(train_price1 - pred14_train))/(abs(train_price1))))*100 
mape14_train

#############################################################################################
#############################################################################################

# LSTM LAYER = 1 
# NEURONS = 20
# BATCH SIZE = 64

model15 <- keras_model_sequential() %>%
  layer_lstm(units = 20, activation = 'tanh', input_shape = c(4, 1))%>%
  layer_dense(units = 1, activation = 'leaky_relu')

model15 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history15 <- model15 %>% fit(
  x = train_reshaped1,  # Data latihan yang telah diubah bentuk
  y = train_price1,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 64,  
  shuffle = FALSE
)
plot(history15)

test_reshaped15 <- array_reshape(test_matrix1, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred15 <- model15 %>% predict(test_reshaped15)

mse15 = mean(test_price1 - pred15)^2
mse15

rmse15 = sqrt((mean(test_price1 - pred15)^2))
rmse15

mape15 = (mean((abs(test_price1 - pred15))/(abs(test_price1))))*100 
mape15

train_reshaped15 <- array_reshape(train_matrix1, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred15_train <- model15 %>% predict(train_reshaped15)

mse15_train = mean(train_price1 - pred15_train)^2
mse15_train

rmse15_train = sqrt((mean(train_price1 - pred15_train)^2))
rmse15_train

mape15_train = (mean((abs(train_price1 - pred15_train))/(abs(train_price1))))*100 
mape15_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 30
# BATCH SIZE = 64

model16 <- keras_model_sequential() %>%
  layer_lstm(units = 30, activation = 'tanh', input_shape = c(4, 1))%>%
  layer_dense(units = 1, activation = 'leaky_relu')

model16 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history16 <- model16 %>% fit(
  x = train_reshaped1,  # Data latihan yang telah diubah bentuk
  y = train_price1,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 64,  
  shuffle = FALSE
)
plot(history16)

test_reshaped16 <- array_reshape(test_matrix1, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred16 <- model16 %>% predict(test_reshaped16)

mse16 = mean(test_price1 - pred16)^2
mse16

rmse16 = sqrt((mean(test_price1 - pred16)^2))
rmse16

mape16 = (mean((abs(test_price1 - pred16))/(abs(test_price1))))*100 
mape16

train_reshaped16 <- array_reshape(train_matrix1, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred16_train <- model16 %>% predict(train_reshaped16)

mse16_train = mean(train_price1 - pred16_train)^2
mse16_train

rmse16_train = sqrt((mean(train_price1 - pred16_train)^2))
rmse16_train

mape16_train = (mean((abs(train_price1 - pred16_train))/(abs(train_price1))))*100 
mape16_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 40
# BATCH SIZE = 64

model17 <- keras_model_sequential() %>%
  layer_lstm(units = 40, activation = 'tanh', input_shape = c(4, 1))%>%
  layer_dense(units = 1, activation = 'leaky_relu')

model17 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history17 <- model17 %>% fit(
  x = train_reshaped1,  # Data latihan yang telah diubah bentuk
  y = train_price1,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 64,  
  shuffle = FALSE
)
plot(history17)

test_reshaped17 <- array_reshape(test_matrix1, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred17 <- model17 %>% predict(test_reshaped17)

mse17 = mean(test_price1 - pred17)^2
mse17

rmse17 = sqrt((mean(test_price1 - pred17)^2))
rmse17

mape17 = (mean((abs(test_price1 - pred17))/(abs(test_price1))))*100 
mape17

train_reshaped17 <- array_reshape(train_matrix1, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred17_train <- model17 %>% predict(train_reshaped17)

mse17_train = mean(train_price1 - pred17_train)^2
mse17_train

rmse17_train = sqrt((mean(train_price1 - pred17_train)^2))
rmse17_train

mape17_train = (mean((abs(train_price1 - pred17_train))/(abs(train_price1))))*100 
mape17_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 50
# BATCH SIZE = 64

model18 <- keras_model_sequential() %>%
  layer_lstm(units = 50, activation = 'tanh', input_shape = c(4, 1))%>%
  layer_dense(units = 1, activation = 'leaky_relu')

model18 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history18 <- model18 %>% fit(
  x = train_reshaped1,  # Data latihan yang telah diubah bentuk
  y = train_price1,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 64,  
  shuffle = FALSE
)
plot(history18)

test_reshaped18 <- array_reshape(test_matrix1, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred18 <- model18 %>% predict(test_reshaped18)

mse18 = mean(test_price1 - pred18)^2
mse18

rmse18 = sqrt((mean(test_price1 - pred18)^2))
rmse18

mape18 = (mean((abs(test_price1 - pred18))/(abs(test_price1))))*100 
mape18

train_reshaped18 <- array_reshape(train_matrix1, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred18_train <- model18 %>% predict(train_reshaped18)

mse18_train = mean(train_price1 - pred18_train)^2
mse18_train

rmse18_train = sqrt((mean(train_price1 - pred18_train)^2))
rmse18_train

mape18_train = (mean((abs(train_price1 - pred18_train))/(abs(train_price1))))*100 
mape18_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 20
# BATCH SIZE = 96

model19 <- keras_model_sequential() %>%
  layer_lstm(units = 20, activation = 'tanh', input_shape = c(4, 1))%>%
  layer_dense(units = 1, activation = 'leaky_relu')

model19 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history19 <- model19 %>% fit(
  x = train_reshaped1,  # Data latihan yang telah diubah bentuk
  y = train_price1,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 96,  
  shuffle = FALSE
)
plot(history19)

test_reshaped19 <- array_reshape(test_matrix1, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred19 <- model19 %>% predict(test_reshaped19)

mse19 = mean(test_price1 - pred19)^2
mse19

rmse19 = sqrt((mean(test_price1 - pred19)^2))
rmse19

mape19 = (mean((abs(test_price1 - pred19))/(abs(test_price1))))*100 
mape19

train_reshaped19 <- array_reshape(train_matrix1, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred19_train <- model19 %>% predict(train_reshaped19)

mse19_train = mean(train_price1 - pred19_train)^2
mse19_train

rmse19_train = sqrt((mean(train_price1 - pred19_train)^2))
rmse19_train

mape19_train = (mean((abs(train_price1 - pred19_train))/(abs(train_price1))))*100 
mape19_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 30
# BATCH SIZE = 96

model110 <- keras_model_sequential() %>%
  layer_lstm(units = 30, activation = 'tanh', input_shape = c(4, 1))%>%
  layer_dense(units = 1, activation = 'leaky_relu')

model110 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history110 <- model110 %>% fit(
  x = train_reshaped1,  # Data latihan yang telah diubah bentuk
  y = train_price1,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 96,  
  shuffle = FALSE
)
plot(history110)

test_reshaped110 <- array_reshape(test_matrix1, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred110 <- model110 %>% predict(test_reshaped110)

mse110 = mean(test_price1 - pred110)^2
mse110

rmse110 = sqrt((mean(test_price1 - pred110)^2))
rmse110

mape110 = (mean((abs(test_price1 - pred110))/(abs(test_price1))))*100 
mape110

train_reshaped110 <- array_reshape(train_matrix1, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred110_train <- model110 %>% predict(train_reshaped110)

mse110_train = mean(train_price1 - pred110_train)^2
mse110_train

rmse110_train = sqrt((mean(train_price1 - pred110_train)^2))
rmse110_train

mape110_train = (mean((abs(train_price1 - pred110_train))/(abs(train_price1))))*100 
mape110_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 40
# BATCH SIZE = 96

model111 <- keras_model_sequential() %>%
  layer_lstm(units = 40, activation = 'tanh', input_shape = c(4, 1))%>%
  layer_dense(units = 1, activation = 'leaky_relu')

model111 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history111 <- model111 %>% fit(
  x = train_reshaped1,  # Data latihan yang telah diubah bentuk
  y = train_price1,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 96,  
  shuffle = FALSE
)
plot(history111)

test_reshaped111 <- array_reshape(test_matrix1, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred111 <- model111 %>% predict(test_reshaped111)

mse111 = mean(test_price1 - pred111)^2
mse111

rmse111 = sqrt((mean(test_price1 - pred111)^2))
rmse111

mape111 = (mean((abs(test_price1 - pred111))/(abs(test_price1))))*100 
mape111

train_reshaped111 <- array_reshape(train_matrix1, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred111_train <- model111 %>% predict(train_reshaped111)

mse111_train = mean(train_price1 - pred111_train)^2
mse111_train

rmse111_train = sqrt((mean(train_price1 - pred111_train)^2))
rmse111_train

mape111_train = (mean((abs(train_price1 - pred111_train))/(abs(train_price1))))*100 
mape111_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 50
# BATCH SIZE = 96

model112 <- keras_model_sequential() %>%
  layer_lstm(units = 50, activation = 'tanh', input_shape = c(4, 1))%>%
  layer_dense(units = 1, activation = 'leaky_relu')

model112 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history112 <- model112 %>% fit(
  x = train_reshaped1,  # Data latihan yang telah diubah bentuk
  y = train_price1,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 96,  
  shuffle = FALSE
)
plot(history112)

test_reshaped112 <- array_reshape(test_matrix1, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred112 <- model112 %>% predict(test_reshaped112)

mse112 = mean(test_price1 - pred112)^2
mse112

rmse112 = sqrt((mean(test_price1 - pred112)^2))
rmse112

mape112 = (mean((abs(test_price1 - pred112))/(abs(test_price1))))*100 
mape112

train_reshaped112 <- array_reshape(train_matrix1, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred112_train <- model112 %>% predict(train_reshaped112)

mse112_train = mean(train_price1 - pred112_train)^2
mse112_train

rmse112_train = sqrt((mean(train_price1 - pred112_train)^2))
rmse112_train

mape112_train = (mean((abs(train_price1 - pred112_train))/(abs(train_price1))))*100 
mape112_train

#############################################################################################
#############################################################################################
# LSTM LAYER = 2
# NEURONS = 20
# BATCH SIZE = 32

model113 <- keras_model_sequential() %>%
  layer_lstm(units = 20, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 20, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model113 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history113 <- model113 %>% fit(
  x = train_reshaped1,  # Data latihan yang telah diubah bentuk
  y = train_price1,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 32,  
  shuffle = FALSE
)
plot(history113)

test_reshaped113 <- array_reshape(test_matrix1, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred113 <- model113 %>% predict(test_reshaped113)

mse113 = mean(test_price1 - pred113)^2
mse113

rmse113 = sqrt((mean(test_price1 - pred113)^2))
rmse113

mape113 = (mean((abs(test_price1 - pred113))/(abs(test_price1))))*100 
mape113

train_reshaped113 <- array_reshape(train_matrix1, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred113_train <- model113 %>% predict(train_reshaped113)

mse113_train = mean(train_price1 - pred113_train)^2
mse113_train

rmse113_train = sqrt((mean(train_price1 - pred113_train)^2))
rmse113_train

mape113_train = (mean((abs(train_price1 - pred113_train))/(abs(train_price1))))*100 
mape113_train

#############################################################################################
# LSTM LAYER = 2
# NEURONS = 30
# BATCH SIZE = 32

model114 <- keras_model_sequential() %>%
  layer_lstm(units = 30, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 30, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model114 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history114 <- model114 %>% fit(
  x = train_reshaped1,  # Data latihan yang telah diubah bentuk
  y = train_price1,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 32,  
  shuffle = FALSE
)
plot(history114)

test_reshaped114 <- array_reshape(test_matrix1, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred114 <- model114 %>% predict(test_reshaped114)

mse114 = mean(test_price1 - pred114)^2
mse114

rmse114 = sqrt((mean(test_price1 - pred114)^2))
rmse114

mape114 = (mean((abs(test_price1 - pred114))/(abs(test_price1))))*100 
mape114

train_reshaped114 <- array_reshape(train_matrix1, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred114_train <- model114 %>% predict(train_reshaped114)

mse114_train = mean(train_price1 - pred114_train)^2
mse114_train

rmse114_train = sqrt((mean(train_price1 - pred114_train)^2))
rmse114_train

mape114_train = (mean((abs(train_price1 - pred114_train))/(abs(train_price1))))*100 
mape114_train

#############################################################################################
# LSTM LAYER = 2
# NEURONS = 40
# BATCH SIZE = 32

model115 <- keras_model_sequential() %>%
  layer_lstm(units = 40, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 40, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model115 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history115 <- model115 %>% fit(
  x = train_reshaped1,  # Data latihan yang telah diubah bentuk
  y = train_price1,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 32,  
  shuffle = FALSE
)
plot(history115)

test_reshaped115 <- array_reshape(test_matrix1, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred115 <- model115 %>% predict(test_reshaped115)

mse115 = mean(test_price1 - pred115)^2
mse115

rmse115 = sqrt((mean(test_price1 - pred115)^2))
rmse115

mape115 = (mean((abs(test_price1 - pred115))/(abs(test_price1))))*100 
mape115

train_reshaped115 <- array_reshape(train_matrix1, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred115_train <- model115 %>% predict(train_reshaped115)

mse115_train = mean(train_price1 - pred115_train)^2
mse115_train

rmse115_train = sqrt((mean(train_price1 - pred115_train)^2))
rmse115_train

mape115_train = (mean((abs(train_price1 - pred115_train))/(abs(train_price1))))*100 
mape115_train

#############################################################################################
# LSTM LAYER = 2
# NEURONS = 50
# BATCH SIZE = 32

model116 <- keras_model_sequential() %>%
  layer_lstm(units = 50, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 50, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model116 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history116 <- model116 %>% fit(
  x = train_reshaped1,  # Data latihan yang telah diubah bentuk
  y = train_price1,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 32,  
  shuffle = FALSE
)
plot(history116)

test_reshaped116 <- array_reshape(test_matrix1, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred116 <- model116 %>% predict(test_reshaped116)

mse116 = mean(test_price1 - pred116)^2
mse116

rmse116 = sqrt((mean(test_price1 - pred116)^2))
rmse116

mape116 = (mean((abs(test_price1 - pred116))/(abs(test_price1))))*100 
mape116

train_reshaped116 <- array_reshape(train_matrix1, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred116_train <- model116 %>% predict(train_reshaped116)

mse116_train = mean(train_price1 - pred116_train)^2
mse116_train

rmse116_train = sqrt((mean(train_price1 - pred116_train)^2))
rmse116_train

mape116_train = (mean((abs(train_price1 - pred116_train))/(abs(train_price1))))*100 
mape116_train

#############################################################################################
# LSTM LAYER = 2
# NEURONS = 20
# BATCH SIZE = 64

model117 <- keras_model_sequential() %>%
  layer_lstm(units = 20, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 20, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model117 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history117 <- model117 %>% fit(
  x = train_reshaped1,  # Data latihan yang telah diubah bentuk
  y = train_price1,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 64,  
  shuffle = FALSE
)
plot(history117)

test_reshaped117 <- array_reshape(test_matrix1, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred117 <- model117 %>% predict(test_reshaped117)

mse117 = mean(test_price1 - pred117)^2
mse117

rmse117 = sqrt((mean(test_price1 - pred117)^2))
rmse117

mape117 = (mean((abs(test_price1 - pred117))/(abs(test_price1))))*100 
mape117

train_reshaped117 <- array_reshape(train_matrix1, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred117_train <- model117 %>% predict(train_reshaped117)

mse117_train = mean(train_price1 - pred117_train)^2
mse117_train

rmse117_train = sqrt((mean(train_price1 - pred117_train)^2))
rmse117_train

mape117_train = (mean((abs(train_price1 - pred117_train))/(abs(train_price1))))*100 
mape117_train

#############################################################################################
# LSTM LAYER = 2
# NEURONS = 30
# BATCH SIZE = 64

model118 <- keras_model_sequential() %>%
  layer_lstm(units = 30, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 30, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model118 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history118 <- model118 %>% fit(
  x = train_reshaped1,  # Data latihan yang telah diubah bentuk
  y = train_price1,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 64,  
  shuffle = FALSE
)
plot(history118)

test_reshaped118 <- array_reshape(test_matrix1, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred118 <- model118 %>% predict(test_reshaped118)

mse118 = mean(test_price1 - pred118)^2
mse118

rmse118 = sqrt((mean(test_price1 - pred118)^2))
rmse118

mape118 = (mean((abs(test_price1 - pred118))/(abs(test_price1))))*100 
mape118

train_reshaped118 <- array_reshape(train_matrix1, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred118_train <- model118 %>% predict(train_reshaped118)

mse118_train = mean(train_price1 - pred118_train)^2
mse118_train

rmse118_train = sqrt((mean(train_price1 - pred118_train)^2))
rmse118_train

mape118_train = (mean((abs(train_price1 - pred118_train))/(abs(train_price1))))*100 
mape118_train

#############################################################################################
# LSTM LAYER = 2
# NEURONS = 40
# BATCH SIZE = 64

model119 <- keras_model_sequential() %>%
  layer_lstm(units = 40, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 40, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model119 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history119 <- model119 %>% fit(
  x = train_reshaped1,  # Data latihan yang telah diubah bentuk
  y = train_price1,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 64,  
  shuffle = FALSE
)
plot(history119)

test_reshaped119 <- array_reshape(test_matrix1, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred119 <- model119 %>% predict(test_reshaped119)

mse119 = mean(test_price1 - pred119)^2
mse119

rmse119 = sqrt((mean(test_price1 - pred119)^2))
rmse119

mape119 = (mean((abs(test_price1 - pred119))/(abs(test_price1))))*100 
mape119

train_reshaped119 <- array_reshape(train_matrix1, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred119_train <- model119 %>% predict(train_reshaped119)

mse119_train = mean(train_price1 - pred119_train)^2
mse119_train

rmse119_train = sqrt((mean(train_price1 - pred119_train)^2))
rmse119_train

mape119_train = (mean((abs(train_price1 - pred119_train))/(abs(train_price1))))*100 
mape119_train

#############################################################################################
# LSTM LAYER = 2
# NEURONS = 50
# BATCH SIZE = 64

model120 <- keras_model_sequential() %>%
  layer_lstm(units = 50, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 50, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model120 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history120 <- model120 %>% fit(
  x = train_reshaped1,  # Data latihan yang telah diubah bentuk
  y = train_price1,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 64,  
  shuffle = FALSE
)
plot(history120)

test_reshaped120 <- array_reshape(test_matrix1, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred120 <- model120 %>% predict(test_reshaped120)

mse120 = mean(test_price1 - pred120)^2
mse120

rmse120 = sqrt((mean(test_price1 - pred120)^2))
rmse120

mape120 = (mean((abs(test_price1 - pred120))/(abs(test_price1))))*100 
mape120

train_reshaped120 <- array_reshape(train_matrix1, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred120_train <- model120 %>% predict(train_reshaped120)

mse120_train = mean(train_price1 - pred120_train)^2
mse120_train

rmse120_train = sqrt((mean(train_price1 - pred120_train)^2))
rmse120_train

mape120_train = (mean((abs(train_price1 - pred120_train))/(abs(train_price1))))*100 
mape120_train

#############################################################################################
# LSTM LAYER = 2
# NEURONS = 20
# BATCH SIZE = 96

model121 <- keras_model_sequential() %>%
  layer_lstm(units = 20, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 20, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model121 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history121 <- model121 %>% fit(
  x = train_reshaped1,  # Data latihan yang telah diubah bentuk
  y = train_price1,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 96,  
  shuffle = FALSE
)
plot(history121)

test_reshaped121 <- array_reshape(test_matrix1, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred121 <- model121 %>% predict(test_reshaped121)

mse121 = mean(test_price1 - pred121)^2
mse121

rmse121 = sqrt((mean(test_price1 - pred121)^2))
rmse121

mape121 = (mean((abs(test_price1 - pred121))/(abs(test_price1))))*100 
mape121

train_reshaped121 <- array_reshape(train_matrix1, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred121_train <- model121 %>% predict(train_reshaped121)

mse121_train = mean(train_price1 - pred121_train)^2
mse121_train

rmse121_train = sqrt((mean(train_price1 - pred121_train)^2))
rmse121_train

mape121_train = (mean((abs(train_price1 - pred121_train))/(abs(train_price1))))*100 
mape121_train

#############################################################################################
# LSTM LAYER = 2
# NEURONS = 30
# BATCH SIZE = 96

model122 <- keras_model_sequential() %>%
  layer_lstm(units = 30, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 30, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model122 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history122 <- model122 %>% fit(
  x = train_reshaped1,  # Data latihan yang telah diubah bentuk
  y = train_price1,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 96,  
  shuffle = FALSE
)
plot(history122)

test_reshaped122 <- array_reshape(test_matrix1, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred122 <- model122 %>% predict(test_reshaped122)

mse122 = mean(test_price1 - pred122)^2
mse122

rmse122 = sqrt((mean(test_price1 - pred122)^2))
rmse122

mape122 = (mean((abs(test_price1 - pred122))/(abs(test_price1))))*100 
mape122

train_reshaped122 <- array_reshape(train_matrix1, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred122_train <- model122 %>% predict(train_reshaped122)

mse122_train = mean(train_price1 - pred122_train)^2
mse122_train

rmse122_train = sqrt((mean(train_price1 - pred122_train)^2))
rmse122_train

mape122_train = (mean((abs(train_price1 - pred122_train))/(abs(train_price1))))*100 
mape122_train

#############################################################################################
# LSTM LAYER = 2
# NEURONS = 40
# BATCH SIZE = 96

model123 <- keras_model_sequential() %>%
  layer_lstm(units = 40, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 40, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model123 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history123 <- model123 %>% fit(
  x = train_reshaped1,  # Data latihan yang telah diubah bentuk
  y = train_price1,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 96,  
  shuffle = FALSE
)
plot(history123)

test_reshaped123 <- array_reshape(test_matrix1, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred123 <- model123 %>% predict(test_reshaped123)

mse123 = mean(test_price1 - pred123)^2
mse123

rmse123 = sqrt((mean(test_price1 - pred123)^2))
rmse123

mape123 = (mean((abs(test_price1 - pred123))/(abs(test_price1))))*100 
mape123

train_reshaped123 <- array_reshape(train_matrix1, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred123_train <- model123 %>% predict(train_reshaped123)

mse123_train = mean(train_price1 - pred123_train)^2
mse123_train

rmse123_train = sqrt((mean(train_price1 - pred123_train)^2))
rmse123_train

mape123_train = (mean((abs(train_price1 - pred123_train))/(abs(train_price1))))*100 
mape123_train

#############################################################################################
# LSTM LAYER = 2
# NEURONS = 50
# BATCH SIZE = 96

model124 <- keras_model_sequential() %>%
  layer_lstm(units = 50, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 50, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model124 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history124 <- model124 %>% fit(
  x = train_reshaped1,  # Data latihan yang telah diubah bentuk
  y = train_price1,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 96,  
  shuffle = FALSE
)
plot(history124)

test_reshaped124 <- array_reshape(test_matrix1, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred124 <- model124 %>% predict(test_reshaped124)

mse124 = mean(test_price1 - pred124)^2
mse124

rmse124 = sqrt((mean(test_price1 - pred124)^2))
rmse124

mape124 = (mean((abs(test_price1 - pred124))/(abs(test_price1))))*100 
mape124

train_reshaped124 <- array_reshape(train_matrix1, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred124_train <- model124 %>% predict(train_reshaped124)

mse124_train = mean(train_price1 - pred124_train)^2
mse124_train

rmse124_train = sqrt((mean(train_price1 - pred124_train)^2))
rmse124_train

mape124_train = (mean((abs(train_price1 - pred124_train))/(abs(train_price1))))*100 
mape124_train

#############################################################################################
mse_test1 <- c(mse11, mse12, mse13, mse14, mse15, mse16, mse17, mse18, mse19, mse110,
               mse111, mse112, mse113, mse114, mse115, mse116, mse117, mse118, mse119, mse120,
               mse121, mse122, mse123, mse124)
print(mse_test1)

rmse_test1 <- c(rmse11, rmse12, rmse13, rmse14, rmse15, rmse16, rmse17, rmse18, rmse19, rmse110,
                rmse111, rmse112, rmse113, rmse114, rmse115, rmse116, rmse117, rmse118, rmse119, rmse120,
                rmse121, rmse122, rmse123, rmse124)
print(rmse_test1)

mape_test1 <- c(mape11, mape12, mape13, mape14, mape15, mape16, mape17, mape18, mape19, mape110,
                mape111, mape112, mape113, mape114, mape115, mape116, mape117, mape118, mape119, mape120,
                mape121, mape122, mape123, mape124)
print(mape_test1)

mse_train1 <- c(mse11_train, mse12_train, mse13_train, mse14_train, mse15_train, mse16_train,
                mse17_train, mse18_train, mse19_train, mse110_train, mse111_train, mse112_train,
                mse113_train, mse114_train, mse115_train, mse116_train, mse117_train, mse118_train, 
                mse119_train, mse120_train, mse121_train, mse122_train, mse123_train, mse124_train)
print(mse_train1)

rmse_train1 <- c(rmse11_train, rmse12_train, rmse13_train, rmse14_train, rmse15_train, rmse16_train,
                 rmse17_train, rmse18_train, rmse19_train, rmse110_train, rmse111_train, rmse112_train,
                 rmse113_train, rmse114_train, rmse115_train, rmse116_train, rmse117_train, rmse118_train, 
                 rmse119_train, rmse120_train, rmse121_train, rmse122_train, rmse123_train, rmse124_train)
print(rmse_train1)

mape_train1 <- c(mape11_train, mape12_train, mape13_train, mape14_train, mape15_train, mape16_train,
                 mape17_train, mape18_train, mape19_train, mape110_train, mape111_train, mape112_train,
                 mape113_train, mape114_train, mape115_train, mape116_train, mape117_train, mape118_train, 
                 mape119_train, mape120_train, mape121_train, mape122_train, mape123_train, mape124_train)
print(mape_train1)

no_model1 <- c(1, 2, 3, 4, 5, 6, 7, 8, 9 ,10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)

hasil_kombinasi1 = data.frame(no_model1, mse_test1, rmse_test1, mape_test1, 
                              mse_train1, rmse_train1, mape_train1)
min_mape_test1 = min(mape_test1)
print(min_mape_test1) 

#############################################################################################
#############################################################################################
#############################################################################################
# untuk set 2
dataset2[] <- lapply(dataset2, as.numeric)
str(dataset2)

x2 = data.frame(dataset2$set2.tanggal, dataset2$set2.bulan, dataset2$set2.tahun, 
                dataset2$set2.returndata)

baris1_set3 = head(dataset3$set3.Price,1)
price2 = c(dataset2$set2.Price, baris1_set3)
price2 = tail(price2,516)
y2 = data.frame(price2)

matrix_data2 <- as.matrix(x2)
dv2 <- matrix_data2

# Pembagian data train dan test
n_rows2 <- nrow(dv2)

train_size2 <- 491
#train_index2 = sample(seq_len(nrow(x2)), size = 491, replace = FALSE)
#train2 = x2[train_index2, ]
#train_price2 = y2[train_index2,]

#test2 = x2[-train_index2,]
#test_price2 = y2[-train_index2,]

# Membuat indeks sekuensial untuk pemilihan baris
sequential_indices2 <- seq_len(n_rows2)

# Membagi matriks menjadi set pelatihan dan pengujian
train_matrix2 <- dv2[sequential_indices2[1:train_size2], ]
test_matrix2 <- dv2[sequential_indices2[(train_size2 + 1):n_rows2], ]
train_price2 = y2[sequential_indices2[1:train_size2], ]
test_price2 = y2[sequential_indices2[(train_size2 + 1):n_rows2], ]

# Tampilkan dimensi dari set pelatihan dan pengujian
cat("Dimensi Set Pelatihan:", dim(train_matrix2), "\n")
cat("Dimensi Set Pengujian:", dim(test_matrix2), "\n")

train_reshaped2 = array_reshape(train_matrix2, c(491, 4, 1))
test_reshaped2 = array_reshape(test_matrix2,c(25, 4, 1))

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 20
# BATCH SIZE = 32

model21 <- keras_model_sequential() %>%
  layer_lstm(units = 20, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model21 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history21 <- model21 %>% fit(
  x = train_reshaped2,  # Data latihan yang telah diubah bentuk
  y = train_price2,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 32,  
  shuffle = FALSE
)

plot(history21)

test_reshaped21 <- array_reshape(test_matrix2, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred21 <- model21 %>% predict(test_reshaped21)

mse21 = mean(test_price2 - pred21)^2
mse21

rmse21 = sqrt((mean(test_price2 - pred21)^2))
rmse21

mape21 = (mean((abs(test_price2 - pred21))/(abs(test_price2))))*100 
mape21

train_reshaped21 <- array_reshape(train_matrix2, c(491, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred21_train <- model21 %>% predict(train_reshaped21)

mse21_train = mean(train_price2 - pred21_train)^2
mse21_train

rmse21_train = sqrt((mean(train_price2 - pred21_train)^2))
rmse21_train

mape21_train = (mean((abs(train_price2 - pred21_train))/(abs(train_price2))))*100 
mape21_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 30
# BATCH SIZE = 32

model22 <- keras_model_sequential() %>%
  layer_lstm(units = 30, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model22 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history22 <- model22 %>% fit(
  x = train_reshaped2,  # Data latihan yang telah diubah bentuk
  y = train_price2,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 32,  
  shuffle = FALSE
)

plot(history22)

test_reshaped22 <- array_reshape(test_matrix2, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred22 <- model22 %>% predict(test_reshaped22)

mse22 = mean(test_price2 - pred22)^2
mse22

rmse22 = sqrt((mean(test_price2 - pred22)^2))
rmse22

mape22 = (mean((abs(test_price2 - pred22))/(abs(test_price2))))*100 
mape22

train_reshaped22 <- array_reshape(train_matrix2, c(491, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred22_train <- model22 %>% predict(train_reshaped22)

mse22_train = mean(train_price2 - pred22_train)^2
mse22_train

rmse22_train = sqrt((mean(train_price2 - pred22_train)^2))
rmse22_train

mape22_train = (mean((abs(train_price2 - pred22_train))/(abs(train_price2))))*100 
mape22_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 40
# BATCH SIZE = 32

model23 <- keras_model_sequential() %>%
  layer_lstm(units = 40, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model23 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history23 <- model23 %>% fit(
  x = train_reshaped2,  # Data latihan yang telah diubah bentuk
  y = train_price2,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 32,  
  shuffle = FALSE
)

plot(history23)

test_reshaped23 <- array_reshape(test_matrix2, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred23 <- model23 %>% predict(test_reshaped23)

mse23 = mean(test_price2 - pred23)^2
mse23

rmse23 = sqrt((mean(test_price2 - pred23)^2))
rmse23

mape23 = (mean((abs(test_price2 - pred23))/(abs(test_price2))))*100 
mape23

train_reshaped23 <- array_reshape(train_matrix2, c(491, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred23_train <- model23 %>% predict(train_reshaped23)

mse23_train = mean(train_price2 - pred23_train)^2
mse23_train

rmse23_train = sqrt((mean(train_price2 - pred23_train)^2))
rmse23_train

mape23_train = (mean((abs(train_price2 - pred23_train))/(abs(train_price2))))*100 
mape23_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 50
# BATCH SIZE = 32

model24 <- keras_model_sequential() %>%
  layer_lstm(units = 50, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model24 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history24 <- model24 %>% fit(
  x = train_reshaped2,  # Data latihan yang telah diubah bentuk
  y = train_price2,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 32,  
  shuffle = FALSE
)

plot(history24)

test_reshaped24 <- array_reshape(test_matrix2, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred24 <- model24 %>% predict(test_reshaped24)

mse24 = mean(test_price2 - pred24)^2
mse24

rmse24 = sqrt((mean(test_price2 - pred24)^2))
rmse24

mape24 = (mean((abs(test_price2 - pred24))/(abs(test_price2))))*100 
mape24

train_reshaped24 <- array_reshape(train_matrix2, c(491, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred24_train <- model24 %>% predict(train_reshaped24)

mse24_train = mean(train_price2 - pred24_train)^2
mse24_train

rmse24_train = sqrt((mean(train_price2 - pred24_train)^2))
rmse24_train

mape24_train = (mean((abs(train_price2 - pred24_train))/(abs(train_price2))))*100 
mape24_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 20
# BATCH SIZE = 64

model25 <- keras_model_sequential() %>%
  layer_lstm(units = 20, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model25 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history25 <- model25 %>% fit(
  x = train_reshaped2,  # Data latihan yang telah diubah bentuk
  y = train_price2,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 64,  
  shuffle = FALSE
)

plot(history25)

test_reshaped25 <- array_reshape(test_matrix2, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred25 <- model25 %>% predict(test_reshaped25)

mse25 = mean(test_price2 - pred25)^2
mse25

rmse25 = sqrt((mean(test_price2 - pred25)^2))
rmse25

mape25 = (mean((abs(test_price2 - pred25))/(abs(test_price2))))*100 
mape25

train_reshaped25 <- array_reshape(train_matrix2, c(491, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred25_train <- model25 %>% predict(train_reshaped25)

mse25_train = mean(train_price2 - pred25_train)^2
mse25_train

rmse25_train = sqrt((mean(train_price2 - pred25_train)^2))
rmse25_train

mape25_train = (mean((abs(train_price2 - pred25_train))/(abs(train_price2))))*100 
mape25_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 30
# BATCH SIZE = 64

model26 <- keras_model_sequential() %>%
  layer_lstm(units = 30, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model26 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history26 <- model26 %>% fit(
  x = train_reshaped2,  # Data latihan yang telah diubah bentuk
  y = train_price2,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 64,  
  shuffle = FALSE
)

plot(history26)

test_reshaped26 <- array_reshape(test_matrix2, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred26 <- model26 %>% predict(test_reshaped26)

mse26 = mean(test_price2 - pred26)^2
mse26

rmse26 = sqrt((mean(test_price2 - pred26)^2))
rmse26

mape26 = (mean((abs(test_price2 - pred26))/(abs(test_price2))))*100 
mape26

train_reshaped26 <- array_reshape(train_matrix2, c(491, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred26_train <- model26 %>% predict(train_reshaped26)

mse26_train = mean(train_price2 - pred26_train)^2
mse26_train

rmse26_train = sqrt((mean(train_price2 - pred26_train)^2))
rmse26_train

mape26_train = (mean((abs(train_price2 - pred26_train))/(abs(train_price2))))*100 
mape26_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 40
# BATCH SIZE = 64

model27 <- keras_model_sequential() %>%
  layer_lstm(units = 40, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model27 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history27 <- model27 %>% fit(
  x = train_reshaped2,  # Data latihan yang telah diubah bentuk
  y = train_price2,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 64,  
  shuffle = FALSE
)

plot(history27)

test_reshaped27 <- array_reshape(test_matrix2, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred27 <- model27 %>% predict(test_reshaped27)

mse27 = mean(test_price2 - pred27)^2
mse27

rmse27 = sqrt((mean(test_price2 - pred27)^2))
rmse27

mape27 = (mean((abs(test_price2 - pred27))/(abs(test_price2))))*100 
mape27

train_reshaped27 <- array_reshape(train_matrix2, c(491, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred27_train <- model27 %>% predict(train_reshaped27)

mse27_train = mean(train_price2 - pred27_train)^2
mse27_train

rmse27_train = sqrt((mean(train_price2 - pred27_train)^2))
rmse27_train

mape27_train = (mean((abs(train_price2 - pred27_train))/(abs(train_price2))))*100 
mape27_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 50
# BATCH SIZE = 64

model28 <- keras_model_sequential() %>%
  layer_lstm(units = 50, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model28 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history28 <- model28 %>% fit(
  x = train_reshaped2,  # Data latihan yang telah diubah bentuk
  y = train_price2,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 64,  
  shuffle = FALSE
)

plot(history28)

test_reshaped28 <- array_reshape(test_matrix2, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred28 <- model28 %>% predict(test_reshaped28)

mse28 = mean(test_price2 - pred28)^2
mse28

rmse28 = sqrt((mean(test_price2 - pred28)^2))
rmse28

mape28 = (mean((abs(test_price2 - pred28))/(abs(test_price2))))*100 
mape28

train_reshaped28 <- array_reshape(train_matrix2, c(491, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred28_train <- model28 %>% predict(train_reshaped28)

mse28_train = mean(train_price2 - pred28_train)^2
mse28_train

rmse28_train = sqrt((mean(train_price2 - pred28_train)^2))
rmse28_train

mape28_train = (mean((abs(train_price2 - pred28_train))/(abs(train_price2))))*100 
mape28_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 20
# BATCH SIZE = 96

model29 <- keras_model_sequential() %>%
  layer_lstm(units = 20, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model29 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history29 <- model29 %>% fit(
  x = train_reshaped2,  # Data latihan yang telah diubah bentuk
  y = train_price2,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 96,  
  shuffle = FALSE
)

plot(history29)

test_reshaped29 <- array_reshape(test_matrix2, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred29 <- model29 %>% predict(test_reshaped29)

mse29 = mean(test_price2 - pred29)^2
mse29

rmse29 = sqrt((mean(test_price2 - pred29)^2))
rmse29

mape29 = (mean((abs(test_price2 - pred29))/(abs(test_price2))))*100 
mape29

train_reshaped29 <- array_reshape(train_matrix2, c(491, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred29_train <- model29 %>% predict(train_reshaped29)

mse29_train = mean(train_price2 - pred29_train)^2
mse29_train

rmse29_train = sqrt((mean(train_price2 - pred29_train)^2))
rmse29_train

mape29_train = (mean((abs(train_price2 - pred29_train))/(abs(train_price2))))*100 
mape29_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 30
# BATCH SIZE = 96

model210 <- keras_model_sequential() %>%
  layer_lstm(units = 30, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model210 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history210 <- model210 %>% fit(
  x = train_reshaped2,  # Data latihan yang telah diubah bentuk
  y = train_price2,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 96,  
  shuffle = FALSE
)

plot(history210)

test_reshaped210 <- array_reshape(test_matrix2, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred210 <- model210 %>% predict(test_reshaped210)

mse210 = mean(test_price2 - pred210)^2
mse210

rmse210 = sqrt((mean(test_price2 - pred210)^2))
rmse210

mape210 = (mean((abs(test_price2 - pred210))/(abs(test_price2))))*100 
mape210

train_reshaped210 <- array_reshape(train_matrix2, c(491, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred210_train <- model210 %>% predict(train_reshaped210)

mse210_train = mean(train_price2 - pred210_train)^2
mse210_train

rmse210_train = sqrt((mean(train_price2 - pred210_train)^2))
rmse210_train

mape210_train = (mean((abs(train_price2 - pred210_train))/(abs(train_price2))))*100 
mape210_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 40
# BATCH SIZE = 96

model211 <- keras_model_sequential() %>%
  layer_lstm(units = 40, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model211 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history211 <- model211 %>% fit(
  x = train_reshaped2,  # Data latihan yang telah diubah bentuk
  y = train_price2,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 96,  
  shuffle = FALSE
)

plot(history211)

test_reshaped211 <- array_reshape(test_matrix2, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred211 <- model211 %>% predict(test_reshaped211)

mse211 = mean(test_price2 - pred211)^2
mse211

rmse211 = sqrt((mean(test_price2 - pred211)^2))
rmse211

mape211 = (mean((abs(test_price2 - pred211))/(abs(test_price2))))*100 
mape211

train_reshaped211 <- array_reshape(train_matrix2, c(491, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred211_train <- model211 %>% predict(train_reshaped211)

mse211_train = mean(train_price2 - pred211_train)^2
mse211_train

rmse211_train = sqrt((mean(train_price2 - pred211_train)^2))
rmse211_train

mape211_train = (mean((abs(train_price2 - pred211_train))/(abs(train_price2))))*100 
mape211_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 50
# BATCH SIZE = 96

model212 <- keras_model_sequential() %>%
  layer_lstm(units = 50, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model212 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history212 <- model212 %>% fit(
  x = train_reshaped2,  # Data latihan yang telah diubah bentuk
  y = train_price2,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 96,  
  shuffle = FALSE
)

plot(history212)

test_reshaped212 <- array_reshape(test_matrix2, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred212 <- model212 %>% predict(test_reshaped212)

mse212 = mean(test_price2 - pred212)^2
mse212

rmse212 = sqrt((mean(test_price2 - pred212)^2))
rmse212

mape212 = (mean((abs(test_price2 - pred212))/(abs(test_price2))))*100 
mape212

train_reshaped212 <- array_reshape(train_matrix2, c(491, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred212_train <- model212 %>% predict(train_reshaped212)

mse212_train = mean(train_price2 - pred212_train)^2
mse212_train

rmse212_train = sqrt((mean(train_price2 - pred212_train)^2))
rmse212_train

mape212_train = (mean((abs(train_price2 - pred212_train))/(abs(train_price2))))*100 
mape212_train

#############################################################################################
# LSTM LAYER = 2 
# NEURONS = 20
# BATCH SIZE = 32

model213 <- keras_model_sequential() %>%
  layer_lstm(units = 20, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 20, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model213 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history213 <- model213 %>% fit(
  x = train_reshaped2,  # Data latihan yang telah diubah bentuk
  y = train_price2,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 32,  
  shuffle = FALSE
)

plot(history213)

test_reshaped213 <- array_reshape(test_matrix2, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred213 <- model213 %>% predict(test_reshaped213)

mse213 = mean(test_price2 - pred213)^2
mse213

rmse213 = sqrt((mean(test_price2 - pred213)^2))
rmse213

mape213 = (mean((abs(test_price2 - pred213))/(abs(test_price2))))*100 
mape213

train_reshaped213 <- array_reshape(train_matrix2, c(491, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred213_train <- model213 %>% predict(train_reshaped213)

mse213_train = mean(train_price2 - pred213_train)^2
mse213_train

rmse213_train = sqrt((mean(train_price2 - pred213_train)^2))
rmse213_train

mape213_train = (mean((abs(train_price2 - pred213_train))/(abs(train_price2))))*100 
mape213_train

#############################################################################################
# LSTM LAYER = 2 
# NEURONS = 30
# BATCH SIZE = 32

model214 <- keras_model_sequential() %>%
  layer_lstm(units = 30, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 30, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model214 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history214 <- model214 %>% fit(
  x = train_reshaped2,  # Data latihan yang telah diubah bentuk
  y = train_price2,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 32,  
  shuffle = FALSE
)

plot(history214)

test_reshaped214 <- array_reshape(test_matrix2, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred214 <- model214 %>% predict(test_reshaped214)

mse214 = mean(test_price2 - pred214)^2
mse214

rmse214 = sqrt((mean(test_price2 - pred214)^2))
rmse214

mape214 = (mean((abs(test_price2 - pred214))/(abs(test_price2))))*100 
mape214

train_reshaped214 <- array_reshape(train_matrix2, c(491, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred214_train <- model214 %>% predict(train_reshaped214)

mse214_train = mean(train_price2 - pred214_train)^2
mse214_train

rmse214_train = sqrt((mean(train_price2 - pred214_train)^2))
rmse214_train

mape214_train = (mean((abs(train_price2 - pred214_train))/(abs(train_price2))))*100 
mape214_train

#############################################################################################
# LSTM LAYER = 2 
# NEURONS = 40
# BATCH SIZE = 32

model215 <- keras_model_sequential() %>%
  layer_lstm(units = 40, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 40, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model215 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history215 <- model215 %>% fit(
  x = train_reshaped2,  # Data latihan yang telah diubah bentuk
  y = train_price2,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 32,  
  shuffle = FALSE
)

plot(history215)

test_reshaped215 <- array_reshape(test_matrix2, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred215 <- model215 %>% predict(test_reshaped215)

mse215 = mean(test_price2 - pred215)^2
mse215

rmse215 = sqrt((mean(test_price2 - pred215)^2))
rmse215

mape215 = (mean((abs(test_price2 - pred215))/(abs(test_price2))))*100 
mape215

train_reshaped215 <- array_reshape(train_matrix2, c(491, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred215_train <- model215 %>% predict(train_reshaped215)

mse215_train = mean(train_price2 - pred215_train)^2
mse215_train

rmse215_train = sqrt((mean(train_price2 - pred215_train)^2))
rmse215_train

mape215_train = (mean((abs(train_price2 - pred215_train))/(abs(train_price2))))*100 
mape215_train

#############################################################################################
# LSTM LAYER = 2 
# NEURONS = 50
# BATCH SIZE = 32

model216 <- keras_model_sequential() %>%
  layer_lstm(units = 50, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 50, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model216 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history216 <- model216 %>% fit(
  x = train_reshaped2,  # Data latihan yang telah diubah bentuk
  y = train_price2,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 32,  
  shuffle = FALSE
)

plot(history216)

test_reshaped216 <- array_reshape(test_matrix2, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred216 <- model216 %>% predict(test_reshaped216)

mse216 = mean(test_price2 - pred216)^2
mse216

rmse216 = sqrt((mean(test_price2 - pred216)^2))
rmse216

mape216 = (mean((abs(test_price2 - pred216))/(abs(test_price2))))*100 
mape216

train_reshaped216 <- array_reshape(train_matrix2, c(491, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred216_train <- model216 %>% predict(train_reshaped216)

mse216_train = mean(train_price2 - pred216_train)^2
mse216_train

rmse216_train = sqrt((mean(train_price2 - pred216_train)^2))
rmse216_train

mape216_train = (mean((abs(train_price2 - pred216_train))/(abs(train_price2))))*100 
mape216_train

#############################################################################################
# LSTM LAYER = 2 
# NEURONS = 20
# BATCH SIZE = 64

model217 <- keras_model_sequential() %>%
  layer_lstm(units = 20, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 20, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model217 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history217 <- model217 %>% fit(
  x = train_reshaped2,  # Data latihan yang telah diubah bentuk
  y = train_price2,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 64,  
  shuffle = FALSE
)

plot(history217)

test_reshaped217 <- array_reshape(test_matrix2, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred217 <- model217 %>% predict(test_reshaped217)

mse217 = mean(test_price2 - pred217)^2
mse217

rmse217 = sqrt((mean(test_price2 - pred217)^2))
rmse217

mape217 = (mean((abs(test_price2 - pred217))/(abs(test_price2))))*100 
mape217

train_reshaped217 <- array_reshape(train_matrix2, c(491, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred217_train <- model217 %>% predict(train_reshaped217)

mse217_train = mean(train_price2 - pred217_train)^2
mse217_train

rmse217_train = sqrt((mean(train_price2 - pred217_train)^2))
rmse217_train

mape217_train = (mean((abs(train_price2 - pred217_train))/(abs(train_price2))))*100 
mape217_train

#############################################################################################
# LSTM LAYER = 2 
# NEURONS = 30
# BATCH SIZE = 64

model218 <- keras_model_sequential() %>%
  layer_lstm(units = 30, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 30, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model218 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history218 <- model218 %>% fit(
  x = train_reshaped2,  # Data latihan yang telah diubah bentuk
  y = train_price2,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 64,  
  shuffle = FALSE
)

plot(history218)

test_reshaped218 <- array_reshape(test_matrix2, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred218 <- model218 %>% predict(test_reshaped218)

mse218 = mean(test_price2 - pred218)^2
mse218

rmse218 = sqrt((mean(test_price2 - pred218)^2))
rmse218

mape218 = (mean((abs(test_price2 - pred218))/(abs(test_price2))))*100 
mape218

train_reshaped218 <- array_reshape(train_matrix2, c(491, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred218_train <- model218 %>% predict(train_reshaped218)

mse218_train = mean(train_price2 - pred218_train)^2
mse218_train

rmse218_train = sqrt((mean(train_price2 - pred218_train)^2))
rmse218_train

mape218_train = (mean((abs(train_price2 - pred218_train))/(abs(train_price2))))*100 
mape218_train

#############################################################################################
# LSTM LAYER = 2 
# NEURONS = 40
# BATCH SIZE = 64

model219 <- keras_model_sequential() %>%
  layer_lstm(units = 40, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 40, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model219 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history219 <- model219 %>% fit(
  x = train_reshaped2,  # Data latihan yang telah diubah bentuk
  y = train_price2,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 64,  
  shuffle = FALSE
)

plot(history219)

test_reshaped219 <- array_reshape(test_matrix2, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred219 <- model219 %>% predict(test_reshaped219)

mse219 = mean(test_price2 - pred219)^2
mse219

rmse219 = sqrt((mean(test_price2 - pred219)^2))
rmse219

mape219 = (mean((abs(test_price2 - pred219))/(abs(test_price2))))*100 
mape219

train_reshaped219 <- array_reshape(train_matrix2, c(491, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred219_train <- model219 %>% predict(train_reshaped219)

mse219_train = mean(train_price2 - pred219_train)^2
mse219_train

rmse219_train = sqrt((mean(train_price2 - pred219_train)^2))
rmse219_train

mape219_train = (mean((abs(train_price2 - pred219_train))/(abs(train_price2))))*100 
mape219_train

#############################################################################################
# LSTM LAYER = 2 
# NEURONS = 50
# BATCH SIZE = 64

model220 <- keras_model_sequential() %>%
  layer_lstm(units = 50, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 50, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model220 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history220 <- model220 %>% fit(
  x = train_reshaped2,  # Data latihan yang telah diubah bentuk
  y = train_price2,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 64,  
  shuffle = FALSE
)

plot(history220)

test_reshaped220 <- array_reshape(test_matrix2, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred220 <- model220 %>% predict(test_reshaped220)

mse220 = mean(test_price2 - pred220)^2
mse220

rmse220 = sqrt((mean(test_price2 - pred220)^2))
rmse220

mape220 = (mean((abs(test_price2 - pred220))/(abs(test_price2))))*100 
mape220

train_reshaped220 <- array_reshape(train_matrix2, c(491, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred220_train <- model220 %>% predict(train_reshaped220)

mse220_train = mean(train_price2 - pred220_train)^2
mse220_train

rmse220_train = sqrt((mean(train_price2 - pred220_train)^2))
rmse220_train

mape220_train = (mean((abs(train_price2 - pred220_train))/(abs(train_price2))))*100 
mape220_train

#############################################################################################
# LSTM LAYER = 2 
# NEURONS = 20
# BATCH SIZE = 96

model221 <- keras_model_sequential() %>%
  layer_lstm(units = 20, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 20, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model221 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history221 <- model221 %>% fit(
  x = train_reshaped2,  # Data latihan yang telah diubah bentuk
  y = train_price2,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 96,  
  shuffle = FALSE
)

plot(history221)

test_reshaped221 <- array_reshape(test_matrix2, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred221 <- model221 %>% predict(test_reshaped221)

mse221 = mean(test_price2 - pred221)^2
mse221

rmse221 = sqrt((mean(test_price2 - pred221)^2))
rmse221

mape221 = (mean((abs(test_price2 - pred221))/(abs(test_price2))))*100 
mape221

train_reshaped221 <- array_reshape(train_matrix2, c(491, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred221_train <- model221 %>% predict(train_reshaped221)

mse221_train = mean(train_price2 - pred221_train)^2
mse221_train

rmse221_train = sqrt((mean(train_price2 - pred221_train)^2))
rmse221_train

mape221_train = (mean((abs(train_price2 - pred221_train))/(abs(train_price2))))*100 
mape221_train

#############################################################################################
# LSTM LAYER = 2 
# NEURONS = 30
# BATCH SIZE = 96

model222 <- keras_model_sequential() %>%
  layer_lstm(units = 30, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 30, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model222 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history222 <- model222 %>% fit(
  x = train_reshaped2,  # Data latihan yang telah diubah bentuk
  y = train_price2,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 96,  
  shuffle = FALSE
)

plot(history222)

test_reshaped222 <- array_reshape(test_matrix2, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred222 <- model222 %>% predict(test_reshaped222)

mse222 = mean(test_price2 - pred222)^2
mse222

rmse222 = sqrt((mean(test_price2 - pred222)^2))
rmse222

mape222 = (mean((abs(test_price2 - pred222))/(abs(test_price2))))*100 
mape222

train_reshaped222 <- array_reshape(train_matrix2, c(491, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred222_train <- model222 %>% predict(train_reshaped222)

mse222_train = mean(train_price2 - pred222_train)^2
mse222_train

rmse222_train = sqrt((mean(train_price2 - pred222_train)^2))
rmse222_train

mape222_train = (mean((abs(train_price2 - pred222_train))/(abs(train_price2))))*100 
mape222_train

#############################################################################################
# LSTM LAYER = 2 
# NEURONS = 40
# BATCH SIZE = 96

model223 <- keras_model_sequential() %>%
  layer_lstm(units = 40, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 40, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model223 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history223 <- model223 %>% fit(
  x = train_reshaped2,  # Data latihan yang telah diubah bentuk
  y = train_price2,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 96,  
  shuffle = FALSE
)

plot(history223)

test_reshaped223 <- array_reshape(test_matrix2, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred223 <- model223 %>% predict(test_reshaped223)

mse223 = mean(test_price2 - pred223)^2
mse223

rmse223 = sqrt((mean(test_price2 - pred223)^2))
rmse223

mape223 = (mean((abs(test_price2 - pred223))/(abs(test_price2))))*100 
mape223

train_reshaped223 <- array_reshape(train_matrix2, c(491, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred223_train <- model223 %>% predict(train_reshaped223)

mse223_train = mean(train_price2 - pred223_train)^2
mse223_train

rmse223_train = sqrt((mean(train_price2 - pred223_train)^2))
rmse223_train

mape223_train = (mean((abs(train_price2 - pred223_train))/(abs(train_price2))))*100 
mape223_train

#############################################################################################
# LSTM LAYER = 2 
# NEURONS = 50
# BATCH SIZE = 96

model224 <- keras_model_sequential() %>%
  layer_lstm(units = 50, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 50, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model224 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history224 <- model224 %>% fit(
  x = train_reshaped2,  # Data latihan yang telah diubah bentuk
  y = train_price2,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 96,  
  shuffle = FALSE
)

plot(history224)

test_reshaped224 <- array_reshape(test_matrix2, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred224 <- model224 %>% predict(test_reshaped224)

mse224 = mean(test_price2 - pred224)^2
mse224

rmse224 = sqrt((mean(test_price2 - pred224)^2))
rmse224

mape224 = (mean((abs(test_price2 - pred224))/(abs(test_price2))))*100 
mape224

train_reshaped224 <- array_reshape(train_matrix2, c(491, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred224_train <- model224 %>% predict(train_reshaped224)

mse224_train = mean(train_price2 - pred224_train)^2
mse224_train

rmse224_train = sqrt((mean(train_price2 - pred224_train)^2))
rmse224_train

mape224_train = (mean((abs(train_price2 - pred224_train))/(abs(train_price2))))*100 
mape224_train

#############################################################################################
mse_test2 <- c(mse21, mse22, mse23, mse24, mse25, mse26, mse27, mse28, mse29, mse210,
               mse211, mse212, mse213, mse214, mse215, mse216, mse217, mse218, mse219, mse220,
               mse221, mse222, mse223, mse224)
print(mse_test2)

rmse_test2 <- c(rmse21, rmse22, rmse23, rmse24, rmse25, rmse26, rmse27, rmse28, rmse29, rmse210,
                rmse211, rmse212, rmse213, rmse214, rmse215, rmse216, rmse217, rmse218, rmse219, rmse220,
                rmse221, rmse222, rmse223, rmse224)
print(rmse_test2)

mape_test2 <- c(mape21, mape22, mape23, mape24, mape25, mape26, mape27, mape28, mape29, mape210,
                mape211, mape212, mape213, mape214, mape215, mape216, mape217, mape218, mape219, mape220,
                mape221, mape222, mape223, mape224)
print(mape_test2)

mse_train2 <- c(mse21_train, mse22_train, mse23_train, mse24_train, mse25_train, mse26_train,
                mse27_train, mse28_train, mse29_train, mse210_train, mse211_train, mse212_train,
                mse213_train, mse214_train, mse215_train, mse216_train, mse217_train, mse218_train, 
                mse219_train, mse220_train, mse221_train, mse222_train, mse223_train, mse224_train)
print(mse_train2)

rmse_train2 <- c(rmse21_train, rmse22_train, rmse23_train, rmse24_train, rmse25_train, rmse26_train,
                 rmse27_train, rmse28_train, rmse29_train, rmse210_train, rmse211_train, rmse212_train,
                 rmse213_train, rmse214_train, rmse215_train, rmse216_train, rmse217_train, rmse218_train, 
                 rmse219_train, rmse220_train, rmse221_train, rmse222_train, rmse223_train, rmse224_train)
print(rmse_train2)

mape_train2 <- c(mape21_train, mape22_train, mape23_train, mape24_train, mape25_train, mape26_train,
                 mape27_train, mape28_train, mape29_train, mape210_train, mape211_train, mape212_train,
                 mape213_train, mape214_train, mape215_train, mape216_train, mape217_train, mape218_train, 
                 mape219_train, mape220_train, mape221_train, mape222_train, mape223_train, mape224_train)
print(mape_train2)

no_model2 <- c(1, 2, 3, 4, 5, 6, 7, 8, 9 ,10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)

hasil_kombinasi2 = data.frame(no_model2, mse_test2, rmse_test2, mape_test2, 
                              mse_train2, rmse_train2, mape_train2)
min_mape_test2 = min(mape_test2)
print(min_mape_test2) 

#############################################################################################
#############################################################################################
#############################################################################################
# untuk set 3
dataset3[] <- lapply(dataset3, as.numeric)
str(dataset3)

x3 = data.frame(dataset3$set3.tanggal, dataset3$set3.bulan, dataset3$set3.tahun, 
                dataset3$set3.returndata)
x3 = head(x3,515)

price3 = tail(dataset3$set3.Price,515)
y3 = data.frame(price3)

matrix_data3 <- as.matrix(x3)
dv3 <- matrix_data3

# Pembagian data train dan test
n_rows3 <- nrow(dv3)

train_size3 <- 490
#train_index3 = sample(seq_len(nrow(x3)), size = 490, replace = FALSE)
#train3 = x3[train_index3, ]
#train_price3 = y3[train_index3,]

#test3 = x3[-train_index3,]
#test_price3 = y3[-train_index3,]

# Membuat indeks sekuensial untuk pemilihan baris
sequential_indices3 <- seq_len(n_rows3)

# Membagi matriks menjadi set pelatihan dan pengujian
train_matrix3 <- dv3[sequential_indices3[1:train_size3], ]
test_matrix3 <- dv3[sequential_indices3[(train_size3 + 1):n_rows3], ]
train_price3 = y3[sequential_indices3[1:train_size3], ]
test_price3 = y3[sequential_indices3[(train_size3 + 1):n_rows3], ]

# Tampilkan dimensi dari set pelatihan dan pengujian
cat("Dimensi Set Pelatihan:", dim(train_matrix3), "\n")
cat("Dimensi Set Pengujian:", dim(test_matrix3), "\n")

train_reshaped3 = array_reshape(train_matrix3, c(490, 4, 1))
test_reshaped3 = array_reshape(test_matrix3,c(25, 4, 1))

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 20
# BATCH SIZE = 32

model31 <- keras_model_sequential() %>%
  layer_lstm(units = 20, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model31 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history31 <- model31 %>% fit(
  x = train_reshaped3,  # Data latihan yang telah diubah bentuk
  y = train_price3,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 32,  
  shuffle = FALSE
)

plot(history31)

test_reshaped31 <- array_reshape(test_matrix3, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred31 <- model31 %>% predict(test_reshaped31)

mse31 = mean(test_price3 - pred31)^2
mse31

rmse31 = sqrt((mean(test_price3 - pred31)^2))
rmse31

mape31 = (mean((abs(test_price3 - pred31))/(abs(test_price3))))*100 
mape31

train_reshaped31 <- array_reshape(train_matrix3, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred31_train <- model31 %>% predict(train_reshaped31)

mse31_train = mean(train_price3 - pred31_train)^2
mse31_train

rmse31_train = sqrt((mean(train_price3 - pred31_train)^2))
rmse31_train

mape31_train = (mean((abs(train_price3 - pred31_train))/(abs(train_price3))))*100 
mape31_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 30
# BATCH SIZE = 32

model32 <- keras_model_sequential() %>%
  layer_lstm(units = 30, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model32 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history32 <- model32 %>% fit(
  x = train_reshaped3,  # Data latihan yang telah diubah bentuk
  y = train_price3,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 32,  
  shuffle = FALSE
)

plot(history32)

test_reshaped32 <- array_reshape(test_matrix3, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred32 <- model32 %>% predict(test_reshaped32)

mse32 = mean(test_price3 - pred32)^2
mse32

rmse32 = sqrt((mean(test_price3 - pred32)^2))
rmse32

mape32 = (mean((abs(test_price3 - pred32))/(abs(test_price3))))*100 
mape32

train_reshaped32 <- array_reshape(train_matrix3, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred32_train <- model32 %>% predict(train_reshaped32)

mse32_train = mean(train_price3 - pred32_train)^2
mse32_train

rmse32_train = sqrt((mean(train_price3 - pred32_train)^2))
rmse32_train

mape32_train = (mean((abs(train_price3 - pred32_train))/(abs(train_price3))))*100 
mape32_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 40
# BATCH SIZE = 32

model33 <- keras_model_sequential() %>%
  layer_lstm(units = 40, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model33 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history33 <- model33 %>% fit(
  x = train_reshaped3,  # Data latihan yang telah diubah bentuk
  y = train_price3,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 32,  
  shuffle = FALSE
)

plot(history33)

test_reshaped33 <- array_reshape(test_matrix3, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred33 <- model33 %>% predict(test_reshaped33)

mse33 = mean(test_price3 - pred33)^2
mse33

rmse33 = sqrt((mean(test_price3 - pred33)^2))
rmse33

mape33 = (mean((abs(test_price3 - pred33))/(abs(test_price3))))*100 
mape33

train_reshaped33 <- array_reshape(train_matrix3, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred33_train <- model33 %>% predict(train_reshaped33)

mse33_train = mean(train_price3 - pred33_train)^2
mse33_train

rmse33_train = sqrt((mean(train_price3 - pred33_train)^2))
rmse33_train

mape33_train = (mean((abs(train_price3 - pred33_train))/(abs(train_price3))))*100 
mape33_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 50
# BATCH SIZE = 32

model34 <- keras_model_sequential() %>%
  layer_lstm(units = 50, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model34 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history34 <- model34 %>% fit(
  x = train_reshaped3,  # Data latihan yang telah diubah bentuk
  y = train_price3,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 32,  
  shuffle = FALSE
)

plot(history34)

test_reshaped34 <- array_reshape(test_matrix3, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred34 <- model34 %>% predict(test_reshaped34)

mse34 = mean(test_price3 - pred34)^2
mse34

rmse34 = sqrt((mean(test_price3 - pred34)^2))
rmse34

mape34 = (mean((abs(test_price3 - pred34))/(abs(test_price3))))*100 
mape34

train_reshaped34 <- array_reshape(train_matrix3, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred34_train <- model34 %>% predict(train_reshaped34)

mse34_train = mean(train_price3 - pred34_train)^2
mse34_train

rmse34_train = sqrt((mean(train_price3 - pred34_train)^2))
rmse34_train

mape34_train = (mean((abs(train_price3 - pred34_train))/(abs(train_price3))))*100 
mape34_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 20
# BATCH SIZE = 64

model35 <- keras_model_sequential() %>%
  layer_lstm(units = 20, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model35 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history35 <- model35 %>% fit(
  x = train_reshaped3,  # Data latihan yang telah diubah bentuk
  y = train_price3,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 64,  
  shuffle = FALSE
)

plot(history35)

test_reshaped35 <- array_reshape(test_matrix3, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred35 <- model35 %>% predict(test_reshaped35)

mse35 = mean(test_price3 - pred35)^2
mse35

rmse35 = sqrt((mean(test_price3 - pred35)^2))
rmse35

mape35 = (mean((abs(test_price3 - pred35))/(abs(test_price3))))*100 
mape35

train_reshaped35 <- array_reshape(train_matrix3, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred35_train <- model35 %>% predict(train_reshaped35)

mse35_train = mean(train_price3 - pred35_train)^2
mse35_train

rmse35_train = sqrt((mean(train_price3 - pred35_train)^2))
rmse35_train

mape35_train = (mean((abs(train_price3 - pred35_train))/(abs(train_price3))))*100 
mape35_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 30
# BATCH SIZE = 64

model36 <- keras_model_sequential() %>%
  layer_lstm(units = 30, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model36 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history36 <- model36 %>% fit(
  x = train_reshaped3,  # Data latihan yang telah diubah bentuk
  y = train_price3,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 64,  
  shuffle = FALSE
)

plot(history36)

test_reshaped36 <- array_reshape(test_matrix3, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred36 <- model36 %>% predict(test_reshaped36)

mse36 = mean(test_price3 - pred36)^2
mse36

rmse36 = sqrt((mean(test_price3 - pred36)^2))
rmse36

mape36 = (mean((abs(test_price3 - pred36))/(abs(test_price3))))*100 
mape36

train_reshaped36 <- array_reshape(train_matrix3, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred36_train <- model36 %>% predict(train_reshaped36)

mse36_train = mean(train_price3 - pred36_train)^2
mse36_train

rmse36_train = sqrt((mean(train_price3 - pred36_train)^2))
rmse36_train

mape36_train = (mean((abs(train_price3 - pred36_train))/(abs(train_price3))))*100 
mape36_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 40
# BATCH SIZE = 64

model37 <- keras_model_sequential() %>%
  layer_lstm(units = 40, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model37 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history37 <- model37 %>% fit(
  x = train_reshaped3,  # Data latihan yang telah diubah bentuk
  y = train_price3,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 64,  
  shuffle = FALSE
)

plot(history37)

test_reshaped37 <- array_reshape(test_matrix3, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred37 <- model37 %>% predict(test_reshaped37)

mse37 = mean(test_price3 - pred37)^2
mse37

rmse37 = sqrt((mean(test_price3 - pred37)^2))
rmse37

mape37 = (mean((abs(test_price3 - pred37))/(abs(test_price3))))*100 
mape37

train_reshaped37 <- array_reshape(train_matrix3, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred37_train <- model37 %>% predict(train_reshaped37)

mse37_train = mean(train_price3 - pred37_train)^2
mse37_train

rmse37_train = sqrt((mean(train_price3 - pred37_train)^2))
rmse37_train

mape37_train = (mean((abs(train_price3 - pred37_train))/(abs(train_price3))))*100 
mape37_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 50
# BATCH SIZE = 64

model38 <- keras_model_sequential() %>%
  layer_lstm(units = 50, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model38 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history38 <- model38 %>% fit(
  x = train_reshaped3,  # Data latihan yang telah diubah bentuk
  y = train_price3,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 64,  
  shuffle = FALSE
)

plot(history38)

test_reshaped38 <- array_reshape(test_matrix3, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred38 <- model38 %>% predict(test_reshaped38)

mse38 = mean(test_price3 - pred38)^2
mse38

rmse38 = sqrt((mean(test_price3 - pred38)^2))
rmse38

mape38 = (mean((abs(test_price3 - pred38))/(abs(test_price3))))*100 
mape38

train_reshaped38 <- array_reshape(train_matrix3, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred38_train <- model38 %>% predict(train_reshaped38)

mse38_train = mean(train_price3 - pred38_train)^2
mse38_train

rmse38_train = sqrt((mean(train_price3 - pred38_train)^2))
rmse38_train

mape38_train = (mean((abs(train_price3 - pred38_train))/(abs(train_price3))))*100 
mape38_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 20
# BATCH SIZE = 96

model39 <- keras_model_sequential() %>%
  layer_lstm(units = 20, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model39 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history39 <- model39 %>% fit(
  x = train_reshaped3,  # Data latihan yang telah diubah bentuk
  y = train_price3,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 96,  
  shuffle = FALSE
)

plot(history39)

test_reshaped39 <- array_reshape(test_matrix3, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred39 <- model39 %>% predict(test_reshaped39)

mse39 = mean(test_price3 - pred39)^2
mse39

rmse39 = sqrt((mean(test_price3 - pred39)^2))
rmse39

mape39 = (mean((abs(test_price3 - pred39))/(abs(test_price3))))*100 
mape39

train_reshaped39 <- array_reshape(train_matrix3, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred39_train <- model39 %>% predict(train_reshaped39)

mse39_train = mean(train_price3 - pred39_train)^2
mse39_train

rmse39_train = sqrt((mean(train_price3 - pred39_train)^2))
rmse39_train

mape39_train = (mean((abs(train_price3 - pred39_train))/(abs(train_price3))))*100 
mape39_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 30
# BATCH SIZE = 96

model310 <- keras_model_sequential() %>%
  layer_lstm(units = 30, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model310 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history310 <- model310 %>% fit(
  x = train_reshaped3,  # Data latihan yang telah diubah bentuk
  y = train_price3,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 96,  
  shuffle = FALSE
)

plot(history310)

test_reshaped310 <- array_reshape(test_matrix3, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred310 <- model310 %>% predict(test_reshaped310)

mse310 = mean(test_price3 - pred310)^2
mse310

rmse310 = sqrt((mean(test_price3 - pred310)^2))
rmse310

mape310 = (mean((abs(test_price3 - pred310))/(abs(test_price3))))*100 
mape310

train_reshaped310 <- array_reshape(train_matrix3, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred310_train <- model310 %>% predict(train_reshaped310)

mse310_train = mean(train_price3 - pred310_train)^2
mse310_train

rmse310_train = sqrt((mean(train_price3 - pred310_train)^2))
rmse310_train

mape310_train = (mean((abs(train_price3 - pred310_train))/(abs(train_price3))))*100 
mape310_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 40
# BATCH SIZE = 96

model311 <- keras_model_sequential() %>%
  layer_lstm(units = 40, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model311 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history311 <- model311 %>% fit(
  x = train_reshaped3,  # Data latihan yang telah diubah bentuk
  y = train_price3,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 96,  
  shuffle = FALSE
)

plot(history311)

test_reshaped311 <- array_reshape(test_matrix3, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred311 <- model311 %>% predict(test_reshaped311)

mse311 = mean(test_price3 - pred311)^2
mse311

rmse311 = sqrt((mean(test_price3 - pred311)^2))
rmse311

mape311 = (mean((abs(test_price3 - pred311))/(abs(test_price3))))*100 
mape311

train_reshaped311 <- array_reshape(train_matrix3, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred311_train <- model311 %>% predict(train_reshaped311)

mse311_train = mean(train_price3 - pred311_train)^2
mse311_train

rmse311_train = sqrt((mean(train_price3 - pred311_train)^2))
rmse311_train

mape311_train = (mean((abs(train_price3 - pred311_train))/(abs(train_price3))))*100 
mape311_train

#############################################################################################
# LSTM LAYER = 1 
# NEURONS = 50
# BATCH SIZE = 96

model312 <- keras_model_sequential() %>%
  layer_lstm(units = 50, activation = 'tanh', input_shape = c(4, 1)) %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model312 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history312 <- model312 %>% fit(
  x = train_reshaped3,  # Data latihan yang telah diubah bentuk
  y = train_price3,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 96,  
  shuffle = FALSE
)

plot(history312)

test_reshaped312 <- array_reshape(test_matrix3, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred312 <- model312 %>% predict(test_reshaped312)

mse312 = mean(test_price3 - pred312)^2
mse312

rmse312 = sqrt((mean(test_price3 - pred312)^2))
rmse312

mape312 = (mean((abs(test_price3 - pred312))/(abs(test_price3))))*100 
mape312

train_reshaped312 <- array_reshape(train_matrix3, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred312_train <- model312 %>% predict(train_reshaped312)

mse312_train = mean(train_price3 - pred312_train)^2
mse312_train

rmse312_train = sqrt((mean(train_price3 - pred312_train)^2))
rmse312_train

mape312_train = (mean((abs(train_price3 - pred312_train))/(abs(train_price3))))*100 
mape312_train

#############################################################################################
# LSTM LAYER = 2 
# NEURONS = 20
# BATCH SIZE = 32

model313 <- keras_model_sequential() %>%
  layer_lstm(units = 20, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 20, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model313 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history313 <- model313 %>% fit(
  x = train_reshaped3,  # Data latihan yang telah diubah bentuk
  y = train_price3,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 32,  
  shuffle = FALSE
)

plot(history313)

test_reshaped313 <- array_reshape(test_matrix3, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred313 <- model313 %>% predict(test_reshaped313)

mse313 = mean(test_price3 - pred313)^2
mse313

rmse313 = sqrt((mean(test_price3 - pred313)^2))
rmse313

mape313 = (mean((abs(test_price3 - pred313))/(abs(test_price3))))*100 
mape313

train_reshaped313 <- array_reshape(train_matrix3, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred313_train <- model313 %>% predict(train_reshaped313)

mse313_train = mean(train_price3 - pred313_train)^2
mse313_train

rmse313_train = sqrt((mean(train_price3 - pred313_train)^2))
rmse313_train

mape313_train = (mean((abs(train_price3 - pred313_train))/(abs(train_price3))))*100 
mape313_train

#############################################################################################
# LSTM LAYER = 2 
# NEURONS = 30
# BATCH SIZE = 32

model314 <- keras_model_sequential() %>%
  layer_lstm(units = 30, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 30, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model314 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history314 <- model314 %>% fit(
  x = train_reshaped3,  # Data latihan yang telah diubah bentuk
  y = train_price3,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 32,  
  shuffle = FALSE
)

plot(history314)

test_reshaped314 <- array_reshape(test_matrix3, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred314 <- model314 %>% predict(test_reshaped314)

mse314 = mean(test_price3 - pred314)^2
mse314

rmse314 = sqrt((mean(test_price3 - pred314)^2))
rmse314

mape314 = (mean((abs(test_price3 - pred314))/(abs(test_price3))))*100 
mape314

train_reshaped314 <- array_reshape(train_matrix3, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred314_train <- model314 %>% predict(train_reshaped314)

mse314_train = mean(train_price3 - pred314_train)^2
mse314_train

rmse314_train = sqrt((mean(train_price3 - pred314_train)^2))
rmse314_train

mape314_train = (mean((abs(train_price3 - pred314_train))/(abs(train_price3))))*100 
mape314_train

#############################################################################################
# LSTM LAYER = 2 
# NEURONS = 40
# BATCH SIZE = 32

model315 <- keras_model_sequential() %>%
  layer_lstm(units = 40, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 40, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model315 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history315 <- model315 %>% fit(
  x = train_reshaped3,  # Data latihan yang telah diubah bentuk
  y = train_price3,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 32,  
  shuffle = FALSE
)

plot(history315)

test_reshaped315 <- array_reshape(test_matrix3, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred315 <- model315 %>% predict(test_reshaped315)

mse315 = mean(test_price3 - pred315)^2
mse315

rmse315 = sqrt((mean(test_price3 - pred315)^2))
rmse315

mape315 = (mean((abs(test_price3 - pred315))/(abs(test_price3))))*100 
mape315

train_reshaped315 <- array_reshape(train_matrix3, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred315_train <- model315 %>% predict(train_reshaped315)

mse315_train = mean(train_price3 - pred315_train)^2
mse315_train

rmse315_train = sqrt((mean(train_price3 - pred315_train)^2))
rmse315_train

mape315_train = (mean((abs(train_price3 - pred315_train))/(abs(train_price3))))*100 
mape315_train

#############################################################################################
# LSTM LAYER = 2 
# NEURONS = 50
# BATCH SIZE = 32

model316 <- keras_model_sequential() %>%
  layer_lstm(units = 50, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 50, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model316 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history316 <- model316 %>% fit(
  x = train_reshaped3,  # Data latihan yang telah diubah bentuk
  y = train_price3,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 32,  
  shuffle = FALSE
)

plot(history316)

test_reshaped316 <- array_reshape(test_matrix3, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred316 <- model316 %>% predict(test_reshaped316)

mse316 = mean(test_price3 - pred316)^2
mse316

rmse316 = sqrt((mean(test_price3 - pred316)^2))
rmse316

mape316 = (mean((abs(test_price3 - pred316))/(abs(test_price3))))*100 
mape316

train_reshaped316 <- array_reshape(train_matrix3, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred316_train <- model316 %>% predict(train_reshaped316)

mse316_train = mean(train_price3 - pred316_train)^2
mse316_train

rmse316_train = sqrt((mean(train_price3 - pred316_train)^2))
rmse316_train

mape316_train = (mean((abs(train_price3 - pred316_train))/(abs(train_price3))))*100 
mape316_train

#############################################################################################
# LSTM LAYER = 2 
# NEURONS = 20
# BATCH SIZE = 64

model317 <- keras_model_sequential() %>%
  layer_lstm(units = 20, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 20, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model317 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history317 <- model317 %>% fit(
  x = train_reshaped3,  # Data latihan yang telah diubah bentuk
  y = train_price3,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 64,  
  shuffle = FALSE
)

plot(history317)

test_reshaped317 <- array_reshape(test_matrix3, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred317 <- model317 %>% predict(test_reshaped317)

mse317 = mean(test_price3 - pred317)^2
mse317

rmse317 = sqrt((mean(test_price3 - pred317)^2))
rmse317

mape317 = (mean((abs(test_price3 - pred317))/(abs(test_price3))))*100 
mape317

train_reshaped317 <- array_reshape(train_matrix3, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred317_train <- model317 %>% predict(train_reshaped317)

mse317_train = mean(train_price3 - pred317_train)^2
mse317_train

rmse317_train = sqrt((mean(train_price3 - pred317_train)^2))
rmse317_train

mape317_train = (mean((abs(train_price3 - pred317_train))/(abs(train_price3))))*100 
mape317_train

#############################################################################################
# LSTM LAYER = 2 
# NEURONS = 30
# BATCH SIZE = 64

model318 <- keras_model_sequential() %>%
  layer_lstm(units = 30, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 30, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model318 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history318 <- model318 %>% fit(
  x = train_reshaped3,  # Data latihan yang telah diubah bentuk
  y = train_price3,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 64,  
  shuffle = FALSE
)

plot(history318)

test_reshaped318 <- array_reshape(test_matrix3, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred318 <- model318 %>% predict(test_reshaped318)

mse318 = mean(test_price3 - pred318)^2
mse318

rmse318 = sqrt((mean(test_price3 - pred318)^2))
rmse318

mape318 = (mean((abs(test_price3 - pred318))/(abs(test_price3))))*100 
mape318

train_reshaped318 <- array_reshape(train_matrix3, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred318_train <- model318 %>% predict(train_reshaped318)

mse318_train = mean(train_price3 - pred318_train)^2
mse318_train

rmse318_train = sqrt((mean(train_price3 - pred318_train)^2))
rmse318_train

mape318_train = (mean((abs(train_price3 - pred318_train))/(abs(train_price3))))*100 
mape318_train

#############################################################################################
# LSTM LAYER = 2 
# NEURONS = 40
# BATCH SIZE = 64

model319 <- keras_model_sequential() %>%
  layer_lstm(units = 40, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 40, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model319 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history319 <- model319 %>% fit(
  x = train_reshaped3,  # Data latihan yang telah diubah bentuk
  y = train_price3,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 64,  
  shuffle = FALSE
)

plot(history319)

test_reshaped319 <- array_reshape(test_matrix3, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred319 <- model319 %>% predict(test_reshaped319)

mse319 = mean(test_price3 - pred319)^2
mse319

rmse319 = sqrt((mean(test_price3 - pred319)^2))
rmse319

mape319 = (mean((abs(test_price3 - pred319))/(abs(test_price3))))*100 
mape319

train_reshaped319 <- array_reshape(train_matrix3, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred319_train <- model319 %>% predict(train_reshaped319)

mse319_train = mean(train_price3 - pred319_train)^2
mse319_train

rmse319_train = sqrt((mean(train_price3 - pred319_train)^2))
rmse319_train

mape319_train = (mean((abs(train_price3 - pred319_train))/(abs(train_price3))))*100 
mape319_train

#############################################################################################
# LSTM LAYER = 2 
# NEURONS = 50
# BATCH SIZE = 64

model320 <- keras_model_sequential() %>%
  layer_lstm(units = 50, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 50, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model320 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history320 <- model320 %>% fit(
  x = train_reshaped3,  # Data latihan yang telah diubah bentuk
  y = train_price3,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 64,  
  shuffle = FALSE
)

plot(history320)

test_reshaped320 <- array_reshape(test_matrix3, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred320 <- model320 %>% predict(test_reshaped320)

mse320 = mean(test_price3 - pred320)^2
mse320

rmse320 = sqrt((mean(test_price3 - pred320)^2))
rmse320

mape320 = (mean((abs(test_price3 - pred320))/(abs(test_price3))))*100 
mape320

train_reshaped320 <- array_reshape(train_matrix3, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred320_train <- model320 %>% predict(train_reshaped320)

mse320_train = mean(train_price3 - pred320_train)^2
mse320_train

rmse320_train = sqrt((mean(train_price3 - pred320_train)^2))
rmse320_train

mape320_train = (mean((abs(train_price3 - pred320_train))/(abs(train_price3))))*100 
mape320_train

#############################################################################################
# LSTM LAYER = 2 
# NEURONS = 20
# BATCH SIZE = 96

model321 <- keras_model_sequential() %>%
  layer_lstm(units = 20, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 20, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model321 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history321 <- model321 %>% fit(
  x = train_reshaped3,  # Data latihan yang telah diubah bentuk
  y = train_price3,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 96,  
  shuffle = FALSE
)

plot(history321)

test_reshaped321 <- array_reshape(test_matrix3, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred321 <- model321 %>% predict(test_reshaped321)

mse321 = mean(test_price3 - pred321)^2
mse321

rmse321 = sqrt((mean(test_price3 - pred321)^2))
rmse321

mape321 = (mean((abs(test_price3 - pred321))/(abs(test_price3))))*100 
mape321

train_reshaped321 <- array_reshape(train_matrix3, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred321_train <- model321 %>% predict(train_reshaped321)

mse321_train = mean(train_price3 - pred321_train)^2
mse321_train

rmse321_train = sqrt((mean(train_price3 - pred321_train)^2))
rmse321_train

mape321_train = (mean((abs(train_price3 - pred321_train))/(abs(train_price3))))*100 
mape321_train

#############################################################################################
# LSTM LAYER = 2 
# NEURONS = 30
# BATCH SIZE = 96

model322 <- keras_model_sequential() %>%
  layer_lstm(units = 30, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 30, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model322 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history322 <- model322 %>% fit(
  x = train_reshaped3,  # Data latihan yang telah diubah bentuk
  y = train_price3,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 96,  
  shuffle = FALSE
)

plot(history322)

test_reshaped322 <- array_reshape(test_matrix3, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred322 <- model322 %>% predict(test_reshaped322)

mse322 = mean(test_price3 - pred322)^2
mse322

rmse322 = sqrt((mean(test_price3 - pred322)^2))
rmse322

mape322 = (mean((abs(test_price3 - pred322))/(abs(test_price3))))*100 
mape322

train_reshaped322 <- array_reshape(train_matrix3, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred322_train <- model322 %>% predict(train_reshaped322)

mse322_train = mean(train_price3 - pred322_train)^2
mse322_train

rmse322_train = sqrt((mean(train_price3 - pred322_train)^2))
rmse322_train

mape322_train = (mean((abs(train_price3 - pred322_train))/(abs(train_price3))))*100 
mape322_train

#############################################################################################
# LSTM LAYER = 2 
# NEURONS = 40
# BATCH SIZE = 96

model323 <- keras_model_sequential() %>%
  layer_lstm(units = 40, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 40, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model323 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history323 <- model323 %>% fit(
  x = train_reshaped3,  # Data latihan yang telah diubah bentuk
  y = train_price3,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 96,  
  shuffle = FALSE
)

plot(history323)

test_reshaped323 <- array_reshape(test_matrix3, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred323 <- model323 %>% predict(test_reshaped323)

mse323 = mean(test_price3 - pred323)^2
mse323

rmse323 = sqrt((mean(test_price3 - pred323)^2))
rmse323

mape323 = (mean((abs(test_price3 - pred323))/(abs(test_price3))))*100 
mape323

train_reshaped323 <- array_reshape(train_matrix3, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred323_train <- model323 %>% predict(train_reshaped323)

mse323_train = mean(train_price3 - pred323_train)^2
mse323_train

rmse323_train = sqrt((mean(train_price3 - pred323_train)^2))
rmse323_train

mape323_train = (mean((abs(train_price3 - pred323_train))/(abs(train_price3))))*100 
mape323_train

#############################################################################################
# LSTM LAYER = 2 
# NEURONS = 50
# BATCH SIZE = 96

model324 <- keras_model_sequential() %>%
  layer_lstm(units = 50, activation = 'tanh', input_shape = c(4, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 50, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'leaky_relu')

model324 %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam')

history324 <- model324 %>% fit(
  x = train_reshaped3,  # Data latihan yang telah diubah bentuk
  y = train_price3,  # Gantilah dengan target variable sesuai kasus Anda
  epochs = 100,  # Jumlah epoch (iterasi) yang diinginkan
  batch_size = 96,  
  shuffle = FALSE
)

plot(history324)

test_reshaped324 <- array_reshape(test_matrix3, c(25, 4, 1))  # Sesuaikan dengan data uji Anda
pred324 <- model324 %>% predict(test_reshaped324)

mse324 = mean(test_price3 - pred324)^2
mse324

rmse324 = sqrt((mean(test_price3 - pred324)^2))
rmse324

mape324 = (mean((abs(test_price3 - pred324))/(abs(test_price3))))*100 
mape324

train_reshaped324 <- array_reshape(train_matrix3, c(490, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred324_train <- model324 %>% predict(train_reshaped324)

mse324_train = mean(train_price3 - pred324_train)^2
mse324_train

rmse324_train = sqrt((mean(train_price3 - pred324_train)^2))
rmse324_train

mape324_train = (mean((abs(train_price3 - pred324_train))/(abs(train_price3))))*100 
mape324_train

#############################################################################################
mse_test3 <- c(mse31, mse32, mse33, mse34, mse35, mse36, mse37, mse38, mse39, mse310,
               mse311, mse312, mse313, mse314, mse315, mse316, mse317, mse318, mse319, mse320,
               mse321, mse323, mse323, mse324)
print(mse_test3)

rmse_test3 <- c(rmse31, rmse32, rmse33, rmse34, rmse35, rmse36, rmse37, rmse38, rmse39, rmse310,
                rmse311, rmse312, rmse313, rmse314, rmse315, rmse316, rmse317, rmse318, rmse319, rmse320,
                rmse321, rmse322, rmse323, rmse324)
print(rmse_test3)

mape_test3 <- c(mape31, mape32, mape33, mape34, mape35, mape36, mape37, mape38, mape39, mape310,
                mape311, mape312, mape313, mape314, mape315, mape316, mape317, mape318, mape319, mape320,
                mape321, mape322, mape323, mape324)
print(mape_test3)

mse_train3 <- c(mse31_train, mse32_train, mse33_train, mse34_train, mse35_train, mse36_train,
                mse37_train, mse38_train, mse39_train, mse310_train, mse311_train, mse312_train,
                mse313_train, mse314_train, mse315_train, mse316_train, mse317_train, mse318_train, 
                mse319_train, mse320_train, mse321_train, mse322_train, mse323_train, mse324_train)
print(mse_train3)

rmse_train3 <- c(rmse31_train, rmse32_train, rmse33_train, rmse34_train, rmse35_train, rmse36_train,
                 rmse37_train, rmse38_train, rmse39_train, rmse310_train, rmse311_train, rmse312_train,
                 rmse313_train, rmse314_train, rmse315_train, rmse316_train, rmse317_train, rmse318_train, 
                 rmse319_train, rmse320_train, rmse321_train, rmse322_train, rmse323_train, rmse324_train)
print(rmse_train3)

mape_train3 <- c(mape31_train, mape32_train, mape33_train, mape34_train, mape35_train, mape36_train,
                 mape37_train, mape38_train, mape39_train, mape310_train, mape311_train, mape312_train,
                 mape313_train, mape314_train, mape315_train, mape316_train, mape317_train, mape318_train, 
                 mape319_train, mape320_train, mape321_train, mape322_train, mape323_train, mape324_train)
print(mape_train3)

no_model3 <- c(1, 2, 3, 4, 5, 6, 7, 8, 9 ,10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)

hasil_kombinasi3 = data.frame(no_model3, mse_test3, rmse_test3, mape_test3, 
                              mse_train3, rmse_train3, mape_train3)
min_mape_test3 = min(mape_test3)
print(min_mape_test3) 

########################################
dataakhir1 <- tail(set1,25)
data_set1 <- data.frame(time = set1$Date, price = set1$Price)
predprice1 <- data.frame(time = dataakhir1$Date, price = pred122)
combined_plot1 <- ggplot() +
  geom_line(data = data_set1, aes(x = time, y = price, color = "Actual Price")) +
  geom_line(data = predprice1, aes(x = time, y = price, color = "Predicted Price")) +
  scale_color_manual(values = c("Actual Price" = "magenta", "Predicted Price" = "black")) +
  labs(color = "") +
  theme_minimal()

print(combined_plot1)

########################################

dataakhir2 = tail(set2,25)
data_set2 <- data.frame(time = set2$Date, price = set2$Price)
predprice2 <- data.frame(time = dataakhir2$Date, price = pred216)
combined_plot2 <- ggplot() +
  geom_line(data = data_set2, aes(x = time, y = price, color = "Actual Price")) +
  geom_line(data = predprice2, aes(x = time, y = price, color = "Predicted Price")) +
  scale_color_manual(values = c("Actual Price" = "magenta", "Predicted Price" = "black")) +
  labs(color = "") +
  theme_minimal()

print(combined_plot2)

########################################

dataakhir3 = tail(set3,25)
data_set3 <- data.frame(time = set3$Date, price = set3$Price)
predprice3 <- data.frame(time = dataakhir3$Date, price = 316)
combined_plot3 <- ggplot() +
  geom_line(data = data_set3, aes(x = time, y = price, color = "Actual Price")) +
  geom_line(data = predprice3, aes(x = time, y = price, color = "Predicted Price")) +
  scale_color_manual(values = c("Actual Price" = "magenta", "Predicted Price" = "black")) +
  labs(color = "") +
  theme_minimal()

print(combined_plot3)
