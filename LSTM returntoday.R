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

returntoday = matrix(nrow = 1548, ncol = 1)
for (k in 2:1548) {
  returntoday[k,] = newdata$Price[k] - newdata$Price[k-1]
}
returntoday

newdata$returntoday = c(returntoday)

newdata[] <- lapply(newdata, as.numeric)
str(newdata)

newdata1 = head(newdata, 1547)
newdata2 = tail(newdata,1547)


x1 = data.frame(newdata1$tanggal, newdata1$bulan, newdata1$tahun, 
                newdata1$Price)
y1 = data.frame(newdata2$returntoday)


matrix_data1 <- as.matrix(x1)
dv1 <- matrix_data1

# Pembagian data train dan test
n_rows1 <- nrow(dv1)

train_size1 <- 1030
#train_index1 = sample(seq_len(nrow(x1)), size = 1030, replace = FALSE)
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
# install_tensorflow(envname = "r-tensorflow")

# install.packages("keras")
# keras::install_keras()

library(keras)
library(tensorflow)

train_reshaped1 = array_reshape(train_matrix1, c(1030, 4, 1))
test_reshaped1 = array_reshape(test_matrix1,c(517, 4, 1))

# test : 1 may 2022 - 29 dec 2023
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
test_reshaped11 <- array_reshape(test_matrix1, c(517, 4, 1))  # Sesuaikan dengan data uji Anda
pred11 <- model11 %>% predict(test_reshaped11)

test1 = data.frame(test_matrix1)
test1 <- cbind(test1, pred11 = pred11)
price1 <- test1$pred11 + test1$newdata1.Price #price1 = price tomorrow
test1 = cbind(test1, price1)
# price1[1:516] - test1$newdata.Price[2:517]

mse11 = mean((test1$newdata1.Price[2:517] - price1[1:516])^2)
mse11

rmse11 = sqrt((mean(test1$newdata1.Price[2:517] - price1[1:516])^2))
rmse11

mape11 = (mean((abs(test1$newdata1.Price[2:517] - price1[1:516]))/(abs(test1$newdata1.Price[2:517]))))*100 
mape11

# Menggunakan model untuk prediksi pada data pelatihan 
train_reshaped11 <- array_reshape(train_matrix1, c(1030, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred11_train <- model11 %>% predict(train_reshaped11)

train1 = data.frame(train_matrix1)
train1 <- cbind(train1, pred11_train)
price1_train = train1$newdata1.Price + train1$pred11_train
train1 = cbind(train1, price1_train)
# price1_train[1:1029] - train1$newdata.Price[2:1030]

mse11_train = mean((train1$newdata1.Price[2:1030] - price1_train[1:1029])^2)
mse11_train

rmse11_train = sqrt((mean(train1$newdata1.Price[2:1030] - price1_train[1:1029])^2))
rmse11_train

mape11_train = (mean((abs(train1$newdata1.Price[2:1030] - price1_train[1:1029]))/(abs(train1$newdata1.Price[2:1030]))))*100 
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

test_reshaped12 <- array_reshape(test_matrix1, c(517, 4, 1))  # Sesuaikan dengan data uji Anda
pred12 <- model12 %>% predict(test_reshaped12)

test2 = data.frame(test_matrix1)
test2 <- cbind(test2, pred12 = pred12)
price2 <- test2$pred12 + test2$newdata1.Price
# price2[1:516] - test2$newdata.Price[2:517]

mse12 = mean((test2$newdata1.Price[2:517] - price2[1:516])^2)
mse12

rmse12 = sqrt((mean(test2$newdata1.Price[2:517] - price2[1:516])^2))
rmse12

mape12 = (mean((abs(test2$newdata1.Price[2:517] - price2[1:516]))/(abs(test2$newdata1.Price[2:517]))))*100 
mape12

train_reshaped12 <- array_reshape(train_matrix1, c(1030, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred12_train <- model12 %>% predict(train_reshaped12)

train2 = data.frame(train_matrix1)
train2 <- cbind(train2, pred12_train)
price2_train = train2$newdata1.Price + train2$pred12_train
# price2_train[1:1029] - train2$newdata.Price[2:1030]

mse12_train = mean((train2$newdata1.Price[2:1030] - price2_train[1:1029])^2)
mse12_train

rmse12_train = sqrt((mean(train2$newdata1.Price[2:1030] - price2_train[1:1029])^2))
rmse12_train

mape12_train = (mean((abs(train2$newdata1.Price[2:1030] - price2_train[1:1029])/(abs(train2$newdata1.Price[2:1030])))))*100 
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

test_reshaped13 <- array_reshape(test_matrix1, c(517, 4, 1))  # Sesuaikan dengan data uji Anda
pred13 <- model13 %>% predict(test_reshaped13)

test3 = data.frame(test_matrix1)
test3 <- cbind(test3, pred13 = pred13)
price3 <- test3$pred13 + test3$newdata1.Price
# price3[1:516] - test3$newdata1.Price[2:517]

mse13 = mean((test3$newdata1.Price[2:517] - price3[1:516])^2)
mse13

rmse13 = sqrt((mean(test3$newdata1.Price[2:517] - price3[1:516])^2))
rmse13

mape13 = (mean((abs(test3$newdata1.Price[2:517] - price3[1:516]))/(abs(test3$newdata1.Price[2:517]))))*100 
mape13

train_reshaped13 <- array_reshape(train_matrix1, c(1030, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred13_train <- model13 %>% predict(train_reshaped13)

train3 = data.frame(train_matrix1)
train3 <- cbind(train3, pred13_train)
price3_train = train3$newdata1.Price + train3$pred13_train
# price3_train[1:1029] - train3$newdata1.Price[2:1030]

mse13_train = mean((train3$newdata1.Price[2:1030] - price3_train[1:1029])^2)
mse13_train

rmse13_train = sqrt((mean(train3$newdata1.Price[2:1030] - price3_train[1:1029])^2))
rmse13_train

mape13_train = (mean((abs(train3$newdata1.Price[2:1030] - price3_train[1:1029]))/(abs(train3$newdata1.Price[2:1030]))))*100 
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

test_reshaped14 <- array_reshape(test_matrix1, c(517, 4, 1))  # Sesuaikan dengan data uji Anda
pred14 <- model14 %>% predict(test_reshaped14)

test4 = data.frame(test_matrix1)
test4 <- cbind(test4, pred14 = pred14)
price4 <- test4$pred14 + test4$newdata1.Price
# price4[1:516] - test4$newdata1.Price[2:517]

mse14 = mean((test4$newdata1.Price[2:517] - price4[1:516])^2)
mse14

rmse14 = sqrt((mean(test4$newdata1.Price[2:517] - price4[1:516])^2))
rmse14

mape14 = (mean((abs(test4$newdata1.Price[2:517] - price4[1:516]))/(abs(test4$newdata1.Price[2:517]))))*100 
mape14

train_reshaped14 <- array_reshape(train_matrix1, c(1030, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred14_train <- model14 %>% predict(train_reshaped14)

train4 = data.frame(train_matrix1)
train4 <- cbind(train4, pred14_train)
price4_train = train4$newdata1.Price + train4$pred14_train
# price4_train[1:1029] - train4$newdata1.Price[2:1030]

mse14_train = mean((train4$newdata1.Price[2:1030] - price4_train[1:1029])^2)
mse14_train

rmse14_train = sqrt((mean(train4$newdata1.Price[2:1030] - price4_train[1:1029])^2))
rmse14_train

mape14_train = (mean((abs(train4$newdata1.Price[2:1030] - price4_train[1:1029]))/(abs(train4$newdata1.Price[2:1030]))))*100 
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

test_reshaped15 <- array_reshape(test_matrix1, c(517, 4, 1))  # Sesuaikan dengan data uji Anda
pred15 <- model15 %>% predict(test_reshaped15)

test5 = data.frame(test_matrix1)
test5 <- cbind(test5, pred15 = pred15)
price5 <- test5$pred15 + test5$newdata1.Price
# price5[1:516] - test5$newdata1.Price[2:517]

mse15 = mean((test5$newdata1.Price[2:517] - price5[1:516])^2)
mse15

rmse15 = sqrt((mean(test5$newdata1.Price[2:517] - price5[1:516])^2))
rmse15

mape15 = (mean((abs(test5$newdata1.Price[2:517] - price5[1:516]))/(abs(test5$newdata1.Price[2:517]))))*100 
mape15

train_reshaped15 <- array_reshape(train_matrix1, c(1030, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred15_train <- model15 %>% predict(train_reshaped15)

train5 = data.frame(train_matrix1)
train5 <- cbind(train5, pred15_train)
price5_train = train5$newdata1.Price + train5$pred15_train
# price5_train[1:1029] - train5$newdata1.Price[2:1030]

mse15_train = mean((train5$newdata1.Price[2:1030] - price5_train[1:1029])^2)
mse15_train

rmse15_train = sqrt((mean(train5$newdata1.Price[2:1030] - price5_train[1:1029])^2))
rmse15_train

mape15_train = (mean((abs(train5$newdata1.Price[2:1030] - price5_train[1:1029]))/(abs(train5$newdata1.Price[2:1030]))))*100 
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

test_reshaped16 <- array_reshape(test_matrix1, c(517, 4, 1))  # Sesuaikan dengan data uji Anda
pred16 <- model16 %>% predict(test_reshaped16)

test6 = data.frame(test_matrix1)
test6 <- cbind(test6, pred16 = pred16)
price6 <- test6$pred16 + test6$newdata1.Price
# price6[1:516] - test6$newdata1.Price[2:517]

mse16 = mean((test6$newdata1.Price[2:517] - price6[1:516])^2)
mse16

rmse16 = sqrt((mean(test6$newdata1.Price[2:517] - price6[1:516])^2))
rmse16

mape16 = (mean((abs(test6$newdata1.Price[2:517] - price6[1:516]))/(abs(test6$newdata1.Price[2:517]))))*100 
mape16

train_reshaped16 <- array_reshape(train_matrix1, c(1030, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred16_train <- model16 %>% predict(train_reshaped16)

train6 = data.frame(train_matrix1)
train6 <- cbind(train6, pred16_train)
price6_train = train6$newdata1.Price + train6$pred16_train
# price6_train[1:1029] - train6$newdata1.Price[2:1030]

mse16_train = mean((train6$newdata1.Price[2:1030] - price6_train[1:1029])^2)
mse16_train

rmse16_train = sqrt((mean(train6$newdata1.Price[2:1030] - price6_train[1:1029])^2))
rmse16_train

mape16_train = (mean((abs(train6$newdata1.Price[2:1030] - price6_train[1:1029]))/(abs(train6$newdata1.Price[2:1030]))))*100 
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

test_reshaped17 <- array_reshape(test_matrix1, c(517, 4, 1))  # Sesuaikan dengan data uji Anda
pred17 <- model17 %>% predict(test_reshaped17)

test7 = data.frame(test_matrix1)
test7 <- cbind(test7, pred17 = pred17)
price7 <- test7$pred17 + test7$newdata1.Price
# price7[1:516] - test7$newdata1.Price[2:517]

mse17 = mean((test7$newdata1.Price[2:517] - price7[1:516])^2)
mse17

rmse17 = sqrt((mean(test7$newdata1.Price[2:517] - price7[1:516])^2))
rmse17

mape17 = (mean((abs(test7$newdata1.Price[2:517] - price7[1:516]))/(abs(test7$newdata1.Price[2:517]))))*100 
mape17

train_reshaped17 <- array_reshape(train_matrix1, c(1030, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred17_train <- model17 %>% predict(train_reshaped17)

train7 = data.frame(train_matrix1)
train7 <- cbind(train7, pred17_train)
price7_train = train7$newdata1.Price + train7$pred17_train
# price7_train[1:1029] - train7$newdata1.Price[2:1030]

mse17_train = mean((train7$newdata1.Price[2:1030] - price7_train[1:1029])^2)
mse17_train

rmse17_train = sqrt((mean(train7$newdata1.Price[2:1030] - price7_train[1:1029])^2))
rmse17_train

mape17_train = (mean((abs(train7$newdata1.Price[2:1030] - price7_train[1:1029]))/(abs(train7$newdata1.Price[2:1030]))))*100 
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

test_reshaped18 <- array_reshape(test_matrix1, c(517, 4, 1))  # Sesuaikan dengan data uji Anda
pred18 <- model18 %>% predict(test_reshaped18)

test8 = data.frame(test_matrix1)
test8 <- cbind(test8, pred18 = pred18)
price8 <- test8$pred18 + test8$newdata1.Price
# price8[1:516] - test8$newdata1.Price[2:517]

mse18 = mean((test8$newdata1.Price[2:517] - price8[1:516])^2)
mse18

rmse18 = sqrt((mean(test8$newdata1.Price[2:517] - price8[1:516])^2))
rmse18

mape18 = (mean((abs(test8$newdata1.Price[2:517] - price8[1:516]))/(abs(test8$newdata1.Price[2:517]))))*100 
mape18

train_reshaped18 <- array_reshape(train_matrix1, c(1030, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred18_train <- model18 %>% predict(train_reshaped18)

train8 = data.frame(train_matrix1)
train8 <- cbind(train8, pred18_train)
price8_train = train8$newdata1.Price + train8$pred18_train
# price8_train[1:1029] - train8$newdata1.Price[2:1030]

mse18_train = mean((train8$newdata1.Price[2:1030] - price8_train[1:1029])^2)
mse18_train

rmse18_train = sqrt((mean(train8$newdata1.Price[2:1030] - price8_train[1:1029])^2))
rmse18_train

mape18_train = (mean((abs(train8$newdata1.Price[2:1030] - price8_train[1:1029]))/(abs(train8$newdata1.Price[2:1030]))))*100 
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

test_reshaped19 <- array_reshape(test_matrix1, c(517, 4, 1))  # Sesuaikan dengan data uji Anda
pred19 <- model19 %>% predict(test_reshaped19)

test9 = data.frame(test_matrix1)
test9 <- cbind(test9, pred19 = pred19)
price9 <- test9$pred19 + test9$newdata1.Price
# price9[1:516] - test9$newdata1.Price[2:517]

mse19 = mean((test9$newdata1.Price[2:517] - price9[1:516])^2)
mse19

rmse19 = sqrt((mean(test9$newdata1.Price[2:517] - price9[1:516])^2))
rmse19

mape19 = (mean((abs(test9$newdata1.Price[2:517] - price9[1:516]))/(abs(test9$newdata1.Price[2:517]))))*100 
mape19

train_reshaped19 <- array_reshape(train_matrix1, c(1030, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred19_train <- model19 %>% predict(train_reshaped19)

train9 = data.frame(train_matrix1)
train9 <- cbind(train9, pred19_train)
price9_train = train9$newdata1.Price + train9$pred19_train
# price9_train[1:1029] - train9$newdata1.Price[2:1030]

mse19_train = mean((train9$newdata1.Price[2:1030] - price9_train[1:1029])^2)
mse19_train

rmse19_train = sqrt((mean(train9$newdata1.Price[2:1030] - price9_train[1:1029])^2))
rmse19_train

mape19_train = (mean((abs(train9$newdata1.Price[2:1030] - price9_train[1:1029]))/(abs(train9$newdata1.Price[2:1030]))))*100 
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

test_reshaped110 <- array_reshape(test_matrix1, c(517, 4, 1))  # Sesuaikan dengan data uji Anda
pred110 <- model110 %>% predict(test_reshaped110)

test10 = data.frame(test_matrix1)
test10 <- cbind(test10, pred110 = pred110)
price10 <- test10$pred110 + test10$newdata1.Price
# price10[1:516] - test10$newdata1.Price[2:517]

mse110 = mean((test10$newdata1.Price[2:517] - price10[1:516])^2)
mse110

rmse110 = sqrt((mean(test10$newdata1.Price[2:517] - price10[1:516])^2))
rmse110

mape110 = (mean((abs(test10$newdata1.Price[2:517] - price10[1:516]))/(abs(test10$newdata1.Price[2:517]))))*100 
mape110

train_reshaped110 <- array_reshape(train_matrix1, c(1030, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred110_train <- model110 %>% predict(train_reshaped110)

train10 = data.frame(train_matrix1)
train10 <- cbind(train10, pred110_train)
price10_train = train10$newdata1.Price + train10$pred110_train
# price10_train[1:1029] - train10$newdata1.Price[2:1030]

mse110_train = mean((train10$newdata1.Price[2:1030] - price10_train[1:1029])^2)
mse110_train

rmse110_train = sqrt((mean(train10$newdata1.Price[2:1030] - price10_train[1:1029])^2))
rmse110_train

mape110_train = (mean((abs(train10$newdata1.Price[2:1030] - price10_train[1:1029]))/(abs(train10$newdata1.Price[2:1030]))))*100 
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

test_reshaped111 <- array_reshape(test_matrix1, c(517, 4, 1))  # Sesuaikan dengan data uji Anda
pred111 <- model111 %>% predict(test_reshaped111)

test11 = data.frame(test_matrix1)
test11 <- cbind(test11, pred111 = pred111)
price11 <- test11$pred111 + test11$newdata1.Price
# price11[1:516] - test11$newdata1.Price[2:517]

mse111 = mean((test11$newdata1.Price[2:517] - price11[1:516])^2)
mse111

rmse111 = sqrt((mean(test11$newdata1.Price[2:517] - price11[1:516])^2))
rmse111

mape111 = (mean((abs(test11$newdata1.Price[2:517] - price11[1:516]))/(abs(test11$newdata1.Price[2:517]))))*100 
mape111

train_reshaped111 <- array_reshape(train_matrix1, c(1030, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred111_train <- model111 %>% predict(train_reshaped111)

train11 = data.frame(train_matrix1)
train11 <- cbind(train11, pred111_train)
price11_train = train11$newdata1.Price + train11$pred111_train
# price11_train[1:1029] - train11$newdata1.Price[2:1030]

mse111_train = mean((train11$newdata1.Price[2:1030] - price11_train[1:1029])^2)
mse111_train

rmse111_train = sqrt((mean(train11$newdata1.Price[2:1030] - price11_train[1:1029])^2))
rmse111_train

mape111_train = (mean((abs(train11$newdata1.Price[2:1030] - price11_train[1:1029]))/(abs(train11$newdata1.Price[2:1030]))))*100 
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

test_reshaped112 <- array_reshape(test_matrix1, c(517, 4, 1))  # Sesuaikan dengan data uji Anda
pred112 <- model112 %>% predict(test_reshaped112)

test12 = data.frame(test_matrix1)
test12 <- cbind(test12, pred112 = pred112)
price12 <- test12$pred112 + test12$newdata1.Price
# price12[1:516] - test12$newdata1.Price[2:517]

mse112 = mean((test12$newdata1.Price[2:517] - price12[1:516])^2)
mse112

rmse112 = sqrt((mean(test12$newdata1.Price[2:517] - price12[1:516])^2))
rmse112

mape112 = (mean((abs(test12$newdata1.Price[2:517] - price12[1:516]))/(abs(test12$newdata1.Price[2:517]))))*100 
mape112

train_reshaped112 <- array_reshape(train_matrix1, c(1030, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred112_train <- model112 %>% predict(train_reshaped112)

train12 = data.frame(train_matrix1)
train12 <- cbind(train12, pred112_train)
price12_train = train12$newdata1.Price + train12$pred112_train
# price12_train[1:1029] - train12$newdata1.Price[2:1030]

mse112_train = mean((train12$newdata1.Price[2:1030] - price12_train[1:1029])^2)
mse112_train

rmse112_train = sqrt((mean(train12$newdata1.Price[2:1030] - price12_train[1:1029])^2))
rmse112_train

mape112_train = (mean((abs(train12$newdata1.Price[2:1030] - price12_train[1:1029]))/(abs(train12$newdata1.Price[2:1030]))))*100 
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

test_reshaped113 <- array_reshape(test_matrix1, c(517, 4, 1))  # Sesuaikan dengan data uji Anda
pred113 <- model113 %>% predict(test_reshaped113)

test13 = data.frame(test_matrix1)
test13 <- cbind(test13, pred113 = pred113)
price13 <- test13$pred113 + test13$newdata1.Price
# price13[1:516] - test13$newdata1.Price[2:517]

mse113 = mean((test13$newdata1.Price[2:517] - price13[1:516])^2)
mse113

rmse113 = sqrt((mean(test13$newdata1.Price[2:517] - price13[1:516])^2))
rmse113

mape113 = (mean((abs(test13$newdata1.Price[2:517] - price13[1:516]))/(abs(test13$newdata1.Price[2:517]))))*100 
mape113

train_reshaped113 <- array_reshape(train_matrix1, c(1030, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred113_train <- model113 %>% predict(train_reshaped113)

train13 = data.frame(train_matrix1)
train13 <- cbind(train13, pred113_train)
price13_train = train13$newdata1.Price + train13$pred113_train
# price13_train[1:1029] - train13$newdata1.Price[2:1030]

mse113_train = mean((train13$newdata1.Price[2:1030] - price13_train[1:1029])^2)
mse113_train

rmse113_train = sqrt((mean(train13$newdata1.Price[2:1030] - price13_train[1:1029])^2))
rmse113_train

mape113_train = (mean((abs(train13$newdata1.Price[2:1030] - price13_train[1:1029]))/(abs(train13$newdata1.Price[2:1030]))))*100 
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

test_reshaped114 <- array_reshape(test_matrix1, c(517, 4, 1))  # Sesuaikan dengan data uji Anda
pred114 <- model114 %>% predict(test_reshaped114)

test14 = data.frame(test_matrix1)
test14 <- cbind(test14, pred114 = pred114)
price14 <- test14$pred114 + test14$newdata1.Price
# price14[1:516] - test14$newdata1.Price[2:517]

mse114 = mean((test14$newdata1.Price[2:517] - price14[1:516])^2)
mse114

rmse114 = sqrt((mean(test14$newdata1.Price[2:517] - price14[1:516])^2))
rmse114

mape114 = (mean((abs(test14$newdata1.Price[2:517] - price14[1:516]))/(abs(test14$newdata1.Price[2:517]))))*100 
mape114

train_reshaped114 <- array_reshape(train_matrix1, c(1030, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred114_train <- model114 %>% predict(train_reshaped114)

train14 = data.frame(train_matrix1)
train14 <- cbind(train14, pred114_train)
price14_train = train14$newdata1.Price + train14$pred114_train
# price14_train[1:1029] - train14$newdata1.Price[2:1030]

mse114_train = mean((train14$newdata1.Price[2:1030] - price14_train[1:1029])^2)
mse114_train

rmse114_train = sqrt((mean(train14$newdata1.Price[2:1030] - price14_train[1:1029])^2))
rmse114_train

mape114_train = (mean((abs(train14$newdata1.Price[2:1030] - price14_train[1:1029]))/(abs(train14$newdata1.Price[2:1030]))))*100 
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

test_reshaped115 <- array_reshape(test_matrix1, c(517, 4, 1))  # Sesuaikan dengan data uji Anda
pred115 <- model115 %>% predict(test_reshaped115)

test15 = data.frame(test_matrix1)
test15 <- cbind(test15, pred115 = pred115)
price15 <- test15$pred115 + test15$newdata1.Price
# price15[1:516] - test15$newdata1.Price[2:517]

mse115 = mean((test15$newdata1.Price[2:517] - price15[1:516])^2)
mse115

rmse115 = sqrt((mean(test15$newdata1.Price[2:517] - price15[1:516])^2))
rmse115

mape115 = (mean((abs(test15$newdata1.Price[2:517] - price15[1:516]))/(abs(test15$newdata1.Price[2:517]))))*100 
mape115

train_reshaped115 <- array_reshape(train_matrix1, c(1030, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred115_train <- model115 %>% predict(train_reshaped115)

train15 = data.frame(train_matrix1)
train15 <- cbind(train15, pred115_train)
price15_train = train15$newdata1.Price + train15$pred115_train
# price15_train[1:1029] - train15$newdata1.Price[2:1030]

mse115_train = mean((train15$newdata1.Price[2:1030] - price15_train[1:1029])^2)
mse115_train

rmse115_train = sqrt((mean(train15$newdata1.Price[2:1030] - price15_train[1:1029])^2))
rmse115_train

mape115_train = (mean((abs(train15$newdata1.Price[2:1030] - price15_train[1:1029]))/(abs(train15$newdata1.Price[2:1030]))))*100 
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

test_reshaped116 <- array_reshape(test_matrix1, c(517, 4, 1))  # Sesuaikan dengan data uji Anda
pred116 <- model116 %>% predict(test_reshaped116)

test16 = data.frame(test_matrix1)
test16 <- cbind(test16, pred116 = pred116)
price16 <- test16$pred116 + test16$newdata1.Price
# price16[1:516] - test16$newdata1.Price[2:517]

mse116 = mean((test16$newdata1.Price[2:517] - price16[1:516])^2)
mse116

rmse116 = sqrt((mean(test16$newdata1.Price[2:517] - price16[1:516])^2))
rmse116

mape116 = (mean((abs(test16$newdata1.Price[2:517] - price16[1:516]))/(abs(test16$newdata1.Price[2:517]))))*100 
mape116

train_reshaped116 <- array_reshape(train_matrix1, c(1030, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred116_train <- model116 %>% predict(train_reshaped116)

train16 = data.frame(train_matrix1)
train16 <- cbind(train16, pred116_train)
price16_train = train16$newdata1.Price + train16$pred116_train
# price16_train[1:1029] - train16$newdata1.Price[2:1030]

mse116_train = mean((train16$newdata1.Price[2:1030] - price16_train[1:1029])^2)
mse116_train

rmse116_train = sqrt((mean(train16$newdata1.Price[2:1030] - price16_train[1:1029])^2))
rmse116_train

mape116_train = (mean((abs(train16$newdata1.Price[2:1030] - price16_train[1:1029]))/(abs(train16$newdata1.Price[2:1030]))))*100 
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

test_reshaped117 <- array_reshape(test_matrix1, c(517, 4, 1))  # Sesuaikan dengan data uji Anda
pred117 <- model117 %>% predict(test_reshaped117)

test17 = data.frame(test_matrix1)
test17 <- cbind(test17, pred117 = pred117)
price17 <- test17$pred117 + test17$newdata1.Price
# price17[1:516] - test17$newdata1.Price[2:517]

mse117 = mean((test17$newdata1.Price[2:517] - price17[1:516])^2)
mse117

rmse117 = sqrt((mean(test17$newdata1.Price[2:517] - price17[1:516])^2))
rmse117

mape117 = (mean((abs(test17$newdata1.Price[2:517] - price17[1:516]))/(abs(test17$newdata1.Price[2:517]))))*100 
mape117

train_reshaped117 <- array_reshape(train_matrix1, c(1030, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred117_train <- model117 %>% predict(train_reshaped117)

train17 = data.frame(train_matrix1)
train17 <- cbind(train17, pred117_train)
price17_train = train17$newdata1.Price + train17$pred117_train
# price17_train[1:1029] - train17$newdata1.Price[2:1030]

mse117_train = mean((train17$newdata1.Price[2:1030] - price17_train[1:1029])^2)
mse117_train

rmse117_train = sqrt((mean(train17$newdata1.Price[2:1030] - price17_train[1:1029])^2))
rmse117_train

mape117_train = (mean((abs(train17$newdata1.Price[2:1030] - price17_train[1:1029]))/(abs(train17$newdata1.Price[2:1030]))))*100 
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

test_reshaped118 <- array_reshape(test_matrix1, c(517, 4, 1))  # Sesuaikan dengan data uji Anda
pred118 <- model118 %>% predict(test_reshaped118)

test18 = data.frame(test_matrix1)
test18 <- cbind(test18, pred118 = pred118)
price18 <- test18$pred118 + test18$newdata1.Price
# price18[1:516] - test18$newdata1.Price[2:517]

mse118 = mean((test18$newdata1.Price[2:517] - price18[1:516])^2)
mse118

rmse118 = sqrt((mean(test18$newdata1.Price[2:517] - price18[1:516])^2))
rmse118

mape118 = (mean((abs(test18$newdata1.Price[2:517] - price18[1:516]))/(abs(test18$newdata1.Price[2:517]))))*100 
mape118

train_reshaped118 <- array_reshape(train_matrix1, c(1030, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred118_train <- model118 %>% predict(train_reshaped118)

train18 = data.frame(train_matrix1)
train18 <- cbind(train18, pred118_train)
price18_train = train18$newdata1.Price + train18$pred118_train
# price18_train[1:1029] - train18$newdata1.Price[2:1030]

mse118_train = mean((train18$newdata1.Price[2:1030] - price18_train[1:1029])^2)
mse118_train

rmse118_train = sqrt((mean(train18$newdata1.Price[2:1030] - price18_train[1:1029])^2))
rmse118_train

mape118_train = (mean((abs(train18$newdata1.Price[2:1030] - price18_train[1:1029]))/(abs(train18$newdata1.Price[2:1030]))))*100 
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

test_reshaped119 <- array_reshape(test_matrix1, c(517, 4, 1))  # Sesuaikan dengan data uji Anda
pred119 <- model119 %>% predict(test_reshaped119)

test19 = data.frame(test_matrix1)
test19 <- cbind(test19, pred119 = pred119)
price19 <- test19$pred119 + test19$newdata1.Price
# price19[1:516] - test19$newdata1.Price[2:517]

mse119 = mean((test19$newdata1.Price[2:517] - price19[1:516])^2)
mse119

rmse119 = sqrt((mean(test19$newdata1.Price[2:517] - price19[1:516])^2))
rmse119

mape119 = (mean((abs(test19$newdata1.Price[2:517] - price19[1:516]))/(abs(test19$newdata1.Price[2:517]))))*100 
mape119

train_reshaped119 <- array_reshape(train_matrix1, c(1030, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred119_train <- model119 %>% predict(train_reshaped119)

train19 = data.frame(train_matrix1)
train19 <- cbind(train19, pred119_train)
price19_train = train19$newdata1.Price + train19$pred119_train
# price19_train[1:1029] - train19$newdata1.Price[2:1030]

mse119_train = mean((train19$newdata1.Price[2:1030] - price19_train[1:1029])^2)
mse119_train

rmse119_train = sqrt((mean(train19$newdata1.Price[2:1030] - price19_train[1:1029])^2))
rmse119_train

mape119_train = (mean((abs(train19$newdata1.Price[2:1030] - price19_train[1:1029]))/(abs(train19$newdata1.Price[2:1030]))))*100 
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

test_reshaped120 <- array_reshape(test_matrix1, c(517, 4, 1))  # Sesuaikan dengan data uji Anda
pred120 <- model120 %>% predict(test_reshaped120)

test20 = data.frame(test_matrix1)
test20 <- cbind(test20, pred120 = pred120)
price20 <- test20$pred120 + test20$newdata1.Price
# price20[1:516] - test20$newdata1.Price[2:517]

mse120 = mean((test20$newdata1.Price[2:517] - price20[1:516])^2)
mse120

rmse120 = sqrt((mean(test20$newdata1.Price[2:517] - price20[1:516])^2))
rmse120

mape120 = (mean((abs(test20$newdata1.Price[2:517] - price20[1:516]))/(abs(test20$newdata1.Price[2:517]))))*100 
mape120

train_reshaped120 <- array_reshape(train_matrix1, c(1030, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred120_train <- model120 %>% predict(train_reshaped120)

train20 = data.frame(train_matrix1)
train20 <- cbind(train20, pred120_train)
price20_train = train20$newdata1.Price + train20$pred120_train
# price20_train[1:1029] - train20$newdata1.Price[2:1030]

mse120_train = mean((train20$newdata1.Price[2:1030] - price20_train[1:1029])^2)
mse120_train

rmse120_train = sqrt((mean(train20$newdata1.Price[2:1030] - price20_train[1:1029])^2))
rmse120_train

mape120_train = (mean((abs(train20$newdata1.Price[2:1030] - price20_train[1:1029]))/(abs(train20$newdata1.Price[2:1030]))))*100 
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

test_reshaped121 <- array_reshape(test_matrix1, c(517, 4, 1))  # Sesuaikan dengan data uji Anda
pred121 <- model121 %>% predict(test_reshaped121)

test21 = data.frame(test_matrix1)
test21 <- cbind(test21, pred121 = pred121)
price21 <- test21$pred121 + test21$newdata1.Price
# price21[1:516] - test21$newdata1.Price[2:517]

mse121 = mean((test21$newdata1.Price[2:517] - price21[1:516])^2)
mse121

rmse121 = sqrt((mean(test21$newdata1.Price[2:517] - price21[1:516])^2))
rmse121

mape121 = (mean((abs(test21$newdata1.Price[2:517] - price21[1:516]))/(abs(test21$newdata1.Price[2:517]))))*100 
mape121

train_reshaped121 <- array_reshape(train_matrix1, c(1030, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred121_train <- model121 %>% predict(train_reshaped121)

train21 = data.frame(train_matrix1)
train21 <- cbind(train21, pred121_train)
price21_train = train21$newdata1.Price + train21$pred121_train
# price21_train[1:1029] - train21$newdata1.Price[2:1030]

mse121_train = mean((train21$newdata1.Price[2:1030] - price21_train[1:1029])^2)
mse121_train

rmse121_train = sqrt((mean(train21$newdata1.Price[2:1030] - price21_train[1:1029])^2))
rmse121_train

mape121_train = (mean((abs(train21$newdata1.Price[2:1030] - price21_train[1:1029]))/(abs(train21$newdata1.Price[2:1030]))))*100 
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

test_reshaped122 <- array_reshape(test_matrix1, c(517, 4, 1))  # Sesuaikan dengan data uji Anda
pred122 <- model122 %>% predict(test_reshaped122)

test22 = data.frame(test_matrix1)
test22 <- cbind(test22, pred122 = pred122)
price22 <- test22$pred122 + test22$newdata1.Price
# price22[1:516] - test22$newdata1.Price[2:517]

mse122 = mean((test22$newdata1.Price[2:517] - price22[1:516])^2)
mse122

rmse122 = sqrt((mean(test22$newdata1.Price[2:517] - price22[1:516])^2))
rmse122

mape122 = (mean((abs(test22$newdata1.Price[2:517] - price22[1:516]))/(abs(test22$newdata1.Price[2:517]))))*100 
mape122

train_reshaped122 <- array_reshape(train_matrix1, c(1030, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred122_train <- model122 %>% predict(train_reshaped122)

train22 = data.frame(train_matrix1)
train22 <- cbind(train22, pred122_train)
price22_train = train22$newdata1.Price + train22$pred122_train
# price22_train[1:1029] - train22$newdata1.Price[2:1030]

mse122_train = mean((train22$newdata1.Price[2:1030] - price22_train[1:1029])^2)
mse122_train

rmse122_train = sqrt((mean(train22$newdata1.Price[2:1030] - price22_train[1:1029])^2))
rmse122_train

mape122_train = (mean((abs(train22$newdata1.Price[2:1030] - price22_train[1:1029]))/(abs(train22$newdata1.Price[2:1030]))))*100 
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

test_reshaped123 <- array_reshape(test_matrix1, c(517, 4, 1))  # Sesuaikan dengan data uji Anda
pred123 <- model123 %>% predict(test_reshaped123)

test23 = data.frame(test_matrix1)
test23 <- cbind(test23, pred123 = pred123)
price23 <- test23$pred123 + test23$newdata1.Price
# price23[1:516] - test23$newdata1.Price[2:517]

mse123 = mean((test23$newdata1.Price[2:517] - price23[1:516])^2)
mse123

rmse123 = sqrt((mean(test23$newdata1.Price[2:517] - price23[1:516])^2))
rmse123

mape123 = (mean((abs(test23$newdata1.Price[2:517] - price23[1:516]))/(abs(test23$newdata1.Price[2:517]))))*100 
mape123

train_reshaped123 <- array_reshape(train_matrix1, c(1030, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred123_train <- model123 %>% predict(train_reshaped123)

train23 = data.frame(train_matrix1)
train23 <- cbind(train23, pred123_train)
price23_train = train23$newdata1.Price + train23$pred123_train
# price23_train[1:1029] - train23$newdata1.Price[2:1030]

mse123_train = mean((train23$newdata1.Price[2:1030] - price23_train[1:1029])^2)
mse123_train

rmse123_train = sqrt((mean(train23$newdata1.Price[2:1030] - price23_train[1:1029])^2))
rmse123_train

mape123_train = (mean((abs(train23$newdata1.Price[2:1030] - price23_train[1:1029]))/(abs(train23$newdata1.Price[2:1030]))))*100 
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

test_reshaped124 <- array_reshape(test_matrix1, c(517, 4, 1))  # Sesuaikan dengan data uji Anda
pred124 <- model124 %>% predict(test_reshaped124)

test24 = data.frame(test_matrix1)
test24 <- cbind(test24, pred124 = pred124)
price24 <- test24$pred124 + test24$newdata1.Price
# price24[1:516] - test24$newdata1.Price[2:517]

mse124 = mean((test24$newdata1.Price[2:517] - price24[1:516])^2)
mse124

rmse124 = sqrt((mean(test24$newdata1.Price[2:517] - price24[1:516])^2))
rmse124

mape124 = (mean((abs(test24$newdata1.Price[2:517] - price24[1:516]))/(abs(test24$newdata1.Price[2:517]))))*100 
mape124

train_reshaped124 <- array_reshape(train_matrix1, c(1030, 4, 1))  # Sesuaikan dengan data pelatihan Anda
pred124_train <- model124 %>% predict(train_reshaped124)

train24 = data.frame(train_matrix1)
train24 <- cbind(train24, pred124_train)
price24_train = train24$newdata1.Price + train24$pred124_train
# price24_train[1:1029] - train24$newdata1.Price[2:1030]

mse124_train = mean((train24$newdata1.Price[2:1030] - price24_train[1:1029])^2)
mse124_train

rmse124_train = sqrt((mean(train24$newdata1.Price[2:1030] - price24_train[1:1029])^2))
rmse124_train

mape124_train = (mean((abs(train24$newdata1.Price[2:1030] - price24_train[1:1029]))/(abs(train24$newdata1.Price[2:1030]))))*100 
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

########################################
########################################

library(ggplot2)
dataakhir1 <- tail(newdata, 517)
data_set1 <- data.frame(time = newdata$Date, price = newdata$Price)
predprice1 <- data.frame(time = dataakhir1$Date, price = price5)
combined_plot1 <- ggplot() +
  geom_line(data = data_set1, aes(x = time, y = price, color = "Actual Price")) +
  geom_line(data = predprice1, aes(x = time, y = price, color = "Predicted Price")) +
  scale_color_manual(values = c("Actual Price" = "magenta", "Predicted Price" = "black")) +
  labs(color = "") +
  theme_minimal()

print(combined_plot1)

