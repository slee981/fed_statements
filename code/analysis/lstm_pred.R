#--------------------------------- Imports -------------------------------------
library(keras)
library(tidyverse)
library(lubridate)

#-------------------------------- Functions ------------------------------------

# override the built in to_categorical 
to_categorical <- function(x) {
    if (x > 0) {
        res <- c(0,0,1)    # increased 
    } else if (x == 0) {
        res <- c(0,1,0)    # no change 
    } else {
        res <- c(1,0,0)    # decreased 
    }
    res <- res %>% 
        as.matrix() 
    return(res)
}

#--------------------------------- Storage -------------------------------------

fname <- './data/asset_prices.csv'
df_raw <- read_csv(fname) 

#---------------------------------- Main ---------------------------------------

# get data
df <- df_raw %>% 
    mutate(Date = as_date(Row, format="%d-%b-%Y", tz = "America/New_York"))  %>%
    select(-Row, -PFNP, -PFNPEC, -SPSPF) %>% 
    drop_na()

# transform into a matrix of price / interest rate changes since "yesterday"
X <- df %>% 
    select(FCM1, FCM5, FCM10) %>% 
    as.matrix() %>% 
    diff()                          # take diff to get change from yesterday 

Y <- df %>% 
    select(SP500) %>% 
    as.matrix() %>%
    diff() %>% 
    sapply(to_categorical) %>% 
    t()

# make sure x and y have the same number of observations 
obs <- ifelse(dim(X)[1] == dim(Y)[1], dim(X)[1], -1)
stopifnot(obs != -1)

# split into training and test
n <- obs * 0.8
x_train <- X[1:n,]
y_train <- Y[1:n,]
x_test <- X[n:obs,]
y_test <- Y[n:obs,]

# setup model
model <- keras_model_sequential() 
model %>% 
    layer_dense(units = 64, activation = 'relu', input_shape = c(3)) %>% 
    layer_dropout(rate = 0.2) %>% 
    layer_dense(units = 32, activation = 'relu') %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 3, activation = 'softmax')

summary(model)

model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
)

# fit model
history <- model %>% fit(
    x_train, y_train, 
    epochs = 30, batch_size = 10, 
    validation_split = 0.2
)

jpeg(file="training_results.jpeg")
plot(history)
dev.off()

model %>% evaluate(x_test, y_test)
