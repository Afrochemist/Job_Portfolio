#Neccessary Librairies
library(keras)
library(lime)
library(tidyquant)
library(rsample)
library(recipes)
library(yardstick)

#Uploading the dataset
churn_data <- read_csv("/Users/afrochemist/Desktop/datasets/ChurnData.csv")
glimpse(churn_data)

#Remove unnecessary customerID column
churn_data_tbl <- churn_data %>%
  select(-customerID) %>%
  drop_na() %>%
  select(Churn, everything())


# Train/test split
set.seed(100)
train_test_split <- initial_split(churn_data_tbl, prop = 0.8)
train_test_split


# Obtaining train and test sets
train_tbl <- training(train_test_split)
test_tbl  <- testing(train_test_split) 

# Initialize the recipe
rec_obj <- recipe(Churn ~ ., data = train_tbl) %>%
  step_discretize(tenure, options = list(cuts = 6)) %>%
  step_log(TotalCharges) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  prep(data = train_tbl)

#Printing the recipe 
rec_obj


#Creating the predictors
#Time to shake and bake
x_train_tbl <- bake(rec_obj, newdata = train_tbl) %>% select(-Churn)
x_test_tbl <- bake(rec_obj, newdata = test_tbl) %>% select(-Churn) 



# Response variables for training and testing sets
y_train_vec <- ifelse(pull(train_tbl, Churn) == "Yes", 1, 0)
y_test_vec  <- ifelse(pull(test_tbl, Churn) == "Yes", 1, 0)


#Now building the Neural network with Keras
model_keras <- keras_model_sequential()

model_keras %>% 
  # First hidden layer
  layer_dense(
    units              = 16, 
    kernel_initializer = "uniform", 
    activation         = "relu", 
    input_shape        = ncol(x_train_tbl)) %>% 
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.1) %>%
  # Second hidden layer
  layer_dense(
    units              = 16, 
    kernel_initializer = "uniform", 
    activation         = "relu") %>% 
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.1) %>%
  # Output layer
  layer_dense(
    units              = 1, 
    kernel_initializer = "uniform", 
    activation         = "sigmoid") %>% 
  # Compile ANN
  compile(
    optimizer = 'adam',
    loss      = 'binary_crossentropy',
    metrics   = c('accuracy')
  )
model_keras


# Fit the keras model to the training data
fit_keras <- fit(
  object           = model_keras, 
  x                = as.matrix(x_train_tbl), 
  y                = y_train_vec,
  batch_size       = 50, 
  epochs           = 35,
  validation_split = 0.30
)


#Looking at the final model
#It is 81% accurate
fit_keras



# Plot the training/validation history of our Keras model
plot(fit_keras) +
  theme_tq() +
  scale_color_tq() +
  scale_fill_tq() +
  labs(title = "Neural Net Training Results")





# Predicted Class
yhat_keras_class_vec <- predict_classes(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()

# Predicted Class Probability
yhat_keras_prob_vec  <- predict_proba(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()


# Format test data and predictions for yardstick metrics
estimates_keras_tbl <- tibble(
  truth      = as.factor(y_test_vec) %>% fct_recode(yes = "1", no = "0"),
  estimate   = as.factor(yhat_keras_class_vec) %>% fct_recode(yes = "1", no = "0"),
  class_prob = yhat_keras_prob_vec
)

estimates_keras_tbl


class(model_keras)


# Setup lime::model_type() function for keras
model_type.keras.models.Sequential <- function(x, ...) {
  return("classification")
}


# Setup lime::predict_model() function for keras
predict_model.keras.models.Sequential <- function(x, newdata, type, ...) {
  pred <- predict_proba(object = x, x = as.matrix(newdata))
  return(data.frame(Yes = pred, No = 1 - pred))
}


# Test our predict_model() function
predict_model(x = model_keras, newdata = x_test_tbl, type = 'raw') %>%
  tibble::as_tibble()


#Below are graphs that show churn rates based on each feature

# Tenure
churn_data %>%
  ggplot(aes(x = Churn, y = tenure)) +
  geom_jitter(alpha = 0.25, color = palette_light()[[6]]) +
  geom_violin(alpha = 0.6, fill = palette_light()[[1]]) +
  theme_tq() +
  labs(
    title = "Tenure",
    subtitle = "Customers with lower tenure are more likely to leave"
  )

# Contract
churn_data %>%
  mutate(Churn = ifelse(Churn == "Yes", 1, 0)) %>%
  ggplot(aes(x = as.factor(Contract), y = Churn)) +
  geom_jitter(alpha = 0.25, color = palette_light()[[6]]) +
  geom_violin(alpha = 0.6, fill = palette_light()[[1]]) +
  theme_tq() +
  labs(
    title = "Contract Type",
    subtitle = "Two and one year contracts much less likely to leave",
    x = "Contract"
  )

# Internet Service
churn_data %>%
  mutate(Churn = ifelse(Churn == "Yes", 1, 0)) %>%
  ggplot(aes(x = as.factor(InternetService), y = Churn)) +
  geom_jitter(alpha = 0.25, color = palette_light()[[6]]) +
  geom_violin(alpha = 0.6, fill = palette_light()[[1]]) +
  theme_tq() +
  labs(
    title = "Internet Service",
    subtitle = "Fiber optic more likely to leave",
    x = "Internet Service"
  )

# Payment Method
churn_data %>%
  mutate(Churn = ifelse(Churn == "Yes", 1, 0)) %>%
  ggplot(aes(x = as.factor(PaymentMethod), y = Churn)) +
  geom_jitter(alpha = 0.25, color = palette_light()[[6]]) +
  geom_violin(alpha = 0.6, fill = palette_light()[[1]]) +
  theme_tq() +
  labs(
    title = "Payment Method",
    subtitle = "Electronic check more likely to leave",
    x = "Payment Method"
  )

# Senior Citizen
churn_data %>%
  mutate(Churn = ifelse(Churn == "Yes", 1, 0)) %>%
  ggplot(aes(x = as.factor(SeniorCitizen), y = Churn)) +
  geom_jitter(alpha = 0.25, color = palette_light()[[6]]) +
  geom_violin(alpha = 0.6, fill = palette_light()[[1]]) +
  theme_tq() +
  labs(
    title = "Senior Citizen",
    subtitle = "Non-senior citizens less likely to leave",
    x = "Senior Citizen (Yes = 1)"
  )

# Online Security
churn_data %>%
  mutate(Churn = ifelse(Churn == "Yes", 1, 0)) %>%
  ggplot(aes(x = OnlineSecurity, y = Churn)) +
  geom_jitter(alpha = 0.25, color = palette_light()[[6]]) +
  geom_violin(alpha = 0.6, fill = palette_light()[[1]]) +
  theme_tq() +
  labs(
    title = "Online Security",
    subtitle = "Customers without online security are more likely to leave"
  )
