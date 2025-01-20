# Load Required Libraries
library(dplyr)
library(ggplot2)
library(caret)
library(glmnet)
library(nnet)
library(randomForest)
library(gbm)
library(e1071)
library(earth)

# Load Required Scripts
source("../src/load_data.R")
source("../src/preprocess_data.R")

# Load Data
preprocessed_file <- "../data/processed/regression_data.csv"
data_path <- Sys.getenv("DATA_PATH", "../data/raw/train_data.csv")

if (file.exists(preprocessed_file)) {
  regression_data <- read.csv(preprocessed_file)
} else {
  training_data <- load_data(data_path)
  processed_data <- preprocess_data(training_data)
  regression_data <- processed_data$regression
}

# Train Models
set.seed(123)
train_ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

# Linear Models
linear_mod01 <- train(y ~ Lightness + Saturation, data = regression_data, method = "lm", preProcess = c("center", "scale"), metric = "RMSE", trControl = train_ctrl)
linear_mod02 <- train(y ~ R + G + B + Hue, data = regression_data, method = "lm", preProcess = c("center", "scale"), metric = "RMSE", trControl = train_ctrl)
linear_mod03 <- train(y ~ (Lightness + Saturation) + (R + G + B + Hue)^2, data = regression_data, method = "lm", preProcess = c("center", "scale"), metric = "RMSE", trControl = train_ctrl)
linear_mod04 <- train(y ~ (Lightness + Saturation) * (R + G + B + Hue)^2, data = regression_data, method = "lm", preProcess = c("center", "scale"), metric = "RMSE", trControl = train_ctrl)

# Elastic Net
enet_model <- train(
  y ~ (Lightness + Saturation) * (R + G + B + Hue)^2,
  data = regression_data,
  method = "glmnet",
  metric = "RMSE",
  trControl = train_ctrl,
  tuneGrid = expand.grid(alpha = seq(0, 1, length = 5), lambda = seq(0.001, 0.1, length = 5))
)

# Random Forest
rf_model <- train(y ~ ., data = regression_data, method = "rf", metric = "RMSE", trControl = train_ctrl)

# Gradient Boosted Trees
gbm_model <- train(y ~ ., data = regression_data, method = "gbm", metric = "RMSE", trControl = train_ctrl)

# Neural Network
nn_model <- train(y ~ ., data = regression_data, method = "nnet", metric = "RMSE", trControl = train_ctrl)

# Support Vector Machines
svm_model <- train(y ~ ., data = regression_data, method = "svmRadial", metric = "RMSE", trControl = train_ctrl)

# MARS
mars_model <- train(y ~ ., data = regression_data, method = "earth", metric = "RMSE", trControl = train_ctrl)

# Evaluate Models
model_results <- data.frame(
  Model = c("Linear Mod 1", "Linear Mod 2", "Linear Mod 3", "Linear Mod 4", "Elastic Net", "Random Forest", "GBM", "Neural Network", "SVM", "MARS"),
  RMSE = c(
    linear_mod01$results$RMSE,
    linear_mod02$results$RMSE,
    linear_mod03$results$RMSE,
    linear_mod04$results$RMSE,
    min(enet_model$results$RMSE),
    min(rf_model$results$RMSE),
    min(gbm_model$results$RMSE),
    min(nn_model$results$RMSE),
    min(svm_model$results$RMSE),
    min(mars_model$results$RMSE)
  )
)

# Sort Results by RMSE
model_results <- model_results[order(model_results$RMSE), ]
print("Model Performance (RMSE):")
print(model_results)

# Save Results
results_file <- "../results/regression_metrics_results.csv"
write.csv(model_results, results_file, row.names = FALSE)

# Save All Models
dir.create("../results/models/regression", recursive = TRUE, showWarnings = FALSE)
saveRDS(linear_mod01, "../results/models/regression/linear_mod01.rds")
saveRDS(linear_mod02, "../results/models/regression/linear_mod02.rds")
saveRDS(linear_mod03, "../results/models/regression/linear_mod03.rds")
saveRDS(linear_mod04, "../results/models/regression/linear_mod04.rds")
saveRDS(enet_model, "../results/models/regression/elastic_net_model.rds")
saveRDS(rf_model, "../results/models/regression/random_forest_model.rds")
saveRDS(gbm_model, "../results/models/regression/gbm_model.rds")
saveRDS(nn_model, "../results/models/regression/neural_network_model.rds")
saveRDS(svm_model, "../results/models/regression/svm_model.rds")
saveRDS(mars_model, "../results/models/regression/mars_model.rds")
