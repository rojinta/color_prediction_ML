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
library(pROC)
library(rstanarm)
library(loo)

# Load Required Scripts
source("../src/load_data.R")
source("../src/preprocess_data.R")

# Load and Preprocess Data
preprocessed_file <- "../data/processed/classification_data.csv"
data_path <- Sys.getenv("DATA_PATH", "../data/raw/paint_project_train_data.csv")

if (file.exists(preprocessed_file)) {
  classification_data <- read.csv(preprocessed_file)
} else {
  training_data <- load_data(data_path)
  processed_data <- preprocess_data(training_data)
  classification_data <- processed_data$classification
}

# Train Control for Cross-Validation
set.seed(123)
train_ctrl_roc <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)

train_ctrl_acc <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5,
  classProbs = TRUE,
  summaryFunction = defaultSummary,
  savePredictions = TRUE
)

# Define Metrics for Evaluation
my_metric_roc <- "ROC"
my_metric_acc <- "Accuracy"

# Train Models
models_roc <- list()
models_acc <- list()

# Generalized Linear Models
models_roc$glm_mod01 <- train(outcome ~ Lightness + Saturation, data = classification_data, method = "glm", family = binomial, metric = my_metric_roc, trControl = train_ctrl_roc, preProcess = c("center", "scale"))
models_roc$glm_mod02 <- train(outcome ~ R + G + B + Hue, data = classification_data, method = "glm", family = binomial, metric = my_metric_roc, trControl = train_ctrl_roc, preProcess = c("center", "scale"))
models_roc$glm_mod03 <- train(outcome ~ (Lightness + Saturation) + (R + G + B + Hue)^2, data = classification_data, method = "glm", family = binomial, metric = my_metric_roc, trControl = train_ctrl_roc, preProcess = c("center", "scale"))
models_roc$glm_mod04 <- train(outcome ~ (Lightness + Saturation) * (R + G + B + Hue)^2, data = classification_data, method = "glm", family = binomial, metric = my_metric_roc, trControl = train_ctrl_roc, preProcess = c("center", "scale"))

models_acc$glm_mod01 <- train(outcome ~ Lightness + Saturation, data = classification_data, method = "glm", family = binomial, metric = my_metric_acc, trControl = train_ctrl_acc, preProcess = c("center", "scale"))
models_acc$glm_mod02 <- train(outcome ~ R + G + B + Hue, data = classification_data, method = "glm", family = binomial, metric = my_metric_acc, trControl = train_ctrl_acc, preProcess = c("center", "scale"))
models_acc$glm_mod03 <- train(outcome ~ (Lightness + Saturation) + (R + G + B + Hue)^2, data = classification_data, method = "glm", family = binomial, metric = my_metric_acc, trControl = train_ctrl_acc, preProcess = c("center", "scale"))
models_acc$glm_mod04 <- train(outcome ~ (Lightness + Saturation) * (R + G + B + Hue)^2, data = classification_data, method = "glm", family = binomial, metric = my_metric_acc, trControl = train_ctrl_acc, preProcess = c("center", "scale"))

# Regularized Regression
enet_grid <- expand.grid(alpha = seq(0, 1, length = 5), lambda = seq(0.001, 0.1, length = 5))
models_roc$enet_mod <- train(outcome ~ (Lightness + Saturation) * (R + G + B + Hue)^2, data = classification_data, method = "glmnet", metric = my_metric_roc, trControl = train_ctrl_roc, preProcess = c("center", "scale"), tuneGrid = enet_grid)
models_acc$enet_mod <- train(outcome ~ (Lightness + Saturation) * (R + G + B + Hue)^2, data = classification_data, method = "glmnet", metric = my_metric_acc, trControl = train_ctrl_acc, preProcess = c("center", "scale"), tuneGrid = enet_grid)

# Advanced Models
models_roc$rf_mod <- train(outcome ~ ., data = classification_data, method = "rf", metric = my_metric_roc, trControl = train_ctrl_roc, preProcess = c("center", "scale"))
models_acc$rf_mod <- train(outcome ~ ., data = classification_data, method = "rf", metric = my_metric_acc, trControl = train_ctrl_acc, preProcess = c("center", "scale"))

models_roc$gbm_mod <- train(outcome ~ ., data = classification_data, method = "gbm", metric = my_metric_roc, trControl = train_ctrl_roc, preProcess = c("center", "scale"))
models_acc$gbm_mod <- train(outcome ~ ., data = classification_data, method = "gbm", metric = my_metric_acc, trControl = train_ctrl_acc, preProcess = c("center", "scale"))

models_roc$svm_mod <- train(outcome ~ ., data = classification_data, method = "svmRadial", metric = my_metric_roc, trControl = train_ctrl_roc, preProcess = c("center", "scale"))
models_acc$svm_mod <- train(outcome ~ ., data = classification_data, method = "svmRadial", metric = my_metric_acc, trControl = train_ctrl_acc, preProcess = c("center", "scale"))

models_roc$mars_mod <- train(outcome ~ ., data = classification_data, method = "earth", metric = my_metric_roc, trControl = train_ctrl_roc, preProcess = c("center", "scale"))
models_acc$mars_mod <- train(outcome ~ ., data = classification_data, method = "earth", metric = my_metric_acc, trControl = train_ctrl_acc, preProcess = c("center", "scale"))

# Evaluate Models
model_results_roc <- lapply(models_roc, function(mod) max(mod$results$ROC))
model_results_acc <- lapply(models_acc, function(mod) max(mod$results$Accuracy))

results_df <- data.frame(
  Model = names(models_roc),
  AUC = unlist(model_results_roc),
  Accuracy = unlist(model_results_acc)
)

results_df <- results_df[order(-results_df$AUC), ]
print("Model Performance (AUC and Accuracy):")
print(results_df)

# Save Results
results_file <- "../results/classification_metrics_results.csv"
write.csv(results_df, results_file, row.names = FALSE)

# Save All Models
dir.create("../results/models/classification", recursive = TRUE, showWarnings = FALSE)
for (model_name in names(models_roc)) {
  saveRDS(models_roc[[model_name]], file.path("../results/models/classification", paste0(model_name, ".rds")))
}
for (model_name in names(models_acc)) {
  saveRDS(models_acc[[model_name]], file.path("../results/models/classification", paste0(model_name, "_acc.rds")))
}
