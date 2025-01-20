# train_models.R: Functions to train regression and classification models
train_regression_model <- function(data, method = "lm", metric = "RMSE", ctrl = NULL) {
  library(caret)
  set.seed(2001)
  ctrl <- ctrl %||% trainControl(method = "repeatedcv", number = 10, repeats = 5)
  model <- train(y ~ ., data = data, method = method, metric = metric, trControl = ctrl)
  return(model)
}

train_classification_model <- function(data, method = "rf", metric = "ROC", ctrl = NULL) {
  library(caret)
  set.seed(123)
  ctrl <- ctrl %||% trainControl(method = "repeatedcv", number = 10, repeats = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
  model <- train(outcome ~ ., data = data, method = method, metric = metric, trControl = ctrl)
  return(model)
}

