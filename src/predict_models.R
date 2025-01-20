# predict_models.R: Functions for making predictions
predict_holdout <- function(reg_model, class_model, holdout_data) {
  regression_preds <- predict(reg_model, newdata = holdout_data)
  classification_preds <- predict(class_model, newdata = holdout_data)
  probabilities <- predict(class_model, newdata = holdout_data, type = "prob")[, "event"]
  predictions <- tibble::tibble(
    id = seq_len(nrow(holdout_data)),
    y = regression_preds,
    outcome = classification_preds,
    probability = probabilities
  )
  return(predictions)
}

