# Load Required Libraries
library(dplyr)
library(readr)
library(caret)
library(tibble)

# Paths
holdout_path <- "../data/raw/holdout_data.csv"
output_path <- "../results/holdout/"
regression_model_path <- "../results/models/regression/linear_mod04.rds"
classification_model_path <- "../results/models/classification/rf_mod.rds"
dir.create(output_path, recursive = TRUE, showWarnings = FALSE)

# Load Holdout Data
holdout_data <- read_csv(holdout_path)

# Load Pre-trained Models
mod04 <- readRDS(regression_model_path)
rf_mod <- readRDS(classification_model_path)

# 1. Predict Holdout Data
## Regression Predictions
regression_preds <- predict(mod04, newdata = holdout_data)

## Classification Predictions
classification_preds <- predict(rf_mod, newdata = holdout_data)
classification_probs <- predict(rf_mod, newdata = holdout_data, type = "prob")

# 2. Compile and Save Predictions
final_preds <- tibble::tibble(
  y_pred = regression_preds,
  class_pred = classification_preds,
  event_prob = classification_probs$event
) %>%
  rowid_to_column("id")

# Save Predictions
write_csv(final_preds, file.path(output_path, "holdout_predictions.csv"))

# Print Confirmation
print("Holdout predictions saved successfully.")
