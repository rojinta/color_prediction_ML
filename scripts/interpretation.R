# Load Required Libraries
library(dplyr)
library(ggplot2)
library(caret)
library(tidyr)
library(tibble)

# Load Required Scripts
source("../src/load_data.R")
source("../src/preprocess_data.R")

# Load and Preprocess Data
data_path <- Sys.getenv("DATA_PATH", "../data/raw/paint_project_train_data.csv")
processed_files <- list(
  regression = "../data/processed/regression_data.csv",
  classification = "../data/processed/classification_data.csv"
)

# Load Preprocessed Data
if (file.exists(processed_files$regression)) {
  regression_data <- read.csv(processed_files$regression)
} else {
  training_data <- load_data(data_path)
  processed_data <- preprocess_data(training_data)
  regression_data <- processed_data$regression
}

if (file.exists(processed_files$classification)) {
  classification_data <- read.csv(processed_files$classification)
} else {
  training_data <- load_data(data_path)
  processed_data <- preprocess_data(training_data)
  classification_data <- processed_data$classification
}

# Set Output Directory for Plots
dir.create("../results/plots/interpretation", recursive = TRUE, showWarnings = FALSE)

# Input Importance
## Best Regression Model (Linear Model 4)
set.seed(123)
mod04 <- train(
  y ~ (Lightness + Saturation) * (R + G + B + Hue)^2,
  data = regression_data,
  method = "lm",
  preProcess = c("center", "scale"),
  trControl = trainControl(method = "repeatedcv", number = 10, repeats = 5, savePredictions = "final")
)

## Coefficient Summary
coef_df <- as.data.frame(summary(mod04$finalModel)$coefficients)
coef_df <- coef_df %>%
  rownames_to_column("Term") %>%
  arrange(desc(abs(Estimate)))

# Save Coefficient Plot
ggplot(coef_df, aes(x = reorder(Term, Estimate), y = Estimate)) +
  geom_bar(stat = "identity", fill = "blue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Coefficient Importance for Regression Model", x = "Term", y = "Estimate")
ggsave("../results/plots/interpretation/regression_importance.png", width = 8, height = 6)

## Best Classification Model (Random Forest)
set.seed(123)
rf_mod <- train(
  outcome ~ .,
  data = classification_data,
  method = "rf",
  trControl = trainControl(
    method = "repeatedcv",
    number = 10,
    repeats = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary,
    savePredictions = "final"
  ),
  metric = "ROC"
)

## Variable Importance
rf_importance <- varImp(rf_mod)
ggplot(rf_importance) +
  labs(title = "Variable Importance for Classification Model", x = "Variable", y = "Importance")
ggsave("../results/plots/interpretation/classification_importance.png", width = 8, height = 6)

# Hardest and Easiest Combinations
## Regression Hardest/Easiest
regression_data$rowIndex <- 1:nrow(regression_data)
predicted_regression <- mod04$pred %>%
  left_join(regression_data, by = "rowIndex") %>%
  mutate(squared_error = (obs - pred)^2)

grouped_regression <- predicted_regression %>%
  group_by(Lightness, Saturation) %>%
  summarize(
    RMSE = sqrt(mean(squared_error, na.rm = TRUE))
  )

hardest_regression <- grouped_regression %>% slice_max(RMSE, n = 1)
easiest_regression <- grouped_regression %>% slice_min(RMSE, n = 1)

## Classification Hardest/Easiest
classification_data$rowIndex <- 1:nrow(classification_data)
predicted_classification <- rf_mod$pred %>%
  left_join(classification_data, by = "rowIndex") %>%
  mutate(correct = pred == obs)

grouped_classification <- predicted_classification %>%
  group_by(Lightness, Saturation) %>%
  summarize(Accuracy = mean(correct, na.rm = TRUE))

hardest_classification <- grouped_classification %>% slice_min(Accuracy, n = 1)
easiest_classification <- grouped_classification %>% slice_max(Accuracy, n = 1)

# Save Hardest/Easiest Results
dir.create("../results/metrics", recursive = TRUE, showWarnings = FALSE)
write.csv(hardest_regression, "../results/metrics/hardest_regression.csv", row.names = FALSE)
write.csv(easiest_regression, "../results/metrics/easiest_regression.csv", row.names = FALSE)
write.csv(hardest_classification, "../results/metrics/hardest_classification.csv", row.names = FALSE)
write.csv(easiest_classification, "../results/metrics/easiest_classification.csv", row.names = FALSE)

# Visualizations
## Regression Predictions
ggplot(hardest_regression, aes(x = Lightness, y = RMSE, fill = Saturation)) +
  geom_col() +
  labs(title = "Hardest Combinations for Regression Model")
ggsave("../results/plots/interpretation/hardest_regression.png", width = 8, height = 6)
