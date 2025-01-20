# preprocess_data.R: Data preprocessing utilities
preprocess_data <- function(data) {
  library(dplyr)
  library(boot)
  
  # Preprocess for regression
  regression_data <- data %>%
    mutate(
      y = boot::logit((response - 0) / (100 - 0)),
      Lightness = as.factor(Lightness),
      Saturation = as.factor(Saturation)
    ) %>%
    select(R, G, B, Lightness, Saturation, Hue, y)
  
  # Preprocess for classification
  classification_data <- data %>%
    select(-response) %>%
    mutate(
      outcome = ifelse(outcome == 1, "event", "non_event"),
      outcome = factor(outcome, levels = c("event", "non_event")),
      Lightness = as.factor(Lightness),
      Saturation = as.factor(Saturation)
    )
  
  # Save preprocessed data
  output_dir <- "../data/processed"
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  write.csv(regression_data, file.path(output_dir, "regression_data.csv"), row.names = FALSE)
  write.csv(classification_data, file.path(output_dir, "classification_data.csv"), row.names = FALSE)
  
  return(list(regression = regression_data, classification = classification_data))
}
