# visualize.R: Plotting utilities
plot_feature_importance <- function(model) {
  library(ggplot2)
  importance <- varImp(model, scale = TRUE)
  ggplot(importance, aes(x = reorder(Variables, Importance), y = Importance)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    theme_minimal() +
    labs(title = "Feature Importance", x = "Features", y = "Importance")
}

plot_performance <- function(results, metric) {
  library(ggplot2)
  ggplot(results, aes(x = Model, y = !!sym(metric))) +
    geom_col() +
    theme_minimal() +
    labs(title = paste(metric, "Comparison"), x = "Model", y = metric)
}

