# Load Required Libraries
library(dplyr)
library(ggplot2)
library(tidyr)
library(reshape2)

# Load Required Scripts
source("../src/load_data.R")

# Load Data
# Set data path using an environment variable or default
data_path <- Sys.getenv("DATA_PATH", "../data/raw/train_data.csv")
training_data <- load_data(data_path)

# Visualize Data

# Distribution of Variables
plot_variable_distributions <- function(data) {
  data %>%
    select(where(is.numeric)) %>%
    tidyr::gather(key = "Variable", value = "Value") %>%
    ggplot(aes(x = Value)) +
    geom_histogram(fill = "blue", color = "white", bins = 30) +
    facet_wrap(~ Variable, scales = "free", ncol = 3) +
    theme_minimal() +
    labs(title = "Distributions of Numeric Variables", x = "Value", y = "Count")
}

# Correlation Heatmap
plot_correlation_heatmap <- function(data) {
  corr_matrix <- data %>%
    select(where(is.numeric)) %>%
    cor()
  
  melted_corr <- melt(corr_matrix)
  
  ggplot(melted_corr, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile(color = "white") +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1, 1)) +
    theme_minimal() +
    labs(title = "Correlation Heatmap", x = "Variable", y = "Variable", fill = "Correlation")
}

# Relationships Between Variables
plot_relationships <- function(data, x_var, y_var) {
  ggplot(data, aes_string(x = x_var, y = y_var)) +
    geom_point(alpha = 0.7) +
    theme_minimal() +
    labs(title = paste("Relationship Between", x_var, "and", y_var), x = x_var, y = y_var)
}

# Generate Plots
dist_plot <- plot_variable_distributions(training_data)
heatmap_plot <- plot_correlation_heatmap(training_data)

# Save Outputs
dir.create("../results/plots/exploration", recursive = TRUE, showWarnings = FALSE)
ggsave("../results/plots/exploration/distributions.png", dist_plot, width = 8, height = 6)
ggsave("../results/plots/exploration/correlation_heatmap.png", heatmap_plot, width = 8, height = 6)

# Example Relationship Plot
plot_relationships(training_data, "R", "G")
