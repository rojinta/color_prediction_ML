generate_synthetic_data <- function(output_path) {
  set.seed(123)
  synthetic_data <- data.frame(
    R = runif(100, 0, 255),
    G = runif(100, 0, 255),
    B = runif(100, 0, 255),
    Hue = runif(100, 0, 360),
    Saturation = runif(100, 0, 100),
    Lightness = runif(100, 0, 100),
    response = runif(100, 0, 100),
    outcome = sample(0:1, 100, replace = TRUE)
  )
  write.csv(synthetic_data, output_path, row.names = FALSE)
}

# Generate synthetic data
dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)
generate_synthetic_data("data/raw/synthetic_data.csv")
