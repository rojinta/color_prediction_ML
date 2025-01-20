load_data <- function(file_path) {
  library(readr)
  data <- read_csv(file_path, col_names = TRUE)
  return(data)
}
