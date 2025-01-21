# Exploring Paint Color Trends: Regression, Classification, and Interpretability

## Overview
This project explores machine learning techniques to analyze and predict trends in a paint color library from PPG Paints. The project involves the following tasks:

1. **Exploration**: Understand the dataset through visualizations and statistical summaries.
2. **Regression**: Predict a continuous response variable representing a paint property.
3. **Classification**: Predict a binary outcome (`event` or `non_event`) indicating whether a paint color is popular.
4. **Interpretation**: Analyze model insights, including variable importance and prediction accuracy for different combinations of input features.
5. **Holdout Prediction**: Use trained models to generate predictions for a holdout dataset.

Note: The original dataset from PPG Paints is not included in this repository due to privacy constraints. However, a synthetic dataset can be generated for testing purposes using the included `generate_synthetic_data.R` script.

## Project Structure
The repository is organized as follows:

```
├── data/
│   ├── raw/
│   ├── processed/
├── src/
├── scripts/
├── results/
│   ├── models/
│   |   ├── regression/
│   |   ├── classification/
│   ├── plots/
│   |   ├── exploration/
│   |   ├── interpretation/
│   ├── metrics/
│   ├── holdout/
├── README.md
├── .gitignore
```

## Key Files and Scripts

### Source Scripts (`src/`)
- **`load_data.R`**: Loads raw data from the `data/raw/` directory.
- **`preprocess_data.R`**: Preprocesses data for regression and classification tasks.
- **`generate_synthetic_data.R`**: Generates synthetic data to replicate testing scenarios.

### Workflow Scripts (`scripts/`)
1. **`exploration.R`**:
   - Visualize distributions, correlations, and relationships between variables.
   - Save plots to `results/plots/exploration/`.

2. **`regression.R`**:
   - Train models to predict the continuous variable `response`, saving trained models to `results/models/regression/`.
   - Evaluate models using RMSE and save results to `results/metrics/regression_metrics_results.csv`.

3. **`classification.R`**:
   - Train models to classify the binary variable `outcome` and save trained models to `results/models/classification/`.
   - Evaluate models using AUC-ROC and accuracy metrics, saving results to `results/metrics/classification_metrics_results.csv`.

4. **`interpretation.R`**:
   - Analyze model insights (e.g., variable importance, hardest/easiest combinations).
   - Save plots and metrics to `results/plots/interpretation/` and `results/metrics/interpretation/`.

5. **`holdout_prediction.R`**:
   - Use saved models to generate predictions for a holdout dataset.
   - Save predictions to `results/holdout/holdout_predictions.csv`.

## Results
The trained models were evaluated on a holdout dataset to assess their performance in predicting both the continuous paint property and paint color popularity. Both regression and classification tasks demonstrated strong predictive capabilities:

- Regression Task: The best regression model achieved an impressive RMSE of 0.053 and an R² of 0.998, indicating exceptional accuracy in predicting the continuous paint property with minimal error.
- Classification Task: The best classification model performed robustly, achieving an accuracy of 87%, an AUC-ROC of 0.91, and a high specificity of 92.6%, showcasing its reliability in predicting paint color popularity.

Detailed performance metrics can be found in the `results/holdout/holdout_metrics.csv` file.

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/rojinta/color_prediction_ML.git
   ```
2. Set up the required R environment and install dependencies:
   ```r
   install.packages(c("readr", "dplyr", "ggplot2", "tidyr", "tidyverse", "caret", "boot", "tibble", "reshape2", "coefplot", "loo", "randomForest", "glmnet", "e1071", "gbm", "nnet", "earth", "rstanarm", "pROC"))
   ```
3. Add raw data to `data/raw/` (excluded from Git).
    - You can generate synthetic data using the following script:
    ```
    source("src/generate_synthetic_data.R")
    ```

4. Run workflow scripts in sequence:
   - `scripts/exploration.R`
   - `scripts/regression.R`
   - `scripts/classification.R`
   - `scripts/interpretation.R`
   - `scripts/holdout_prediction.R`
