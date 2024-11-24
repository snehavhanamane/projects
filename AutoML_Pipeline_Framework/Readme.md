# AutoML Pipeline Framework

## Overview

The AutoML Pipeline Framework simplifies machine learning workflows by automating data preprocessing, feature engineering, model selection, hyperparameter tuning, and evaluation. It supports both classification and regression tasks, using configurations defined in JSON files.

## Features

- **Configurable Workflow**: Use JSON files to define feature handling, model selection, and hyperparameter tuning strategies.
- **Supported Algorithms**: Includes popular algorithms like Random Forest, XGBoost, Gradient Boosting, Neural Networks, and more.
- **Feature Engineering**: Handles missing data, scaling, interaction generation, and feature reduction techniques like PCA and tree-based methods.
- **Customizable Hyperparameters**: Tune hyperparameters using grid search or custom strategies.
- **Metrics and Evaluation**: Supports metrics like accuracy, R², F1 score, and mean squared error.

## Requirements

- Python 3.8 or later
- Libraries: Scikit-learn, XGBoost, pandas, numpy, etc.

## Installation

1. Clone the repository:
   
   git clone https://github.com/your-username/automl-pipeline-framework.git AutoML_Pipeline_Framework
   cd automl-pipeline-framework

AutoML_Pipeline_Framework https://github.com/snehavhanamane/projects.git

**Install the required Python packages**:

pip install -r requirements.txt

### Usage

**Prepare your dataset**: Ensure the dataset is in CSV format and matches the configurations in the JSON file.

**Update the configuration**: Modify algoparams_from_ui.json to define features, target variable, algorithms, and hyperparameter settings.

### Run the pipeline:

python AutoML_Pipeline_Framework.py

Replace the csv_path and config_path variables in the script with the paths to your dataset and configuration files.

### Configuration Details

The configuration file (algoparams_from_ui.json) contains:
**Feature Handling**: Imputation, scaling, interaction terms, etc.
**Algorithms**: Enable or disable specific models and set their hyperparameters.
**Hyperparameter Tuning**: Specify the grid search strategy and ranges.
**Feature Reduction**: Methods like PCA or tree-based feature selection.

### Output

The script outputs:

Best hyperparameters for selected models.
Evaluation metrics (e.g., accuracy, R², or F1 score).
Predictions on the test dataset.

### Example:

Modify algoparams_from_ui.json as follows:

{
  "target": {
    "prediction_type": "Regression",
    "target": "petal_width"
  },
  "algorithms": {
    "RandomForestRegressor": {
      "is_selected": true,
      "min_trees": 10,
      "max_trees": 20
    }
  }
}

### Run the script:

python AutoML_Pipeline_Framework.py

### Acknowledgments

Scikit-learn: For providing robust tools for machine learning.
XGBoost: For efficient and scalable boosting algorithms.
