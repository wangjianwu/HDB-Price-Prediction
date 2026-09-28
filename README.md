# HDB Resale Price Prediction

## Project Overview

This project builds and evaluates regression models for predicting Singapore HDB resale prices. It cleans the source dataset, engineers useful features, preprocesses numerical and categorical variables, trains baseline and tuned regression models, and selects the best-performing model using the validation R-squared score.

The final selected model is evaluated on a separate test set using MAE, MSE, RMSE, and R-squared.

## Project Structure

root/
├── .gitignore
├── README.md
├── requirements.txt
├── eda.ipynb
├── data/
│   └── data.csv
├── src/
│   ├── __init__.py
│   ├── config.yaml
│   ├── data_preparation.py
│   └── model_training.py
└── main.py
```

## File Descriptions

- `eda.ipynb`: Contains exploratory data analysis and visualisation.
- `data/data.csv`: Contains the input HDB resale dataset.
- `src/config.yaml`: Stores the data path, target column, feature groups, data-split settings, cross-validation settings, and hyperparameter grid.
- `src/data_preparation.py`: Cleans the data and creates the Scikit-learn preprocessing pipeline.
- `src/model_training.py`: Splits the data, trains baseline and tuned models, compares model performance, and evaluates the final model.
- `main.py`: Runs the complete machine-learning workflow.
- `requirements.txt`: Lists the required Python packages.
- `src/__init__.py`: Marks `src` as a Python package.

## Data Preparation

The data preparation stage performs the following operations:

1. Removes duplicate rows.
2. Standardises `FOUR ROOM` as `4 ROOM`.
3. Converts negative lease commencement years to positive values.
4. Converts each storey range into its average numerical value.
5. Fills missing town and flat-model names using their corresponding IDs.
6. Extracts year and month from the original month field.
7. Converts remaining lease information into total months.
8. Removes columns that are not used for model training.

The preprocessing pipeline applies:

- Standard scaling to numerical features.
- One-hot encoding to nominal features.
- Ordinal encoding to `flat_type`.
- Passthrough processing to `storey_range`.

## Models

The project trains these baseline regression models:

- Linear Regression
- Ridge Regression
- Lasso Regression

It also uses `GridSearchCV` to tune Ridge and Lasso models. The configured values of `alpha` and `fit_intercept` are tested using 5-fold cross-validation and R-squared scoring.

## Data Split

The configured two-stage split produces:

- 80% training data
- 10% validation data
- 10% test data

A fixed `random_state=42` is used so that the split is reproducible.

## Evaluation Metrics

The models are evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

The model with the highest validation R-squared score is selected and evaluated on the test set.

## Installation

Open PowerShell in the project root folder and install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

If the computer does not permit installation into the system Python environment, pip may automatically use a user-level installation. This is acceptable for running the project.

A suitable `requirements.txt` is:

```text
pandas
PyYAML
scikit-learn
jupyter
matplotlib
seaborn
```

## Running the Project

Ensure that the dataset is saved as:

```text
data/data.csv
```

From the project root folder, run:

```powershell
python main.py
```

The program will log the data preparation, training, validation, hyperparameter tuning, best-model selection, and final test metrics.

## Configuration

The main project settings are stored in `src/config.yaml`. Update this file when changing:

- Dataset location
- Target column
- Feature lists
- Train-validation-test split
- Hyperparameter values
- Cross-validation folds
- Model scoring method

## Notes

- Run `main.py` from the project root so that `./src/config.yaml` and `./data/data.csv` resolve correctly.
- Keep `src/__init__.py` in the `src` folder. It may remain empty for this project.
