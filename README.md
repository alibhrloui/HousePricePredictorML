# House Price Prediction Analysis

This repository contains Python code for analyzing and predicting house prices using various machine learning models. The analysis covers data preprocessing, data cleaning, and model training, aiming to provide accurate predictions for house prices.

## Overview

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
```

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
   cd house-price-prediction
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook:**
   ```bash
   jupyter notebook House_Price_Prediction.ipynb
   ```

## Dataset

The dataset (`HousePricePrediction.csv`) used for analysis contains various features related to house attributes. Initial exploration includes loading the dataset and printing the first 5 records to understand its structure.

## Data Preprocessing and Cleaning

Data preprocessing involves categorizing features into different data types, identifying categorical, integer, and float variables, and creating a correlation heatmap. Data cleaning includes dropping unnecessary columns, handling missing values, and creating a clean dataset for further analysis.

## OneHotEncoder for Categorical Features

Utilizing the `OneHotEncoder` to convert categorical features into a suitable format for machine learning models enhances the accuracy and effectiveness of the predictions.

## Model Training and Evaluation

The repository implements several regression models for predicting house prices:

- **Support Vector Machine (SVM) Regression**
- **Random Forest Regression**
- **Linear Regression**
- **CatBoost Regression**

Each model is trained and evaluated using metrics such as Mean Absolute Percentage Error (MAPE) and R2 Score.

## Contributions

Contributions to the project are welcome! Feel free to fork the repository, create branches, and submit pull requests with improvements or additional features.


## Acknowledgments

Special thanks to the contributors and open-source community for their valuable support and resources.

For more details and insights, refer to the Jupyter Notebook (`HousePricing.ipynb`) in the repository.
