# 🏠 House Sales Price Prediction — King County, USA

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)

A complete end-to-end **Data Analysis with Python** project: cleaning, exploring, visualizing, and modeling real-world house sale prices from King County, USA (21,613 records). This repository doubles as a **from-zero-to-hero learning guide** — every concept used in the notebook is explained here in plain language.

---

## 📌 Project Overview

| | |
|---|---|
| **Dataset** | House Sales in King County, USA (May 2014 – May 2015) |
| **Rows** | 21,613 houses |
| **Goal** | Predict house `price` from features like size, location, and condition |
| **Tools** | Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn |
| **Techniques** | Data Wrangling, EDA, Linear Regression, Polynomial Regression, Ridge Regression |

---

## 📂 Repository Structure

```
├── House_Sales_King_County_Final_Project.ipynb   # Main analysis notebook
├── kc_house_data_NaN.csv                          # Dataset
└── README.md                                      # This file
```

---

## 🧠 Concepts: From Zero to Hero

This section explains **every concept** used in the project, in order, starting from the absolute basics.

### 1. What is Data Analysis?

Data analysis is the process of **inspecting, cleaning, transforming, and modeling data** to discover useful information and support decisions. In Python, this is done mainly with the **pandas** library (for tables of data) and **NumPy** (for numerical operations).

### 2. Importing and Loading Data

```python
import pandas as pd
df = pd.read_csv("kc_house_data_NaN.csv")
```

- `pd.read_csv()` reads a CSV (Comma-Separated Values) file into a **DataFrame** — a table-like structure with rows and columns, similar to an Excel sheet.
- `df.head()` shows the first 5 rows.
- `df.dtypes` shows the data type of each column (integer, float, object/text).
- `df.describe()` gives summary statistics (mean, min, max, standard deviation) for every numeric column.

### 3. Data Wrangling (Cleaning)

Real-world data is rarely perfect. Common cleaning tasks:

- **Dropping irrelevant columns** — e.g., an `id` column has no predictive value:
  ```python
  df.drop(["id", "Unnamed: 0"], axis=1, inplace=True)
  ```
- **Handling missing values (NaN)** — Two common strategies:
  - Replace with the **mean** (average) — good for continuous numeric columns like `bedrooms`, `bathrooms`.
  - Replace with the **mode** (most frequent value) — good for categorical columns.
  ```python
  mean_val = df["bedrooms"].mean()
  df["bedrooms"] = df["bedrooms"].fillna(mean_val)
  ```

### 4. Exploratory Data Analysis (EDA)

EDA means **visually and statistically exploring data** before modeling, to understand patterns and relationships.

- **`value_counts()`** — counts how many times each unique value appears in a column.
- **Boxplot** — shows the spread, median, and outliers of a numeric variable across categories:
  ```python
  sns.boxplot(x="waterfront", y="price", data=df)
  ```
- **Regression plot (`regplot`)** — a scatter plot with a fitted trend line, showing the relationship between two numeric variables:
  ```python
  sns.regplot(x="sqft_above", y="price", data=df)
  ```
- **Correlation (`corr()`)** — a number between **-1 and +1** showing how strongly two variables move together:
  - **+1** → perfect positive relationship (both increase together)
  - **-1** → perfect negative relationship (one increases, other decreases)
  - **0** → no linear relationship
  ```python
  df.corr()["price"].sort_values(ascending=False)
  ```

### 5. Simple Linear Regression

Predicts a target variable (`Y`, e.g., `price`) using **one** input feature (`X`, e.g., `sqft_living`) by fitting a straight line:

```
Y = b0 + b1 * X
```

```python
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X, Y)
lm.score(X, Y)   # R² score
```

### 6. R² (R-squared) — Measuring Model Quality

R² tells you **what percentage of the variation** in the target variable is explained by the model.

- **R² = 1** → perfect predictions
- **R² = 0** → model explains nothing (as good as guessing the average)
- Example from this project: `sqft_living` alone gives R² ≈ 0.49 (explains ~49% of price variation).

### 7. Multiple Linear Regression

Uses **several features at once** to predict the target — usually more accurate than a single feature, since real outcomes depend on many factors:

```python
features = ["floors", "waterfront", "lat", "bedrooms", "bathrooms",
            "sqft_living15", "sqft_above", "grade", "sqft_living"]
lm.fit(df[features], df["price"])
```

### 8. Feature Scaling (Standardization)

Different features have very different ranges (e.g., `bedrooms` is 1–8, but `sqft_living` is in the thousands). **`StandardScaler`** rescales every feature to have a mean of 0 and standard deviation of 1, so no single feature unfairly dominates the model.

```python
from sklearn.preprocessing import StandardScaler
```

### 9. Polynomial Regression & Pipelines

Real relationships aren't always straight lines. **Polynomial features** add powers and interactions of the original features (like `x²`, `x·y`) so the model can fit curves, not just lines.

A **Pipeline** chains multiple steps (scaling → polynomial transform → regression) into a single object, so they always run in the right order:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("polynomial", PolynomialFeatures()),
    ("model", LinearRegression())
])
pipe.fit(X, Y)
```

### 10. Overfitting vs. Underfitting

- **Underfitting** — model is too simple, misses real patterns (high error on both training and test data).
- **Overfitting** — model is too complex, "memorizes" the training data (great on training data, poor on new/unseen data).
- The goal is a model that generalizes well — good performance on **unseen** data, not just the data it was trained on.

### 11. Train/Test Split

To check if a model generalizes, we split the data:
- **Training set** — used to fit (teach) the model.
- **Test set** — held back and used only to evaluate performance on unseen data.

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)
```

### 12. Ridge Regression

A improved version of Linear Regression that **penalizes large coefficients** using a hyperparameter called **alpha (α)**. This helps prevent overfitting, especially when using polynomial features or when input features are strongly correlated with each other.

```python
from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(x_train, y_train)
ridge_model.score(x_test, y_test)
```

- **Small alpha** → behaves close to normal Linear Regression (risk of overfitting).
- **Large alpha** → coefficients shrink a lot (risk of underfitting).
- The **best alpha** is found by testing several values and picking the one that gives the highest R² on test/validation data.

### 13. Cross-Validation & Grid Search *(concepts to explore next)*

- **Cross-validation** splits data into multiple folds, trains/tests on different combinations, and averages the results — a more robust way to estimate performance than a single train/test split.
- **GridSearchCV** automatically tries many combinations of hyperparameters (like alpha) using cross-validation, and returns the best-performing combination.

---

## 📊 Results Summary

| Model | R² Score |
|---|---|
| Simple Linear Regression (`long` only) | ~0.0005 |
| Simple Linear Regression (`sqft_living` only) | ~0.493 |
| Multiple Linear Regression (11 features) | ~0.658 |
| Pipeline (scaled polynomial features) | ~0.751 |
| Ridge Regression (α=0.1, test data) | ~0.648 |
| Ridge Regression (2nd-order polynomial, α=0.1, test data) | ~0.700 |
| Ridge Regression (2nd-order polynomial, best α) | ~0.708 |

**Key takeaway:** Location alone (`long`) barely predicts price, but living area (`sqft_living`) is a strong single predictor. Combining multiple features — and carefully tuning model complexity with polynomial features and Ridge regularization — gives the best, most generalizable predictions.

---
## 🚀 How to Run

1. Clone this repository or download the files.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook House_Sales_King_County_Final_Project.ipynb
   ```
4. Run all cells from top to bottom (`Cell → Run All`).

---
## 📚 Dataset Source

Adapted from the **House Sales in King County, USA** dataset, originally sourced from Kaggle, and used as part of the IBM "Data Analysis with Python" course.
I Have Shared with you.
Do it yourself
Thank You 
---

## ✍️ Author

Built as a learning project — from data cleaning to regression modeling — as part of practicing **Python for Data Analysis**.



Follow on LinkedIn: www.linkedin.com/in/sultan-mehmood-817626398
