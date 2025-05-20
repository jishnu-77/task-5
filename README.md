# Titanic Dataset Visualization

This project merges the **Titanic train and test datasets**, performs basic **data cleaning**, and creates several **visualizations** to understand the structure and relationships in the data.


## Dataset
- **train.csv** and **test.csv** from the Titanic dataset.
- Columns include features like `Survived`, `Age`, `Fare`, `Pclass`, etc.


## Steps and Code Explanation

### 1. **Import Required Libraries**
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
```
> Used for data manipulation, visualization, and numerical operations.


### 2. **Load the Datasets**
```python
df_train = pd.read_csv("/path/to/train.csv")
df_test = pd.read_csv("/path/to/test.csv")
```
> Load the train and test CSV files into separate DataFrames.


### 3. **Merge the Datasets**
```python
df_titanic = pd.concat([df_train, df_test], ignore_index=True)
```
> Combine the two datasets into one, for easier overall analysis.


### 4. **Basic Exploration**
```python
df_titanic.describe()
df_titanic.info()
```
> Get summary statistics and general info about data types and missing values.


### 5. **Data Cleaning**
- **Fill missing Age values** with the **mean** and cast to integer.
- **Map Pclass** values (1, 2, 3) to string labels ('First', 'Second', 'Third').
- **Fill missing Survived values** with the mean and cast to integer.
- **Ensure Fare is float type** for consistency.

```python
df_titanic["Age"] = df_titanic["Age"].fillna(df_titanic["Age"].mean()).astype(int)

mapping = {1: "First", 2: "Second", 3: "Third", np.nan: "NA"}
df_titanic["Pclass"] = df_titanic["Pclass"].map(mapping)

df_titanic["Fare"] = df_titanic["Fare"].astype(float)
df_titanic["Survived"] = df_titanic["Survived"].fillna(df_titanic["Survived"].mean()).astype(int)
```


### 6. **Identify Numeric and Categorical Columns**
```python
numeric_cols = df_titanic.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = df_titanic.select_dtypes(include=['object']).columns.tolist()
```
> Useful for plotting and correlation analysis later.


## Visualizations

### Box Plot of Fare
```python
sns.boxplot(y=df_titanic['Fare'])
```
- Shows distribution, outliers, and median Fare values.

![Box Plot](images/boxplot.png)

### Histogram of Age
```python
sns.histplot(data=df_titanic, x='Age', kde=True)
```
- Displays distribution of passengers' ages.
- KDE (Kernel Density Estimate) curve shows smooth distribution.
  
![Histogram](images/hist.png)

### Scatter Plot: Age vs Survived
```python
sns.scatterplot(data=df_titanic, x='Age', y='Survived')
```
- Shows the relationship between Age and survival.
  
![Scatter Plot](images/scatter_plot.png)

### Pair Plot
```python
sns.pairplot(df_titanic, diag_kind='kde')
```
- Plots scatter plots and KDE plots for all numeric features.
- Useful for visual correlation analysis between all numeric variables.
  
![Pair Plot](images/pairplot.png)

### Correlation Heatmap
```python
correlation_matrix = df_titanic[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
```
- Shows the correlation matrix between numeric variables.
- Positive correlations (red) and negative correlations (blue).
 
![Correlation Heatmap](images/corr_heatmap.png)

## Summary of Insights
- **Fare** has outliers (high-paying passengers).
- **Age** is relatively normally distributed but slightly skewed toward younger ages.
- **Younger passengers** (children) seem slightly more likely to survive.
- Some features show moderate correlation (e.g., Fare and Pclass).
- **Survival** has correlations with features like **Pclass** and **Fare**.


## Requirements
- Python 3.x
- pandas
- matplotlib
- seaborn
- numpy


## How to Run
1. Place the `train.csv` and `test.csv` files inside your working directory (or update the file paths).
2. Run the Python script or Jupyter notebook.
3. Visualizations will appear one by one.
