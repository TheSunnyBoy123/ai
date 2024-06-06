## Types

### Simple regression (single independent variable)
1. Simple Linear Regression
2. Simple Non-Linear Regression

### Multiple regression (multiple independent variables)
1. Multiple Linear Regression
2. Multiple Non-Linear Regression


## Applications
- Sales forcasting
- Price Estimation 
- Demand and Supply Estimation
- Income Prediction

## Regression Algorithms
1. Ordinal Regression
2. Poisson Regression
3. Fast Forest Quantile Regression
4. Linear, Polynomial, Lasso, Stepwise, Ridge Regression
5. Bayesian Linear Regression
6. Neural Network Regression
7. Decision Forest Regression
8. Boosted Decision Tree Regression
9. KNN (K-Nearest Neighbors)

## Model Evaluation
1. Training and Testing on the same Dataset
    - High training accuracy
    - Low out-of-sample accuracy
    - Overfitting
2. Train Test dataset split
    - Accurate out-of-sample accuracy
    - Innacuracy generated due to train-test split
    - K-Fold Cross Validation solution
        - Repeatedly train and test the model on different train-test splits to remove the inaccuracy
        - Accuracy = Average the accuracy of all tests

## Evaluation Metrics

1. Mean Absolute Error (MAE)  
$\frac{1}{n}\sum_{j=1}^{n}|y_{j} - y\hat{}_{j}|$

2. Mean Squared Error (MSE)  
$\frac{1}{n}\sum_{j=1}^{n}(y_{j} - y\hat{}_{j})^2$

3. Root Mean Squared Error (RMSE)  
$\sqrt{\frac{1}{n}\sum_{j=1}^{n}(y_{j} - y\hat{}_{j})^2}$
- Has same units as dependent variable

4. Relative Absolute Error (RAE)  
$\frac{\sum_{j=1}^{n}|y_{j} - y\hat{}_{j}|}{\sum_{j=1}^{n}|y_{j} - \bar{y}|}$

5. Relative Squared Error (RSE)
$\frac{\sum_{j=1}^{n}(y_{j} - y\hat{}_{j})^2}{\sum_{j=1}^{n}(y_{j} - \bar{y})^2}$

6. Coefficient of Determination ($R^2$)  
$R^2 = 1 - RSE$
- Describes how well the regression model fits the observed data. Higher the better.

## Outlined Steps for Simple Linear Regression

### Import Libraries
~~~
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
~~~

### Import Dataset
~~~
df = pd.read_csv("fileName.csv")
df.head() #this shows a preview of the dataset
df.describe() # see a stastical summary of the data
~~~

