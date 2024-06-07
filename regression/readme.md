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
$\frac{1}{n}\sum_{j=1}^{n}|y_{j} - \hat{y}_{j}|$

2. Mean Squared Error (MSE)  
$\frac{1}{n}\sum_{j=1}^{n}(y_{j} - \hat{y}_{j})^2$

3. Root Mean Squared Error (RMSE)  
$\sqrt{\frac{1}{n}\sum_{j=1}^{n}(y_{j} - \hat{y}_{j})^2}$
- Has same units as dependent variable

4. Relative Absolute Error (RAE)  
$\frac{\sum_{j=1}^{n}|y_{j} - \hat{y}_{j}|}{\sum_{j=1}^{n}|y_{j} - \bar{y}|}$

5. Relative Squared Error (RSE)
$\frac{\sum_{j=1}^{n}(y_{j} - \hat{y}_{j})^2}{\sum_{j=1}^{n}(y_{j} - \bar{y})^2}$

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

Plotting features as histograms:
~~~
visualise = df[['feature1', 'feature2', 'feature3', 'feature4']]
visualise.hist()
plt.show()
~~~

Plotting features against each other:
~~~
plt.scatter(df.feature1, df.feature2, color='color_of_points')
plt.xlabel("feature1")
plt.ylabel("feature2")
plt.show()
~~~

Create train and test dataset:
~~~
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]
~~~

### Simple Regression Model

Use sklearn to model data:
~~~
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['feature1']])
train_y = np.asanyarray(train[['dependent_variable']])
regr.fit(train_x, train_y)
print(regr.coef_, regr.intercept_) #m and c values for the line of best fit y=mx+c
~~~

Plot this line
~~~
plt.scatter(train.feature1, train.dependent_variable, color='color_of_points')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("feature1")
plt.ylabel("dependent_variable")
plt.show()
~~~

### Evaluation

Calculate metrics to check accuracy:
~~~
from sklearn.metrics import r2_score
test_x = np.asanyarray(test[['feature1']])
test_y = np.asanyarray(test[['dependent_variable']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )
~~~

## Multiple Linear Regression

### Estimating Coefficients

How can we initially estimate the coefficients of the model?  
- Ordinary Least Squares
    - Linear algebraic methods to estimate the model coefficients
    - Takes long time for large datasets
- Some optimization algorithm (uses multiple iterations to find the best coefficients)
    - Gradient Descent (great for large datasets)

