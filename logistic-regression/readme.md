## Basic Idea
Logistic Regression is a classification algorithm. It is used to predict a binary outcome as a proability value between 0 and 1.

## When to use Logistic Regression
1. If label is binary - 0 or 1, True or False, Yes or No
2. If you need a probabilistic value of the label
3. If the data is linearly separable, i.e., the data can be separated using a straight line/plane

## Logistic Regression Training

In training a model we change the parameters of the model to minimize the cost function. The cost function is the difference between the predicted value and the actual value.

The cost function for this model given by:
$Cost(\hat{y}, y) = \frac{1}{2} [\sigma(\theta^TX) -y]^2$

**Note:** Here the term is squared to account for negative values. Also, the $\frac{1}{2}$ multiplier is simply used to make the derivative of the cost function easier to calculate.

**Note:** Here $\sigma$ is the sigmoid function given by: $\sigma(z) = \frac{1}{1+e^{-z}}$. In basic terms this function squashes the value of z between 0 and 1 (probability value).

The cost function for all the training data is given by: $J(\theta) = \frac{1}{m} \sum_{i=1}^{m} Cost(\hat{y}, y)$ or simply the mean of the cost function over the entire training data/vector.


Now, we can see that the cost function is a function that depends on the parameter $\theta$. Thus we can study it as such and minimize the value of the cost function to get the best parameters.

But, this cost function is complicated so instead we replace it with a function that behaves similarly but is easier to calculate.

$Cost = -log(\hat{y}) $,  if y = 1  
$Cost = - log(1-\hat{y})$, if y = 0

Now our cost function becomes:  

$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [ylog(\hat{y}) + (1-y)log(1-\hat{y})]$

This is the cost function we will use to train our model.

### Minimize the cost function - Gradient Descent
If we were to use a dataset with `n` features, then the cost function would be a function of `n` variables. We can now imagine a `n` dimensional space where each dimension represents a feature and the value of the cost function is the height of the space at that point.  

What we are trying to do is find the lowest point in this space. This is the point where the cost function is minimum and the model is best. Thus giving us the weights for this model.

To do this we use the gradient descent algorithm. The algorithm is as follows:
1. Initialize the weights randomly
2. Calculate the cost function at the current point
3. Calculate the gradient of the cost function at the current point
4. Update the weights using the formula: $\theta_{new} = \theta_{prev} - \alpha \nabla J(\theta)$  
where $\alpha$ is the learning rate and $\nabla J(\theta)$ is the gradient of the cost function at the current point.
5. Repeat steps 2-4 until the cost function until the cost function is within some tolerance level or the number of iterations is reached.

## Outlined Steps for Logistic Regression

### Import Libraries
~~~
import pandas as pd
import numpy as np
import pylab as pl
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import log_loss
~~~

### Load Data
~~~
df =  pd.read_csv("data.csv")
df['label'] = df['label'].astype('int') # Convert label to int as required by the sci-kit learn library
df.head()
~~~

### Data processing
~~~
X = np.asarray(df[['feature1', 'feature2', 'feature3']])
y = np.asarray(df['label'])
X = preprocessing.StandardScaler().fit(X).transform(X)
~~~

Split into training and testing data
~~~
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
~~~

### Training the model
Now we build the model using `LogisticRegression` from Sci-kit learn

~~~
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
~~~

### Predictions
~~~
y_hat = LR.predict(X_test) # Predictions
y_hat_prob = LR.predict_proba(X_test) # Probability of predictions for each class
~~~

### Evaluation

**Jaccard Index**
~~~
jaccard_score(y_test, y_hat)
~~~
**Confusion Matrix**
~~~
confusion_matrix = confusion_matrix(y_test, y_hat, labels=[1,0])
np.set_printoptions(precision=2) #this just changes the output settings for easier reading

plt.figure()
plot_confusion_matrix(confusion_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
~~~

**Log Loss**
~~~
log_loss(y_test, y_hat_prob)
~~~




