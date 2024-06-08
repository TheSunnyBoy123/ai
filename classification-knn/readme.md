# KNN 
## Basic Idea
KNN is a simple algorithm that stores all available cases and classifies new cases based on a similarity measure which is usually the distance.  

A good example would be having a dataset of points with two features and a label. Plot this dataset and now for a test point, find the k nearest points to this test point. The label of the test point would be the majority label of the k nearest points.

## Evaluation Metrics
### Jaccard Index
$y =$ Actual labels  
$\hat{y} =$ Predicted label  
$J(y, \hat{y}) = \frac{|y \cap y\hat{}|}{|y \cup y\hat{}|}$

An example:  
$y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]$  
$\hat{y} = [1, 1, 0, 0, 0, 1, 1, 1, 1, 1]$  
$J(y, \hat{y}) = \frac{8}{12} = 0.66$

### F1-score
$TP = $ True Positives  
$FP = $ False Positives  
$FN = $ False Negatives  
$TN = $ True Negatives  

$Precision = \frac{TP}{TP + FP}$
> A measure of accuracy  

$Recall = \frac{TP}{TP + FN}$  
> A measure of how well we can find true positives


**Note:** Both of these metrics are used and calculated separately for each class/groups in the classification problem.

$F1_{score} = 2 \times \frac{Precision \times Recall}{Precision + Recall}$

The F1 score lets us see how well our model is at finding the true positives and how well it is at not misclassifying the negatives. By taking the harmonic mean, we are able to include both metrics in one score.

## Outlined Steps for KNN Classification
### Import Libraries
~~~
import numpy as np
import matplotlib.pyplot as plt
import pandas as pdfrom sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
~~~

### Import Dataset
~~~
df = pd.read_csv("fileName.csv")
df.head() #this shows a preview of the dataset
~~~

Check features and labels:
~~~
df.columns
~~~

### Preprocessing

Convert data into a numpy array:
~~~
X = df[['feature1', 'feature2', 'feature3', 'feature4']].values
y = df['label'].values
~~~

Normalize the data (transform the data so that it has a mean of 0 and a standard deviation of 1):
~~~
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
~~~

### Train Test Split
~~~
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
# test_size is the proportion of the dataset to include in the test split
# random_state is the seed used by the random number generator, this can be any integer
~~~


### KNN Model
~~~
k = 5
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
~~~

### Predict
~~~
y_hat = neigh.predict(X_test)
~~~

`y_hat` is the predicted labels for the test set in the form of a numpy array.

Accuracy could be higher for a different value of k. To find the best value of k, we can loop through a range of values and calculate the accuracy for each value of k. Then finally set this value of k for the model.

~~~
accuracies = []
for k in range(1, 10):
    neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
    y_hat = neigh.predict(X_test)
    accuracies.append(np.mean(y_hat == y_test))

# Plot the accuracies
plt.plot(range(1, 10), accuracies)
plt.xlabel('Value of K')
plt.ylabel('Accuracy')
plt.show()
~~~

# Decision Tree
## Basic Idea
In a decision tree, we essentially have nodes which represent values of a feature. The edges between nodes represent the decision rules. The leaf nodes represent the outcome. The tree is built by splitting the dataset into subsets based on the value of a feature. Choosing the final class label is done by traversing the tree from the root to the leaf node. 

## Making a Decision Tree
- Choose the best feature to split the data on.
    - The best feature will give a (or the most) clear separation between the classes.
    - This introduces two terms:
        - Predictiveness: How well a feature can predict the class (maximise at this step)
        - Impurity: How mixed the classes are in a subset (minimise at this step)
2. Split the data into subsets.
3. Repeat the process for each feature that we are trying to include in the model.

### Choosing a feature - detailed overview
Let's take an example of a dataset.  
We have a dataset with 2 features - `age` (`young` or `old`), `gender` (`male` or `female`) and a label -  `fav_color` (`blue` or `green`).

We first define an important measure of impurity - Entropy.  

$Entropy = -p(A)log_{2}(p(A)) - p(B)log_{2}(p(B))$  
where  
$p(A) = $ probability of class A 


The lower the entropy, the better the feature is at predicting the class, essentially telling us that at this recursive step, this feature can be used to create nodes of the tree.

Continuing with our example, we are at the first node of our decision tree, so currently we have the entire dataset. We calculate the entropy of the entire dataset.  
Say there are 9 `Blue` and 5 `Green` in the dataset.  
$Entropy = -\frac{9}{14}log_{2}(\frac{9}{14}) - \frac{5}{14}log_{2}(\frac{5}{14}) = 0.940$

Now say we do a split based on the `age` feature.
- For the `young` feature, we have 6 `Blue` and 2 `Green`.
    - Entropy = 0.811
- For the `old` feature, we have 3 `Blue` and 3 `Green`.
    - Entropy = 1.00

Let's check for the `gender` feature.
- For the `male` feature, we have 3 `Blue` and 4 `Green`.
    - Entropy = 0.985
- For the `female` feature, we have 6 `Blue` and 1 `Green`.
    - Entropy = 0.592

At this point we have metrics for each feature. We can now calculate the information gain for each feature.

**Information Gain** is a measure of how well a feature can predict the class.

Information Gain = Entropy of the parent node - Weighted sum of the entropy of the child nodes.

For the `gender` feature, the information gain would be:  
$IG(gender) = 0.940 - [\frac{7}{14}0.985 + \frac{7}{14}0/592] = 0.151$

For the `age` feature, the information gain would be:  
$IG(age) = 0.940 - [\frac{8}{14}0.811 + \frac{6}{14}1.00] = 0.048$

Thus, we would choose the `gender` feature to split the dataset on as it has a higher Information Gain at this step.

## Outlined Steps for Decision Tree Classification
# `ToDo`

# Regression Tree
# `ToDo`