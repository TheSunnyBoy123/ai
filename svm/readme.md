## Basic Idea
   
SVM is a great model to use for classification problems when the data is not linearly separable. It is a powerful model that can be used for both linear and non-linear classification problems.  
The basic idea behind SVM is to find the hyperplane (a line in higher dimensions) that best separates the data into two classes.  
Now, say our data has 2 features which do not have a linear separation. We can use a kernel function to transform the data into a higher dimension where it can be linearly separated.

## Kernels/Kernelling

Let's take an example of a linear dataset, i.e, data with just one feature and one label.

| feature | class |
|---------|-------|
| -3      | blue  |
| -2      | blue  |
| -1      | blue  |
| -0.75   | green |
| -0.50   | green |
|  0      | green |
|  0.75   | green |
| 0.5     | green |
| 1       | blue  |
| 2       | blue  |
| 3       | blue  |

Here we can see that there is no straight line/plane/hyperplane that can separate the data into two classes. So now we can use a kernel function to transform the data into a higher dimension where it can be linearly separated.

$\psi(x) = [x, x^2]$

This transforms our dataset into 

| feature | class |
|---------|-------|
| 9      | blue  |
| 4      | blue  |
| 1      | blue  |
| 0.5625 | green |
| 0.25   | green |
| 0      | green |
| 0.5625 | green |
| 0.25   | green |
| 1      | blue  |
| 4      | blue  |
| 9      | blue  |

Now we can see that the data can be separated using a straight line $y = 0.9x$  for example.
This line becomes our separator and our kernel is the function $\psi(x)$.  

### Types of Kernels
1. Linear 
2. Polynomial
3. Radial Basis Function (RBF)
4. Sigmoid


## Finding Separating Hyperplane

The goal of SVM is to find the hyperplane that best separates the data into two classes.  
For this, we first define a term Margin, this is the distance between the hyperplane and the nearest point from either class.  
Our goal is to maximise this margin. From this wew can also see that only the dataset points closest to the separator need to be studied. These points are called support vectors.  

## Pros and Cons

### Advantages
1. Accurate for multiple features dataset
2. Versatile for different types of data

### Disadvantages
1. Prone to overfitting
2. No confidence/probability score
3. Can not be efffectively used for large datasets

## Use cases
1. Image classification/recongnition
2. Sentiment analysis
3. Text classification
4. Clustering

## Outlined Steps for SVM

## `Todo`