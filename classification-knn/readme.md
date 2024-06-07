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
~~~

### Import Dataset
~~~
df = pd.read_csv("fileName.csv")
df.head() #this shows a preview of the dataset
~~~


