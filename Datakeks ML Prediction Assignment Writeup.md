---
title: "Machine Learning Prediction Assignment Writeup"
author: "datakeks"
date: "January 31, 2016"
output: html_document
---
##Executive Summary
Leveraging the [Weight Lifting Exercise Dataset](http://groupware.les.inf.puc-rio.br/har), this analysis aims to determine whether or not exercises were performed correctly or incorrectly by the participant. The data are collected from six participants, men aged 20-28, performing five sets of 10 repetitions of the Unilateral Dumbbell Biceps Curl. The sets were performed as follows: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). The model used in this analysis uses a Random Forest to predict if correct form is or is not being used, which requires no cross validation. The random forest produces an expected out of sample error of 0.03%, which was reflected when the model was applied to the test dataset and correctly determined the classe for all 20 test observations.

##Analysis
###Pre-Processing
1. Load the necessary packages

```r
suppressMessages(library(caret))
suppressMessages(library(randomForest))
```

2. Download training and testing files from the internet

```r
if (!file.exists("data")) {dir.create("data")}
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "data/pml-training.csv", method = "curl", cacheOK = FALSE)
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "data/pml-testing.csv", method = "curl", cacheOK = FALSE)
setwd("./data")
trainingRaw <- read.csv("pml-training.csv", header = TRUE, na.strings = c("NA", "", " "))
testingRaw <- read.csv("pml-testing.csv", header = TRUE, na.strings = c("NA", "", " "))
```

3. Subsetting data to remove problematic variables
The raw training and testing datasets contain 160 variables for their respective 19,622 training observations and 20 testing observations. However, not all of these 160 variables would benefit the model as predictors, due to the prevalance of missing values in several columns and the inclusion of several variables that did not provide predictive measurement values. 

```r
trainingSubset <- trainingRaw[, 7:ncol(testingRaw)]  # remove unnecesary metadata columns
training <- trainingSubset[, colSums(is.na(trainingSubset)) == 0] # subset columns that do not contain NAs
testingSubset <- testingRaw[, 7:ncol(testingRaw)]
test <- testingSubset[, colSums(is.na(testingSubset)) == 0]
predictors <- names(training[, 1:53]) # subset the column names to exclude non-predictors problem_id and classe
```

4. Partition the training data further with 60% for training and 40% for validation
This data splitting of the training dataset provides one round of cross validation. This is helpful in that it provides a validation dataset to run the model on to estimate the out of sample error rate. Further rounds of cross validation are not necessary since Random Forests do not require seperate cross validation. The one round of cross validation is enough to get an estimate of how the model will preform on the test data without using the test data. 

```r
set.seed(30303)
inTrain <- createDataPartition (y = training$classe, p = 0.6, list = FALSE)
train <- training[inTrain,]
validate <- training[-inTrain,]
```

###Random Forest Model
A random forest model was selected due to the accuracy benefits of random forests. Testing the model against the validate dataset created in the training set data split allows the predicted values to be assessed against the given results. Comparing the predicted values to the classe values from the validated dataset shows that the model predicts on the validation dataset with an accuracy of 99.7% and an out of sample error rate to be expected of 0.3%. 

```r
modFit <- randomForest(classe ~., data = train, ntree = 300) # fit the model
modFit
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = train, ntree = 300) 
##                Type of random forest: classification
##                      Number of trees: 300
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.36%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3348    0    0    0    0 0.0000000000
## B    6 2271    2    0    0 0.0035103115
## C    0   13 2039    2    0 0.0073028238
## D    0    0   17 1912    1 0.0093264249
## E    0    0    0    1 2164 0.0004618938
```

```r
predVal <- predict(modFit, validate[, predictors]) #removing classe as it is not a predictor for itself
confusionMatrix(validate$classe, predVal)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    1    0    0    0
##          B    2 1515    1    0    0
##          C    0    4 1364    0    0
##          D    0    0    6 1279    1
##          E    0    0    0    3 1439
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9977          
##                  95% CI : (0.9964, 0.9986)
##     No Information Rate : 0.2846          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9971          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9967   0.9949   0.9977   0.9993
## Specificity            0.9998   0.9995   0.9994   0.9989   0.9995
## Pos Pred Value         0.9996   0.9980   0.9971   0.9946   0.9979
## Neg Pred Value         0.9996   0.9992   0.9989   0.9995   0.9998
## Prevalence             0.2846   0.1937   0.1747   0.1634   0.1835
## Detection Rate         0.2843   0.1931   0.1738   0.1630   0.1834
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9995   0.9981   0.9971   0.9983   0.9994
```

##Results
When the model was applied to the test dataset, it correctly predicted the type of form being used for all 20 observations. This is reflective of the low sample error rate and high accuracy of the random forest model.

```r
predTest <- predict(modFit, test[, predictors])
predTest
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

##Dataset Citation
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13)

