---
title: "Practical Machine Learning Course Project"
author: "Mohammad Ashekur Rahman"
date: "June 14, 2019"
output: 
  html_document: 
    keep_md: yes
---



## Overview

The following steps have been taken to complete the project:

1. Loading the Packages
2. Data Loading and Cleaning
3. Exploratory Data Analysis
4. Training Models
5. Model Performances Visualization
6. Prediction Model Selection
7. Prediction on New Dataset

## 1. Loading the Packages

For analyzing the given dataset for this project, the following R packages are needed to load prior running the analysis.


```r
library(lattice)
library(ggplot2)
library(caret)
library(corrplot)
library(rpart)
library(rpart.plot)
library(rattle)
```

## 2. Data Loading and Cleaning

## i. Data Loading

Load the training and test data set after downloading the datasets from the course content and then split the training dataset further into training and test datasets.



```r
setwd("D:/Learning/Coursera/Practical Machine Learning/Week 4/practicalmachinelearning/")
trainURL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testURL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

trainingDS <- read.csv(url(trainURL))
testDS <- read.csv(url(testURL))

label <- createDataPartition(trainingDS$classe, p = 0.7, list = FALSE)
train <- trainingDS[label, ]
test <- trainingDS[-label, ]
```

## ii. Data Cleaning

Before proceed further, data cleaning is necessary. In the training dataset, some variables contain a lot of NA, some have nearly zero variance and some are used for identification. Need to remove these type of variables from final features.


```r
label <- apply(train, 2, function(x) mean(is.na(x))) > 0.95
train <- train[, -which(label, label == FALSE)]
test <- test[, -which(label, label == FALSE)]
NZV <- nearZeroVar(train)
train <- train[ ,-NZV]
test <- test[ ,-NZV]
train <- train[ , -(1:5)]
test <- test[ , -(1:5)]
```

Finally, 54 variables are remaining out of 160 variables after data cleaning.

## 3. Exploratory Data Analysis

After cleaning the dataset, the correlation of the variables are as below:


```r
corr <- cor(train[,-54])
corrplot(corr, method = "color", type = "lower", tl.cex = 0.8, tl.col = rgb(0,0,0))
```

![](index_files/figure-html/CorrelationPlot-1.png)<!-- -->

Darker gradient in the above plot shows higher correlation. As number of correlations are few, Princal Component Analysis is not needed here.

## 4. Training Models

Now different machine learning mothods will be implemented to model the training set and model with best accuracy will be chosen to predict the outcome in the testing dataset. The methods are Decision Tree, Random Forest, Generalized Boosted Model and Naive Bayes.

To help to visualize better, a confusion matrix will be plotted at the end of each model.

### a. Random Forest (RM)


```r
set.seed(1833)
control <- trainControl(method = "cv", number = 3, verboseIter=FALSE)
modelRF <- train(classe ~ ., data = train, method = "rf", trControl = control)

predictRF <- predict(modelRF, test)
confMatRF <- confusionMatrix(predictRF, test$classe)
accRF <- confusionMatrix(predictRF, test$classe)$overall['Accuracy']
confMatRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    4    0    0    0
##          B    0 1135    3    0    0
##          C    0    0 1022    7    0
##          D    0    0    1  957    1
##          E    0    0    0    0 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9973          
##                  95% CI : (0.9956, 0.9984)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9966          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9965   0.9961   0.9927   0.9991
## Specificity            0.9991   0.9994   0.9986   0.9996   1.0000
## Pos Pred Value         0.9976   0.9974   0.9932   0.9979   1.0000
## Neg Pred Value         1.0000   0.9992   0.9992   0.9986   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1929   0.1737   0.1626   0.1837
## Detection Prevalence   0.2851   0.1934   0.1749   0.1630   0.1837
## Balanced Accuracy      0.9995   0.9979   0.9973   0.9962   0.9995
```

### b. Generalized Boosted Model (GBM)


```r
set.seed(1847)
control <- trainControl(method = "repeatedcv", number = 5, repeats = 1, verboseIter = FALSE)
modelGBM <- train(classe ~ ., data = train, trControl = control, method = "gbm", verbose = FALSE)
modelGBM$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 53 predictors of which 52 had non-zero influence.
```

```r
predictGBM <- predict(modelGBM, test)
confMatGBM <- confusionMatrix(predictGBM, test$classe)
accGBM <- confusionMatrix(predictGBM, test$classe)$overall['Accuracy']
confMatGBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1669   16    0    0    3
##          B    3 1115    8    3    3
##          C    1    7 1006   13    4
##          D    1    1   10  947   12
##          E    0    0    2    1 1060
## 
## Overall Statistics
##                                          
##                Accuracy : 0.985          
##                  95% CI : (0.9816, 0.988)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : <2e-16         
##                                          
##                   Kappa : 0.9811         
##                                          
##  Mcnemar's Test P-Value : 0.0016         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9970   0.9789   0.9805   0.9824   0.9797
## Specificity            0.9955   0.9964   0.9949   0.9951   0.9994
## Pos Pred Value         0.9887   0.9850   0.9758   0.9753   0.9972
## Neg Pred Value         0.9988   0.9950   0.9959   0.9965   0.9954
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2836   0.1895   0.1709   0.1609   0.1801
## Detection Prevalence   0.2868   0.1924   0.1752   0.1650   0.1806
## Balanced Accuracy      0.9963   0.9877   0.9877   0.9887   0.9895
```

### c. Decision Tree (DT)


```r
set.seed(1671)
modelDT <- rpart(classe ~ ., data = train, method = "class")
fancyRpartPlot(modelDT)
```

![](index_files/figure-html/DecisionTree-1.png)<!-- -->

```r
predictDT <- predict(modelDT, test, type = "class")
confMatDT <- confusionMatrix(predictDT, test$classe)
accDT <- confusionMatrix(predictDT, test$classe)$overall['Accuracy']
confMatDT
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1444  190   46   49   53
##          B   86  695   69   86  120
##          C   22   80  819  148   86
##          D   99  139   66  630  116
##          E   23   35   26   51  707
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7298          
##                  95% CI : (0.7183, 0.7411)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6577          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8626   0.6102   0.7982   0.6535   0.6534
## Specificity            0.9197   0.9239   0.9308   0.9147   0.9719
## Pos Pred Value         0.8103   0.6581   0.7091   0.6000   0.8397
## Neg Pred Value         0.9439   0.9081   0.9562   0.9309   0.9256
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2454   0.1181   0.1392   0.1071   0.1201
## Detection Prevalence   0.3028   0.1794   0.1963   0.1784   0.1431
## Balanced Accuracy      0.8912   0.7671   0.8645   0.7841   0.8127
```

### d. Naive Bayes (NB)


```r
set.seed(1819)
control <- trainControl(method = "repeatedcv", number = 4, repeats = 1, verboseIter = FALSE)
modelNB <- train(classe ~ ., data = train, trControl = control, method = "nb", verbose = FALSE)

predictNB <- predict(modelNB, test)
confMatNB <- confusionMatrix(predictNB, test$classe)
accNB <- confusionMatrix(predictNB, test$classe)$overall['Accuracy']
confMatNB
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1517  219  192  168   67
##          B   24  761   76    5   92
##          C   30   95  717  148   47
##          D   94   53   39  600   36
##          E    9   11    2   43  840
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7536          
##                  95% CI : (0.7424, 0.7646)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.685           
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9062   0.6681   0.6988   0.6224   0.7763
## Specificity            0.8466   0.9585   0.9341   0.9549   0.9865
## Pos Pred Value         0.7013   0.7944   0.6914   0.7299   0.9282
## Neg Pred Value         0.9578   0.9233   0.9363   0.9281   0.9514
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2578   0.1293   0.1218   0.1020   0.1427
## Detection Prevalence   0.3675   0.1628   0.1762   0.1397   0.1538
## Balanced Accuracy      0.8764   0.8133   0.8165   0.7886   0.8814
```

## 5. Model Performances Visualization

Visualizing the model performance will give better understanding for identifying the best method to predict the outcome in test dataset.


```r
accuracyValues <- c(accRF, accGBM, accDT, accNB)
modelNames <- c("Random Forest", "Generalized Boosted Model", "Decision Tree", "Naive Bayes")
x <- data.frame(Model = modelNames,
                Accuracy = accuracyValues)
ggplot(x, aes(x = Model, y = Accuracy)) + 
  geom_bar(stat = "identity", aes(fill = Model)) +
  theme_bw() + theme(legend.position = "none")
```

## 6. Prediction Model Selection

Random forest is the best performing model, followed by Generalized Boosted Model.

## 7. Prediction on New Dataset

Now, Random Forest model will be used for final prediction of the given test dataset.


```r
predictRF <- predict(modelRF, testDS)
predictRF
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
