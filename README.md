Practical Machine Learning - Prediction Assignment Writeup
========================================================

This document describe the analysis done for the prediction assignment of the practical machine learning course.

The first part is the declaration of the package which will be used. In addition to caret & randomForest already seen on the course, I used Hmisc to help me on the data analysis phases & foreach & doParallel to decrease the random forrest processing time by parallelising the operation.
Note : to be reproductible, I also set the seed value.


```r
options(warn=-1)
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(Hmisc)
```

```
## Loading required package: grid
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## L'objet suivant est masqué from 'package:caret':
## 
##     cluster
## 
## Loading required package: Formula
## 
## Attaching package: 'Hmisc'
## 
## L'objet suivant est masqué from 'package:randomForest':
## 
##     combine
## 
## Les objets suivants sont masqués from 'package:base':
## 
##     format.pval, round.POSIXt, trunc.POSIXt, units
```

```r
library(foreach)
library(doParallel)
```

```
## Loading required package: iterators
## Loading required package: parallel
```

```r
set.seed(4356)
```

The first step is to load the csv file data to dataframe and analyze the type & the completion rate of the data (commands are commented to limit the output size. You can run it deleting the "#" ) :


```r
data <- read.csv("/projects/Coursera-PracticalMachineLearning/data//pml-training.csv")
#summary(data)
#describe(data)
#sapply(data, class)
#str(data)
```

This analysis allows us to note two main points :
 1 - Some numeric data have been imported as factor because of the presence of some characters ("#DIV/0!")
 2 - Some columns have a really low completion rate (a lot of missing data)
 
To manage the first issue we need to reimport data ignoring "#DIV/0!" values :


```r
data <- read.csv("/projects/Coursera-PracticalMachineLearning/data//pml-training.csv", na.strings=c("#DIV/0!") )
```

And force the cast to numeric values for the specified columns (i.e.: 8 to end) :


```r
cData <- data
for(i in c(8:ncol(cData)-1)) {cData[,i] = as.numeric(as.character(cData[,i]))}
```

To manage the second issue we will select as feature only the column with a 100% completion rate ( as seen in analysis phase, the completion rate in this dataset is very binary) We will also filter some features which seem to be useless like "X"", timestamps, "new_window" and "num_window". We filter also user_name because we don't want learn from this feature (name cannot be a good feature in our case and we don't want to limit the classifier to the name existing in our training dataset)


```r
featuresnames <- colnames(cData[colSums(is.na(cData)) == 0])[-(1:7)]
features <- cData[featuresnames]
```


We have now a dataframe "features which contains all the workable features. So the first step is to split the dataset in two part : the first for training and the second for testing.


```r
xdata <- createDataPartition(y=features$classe, p=3/4, list=FALSE )
training <- features[xdata,]
testing <- features[-xdata,]
```


We can now train a classifier with the training data. To do that we will use parallelise the processing with the foreach and doParallel package : we call registerDoParallel to instantiate the configuration. (By default it's assign the half of the core available on your laptop, for me it's 4, because of hyperthreading) So we ask to process 4 random forest with 150 trees each and combine then to have a random forest model with a total of 600 trees.

```r
registerDoParallel()
model <- foreach(ntree=rep(150, 4), .combine=randomForest::combine) %dopar% randomForest(training[-ncol(training)], training$classe, ntree=ntree)
```

To evaluate the model we will use the confusionmatrix method and we will focus on accuracy, sensitivity & specificity metrics :

```r
predictionsTr <- predict(model, newdata=training)
confusionMatrix(predictionsTr,training$classe)
```

```
## 
## Attaching package: 'e1071'
## 
## L'objet suivant est masqué from 'package:Hmisc':
## 
##     impute
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4185    0    0    0    0
##          B    0 2848    0    0    0
##          C    0    0 2567    0    0
##          D    0    0    0 2412    0
##          E    0    0    0    0 2706
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

```r
predictionsTe <- predict(model, newdata=testing)
confusionMatrix(predictionsTe,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    1    0    0    0
##          B    0  946    6    0    0
##          C    0    2  849    6    1
##          D    0    0    0  798    1
##          E    0    0    0    0  899
## 
## Overall Statistics
##                                         
##                Accuracy : 0.997         
##                  95% CI : (0.994, 0.998)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.996         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.997    0.993    0.993    0.998
## Specificity             1.000    0.998    0.998    1.000    1.000
## Pos Pred Value          0.999    0.994    0.990    0.999    1.000
## Neg Pred Value          1.000    0.999    0.999    0.999    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.173    0.163    0.183
## Detection Prevalence    0.285    0.194    0.175    0.163    0.183
## Balanced Accuracy       1.000    0.998    0.995    0.996    0.999
```

As seen by the result of the confusionmatrix, the model is good and efficient because it has an accuracy of 0.997 and very good sensitivity & specificity values on the testing dataset. (the lowest value is 0.992 for the sensitivity of the class C)

It seems also very good because It scores 100% (20/20) on the Course Project Submission (the 20 values to predict)


I also try to play with preprocessing generating PCA or scale & center the features but the accuracy was lower.
