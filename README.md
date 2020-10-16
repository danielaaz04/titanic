### Name

Predicting Survival in Titanic

### The problem

The sinking of the Titanic resulted in the death of 1502 out of 2224 passengers and crew. Apparently there were some patterns in the people who died and the people who survived. The problem here is that I have certain data with specific characteristics of each passenger and the data is already labeled which lets me know if the passenger lived or died. I have been given also another dataset with more Titanic passengers and their characteristics but this dataset is not labeled, so I don't know who lived and who died. To be able to predict which passengers were more likely to survive I will use a couple of algorithms to train the first dataset and when I decide which one is the best I will use it to predict what passengers in the unlabeled dataset survived.

### Description

I was given two datasets:
+ Train.csv
+ Test.csv

To begin I visualized what features did each dataset had and each feature's type.

### EDA

I used a heatmap to help me find some correlations between variables and understand better the data.
I also used a couple more plots and confirmed the linear relation between **Fare** and the target variable, as well as **Age** and target variable. The following heatmap shows some strong and weak correlations between variables.

![Correlation Heatmap](/images/heatmap.png)

### Cleaning the Data

Considering only my training set as if my test set is not there, I started to clean my data.

1. I deleted some features which I considered not relevant:

    + Name
    + PassengerId
    + Ticket Number

2. I deleted 1 feature for having around 77% of missing values:

    + Cabin

3. To be able to have clean data I imputed missing values in relevant features:

    + Age missing values were imputed with the Age mean.
    + Embarked missing values were imputed with the most frequent value in that feature.

4. Next, I encoded my category features "Sex" and "Embarked" with OneHotEncoding as the algorithms understand numbers better.

### Selecting the best algorithm

I selected two algorithms to train in 5 different folds using cross validation StratifiedKFold.
The algorithms were:

1. KNearestNeighbor: I first scaled features because Age and Fare had a different scale from the rest of the features.
2. RandomForest

According to the cross validation scores I decided to select Random Forest to train the entire train dataset.
According to the feature importance score of the model the following image shows the feature's order of importance.

![Feature Importances](/images/feature_importances.png)

### Make Predictions

Now working on my test dataset and before making predictions I had to make the same transformations I did on my training dataset but still using the Age mean and Fare mean from the training dataset, to do not have a data leakage.

Then I applied my selected algorithm to my final test set to predict which passengers were more likely to survive.

### Saving the model, predictions and important features.

Finally I saved this feature importances in a csv file, as well as the predictions obtained. 
I used pickle to save the model so it can be loaded from disk in the future and applied to new data.

```
 loaded_model = pickle.load(open(filename, 'rb'))
 result = loaded_model.score(X_test, Y_test)
 print(result)
```
 
 ### Files
 
 Train and test dataset can be found in the data folder.
 Feature importances can be found in the following file: "feature_importances.csv"
 Predictions of the model can be found in the following file: "predictions.csv"
 The saved model can be found as: "final_model.sav"
