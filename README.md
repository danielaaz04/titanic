### Name

Predicting Survival in Titanic

### Description

I was given two datasets:
+ Train.csv
+ Test.csv

To start I visualized what features did each dataset had and each features' type.

As part of the EDA I used a heatmap to help me find some correlations between variables and understand better the data.
I also used a couple more plots and confirmed the linear relation between **Fare** and the target variable, as well as **Age** and target variable. The following heatmap shows some strong and weak correlations between variables.

![Correlation Heatmap](/images/heatmap.png)


I decided to merge train and test dataset to clean data and input missing values.
Once I had clean data, I separated the merged dataset again into train and test datasets.

I selected two models to train in 5 different folds using cross validation StratifiedKFold.
The models were:

1. KNearestNeighbor: I first scaled features as Age and Fare had a different scale from the rest of the features.
2. RandomForest

According to the cross validation scores I decided to select Random Forest to train the entire train dataset.
According to the feature importance score of the model the following image shows the features'order of importance.

![Feature Importances](/images/feature_importances.png)


Finally I saved this feature importances in a csv file, as well as the predictions of the model applied to the test dataset. 
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
