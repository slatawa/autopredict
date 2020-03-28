# autopredict

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://github.com/slatawa/autopredict)

Autopredict is a simple yet powerful library which can be used by Data Scientists to create multiple prediction (regression,classification) data models.Pass in the data to autopredict and sit back as it does all the work for you. It is very powerfull in creating intial baseline models and also has ready made tweaked parameters for multiple models to generate highly accurate predictions.

  - Automate Model Selection
  - Hyperparameter tuning 
  - Feature selection/ranking
  - Feature Compression

This software has been designed with much Joy,
by Sanchit Latawa & is protected by The Apache Licensev2.0.

# New Features!

  - Added new classification Models
  - Allow grid tuning parameters to be passed in as argument 



### Tech

### Sample Usage
```sh
>> from autopredict.classification import autoClassify
>> model =autoClassify(encoder='label',scaler='minmax',useGridtuning=False)
>> model.train(X,y)
>>print(model.getModelScores())  
```

### Output
```sh
modelName                  score     roc_auc_score  f1_score
LogisticRegression         0.927464   0.639570      0.000000
DecisionTreeClassifier     0.937422   0.788967      0.285612
GaussianNB                 0.935352   0.760670      0.203207
RandomForestClassifier     0.937297   0.791552      0.248444
GradientBoostingClassifier 0.937472   0.792435      0.257557
```

### Sample Run for Iris Dataset

Below shows sample code flow for using autopredict , you can get this sample file
from -> */autopredict/tests/sample_iris_classification.py

```sh
# Loading Libraries
import pandas as pd
from autopredict.classification import autoClassify
from autopredict.features import rankFeatures,reduce_memory

# Setting display options
pd.set_option('display.max_columns',50000)
pd.set_option('display.width', 500000)

# Load the data into a dataframe
df = pd.read_csv('./tests/iris.csv')

# St target and feature values
X=df.drop('Species',axis=1)
y=df['Species']

# step 1  Feature Importance/Evaluation
# rankFeatures is a function in autopredict which you can
# use for feature evaluation it will give you the ranking of your features
# based on importance , with the most important feature starting from 1
print(rankFeatures(X,y))
## Sample Output - showing features along with their realtive rank  ########
#     Column-name  Importance-Rank
# 0   Petal.Width              1.0
# 1  Petal.Length              1.5
# 2  Sepal.Length              3.0
# 3   Sepal.Width              3.0

## Once you have the list of importance of the features
## you can either drop or add some new features which 
## would be used in the prediction modeling

## step 2 Train the model/evaluate

################ sample usage 2.1 ########################
# the below fit statement trains the model
model = autoClassify(scaler='standard',useGridtuning=False,gridDict = None).fit(X,y)
## get model scores 
print(model.getModelScores())
################ sample usage 2.2 ########################
## the below fit statement would ask autopredict to perform
## hyper parameter tuning using Gridsearch
model = autoClassify(scaler='standard',useGridtuning=True,gridDict = None).fit(X,y)
####### sample useage 2.3 ##################
##### the below fit uses grid tuning to fit models but over-rides
### auto-predict's base grid search parameters and models

# Define the grid that you want to run 
grid = {'LogisticRegression':{'penalty':['l2']
                               ,'C':[0.001,0.1,1,10]}
        ,'DecisionTreeClassifier': {'max_depth':[4,5,6,7,8,9,10]}
        ,'RandomForestClassifier':{'n_estimators':[100,500,1000],
                                   'max_depth':[4,5]}
        ,'GradientBoostingClassifier':{'learning_rate':[0.01,0.1,0.2,0.3],
                                      'n_estimaors':[1000]}
                           }

# train the model , passing useGridtuning as True which tells
# the function to use Grid Tuning and pass the grid to be used
# in case you pass gridDict as NUll default options set in autopredict
# would be used
model = autoClassify(scaler='standard',useGridtuning=True,gridDict = grid).fit(X,y)

# Step 3 get the score board
print(model.getModelScores())
print(model._predict_df)

# step 4 if you want to get a model object back to predict ouput
# below gets the best model object based onb accuracy score 
# you can over-ride the default scoring mechanism by using 
# score paramter in the the getBestModel Function
model_obj = model.getBestModel()

# Step 4.1 In case you want to select any other model 
# the model Name is derived from the output 
# you get when you print print(model.getModelScores())
model_obj = model.getModelObject('DecisionTreeClassifier')

# Step 5 To predict using the model object use the below statement
y_predict = model_obj.predict(validTestData)

# Other Misc features

# 1 If you want to compress memory usage of your datframe use the
# reduce_memory utilty this will compress your feature set and display
# the compress percentage 
df = reduce_memory(df)

```



### Development

Want to contribute? 
Please reach out to me at slatawa@yahoo.in and we can go over the Queue items planned for 
the next release 

### Todos

 - Write MORE Tests
 - Build catboost,LGB,XGB as a seperate feature

License
----
Apache v2.0


**Free Software, Hell Yeah!**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
