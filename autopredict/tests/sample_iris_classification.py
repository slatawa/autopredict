import pandas as pd
import os
from autopredict.classification import autoClassify
from autopredict.features import rankFeatures


pd.set_option('display.max_columns',50000)
pd.set_option('display.width', 500000)

#print(os.listdir())
# load the data into a dataframe
df = pd.read_csv('iris.csv')

# set target and feature values

X=df.drop('Species',axis=1)
y=df['Species']


# step 1  Feature Importance/Evaluation

# rankFeatures is a function in autopredict which you
# can use for feature evaluation it will give you the ranking of your features
# based on importance , with the most important feature starting from 1
print(y.dtypes)
print(rankFeatures(X,y))
################ Output #############
#     Column-name  Importance-Rank
# 0   Petal.Width              1.0
# 1  Petal.Length              1.5
# 2  Sepal.Length              3.0
# 3   Sepal.Width              3.0


## step 2 Train the model/evaluate


################ sample usage 1 ########################
# the below fit statement trains the model
#model = autoClassify(scaler='standard',useGridtuning=False,gridDict = None).fit(X,y)
# model = autoClassify(scaler=None,useGridtuning=False,gridDict = None).fit(X,y)
#print(model.getModelScores())
################ sample usage 2 ########################
## the below fit statement trains the model using grid search parameters to do
## hyper parameter tuning
#model = autoClassify(scaler='standard',useGridtuning=True,gridDict = None).fit(X,y)

####### sample useage 3#################################
##### the below fit uses grid tuning to fit models but over-rides
### auto-predict's base grid search parameters and models


### get the score board
#print(model.getModelScores())
#print(model._predict_df)

## step 3 if you want to get a model object back to predict ouput

# below gets the best model object
#model_obj = model.getBestModel()

## to predict using the model object use the below statement
## model_obj.predict(validTestData)

## in case you want to get a specific model and apply that use below code
#print(model.getModelObject('DecisionTreeClassifier'))

grid = {'LogisticRegression':{'penalty':['l2']
                               ,'C':[0.001,0.1,1,10]}
             ,'DecisionTreeClassifier': {'max_depth':[4,5,6,7,8,9,10]}

             ,'GaussianNB':{}
            ,'RandomForestClassifier':{'n_estimators':[100,500,1000],'max_depth':[4,5]}
            ,'GradientBoostingClassifier':{'learning_rate':[0.01,0.1,0.2,0.3],'n_estimators':[1000]}
                           }



model = autoClassify(scaler='standard',useGridtuning=True,gridDict = grid).fit(X,y)
print(model.getModelScores())
