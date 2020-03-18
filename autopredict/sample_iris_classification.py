from datasets import loadiris
# add autopredict. to all packages
from classification import autoClassify
import pandas as pd

pd.set_option('display.max_columns',50000)
pd.set_option('display.width', 500000)

df = loadiris()
#print(df)

## step 1  Feature Importance/Evaluation


## step 2 Train the model/evaluate

X=df.drop('Species',axis=1)
y=df['Species']

################ sample usage 1 ########################
# the below fit statement trains the model
model = autoClassify(scaler='standard',useGridtuning=False,gridDict = None).fit(X,y)
model = autoClassify(scaler='standard',useGridtuning=True,gridDict = None).fit(X,y)
model = autoClassify(scaler=None,useGridtuning=True,gridDict = None).fit(X,y)
model = autoClassify(scaler=None,useGridtuning=False,gridDict = None).fit(X,y)
print(model.getModelScores())
################ sample usage 2 ########################
## the below fit statement trains the model using grid search parameters to do
## hyper parameter tuning
#model = autoClassify(scaler='standard',useGridtuning=True,gridDict = None).fit(X,y)

####### sample useage 3#################################
##### the below fit uses grid tuning to fit models but over-rides
### auto-predict's base grid search parameters and models


### get the score board
print(model.getModelScores())
#print(model._predict_df)

## step 3 get scores
