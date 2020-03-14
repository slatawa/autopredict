# autopredict
Python Library to Automate building of prediction models - classification and regression and feature selection/ranking.

Project Description 

autopredict is a simple yet powerful library which can be used by Data Scientists to apply multiple prediction (regression,classification) data models
to a data set. Just pass in the train set with the dependent variable and sit back as autopredict does all the work for you.

This software has been designed with much Joy,
by Sanchit Latawa & is protected by The Apache Licensev2.0.

Objective

Aim of this library is to automate the model building task, feature selection with limited human intervention. Once you have baseline ready from autopredict 
you can tweak parameter/hyper parameter tuning to improve prediction scores further.

**********Sample Usage ***************
>> from classification import autoClassify
>> model =autoClassify(encoder='label',scaler='minmax',use_grid_tuning=False)
>> model.train(X,y)
>> print(model.getModelScores())  

