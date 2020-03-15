# autopredict

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://github.com/slatawa/autopredict)

Autopredict is a simple yet powerful library which can be used by Data Scientists to create multiple prediction (regression,classification) data models.Pass in the data to autopredict and sit back as it does all the work for you. It is very powerfull in creating intial baseline models and also has ready made tweaked parameters for multiple models to generate highly accurate predictions.

  - Automate Model Selection
  - Hyperparameter tuning 
  - Feature selection/ranking

This software has been designed with much Joy,
by Sanchit Latawa & is protected by The Apache Licensev2.0.

# New Features!

  - Added new classification Models
  - Allow grid tuning parameters to be passed in as argument 



### Tech

### Sample Usage
```sh
>> from autopredict.classification import autoClassify
>> model =autoClassify(encoder='label',scaler='minmax',use_grid_tuning=False)
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
### Development

Want to contribute? 
Please reach out to me at slatawa@yahoo.in and we can go over the Queue items planned for 
the next release 

### Todos

 - Write MORE Tests
 - Add option for RFE

License
----
Apache v2.0


**Free Software, Hell Yeah!**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   