# -*- encoding: utf-8 -*-
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from grid._base import gridDict



class autoClassify:
    """
    Automate classification prediction

    The function runClassification automates finding supervised
    classification model.The runClassification function supports the following
    models right now

    """

    def __init__(self,cv=3,verbosity="warn",models=None,
                 encoder='label',scaler=None
                 ,use_grid_tuning=False,score='roc_auc',random_state=None):
        """

        :param cv:         cross validation sets
        :param verbosity:  level of logging - 0,1,2
        :param models:     List of model objects for which you want to run the train data through
                           below is a sample input, by default this is null in which case train data
                           ,could be run through all supported models with default parameters of autopredict
                           if this argument is passed in use_grid_tuning would be over-riden to False
                           i.e - you can not grid search for parameter tuning
                           [LogisticRegression(random_state=100)
                           ,DecisionTreeClassifier(random_state=100,max_depth=3)]
        :param encoder:    this variable decides how to convert string/object column in input data
                           autopredict for now supports only labelencoder, if you choose
                           to with some other encoding scheme like onehotencoder tranform the
                           input array before passing it in
        :param scaler:     this would decide the scaling strategy,some of prediction models
                           perform better with scaled features while few others like trees can handle
                           unscaled values,default value for this is None, supported values- 'minmax'
                           for sklearn's minmax sclare ,'standard' - for sklearn's standard scaler
        :param use_grid_tuning: set this to True if you want to use grid search over the
                               supported classifier, the grid is selected based on configuration
                               saved in ./grid/_bases.py file in Dictionary gridDict
        :param score: scoring parameter to be used for grid search
        :param random_state: seed parameter
        """
        self.cv = cv
        self.verbosity=verbosity
        self.random_state = random_state

        if models:
            self.models=models
            self.use_grid_tuning = False
        else:
            self.models=[LogisticRegression(random_state=self.random_state)
                     ,DecisionTreeClassifier(random_state=self.random_state)]
            self.use_grid_tuning = use_grid_tuning

        self._predict_df = pd.DataFrame(columns=['modelName','modelObject','modelParams'])
        self._encode_dict = {'label':LabelEncoder(),'hot':OneHotEncoder()}
        self.encoder=encoder
        self.scaler=scaler
        self.score = score


    ## add checks here


    def _encodeData(self,X,encodeObj):
        for rec in X.columns:
            if X[rec].dtype == 'object':
                print(rec)
                X[rec] = encodeObj.fit_transform(X[rec])
        return X

    def _score(self,X,y):
        ### add scorers here
        self._predict_df['score'] = self._predict_df['modelObject']. \
            apply(lambda x: x.score(X,y))
        self._predict_df['roc_auc_score'] = self._predict_df['modelObject'].\
            apply(lambda x : roc_auc_score(y,x.predict_proba(X)[:,1]))
        self._predict_df['f1_score'] = self._predict_df['modelObject']. \
            apply(lambda x: f1_score(y,x.predict(X)))

    def _applyModel(self,model,X,y,params=None):
        key = str(model).split('(')[0]
        model.fit(X,y,params)
        self._predict_df= self._predict_df.append({'modelName':key,
                                                   'modelObject':model,
                                                  'modelParams':model.get_params()},
                                                  ignore_index=True)

    def train(self,X,y):
        ## check if any data to be converted from str/object
        X= self._encodeData(X,self._encode_dict[self.encoder])

        if self.use_grid_tuning:
            for rec in self.models:
                key = str(rec).split('(')[0]
                if key not in gridDict.keys():
                    print(key)
                    raise ValueError(f' {key} is not supported by Gridsearch paramter dict, look at ./grid/_.base'
                          f'-> gridDict')
                gridvalues = gridDict[key]
                gsModel = GridSearchCV(estimator=rec
                                       ,param_grid=gridvalues
                                       ,scoring=self.score
                                       ,cv=self.cv).fit(X, y)
                self._predict_df = self._predict_df.append({'modelName': key,
                                                            'modelObject': gsModel.best_estimator_,
                                                            'modelParams': gsModel.best_params_,
                                                            'gridSearchScore':gsModel.best_score_},
                                                           ignore_index=True)
                self._score(X, y)
        else:
            ## split the data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=self.random_state)

            for rec in self.models:
            ## Fit the data into the model
                self._applyModel(rec,X_train,y_train)
                self._score(X_test,y_test)

    def getModelScores(self):
        """
        :return: return models trained by autoclassify along with respective stores
        sample
        autoClassify().getModelScores()
        """
        return self._predict_df

    def getModelObject(self,modelName):
        """
        this function is used
        :param modelName:  name of the model whose object you want back, all possible options
                        can be retrieved by using autoClassify().getModelScores()
        :return: returns trained model object which was trained under autopredict's train function
        sample call
        autoClassify().getModelObject(modelName='LogisticRegression')
        """
        return self._predict_df.loc[self._predict_df['modelName']==modelName,'modelObject'].iloc[0]
    
    def getBestModel(self,score='score'):
        """
        :param score:  by default this is set to score method, scorers supported by
        autoClassify can be passed here
        :return:
        sample call
        autoClassify().getBestModel()
        """
        if score not in self._predict_df:
            raise('Scorer not supported by autoClassify')
        return self._predict_df.sort_values(by=score,ascending=False).loc[0,'modelObject']

    def predict(self,testSet,model=None):
        """
        Returns preidct array for testSet passed as input
        :param testSet: input on which you want model to predict output
        :param model: model to be used to predict the ouput
        :return: an array of prediction values
        sample call
        autoClassify.predict(X,model=tmp.getModelObject('DecisionTreeClassifier'))
        """
        if not model:
            model = self.getBestModel()
        return model.predict(testSet)






