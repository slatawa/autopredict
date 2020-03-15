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
from grid import getClassificationGridDict
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from classification import _base


class autoClassify:
    """
    Automate classification prediction

    The function runClassification automates finding supervised
    classification model.The runClassification function supports the following
    models right now

    """

    def __init__(self,cv=3,verbosity="warn",models=None,
                 encoder='label',scaler=None
                 ,useGridtuning=False,gridDict = None
                 ,score='roc_auc',random_state=None):
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
                           for sklearn's minmax sclaer ,'standard' - for sklearn's standard scaler
        :param useGridtuning: set this to True if you want to use grid search over the
                               supported classifier, the grid is selected based on configuration
                               saved in ./grid/_bases.py file in Dictionary gridDict
        :param gridDict: This variable is required if you use the Grid tuning option in autopredict
                         by setting useGridtuning= True in such scenario autopredict has ready made ranges for paramter
                         tweaking that it is going to test the model performance on , in case you want to over-ride those
                         parameters you can pass values in this argument , sample
                         gridDict={'LogisticRegression': {'penalty': ['l2'], 'C': [0.001, 0.1, 1, 10]}, 'DecisionTreeClassifier': {'max_depth': [4, 5, 6, 7, 8, 9, 10]}}
                         if you want to see the default config used by autopredict,use below function in autopredict.grid
                         grid.getClassificationGridDict() , this function will return the possible values
                         you can use the output, tweak the values and pass it as input to autopredict. You can not add
                         new keys in the dict as the keys present are the only ones supported, you could edit the values
                         of the dict to change the grid tuning params
        :param score: scoring parameter to be used for grid search
        :param random_state: seed parameter
        """
        self.cv = cv
        self.verbosity=verbosity
        self.random_state = random_state

        if models:
            self.models=models
            self.useGridtuning = False
        else:
            self.models=_base._getClassModelsMetadata(self.random_state)
            self.useGridtuning = useGridtuning

        self._predict_df = pd.DataFrame(columns=['modelName','modelObject','modelParams'])
        self._encode_dict = {'label':LabelEncoder(),'hot':OneHotEncoder()}
        self._scaler_dict = {'standard':StandardScaler(),'minmax':MinMaxScaler()}
        self.encoder=encoder
        self.scaler=scaler
        self.score = score


    ## add checks here

    def _scaleData(self,X,scaleObj):
        for rec in X.columns:
            if str(X[rec].dtype).startswith('int') or str(X[rec].dtype).startswith('float'):
                X[rec] = scaleObj.fit_transform(X[rec].to_numpy().reshape(-1,1))
        return X


    def _encodeData(self,X,encodeObj):
        for rec in X.columns:
            if X[rec].dtype == 'object':
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
        model.fit(X,y)
        self._predict_df= self._predict_df.append({'modelName':key,
                                                   'modelObject':model,
                                                  'modelParams':model.get_params()},
                                                  ignore_index=True)

    def getModelMetadata(self):
        return self.models

    def train(self,X,y):

        ## check for null values
        assert X.isna().any().sum() == 0, 'Dataframe X passed as input has null values ' \
                                                 'pleae run df.isna().any() to indetify null' \
                                                 'columns and either drop or fill these null values' \
                                                 'before passing to autopredict'
        # print(X.isna().any())

        if self.scaler:
            if self.scaler not in self._scaler_dict.keys():
                raise ValueError(f'Scaler key not defined, look at the scaler parameter '
                                 f'that is being passed in {self.scaler}')
            X = self._scaleData(X,self._scaler_dict[self.scaler])

        ## check if any data to be converted from str/object
        X = self._encodeData(X, self._encode_dict[self.encoder])

        if self.useGridtuning:
            for rec in self.models:
                key = str(rec).split('(')[0]
                gridDict= getClassificationGridDict()
                if key not in gridDict.keys():
                    print(key)
                    raise ValueError(f' {key} is not supported by Gridsearch paramter dict, look at ./grid/_.base'
                          f'-> gridDict')
                gridvalues = gridDict[key]
                gsModel = GridSearchCV(estimator=rec
                                       ,param_grid=gridvalues
                                       ,scoring=self.score
                                       ,cv=self.cv
                                       ,n_jobs=-1).fit(X, y)
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






