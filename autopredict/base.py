from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from classification import _base
from grid import getClassificationGridDict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import scorers
import logging


class basePredict:
    _classify = 'classify'
    _regression = 'regress'
    _logSupport = ['WARNING', 'INFO', 'DEBUG']
    def __init__(self,objective,cv=3,verbosity="warn",models=None,
                 encoder='label',scaler=None
                 ,useGridtuning=False,gridDict = None
                 ,score='roc_auc',random_state=None):
        self.cv = cv
        self.verbosity = verbosity
        self.random_state = random_state
        self.objective = objective

        if models:
            self.models = models
            self.useGridtuning = False
        else:
            if self.objective == self._classify :
                self.models = _base._getClassModelsMetadata(self.random_state)
                self.useGridtuning = useGridtuning
            else:
                pass

        self._predict_df = pd.DataFrame(columns=['modelName', 'modelObject', 'modelParams'])
        self._encode_dict = {'label': LabelEncoder(), 'hot': OneHotEncoder()}
        self._scaler_dict = {'standard': StandardScaler(), 'minmax': MinMaxScaler()}
        self.encoder = encoder
        self.scaler = scaler
        self.score = score
        self.scaleModels = _base._getScaleModels()

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

    def _score(self,X,y,X_scaled_test=None,y_scaled_test=None,scalerData=False):
        ### add scorers here
        if self.objective == self._classify:
            self._predict_df = scorers.classifyScore(self._predict_df,X,y,X_scaled_test,y_scaled_test
                                                     ,self.multiClass,scalerData)
        else:
            self._predict_df = scorers.regressScore(self._predict_df, X, y,X_scaled_test,y_scaled_test)

    def _applyModel(self,model,X,y,params=None):
        key = str(model).split('(')[0]
        model.fit(X,y)
        self._predict_df= self._predict_df.append({'modelName':key,
                                                   'modelObject':model,
                                                  'modelParams':model.get_params()},
                                                  ignore_index=True)

    def getModelMetadata(self):
        return self.models

    def fit(self, X,X_scaled, y,multiClass=False):
        self.multiClass = multiClass
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=self.random_state)
        if self.scaler:
            X_scaled_train, X_scaled_test, y_scaled_train, y_scaled_test = \
                train_test_split(X_scaled, y, random_state=self.random_state)

        if self.useGridtuning:
            for rec in self.models:
                key = str(rec).split('(')[0]
                gridDict = getClassificationGridDict()
                if key not in gridDict.keys():
                    raise ValueError(f' {key} is not supported by Gridsearch paramter dict, look at ./grid/_.base'
                                     f'-> gridDict')
                gridvalues = gridDict[key]

                if key in self.scaleModels and self.scaler:
                    gsModel = GridSearchCV(estimator=rec
                                           , param_grid=gridvalues
                                           , scoring=self.score
                                           , cv=self.cv
                                           , n_jobs=-1).fit(X_scaled_train, y_scaled_train)
                else:
                    gsModel = GridSearchCV(estimator=rec
                                           , param_grid=gridvalues
                                           , scoring=self.score
                                           , cv=self.cv
                                           , n_jobs=-1).fit(X_train, y_train)
                self._predict_df = self._predict_df.append({'modelName': key,
                                                            'modelObject': gsModel.best_estimator_,
                                                            'modelParams': gsModel.best_params_,
                                                            'gridSearchScore': gsModel.best_score_},
                                                           ignore_index=True)
        else:
            ## split the data into train and test sets
            for rec in self.models:
                key = str(rec).split('(')[0]
                ## Fit the data into the model
                if key in self.scaleModels and self.scaler:
                    self._applyModel(rec, X_scaled_train, y_scaled_train)
                else:
                    self._applyModel(rec, X_train, y_train)
        if self.scaler:
            self._score(X_test, y_test,X_scaled_test,y_scaled_test,self.scaler)
        else:
            self._score(X_test, y_test, self.scaler)


    def getModelScores(self):
        """
        :return: return models trained by autoclassify along with respective stores
        sample
        autoClassify().getModelScores()
        """
        return self._predict_df

    def getModelObject(self, modelName):
        """
        this function is used
        :param modelName:  name of the model whose object you want back, all possible options
                        can be retrieved by using autoClassify().getModelScores()
        :return: returns trained model object which was trained under autopredict's train function
        sample call
        autoClassify().getModelObject(modelName='LogisticRegression')
        """
        return self._predict_df.loc[self._predict_df['modelName'] == modelName, 'modelObject'].iloc[0]

    def getBestModel(self, score='score'):
        """
        :param score:  by default this is set to score method, scorers supported by
        autoClassify can be passed here
        :return:
        sample call
        autoClassify().getBestModel()
        """
        if score not in self._predict_df:
            raise ('Scorer not supported by autoClassify')
        return self._predict_df.sort_values(by=score, ascending=False).loc[0, 'modelObject']

    def predict(self, testSet, model=None):
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


