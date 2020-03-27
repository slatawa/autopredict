# -*- encoding: utf-8 -*-
import sys
sys.path.append("..")
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from ..grid import getClassificationGridDict
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from ..base import basePredict
import autopredict.classification._base
import logging
import time



class autoClassify(basePredict):
    """
    Automate classification prediction

    The function runClassification automates finding supervised
    classification model.The runClassification function supports the following
    models right now

    """


    def __init__(self,cv=3,verbosity="INFO",models=None,
                 encoder='label',scaler=None
                 ,useGridtuning=False,gridDict = None
                 ,score='roc_auc',random_state=None):
        """

        :param cv:         cross validation sets
        :param verbosity:  level of logging - 0,1,2
        :param models:     List of model objects for which you want to run the train data through
                           below is a sample input, by default this is null in which case train data
                           ,could be run through all supported models with default parameters of autopredict.
                           if this argument is passed in then use_grid_tuning would be over-riden to False
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
        ## add checks here
        if verbosity not in self._logSupport:
            raise Exception('verbosity paramater can only have the values {_logSupport}')
        logging.basicConfig(level=verbosity)

        self.multiClass = False

        super().__init__(objective=self._classify, cv=cv, verbosity=verbosity, models=models,
                         encoder=encoder, scaler=scaler
                         ,useGridtuning=useGridtuning, gridDict=gridDict
                         , score=score, random_state=random_state)

    def fit(self, X, y):
        startTime = time.time()

        if X.isna().any().sum() > 0:
            logging.warning('Input DataFrame passed has null values - autopredict '
                            'is going to replace the Nan with the most frequent occuring '
                            'value in each column')
        try:
            for rec in X.columns:
                X = X.fillna(X.mode().iloc[0])
        except Exception as e:
            raise Exception(f'Failed to replace NAN values in the dataframe {str(e)}')


        if self.scaler:
            if self.scaler not in self._scaler_dict.keys():
                raise ValueError(f'Scaler key not defined, look at the scaler parameter '
                                 f'that is being passed in {self.scaler}')
            X_scaled = self._scaleData(X, self._scaler_dict[self.scaler])

        ## check if any data to be converted from str/object
        X = self._encodeData(X, self._encode_dict[self.encoder])
        if self.scaler:
            X_scaled = self._encodeData(X_scaled, self._encode_dict[self.encoder])

        ## check if binary classification

        if len(y.value_counts()) > 2:
            logging.warning('More then 2 target classes detected , scoring '
                            'for grid search will be over-ridden to - ''Accuracy''')
            self.score = 'accuracy'
            self.multiClass = True

        X_scaled = X_scaled if self.scaler else None
        super().fit(X,X_scaled, y,self.multiClass)


        ## get scores
        logging.info('Training of models complete')
        logging.info(f'Total training time {round(time.time()-startTime,1)} seconds')
        return self










