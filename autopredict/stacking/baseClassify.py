from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold
import numpy as np
import logging

logging.basicConfig(level=20)

class stackClassify():

    def __init__(self,splits,stackerModel,models,score='roc_auc',seed=100):
        """
        This is initializer for class stackClassify which can be used
        to stack multiple methods for generating a prediction outcome
        :param splits: decides how many split to do for Kfold
        :param stackerModel: the model that you want to use to give final output
        :param models: base models , the outputs from these would be used as input to stacker model
        :param score: scoring mechanism to be used by default roc_auc
        :param seed: random seed to generate consistent result
        """
        self._splits = splits
        self._seed = seed
        self._score = score
        self._models = models
        self._stackerModel = stackerModel

    def fit_predict(self,X,y,test):
        """
        This is a method of stackClassify class and is used to fit and predict output
        X and y are inputs on which all the base models are fitted,test is the final set
        on which the output is given
        :param X: Input features
        :param y: Target variable
        :param test: Test feature set
        :return: return predict_proba for test feature set

        # sample usage below for binary classification
        # which used lgb and catboost as base models and
        # logistic regression as the stack model to give final output probabilities

        from lightgbm import LGBMClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.model_selection import cross_val_score
        from autopredict.stacking import stackClassify
        from catboost import CatBoostClassifier

        # LightGBM params
        lgb_params = {}
        lgb_params['learning_rate'] = 0.01
        lgb_params['n_estimators'] = 250
        lgb_params['max_bin'] = 9
        lgb_params['subsample'] = 0.7
        lgb_params['subsample_freq'] = 10
        lgb_params['colsample_bytree'] = 0.8
        lgb_params['min_child_samples'] = 500
        lgb_params['seed'] = 99

        logmodel = LogisticRegression()
        lgmodel = LGBMClassifier(**lgb_params)


        cat_params = {}
        cat_params['iterations'] = 700
        cat_params['depth'] = 7
        cat_params['rsm'] = 0.90
        cat_params['learning_rate'] = 0.03
        cat_params['l2_leaf_reg'] = 3.5
        cat_params['border_count'] = 8

        catmodel = CatBoostClassifier(**cat_params)

        tmp = stackClassify(splits=2,stackerModel=logmodel ,
                      models= [lgmodel,catmodel],score='roc_auc',seed=100)

        _ = tmp.fit_predict(X=train,y=target_train,test=test)


        """
        if self._splits <= 1:
            logging.warning('Splits need to be more then 1 for Stacking to work')
            exit(250)
        X = np.array(X)
        y = np.array(y)
        test = np.array(test)

        # lets get our score hold arrays ready
        hold_score = np.zeros((X.shape[0], len(self._models)))
        test_score = np.zeros((test.shape[0], len(self._models)))

        # get the folds
        folds = StratifiedKFold(n_splits=self._splits,random_state=self._seed,shuffle=True).split(X,y)

        # start getting the scores
        for i, model in enumerate(self._models):
            test_score_temp = np.zeros((test.shape[0], self._splits))
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_valid = X[test_idx]
                modName = str(model).split('(')[0]
                logging.info(f'Fitting Model - {modName} fit - Split {j+1}')
                model.fit(X_train, y_train)
                y_pred = model.predict_proba(X_valid)[:, 1]
                hold_score[test_idx, i] = y_pred
                test_score_temp[:, j] = model.predict_proba(test)[:, 1]
            test_score[:,i] = test_score_temp.mean(axis=1)

        result = cross_val_score(self._stackerModel, hold_score, y, cv=self._splits, scoring=self._score )
        logging.info(f'score {result.mean()}')
        self._stackerModel.fit(hold_score, y)
        res = self._stackerModel.predict_proba(test_score)[:, 1]
        return res






