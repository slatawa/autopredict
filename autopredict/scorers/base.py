from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from ..classification import _base

classifyScorers = [roc_auc_score,accuracy_score,f1_score,recall_score,precision_score,confusion_matrix]
regressScores=[]
scaleModels = _base._getScaleModels()


def classifyScore(scoreBoard, X, y,X_scaled_test,y_scaled_test,multiClass,scalerData):

    #multi_class = 'ovo' if multiClass else None
    for rec in classifyScorers:
        if rec.__name__ == 'roc_auc_score':
            if not multiClass:
                scoreBoard[str(rec.__name__)] = scoreBoard.apply(
                    lambda x: rec(y_scaled_test, x['modelObject'].predict_proba(X_scaled_test)[:,1])
                    if x['modelName'] in scaleModels and scalerData
                    else rec(y, x['modelObject'].predict_proba(X)[:,1]), axis=1)
            else:
                scoreBoard[str(rec.__name__)] = scoreBoard.apply(
                    lambda x: rec(y_scaled_test, x['modelObject'].predict_proba(X_scaled_test),multi_class='ovo')
                    if x['modelName'] in scaleModels and scalerData
                    else rec(y, x['modelObject'].predict_proba(X),multi_class='ovo'), axis=1)
        elif rec.__name__ in ('accuracy_score','confusion_matrix'):
            #scoreBoard[str(rec.__name__)] = scoreBoard['modelObject'].apply(lambda x: rec(y, x.predict(X)))
            scoreBoard[str(rec.__name__)] = scoreBoard.apply(lambda x: rec(y_scaled_test, x['modelObject'].predict(X_scaled_test))
            if x['modelName'] in scaleModels and scalerData else rec(y, x['modelObject'].predict(X)),axis=1)
        else:
            scoreBoard[str(rec.__name__)] = scoreBoard.apply(lambda x: rec(y_scaled_test, x['modelObject'].predict(X_scaled_test),average='macro')
            if x['modelName'] in scaleModels and scalerData else rec(y, x['modelObject'].predict(X),average='macro'),axis=1)
            #scoreBoard[str(rec.__name__)] = scoreBoard['modelObject'].apply(lambda x: rec(y, x.predict(X),average='macro'))
    return scoreBoard


def regressScore(scoreBoard, X, y):
    for rec in classifyScorers:
        if rec.__name__ == 'roc_auc_score':
            scoreBoard[str(rec.__name__)] = scoreBoard['modelObject'].apply(
                lambda x: rec(y, x.predict_proba(X)[:, 1]))
        else:
            scoreBoard[str(rec.__name__)] = scoreBoard['modelObject'].apply(lambda x: rec(y, x.predict(X)))
    return scoreBoard