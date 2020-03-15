classificationGridDict =  {'LogisticRegression':{'penalty':['l2']
                               ,'C':[0.001,0.1,1,10]}
             ,'DecisionTreeClassifier': {'max_depth':[4,5,6,7,8,9,10]}

             #,'KNeighborsClassifier':{'n_neighbors':[5,6,7]}
             ,'GaussianNB':{}
            ,'RandomForestClassifier':{'n_estimators':[100,500,1000]
                                       ,'max_depth':[4,5,6,7]}
            ,'GradientBoostingClassifier':{'learning_rate':[0.01,0.1,0.2,0.3]
                                           ,'n_estimators':[100,500,1000]}
                           }

def getClassificationGridDict():
    return classificationGridDict
