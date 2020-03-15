from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


def _getClassModelsMetadata(seed):
    return [LogisticRegression(random_state=seed)
                     ,DecisionTreeClassifier(random_state=seed)
            #,KNeighborsClassifier(n_jobs=-1)
            ,GaussianNB()
            ,RandomForestClassifier(random_state=seed,n_jobs=-1)
            ,GradientBoostingClassifier(random_state=seed)]