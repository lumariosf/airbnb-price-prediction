import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge

# assumindo que as features já estão normalizadas
# vamos usar random search para tunagem
# como temos features com > 0.7 de correlação, vamos usar Ridge
def analise_hiperparametro(model, features, target):

    param_space = {"alpha": np.logspace(-4, 4, 100),     
                   "fit_intercept": [True, False],
                    "solver": ["auto", "svd", "lsqr", "saga"],
                    "positive": [False, True] }
    
    random_search = RandomizedSearchCV(model, param_space, n_iter=100, cv=5)
    random_search.fit(features, target)

    return random_search.best_estimator_

def linear_regression(features, target, test_features):
    
    model = Ridge()
    best_model = analise_hiperparametro(model, features, target)
    
    target_pred = best_model.predict(test_features)

    return target_pred