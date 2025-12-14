from sklearn.neighbors import  KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

def analise_hiperparametros(model, features, target):
    
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'chebyshev']
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error', 
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(features, target)

    return grid_search

def knn(features, target, test_features):

    model = KNeighborsRegressor()
    best_model = analise_hiperparametros(model, features, target)

    target_pred = best_model.predict(test_features)

    return target_pred, best_model