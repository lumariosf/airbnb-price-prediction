from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np

def analise_hiperparametros(features, target):
    
    param_grid = {
    "n_estimators": np.arange(200, 800, 100),     
    "learning_rate": [0.01, 0.03, 0.05, 0.1],     
    "max_depth": [3, 4, 5, 6, 7],                 
    "min_child_weight": [1, 3, 5, 7],             
    "subsample": [0.7, 0.8, 0.9, 1.0],            
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],     
    "gamma": [0, 0.1, 0.3, 1],                    
    "reg_alpha": [0, 0.001, 0.01, 0.1, 1],        
    "reg_lambda": [0.1, 1, 5, 10]                 
    }

    grid = GridSearchCV(
    XGBRegressor(objective="reg:squarederror"), 
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1
    )
    grid.fit(features, target)

    return grid.best_estimator_

def xgb(features, target, test_features):

    model = XGBRegressor()
    best_model = analise_hiperparametros(model, features, target)

    target_pred = best_model.predict(test_features)
    return target_pred