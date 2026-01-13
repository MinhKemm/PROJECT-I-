from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np

def search_params(base_model, param_grid, X, y, cv=5, search_type='grid'):
    print(f"Bắt đầu Tuning ({search_type})...")
    
    if search_type == 'grid':
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='neg_root_mean_squared_error', 
            cv=cv,
            n_jobs=-1, 
            verbose=1
        )
    else:
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=20, 
            scoring='neg_root_mean_squared_error',
            cv=cv,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
    search.fit(X, y)
    
    print(f"Best RMSE: {-search.best_score_:.4f}")
    print(f"Best Params: {search.best_params_}")
    
    return search.best_estimator_, search.best_params_