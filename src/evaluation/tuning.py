from __future__ import annotations
from sklearn.model_selection import GridSearchCV

def tune_model(pipeline, param_grid, x_train, y_train):
    search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_