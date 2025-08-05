from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from src.mlflow_handler import log_run
from math import sqrt

def get_models():
    return {
        "linear": (LinearRegression(), {}),
        "xgboost": (XGBRegressor(), {"n_estimators": [50, 100, 150]}),
        "lgbm": (LGBMRegressor(), {"learning_rate": [0.05, 0.1]})
    }

def train_and_select(X_train, X_test, y_train, y_test, model_names):
    results = []
    all_models = get_models()

    for name in model_names:
        model, params = all_models[name]
        search = GridSearchCV(model, params, cv=3)
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        preds = best_model.predict(X_test)
        rmse = sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        log_run(name, search.best_params_, {"rmse": rmse, "r2": r2}, best_model)

        results.append((name, rmse, r2))

    return results
