import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy.stats import randint, uniform
from meta_data import *

def tune_random_forest(data_path, target_name, colsbase, task_type, model_rand_seed=1, iter=200):
    # Load the dataset
    df = pd.read_csv(data_path)
    
    # Separate the target variable and the features
    y = df[target_name]
    X = df[colsbase]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=model_rand_seed)
    
    # Initialize the RandomForestRegressor
    if task_type == "regress":
        rf = RandomForestRegressor(random_state=1)
        param_distributions = {
            'n_estimators': randint(100, 600),  # Number of trees in the forest
            'max_features': ['sqrt'],   # Number of features to consider at every split
            'max_depth': randint(10, 100),      # Maximum number of levels in tree
            # 'min_samples_split': randint(2, 10),# Minimum number of samples required to split a node
            # 'min_samples_leaf': randint(1, 4),  # Minimum number of samples required at each leaf node
        }
    elif task_type == "class":
        rf = RandomForestClassifier(random_state=1)
        param_distributions = {
            'n_estimators': randint(100, 600),            # Number of trees in the forest
            'max_features': ['log2'],       # Number of features to consider at every split
            'max_depth': randint(10, 100),                # Maximum number of levels in tree
            # 'min_samples_split': randint(2, 10),          # Minimum number of samples required to split a node
            # 'min_samples_leaf': randint(1, 4),            # Minimum number of samples required at each leaf node
            # 'criterion': ['gini'],             # The function to measure the quality of a split
        }
    
    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, n_iter=iter, cv=3, verbose=0, random_state=1, n_jobs=-1)
    
    # Fit the model
    random_search.fit(X_train, y_train)
    
    # Retrieve the best model
    best_model = random_search.best_estimator_
    
    # Evaluate the model
    score = best_model.score(X_test, y_test)
    
    return best_model, score


if __name__ == "__main__":
    data_name = "gender"
    
    info = info_dict[data_name]
    total_size = info["total_size"]
    target_name = info["target"]
    colsbase = info["colsbase"]
    task_type = info["task_type"]
    raw_size = info["raw_size"]
    shap_raw = info["shap_raw"]
    
    parent_path = "../data/{}/{}_see_small_raw{}{}".format(data_name, data_name, shap_raw, post_shap)
    data_name = "/train_colsbase_size{}_total{}_rand1.csv".format(shap_raw, total_size)
    data_path = parent_path + data_name

    
    best_model, best_score = tune_random_forest(data_path, target_name, colsbase, task_type)
    
    print(best_model)
