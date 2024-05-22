

import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from meta_data import *
from meta_data import get_data, get_meta, heart_feature_cols
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform
    

def fit_model(files, train_datasets, parent_path, cols_base, target, method="logistic"):
    ''' set hyper para '''
    # datasets, dataframes, test_name = get_data(csv_train50_rand64, train50_rand64_datasets)
    datasets, dataframes, test_name = get_data(files, train_datasets, parent_path)


    # test = dataframes['test_cols_base_size7763_rand42']
    test = dataframes[test_name]

    accuracy_trace = []

    ''' experiments '''
    for dataset in datasets:
        train = pd.concat([dataframes[file.split('.')[0]] for file in dataset])
        X_train = train[cols_base]
        y_train = train[target]

        # Add constant to predictors
        X_train = sm.add_constant(X_train)
        
        if method in method_list:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)

        # Fit logistic regression model
        if method == "logistic":
            model_train = sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit()
        elif method == "randomForest":
            # model_train = RandomForestClassifier()
            # model_train.fit(X_train, y_train)
            
            # param_grid = { 
            #     'n_estimators': [50, 100, 200, 400, 600],
            #     'max_features': ['sqrt', 'log2', len(X_train.axes[1])],
            #     'max_depth' : randint(4, 8),
            #     'criterion' :['gini', 'entropy'],
            #     'min_samples_split': [2, 5, 10],
            #     'min_samples_leaf': [1, 2, 4]
            # }
            

            # rfc = RandomForestClassifier(random_state=42)

            # # Initialize RandomizedSearchCV
            # random_search = RandomizedSearchCV(
            #     rfc, 
            #     param_distributions=param_grid, 
            #     n_iter=50, 
            #     cv=5, 
            #     verbose=0, 
            #     random_state=42, 
            #     n_jobs=-1
            # )

            # # Perform Randomized Search
            # random_search.fit(X_train, y_train)
            # model_train = random_search
            
            # print("Best Parameters Found: ")
            # print(model_train.best_params_)
            
            rfc = RandomForestClassifier(
                n_estimators=100,      # Number of trees in the forest. 100 is a good balance for most cases.
                max_features='auto',   # 'auto' lets the model choose sqrt(n_features).
                max_depth=None,        # No maximum depth to let the trees grow as much as needed.
                criterion='gini',      # 'gini' is a good default for classification tasks.
                min_samples_split=2,   # Minimum number of samples required to split an internal node.
                min_samples_leaf=1,    # Minimum number of samples required to be at a leaf node.
                random_state=42        # For reproducibility of results.
            )
            
            rfc.fit(X_train, y_train)
            model_train = rfc
            
        elif method == "svm":
            # model_train = SVC(random_state=42)
            # model_train.fit(X_train, y_train)
            svm = SVC(random_state=42)
            
            grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001], 
              'kernel': ['rbf', 'sigmoid']}
            gscv = GridSearchCV(estimator=svm ,param_grid=grid, scoring ='accuracy', refit=True, cv=3, verbose=1)
            gscv.fit(X_train,y_train)
            model_train = gscv
            
            print("Best Parameters Found: ")
            print(model_train.best_params_)
            
        elif method == "bayes":
            model_train = GaussianNB()
            model_train.fit(X_train, y_train)
        elif method == "catBoost":
            # model_train = CatBoostClassifier(verbose=0)  
            # model_train.fit(X_train, y_train)
            
            # grid = {
            #     'depth': [4, 6, 8, 10],
            #     'learning_rate': [0.01, 0.05, 0.1, 0.3],
            #     'iterations'    : [30, 50, 100]
            # }
            # cbc = CatBoostClassifier(random_state=42, verbose=0)

            # # Set up GridSearchCV
            # gscv = GridSearchCV (estimator = cbc, param_grid = grid, scoring ='accuracy', cv = 5)

            # #fit the model
            # gscv.fit(X_train,y_train)
            # model_train = gscv
            
            # print("Best Parameters Found: ")
            # print(model_train.best_params_)
            
            
            cbc = CatBoostClassifier(
                iterations=100,         # A moderate number of trees, balancing performance and training time.
                learning_rate=0.05,     # A slightly smaller learning rate for more robust convergence.
                depth=6,                # A moderate depth, balancing model complexity and overfitting risk.
                random_state=42,        # For reproducibility of results.
                verbose=0               # Suppress training output for cleaner logs.
            )
            
            cbc.fit(X_train, y_train)
            model_train = cbc


        # Predict on test set
        X_test = test[cols_base]
        X_test = sm.add_constant(X_test)
        y_test = test[target]
        if (method in method_list):
            X_test = scaler.transform(X_test)
        
        y_pred_test = model_train.predict(X_test)
        y_pred_test = [1 if p > 0.5 else 0 for p in y_pred_test]

        # Calculate accuracy
        accuracy_train = accuracy_score(y_test, y_pred_test)

        # print("test acc: {}".format(accuracy_train))
        accuracy_trace.append(accuracy_train)
    
    
    # diff = [add_acc - accuracy_trace[0] for add_acc in accuracy_trace[1:len(accuracy_trace)-1]]
    # print(diff)
    # return diff
    
    return accuracy_trace



def fit_model_chunk(train_name, traina_name, test_name, traina_num, traina_size, parent_path, cols_base, target, method="logistic", model_random_seed = 1, data_name=None, metric = "accuracy"):
    ''' set hyper para '''
    # datasets, dataframes, test_name = get_data(csv_train50_rand64, train50_rand64_datasets)

    error_trace = []
    train_init, traina_init, test_init = get_data_chunk(train_name, traina_name, test_name, parent_path)

    ''' experiments '''
    for tmp_traina_num in range(traina_num + 1):
        train, traina, test = train_init, traina_init, test_init
        traina_chunck = traina.iloc[0:traina_size*tmp_traina_num]
        train = pd.concat([train, traina_chunck])
        X_train = train[cols_base]
        y_train = train[target]

        # Add constant to predictors
        X_train = sm.add_constant(X_train)
        
        if method in method_list:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)

        # Fit logistic regression model
        if method == "logistic":
            model_train = sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit()
        elif method == "randomForest":
            
            # param_grid = { 
            #     'n_estimators': [50, 100, 200, 400, 600],
            #     'max_features': ['sqrt', 'log2', len(X_train.axes[1])],
            #     'max_depth' : randint(4, 8),
            #     'criterion' :['gini', 'entropy'],
            #     'min_samples_split': [2, 5, 10],
            #     'min_samples_leaf': [1, 2, 4]
            # }
            

            # rfc = RandomForestClassifier(random_state=42)

            # # Initialize RandomizedSearchCV
            # random_search = RandomizedSearchCV(
            #     rfc, 
            #     param_distributions=param_grid, 
            #     n_iter=50, 
            #     cv=5, 
            #     verbose=0, 
            #     random_state=42, 
            #     n_jobs=-1
            # )

            # # Perform Randomized Search
            # random_search.fit(X_train, y_train)
            # model_train = random_search
            
            # print("Best Parameters Found: ")
            # print(model_train.best_params_)
            
            if data_name == "openmlDiabetes":
                # rfc = RandomForestClassifier(class_weight='balanced', criterion='entropy',
                #        max_depth=51, max_features='log2', min_samples_leaf=2,
                #        min_samples_split=9, n_estimators=144, random_state=1)
                rfc = RandomForestClassifier(max_depth=21, max_features='log2', n_estimators=352,
                       random_state=1)
                
            elif data_name == "heart_failure":
                rfc = RandomForestClassifier(max_depth=74, max_features='log2', n_estimators=244,
                       random_state=1)
            elif data_name == "gender":
                rfc = RandomForestClassifier(max_depth=47, max_features='log2', n_estimators=335,
                       random_state=1)
            
            else:
                rfc = RandomForestClassifier(
                    n_estimators=200,      # Number of trees in the forest. 100 is a good balance for most cases.
                    max_features='sqrt',   # 'auto' lets the model choose sqrt(n_features).
                    max_depth=None,        # No maximum depth to let the trees grow as much as needed.
                    criterion='gini',      # 'gini' is a good default for classification tasks.
                    min_samples_split=2,   # Minimum number of samples required to split an internal node.
                    min_samples_leaf=2,    # Minimum number of samples required to be at a leaf node.
                    random_state=model_random_seed       # For reproducibility of results.
                )
            
            rfc.fit(X_train, y_train)
            model_train = rfc
            
        elif method == "svm":
            # model_train = SVC(random_state=42)
            # model_train.fit(X_train, y_train)
            svm = SVC(random_state=42)
            
            grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001], 
              'kernel': ['rbf', 'sigmoid']}
            gscv = GridSearchCV(estimator=svm ,param_grid=grid, scoring ='accuracy', refit=True, cv=3, verbose=1)
            gscv.fit(X_train,y_train)
            model_train = gscv
            
            print("Best Parameters Found: ")
            print(model_train.best_params_)
            
        elif method == "bayes":
            model_train = GaussianNB()
            model_train.fit(X_train, y_train)
        elif method == "catBoost":
            # model_train = CatBoostClassifier(verbose=0)  
            # model_train.fit(X_train, y_train)
            
            # grid = {
            #     'depth': [4, 6, 8, 10],
            #     'learning_rate': [0.01, 0.05, 0.1, 0.3],
            #     'iterations'    : [30, 50, 100]
            # }
            # cbc = CatBoostClassifier(random_state=42, verbose=0)

            # # Set up GridSearchCV
            # gscv = GridSearchCV (estimator = cbc, param_grid = grid, scoring ='accuracy', cv = 5)

            # #fit the model
            # gscv.fit(X_train,y_train)
            # model_train = gscv
            
            # print("Best Parameters Found: ")
            # print(model_train.best_params_)
            
            cbc = CatBoostClassifier(
                iterations=200,         # A moderate number of trees, balancing performance and training time.
                learning_rate=0.05,     # A slightly smaller learning rate for more robust convergence.
                depth=6,                # A moderate depth, balancing model complexity and overfitting risk.
                random_state=42,        # For reproducibility of results.
                verbose=0               # Suppress training output for cleaner logs.
            )
            
            cbc.fit(X_train, y_train)
            model_train = cbc

        # Predict on test set
        X_test = test[cols_base]
        X_test = sm.add_constant(X_test)
        y_test = test[target]
        if (method in method_list):
            X_test = scaler.transform(X_test)
        
        y_pred_test = model_train.predict(X_test)
        y_pred_test = [1 if p > 0.5 else 0 for p in y_pred_test]
        
        if metric == "accuracy":
            # Calculate accuracy
            accuracy_train = accuracy_score(y_test, y_pred_test)

            # print("test acc: {}".format(accuracy_train))
            error_trace.append(1 - accuracy_train)
        
        elif metric == "f1":
            # Calculate F1 score
            f1_score_train = f1_score(y_test, y_pred_test, average='binary')  # Use 'binary' for binary classification, 'macro'/'micro'/'weighted' for multiclass

            # Append 1 - F1 score to error_trace
            error_trace.append(1 - f1_score_train)

    return error_trace


if __name__ == "__main__":
    
    # cols_base = ['Age', 'Diabetes', 'Smoking', 'Obesity', 'Income']
    cols_base = heart_feature_cols["colsbase"][:-1]
    
    feature_cols_name = "colsbase"
    train_size = 50
    traina_size = 50
    train_total_size = 7776
    files, train_datasets = get_meta(feature_cols_name, train_size, traina_size, train_total_size, 8, 4)
    current_path = "../data/heart_data"
    
    files, train_datasets = get_meta(feature_cols_name, train_size, traina_size, train_total_size, 8, 4)
    target = 'Heart Attack Risk'
    
    # print(files)
    # print(50*"*")
    # print(train_datasets)
    
    
    accuracy_trace = fit_model(files, train_datasets, current_path, cols_base, target, "catBoost")
    
    # print(cols_base)
    
    print(accuracy_trace)