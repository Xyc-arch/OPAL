import pandas as pd
import numpy as np
from meta_data import * 
from split import sample_remain_no_replace
from sklearn.utils import shuffle

import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import statistics

def calculate_proportions(data_file, target_name, attribute_name):
    data = pd.read_csv(data_file)
    data[f'{attribute_name}_cat'] = data[attribute_name].apply(lambda x: '0' if x == 0 else 'positive')
    
    total_proportions = data.groupby([f'{attribute_name}_cat', target_name]).size() / len(data)
    total_proportions = total_proportions.reset_index(name='proportion')
    
    print(total_proportions)



def feature_target_correlation(data_file, target_name):
    data = pd.read_csv(data_file)
    correlations = data.corr()[target_name].drop(target_name)  
    correlations = correlations.sort_values(key=abs)  
    for feature, value in correlations.items():
        print(f"{feature}: {value}")
        
        
        
        
def spurious_split(feature_cols, feature_cols_name, parent_path, split_file, target, spurious_attr, associate, file_size, test_size, rand_seed, major_size, minor_size):
    cols = feature_cols[feature_cols_name]
    df = pd.read_csv(parent_path + split_file)
    df = df[cols]
    df['spurious_cat'] = df[spurious_attr].apply(lambda x: '0' if x == 0 else 'positive')
    remain = df
    test, remain = sample_remain_no_replace(remain, test_size, rand_seed)
    train_full_size = file_size - test_size
    
    test = test.drop('spurious_cat', axis=1)
    test.to_csv(parent_path + "/test_{}_size{}_rand{}.csv".format(feature_cols_name, train_full_size, rand_seed), index=False)
    
    # Classify into groups
    groups = remain.groupby(['spurious_cat', target])
    
    group11 = groups.get_group(('positive', 1))
    group00 = groups.get_group(('0', 0))
    group01 = groups.get_group(('0', 1))
    group10 = groups.get_group(('positive', 0))
    
    if associate == 1:
        maj1 = group11.sample(n=major_size, random_state=rand_seed)
        maj2 = group00.sample(n=major_size, random_state=rand_seed)
        maj1_mix, maj1_remain = sample_remain_no_replace(maj1, minor_size, rand_seed)
        maj2_mix, maj2_remain = sample_remain_no_replace(maj2, minor_size, rand_seed)
        min1 = group01.sample(n=minor_size, random_state=rand_seed)
        min2 = group10.sample(n=minor_size, random_state=rand_seed)
        
        # maj_sample1 = maj11.sample(n=major_size, random_state=rand_seed)
        # maj_sample2 = maj00.sample(n=major_size, random_state=rand_seed)
        # combined_major = pd.concat([maj_sample1, maj_sample2]).sample(n=2*minor_size, random_state=rand_seed)
    else:
        maj1 = group10.sample(n=major_size, random_state=rand_seed)
        maj2 = group01.sample(n=major_size, random_state=rand_seed)
        maj1_mix, maj1_remain = sample_remain_no_replace(maj1, minor_size, rand_seed)
        maj2_mix, maj2_remain = sample_remain_no_replace(maj2, minor_size, rand_seed)
        min1 = group00.sample(n=minor_size, random_state=rand_seed)
        min2 = group11.sample(n=minor_size, random_state=rand_seed)
        
    seed_data = pd.concat([maj1_mix, maj2_mix, min1, min2]).sample(frac=1, random_state=rand_seed)
    maj_remain = pd.concat([maj1_remain, maj2_remain]).sample(frac=1, random_state=rand_seed)
    final_df = pd.concat([seed_data, maj_remain])
    final_df = final_df.drop('spurious_cat', axis=1)

    save_path = parent_path + "/train_{}_size{}_total{}_rand{}.csv".format(feature_cols_name, raw_size, train_full_size, rand_seed)
    final_df.to_csv(save_path, index=False)
    
    print(f"Data processed and saved to {save_path}")
    


def spurious_split_rand(feature_cols, feature_cols_name, parent_path, split_file, target, spurious_attr, associate, file_size, test_size, rand_seed_set, major_size, minor_size):
    for rand_seed in rand_seed_set:
        spurious_split(feature_cols, feature_cols_name, parent_path, split_file, target, spurious_attr, associate, file_size, test_size, rand_seed, major_size, minor_size)


# minority_associate = 1, then minority association are 11 or 00 ; minority_gate = 0: overall, 1 minority, -1 majority
def fit_model_chunk_spurious(minority_associate, spurious_attr, train_name, traina_name, test_name, traina_num, traina_size, parent_path, cols_base, target, method="logistic", model_random_seed = 1, data_name=None, metric = "accuracy", minority_gate = True):
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

        if minority_gate:
            test['spurious_cat'] = test[spurious_attr].apply(lambda x: '0' if x == 0 else 'positive')
            test_groups = test.groupby(['spurious_cat', target])
        
            test_group11 = test_groups.get_group(('positive', 1))
            test_group00 = test_groups.get_group(('0', 0))
            test_group01 = test_groups.get_group(('0', 1))
            test_group10 = test_groups.get_group(('positive', 0))
            
            test_group11 = test_group11.drop('spurious_cat', axis=1)
            test_group00 = test_group00.drop('spurious_cat', axis=1)
            test_group01 = test_group01.drop('spurious_cat', axis=1)
            test_group10 = test_group10.drop('spurious_cat', axis=1)
            
            if minority_gate == 1: # minority
                if minority_associate == 1:
                    test = pd.concat([test_group11, test_group00])
                elif minority_associate == 0:
                    test = pd.concat([test_group10, test_group01])
            
            elif minority_gate == -1: # majority
                if minority_associate == 1:
                    test = pd.concat([test_group10, test_group01])
                elif minority_associate == 0:
                    test = pd.concat([test_group11, test_group00])
        
        # Add constant to predictors
        X_train = sm.add_constant(X_train)
        
        if method in method_list:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)

        # Fit logistic regression model
        if method == "logistic":
            model_train = sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit()
        elif method == "randomForest":
            if data_name == "openmlDiabetes":
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


# output worst group
def fit_model_chunk_spurious_worst(minority_associate, spurious_attr, train_name, traina_name, test_name, traina_num, traina_size, parent_path, cols_base, target, method="logistic", model_random_seed=1, data_name=None, metric="accuracy", minority_gate=True):
    ''' Set hyperparameters '''
    error_trace = []
    train_init, traina_init, test_init = get_data_chunk(train_name, traina_name, test_name, parent_path)

    ''' Experiments '''
    for tmp_traina_num in range(traina_num + 1):
        train, traina, test = train_init, traina_init.copy(), test_init.copy()
        traina_chunk = traina.iloc[0:traina_size*tmp_traina_num]
        train = pd.concat([train, traina_chunk])
        X_train = train[cols_base]
        y_train = train[target]

        test['spurious_cat'] = test[spurious_attr].apply(lambda x: '0' if x == 0 else 'positive')
        test_groups = test.groupby(['spurious_cat', target])
        X_train.insert(0, 'Intercept', 1)
        errors = []

        for group_key in [('positive', 1), ('0', 0), ('0', 1), ('positive', 0)]:
            try:
                test_group = test_groups.get_group(group_key).drop(['spurious_cat'], axis=1)
                
                X_test = test_group[cols_base]
                y_test = test_group[target]

                X_test.insert(0, 'Intercept', 1)
                
                # print(X_test.shape)

                if method == "logistic":
                    model = sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit()
                elif method == "randomForest":
                    if data_name == "openmlDiabetes":
                        model = RandomForestClassifier(max_depth=21, max_features='log2', n_estimators=352,
                            random_state=1)
                        
                    elif data_name == "heart_failure":
                        model = RandomForestClassifier(max_depth=74, max_features='log2', n_estimators=244,
                            random_state=1)
                    elif data_name == "gender":
                        model = RandomForestClassifier(max_depth=47, max_features='log2', n_estimators=335,
                            random_state=1)
                    
                    else:
                        model = RandomForestClassifier(
                            n_estimators=200,      # Number of trees in the forest. 100 is a good balance for most cases.
                            max_features='sqrt',   # 'auto' lets the model choose sqrt(n_features).
                            max_depth=None,        # No maximum depth to let the trees grow as much as needed.
                            criterion='gini',      # 'gini' is a good default for classification tasks.
                            min_samples_split=2,   # Minimum number of samples required to split an internal node.
                            min_samples_leaf=2,    # Minimum number of samples required to be at a leaf node.
                            random_state=model_random_seed       # For reproducibility of results.
                        )
                    model.fit(X_train, y_train)
                        
                elif method == "catBoost":
                    model = CatBoostClassifier(
                        iterations=200,         # A moderate number of trees, balancing performance and training time.
                        learning_rate=0.05,     # A slightly smaller learning rate for more robust convergence.
                        depth=6,                # A moderate depth, balancing model complexity and overfitting risk.
                        random_state=42,        # For reproducibility of results.
                        verbose=0               # Suppress training output for cleaner logs.
                    )
                    
                    model.fit(X_train, y_train)


                # Model prediction
                y_pred = model.predict(X_test)
                y_pred = [1 if p > 0.5 else 0 for p in y_pred]

                # Error calculation
                if metric == "accuracy":
                    accuracy = accuracy_score(y_test, y_pred)
                    errors.append(1 - accuracy)
                elif metric == "f1":
                    f1 = f1_score(y_test, y_pred, average='binary')
                    errors.append(1 - f1)
            except KeyError:  # Handle missing group combination
                continue

        # After iterating through all groups, find the worst error
        if errors:  # Ensure there is at least one error recorded
            worst_error = max(errors)
            error_trace.append(worst_error)

    return error_trace


def fit_model_chunk_spurious_diff(spurious_attr, train_name, traina_name, test_name, traina_num, traina_size, parent_path, cols_base, target, method="logistic", model_random_seed=1, data_name=None, metric="accuracy"):
    error_trace = []
    train_init, traina_init, test_init = get_data_chunk(train_name, traina_name, test_name, parent_path)

    for tmp_traina_num in range(traina_num + 1):
        train, traina, test = train_init.copy(), traina_init.copy(), test_init.copy()
        traina_chunk = traina.iloc[0:traina_size * tmp_traina_num]
        train = pd.concat([train, traina_chunk])
        X_train = train[cols_base].values
        y_train = train[target].values

        # Prepare the test set
        test['spurious_cat'] = test[spurious_attr].apply(lambda x: '0' if x == 0 else 'positive')
        
        test_groups = test.groupby(['spurious_cat', target])
        
        errors = []
        
        ones_column = np.ones((X_train.shape[0], 1))  # Create a new column of ones
        X_train = np.hstack((X_train, ones_column))
        
        if method in method_list:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
        if method == "logistic":
            model_train = sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit()
        elif method == "randomForest":
            if data_name == "openmlDiabetes":
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
            
            cbc = CatBoostClassifier(
                iterations=200,         # A moderate number of trees, balancing performance and training time.
                learning_rate=0.05,     # A slightly smaller learning rate for more robust convergence.
                depth=6,                # A moderate depth, balancing model complexity and overfitting risk.
                random_state=42,        # For reproducibility of results.
                verbose=0               # Suppress training output for cleaner logs.
            )
            
            cbc.fit(X_train, y_train)
            model_train = cbc
            

        # Calculate errors for each group
        for group_name, group_df in test_groups:
            X_test_group = group_df[cols_base].values
            ones_column = np.ones((X_test_group.shape[0], 1))  # Create a new column of ones
            X_test_group = np.hstack((X_test_group, ones_column))
            y_test_group = group_df[target].values
            if (method in method_list):
                X_test_group = scaler.transform(X_test_group)
            y_pred_group = model_train.predict(X_test_group)
            y_pred_group = [1 if p > 0.5 else 0 for p in y_pred_group]

            # Calculate error
            if metric == "accuracy":
                accuracy = accuracy_score(y_test_group, y_pred_group)
                error = 1 - accuracy
            elif metric == "f1":
                error = 1 - f1_score(y_test_group, y_pred_group, average='binary')
            errors.append(error)

        # Calculate the maximum difference in errors
        max_diff = max(errors) - min(errors)
        error_trace.append(max_diff)

    return error_trace



def get_min_average(df, row_idx = 5):
    row = df.iloc[row_idx]
    row_min = min(row)
    row_avg = statistics.mean(row)
    
    return row_min, row_avg


def batch_run_rand_seed_chunk_spurious(minority_associate, spurious_attr, feature_cols_name, train_size, traina_size, trainfull_size, traina_num, rand_seed_set, cols_base, data_path, save_path, target, method="linear", traina_half_name = None, data_name = None, fix_raw = None, metric="accuracy", minority_gate = True):
   
    record = []
    print("The traina name is {}".format(traina_half_name))
      
    for rand_seed in rand_seed_set:
        if fix_raw:
            train_name = "train_{}_size{}_total{}_rand{}.csv".format(feature_cols_name, train_size, trainfull_size, fix_raw)
        else:
            train_name = "train_{}_size{}_total{}_rand{}.csv".format(feature_cols_name, train_size, trainfull_size, rand_seed)
        if traina_half_name:
            traina_name = traina_half_name + "{}.csv".format(rand_seed)
        else:
            traina_name = "train_generate_colsbase_rand{}.csv".format(rand_seed)
        if fix_raw:
            test_name = "test_{}_size{}_rand{}.csv".format(feature_cols_name, trainfull_size, fix_raw)
        else:
            test_name = "test_{}_size{}_rand{}.csv".format(feature_cols_name, trainfull_size, rand_seed)
            
        
        error = fit_model_chunk_spurious(minority_associate, spurious_attr, train_name, traina_name, test_name, traina_num, traina_size, data_path, cols_base, target, method=method, model_random_seed = 1, data_name=None, metric = "accuracy", minority_gate = minority_gate)
        record.append(error)

        df_record = pd.DataFrame(record)
    
    col_mean = df_record.mean(axis=0)
    col_std = df_record.std(axis=0)
    record.append(col_mean)
    record.append(col_std)
    df_record_to_csv = pd.DataFrame(record)
    df_avg_min, df_avg_mean = get_min_average(df_record_to_csv, row_idx=len(rand_seed_set))
    for index, row in df_record_to_csv.iterrows():
        row_as_list = list(row.values)
        formatted_row = "[" + ", ".join(repr(e) for e in row_as_list) + "]"
        print(f"Row {index}: {formatted_row}")
    # print(df_record_to_csv)
    print("min: {}, mean: {}".format(df_avg_min, df_avg_mean))
    df_record_to_csv.to_csv(save_path + 
                     "/perform_train_{}_size{}_total{}_addnum{}_{}.csv".format(feature_cols_name, train_size, trainfull_size, traina_num, method))
    
    plt.figure("mean")
    x_axis =[x for x in range(traina_num+1)]
    plt.errorbar(x_axis, col_mean, yerr=col_std, fmt='-o', capsize=5)
    plt.axhline(y = col_mean[0], color = 'r', linestyle = '-') 
    plt.xlabel('add num')
    plt.ylabel('error')
    save_name = "/{}_train_{}_size{}_total{}_addnum{}_{}".format("mean", feature_cols_name, train_size, trainfull_size, traina_num, method)
    plt.title("{} on {}".format(method, target))
    plt.savefig(save_path + save_name + '.png')
    
    return col_mean, col_std


def batch_run_rand_seed_chunk_spurious_worst(minority_associate, spurious_attr, feature_cols_name, train_size, traina_size, trainfull_size, traina_num, rand_seed_set, cols_base, data_path, save_path, target, method="linear", traina_half_name = None, data_name = None, fix_raw = None, metric="accuracy", minority_gate = True):
   
    record = []
    print("The traina name is {}".format(traina_half_name))
      
    for rand_seed in rand_seed_set:
        if fix_raw:
            train_name = "train_{}_size{}_total{}_rand{}.csv".format(feature_cols_name, train_size, trainfull_size, fix_raw)
        else:
            train_name = "train_{}_size{}_total{}_rand{}.csv".format(feature_cols_name, train_size, trainfull_size, rand_seed)
        if traina_half_name:
            traina_name = traina_half_name + "{}.csv".format(rand_seed)
        else:
            traina_name = "train_generate_colsbase_rand{}.csv".format(rand_seed)
        if fix_raw:
            test_name = "test_{}_size{}_rand{}.csv".format(feature_cols_name, trainfull_size, fix_raw)
        else:
            test_name = "test_{}_size{}_rand{}.csv".format(feature_cols_name, trainfull_size, rand_seed)
            
        
        error = fit_model_chunk_spurious_worst(minority_associate, spurious_attr, train_name, traina_name, test_name, traina_num, traina_size, data_path, cols_base, target, method=method, model_random_seed = 1, data_name=None, metric = "accuracy", minority_gate = minority_gate)
        record.append(error)

        df_record = pd.DataFrame(record)
    
    col_mean = df_record.mean(axis=0)
    col_std = df_record.std(axis=0)
    record.append(col_mean)
    record.append(col_std)
    df_record_to_csv = pd.DataFrame(record)
    df_avg_min, df_avg_mean = get_min_average(df_record_to_csv, row_idx=len(rand_seed_set))
    for index, row in df_record_to_csv.iterrows():
        row_as_list = list(row.values)
        formatted_row = "[" + ", ".join(repr(e) for e in row_as_list) + "]"
        print(f"Row {index}: {formatted_row}")
    # print(df_record_to_csv)
    print("min: {}, mean: {}".format(df_avg_min, df_avg_mean))
    
    return col_mean, col_std




def batch_run_rand_seed_chunk_spurious_diff(spurious_attr, feature_cols_name, train_size, traina_size, trainfull_size, traina_num, rand_seed_set, cols_base, data_path, save_path, target, method="linear", traina_half_name = None, data_name = None, fix_raw = None, metric="accuracy"):
   
    record = []
    print("The traina name is {}".format(traina_half_name))
      
    for rand_seed in rand_seed_set:
        if fix_raw:
            train_name = "train_{}_size{}_total{}_rand{}.csv".format(feature_cols_name, train_size, trainfull_size, fix_raw)
        else:
            train_name = "train_{}_size{}_total{}_rand{}.csv".format(feature_cols_name, train_size, trainfull_size, rand_seed)
        if traina_half_name:
            traina_name = traina_half_name + "{}.csv".format(rand_seed)
        else:
            traina_name = "train_generate_colsbase_rand{}.csv".format(rand_seed)
        if fix_raw:
            test_name = "test_{}_size{}_rand{}.csv".format(feature_cols_name, trainfull_size, fix_raw)
        else:
            test_name = "test_{}_size{}_rand{}.csv".format(feature_cols_name, trainfull_size, rand_seed)
            
        
        error = fit_model_chunk_spurious_diff(spurious_attr, train_name, traina_name, test_name, traina_num, traina_size, data_path, cols_base, target, method=method, model_random_seed = 1, data_name=None, metric = "accuracy")
        record.append(error)

        df_record = pd.DataFrame(record)
    
    col_mean = df_record.mean(axis=0)
    col_std = df_record.std(axis=0)
    record.append(col_mean)
    record.append(col_std)
    df_record_to_csv = pd.DataFrame(record)
    df_avg_min, df_avg_mean = get_min_average(df_record_to_csv, row_idx=len(rand_seed_set))
    for index, row in df_record_to_csv.iterrows():
        row_as_list = list(row.values)
        formatted_row = "[" + ", ".join(repr(e) for e in row_as_list) + "]"
        print(f"Row {index}: {formatted_row}")
    # print(df_record_to_csv)
    print("min: {}, mean: {}".format(df_avg_min, df_avg_mean))
    
    return col_mean, col_std


if __name__ == "__main__":
    
    control_ls = {0: "try", 1: "split", 2: "aug", 3: "diff", 4: "worst"}
    control = control_ls[0]
    
    dataset_name_ls = {0: "openmlDiabetes", 1: "heart_failure", 2: "gender"}
    spurious_attr_ls = {"openmlDiabetes": "skin", "heart_failure": "sex", "gender": "long_hair"}
    feature_cols_ls = {"openmlDiabetes": openmlDiabetes_feature_cols, "heart_failure": uci_heart_failure_feature_cols, "gender": gender_feature_cols}
    post_ls = {0: post_spurious_duplicate, 1: post_spurious, 2: post_spurious_focus}
    method_list = {0: "logistic", 1: "catBoost", 2: "randomForest"}
    
    # !!!
    current_method = method_list[2]
    dataset_name = dataset_name_ls[2]
    current_post = post_ls[2]
    minority_gate = -1
    
    info = info_dict[dataset_name]
    raw_size = info["raw_size"]
    file_size = info["file_size"]
    test_size = info["test_size"]
    cols_base = info["colsbase"]
    
    
    
    split_name_ls = {"openmlDiabetes": "diabetes", "gender": "gender", "heart_failure": "heart_failure", "gender": "gender"}
    associate_ls = {"openmlDiabetes": 1, "heart_failure": 1, "gender": 1}
    major_size_ls = {"openmlDiabetes": 45, "heart_failure": 20, "gender": 20}
    minor_size_ls = {"openmlDiabetes": 5, "heart_failure": 5, "gender": 5}
    
    current_parent_path = "../data/{}/{}_see_small_raw{}{}".format(dataset_name, dataset_name, raw_size, current_post) #!!!
    feature_cols_name = "colsbase"
    
    
    target = info["target"]
    spurious_attr = spurious_attr_ls[dataset_name]
    associate = associate_ls[dataset_name]
    major_size = major_size_ls[dataset_name]
    minor_size = minor_size_ls[dataset_name]
    feature_cols = feature_cols_ls[dataset_name]
    
    split_file = "/{}.csv".format(split_name_ls[dataset_name])
    
    rand_seed = 42
    
    if control == "try":
        split_file_abs_path = current_parent_path + split_file
        feature_target_correlation(split_file_abs_path, target)
        calculate_proportions(split_file_abs_path, target, spurious_attr)

    elif control == "split":
        rand_seed_set = [1, 2, 6, 8, 42]
        spurious_split_rand(feature_cols, feature_cols_name, current_parent_path, split_file, target, spurious_attr, associate, file_size, test_size, rand_seed_set, major_size, minor_size)
        
        
    elif control == "aug":
        minority_associate = abs(1-associate)
        rand_seed_set = [1, 2, 6, 8, 42]
        current_save_path = current_parent_path
        target = info["target"]
        
        feature_cols_name = "colsbase"
        train_size = info["raw_size"]
        traina_size = int(train_size/5)
        trainfull_size = info["total_size"]
        traina_num = 5
        batch_run_rand_seed_chunk_spurious(minority_associate, spurious_attr, feature_cols_name, train_size, traina_size, trainfull_size, traina_num, rand_seed_set, cols_base, current_parent_path, current_save_path, target, method=current_method, traina_half_name = None, data_name = None, fix_raw = None, metric="accuracy", minority_gate=minority_gate)

    elif control == "diff":
        target = info["target"]
        rand_seed_set = [1, 2, 6, 8, 42]
        feature_cols_name = "colsbase"
        train_size = info["raw_size"]
        traina_size = int(train_size/5)
        trainfull_size = info["total_size"]
        current_save_path = current_parent_path
        traina_num = 5
        batch_run_rand_seed_chunk_spurious_diff(spurious_attr, feature_cols_name, train_size, traina_size, trainfull_size, traina_num, rand_seed_set, cols_base, current_parent_path, current_save_path, target, method=current_method, traina_half_name = None, data_name = None, fix_raw = None, metric="accuracy")
        
        
    elif control == "worst":
        minority_associate = abs(1-associate)
        rand_seed_set = [1, 2, 6, 8, 42]
        current_save_path = current_parent_path
        target = info["target"]
        
        feature_cols_name = "colsbase"
        train_size = info["raw_size"]
        traina_size = int(train_size/5)
        trainfull_size = info["total_size"]
        traina_num = 5
        batch_run_rand_seed_chunk_spurious_worst(minority_associate, spurious_attr, feature_cols_name, train_size, traina_size, trainfull_size, traina_num, rand_seed_set, cols_base, current_parent_path, current_save_path, target, method=current_method, traina_half_name = None, data_name = None, fix_raw = None, metric="accuracy", minority_gate=minority_gate)
