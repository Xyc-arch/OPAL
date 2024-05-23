import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


from model import fit_model, fit_model_chunk
from meta_data import *
import statistics
            
            
def batch_run_rand_seed(feature_cols_name, train_size, traina_size, train_total_size, rand_seed_set, cols, add_num, data_path, save_path, target, method="logistic"):

    rate_mean = []
    rate_std = []
    
    record = []
    
    for rand_seed in rand_seed_set:
        files, train_datasets = get_meta(feature_cols_name, train_size, traina_size, train_total_size, rand_seed, add_num)
        acc = fit_model(files, train_datasets, data_path, cols, target, method)
        record.append(acc)

        
    df_record = pd.DataFrame(record)
    
    col_mean = df_record.mean(axis=0)
    col_std = df_record.std(axis=0)
    record.append(col_mean)
    record.append(col_std)
    df_record_to_csv = pd.DataFrame(record)
    print(df_record_to_csv)
    df_record_to_csv.to_csv(save_path + 
                     "/perform_train_{}_size{}_total{}_addnum{}_{}.csv".format(feature_cols_name, train_size, train_total_size, add_num, method))
    
    plt.figure("mean")
    x_axis =[x for x in range(add_num+2)]
    plt.errorbar(x_axis, col_mean, yerr=col_std, fmt='-o', capsize=5)
    plt.axhline(y = col_mean[0], color = 'r', linestyle = '-') 
    plt.xlabel('add num')
    plt.ylabel('accuracy')
    save_name = "/{}_train_{}_size{}_total{}_addnum{}_{}".format("mean", feature_cols_name, train_size, train_total_size, add_num, method)
    plt.title("{} on {}".format(method, target))
    plt.savefig(save_path + save_name + '.png')
    
    
    plt.figure("std")
    x_axis =[x for x in range(add_num+2)]
    plt.bar(x_axis, col_std)
    plt.xlabel('add num')
    plt.ylabel('std')
    save_name = "/{}_train_{}_size{}_total{}_addnum{}_{}".format("std", feature_cols_name, train_size, train_total_size, add_num, method)
    plt.title("{} on {}".format(method, target))
    plt.savefig(save_path + save_name + '.png')
        
        
    return col_mean, col_std


def get_max_average(df, row_idx = 5):
    row = df.iloc[row_idx]
    row_max = max(row)
    row_avg = statistics.mean(row)
    
    return row_max, row_avg

def get_min_average(df, row_idx = 5):
    row = df.iloc[row_idx]
    row_min = min(row)
    row_avg = statistics.mean(row)
    
    return row_min, row_avg

def batch_run_rand_seed_chunk(feature_cols_name, train_size, traina_size, trainfull_size, traina_num, rand_seed_set, cols_base, data_path, save_path, target, method="linear", traina_half_name = None, data_name = None, fix_raw = None, metric="accuracy", current_post = None):
   
    record = []
    
      
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
            
        
        error = fit_model_chunk(train_name, traina_name, test_name, traina_num, traina_size, data_path, cols_base, target, method, data_name=data_name, metric=metric)
        record.append(error)

        
    df_record = pd.DataFrame(record)
    
    col_mean = df_record.mean(axis=0)
    col_std = df_record.std(axis=0)
    record.append(col_mean)
    record.append(col_std)
    df_record_to_csv = pd.DataFrame(record)
    df_avg_min, df_avg_mean = get_min_average(df_record_to_csv, row_idx=len(rand_seed_set))
    Mean = None
    Std = None
    for index, row in df_record_to_csv.iterrows():
        row_as_list = list(row.values)
        formatted_row = "[" + ", ".join(repr(e) for e in row_as_list) + "]"
        if index <= 4:
            print(f"Row {index}: {formatted_row}")
        elif index == 5:
            print(f"Mean: {formatted_row}")
            Mean = formatted_row
        elif index == 6:
            print(f"Std: {formatted_row}")
            Std = formatted_row
    # print(df_record_to_csv)
    print("min: {}, mean: {}".format(df_avg_min, df_avg_mean))
    
    with open(f"{save_path}/{current_post}_{method}_{data_name}.txt", 'w') as f:
        f.write(f"Mean: {Mean}\n")
        f.write(f"Std: {Std}\n")
        
        
    return col_mean, col_std
            

    
   