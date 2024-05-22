

from experiment import *

if __name__ == "__main__":
    current_data_path = "../data/heart_data_v3/heart_data_v3_see_small"
    current_save_path = current_data_path
    target = 'Heart Attack Risk'
    
    feature_cols_name = "cols2"
    train_size = 100
    traina_size = 100
    train_total_size = 7763
    
    rand_seed_set = [1,2,42,6,8,64,72,88,92,96]  
    cols = heart_feature_cols[feature_cols_name][:-1]
    method = method_list[2]
    
    cols_mean, cols_std = batch_run_rand_seed(feature_cols_name, train_size, traina_size, train_total_size, rand_seed_set, cols,
                        add_num=8, data_path=current_data_path, save_path = current_save_path, target=target, method=method)
    
    