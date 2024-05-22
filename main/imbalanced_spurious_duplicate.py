from meta_data import *
import pandas as pd
from sklearn.utils import resample

def duplicate_synthetic_data(raw_data_path, generate_size, save_path, start_row=1, end_row=10):
    raw_data = pd.read_csv(raw_data_path)
    adjusted_start_row = start_row - 1  
    adjusted_end_row = end_row  
    data_to_resample = raw_data.iloc[adjusted_start_row:adjusted_end_row]
    synthetic_data = resample(data_to_resample, replace=True, n_samples=generate_size, random_state=0)
    synthetic_data.to_csv(save_path, index=False)

    return save_path



def duplicate_synthetic_rand(parent_path, rand_set, raw_size, generate_size, total_size, start_row=1, end_row=10):
    for rand in rand_set:
        save_path = parent_path + "/train_generate_colsbase_rand{}.csv".format(rand)
        raw_data_path = parent_path + "/train_colsbase_size{}_total{}_rand{}.csv".format(raw_size, total_size, rand)

        duplicate_synthetic_data(raw_data_path, generate_size, save_path, start_row, end_row)


if __name__ == "__main__":
    control_ls = {0: "imbalanced", 1: "spurious"}
    data_name_ls = {0: "openmlDiabetes", 1: "heart_failure", 2: "gender", 3: "craft"}
    
    
    control = control_ls[1]
    data_name = data_name_ls[2]
    
    if control == "imbalanced":
        info = info_dict[data_name]
        total_size = info["total_size"]
        target_column = info["target"]
        task_type = info["task_type"]
        raw_size = info["raw_size"]
        parent_path = '../data/{}/{}_see_small_raw{}{}'.format(data_name, data_name, raw_size, post_imbalanced_duplicate)
        
        rand_set = [1, 2, 6, 8, 42]
        
        duplicate_synthetic_rand(parent_path, rand_set, raw_size, raw_size, total_size, 1, end_row=10)
        
    elif control == "spurious":
        info = info_dict[data_name]
        total_size = info["total_size"]
        target_column = info["target"]
        task_type = info["task_type"]
        raw_size = info["raw_size"]
        parent_path = '../data/{}/{}_see_small_raw{}{}'.format(data_name, data_name, raw_size, post_spurious_duplicate)
        
        rand_set = [1, 2, 6, 8, 42]
        
        duplicate_synthetic_rand(parent_path, rand_set, raw_size, raw_size, total_size, 1, end_row=20)