from split import *
from meta_data import *



if __name__ == "__main__":
    
    
    dataset_name_ls = {0: "openmlDiabetes", 1: "gender", 2: "heart_failure", 3: "craft"}
    
    dataset_name = dataset_name_ls[3]
    info = info_dict[dataset_name]
    raw_size = info["raw_size"]
    current_post = post_imbalanced_focus
    
    
    split_name = {"openmlDiabetes": "diabetes", "gender": "gender", "heart_failure": "heart_failure", "craft": "craft"}
    raw_minority_rate_ls = {"openmlDiabetes": 0.1, "gender": 0.2, "heart_failure": 0.2, "craft": 0.1}
    
    current_parent_path = "../data/{}/{}_see_small_raw{}{}".format(dataset_name, dataset_name, raw_size, current_post)
    feature_cols_name = "colsbase"
    
    
    rand_seed_set = [1,2,6,8,42] 
    split_file = '/{}.csv'.format(split_name[dataset_name])
    file_size = info["file_size"]
    traina_size = int(raw_size/5)
    test_size = info["test_size"]
    target = info["target"]
    minority = 0
    gate_traina = False
    
    raw_minority_rate = raw_minority_rate_ls[dataset_name]
    imbalanced_split_rand(craft_feature_cols, feature_cols_name, raw_size, test_size, rand_seed_set, current_parent_path, split_file, file_size, gate_traina, raw_minority_rate, minority, target)
    