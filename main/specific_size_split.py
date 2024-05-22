

from split import *
from meta_data import *


if __name__ == "__main__":
    
    post = ""
    post_bootstrap = "_bootstrap"
    post_mixup = "_mixup"
    post_shap = "_shap"
    
    ''' general '''
    data_name_ls = {0: "abalone", 1: "california", 2: "wine", 3: "openmlDiabetes", 4: "gender", 5: "heart_failure", 6: "craft"}
    split_file_ls = {"abalone": "abalone", "california": "california_raw", "wine": "wine", "openmlDiabetes": "diabetes", "gender": "gender", "heart_failure": "heart_failure", "craft": "craft"}
    data_name = data_name_ls[4]
    info = info_dict[data_name]
    
    shap_raw = info["shap_raw"]
    file_size = info["file_size"]
    raw_size = info["raw_size"]
    # raw_size = 50
    shap_raw = info["shap_raw"]
    current_post = post_spurious_duplicate
    current_parent_path = "../data/{}/{}_see_small_raw{}{}".format(data_name, data_name, raw_size, current_post)
    feature_cols_name = "colsbase"
    feature_cols = info["feature_cols"]

    rand_seed_set = [1,2,6,8,42] 
    split_file = '/{}.csv'.format(split_file_ls[data_name])
    traina_size = raw_size
    test_size = info["test_size"]
    traina_num = 1
    specific_size_split_rand(feature_cols, feature_cols_name, raw_size, traina_size, traina_num, test_size, rand_seed_set, current_parent_path, split_file, file_size, False)
    
    
