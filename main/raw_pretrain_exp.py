
from experiment import *
from meta_data import *
import argparse

if __name__ == "__main__":
    
    control_ls = {0: "shap batch", 1: "shap", 2: "general"}
    control = control_ls[2]
    
    post_bootstrap = "_bootstrap"
    post = ""
    post_mixup = "_mixup"
    post_shap = "_shap"
    
    # !!!
    traina_half_name_ls = ["train_real_colsbase_rand", None, "select_randomForest_rand"]
    # traina_half_name_ls = ["train_real_colsbase_rand"]
    
    dataset_name_ls = {1: "openmlDiabetes", 2: "heart_failure", 3: "gender"}
    
    data_name = dataset_name_ls[3] # !!!
    
    
    ''' general '''

    info = info_dict[data_name]
    raw_size = info["raw_size"]
    shap_raw = info["shap_raw"]
    post_bootstrap = "_bootstrap"
    post = ""
    post_mixup = "_mixup"
    current_data_path = "../data/{}/{}_see_small_raw{}{}".format(data_name, data_name, shap_raw, post)
    current_save_path = current_data_path
    target = info["target"]
    
    feature_cols_name = "colsbase"
    train_size = info["shap_raw"] # !!!
    traina_size = int(train_size/5)
    trainfull_size = info["total_size"]
    traina_num = 5
      
    rand_seed_set = [1,2,6,8,42]  # !!!
    # rand_seed_set = [1]
    
    cols_base = info["colsbase"]
    # 1: logistic, 2: random forest, 5: catboost
    method = method_list[2]
    # traina_half_name = "select_randomForest_rand"
    traina_half_name = "train_real_colsbase_rand"
    # traina_half_name = None
    fix_raw = 1
    # data_name = None
    
    if control == "shap batch":
        current_data_path = "../data/{}/{}_see_small_raw{}{}".format(data_name, data_name, shap_raw, post_shap)
        current_save_path = current_data_path
        for traina_half_name in traina_half_name_ls:
            print("-"*50)
            cols_mean, cols_std = batch_run_rand_seed_chunk(feature_cols_name, train_size, traina_size, 
                                                            trainfull_size, traina_num, rand_seed_set, cols_base, 
                                                            current_data_path, current_save_path, target, method, traina_half_name, data_name=data_name, fix_raw=fix_raw)
    elif control == "shap":
        cols_mean, cols_std = batch_run_rand_seed_chunk(feature_cols_name, train_size, traina_size, 
                                                            trainfull_size, traina_num, rand_seed_set, cols_base, 
                                                            current_data_path, current_save_path, target, method, traina_half_name, data_name=data_name)

    
    elif control == "general":
        

        
        dataset_name_ls = {1: "openmlDiabetes", 2: "heart_failure", 3: "gender", 4: "craft"}
        data_name = dataset_name_ls[4]
        all_post = {0: post, 1: post_origin, 2: post_mixup, 3: post_gmm, 4: post_gan, 5: post_bootstrap, 
                    6: post_imbalanced, 7: post_imbalanced_duplicate, 8: post_imbalanced_smote, 9: post_imbalanced_focus}
        current_post = all_post[7]
        method_list_class = {0: "logistic", 1: "catBoost", 2: "randomForest"}
        method = method_list_class[2]
        
        
        parser = argparse.ArgumentParser(description='Input for hyperpara.')
        parser.add_argument('--current_post', type=str, required=True, choices=["post_spurious_focus", "post_imbalanced_focus", "post_spurious_duplicate", "post_imbalanced_duplicate", "post_spurious_smote", "post_imbalanced_smote"], help='The current post variable name (e.g., "post_spurious_focus" or "post_imbalanced_focus")')
        parser.add_argument('--data_name', type=str, required=True, choices=["openmlDiabetes", "gender", "heart_failure", "craft"], help='The name of the dataset (e.g., "openmlDiabetes")')
        parser.add_argument('--classifier', type=str, required=True, choices=["logistic", "catBoost", "randomForest"], help='The classifier you want to evaluate on.')
        
        args = parser.parse_args()
        
        current_post = eval(args.current_post)
        data_name = args.data_name
        method = args.classifier
                    
        info = info_dict[data_name]
        raw_size = info["raw_size"]
        trainfull_size = info["total_size"]

        print("-"*30)
        print(data_name)
        print(current_post)
        print(method)

        current_data_path = "./data/{}/{}_see_small_raw{}{}".format(data_name, data_name, raw_size, current_post)
        current_save_path = current_data_path
        target = info["target"]
        
        feature_cols_name = "colsbase"
        traina_size = int(raw_size/5)
        traina_num = 5
        
        rand_seed_set = [1, 2, 6, 8, 42]
        cols_base = info[feature_cols_name][:-1] 
        traina_half_name = None
        fix_raw = None
        metric = "accuracy"
        
        cols_mean, cols_std = batch_run_rand_seed_chunk(feature_cols_name, raw_size, traina_size, 
                                                                trainfull_size, traina_num, rand_seed_set, cols_base, 
                                                                current_data_path, current_save_path, target, method, traina_half_name, data_name=data_name, metric=metric)