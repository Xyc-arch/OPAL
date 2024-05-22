
from experiment import *
from meta_data import *

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
        
        ''' heart failure '''
        # post_bootstrap = "_bootstrap"
        # post_mixup = "_mixup"
        # post = ""
        # current_data_path = "../data/heart_failure/heart_failure_see_small_raw50{}".format(post_mixup)
        # current_save_path = current_data_path
        # target = 'DEATH_EVENT'
        
        # feature_cols_name = "colsbase"
        # train_size = 50
        # traina_size = 10
        # trainfull_size = 199
        # traina_num = 5
        
        # # rand_seed_set = [1,2,6,8,42] 
        # rand_seed_set = [1, 2, 6, 8, 42]
        # cols_base = uci_heart_failure_feature_cols[feature_cols_name][:-1]
        # # 1: logistic, 2: random forest, 5: catboost
        # method = method_list[1]
        # print(method)
        # cols_mean, cols_std = batch_run_rand_seed_chunk(feature_cols_name, train_size, traina_size, 
        #                                                     trainfull_size, traina_num, rand_seed_set, cols_base, 
        #                                                     current_data_path, current_save_path, target, method)



        ''' openmlDiabetes '''
        # data_name = "openmlDiabetes"
        # print(data_name)
        # # data_name = None
        # post_bootstrap = "_bootstrap"
        # post = ""
        # post_mixup = "_mixup"
        # current_data_path = "../data/openmlDiabetes/openmlDiabetes_see_small_raw100{}".format(post_imbalanced)
        # current_save_path = current_data_path
        # target = "class"
        
        # feature_cols_name = "colsbase"
        # train_size = 100
        # traina_size = 20
        # trainfull_size = 568
        # traina_num = 5
        
        # rand_seed_set = [1,2,6,8,42] 
        # # rand_seed_set = [1, 2, 6, 42]
        # cols_base = openmlDiabetes_feature_cols[feature_cols_name][:-1]
        # # 1: logistic, 2: random forest, 5: catboocurrent_postst
        # method = method_list[2]
        # # traina_half_name = "select_randomForest_rand"
        # # traina_half_name = "train_real_colsbase_rand"
        # traina_half_name = None
        # # fix_raw = 1
        # fix_raw = None
        # batch = False
        # if batch:
        #     for traina_half_name in traina_half_name_ls:
        #         print("-"*50)
        #         cols_mean, cols_std = batch_run_rand_seed_chunk(feature_cols_name, train_size, traina_size, 
        #                                                         trainfull_size, traina_num, rand_seed_set, cols_base, 
        #                                                         current_data_path, current_save_path, target, method, traina_half_name, data_name=data_name, fix_raw=fix_raw)
        # else:
        #     cols_mean, cols_std = batch_run_rand_seed_chunk(feature_cols_name, train_size, traina_size, 
        #                                                         trainfull_size, traina_num, rand_seed_set, cols_base, 
        #                                                         current_data_path, current_save_path, target, method, traina_half_name, data_name=data_name)


        ''' uci breast cancer '''
        
        # current_data_path = "../data/uci_breast_cancer/uci_breast_cancer_see_small_raw100"
        # current_save_path = current_data_path
        # target = 'Class'
        
        # feature_cols_name = "colsbase"
        # train_size = 50
        # traina_size = 20
        # trainfull_size = 186
        # traina_num = 5
        
        # # rand_seed_set = [1,2,6,8,42] 
        # rand_seed_set = [1]
        # cols_base = uci_breast_cancer_1_feature_cols[feature_cols_name][:-1]
        # # 1: logistic, 2: random forest, 5: catboost
        # method = method_list[5]
        # print(method)
        # cols_mean, cols_std = batch_run_rand_seed_chunk(feature_cols_name, train_size, traina_size, 
        #                                                     trainfull_size, traina_num, rand_seed_set, cols_base, 
        #                                                     current_data_path, current_save_path, target, method)
        
        
        ''' gender '''
        # post_bootstrap = "_bootstrap"
        # post = ""
        # post_mixup = "_mixup"
        # current_data_path = "../data/gender/gender_see_small_raw50{}".format(post)
        # current_save_path = current_data_path
        # target = 'gender'
        
        # feature_cols_name = "colsbase"
        # train_size = 50
        # traina_size = 10
        # trainfull_size = 4001
        # traina_num = 5
        
        # rand_seed_set = [1,2,6,8,42] 
        # # rand_seed_set = [1]
        # cols_base = gender_feature_cols[feature_cols_name][:-1]
        # # 1: logistic, 2: random forest, 5: catboost
        # method = method_list[2]
        # print(method)
        # metric = "accuracy"
        
        # cols_mean, cols_std = batch_run_rand_seed_chunk(feature_cols_name, train_size, traina_size, 
        #                                                     trainfull_size, traina_num, rand_seed_set, cols_base, 
        #                                                     current_data_path, current_save_path, target, method, metric=metric)
        
        
        
        dataset_name_ls = {1: "openmlDiabetes", 2: "heart_failure", 3: "gender", 4: "craft"}
        data_name = dataset_name_ls[4]
        all_post = {0: post, 1: post_origin, 2: post_mixup, 3: post_gmm, 4: post_gan, 5: post_bootstrap, 
                    6: post_imbalanced, 7: post_imbalanced_duplicate, 8: post_imbalanced_smote, 9: post_imbalanced_focus}
        current_post = all_post[7]
        method_list_class = {0: "logistic", 1: "catBoost", 2: "randomForest"}
        method = method_list_class[2]
        
        
        # for data_name in ["openmlDiabetes", "heart_failure", "gender"]:
        #     for method in method_list_class.values():
        #         for current_post in all_post.values():
        #             print(50*"-")
        #             print(data_name)
        #             print(current_post)
        #             print(method)
                    
        info = info_dict[data_name]
        # raw_size = info["raw_size"]
        raw_size = 100
        trainfull_size = info["total_size"]

        print("-"*30)
        print(data_name)
        print(current_post)
        print(method)

        current_data_path = "../data/{}/{}_see_small_raw{}{}".format(data_name, data_name, raw_size, current_post)
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