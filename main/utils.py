import pandas as pd
from meta_data import *


def random_generate_from_all(data_name_ls, rand_set, raw_size_set, add_size_set, fix_sample_seed=None, all_half_name = None, save_half_name = None):
    for data_name in data_name_ls:
        add_num = add_size_set[data_name]
        raw_num = raw_size_set[data_name]
        parent_path = '../data/{}/{}_see_small_raw{}_shap'.format(data_name, data_name, raw_num)
        # generate_all_name = "/train_generate_colsbase_randall.csv"
        # generate_all_path = parent_path + generate_all_name
        # df_all = pd.read_csv(generate_all_path)
        
        if len(rand_set) <= 1:
            sample_seed_ls = [1, 2, 6, 8, 42]
            for sample_seed in sample_seed_ls:
                generate_all_name = "/" + all_half_name + ".csv"
                generate_rand_name = "/" + save_half_name + "_rand{}.csv".format(sample_seed)
                generate_all_path = parent_path + generate_all_name
                df_all = pd.read_csv(generate_all_path)
                
                generate_rand_path = parent_path + generate_rand_name
                df_sampled = df_all.sample(n=add_num, random_state=sample_seed)
                df_sampled.to_csv(generate_rand_path, index=False)
                
                print(generate_rand_path)
                
        else:
            for rand in rand_set:
                generate_all_name = "/" + all_half_name + "_rand{}.csv".format(rand)
                generate_rand_name = "/" + save_half_name + "_rand{}.csv".format(rand)
                generate_all_path = parent_path + generate_all_name
                df_all = pd.read_csv(generate_all_path)

                generate_rand_path = parent_path + generate_rand_name
                df_sampled = df_all.sample(n=add_num, random_state=fix_sample_seed)
                df_sampled.to_csv(generate_rand_path, index=False)
                
                print(generate_rand_path)



if __name__ == "__main__":
    
    
    data_type_ls = {0: "real", 1: "syn", 2: "presyn", 3: "real50", 4: "syn50", 
                    5: "presyn50", 6: "trial", 7:"imbalanced_syn_reorder", 8:"spurious_reorder"}
    data_type = data_type_ls[8]
    fix_seed = 42
    rm = False
    
    
    if data_type == "real":
        # data_name = "abalone"
        # data_name = "california"
        # data_name = "openmlDiabetes"
        sample_seed_ls = [1, 2, 6, 8, 42]
        data_name_ls = ["california", "abalone", "openmlDiabetes", "wine", "gender"]
        for data_name in data_name_ls:
            for sample_seed in sample_seed_ls:
                info = info_dict[data_name]
                
                total_size = info["total_size"]
                all_half_name = "trainfull_colsbase_size{}_rand1".format(total_size)
                save_half_name = "train_real_colsbase"
                data_name_ls = [data_name]
                rand_set = [1]
                raw_size_set = {data_name: 100}
                add_size_set = {data_name: 200}
                random_generate_from_all(data_name_ls, rand_set, raw_size_set, add_size_set, fix_seed, all_half_name, save_half_name)
                
            
    elif data_type == "syn":
        all_half_name = None
        save_half_name = None
    
        # data_name_ls = ["openmlDiabetes"]
        data_name_ls = ["california", "abalone", "openmlDiabetes", "wine", "gender"]
        rand_set = [1, 2, 6, 8, 42]
        # raw_size_set = {"heart_failure": 50, "abalone": 100, "openmlDiabetes": 100, "california": 100, "wine": 50}
        # add_size_set = {"heart_failure": 50, "abalone": 100, "openmlDiabetes": 200, "california": 200, "wine": 50}
        
        for data_name in data_name_ls:
            info = info_dict[data_name]
            total_size = info["total_size"]
            all_half_name = "train_generate_colsbase_all"
            save_half_name = "train_generate_colsbase"
            data_name_ls = [data_name]
            raw_size_set = {data_name: 100}
            add_size_set = {data_name: 500}
            random_generate_from_all(data_name_ls, rand_set, raw_size_set, add_size_set, fix_seed, all_half_name, save_half_name)
        
        
    elif data_type == "presyn":
        sample_seed_ls = [1, 2, 6, 8, 42]
        data_name_ls = ["california", "abalone", "openmlDiabetes", "wine", "gender"]
        for data_name in data_name_ls:
            for sample_seed in sample_seed_ls:
                info = info_dict[data_name]
                
                total_size = info["total_size"]
                all_half_name = "train_generate_colsbase_all"
                save_half_name = "train_generate_colsbase_all"
                data_name_ls = [data_name]
                rand_set = [1]
                raw_size_set = {data_name: 100}
                add_size_set = {data_name: 500}
                random_generate_from_all(data_name_ls, rand_set, raw_size_set, add_size_set, fix_seed, all_half_name, save_half_name)
                
                
    elif data_type == "real50":
        sample_seed_ls = [1, 2, 6, 8, 42]
        data_name_ls = ["heart_failure"]
        for data_name in data_name_ls:
            for sample_seed in sample_seed_ls:
                info = info_dict[data_name]
                
                total_size = info["total_size"]
                all_half_name = "trainfull_colsbase_size{}_rand1".format(total_size)
                save_half_name = "train_real_colsbase"
                data_name_ls = [data_name]
                rand_set = [1]
                raw_size_set = {data_name: 50}
                add_size_set = {data_name: 100}
                random_generate_from_all(data_name_ls, rand_set, raw_size_set, add_size_set, fix_seed, all_half_name, save_half_name)
                
                
    elif data_type == "syn50":
        all_half_name = None
        save_half_name = None
    
        data_name_ls = ["heart_failure"]
        rand_set = [1, 2, 6, 8, 42]
        
        for data_name in data_name_ls:
            info = info_dict[data_name]
            total_size = info["total_size"]
            all_half_name = "train_generate_colsbase_all"
            save_half_name = "train_generate_colsbase"
            data_name_ls = [data_name]
            raw_size_set = {data_name: 50}
            add_size_set = {data_name: 100}
            random_generate_from_all(data_name_ls, rand_set, raw_size_set, add_size_set, fix_seed, all_half_name, save_half_name)
            
            
    elif data_type == "presyn50":
        sample_seed_ls = [1, 2, 6, 8, 42]
        data_name_ls = ["heart_failure"]
        for data_name in data_name_ls:
            for sample_seed in sample_seed_ls:
                info = info_dict[data_name]
                
                total_size = info["total_size"]
                all_half_name = "train_generate_colsbase_all"
                save_half_name = "train_generate_colsbase_all"
                data_name_ls = [data_name]
                rand_set = [1]
                raw_size_set = {data_name: 50}
                add_size_set = {data_name: 100}
                random_generate_from_all(data_name_ls, rand_set, raw_size_set, add_size_set, fix_seed, all_half_name, save_half_name)
                
                
    elif data_type == "trial":
        data_name = "openmlDiabetes"
        rand_set = [1, 2, 6, 8, 42]
        info = info_dict[data_name]
        total_size = info["total_size"]
        all_half_name = "train_generate_colsbase_all"
        save_half_name = "train_generate_colsbase"
        data_name_ls = [data_name]
        raw_size_set = {data_name: 100}
        add_size_set = {data_name: 200}
        
        fix_seed = 42
        random_generate_from_all(data_name_ls, rand_set, raw_size_set, add_size_set, fix_seed, all_half_name, save_half_name)
    
    
    elif data_type == "imbalanced_syn_reorder":
        data_name_ls = {0: "openmlDiabetes", 1: "heart_failure", 2: "gender", 4: "craft"}
        data_name = data_name_ls[4]
        info = info_dict[data_name]
        raw_size = info["raw_size"]
        total_size = info["total_size"]
        parent_path = '../data/{}/{}_see_small_raw{}{}'.format(data_name, data_name, raw_size, post_imbalanced_focus)
        rand_ls = [1, 2, 6, 8, 42]
        target = info["target"]
        minority = 0
        
        for rand in rand_ls:
            name = "/train_generate_colsbase_rand{}.csv".format(rand)
            data_path = parent_path + name
            df = pd.read_csv(data_path)
            
            df['minority_first'] = (df[target] == minority)
            df.sort_values(by='minority_first', ascending=False, inplace=True)
            df.drop(columns=['minority_first'], inplace=True)
            
            df.to_csv(data_path, index=False)
            
    
    elif data_type == "spurious_reorder":
        data_name_ls = {0: "openmlDiabetes", 1: "heart_failure", 2: "gender"}
        data_name = data_name_ls[2]
        info = info_dict[data_name]
        raw_size = info["raw_size"]
        total_size = info["total_size"]
        parent_path = '../data/{}/{}_see_small_raw{}{}'.format(data_name, data_name, raw_size, post_spurious_focus)
        rand_ls = [1, 2, 6, 8, 42]
        target = info["target"]
        associate_ls = {"openmlDiabetes": 1, "heart_failure": 1, "gender": 1}
        associate = associate_ls[data_name]
        minority_associate = abs(associate - 1)
        
        
        spurious_attr_ls = {"openmlDiabetes": "skin", "heart_failure": "sex", "gender": "long_hair"}
        feature_cols_ls = {"openmlDiabetes": openmlDiabetes_feature_cols, "heart_failure": uci_heart_failure_feature_cols, "gender": gender_feature_cols}
        
        spurious_attr = spurious_attr_ls[data_name]
        
        for rand in rand_ls:
            name = "/train_generate_colsbase_rand{}.csv".format(rand)
            data_path = parent_path + name
            df = pd.read_csv(data_path)
            df['spurious_cat'] = df[spurious_attr].apply(lambda x: '0' if x == 0 else 'positive')
            
            groups = df.groupby(['spurious_cat', target])
            
            group11 = groups.get_group(('positive', 1))
            group00 = groups.get_group(('0', 0))
            group01 = groups.get_group(('0', 1))
            group10 = groups.get_group(('positive', 0))
            
            
            if minority_associate == 1:
                first_two_groups = pd.concat([group11, group00]).sample(frac=1).reset_index(drop=True)
                last_two_groups = pd.concat([group10, group01]).sample(frac=1).reset_index(drop=True)
                df_reordered = pd.concat([first_two_groups, last_two_groups])
            elif minority_associate == 0:
                first_two_groups = pd.concat([group10, group01]).sample(frac=1).reset_index(drop=True)
                last_two_groups = pd.concat([group11, group00]).sample(frac=1).reset_index(drop=True)
                df_reordered = pd.concat([first_two_groups, last_two_groups])
                
            df_reordered.drop(columns=['spurious_cat'], inplace=True)

            df_reordered.to_csv(data_path, index=False)
