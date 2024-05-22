import pandas as pd
import numpy as np
from meta_data import * 



np.random.seed(1)

    
def vary_split(feature_cols, feature_cols_name, traina_size, train_total_size, rand_seed, parent_path, split_file):
    
    
    
    cols = feature_cols[feature_cols_name]
    df = pd.read_csv(parent_path + split_file)
    
    df = df[cols]
    
    # Take first 7753 rows as train
    train = df.sample(n=train_total_size, replace=False, random_state=rand_seed) 

    # Take the rest as test
    # test = df.iloc[7763:]  
    
    
    test = pd.concat([df, train]).drop_duplicates(keep=False)

    # train a

    train_subset_path = parent_path + "/train_{}_size{}_total{}_rand{}.csv".format(feature_cols_name, traina_size, train_total_size, rand_seed)
    train_subset = train.sample(n=traina_size, replace=False, random_state=rand_seed) 
    train_subset.to_csv(train_subset_path, index=False)

    # subset.to_csv('/train2_100.csv', index=False)

    train.to_csv(parent_path + "/trainfull_{}_size{}_rand{}.csv".format(feature_cols_name, train_total_size, rand_seed), index=False)
    test.to_csv(parent_path + "/test_{}_size{}_rand{}.csv".format(feature_cols_name, train_total_size, rand_seed), index=False)



# vary_split(traina_size=50, rand_seed=64)

def batch_split(feature_cols, cols_name_set, traina_size_set, train_total_size_set, rand_seed_set, parent_path, split_file):
    for feature_cols_name in cols_name_set:
        for traina_size in traina_size_set:
            for train_total_size in train_total_size_set:
                for rand_seed in rand_seed_set:
                    vary_split(feature_cols, feature_cols_name, traina_size, train_total_size, rand_seed, parent_path, split_file)



def raw_pretrain_split(feature_cols, feature_cols_name, prop_list, rand_seed, parent_path, split_file, file_size, gate_traina):
    cols = feature_cols[feature_cols_name]
    df = pd.read_csv(parent_path + split_file)
    part_list = []
    size_list = []
    
    df = df[cols]
    remain = df
    for idx in range(len(prop_list)-1):
        prop = prop_list[idx]
        current_size = int(file_size*prop)
        part_tmp = remain.sample(n=current_size, replace=False, random_state=rand_seed)
        remain = pd.concat([remain, part_tmp]).drop_duplicates(keep=False)
        part_list.append(part_tmp)
        size_list.append(current_size)
    
    part_list.append(remain)
    size_list.append(file_size - sum([size for size in size_list]))
    
    train_full_size = size_list[0] + sum([size_list[i] for i in range(2, len(part_list))])
    
    raw_path = parent_path + "/train_{}_size{}_total{}_rand{}.csv".format(feature_cols_name, size_list[0], train_full_size, rand_seed)
    raw = part_list[0]
    raw.to_csv(raw_path, index=False)
    
    train_full = pd.concat([part_list[i] for i in range(2, len(part_list))])
    train_full = pd.concat([raw, train_full])
    train_full.to_csv(parent_path + "/trainfull_{}_size{}_rand{}.csv".format(feature_cols_name, train_full_size, rand_seed), index=False)
    
    if gate_traina:
        train_full.to_csv(parent_path + "/train_generate_colsbase_rand{}.csv".format(rand_seed), index=False)
        # for i in range(2, len(part_list)):
        #     add_size = size_list[i-1] if i == len(part_list)-1 else size_list[i]
        #     part_list[i].to_csv(parent_path + '/train{}{}_on{}_{}_train{}_rand{}.csv'.format(add_size, alpha_set[i], size_list[0], 
        #                                                             feature_cols_name, train_full_size, rand_seed), index=False)
    
    
    test = part_list[1]
    test.to_csv(parent_path + "/test_{}_size{}_rand{}.csv".format(feature_cols_name, train_full_size, rand_seed), index=False)
    
    alpha_set = {2:"a", 3: "b", 4: "c", 5: "d", 6: "e", 7: "f", 8: "g", 9: "h", 10: "i", 11: "j", 12: "k", 13: "l", 14: "m", 15: "n", 16: "o", 17: "p"}
        
        
    print("-"*20)
    print("Dataset: {}".format(split_file))
    print("cols name: {}".format(feature_cols_name))
    print("raw size: {}".format(size_list[0]))
    print("test size: {}".format(size_list[1]))
    print("add size: {}".format(size_list[-2]))
    print("full/total size: {}".format(train_full_size))
    
    
def raw_pretrain_split_cal(raw_prop, test_prop, train_add_num=4):
    remain_prop = 1 - raw_prop - test_prop
    train_add_size = remain_prop/train_add_num
    prop_list = []
    prop_list.append(raw_prop)
    prop_list.append(test_prop)
    for i in range(train_add_num):
        prop_list.append(train_add_size)
    print(prop_list)
    return prop_list



def raw_pretrain_split_rand(feature_cols, feature_cols_name, prop_list, rand_seed_set, parent_path, split_file, file_size, gate_traina):
    for seed in rand_seed_set:
        raw_pretrain_split(feature_cols, feature_cols_name, prop_list, seed, parent_path, split_file, file_size, gate_traina)
        
        
def sample_remain_no_replace(remain, sample_num, rand_seed):
    target_sample = remain.sample(n=sample_num, replace=False, random_state=rand_seed)
    remain = pd.concat([remain, target_sample]).drop_duplicates(keep=False)
    return target_sample, remain

def specific_size_split(feature_cols, feature_cols_name, raw_size, traina_size, traina_num, test_size, rand_seed, parent_path, split_file, file_size, gate_traina):
    cols = feature_cols[feature_cols_name]
    df = pd.read_csv(parent_path + split_file)
    
    df = df[cols]
    remain = df
    
    # split test
    train_full_size = file_size - test_size
    test, remain = sample_remain_no_replace(remain, test_size, rand_seed)
    test.to_csv(parent_path + "/test_{}_size{}_rand{}.csv".format(feature_cols_name, train_full_size, rand_seed), index=False)


    # split initial train (raw) and save
    raw, remain = sample_remain_no_replace(remain, raw_size, rand_seed)
    raw.to_csv(parent_path + "/train_{}_size{}_total{}_rand{}.csv".format(feature_cols_name, raw_size, train_full_size, rand_seed), index=False)
    
    train_full = remain
    train_full.to_csv(parent_path + "/trainfull_{}_size{}_rand{}.csv".format(feature_cols_name, train_full_size, rand_seed), index=False)

    alpha_set = {0:"a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j", 10: "k", 11: "l", 12: "m", 13: "n", 14: "o", 15: "p"}
    
    if gate_traina:
        # for n in range(traina_num):
        #     traina_tmp, remain = sample_remain_no_replace(remain, traina_size, rand_seed)
        #     traina_tmp.to_csv(parent_path + '/train{}{}_on{}_{}_train{}_rand{}.csv'.format(raw_size, alpha_set[n], traina_size, 
        #                                                             feature_cols_name, train_full_size, rand_seed), index=False)
        
        remain.to_csv(parent_path + "/train_generate_colsbase_rand{}.csv".format(rand_seed), index=False)
            
    print("-"*20)
    print("Dataset: {}".format(split_file))
    print("cols name: {}".format(feature_cols_name))
    print("raw size: {}".format(raw_size))
    print("test size: {}".format(test_size))
    print("add size: {}".format(traina_size))
    print("full/total size: {}".format(train_full_size))

def specific_size_split_rand(feature_cols, feature_cols_name, raw_size, traina_size, traina_num, test_size, rand_seed_set, parent_path, split_file, file_size, gate_traina):
    for rand_seed in rand_seed_set:
        specific_size_split(feature_cols, feature_cols_name, raw_size, traina_size, traina_num, test_size, rand_seed, parent_path, split_file, file_size, gate_traina)     
        
        

def imbalanced_split(feature_cols, feature_cols_name, raw_size, test_size, rand_seed, parent_path, split_file, file_size, gate_traina, raw_minority_rate, minority, target):
    cols = feature_cols[feature_cols_name]
    df = pd.read_csv(parent_path + split_file)
    
    df = df[cols]
    remain = df
    
    # split test
    test, remain = sample_remain_no_replace(remain, test_size, rand_seed)
    train_full_size = file_size - test_size
    
    # save test
    test.to_csv(parent_path + "/test_{}_size{}_rand{}.csv".format(feature_cols_name, train_full_size, rand_seed), index=False)
    
    # Calculate minority instances in raw dataset
    num_minority_raw = int(raw_minority_rate * raw_size)
    
    # Separate minority and majority
    minority_df = remain[remain[target] == minority]
    majority_df = remain[remain[target] != minority]
    
    # Sample minority and majority for raw dataset
    minority_raw = minority_df.sample(n=num_minority_raw, random_state=rand_seed)
    majority_raw = majority_df.sample(n=raw_size - num_minority_raw, random_state=rand_seed)
   
    interleaved_list = []
    for minority_row, majority_row in zip(minority_raw.itertuples(index=False), majority_raw.itertuples(index=False)):
        interleaved_list.append(minority_row)
        interleaved_list.append(majority_row)
    
    remaining_majority = len(majority_raw) - len(minority_raw)
    if remaining_majority > 0:
        interleaved_list.extend(majority_raw.tail(remaining_majority).itertuples(index=False))
    
    raw = pd.DataFrame(interleaved_list, columns=cols)
    # Update remain
    remain = remain.reset_index(drop=True)
    
    # Save raw and update train full size
    train_full_size = file_size - test_size 
    raw.to_csv(parent_path + "/train_{}_size{}_total{}_rand{}.csv".format(feature_cols_name, raw_size, train_full_size, rand_seed), index=False)
    
    if gate_traina:
        # Save remaining dataset if gate_traina is True
        remain.to_csv(parent_path + "/train_generate_colsbase_rand{}.csv".format(rand_seed), index=False)
    
    print("-"*20)
    print("Dataset: {}".format(split_file))
    print("cols name: {}".format(feature_cols_name))
    print("raw size: {}".format(raw_size))
    print("test size: {}".format(test_size))
    print("raw minority rate: {}".format(raw_minority_rate))
    print("minority class: {}".format(minority))
    print("full/total size: {}".format(train_full_size))
        

def imbalanced_split_rand(feature_cols, feature_cols_name, raw_size, test_size, rand_seed_set, parent_path, split_file, file_size, gate_traina, raw_minority_rate, minority, target):
    for rand_seed in rand_seed_set:
        imbalanced_split(feature_cols, feature_cols_name, raw_size, test_size, rand_seed, parent_path, split_file, file_size, gate_traina, raw_minority_rate, minority, target)

