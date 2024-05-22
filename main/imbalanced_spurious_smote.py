from meta_data import *
import pandas as pd
from sklearn.utils import resample
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import argparse
np.random.seed(42)


def smote_synthetic_data(raw_data_path, generate_size, save_path, target_column, minority_target):
    raw_data = pd.read_csv(raw_data_path)
    minority_data = raw_data[raw_data[target_column] == minority_target].reset_index(drop=True)
    features = minority_data.drop(columns=[target_column])
    synthetic_features = []
    
    # Determine if all values in each column are integers
    is_integer_column = features.applymap(lambda x: float(x).is_integer()).all()

    for _ in range(generate_size):
        num_samples_to_use = np.random.randint(1, len(minority_data) + 1)
        num_samples_to_use = 2
        samples = features.sample(n=num_samples_to_use, replace=True)
        synthetic_sample = samples.mean(axis=0)
        
        for col in synthetic_sample.index:
            if is_integer_column[col]:  # If all values in the column are integers
                synthetic_sample[col] = int(round(synthetic_sample[col]))  # Correctly casting to int
            else:
                synthetic_sample[col] = synthetic_sample[col]
                
        synthetic_features.append(synthetic_sample)
    
    synthetic_features_df = pd.DataFrame(synthetic_features, columns=features.columns)
    synthetic_features_df[target_column] = minority_target
    
    synthetic_data = synthetic_features_df.sample(frac=1).reset_index(drop=True)
    synthetic_data.to_csv(save_path, index=False)
    return save_path



def apply_smote(raw_data_path, save_path, target_column, minority_target, generate_size, mixup = False, size = None):
    # Load data
    raw_data = pd.read_csv(raw_data_path)
    
    if size:
        raw_data = raw_data[:size]
    
    # Separate features and target
    X = raw_data.drop(columns=[target_column])
    y = raw_data[target_column]
    
    is_integer_column = raw_data.applymap(lambda x: float(x).is_integer()).all()
    
    # Determine the number of samples to generate
    n_samples_to_add = generate_size 
    
    # Initialize SMOTE with a specified number of samples to generate
    if mixup:
        smote = SMOTE(sampling_strategy={0: 200, 1: 200}, k_neighbors=2, random_state=42)
    else:
        smote = SMOTE(sampling_strategy={minority_target: generate_size}, random_state=42)
    
    # Apply SMOTE
    X_res, y_res = smote.fit_resample(X, y)
    
    # Extract only the synthetic samples
    if mixup:
        resampled_data = pd.concat([X_res, y_res], axis=1)
        resampled_data = shuffle(resampled_data, random_state=42)
        synthetic_data = resampled_data.sample(n=generate_size, random_state=42)
    else:
        synthetic_indices = y_res[y_res == minority_target].index[-n_samples_to_add:]  # Get last 'n_samples_to_add' indices
        synthetic_data = X_res.loc[synthetic_indices]
        synthetic_data[target_column] = y_res[synthetic_indices]
    
    for col in is_integer_column.index[is_integer_column]:
        synthetic_data[col] = synthetic_data[col].round().astype(int)
    
    # Shuffle the synthetic dataset
    synthetic_data = shuffle(synthetic_data).reset_index(drop=True)
    
    # Save the synthetic data to a new CSV file
    synthetic_data.to_csv(save_path, index=False)
    
    return save_path



def smote_synthetic_rand(parent_path, rand_set, raw_size, generate_size, total_size, target_column, minority_target=0, mixup = False, size = None):
    for rand in rand_set:
        save_path = parent_path + "/train_generate_colsbase_rand{}.csv".format(rand)
        raw_data_path = parent_path + "/train_colsbase_size{}_total{}_rand{}.csv".format(raw_size, total_size, rand)

        # smote_synthetic_data(raw_data_path, generate_size, save_path, target_column, minority_target)
        apply_smote(raw_data_path, save_path, target_column, minority_target, generate_size, mixup, size)


if __name__ == "__main__":
    
    control_list = {0: "imbalanced", 1: "spurious"}
    control = control_list[1]

    data_name_ls = {0: "openmlDiabetes", 1: "heart_failure", 2: "gender", 3: "craft"}
    
    
    data_name = data_name_ls[2]
    
    parser = argparse.ArgumentParser(description='Input for hyperpara.')
    parser.add_argument('--control', type=str, required=True, help='The control (e.g., "spurious" or "imbalanced")')
    parser.add_argument('--data_name', type=str, required=True, choices=["openmlDiabetes", "gender", "heart_failure", "craft"], help='The name of the dataset (e.g., "openmlDiabetes")')
    
    args = parser.parse_args()
    
    control = eval(args.control)
    data_name = args.data_name

    info = info_dict[data_name]
    total_size = info["total_size"]
    target_column = info["target"]
    task_type = info["task_type"]
    raw_size = info["raw_size"]
    
    rand_set = [1, 2, 6, 8, 42]
    
    
    if control == "imbalanced":
        parent_path = './data/{}/{}_see_small_raw{}{}'.format(data_name, data_name, raw_size, post_imbalanced_smote)
        smote_synthetic_rand(parent_path, rand_set, raw_size, raw_size, total_size, target_column)
    elif control == "spurious":
        parent_path = './data/{}/{}_see_small_raw{}{}'.format(data_name, data_name, raw_size, post_spurious_smote)
        smote_synthetic_rand(parent_path, rand_set, raw_size, raw_size, total_size, target_column, mixup=True, size=20)