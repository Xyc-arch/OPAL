import pandas as pd
from meta_data import *

def row_to_sentence(row):
    """Transform a DataFrame row into a descriptive sentence."""
    return ', '.join([f"{col} is {row[col]}" for col in row.index]) + '.'

def to_context(parent_path, table_name):
    data = pd.read_csv(parent_path + "/{}.csv".format(table_name))

    # Apply the function to each row and save the sentences to a list
    sentences = data.apply(row_to_sentence, axis=1).tolist()

    # Save the sentences to a text file
    output_file_path = parent_path + "/{}.txt".format(table_name)
    with open(output_file_path, 'w') as file:
        for sentence in sentences:
            file.write(sentence + '\n')


if __name__ == "__main__":


    ''' general '''
    data_name_ls = {0: "abalone", 1: "california", 2: "wine", 3: "gender", 4: "heart_failure", 5: "openmlDiabetes", 6: "craft"}
    data_name = data_name_ls[3]
    info = info_dict[data_name]
    
    shap_raw = info["shap_raw"]
    file_size = info["file_size"]
    raw_size = info["raw_size"]
    # raw_size = 50
    shap_raw = info["shap_raw"]
    total_size = info["total_size"]
    current_post = post_spurious_focus
    
    for rand in [1, 2, 6, 8, 42]:
        parent_path = '../data/{}/{}_see_small_raw{}{}'.format(data_name, data_name, raw_size, current_post)
        table_name = 'train_colsbase_size{}_total{}_rand{}'.format(raw_size, total_size, rand)
        to_context(parent_path, table_name)
