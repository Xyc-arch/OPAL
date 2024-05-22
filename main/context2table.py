import pandas as pd
from meta_data import *

def sentence_to_row(sentence):
    """Transform a descriptive sentence back into a dictionary representing a row."""
    # Remove any trailing characters like newlines and full stops
    cleaned_sentence = sentence.strip().strip('.')
    elements = cleaned_sentence.split(', ')
    row = {}
    for element in elements:
        col, value = element.split(' is ')
        row[col] = float(value)
    return row


def to_table(parent_path, context_name, table_name):
    # Path to the text file containing sentences
    input_file_path = parent_path + '/{}.txt'.format(context_name)

    # Read the sentences from the file, ensuring to strip any whitespace or newline characters
    with open(input_file_path, 'r') as file:
        sentences = [line.strip() for line in file if line.strip()]

    # Transform each sentence back into a row (dictionary)
    rows = [sentence_to_row(sentence) for sentence in sentences]

    # Convert the list of dictionaries into a DataFrame
    reconstructed_data = pd.DataFrame(rows)

    # Path for the reconstructed CSV file
    reconstructed_csv_path = parent_path + '/{}.csv'.format(table_name)

    # Save the DataFrame as a CSV file
    reconstructed_data.to_csv(reconstructed_csv_path, index=False)
    
    print(reconstructed_csv_path)
    

if __name__ == "__main__":

    post_shap = "_shap"
    post_loo = "_loo"
    post_mixup = "_mixup"
    post = ""
    rand_set = [1, 2, 6, 8, 42, "all"]
    
    
    ''' general '''
    data_name_ls = {0: "abalone", 1: "california", 2: "openmlDiabetes", 3: "wine", 4: "gender", 5: "heart_failure", 6: "craft"}
    dataset_name = data_name_ls[4]
    info = info_dict[dataset_name]
    # raw_size = info["raw_size"]
    raw_size = info["raw_size"]
    shap_raw = info["shap_raw"]
    current_post = post_spurious_focus
    
    
    for rand in [1, 2, 6, 8, 42]:
        rand_all = "all"
        parent_path = '../data/{}/{}_see_small_raw{}{}'.format(dataset_name, dataset_name, raw_size, current_post)  
        context_name = "4-rand{}".format(rand)
        table_name = "train_generate_colsbase_rand{}".format(rand)
        to_table(parent_path, context_name, table_name)