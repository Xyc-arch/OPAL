from openai import OpenAI
from fine_tune import upload_file
import pandas as pd
import csv
import time
import math

post = ""
post_shap = "_shap"
post_bootstrap = "_bootstrap"
post_mixup = "_mixup"
post_gan = "_gan"
post_gmm = "_gmm"
post_imbalanced = "_imbalanced"
post_spurious = "_spurious"
post_ablate = "_ablate"
post_imbalanced_focus = "_imbalanced_focus"
post_spurious_focus = "_spurious_focus"


'''
script to use API 
'''

def csv_to_string(file_path, delimiter=',', limit = 0):
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        count = 0

        csv_string = ""

        for row in reader:
            csv_string += ", ".join(row) + "\n"
            count += 1
            if limit:
                if count > limit: break
                
        print("total rows: {}".format(count))

    return csv_string


def csv_to_string_specific_row(file_path, start_row=0, end_row=100, delimiter=','):
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        csv_string = ""
        count = 0

        for row in reader:
            if count < start_row:
                count += 1
                continue

            csv_string += ", ".join(row) + "\n"
            count += 1

            if count > end_row:
                break
    return csv_string


def read_in_text(file_path, num_lines):
    with open(file_path, 'r') as file:
        lines = []
        for _ in range(num_lines):
            line = file.readline()
            if not line:
                break  
            lines.append(line)
        return ''.join(lines)


def read_in_text_specific_row(file_path, start_row=0, end_row=100):
    with open(file_path, 'r') as file:
        lines = []
        for current_row, line in enumerate(file, start=1):
            if current_row < start_row:
                continue
            if current_row > end_row:
                break
            lines.append(line)
        return ''.join(lines)

client = OpenAI(
    api_key='<your openai-api key>', # replace with your own api key
)


def generate_synthetic_data(model, sys_ins, user_ins, data_string_ls):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_ins},
            {"role": "user", "content": user_ins[0]},
            {"role": "user", "content": data_string_ls[0]},
            {"role": "user", "content": user_ins[1]}
        ],
        temperature = 0.60
    )
    return response.choices[0].message.content

import csv


def convert_txt_to_csv(input_file_path, output_csv_path):
    df = pd.read_csv(input_file_path)

    # print(df)
    df.to_csv(output_csv_path, index=False, header=False)
    

# the main function, specify the model name, system instructions, user instructions, string data, and output path
def write_response_to(model_name, sys_ins, user_ins, data_content, output_path):
    response = generate_synthetic_data(model_name, sys_ins, user_ins, data_content)
    # print(response)
    with open(output_path, mode='a', newline='', encoding='utf-8') as file:
        file.write(response + "\n")




if __name__ == "__main__":
    
    output_max_num = 20 # !!!
    
    current_post = post_spurious_focus # !!! # can change it to post_imbalanced_focus
    
    data_name_ls = {0: "openmlDiabetes", 1: "gender", 2: "heart_failure", 3: "craft"}
    data_name = data_name_ls[0]  # !!!
    
    raw_size_ls = {0: 20, 1: 50, 2: 100}
    raw_size = raw_size_ls[1]  # !!!
    
    cut_idx5 = {0:1, 1:5, 2:6, 3:10, 4:11, 5:15, 6:16, 7:20}
    cut_idx20 = {0:1, 1:20, 2:21, 3:40, 4:41, 5:60, 6:61, 7:80}
    cut_idx10 = {0:1, 1:10, 2:11, 3:20, 4:21, 5:30, 6:31, 7:40}
    cut_idx10_cross = {0:1, 1:10, 2:11, 3:20, 4:11, 5:20, 6:1, 7:10}
    cut_idx20_repeat = {0:1, 1:20, 2:21, 3:40, 4:1, 5:20, 6:21, 7:40}
    cut_idx20_cross = {0:1, 1:20, 2:21, 3:40, 4:21, 5:40, 6:1, 7:20}
    cut20_raw20 = {0:1, 1:20, 2:1, 3:20}
    cut25_raw50 = {0:1, 1:25, 2:26, 3:50}
    cut25_raw100 = {0:1, 1:25, 2:26, 3:50, 4:51, 5:75, 6:76, 7:100}
    cut_idx20_window = {0:1, 1: 20, 2:1, 3:20, 4:1, 5:20, 6:1, 7:20, 8:1, 9:20}
    cut_idx = cut_idx20_window  
    
    iteration_num = 1
    if raw_size > output_max_num:
        iteration_num = math.ceil(raw_size/output_max_num)
    
    
    for rand in [1, 2, 6, 8, 42]:
        data_path_openmlDiabetes = '../data/openmlDiabetes/openmlDiabetes_see_small_raw{}{}'.format(raw_size, current_post) + '/train_colsbase_size{}_total568_rand{}.txt'.format(raw_size, rand) 
        data_path_heart_failure = '../data/heart_failure/heart_failure_see_small_raw{}{}'.format(raw_size, current_post) + '/train_colsbase_size{}_total199_rand{}.txt'.format(raw_size, rand)
        data_path_gender = '../data/gender/gender_see_small_raw{}{}'.format(raw_size, current_post) + '/train_colsbase_size{}_total4001_rand{}.txt'.format(raw_size, rand)
        data_path_craft = '../data/craft/craft_see_small_raw100{}'.format(current_post) + '/train_colsbase_size100_total600_rand{}.txt'.format(rand) 
        
        data_path_ls = {"openmlDiabetes": data_path_openmlDiabetes, "gender": data_path_gender, "heart_failure": data_path_heart_failure, "craft": data_path_craft}
        data_path = data_path_ls[data_name]
        # replace with your own current path
        current_data_path = data_path 
        output_path = './synthetic/{}_raw{}_rand{}.txt'.format(data_name, raw_size, rand) # !!!
        
        # system instruction
        
        sys_topic_openmlDiabetes = 'diabetes'
        sys_topic_heart_failure = 'heart failure'
        sys_topic_gender = 'facial characteristic and gender'
        sys_topic_craft = "(artificially) crafted data"
        sys_topic_ls = {"openmlDiabetes": sys_topic_openmlDiabetes, "gender": sys_topic_gender, "heart_failure": sys_topic_heart_failure, "craft": sys_topic_craft}
        sys_topic = sys_topic_ls[data_name]

        sys_app_openmlDiabetes = "developing personalized treatment plans, conducting epidemiological studies, and optimizing healthcare resource allocation"
        sys_app_heart_failure = "developing predictive models for heart failure risk and creating personalized healthcare plans"
        sys_app_gender = "developing policies and strategies for gender equality and inclusion"
        sys_app_craft = "statistical simulation"
        sys_app_ls = {"openmlDiabetes": sys_app_openmlDiabetes, "gender": sys_app_gender, "heart_failure": sys_app_heart_failure, "craft": sys_app_craft}
        sys_app = sys_app_ls[data_name]
        sys_ins_tmp = ''' 
        You are a UC, Berkeley statistician expert in analyzing {} records. 
        Your objective is to predict/guess new chunk of records that closely mirrors the statistical properties of a provided real-world records. 
        This predicted/guessed records collection will be instrumental for downstream tasks such as {}. 
        You are good at in-context learning. You always think step-by-step, use chain-of-thoughts, and your common sense. You can do perfect job!
        '''.format(sys_topic, sys_app) 
        
        
        # user instructions
        
        records_num = output_max_num 
        user_ins0_features_openmlDiabetes = ''' Number of times pregnant, Plasma glucose concentration, Diastolic blood pressure, Triceps skin fold thickness, 2-Hour serum insulin, Body mass index, Diabetes pedigree function, Age, Class (class value 1 is interpreted as "tested positive for diabetes") '''
        user_ins0_features_heart_failure = "age, anaemia, creatinine phosphokinase , whether have diabetes, ejection_fraction, whether have high_blood_pressure, platelets, serum creatinine, serum sodium, sex, whether smoking, follow-up time period, death of heart failure (1 means positive)"
        user_ins0_features_gender = "long hair (1 long), forehead width (cm), forehead height (cm), nose wide (1 wide), nose_long (1 long), lips thin (1 thin), distance from nose to lip long (1 long), gender (0 male 1 female)"
        user_ins0_features_craft = "X1, X2, X3, X4, X5, X6, X7, X8, X9, Target (binary target condition on X)"
        user_ins0_features_ls = {"openmlDiabetes": user_ins0_features_openmlDiabetes, "gender": user_ins0_features_gender, "heart_failure": user_ins0_features_heart_failure, "craft": user_ins0_features_craft}
        user_ins0_features = user_ins0_features_ls[data_name]
        user_ins0_tmp = ''' 
        The following is the text of the observed records of {}. 
        Investigate it carefully. Each row represents a block group with records about {}. 
        Guess and craft new {} records of textural representation as if they were from the same source of the given records. Do not replicate the real records.
        Discover the pattern and trends of the real records. Your guess should preserve statistical properties and causal relationship. All pairs of correlation of variables should be very close to real-world records. All variable's marginal distribution should be closely align with the real dataset. Learn complicated associations and interplays.
        Introduce interpretable variation. The guess should closely resemble real records in terms of trends and patterns. 
        Use your domain knowledge and understanding of {} and other factors when you are predicting. 
        Output predicted records in the same format as real-world records format and do not order or number the guessed records!
        '''.format(sys_topic, user_ins0_features, records_num, sys_topic) # !!!
        
        user_ins1 = '''  
        Your response must only exclusively contain your guessed records with the same format as the provided excellent example (e.g. object is value)! No other words.
        Please always think step-by-step, use chain-of-thoughts, and your common sense. 
        The guessed {} records are: 
        '''.format(records_num)


        # if raw_size == 20:
        #     cut_idx = cut20_raw20
        # elif raw_size == 50:
        #     cut_idx = cut25_raw50
        # elif raw_size == 100:
        #     cut_idx = cut25_raw100

        
        
        start_time = time.time()
        # model_used = "gpt-3.5-turbo-1106"
        model_used = "gpt-4-1106-preview"
        # model_used = "gpt-4-turbo-preview"
        for i in range(iteration_num): # !!!
            data_content0 = read_in_text_specific_row(current_data_path, cut_idx[i*2], cut_idx[i*2+1])

            user_ins_total_tmp = [user_ins0_tmp, user_ins1]
            data_content_ls_tmp = [data_content0]
            
            write_response_to(model_used, sys_ins_tmp, user_ins_total_tmp, data_content_ls_tmp, output_path)
        end_time = time.time()

        elapsed_time = end_time - start_time
        
        print("elapsed time: {}".format(elapsed_time))
        
        