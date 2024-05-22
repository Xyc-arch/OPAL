
import pandas as pd
import numpy as np


def map_categorical_to_numeric(df):
    for column in df.select_dtypes(include=['object']).columns:
        unique_values = df[column].unique()
        mapping = {k: v for v, k in enumerate(unique_values)}
        df[column] = df[column].map(mapping)
    return df

# insurance_df = pd.read_csv('insurance_raw.csv')
# transformed_df = map_categorical_to_numeric(insurance_df.copy())

# print(transformed_df.head())
# transformed_df.to_csv("insurance.csv", index=False)



def process(parent_path, data_name, save_name, cols_to_keep):
    data_path = parent_path + "/{}.csv".format(data_name)
    df_tmp = pd.read_csv(data_path)
    df_num = map_categorical_to_numeric(df_tmp.copy())
    df_final = df_num[cols_to_keep]
    save_path = parent_path + "/{}.csv".format(save_name)
    df_final.to_csv(save_path, index = False)
    

if __name__ == "__main__":
    
    
    ''' heart failure '''
    # parent_path = "/Users/yichen_xu/Desktop/D_disk/research/2023_fall/data/heart_failure/heart_failure_see_small_raw100"
    # data_name = "heart_failure_clinical_records_dataset"
    # save_name = "heart_failure"
    # cols_to_keep = ["age",	"anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction", "high_blood_pressure",
    #              "platelets", "serum_creatinine", "serum_sodium", "sex", "smoking", "time", "DEATH_EVENT"]
    
    ''' gender '''
    parent_path = "/Users/yichen_xu/Desktop/D_disk/research/2023_fall/data/gender/gender_see_small_raw100"
    data_name = "gender_raw"
    save_name = "gender"
    cols_to_keep = ["long_hair", "forehead_width_cm", "forehead_height_cm", "nose_wide", "nose_long", "lips_thin", "distance_nose_to_lip_long", "gender"]
    
    process(parent_path, data_name, save_name, cols_to_keep)