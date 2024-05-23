

import pandas as pd


post = ""
post_shap = "_shap"
post_bootstrap = "_bootstrap"
post_mixup = "_mixup"
post_gan = "_gan"
post_gmm = "_gmm"
post_imbalanced = "_imbalanced"
post_spurious = "_spurious"
post_ablate = "_ablate"
post_origin = "_origin"
post_imbalanced_duplicate = "_imbalanced_duplicate"
post_imbalanced_smote = "_imbalanced_smote"
post_imbalanced_focus = "_imbalanced_focus"
post_spurious_duplicate = "_spurious_duplicate"
post_spurious_focus = "_spurious_focus"
post_spurious_smote = "_spurious_smote"

# train = pd.read_csv(current_path + 'heart_data/train.csv')
# traina = pd.read_csv(current_path + 'heart_data/traina.csv')
# test = pd.read_csv(current_path + 'heart_data/test.csv')
# train.head()

drop_cols_fit = ['LOO_Score']

csv_train50_rand42 = [
    'train_colsbase_size50_total7763_rand42.csv',
    'train50a.csv',
    'train50b.csv',
    'train50c.csv',
    'train50d.csv',
    'trainfull_colsbase_size7763_rand42.csv',
    'test_colsbase_size7763_rand42.csv'
]

csv_train120_rand666 = [
    'train_colsbase_size120_total7763_rand666.csv',
    'train60a.csv',
    'train60b.csv',
    'train60c.csv',
    'train60d.csv',
    'trainfull_colsbase_size7763_rand666.csv',
    'test_colsbase_size7763_rand666.csv'
]

csv_train50_rand64 = [
    'train_colsbase_size50_total7763_rand64.csv',
    'train50a_colsbase_train7763_rand64.csv',
    'train50b_colsbase_train7763_rand64.csv',
    'train50c_colsbase_train7763_rand64.csv',
    'train50d_colsbase_train7763_rand64.csv',
    'trainfull_colsbase_size7763_rand64.csv',
    'test_colsbase_size7763_rand64.csv'
]

train50_rand42_datasets = [
    ['train_colsbase_size50_total7763_rand42.csv'],
    ['train_colsbase_size50_total7763_rand42.csv', 'train50a.csv'],
    ['train_colsbase_size50_total7763_rand42.csv', 'train50a.csv', 'train50b.csv'],
    ['train_colsbase_size50_total7763_rand42.csv', 'train50a.csv', 'train50b.csv', 'train50c.csv'],
    ['train_colsbase_size50_total7763_rand42.csv', 'train50a.csv', 'train50b.csv', 'train50c.csv', 'train50d.csv'],
    ['trainfull_colsbase_size7763_rand42.csv']
]

# test acc: 0.539
# test acc: 0.575
# test acc: 0.574
# test acc: 0.596
# test acc: 0.645
# test acc: 0.647 7776 + 1000(test)

train50_rand64_datasets = [
    ['train_colsbase_size50_total7763_rand64.csv'],
    ['train_colsbase_size50_total7763_rand64.csv', 'train50a_colsbase_train7763_rand64.csv'],
    ['train_colsbase_size50_total7763_rand64.csv', 'train50a_colsbase_train7763_rand64.csv', 'train50b_colsbase_train7763_rand64.csv'],
    ['train_colsbase_size50_total7763_rand64.csv', 'train50a_colsbase_train7763_rand64.csv', 'train50b_colsbase_train7763_rand64.csv', 'train50c_colsbase_train7763_rand64.csv'],
    ['train_colsbase_size50_total7763_rand64.csv', 'train50a_colsbase_train7763_rand64.csv', 'train50b_colsbase_train7763_rand64.csv', 'train50c_colsbase_train7763_rand64.csv', 'train50d_colsbase_train7763_rand64.csv'],
    ['trainfull_colsbase_size7763_rand64.csv']
]

train120_rand666_datasets = [
    ['train_colsbase_size120_total7763_rand666.csv'],
    ['train_colsbase_size120_total7763_rand666.csv', 'train60a.csv'],
    ['train_colsbase_size120_total7763_rand666.csv', 'train60a.csv', 'train60b.csv'],
    ['train_colsbase_size120_total7763_rand666.csv', 'train60a.csv', 'train60b.csv', 'train60c.csv'],
    ['train_colsbase_size120_total7763_rand666.csv', 'train60a.csv', 'train60b.csv', 'train60c.csv', 'train60d.csv'],
    ['trainfull_colsbase_size7763_rand666.csv']
]

# test acc: 0.579
# test acc: 0.616
# test acc: 0.63
# test acc: 0.632
# test acc: 0.638
# test acc: 0.647

method_list = {
    1: "logistic",
    2: "randomForest",
    3: "svm",
    4: "bayes",
    5: "catBoost"
}

method_cts_list = {
    1: "linear",
    2: "catBoost",
    3: "randomForest"
}

scale_method = {"svm", "randomForest"}



# Load datasets, place target in the end

''' binary classification datasets '''

heart_feature_cols = {
    "colsbase": ['Age', 'Diabetes', 'Smoking', 'Obesity', 'Income', 'Heart Attack Risk'],
    "cols1": ['Age', 'Diabetes', 'Smoking', 'Obesity', 'Cholesterol', 'Heart Rate', 'Heart Attack Risk'],
    "cols2": ['Family History', 'Exercise Hours Per Week', 'Diabetes', 'Alcohol Consumption', 'Obesity', 'Cholesterol', 'Heart Rate', 'Heart Attack Risk'],
    "cols3": ['Age', 'Diabetes', 'Smoking', 'Obesity', 'Income', 'Cholesterol', 'Heart Rate', 'Family History', 'Exercise Hours Per Week','Alcohol Consumption', 'Heart Attack Risk'],
}


breast_cancer_feature_cols = {
    "colsbase": ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '1'],
    "cols1": ['7', '8', '9', '10', '1']
}


uci_breast_cancer_1_feature_cols = {
    "colsbase": ["age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad", "irradiat", "Class"]
}

uci_heart_failure_feature_cols = {
    "colsbase": ["age",	"anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction", "high_blood_pressure",
                 "platelets", "serum_creatinine", "serum_sodium", "sex", "smoking", "time", "DEATH_EVENT"]
}


kaggle_heart_indicators_feature_cols = {
    "colsbase": 
        ['PhysicalHealthDays',
       'MentalHealthDays', 'LastCheckupTime', 'PhysicalActivities',
       'SleepHours', 'RemovedTeeth', 'HadAngina',
       'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD',
       'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
       'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
       'DifficultyConcentrating', 'DifficultyWalking',
       'DifficultyDressingBathing', 'DifficultyErrands', 'SmokerStatus',
       'ECigaretteUsage', 'ChestScan', 'RaceEthnicityCategory', 'AgeCategory',
       'HeightInMeters', 'WeightInKilograms', 'BMI', 'AlcoholDrinkers',
       'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 'TetanusLast10Tdap', 'HadHeartAttack'],
    "cols1":
        ['PhysicalActivities',
       'SleepHours', 'RemovedTeeth', 'HadAngina',
       'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD',
       'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
       'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
       'SmokerStatus',
       'ECigaretteUsage', 'ChestScan', 'RaceEthnicityCategory', 'AgeCategory',
       'HeightInMeters', 'WeightInKilograms', 'BMI', 'AlcoholDrinkers',
       'HadHeartAttack'],
    "cols2":
        ['PhysicalActivities',
       'HadAsthma', 'HadSkinCancer', 'HadCOPD',
       'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
       'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
       'ECigaretteUsage', 'AgeCategory',
       'HeightInMeters', 'WeightInKilograms', 'BMI', 'AlcoholDrinkers',
       'HadHeartAttack']
}


''' continuous regression dataset '''

abalone_feature_cols = {
    "colsbase":
        ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight',
       'Viscera_weight', 'Shell_weight', 'Class_number_of_rings']
}


california_feature_cols = {
    "colsbase":
        [
    'median income square', 'median income cubic', 'ln of housing median age', 
    'ln total rooms over population', 'ln of bedrooms over population', 'ln of population over households', 
    'ln of households', 'ln of households'
    ]
}

cal_raw_feature_cols = {
    "colsbase":
        ["housingMedianAge", "totalRooms", "totalBedrooms", "population", "households",
         "medianIncome", "medianHouseValue"]
}


house_feature_cols = {
    "colsbase":
        ['P1', 'P5p1', 'P6p2', 'P11p4', 'P14p9', 'P15p1', 'P15p3', 'P16p2',
       'P18p2', 'P27p4', 'H2p2', 'H8p2', 'H10p1', 'H13p1', 'H18pA', 'H40p4',
       'price']
}


insurance_feature_cols = {
    "colsbase":
        ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
}


openmlDiabetes_feature_cols = {
    "colsbase":
        ['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age', 'class']
}

gender_feature_cols = {
    "colsbase":
        ["long_hair", "forehead_width_cm", "forehead_height_cm", "nose_wide", "nose_long", "lips_thin", "distance_nose_to_lip_long", "gender"]
}

wine_feature_cols = {
    "colsbase":
        ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]
}

craft_feature_cols = {
    "colsbase":
        ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "Target"]
}

''' settings '''


def get_data_chunk(train_name, traina_name, test_name, parent_path):
    train = pd.read_csv(parent_path + f'/{train_name}')
    traina = pd.read_csv(parent_path + f'/{traina_name}')
    test = pd.read_csv(parent_path + f'/{test_name}')
    
    
    return train, traina, test


def get_data(current_files, current_datasets, parent_path):
    dataframes = {}
    for file in current_files:
        dataframes[file.split('.')[0]] = pd.read_csv(parent_path + f'/{file}')
    test_name = current_files[-1].split('.')[0]
    
    print(dataframes)
    return current_datasets, dataframes, test_name




def get_meta(feature_cols_name, train_size, traina_size, train_total_size, rand_seed, add_num):
    # files = [
    #     "train_{}_size{}_total{}_rand{}.csv".format(feature_cols_name, train_size, train_total_size, rand_seed),
    #     'train{}a_on{}_{}_train{}_rand{}.csv'.format(traina_size, train_size, feature_cols_name, train_total_size, rand_seed),
    #     'train{}b_on{}_{}_train{}_rand{}.csv'.format(traina_size, train_size, feature_cols_name, train_total_size, rand_seed),
    #     'train{}c_on{}_{}_train{}_rand{}.csv'.format(traina_size, train_size, feature_cols_name, train_total_size, rand_seed),
    #     'train{}d_on{}_{}_train{}_rand{}.csv'.format(traina_size, train_size, feature_cols_name, train_total_size, rand_seed),
    #     'trainfull_{}_size{}_rand{}.csv'.format(feature_cols_name, train_total_size, rand_seed),
    #     'test_{}_size{}_rand{}.csv'.format(feature_cols_name, train_total_size, rand_seed)
    # ]
    
    # train_datasets = [
    #     [files[0]],
    #     [files[0], files[1]],
    #     [files[0], files[1], files[2]],
    #     [files[0], files[1], files[2], files[3]],
    #     [files[0], files[1], files[2], files[3], files[4]],
    #     [files[5]]
    # ]
    
    add_alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q']
    
    # files meta
    files = []
    files.append("train_{}_size{}_total{}_rand{}.csv".format(feature_cols_name, train_size, train_total_size, rand_seed))
    for add_index in range(add_num):
        alphabet = add_alphabet[add_index]
        files.append('train{}{}_on{}_{}_train{}_rand{}.csv'.format(traina_size, alphabet, train_size, 
                                                                   feature_cols_name, train_total_size, rand_seed))
    files.append('trainfull_{}_size{}_rand{}.csv'.format(feature_cols_name, train_total_size, rand_seed))
    files.append('test_{}_size{}_rand{}.csv'.format(feature_cols_name, train_total_size, rand_seed))
    
    
    # train_datasets meta
    train_datasets = []
    for i in range(add_num+1):
        current_dataset = []
        for j in range(i+1):
            current_dataset.append(files[j])
        train_datasets.append(current_dataset)
    train_datasets.append([files[-2]])
    
    return files, train_datasets


rm_start_dict = {1: 0, 2: 100, 6: 200, 8: 300, 42: 400}
rm_end_dict = {1: 99, 2: 199, 6: 299, 8: 399, 42: 499}

abalone_info = {
    "total_size": 3177,
    "target": 'Class_number_of_rings',
    "metric": "RMSE",
    "colsbase": abalone_feature_cols["colsbase"][:-1],
    "feature_cols": abalone_feature_cols,
    "task_type": "regress",
    "raw_size": 100,
    "shap_raw": 100,
    "file_size": 4177,
    "test_size": 1000
}

california_info = {
    "total_size": 7763,
    "target": "medianHouseValue",
    "metric": "RMSE",
    "colsbase": cal_raw_feature_cols["colsbase"][:-1],
    "feature_cols": cal_raw_feature_cols,
    "task_type": "regress",
    "raw_size": 100,
    "shap_raw": 100,
    "file_size": 8763,
    "test_size": 1000
}

openmlDiabetes_info = {
    "total_size": 568,
    "target": "class",
    "metric": "error",
    "colsbase": openmlDiabetes_feature_cols["colsbase"][:-1],
    "feature_cols": openmlDiabetes_feature_cols,
    "task_type": "class",
    "raw_size": 100,
    "shap_raw": 100,
    "file_size": 768,
    "test_size": 200
}

heart_failure_info = {
    "total_size": 199,
    "target": 'DEATH_EVENT',
    "metric": "error",
    "colsbase": uci_heart_failure_feature_cols["colsbase"][:-1],
    "feature_cols": uci_heart_failure_feature_cols,
    "task_type": "class",
    "raw_size": 50,
    "shap_raw": 50,
    "file_size": 299,
    "test_size": 100
}

gender_info = {
    "total_size": 4001,
    "target": 'gender',
    "metric": "error",
    "colsbase": gender_feature_cols["colsbase"][:-1],
    "feature_cols": gender_feature_cols,
    "task_type": "class",
    "raw_size": 50,
    "shap_raw": 100,
    "file_size": 5001,
    "test_size": 1000
}


wine_info = {
    "total_size": 3898,
    "target": 'quality',
    "metric": "RMSE",
    "colsbase": wine_feature_cols["colsbase"][:-1],
    "feature_cols": wine_feature_cols,
    "task_type": "regress",
    "raw_size": 50,
    "shap_raw": 100,
    "file_size": 4898,
    "test_size": 1000
}


craft_info = {
    "total_size": 600,
    "target": 'Target',
    "metric": "error",
    "colsbase": craft_feature_cols["colsbase"][:-1],
    "feature_cols": craft_feature_cols,
    "task_type": "class",
    "raw_size": 100,
    "shap_raw": 100,
    "file_size": 1000,
    "test_size": 400
}

info_dict = {"abalone": abalone_info, "california": california_info, "openmlDiabetes": openmlDiabetes_info, "heart_failure": heart_failure_info, "wine": wine_info, "gender": gender_info, "craft": craft_info}

if __name__ == "__main__":
    
    feature_cols_name = "colsbase"
    train_size = 50
    traina_size = 50
    train_total_size = 7776
    files, train_datasets = get_meta(feature_cols_name, train_size, traina_size, train_total_size, 42, 4)
    

    
    



