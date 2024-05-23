# Introduction

This repository contains the code for the paper titled "Synthetic Oversampling: Theory and A Practical Approach Using LLMs to Address Data Imbalance", specifically the algorithm OPAL (OversamPling with Artificial LLM-generated data).

Since we have done all data preparation for you, we do not defer the splits (split.py, specific_size_split.py), the classifier tuning (tune.py), and data conversion for LLM seed data (context2table.py, table2context.py) in ./main folder after the synthetic data generation and evaluation.


## Repository Structure

- **data/**: This folder houses all datasets, including splits and the synthetic data generated for the experiments.
- **gpt-api/**: Contains scripts for utilizing GPT-4 Turbo to generate synthetic data.
- **main/**: Includes all necessary code to reproduce the figures and tables presented in the paper.




## Set up Install required packages:


```bash
conda env create -f environment.yml
conda activate opal 
conda install catboost
```


# Generate synthetic data with GPT-4

Please first replace the place-holder in ./gpt-api/new_prompt.py line 85 with your own key. We store all synthetic data generated in ./gpt-api/synthetic. You can input two options: current_post and data_name.

current_post: post_spurious_focus (for spurious correlations) or post_imbalanced_focus (for imbalanced classification).

data_name: openmlDiabetes, heart_failure, gender.

```bash
python gpt-api/new_prompt.py --current_post post_spurious_focus --data_name gender
```


# Smote

You can use the following command to generate synthetic data for both classification and spurious correlations by SMOTE.

control: spurious (for spurious correlations) or imbalanced (for imbalanced classification).

data_name: openmlDiabetes, heart_failure, gender, craft (control=imbalanced)

## Imbalanced class
```bash
python main/imbalanced_spurious_smote.py --control post_imbalanced_focus --data_name gender
```

## Spurious corr
```bash
python main/imbalanced_spurious_smote.py --control post_spurious_focus --data_name gender
```


# Duplication

You can use the following command to generate synthetic data for both classification and spurious correlations by duplication.

control: spurious (for spurious correlations) or imbalanced (for imbalanced classification).

data_name: openmlDiabetes, heart_failure, gender.

## Imbalanced class
```bash
python main/imbalanced_spurious_duplicate.py --control post_imbalanced_focus --data_name gender
```

## Spurious corr
```bash
python main/imbalanced_spurious_duplicate.py --control post_spurious_focus --data_name gender
```


# Train classifiers and evaluation

## Imbalanced classification

```bash
python main/raw_pretrain_exp.py --current_post post_imbalanced_focus --data_name gender --classifier catBoost
```

current_post: post_imbalanced_focus (for OPAL), post_imbalanced_duplicate (for duplication), post_imbalanced_smote (for SMOTE)

data_name: openmlDiabetes, heart_failure, gender, craft

After running this program, you will see the output in the format
```plaintext
gender
_imbalanced_focus
catBoost
Row 0: [0.262, 0.22299999999999998, 0.16300000000000003, 0.09499999999999997, 0.09299999999999997, 0.08899999999999997]
Row 1: [0.09899999999999998, 0.07599999999999996, 0.07599999999999996, 0.06999999999999995, 0.07099999999999995, 0.06799999999999995]
Row 2: [0.061000000000000054, 0.049000000000000044, 0.06999999999999995, 0.06799999999999995, 0.06899999999999995, 0.06899999999999995]
Row 3: [0.07399999999999995, 0.08099999999999996, 0.07899999999999996, 0.08499999999999996, 0.07899999999999996, 0.07999999999999996]
Row 4: [0.06899999999999995, 0.06699999999999995, 0.07399999999999995, 0.07699999999999996, 0.07899999999999996, 0.07799999999999996]
Mean: [0.11299999999999999, 0.09919999999999998, 0.09239999999999997, 0.07899999999999996, 0.07819999999999996, 0.07679999999999995]
Std: [0.08449556201363478, 0.0702723274127163, 0.039601767637316425, 0.01115795680221071, 0.009444575162494084, 0.008642916174532767]
min: 0.07679999999999995, mean: 0.08976666666666663
```

