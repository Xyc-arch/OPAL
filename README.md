# Introduction

This repository contains the code for the paper titled "Synthetic Oversampling: Theory and A Practical Approach Using LLMs to Address Data Imbalance".

Since we have done all data preparation for you, we do not elaborate the splits (split.py, specific_size_split.py), the classifier tuning (tune.py), and data conversion for LLM seed data (context2table.py, table2context.py) in ./main folder.


## Repository Structure

- **data/**: This folder houses all datasets, including splits and the synthetic data generated for the experiments.
- **gpt-api/**: Contains scripts for utilizing GPT-4 Turbo to generate synthetic data.
- **main/**: Includes all necessary code to reproduce the figures and tables presented in the paper.



## Setup Python Environment

First, create and activate a Python virtual environment:

### Windows:
```bash
python -m venv opal
opal\Scripts\activate
```

### macOS/Linux:
```bash
python -m venv opal
source opal/bin/activate
```

Install cmake for catBoost

macOS
```bash
brew install cmake
```

Windows
```bash
choco install cmake
```

Linux
```bash
sudo apt update
sudo apt install cmake
```


### Install required packages:

Install python packages:
```bash
pip install --upgrade pip 
pip install -r requirements.txt
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

data_name: openmlDiabetes, heart_failure, gender.

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
python main/raw_pretrain_exp.py --current_post post_imbalanced_focus --data_name gender --classifier logistic
```


