
## Setup Python Environment

First, create and activate a Python virtual environment:

### Windows:
```bash
python -m venv gpt-oversample-main
gpt-oversample-main\Scripts\activate
```

### macOS/Linux:
```bash
python3 -m venv gpt-oversample-main
source gpt-oversample-main/bin/activate
```

### Install required packages:
```bash
pip install -r requirements.txt
```



## Run the main code

In our code, we use control_ls to switch function we want to call; dataset_name_ls to change the dataset we want to work on; post_ls to switch among oversampling methods; method_list for classifiers.

You can split the dataset using specific_size_split.py

You can run all experiments about spurious correlation in spurious.py with "spurious_reorder" section in utils. 

You can run all experiments about imbalanced classification with imbalanced_smote, imbalanced_split, and imbalanced_spurious_duplicate.


## Crafted data

You can use crafted_data.py to generate crafted data.
