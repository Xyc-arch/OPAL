
## Setup Python Environment

First, create and activate a Python virtual environment:

### Windows:
```bash
python -m venv gpt-oversample-api
gpt-oversample-api\Scripts\activate
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

## Generate data from GPT-4

Please use your own GPT-4 API keys. Please use new new_prompt.py to generate synthetic data. You can change data_name for different datasets and current_post to switch from spurious correlation to imbalanced classification and other way round. 
