import os
from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key='<your openai-api key>',
)



import json
import pandas as pd





def create_jsonl_for_chat_format(csv_file_path, jsonl_file_path):
    df = pd.read_csv(csv_file_path)
    
    with open(jsonl_file_path, 'w') as jsonl_file:
        for _, row in df.iterrows():
            # Create a list to hold the messages in the conversation
            messages = []

            # Simulate a system message (optional, can be adjusted as needed)
            system_message = {"role": "system", "content": "This is a patient data analysis."}
            messages.append(system_message)

            # User message with patient information
            user_message = {"role": "user", "content": f"Patient information: ID {row['Patient ID']}, Age {row['Age']}, Sex {row['Sex']}, Cholesterol {row['Cholesterol']}"}  # Add other fields as needed
            messages.append(user_message)

            # Assistant message with the heart attack risk
            assistant_message = {"role": "assistant", "content": f"The patient's heart attack risk is {row['Heart Attack Risk']}."}
            messages.append(assistant_message)

            # Write the entire conversation as one JSON object
            jsonl_file.write(json.dumps({"messages": messages}) + '\n')




 

def upload_file(file_path):
    # response = openai.File.create(file=open(file_path), purpose='fine-tune')
    response = client.files.create(file=open(file_path, "rb"), purpose="fine-tune")
    return response.id


