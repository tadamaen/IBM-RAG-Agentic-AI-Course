# pip3 install virtualenv 
# virtualenv my_env
# source my_env/bin/activate

# Installing required libraries
# pip install transformers==4.35.2 \
# torch==2.1.1 \
# gradio==5.9.0 \
# langchain==0.3.12 \
# langchain-community==0.3.12 \
# langchain_ibm==0.3.5 \
# ibm-watsonx-ai==1.1.16 \
# pydantic==2.10.3

# sudo apt update
# sudo apt install ffmpeg -y

# url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/hTqGqoC-LrW6S79HjuJUkg/trimmed-02.wav"
# response = requests.get(url)
# audio_file_path = "sample-meeting.wav"

# if response.status_code == 200:
# 	with open(audio_file_path, "wb") as file:
# 		file.write(response.content)
# 		print("File downloaded successfully")
# else:
# 	print("Failed to download the file")

import requests
import torch
from transformers import pipeline

# Initialize the speech-to-text pipeline from Hugging Face Transformers
pipe = pipeline("automatic-speech-recognition", model = "openai/whisper-tiny.en", chunk_length_s = 30)
sample = 'sample-meeting.wav'
prediction = pipe(sample, batch_size = 8)["text"]
print(prediction)

