# %%capture
# %pip install ibm-watsonx-ai==1.1.20 image==1.5.33 requests==2.32.0

from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from ibm_watsonx_ai.foundation_models import ModelInference
import os
import requests

credentials = Credentials(url = "https://us-south.ml.cloud.ibm.com")
project_id = "skills-network"
client = APIClient(credentials)
client.foundation_models.TextModels
client.foundation_models.TextModels.show()

# Image Preparation
url_image_1 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5uo16pKhdB1f2Vz7H8Utkg/image-1.png'
url_image_2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/fsuegY1q_OxKIxNhf6zeYg/image-2.png'
url_image_3 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/KCh_pM9BVWq_ZdzIBIA9Fw/image-3.png'
url_image_4 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/VaaYLw52RaykwrE3jpFv7g/image-4.png'

image_urls = [url_image_1, url_image_2, url_image_3, url_image_4] 
model_id = 'meta-llama/llama-3-2-11b-vision-instruct'

# Checking The Model Parameters
TextChatParameters.show()
params = TextChatParameters(temperature = 0.2, top_p = 0.5)

# Initialize The Model
model = ModelInference(model_id = model_id, credentials = credentials, project_id = project_id, params = params)

# Encoding The Image
def encode_images_to_base64(image_urls):
    encoded_images = []
    for url in image_urls:
        response = requests.get(url)
        if response.status_code == 200:
            encoded_image = base64.b64encode(response.content).decode("utf-8")
            encoded_images.append(encoded_image)
        else:
            print(f"Warning: Failed to fetch image from {url} (Status code: {response.status_code})")
            encoded_images.append(None)
    return encoded_images

encoded_images = encode_images_to_base64(image_urls)

# Multimodal Inference Function
def generate_model_response(encoded_image, user_query, assistant_prompt = "You are a helpful assistant. Answer the following user query in 1 or 2 sentences: "):
    messages = [{"role" : "user",
                 "content" : [{"type" : "text", "text" : assistant_prompt + user_query},
                              {"type" : "image_url", "image_url" : {"url" : "data:image/jpeg;base64," + encoded_image}}]}]
    response = model.chat(messages = messages)
    return response['choices'][0]['message']['content']

# Image Captioning
user_query = "Describe the photo"
for i in range(len(encoded_images)):
    image = encoded_images[i]
    response = generate_model_response(image, user_query)
    print(f"Description for image {i + 1}: {response}/n/n")
    print()

# Object Detection
# Example 1:
image = encoded_images[1]
user_query = "How many cars are in this image?"
print("User Query: ", user_query)
print("Model Response: ", generate_model_response(image, user_query))

# Example 2:
image = encoded_images[2]
user_query = "How severe is the damage in this image?"
print("User Query: ", user_query)
print("Model Response: ", generate_model_response(image, user_query))

# Example 3:
image = encoded_images[3]
user_query = "How much sodium is in this product?"
print("User Query: ", user_query)
print("Model Response: ", generate_model_response(image, user_query))
