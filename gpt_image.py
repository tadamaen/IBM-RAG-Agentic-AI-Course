# %pip install openai==1.64.0 | tail -n 1
import base64
from openai import OpenAI
from IPython.display import Image, display

client = OpenAI()

# The exercise focuses on DALLE-2 and DALLE-3, but they have been deprecated (May 2026). Hence, the models we will test will change to 
# GPT-Image-1 and GPT-Image-1-mini instead. The code format and structure is exactly the same, just changing the model parameter. 

# Example 1.1: Generate Image Of A Cat Using GPT-Image-1
response = client.images.generate(model = "gpt-image-1",
                                  prompt = "a white siamese cat",
                                  size = "1024x1024",
                                  quality = "high",
                                  n = 1)

image_base64 = response.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

with open("cat.png", "wb") as f:
    f.write(image_bytes)

display(Image(filename = "cat.png", width = 512))


display.Image(url = url, width = 512)

# Example 1.2: Generate Image Of A Cat Using GPT-Image-1-mini
response = client.images.generate(model = "gpt-image-1-mini",
                                  prompt = "a white siamese cat",
                                  size = "1024x1024",
                                  quality = "high",
                                  n = 1)

image_base64 = response.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

with open("cat.png", "wb") as f:
    f.write(image_bytes)

display(Image(filename = "cat.png", width = 512))


display.Image(url = url, width = 512)

# Example 2.1: Generate Image Of A Sunset Using GPT-Image-1
response = client.images.generate(model = "gpt-image-1",
                                  prompt = "a beautiful lake with a sunset",
                                  size = "1024x1024",
                                  quality = "high",
                                  n = 1)

image_base64 = response.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

with open("cat.png", "wb") as f:
    f.write(image_bytes)

display(Image(filename = "cat.png", width = 512))

# Example 2.2: Generate Image Of A Sunset Using GPT-Image-1-mini
response = client.images.generate(model = "gpt-image-1-mini",
                                  prompt = "a beautiful lake with a sunset",
                                  size = "1024x1024",
                                  quality = "high",
                                  n = 1)

image_base64 = response.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

with open("cat.png", "wb") as f:
    f.write(image_bytes)

display(Image(filename = "cat.png", width = 512))
