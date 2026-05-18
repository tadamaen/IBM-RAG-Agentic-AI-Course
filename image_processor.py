import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import requests
import base64
from io import BytesIO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ImageProcessor:
    # Function to initialize the image processor with a pre-trained ResNet50 model
    def __init__(self, image_size = (224, 224), norm_mean = [0.485, 0.456, 0.406], norm_std = [0.229, 0.224, 0.225]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(pretrained = True).to(self.device)
        self.model.eval()
        self.preprocess = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize(mean = norm_mean, std = norm_std)])

    # Function to encode the image
    def encode_image(self, image_input, is_url = True):
        try:
            if is_url:
                response = requests.get(image_input)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image_input).convert("RGB")

            buffered = BytesIO()
            image.save(buffered, format = "JPEG")
            base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
            input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(input_tensor)
            feature_vector = features.cpu().numpy().flatten()
            return {"base64": base64_string, "vector": feature_vector}
          
        except Exception as e:
            print(f"Error encoding image: {e}")
            return {"base64": None, "vector": None}
          
    # Function to find the closest match in the dataset based on cosine similarity
    def find_closest_match(self, user_vector, dataset):
        try:
            dataset_vectors = np.vstack(dataset['Embedding'].dropna().values)
            similarities = cosine_similarity(user_vector.reshape(1, -1), dataset_vectors)
            closest_index = np.argmax(similarities)
            similarity_score = similarities[0][closest_index]
            closest_row = dataset.iloc[closest_index]
            return closest_row, similarity_score
          
        except Exception as e:
            print(f"Error finding closest match: {e}")
            return None, None
